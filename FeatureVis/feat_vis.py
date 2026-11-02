# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os
import warnings
import numpy as np

from contextlib import contextmanager
from dataclasses import asdict, replace
from jaxtyping import Float
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from utilities.misc import NestedTensor

from engine.engine_bnn import EngineBNN
from engine.engine_gcnet_lr import EngineGCNet
from config.config_bnn import ConfigBNN
from config.config_gcnet_lr import ConfigGCNet
from config.config_am import ConfigAM

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = (
    False  # analysis in true fp32 (training used bf16 AMP)
)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)


# %%
class RGBNormalize(nn.Module):
    """
    Differentiable input normalization layer that scales raw [0, 1] RGB images
            to match dataset training statistics (e.g., ImageNet mean & std).
            Placed directly inside the optimization graph so gradients flow seamlessly
            back into the stimulus generator parameters.

    """

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):

        super().__init__()
        # self.mean = mean  # [mu_red, mu_blue, mu_green]
        # self.std = std  # [std_red, std_blue, std_green]
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1)
        )  # [1, 3, 1, 1]
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))  # [1, 3, 1, 1]

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        return (x - self.mean) / self.std


def _init_noise(
    shape: tuple[int, ...], seeds: Sequence[int], std: float
) -> Float[Tensor, "B ..."]:
    """
    Generates independent initial random Gaussian noise tensors for each optimization
            job (batch item) using dedicated CPU RNG generators. This ensures complete
            deterministic reproducibility regardless of batch size or chunking order.

    Args:
        shape: Base tensor shape per job, e.g.:
                - StereoFFTParam:   (2, C, H, W // 2 + 1, 2)
                - StereoPixelParam: (2, C, H, W)
        seeds: Sequence of length B (number of concurrent jobs in the batch)

    Returns noise: [B, *shape]


    """

    noise = (
        torch.stack(
            [
                torch.randn(shape, generator=torch.Generator().manual_seed(int(s)))
                for s in seeds
            ]
        )
        * std
    )  # [B, *shape]

    return noise


# ImageNet colour correlation (square root of the RGB covariance, as in Lucid/Captum):
# maps INDEPENDENT raw channels to naturally CORRELATED RGB (a luminance-dominated
# mixture), which removes most of the rainbow speckle of per-channel noise.
_COLOR_SVD_SQRT = torch.tensor(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
)
_COLOR_DECORRELATE = _COLOR_SVD_SQRT / _COLOR_SVD_SQRT.norm(dim=0).max()


class StereoFFTParam(nn.Module):
    """
    Fourier-basis (spectral) parameterization of stereo image pairs.
    Optimizing unconstrained pixel values directly often results in high-frequency
    adversarial noise. By parameterizing the image in the 2D frequency domain and
    attenuating high spatial frequencies via a 1/f^decay power prior (natural scene spectrum),
    gradient descent converges on smooth, structurally interpretable features.
    Both eyes (left and right) are parameterized with independent complex spectra.

    color_decorrelate  mix the 3 raw channels with the ImageNet colour-correlation
                        matrix before the sigmoid (same matrix for both eyes).
    lowpass            soft cutoff exp(-(f/lowpass)^4) on the amplitude prior, f in
                        cycles/px.  Removes the period-2..4 px checkerboards that
                        stride-2 convolutions let the optimiser exploit.
    eye_init="shared"  start the other eye as an exactly shifted copy of the
                        reference eye (Fourier phase ramp, no interpolation), so the
                        optimisation starts from a correlated pair at the probed
                        disparity instead of two unrelated noise fields.  Convention
                        (derived from build_costVol, both ref=+1 and ref=-1):
                            other_eye(x) = reference_eye(x + init_shift_px),
                        i.e. physical xL - xR = ref * init_shift_px.  The spectra
                        remain free parameters; this only changes the starting point.

    """

    def __init__(
        self,
        size: Tuple[int, int],
        seeds: Sequence[int],
        n_channels: int = 3,
        color_mode: str = "rgb",
        decay: float = 1.0,
        init_std: float = 0.05,
        color_decorrelate: bool = False,
        lowpass: float | None = None,
        eye_init: str = "independent",
        init_shift_px: float = 0.0,
        ref=1,
    ):

        super().__init__()
        h, w = size
        self.size = size
        self.n_channels = n_channels
        self.gray = color_mode == "gray"
        self.decorrelate = bool(color_decorrelate) and not self.gray and n_channels == 3
        c = 1 if self.gray else n_channels

        # spec shape: [B, 2, c, h, w // 2 + 1, 2] (the last dim is for real/imaginary parts of complex)
        spec = _init_noise((2, c, h, w // 2 + 1, 2), seeds, init_std)
        fy = torch.fft.fftfreq(h)[:, None]  # [h, 1]
        fx = torch.fft.rfftfreq(w)[None, :]  # [h, w // 2 + 1]
        if eye_init == "shared":
            r, o = (
                (0, 1) if ref == 1 else (1, 0)
            )  # reference eye index, other eye index
            ramp = torch.polar(  # x(n+k) <-> X(f) exp(+2 pi i f k)
                torch.ones(1, w // 2 + 1),
                2 * torch.pi * torch.fft.rfftfreq(w) * init_shift_px,
            )
            shifted = torch.view_as_complex(spec[:, r].contiguous()) * ramp
            spec[:, o] = torch.view_as_real(shifted)
        self.spec = nn.Parameter(spec)

        freq = (fx.square() + fy.square()).sqrt().clamp_min(1 / max(h, w))  # [h, w // 2 + 1]
        scale = freq.pow(-decay)

        # lowpass, cutoff freq above "lowpass"
        if lowpass is not None:
            rolloff_order = 4.0
            scale = scale * torch.exp(-((freq / lowpass).pow(rolloff_order)))

        # remove_dc:
        # scale[0, 0] = 0

        self.register_buffer("scale", scale)  # [h, w // 2 + 1]
        self.register_buffer("color_mix", _COLOR_DECORRELATE.clone())

    def forward(self) -> Float[Tensor, "B 2 C H W"]:
        spectrum = (
            torch.view_as_complex(self.spec) * self.scale
        )  # [B, 2, c, h, w // 2 + 1] (complex tensor)
        raw = (
            torch.fft.irfft2(spectrum, s=self.size, norm="ortho") / 4
        )  # [B, 2, c, h, w]

        if self.gray:
            raw = raw.expand(-1, 1, self.channels, -1, -1)  # [B, 2, n_channels, h, w]
        elif self.decorrelate:
            raw = torch.einsum("ij, bejhw -> beihw", self.color_mix, raw)

        return raw.sigmoid()  # [B, 2, C=3, h, w]


class StereoPixelParam(nn.Module):
    """
    Direct spatial-domain pixel parameterization (unconstrained logits squashed via sigmoid).
    Provides a baseline stimulus representation without Fourier spectrum priors.
    Allows testing whether Fourier priors introduce any frequency-biasing artifacts.

    """

    def __init__(
        self,
        size: Tuple[int, int],
        seeds: Sequence[int],
        n_channels: int = 3,
        color_mode: str = "rgb",
        init_std: float = 0.05,
    ):

        super().__init__()
        self.gray = color_mode == "gray"
        self.n_channels = n_channels
        self.logits = nn.Parameter(
            _init_noise((2, 1 if self.gray else n_channels, *size), seeds, init_std)
        )

    def forward(self) -> Float[Tensor, "B 2 C H W"]:
        x = self.logits
        if self.gray:
            x = x.expand(
                -1, -1, self.n_channels, -1, -1
            )  # [B, 2, C=3, H, W], bounded in (0, 1)

        return x.sigmoid()


class JobRNG:
    """
    Isolated per-job pseudo-random number generator on CPU.
    Ensures jitter and transformation trajectories for any given (channel, restart)
    are completely deterministic and invariant to batch size or chunking strategy.
    """

    def __init__(self, seeds):

        # List of B individual torch.Generator instances
        self.gens = [torch.Generator().manual_seed(int(s) + 777) for s in seeds]

    def draw(self, k) -> Float[Tensor, "B k"]:

        # Output random uniform values: [B, k] in range [0, 1)
        return torch.stack([torch.rand(k, generator=g) for g in self.gens])  # [B, k]


def apply_jitter(image: Float[Tensor, "B 2 C H W"], rng: JobRNG, config: ConfigAM):
    """
    Robustness transformations during optimization. Applies identical spatial translation
        (integer pixel roll) and isotropic scaling to BOTH eyes.
        Crucially:
            1) Jitter must be identical across both eyes so horizontal binocular disparity is preserved.
            2) Integer circular roll (torch.roll) avoids bilinear interpolation blur and artifacts.
            3) No rotation is applied to avoid inducing artificial vertical disparities.

    Args:
        image:      [B, 2, C, H, W]
        Output:     [B, 2, C, H, W] (transformed stereo pair)
    """

    x = image.flatten(1, 2)  # [B, 2C, H, W]
    B = x.shape[0]
    u = rng.draw(3)

    if tuple(config.scale_range) != (1.0, 1.0):
        lo, hi = config.scale_range
        sc = (lo + (hi - lo) * u[:, 2]).to(x.device)
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1] = sc, sc
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

    if config.jitter > 0:
        J = int(config.jitter)
        dy = ((u[:, 0] * (2 * J + 1)).long() - J).tolist()
        dx = ((u[:, 1] * (2 * J + 1)).long() - J).tolist()
        x = torch.stack([torch.roll(x[b], (dy[b], dx[b]), (-2, -1)) for b in range(B)])

    return x.reshape(image.shape)


def image_penalty(
    image: Float[Tensor, "B 2 C H W"],
    config: ConfigAM,
) -> Float[Tensor, "B"]:
    """
    Loss regularizers preventing extreme contrast explosions and high-frequency noise:
        - L2 penalty: penalizes distance from neutral gray (0.5), keeping stimulus contrast bounded.
        - Total Variation (TV) penalty: penalizes squared differences between adjacent pixels,
        encouraging spatially smooth stimuli.

    Args:
        image:      [B, 2, C, H, W]
        Output:     [B] - Total penalty scalar per sample
    """

    dims = (1, 2, 3, 4)  # Reduce over eye, channel, height, width
    pen = image.new_zeros(image.shape[0])  # [B]

    if config.l2: # L2 penalty
        pen = pen + config.l2 * (image - 0.5).square().mean(dims)
        # pen = pen + config.l2 * (image - 0.5).abs().mean(dims)

    if config.tv: # total variation penalty
        pen = pen + config.tv * (
            (image[..., 1:, :] - image[..., :-1, :]).square().mean(dims)
            + (image[..., :, 1:] - image[..., :, :-1]).square().mean(dims)
        )
    return pen


# %% indexing helpers


def resolve_disp_index(
    n_disp_channels: int,
    disp_idx: Optional[int] = None,
    disp_px: Optional[int] = None,
    max_disp: Optional[int] = None,
) -> int:
    """
    Maps a requested physical horizontal disparity (in pixels) or an explicit depth index
        into the internal 0-indexed discrete disparity slice of a 5D cost-volume tensor [B, C, D, H, W].
        Convention: Index d = n_disp_channels // 2 corresponds to ZERO disparity.

    Depth index of a 5D layer.
    Index n_disp_channels // 2 is ZERO disparity (cost-volume
    index d <-> shift d - D//2 feature px; output index j <-> disparity j - max_disp/2).
    """

    if disp_idx is not None and disp_px is not None:
        raise ValueError("give disp_idx or disp_px, not both")

    if disp_px is not None:
        idx = n_disp_channels // 2 + int(round(disp_px * n_disp_channels / max_disp))

    elif disp_idx is not None:
        idx = disp_idx

    else:
        idx = n_disp_channels // 2

    if not 0 <= idx < n_disp_channels:
        raise ValueError(f"disparity index {idx} outside [0, {n_disp_channels})")

    return idx


def index_to_disparity_px(idx: int, n_disp_channels: int, max_disp: int):
    """
    Inverts a 5D cost-volume disparity slice index into signed disparity in physical input pixels.
        Returns physical pixel shift (x_left - x_right).

    Signed disparity (input px) in the model's output convention (reference-eye
    frame; physical xL-xR = ref * this, if training labels follow that convention).
    """
    return (idx - n_disp_channels // 2) * max_disp / n_disp_channels


def center_origin(h: int, w: int, window: int):
    """
    Computes top-left spatial coordinates (y0, x0) to crop an odd-sized window of pixels
        centered at the exact midpoint of a feature map of shape (h, w).
    """

    half = window // 2
    y0, x0 = h // 2 - half, w // 2 - half

    if y0 < 0 or x0 < 0 or y0 + window > h or x0 + window > w:
        raise ValueError("window does not fit; reduce it rather than silently clipping")

    return y0, x0


def center_slice(
    feat_map: Float[Tensor, "..."],
    disp_index: Optional[int] = None,
    window: int = 1,
) -> Float[Tensor, "N C"]:
    """
    Extracts the center spatial region of size (window x window) from a 4D [N, C, H, W]
        or 5D [N, C, D, H, W] feature map, and computes the spatial average across the window.

    [N, C, H, W] or [N, C, D, H, W] -> [N, C];
    d_index must be resolved for 5D.

    Args:
        feat_map:   [N, C, H, W] (encoder feature maps) or
                    [N, C, D, H, W] (decoder cost volume / 3D convs)
        disp_index: int disparity slice index (required if feat_map is 5D)
        Output:     [N, C] - Activation of each channel at the central receptive field
    """

    if window < 1 or window % 2 != 1:
        raise ValueError("window must be odd and positive")

    if feat_map.ndim not in (4, 5):
        raise ValueError(f"Expected [N, C, H, W] or [N, C, D, H, W], got {tuple(feat_map.shape)}")
    y0, x0 = center_origin(*feat_map.shape[-2:], window)

    if feat_map.ndim == 5: # if feat_map contains disparity channel
        feat_map = feat_map[:, :, disp_index]

    elif disp_index is not None: # if feat_map does not have disparity channel
        raise ValueError("a disparity index is only meaningful for a 5D target")

    return feat_map[..., y0 : y0 + window, x0 : x0 + window].mean((-2, -1))


def _check_pair(image: Tensor) -> None:
    if image.ndim != 5 or image.shape[1] != 2:
        raise ValueError("Input must be stereo pairs [B, 2, C, H, W]")


def _split_normalize(
    image: Float[Tensor, "B 2 C H W"],
    normalizer: Optional[RGBNormalize],
) -> Tuple[Float[Tensor, "B C H W"], Float[Tensor, "B C H W"]]:
    """
    Splits a 5D stereo pair into individual left and right 4D camera images,
            applying dataset normalization (RGBNormalize) to each stream.

    Outputs:
        left:  [B, C, H, W]
        right: [B, C, H, W]
    """
    left, right = image[:, 0], image[:, 1]
    if normalizer is not None:
        left, right = normalizer(left), normalizer(right)
    return left, right


# %% targets: image [B, 2, C, H, W] + channels [B] -> raw activation [B]


class _StopForward(Exception):
    """
    Sentinel exception to abort model forward pass immediately once target layer is reached.
    """

    pass


class CenterTarget(nn.Module):
    """
    Generic activation maximization target. Hooks into ANY 2D or 3D layer across the network,
    intercepts the output feature map, crops the center neuron(s), gathers the specific target
    channels, and raises _StopForward to truncate unnecessary downstream forward computations.
    Requires per-sample normalization (analysis_mode) when B > 1 to avoid batch leakage.

    Generic target: one call of one layer, centre neuron, selected channel per sample.

    Shared encoder: call_index=0 is left, 1 is right.
    Decoder layers: call_index=0.
    Early exit via an exception in the forward hook (no downstream work);
    hooks are removed on every forward.
    Requires per-sample normalisation (analysis_mode) when B > 1.
    """

    kind = "generic"

    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        normalizer: Optional[RGBNormalize] = None,
        call_index: int = 0,
        ref: int = 1,
        disp_idx: Optional[int] = None,
        disp_px: Optional[int] = None,
        max_disp: Optional[int] = None,
        window: int = 1,
        default_batch: int = 32,
    ):
        super().__init__()

        if call_index < 0:
            raise ValueError("call_index must be nonnegative")

        self.model = model
        object.__setattr__(self, "layer", layer)  # no second registration of the layer

        self.call_index = call_index
        self.normalizer = normalizer
        self.ref = ref
        self.disp_idx = disp_idx
        self.disp_px = disp_px
        self.max_disp = max_disp
        self.window, self.default_batch = window, default_batch
        self.output_shape = None
        self.d_index = None

    def forward(
        self,
        image: Float[Tensor, "B 2 C H W"],
        channels: Float[Tensor, "B"],
    ) -> Float[Tensor, "B"]:
        """
        Runs model forward up to target layer, returning target channel activations.

        Shapes:
            image:      [B, 2, C, H, W]
            channels:   [B] (long tensor of target channel indices per batch item)
            Output:     [B] (scalar activation per batch item)
        """

        _check_pair(image)
        
        found = []
        count = 0
        def hook(module, inputs, out):
            nonlocal count
            current = count
            count += 1
            if current == self.call_index:
                if not isinstance(out, torch.Tensor):
                    raise TypeError("Select a layer whose output is a single tensor")
                found.append(out)  # not cloned: early exit, ReLUs are not in-place
                raise _StopForward()

        handle = self.layer.register_forward_hook(hook)
        try:
            left, right = _split_normalize(image, self.normalizer)  # [B, C, H, W] each
            ref = image.new_full((image.shape[0],), float(self.ref))  # [B]
            self.model(NestedTensor(left=left, right=right, ref=ref))

        except _StopForward:
            pass

        finally:
            handle.remove()

        if len(found) != 1:
            raise ValueError(
                f"Target call {self.call_index} was not reached (observed {count} calls)"
            )

        feat_map = found[0]  # [B, C_out, H_feat, W_feat] or [B, C_out, D, H_feat, W_feat]
        self.output_shape = (1,) + tuple(feat_map.shape[1:])

        if feat_map.ndim == 5:
            self.d_index = resolve_disp_index(
                feat_map.shape[2], self.disp_idx, self.disp_px, self.max_disp
            )

        elif self.disp_idx is not None or self.disp_px is not None:
            raise ValueError("a disparity is only meaningful for a 5D target")

        value = center_slice(feat_map, self.d_index, self.window)  # [B, C_out]
        if channels.max() >= value.shape[1] or channels.min() < 0:
            raise ValueError(f"channel outside [0,{value.shape[1]})")

        # Gather target channel per batch item: [B, 1] -> [B]
        return value.gather(1, channels.view(-1, 1)).squeeze(1)
    
# %% exact local ("fast") evaluation of the 3D decoder
#
# The target value depends on a small box of every earlier tensor.  We (1) propagate
# the required box backwards through the decoder graph (conv / transposed conv /
# residual add / pointwise BatchNorm3d+ReLU), (2) evaluate every needed node only on
# its box, with exact zero padding wherever a conv reads outside a tensor, and
# (3) read the target.  Valid only if every BatchNorm3d on the path uses fixed running
# statistics (a per-voxel affine map, as in GC_Net_LR); a BatchNorm3d that normalises
# over the whole volume (the BNN) makes the network non-local -> generic path.

AUTO_BATCH_BYTES = (
    6e9  # activation memory budget used to pick the batch size (heuristic)
)

# GC_Net_LR StereoDecoder.forward: node -> input(s).  "V0" is the cost volume,
# ("add", a, b) a residual sum (no ReLU after it); every other entry is a
# conv(+BatchNorm3d+ReLU) block ending in the ReLU output.
_GC_INPUT = {
    "19": "V0",
    "20": "19",
    "21": "V0",
    "22": "21",
    "23": "22",
    "24": "21",
    "25": "24",
    "26": "25",
    "27": "24",
    "28": "27",
    "29": "28",
    "30": "27",
    "31": "30",
    "32": "31",
    "33a": "32",
    "33b": ("add", "33a", "29"),
    "34a": "33b",
    "34b": ("add", "34a", "26"),
    "35a": "34b",
    "35b": ("add", "35a", "23"),
    "36a": "35b",
    "36b": ("add", "36a", "20"),
    "37": "36b",
}


class _Node:
    """
        Encapsulates the metadata and operator references for a single node in the decoder DAG.
        Nodes map directly to PyTorch modules, making it possible to traverse the graph and
        extract corresponding layer weights and biases.
    """
    def __init__(self, name: str,
                 kind,
                 inputs=(),
                 conv: nn.Conv3d | nn.ConvTranspose3d | None=None,
                 bn: nn.BatchNorm3d | None=None,
                 relu: nn.ReLU | None=None,
                 seq: nn.Sequential | None=None):
        # name: Node identifier string (e.g., "21", "33b")
        # kind: "cost" (source volume), "add" (residual elementwise sum), or "block" (conv/norm/activation module).
        # inputs: Tuple of input node name strings
        # conv: Bound nn.Conv3d or nn.ConvTranspose3d instance
        # bn: Bound nn.BatchNorm3d instance (or None)
        # relu: Bound nn.ReLU instance (or None)
        # seq: Parent container nn.Sequential (if applicable)
        self.name, self.kind, self.inputs = name, kind, tuple(inputs)
        self.conv, self.bn, self.relu, self.seq = conv, bn, relu, seq


class DecoderGraph:
    """
        Manages topological traversal and maps PyTorch module object IDs (`id(module)`) 
        back to graph nodes. Used to extract the computational DAG of GC-Net's 3D decoder.
    """
    def __init__(self, nodes: _Node):
        
        self.nodes = nodes
        
        # Map module IDs to their corresponding node name and stage.
        # This allows fast identification of target layers during activation maximization.
        self._lookup = {}
        for n in nodes.values():
            if n.kind != "block":
                continue
            self._lookup[id(n.conv)] = (n.name, "conv")
            if n.bn is not None:
                self._lookup[id(n.bn)] = (n.name, "bn")
            if n.relu is not None:
                self._lookup[id(n.relu)] = (n.name, "relu")
            if n.seq is not None:
                last = (
                    "relu"
                    if n.relu is not None
                    else ("bn" if n.bn is not None else "conv")
                )
                self._lookup[id(n.seq)] = (n.name, last)

    def lookup(self, module):
        """
        Returns `(node_name, stage)` where `stage` is `"conv"`, `"bn"`, or `"relu"`.
        Allows targeting pre-activations, post-normalizations, or post-ReLUs.
        """
        return self._lookup.get(id(module))

    def order(self, target):
        """
        Ancestors of `target` (inclusive), inputs before consumers.
        Executes Depth-First Search (DFS) to return a topologically sorted list of ancestor nodes 
        (inputs before consumers) required to evaluate `target`.
        """
        seen, out = set(), []

        def visit(name):
            if name in seen:
                return
            seen.add(name)
            for i in self.nodes[name].inputs:
                visit(i)
            out.append(name)

        visit(target)
        # Length of out <= 23 (number of nodes in the graph)
        return out


def _supported_conv(conv):
    """
        Verifies that a convolution operator adheres strictly to the kernel size, stride, 
        padding, and dilation assumptions used by the coordinate math in the receptive 
        field calculation.
    """
    
    k3 = (3, 3, 3)
    if isinstance(conv, nn.Conv3d):
        return (
            conv.kernel_size == k3
            and conv.padding == (1, 1, 1)
            and conv.dilation == (1, 1, 1)
            and conv.groups == 1
            and len(set(conv.stride)) == 1
            and conv.stride[0] in (1, 2)
        )
    if isinstance(conv, nn.ConvTranspose3d):
        return (
            conv.kernel_size == k3
            and conv.stride == (2, 2, 2)
            and conv.padding == (1, 1, 1)
            and conv.output_padding == (1, 1, 1)
            and conv.dilation == (1, 1, 1)
            and conv.groups == 1
        )
    return False


def build_decoder_graph(decoder):
    """
        Inspects `dec` (`model.decoder`). Verifies the presence of all GC-Net layers 
        (`layer19` to `layer37`), checks structural validity (`nn.Sequential` containing 
        Conv3d -> BatchNorm3d -> ReLU), and returns a validated `DecoderGraph`. 
        Returns `None` if the decoder does not match the GC-Net specification.
    """
    
    blocks = [n for n, v in _GC_INPUT.items() if not isinstance(v, tuple)]
    if not all(hasattr(decoder, f"layer{n}") for n in blocks):
        return None
    
    nodes = {"V0": _Node("V0", "cost")}
    for n, src in _GC_INPUT.items():
        
        if isinstance(src, tuple):
            nodes[n] = _Node(n, "add", src[1:])
            continue
        m = getattr(decoder, f"layer{n}")
        
        if isinstance(m, nn.Sequential):
            mods = list(m)
            kinds = [type(x) for x in mods]
            ok = kinds in (
                [type(mods[0])],
                [type(mods[0]), nn.BatchNorm3d, nn.ReLU],
            )
            if not ok:
                return None
            conv = mods[0]
            bn = mods[1] if len(mods) == 3 else None
            relu = mods[2] if len(mods) == 3 else None
            seq = m
        
        else:
            conv, bn, relu, seq = m, None, None, None
        
        if not _supported_conv(conv):
            return None
        nodes[n] = _Node(n, "block", (src,), conv, bn, relu, seq)
    
    return DecoderGraph(nodes)


from collections import namedtuple
FastSpec = namedtuple("FastSpec", "graph target stage")


def resolve_fast_target(model, layer):
    """
        Resolves an arbitrary network module into a validated `FastSpec(graph, target_node, stage)` 
        if local evaluation is possible, or `None` if the layer must use the generic fallback.
    """
    
    decoder = getattr(model, "decoder", None)
    if (
        decoder is None
        or not hasattr(model, "encoder")
        or not hasattr(decoder, "binocular_interaction")
        or not hasattr(decoder, "max_disp")
    ):
        return None
    
    # 1. Checks if model.decoder matches GC-Net.
    graph = build_decoder_graph(decoder)
    if graph is not None:  # GC_Net_LR: any conv / BN / ReLU / block of layers 19..37
        # Looks up layer in DecoderGraph.
        hit = graph.lookup(layer)
        if hit is None:
            return None
        target, stage = hit
        
        # Check all ancestor nodes for un-tracked BatchNorm statistics (running_mean is None).
        # If valid, returns FastSpec.
        bad = [
            n
            for n in graph.order(target)
            if graph.nodes[n].bn is not None and graph.nodes[n].bn.running_mean is None
        ]
        if bad:
            warnings.warn(
                f"BatchNorm3d without running statistics in blocks {bad}: the network is "
                "not local there, using the generic target"
            )
            return None
        return FastSpec(graph, target, stage)
    
    # 2. If not GC-Net (e.g., BNN): checks if layer is a 3D convolution directly consuming 
    # the cost volume via consumes_cost_volume(model, layer).
    if (
        isinstance(layer, nn.Conv3d)
        and _supported_conv(layer)
        and consumes_cost_volume(model, layer)
    ):
        # If verified, returns a minimal 2-node graph V0 -> T.
        nodes = {"V0": _Node("V0", "cost"), "T": _Node("T", "block", ("V0",), layer)}
        return FastSpec(DecoderGraph(nodes), "T", "conv")
    
    # 3. Otherwise, returns None.
    return None


def _conv_out_len(conv: nn.Conv3d | nn.ConvTranspose3d, n: int):
    """
        Computes the output 1D spatial/depth dimension given input length n.
        For nn.Conv3d: L_out = (n - 1) // s + 1
        For nn.ConvTranspose3d (s=2, p=1, k=3, op=1): L_out = 2n
    
    Dimensions:
        n (int): Scalar integer
        Output (int): Scalar integer L_out
    """
    if isinstance(conv, nn.ConvTranspose3d):
        return (n - 1) * 2 - 2 + 3 + 1  # = 2n for k3 s2 p1 output_padding 1
    
    return (n + 2 - 3) // conv.stride[0] + 1


def _in_range(conv: nn.Conv3d | nn.ConvTranspose3d, lo: int, hi: int):
    """
        Computes the input index interval [a, b) required to compute output interval [lo, hi).
        
        For nn.Conv3d: Output voxel o reads input voxels i in [s*o - 1, s*o + 1].
            Lower bound: a = s * lo - 1
            Upper bound (exclusive): b = s * (hi - 1) + 2
        For nn.ConvTranspose3d: Output voxel o is generated by inputs i satisfying o = 2i - 1 + k'.
            Lower bound: a = floor(lo / 2)
            Upper bound (exclusive): b = floor(hi / 2) + 1
            
    Inputs/Outputs:
        lo, hi (int): integers
        Outputs (Tuple[int, int]): a, b integers (may produce negative indices or values exceeding tensor bounds, handled by _fetch).
    """
    
    if isinstance(conv, nn.ConvTranspose3d):  # o = 2i - 1 + k, k in 0..2
        return lo // 2, hi // 2 + 1
    s = conv.stride[0]  # o reads inputs s*o-1 .. s*o+1
    
    return s * lo - 1, s * (hi - 1) + 2


def _fetch(x, xbox, req, size):
    """
        Extracts the required bounding box `req` from cached tensor `x` (which covers sub-region `xbox`), 
        zero-padding wherever `req` extends beyond the absolute physical tensor boundaries `[0, size)`.
        
    Variables & Dimensions:
        x (Tensor): Sub-volume tensor of shape [B, C, d_hi - d_lo, y_hi - y_lo, x_hi - x_lo], where (d_hi - d_lo) is xbox[0][1] - xbox[0][0].
        xbox (Tuple): Currently computed region [(d_lo, d_hi), (y_lo, y_hi), (x_lo, x_hi)].
        req (Tuple): Target requested region [(rd_lo, rd_hi), (ry_lo, ry_hi), (rx_lo, rx_hi)].
        size (Tuple): Global tensor dimensions (D_total, H_total, W_total).
        Output (Tensor): Tensor of shape [B, C, rd_hi - rd_lo, ry_hi - ry_lo, rx_hi - rx_lo] with zero padding applied to out-of-bounds boundaries.
    """
    
    sl, pad = [], []
    for (lo, hi), (blo, bhi), n in zip(req, xbox, size):
        clo, chi = max(lo, 0), min(hi, n)
        if clo < blo or chi > bhi:
            raise RuntimeError("internal error: required region not computed")
        sl.append(slice(clo - blo, chi - blo))
        pad.append((clo - lo, hi - chi))
    x = x[:, :, sl[0], sl[1], sl[2]]
    
    if any(v for pp in pad for v in pp):
        x = F.pad(x, (pad[2][0], pad[2][1], pad[1][0], pad[1][1], pad[0][0], pad[0][1]))
    
    return x


def _crop_to(x, xbox, obox):
    """
        Performs a relative internal spatial crop from a computed tensor region `xbox` down to sub-region `obox` 
        for residual skip additions (A + B).
        
    Dimensions:
        x (Tensor): [B, C, d_hi - d_lo, y_hi - y_lo, x_hi - x_lo] corresponding to `xbox`.
        Output (Tensor): [B, C, d_hi' - d_lo', y_hi' - y_lo', x_hi' - x_lo'] corresponding to `obox` where slices are offset by (olo - blo, ohi - blo).
    """
    
    sl = [slice(olo - blo, ohi - blo) for (olo, ohi), (blo, _) in zip(obox, xbox)]
    
    return x[:, :, sl[0], sl[1], sl[2]]


def _bn3d(y, bn, ch=None):
    """
        Exact local application of `nn.BatchNorm3d` in evaluation mode using frozen running statistics.
        Fuses running statistics and affine parameters into a scale vector and shift vector.
        If `ch` is provided (targeting a single channel per job), it slices the scale and shift vectors 
        to avoid broadcasting unneeded channels.
        
    Variables & Dimensions:
        y (Tensor): [B, C, D, H, W] or [B, 1, D, H, W]
        bn (nn.BatchNorm3d): BatchNorm3d layer with fixed statistics.
        ch (Tensor): Integer tensor [B] of channel targets, or None.
        scale, shift: Broadcasted as [1, C, 1, 1, 1] if ch is None, or [-1, 1, 1, 1, 1] (which will be viewed to match) if ch is specified.
        Output (Tensor): y scaled and shifted.
    """
    
    rm, rv = bn.running_mean, bn.running_var
    gamma = bn.weight if bn.weight is not None else torch.ones_like(rm)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(rm)
    scale = gamma * torch.rsqrt(rv + bn.eps)
    shift = beta - rm * scale
    
    if ch is None:
        shape = (1, -1, 1, 1, 1)
    else:
        scale, shift, shape = scale[ch], shift[ch], (-1, 1, 1, 1, 1)
    
    return y * scale.view(shape) + shift.view(shape)


class FastDecoderConvTarget(nn.Module):
    """
    Exact, local evaluation of a 3D decoder layer.
        Activation maximization is inherently spatially localized. Because 3D convolutions have small,
        localized receptive fields, a single output voxel does not depend on the entire cost volume.
        Instead, its dependency traces backward through the network as a narrow spatial-depth cone.

        This module operates in three stages:
        1. Static Graph & Bounding Box Propagation (_plan): Back-propagates a discrete 3D bounding box
            from the target layer to the cost volume V0.
        2. Minimal Volume Slicing (forward): Synthesizes only the micro-patch of V0 falling inside 
            the required box.
        3. Local Forward Evaluation (_block): Computes each intermediate 3D block strictly on its sub-volume
            with exact zero padding at true tensor boundaries. For the final target layer, it evaluates only the
            target channel via grouped 3D convolutions, achieving a massive reduction in memory and compute.

    BNN: a Conv3d that reads the cost volume (`decoder.layer3.0`).
    GC_Net_LR: the conv, BatchNorm3d, ReLU or whole block of any of layers 19..37
    (`decoder.layerN.0/.1/.2` or `decoder.layerN`, N in 19..32, 33a..36a, 37).
    The encoder (and its image-wide normalisation statistics) still runs on the full
    image.  Verified against CenterTarget in `sanity_checks`.
    """

    kind = "fast"

    def __init__(
        self,
        model: nn.Module,
        spec,
        input_hw,
        normalizer: Optional[RGBNormalize] = None,
        ref: int = 1,
        disp_idx: Optional[int] = None,
        disp_px: Optional[int] = None,
        max_disp: Optional[int] = None,
        window: int = 1,
        default_batch: int = 32,
    ):
        
        super().__init__()
        self.model = model
        object.__setattr__(self, "spec", spec)
        self.normalizer = normalizer
        self.ref = ref
        self.disp_idx = disp_idx
        self.disp_px = disp_px
        self.max_disp = max_disp
        self.window = window
        self.output_shape = None
        self.d_index = None
        self._plans = {}
        h, w = input_hw
        plan = self._plan(model.decoder.max_disp // 2, h, w)
        self.summary = plan["summary"]
        conv2d_ch = sum(
            m.out_channels for m in model.encoder.modules() if isinstance(m, nn.Conv2d)
        )
        per_sample = 4 * (2 * 4 * conv2d_ch * h * w + 3 * plan["elems"])  # bytes, rough
        self.default_batch = int(
            max(1, min(default_batch, AUTO_BATCH_BYTES // per_sample))
        )

    @staticmethod
    def _slice(
        decoder: nn.Module,
        feat_left: Float[Tensor, "B C H W"],
        feat_right: Float[Tensor, "B C H W"],
        ref: Float[Tensor, "B"],
        d: int,
    ) -> Float[Tensor, "B 2C H W"]:
        """
        Constructs a single 2D disparity slice of the cost volume at candidate disparity d.

        Shapes:
            feat_left, feat_right:  [B, C, H, W] - 2D feature maps from encoder
            ref:     [B] - Reference camera flag (+1 or -1)
            Output:  [B, 2*C, H, W] - Binocularly matched disparity slice
        """

        D = decoder.max_disp // 2
        w, c = feat_left.shape[-1], feat_left.shape[1]
        pad = (d, D - d)

        # Shifted right and left features
        feat_right_shifted = F.pad(feat_right, pad)[..., D // 2 : D // 2 + w]  # [B, C, H, W]
        feat_left_shifted = F.pad(feat_left, pad)[..., D // 2 : D // 2 + w]  # [B, C, H, W]
        left_is_reference = (ref == 1).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        left = torch.where(left_is_reference, feat_left, feat_left_shifted) # [B, C, H, W]
        right = torch.where(left_is_reference, feat_right_shifted, feat_right) # [B, C, H, W]

        binocular_interaction = decoder.binocular_interaction
        if binocular_interaction == "default":
            chans = (left, right)
        elif binocular_interaction == "bem":
            chans = (left * right / c, (left.square() + right.square()) / c)
        elif binocular_interaction == "cmm":
            p = left * right / c
            chans = (p, F.relu(p))
        elif binocular_interaction == "sum_diff":
            chans = (left + right, left - right)
        else:
            raise ValueError(f"Unknown binocular interaction: {binocular_interaction}")

        # Concatenated interaction channels: [B, 2*C, H, W]
        return torch.cat(chans, dim=1)
    
    def _plan(self, D: int, h: int, w: int):
        """
            Backward Receptive Field Planning.
            Computes the tensor dimensions of each node forward through the DAG.
            Identifies output target coordinates, initializes the target bounding box,
            and iterates in reverse topological order to propagate bounding boxes to the cost volume.

        Variables & Dimensions:
            D (int): Maximum cost volume disparity planes (D = max_disp // 2)
            h, w (int): Spatial dimensions of 2D encoder feature maps
            Output (Dict): Dictionary containing execution order, shapes, bounding boxes, target disparity, etc.
        """
        
        key = (D, h, w)
        if key in self._plans:
            return self._plans[key]
        graph = self.spec.graph, 
        target = self.spec.target
        order = graph.order(target)
        
        shapes, chans = {}, {}
        # 1. Compute the tensor dimensions of each node forward through the DAG.
        for n in order:
            nd = graph.nodes[n]
            
            if nd.kind == "cost":
                shapes[n] = (D, h, w)
            
            elif nd.kind == "add":
                shapes[n] = shapes[nd.inputs[0]]
                chans[n] = chans[nd.inputs[0]]
                if shapes[nd.inputs[1]] != shapes[n]:
                    raise ValueError(f"residual inputs of {n} differ in shape")
            
            else:
                shapes[n] = tuple(
                    _conv_out_len(nd.conv, L) for L in shapes[nd.inputs[0]]
                )
                chans[n] = nd.conv.out_channels

        for n in order:  # cost-volume channels = in_channels of a consumer
            if graph.nodes[n].kind == "cost":
                consumer = next(
                    graph.nodes[m].conv
                    for m in order
                    if n in graph.nodes[m].inputs and graph.nodes[m].kind == "block"
                )
                chans[n] = consumer.in_channels

        # 2. Identify output target coordinates.
        ts = shapes[target]
        d = resolve_disp_index(ts[0], self.disp_idx, self.disp_px, self.max_disp)
        try:
            y0, x0 = center_origin(ts[1], ts[2], self.window)
        except ValueError:
            raise ValueError(
                f"window={self.window} does not fit the {ts[1]}x{ts[2]} output of decoder "
                f"block {target}; use a smaller window for this layer"
            ) from None

        # 3. Initialize the target bounding box.
        # box[target] = ((d, d + 1), (y0, y0 + window), (x0, x0 + window))
        box = {target: ((d, d + 1), (y0, y0 + self.window), (x0, x0 + self.window))}

        # 4. Iterate in reverse topological order.
        for n in reversed(order):
            nd = graph.nodes[n]
            if nd.kind == "cost":
                continue
            for inp in nd.inputs:
                req = (
                    box[n]
                    if nd.kind == "add"
                    else tuple(_in_range(nd.conv, lo, hi) for lo, hi in box[n])
                )
                # Clamp to valid tensor boundaries
                cl = tuple(
                    (max(a, 0), min(b, N)) for (a, b), N in zip(req, shapes[inp])
                )
                old = box.get(inp)
                # Union intersecting footprints from multiple consumers
                box[inp] = (
                    cl
                    if old is None
                    else tuple(
                        (min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(old, cl)
                    )
                )

        vol = lambda b: int(np.prod([hi - lo for lo, hi in b]))
        elems = sum(chans[n] * vol(box[n]) for n in order)
        dims = tuple(hi - lo for lo, hi in box["V0"])
        pct = 100 * vol(box["V0"]) / int(np.prod(shapes["V0"]))

        plan = dict(
            order=order,
            shapes=shapes,
            box=box,
            d=d,
            elems=elems,
            summary=(
                f"fast target {target}.{self.spec.stage}: cost-volume box "
                f"{dims[0]}x{dims[1]}x{dims[2]} = {pct:.1f}% of the volume"
            ),
        )
        self._plans[key] = plan
        return plan

    def _block(self, nd: _Node, x, xbox, xsize, ob, ch):
        """
            Evaluates a single block node in the DAG forward pass.
            Calls `_fetch` to retrieve the input patch, applying padding to out-of-bounds voxels.
            Executes standard 3D convolution or transposed convolution across all channels (if intermediate layer)
            or only the target channel per job (if final target layer, via grouped convs).
            Applies pointwise operations (_bn3d and F.relu).

        Variables & Dimensions:
            nd (_Node): The current block node to evaluate.
            x (Tensor): Input sub-volume tensor. [B, C_in, d_hi - d_lo, y_hi - y_lo, x_hi - x_lo] (from xbox).
            xbox (Tuple): The bounding box corresponding to input `x`.
            xsize (Tuple): The full tensor dimensions of the input.
            ob (Tuple): The required output bounding box.
            ch (Tensor): Channel indices if target layer, None if intermediate.
            Output (Tensor): Evaluated sub-volume tensor for `ob`.
        """
        conv = nd.conv
        convT = isinstance(conv, nn.ConvTranspose3d)
        req = [_in_range(conv, lo, hi) for lo, hi in ob]
        x = _fetch(x, xbox, req, xsize)
        if ch is None:  # all channels
            if convT:
                y = F.conv_transpose3d(x, conv.weight, None, stride=2)
                y = self._crop_convT(y, req, ob)
                if conv.bias is not None:
                    y = y + conv.bias.view(1, -1, 1, 1, 1)
            else:
                y = F.conv3d(x, conv.weight, conv.bias, stride=conv.stride)
        else:  # one filter per job
            B, cin = x.shape[:2]
            xg = x.reshape(1, B * cin, *x.shape[2:])  # [1, B * C_in, D, H, W]
            if convT:
                wsel = (
                    conv.weight[:, ch]
                    .permute(1, 0, 2, 3, 4)
                    .reshape(B * cin, 1, 3, 3, 3)
                )
                y = F.conv_transpose3d(xg, wsel, None, stride=2, groups=B)
                y = self._crop_convT(y, req, ob)
            else:
                y = F.conv3d(xg, conv.weight[ch], None, stride=conv.stride, groups=B)
            y = y.transpose(0, 1)  # [B, 1, 1, window, window]
            if conv.bias is not None:
                y = y + conv.bias[ch].view(-1, 1, 1, 1, 1)
        stage = self.spec.stage if ch is not None else "relu"
        if stage == "conv":
            return y
        if nd.bn is not None:
            y = _bn3d(y, nd.bn, ch)
        if stage == "bn" or nd.relu is None:
            return y
        return F.relu(y)

    @staticmethod
    def _crop_convT(y, req, ob):
        """
            Crops the output of an unpadded transposed convolution back to the required target box.
        """
        # unpadded transposed conv of the crop: q = o - 2*i_lo + 1
        sl = [
            slice(olo - 2 * rlo + 1, ohi - 2 * rlo + 1)
            for (olo, ohi), (rlo, _) in zip(ob, req)
        ]
        return y[:, :, sl[0], sl[1], sl[2]]

    def forward(
        self,
        image: Float[Tensor, "B 2 C H W"],
        channels: Float[Tensor, "B"],
    ) -> Float[Tensor, "B"]:
        
        """
        Execution lifecycle of local evaluation.
            1. Input Normalization & 2D Encoder: Processes left and right inputs on the full spatial canvas.
            2. Backward Receptive Field Planning: Retrieves `_plan`.
            3. Minimal Cost Volume Construction: Builds only the disparity slices needed in the target box.
            4. Topological Sub-Volume Forwarding: Iterates forward through `plan["order"]`, executing `_block` or `_crop_to`.
            5. Center Window Readout: Computes spatial mean over the window for the target layer's output.

        Shapes:
            image (Tensor): [B, 2, 3, H_in, W_in]
            channels (Tensor): [B] target channel index per batch job
            Output (Tensor): [B] final scalar activations per batch sample
        """

        _check_pair(image)

        decoder = self.model.decoder
        B = image.shape[0]
        stride = self.conv.stride[0]
        
        graph = self.spec.graph
        spec = self.spec

        # 1. Run 2D encoder on full image        
        left, right = _split_normalize(image, self.normalizer)  # [B, C, H, W] each
        ref = image.new_full((B,), float(self.ref))  # [B]
        feat_left, feat_right = self.model.encoder(
            NestedTensor(left=left, right=right, ref=ref)
        )  # [B, C_feat, H_feat, W_feat] each
        _, _, h, w = feat_left.shape
        
        # 2. Retrieve Receptive Field Plan
        plan = self._plan(decoder.max_disp // 2, h, w)
        shapes = plan["shapes"]
        box = plan["box"]
        
        # 3 & 4. Cost Volume Construction and Forward Execution
        val = {}
        for n in plan["order"]:
            nd = graph.nodes[n]
            ob = box[n]
            
            if nd.kind == "cost":
                (d0, d1), (y0, y1), (x0, x1) = ob
                feat_left_row = feat_left[:, :, y0:y1]
                feat_right_row = feat_right[:, :, y0:y1]
                
                # Synthesizes micro-patch of the raw 5D cost volume
                # val["V0"] shape: [B, 2C_enc, d1 - d0, y1 - y0, x1 - x0]
                val[n] = torch.stack(
                    [
                        self._slice(decoder, feat_left_row, feat_right_row, ref, dd)[..., x0:x1]
                        for dd in range(d0, d1)
                    ],
                    dim=2,
                )
            
            elif nd.kind == "add":
                a, b = nd.inputs
                val[n] = _crop_to(val[a], box[a], ob) + _crop_to(val[b], box[b], ob)
            
            else:
                src = nd.inputs[0]
                val[n] = self._block(
                    nd,
                    val[src],
                    box[src],
                    shapes[src],
                    ob,
                    channels if n == spec.target else None,
                )
                
        # 5. Output Extraction
        out = val[spec.target]  # [B, 1, 1, window, window]
        conv = graph.nodes[spec.target].conv
        self.output_shape = (1, conv.out_channels, *shapes[spec.target])
        self.d_index = plan["d"]

        # Spatial mean across window: [1, B, 1, 1, 1] -> [B]
        return out.mean((-2, -1)).reshape(B)


@torch.no_grad()
def consumes_cost_volume(
    model: nn.Module, layer: nn.Module, probe_hw: Tuple[int, int] = (64, 128)
):
    """
    True iff the input of `layer` is exactly the tensor returned by
    `model.decoder.build_costVol` (proved by object identity on a tiny probe
    forward), i.e. the layer is a first-level reader of the cost volume.
    Anything else (deeper layers, encoder layers, ...) must NOT use the fast path.

    Args:
        model (torch.nn.Module): The stereo/depth neural network model.
        layer (torch.nn.Module): Target layer being checked for direct cost volume consumption.
        probe_hw (tuple[int, int]): Probe height and width, e.g., (H, W) = (64, 128).

    Returns:
        bool: True if layer's input is identical to build_costVol output, False otherwise.
    """

    decoder = getattr(model, "decoder", None)
    if (
        decoder is None
        or not hasattr(decoder, "build_costVol")
        or not hasattr(model, "encoder")
    ):
        return False

    device = next(model.parameters()).device

    # Dictionary used as a mutable closure box to hold intercepted tensor references
    box = {}
    original = decoder.build_costVol  # bound method on decoder

    def wrapped(*args, **kwargs):
        out = original(
            *args, **kwargs
        )  # [B, C_vol, D, H_vol, W_vol] (or 4D/5D cost volume tensor)
        box["vol"] = out
        return out

    def pre_hook(module, inputs):
        # inputs: tuple containing positional arguments passed to `layer`
        # inputs[0] shape: depends on layer; expected [B, C_in, ...]
        box["inp"] = inputs[0]
        raise _StopForward()

    # Monkey-patch dec.build_costVol to intercept its return value
    decoder.build_costVol = wrapped  # instance attribute shadows the class method
    handle = layer.register_forward_pre_hook(pre_hook)

    try:
        # H, W = probe_hw (e.g., 64, 128)
        # x shape: [B=1, StereoChannels=2, ColorChannels=3, H=64, W=128]
        x = torch.zeros(1, 2, 3, *probe_hw, device=device)
        ref = torch.ones(1, device=device)  # [B]

        # Dummy NestedTensor input structure:
        # left shape:  [B=1, C=3, H=64, W=128]
        # right shape: [B=1, C=3, H=64, W=128]
        # ref shape:   [B=1]
        model(NestedTensor(left=x[:, 0], right=x[:, 1], ref=ref))

    except _StopForward:
        pass  # Execution stopped immediately upon reaching `layer`

    except (
        Exception
    ) as err:  # probe not runnable -> use the (always valid) generic path
        warnings.warn(
            f"cost-volume probe failed ({type(err).__name__}: {err}); "
            "falling back to the generic target"
        )
        return False

    finally:
        # Guarantee handle removal and restore un-patched dec.build_costVol method
        handle.remove()
        decoder.__dict__.pop("build_costVol", None)

    return "vol" in box and box.get("inp") is box["vol"]


def make_target(
    model: nn.Module,
    layer: nn.Module,
    config: ConfigAM,
    normalizer: Optional[RGBNormalize],
    call_index: int,
) -> Union[FastDecoderConvTarget, CenterTarget]:
    """
    Target factory. Inspects the requested layer and returns FastDecoderConvTarget
    if targeting the first 3D cost volume conv, otherwise falls back to CenterTarget.

    The most compute- and memory-intensive operation in a stereo network is constructing
    and convolving over the 5D cost volume. This function checks whether the target
    layer qualifies for the local micro-patch optimization

    make_target inspects the requested layer, its structural parameters, and the network's
    properties to decide between:
    1. An exact micro-volume fast path (FastDecoderConvTarget).
    2. A universal early-exit forward hook (CenterTarget).
    """

    kw = dict(
        normalizer=normalizer,
        ref=config.ref,
        disp_idx=config.disp_idx,
        disp_px=config.disp_px,
        max_disp=config.max_disp,
        window=config.window,
    )
    dec = getattr(model, "decoder", None)
    fast_ok = (config.fast_path and call_index == 0)
    if fast_ok:
        spec = resolve_fast_target(model, layer)
        if spec is not None:
            _, H, W = config.input_size
            return FastDecoderConvTarget(model, spec, (H // 2, W // 2), **kw)

    in_encoder = hasattr(model, "encoder") and any(
        m is layer for m in model.encoder.modules()
    )
    
    return CenterTarget(
        model, layer, call_index=call_index, default_batch=32 if in_encoder else 1, **kw
    )


# %% analysis-mode context managers


@contextmanager
def analysis_mode(model: nn.Module):
    """
    Freeze the model temporarily.

    Prepares the network for multi-sample activation maximization.
    NOTE: BatchNorm layers have `track_running_stats=False`. In standard
    .eval() mode, they still compute batch statistics over the current forward batch,
    which would leak statistics across batched optimization jobs.
    This contextmanager analysis_mode freezes model parameters and dynamically swaps
    BatchNorm layers with per-sample F.instance_norm, guaranteeing complete
    isolation between batched jobs.

    """
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    relus = [(m, m.inplace) for m in model.modules() if isinstance(m, nn.ReLU)]
    bns = [
        m
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        and m.running_mean is None
    ]
    try:
        model.eval().requires_grad_(False)
        for m, _ in relus:
            m.inplace = False
        for m in bns:
            m.forward = lambda x, _m=m: F.instance_norm(
                x, None, None, _m.weight, _m.bias, True, 0.0, _m.eps
            )
        yield
    finally:
        for m in bns:
            m.__dict__.pop("forward", None)
        for m, flag in modes:
            m.training = flag
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, flag in relus:
            m.inplace = flag


class _RedirectedReLUFn(torch.autograd.Function):
    """
    Custom backward pass for nn.ReLU to mitigate the "dead neuron" problem during optimization.
    If a unit is currently inactive (x <= 0), standard ReLU blocks all backward gradient (g = 0).
    Redirected ReLU allows gradients to pass through inactive units IF the gradient direction
    would push the activation toward becoming positive.

    """

    @staticmethod
    def forward(ctx, x: Tensor, sign: int) -> Tensor:
        ctx.save_for_backward(x)
        ctx.sign = sign
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, g: Tensor) -> Tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        return g * ((x > 0) | (ctx.sign * g > 0)).to(g.dtype), None


@contextmanager
def redirected_relu_mode(model: nn.Module, sign: int = 1, enabled: bool = False):
    """
    Context manager that temporarily patches all nn.ReLU forward methods to use
    _RedirectedReLUFn during optimization.
    """

    patched = []
    if enabled:
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.forward = lambda x, _s=sign: _RedirectedReLUFn.apply(x, _s)
                patched.append(m)
    try:
        yield len(patched)
    finally:
        for m in patched:
            m.__dict__.pop("forward", None)


# %% driver


class ActivationMaximization:

    def __init__(self, engine: EngineBNN | EngineGCNet, config_am: ConfigAM):

        self.engine = engine
        self.config = engine.config
        self.config_am = config_am

        self.model = self.engine.model
        self.model_name = self.engine.model_name

        # set up output directories
        self.make_dirs()

        # set up target layers
        self.set_target_layers()

        # print out list of target layers
        self.print_target_layers()

        # print out parameters for activation maximization
        self.print_am_params()

    def make_dirs(self):

        # parent_folder = "/media/wundari/S990Pro2_4TB"
        parent_folder = os.path.abspath(f"{os.curdir}/..")
        if self.config.dataset == "sceneflow_flying":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/"
        elif self.config.dataset == "sceneflow_monkaa":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/"
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        if self.config.dataset_directory:
            self.datadir = os.path.abspath(self.config.dataset_directory)

        # saving directory
        self.save_dir = os.path.join(
            f"{self.model_name}/run",
            self.config.dataset,
            f"bino_interaction_{self.config.binocular_interaction}",
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # experiment directory
        # runs = natsorted(glob.glob(os.path.join(self.save_dir, "experiment_*")))
        # if self.config.load_state:
        #     run_id = self.config.experiment_id
        # else:
        #     # run_id = int(runs[-1].split("_")[-1]) + 1 if runs else 0
        run_id = self.config.seed
        self.experiment_dir = os.path.join(
            self.save_dir,
            f"experiment_{run_id}",
        )
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # activation_maximization directory
        self.am_dir = os.path.join(self.experiment_dir, "activation_maximization")
        if not os.path.exists(self.am_dir):
            os.makedirs(self.am_dir)

    def _list_candidate_layers(self):
        kinds = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        for name, module in self.model.named_modules():
            if isinstance(module, kinds):
                print(
                    f"{name:24} {type(module).__name__:24} channels={module.out_channels}"
                )

    def set_target_layers(self):

        # Each entry targets ONE invocation of a module.
        # channels = None means all.
        # Depth index D//2 is ZERO disparity (use disp_px to pick a disparity in pixels;
        # layer3 has 2 px resolution, layer4 1 px).
        # Shared encoder: call 0=left, 1=right.

        if self.model_name == "BNN":

            self.target_layers = [
                # dict(
                #     name="mono_left_conv1_pre_relu",
                #     module="encoder.in_conv.0",
                #     call=0,  # call = 0 -> left monocular channel; call = 1 -> right monocular channel
                #     channels=None,
                # ),
                # dict(
                #     name="mono_right_conv1_pre_relu",
                #     module="encoder.in_conv.0",
                #     call=1,
                #     channels=None,
                # ),
                # dict(
                #     name="mono_left_layer2",
                #     module="encoder.layer2.0",
                #     call=0,
                #     channels=None,
                # ),
                # dict(
                #     name="mono_right_layer2",
                #     module="encoder.layer2.0",
                #     call=1,
                #     channels=None,
                # ),
                dict(
                    name="decoder_layer3_pre_relu",
                    module="decoder.layer3.0",
                    call=0,
                    channels=None,
                ),
                # disparity sweep:
                # *[dict(name=f"decoder_layer3_pre_d{d:+d}", module="decoder.layer3.0", call=0,
                #        channels=None, disp_px=d, sign=1) for d in (-20, -10, 0, 10, 20)],
                dict(
                    name="decoder_layer3_post_relu",
                    module="decoder.layer3",
                    call=0,
                    channels=list(range(32)),
                    disp_px=0,
                ),
                # Final layer: one channel; model uses softmax(-cost), so sign=-1 (min cost) = preferred.
                dict(
                    name="decoder_layer4_cost",
                    module="decoder.layer4",
                    call=0,
                    channels=[0],
                    disp_px=0,
                ),
            ]
        else:  # gcnet variants
            # layer19.0 (stride 1, 2 px resolution) and layer21.0 (stride 2, 4 px resolution)
            # read the cost volume directly -> exact fast path.  Everything deeper is generic
            # (full decoder forward/backward; heavy, auto batch size is 1 there).
            # NOTE (binocular_interaction="default"): the PRE-activation layers 19.0 / 21.0 are
            # linear in [left, right] features, so the eye-ablation interaction there is exactly
            # 0 by construction.  Look at post-ReLU / deeper layers (or "bem") for interaction.
            self.target_layers = [
                *[dict(
                    name=f"decoder_layer{i}_pre_relu",
                    module=f"decoder.layer{i}.0",
                    call=0,
                    channels=None,
                    disp_px=0,
                ) for i in range(19, 33)],
                
                dict(name)
                
                # disparity sweep:
                # *[dict(name=f"decoder_layer19_pre_d{d:+d}", module="decoder.layer19.0", call=0,
                #        channels=None, disp_px=d, sign=1) for d in (-20, -10, 0, 10, 20)],
                # encoder (weights shared between the eyes; call 0 = left, call 1 = right):
                # dict(name="left_layer1_pre", module="encoder.layer1.0", call=0, channels=None, sign=1),
                # dict(name="right_layer1_pre", module="encoder.layer1.0", call=1, channels=None, sign=1),
                # dict(name="left_res0_conv1", module="encoder.res_block.0.conv2d_block1.0", call=0, channels=None, sign=1),
                # post-BN3d+ReLU (composite module -> list channels explicitly; generic path, heavy):
                # dict(
                #     name="decoder_layer19_post",
                #     module="decoder.layer19",
                #     call=0,
                #     channels=list(range(config.base_channels)),
                #     disp_px=0,
                # ),
                # Final layer: model uses softmax(-cost), so sign=-1 (min cost) = preferred.
                dict(
                    name="decoder_layer37_cost",
                    module="decoder.layer37",
                    call=0,
                    channels=[0],
                    disp_px=0,
                ),
            ]

    def print_target_layers(self):

        print("==============================================================")
        print(f"List of target layers in {self.model_name}")
        print("==============================================================")
        self._list_candidate_layers()
        print("==============================================================\n")

    def print_am_params(self):

        print("==============================================================")
        print("Activation maximization parameters:")
        print("==============================================================")
        print(f"Input size: {self.config_am.input_size}")
        print(f"The number of independent optimization runs: {self.config_am.restarts}")
        print(f"The number of iterations: {self.config_am.iterations}")
        print(f"Learning rate: {self.config_am.lr}")
        print(f"Parameterization [fft or pixel]: {self.config_am.parameterization}")
        print(f"Color mode [rgb or gray]: {self.config_am.color_mode}")
        print(f"Decay rate: {self.config_am.decay}")
        print(f"Physical eye reference [1: left, -1: right]{self.config_am.ref}")
        print(f"Spatial neighborhood size [odd number]: {self.config_am.window}")
        print(
            f"Jitter size [pix, 0: no transform. Large jitter blurs phase-tuned neurons.]: {self.config_am.jitter}"
        )
        print("==============================================================\n")

    def optimize_batch(
        self,
        target: Union[CenterTarget, FastDecoderConvTarget],
        config_am: ConfigAM,
        channels: Sequence[int],
        seeds: Sequence[int],
        device: torch.device,
    ):
        """
        Optimizes B independent stereo image pairs simultaneously in a single batched pass.
            Because the loss is an elementwise sum over jobs and Adam maintains per-element
            moments, running B jobs concurrently is mathematically identical to running B
            sequential passes, but drastically faster on GPU.

        Args:
            channels:    List[int] of length B (target channel index for each job)
            seeds:       List[int] of length B (RNG seeds)
            Return runs: List of B result dictionaries containing:
                            - 'img':         [2, C=3, H, W] - Best optimized stereo pair on CPU
                            - 'act':         float - Best activation achieved
                            - 'score':       float - Best objective score (act - penalty)
                            - 'step':        int - Step index where best score was reached
                            - 'trace':       [n_eval] - Activation trajectory on untransformed image
                            - 'score_trace': [n_eval] - Objective trajectory
                            - 'trace_steps': [n_eval] - Step numbers where evaluation occurred
                            - 'train_trace': [iterations] - Activation at each jittered training step
                            - 'seed':        int - Random seed used
        """

        B = len(channels)
        ch = torch.tensor(channels, device=device, dtype=torch.long)
        c, h, w = config_am.input_size

        if config_am.parameterization == "fft":
            shift = config_am.eye_init_shift_px
            if shift is None:
                shift = config_am.disp_px if config_am.disp_px is not None else 0.0
            param = StereoFFTParam(
                (h, w),
                seeds,
                c,
                config_am.color_mode,
                config_am.decay,
                config_am.init_std,
                color_decorrelate=config_am.color_decorrelate,
                lowpass=config_am.lowpass,
                eye_init=config_am.eye_init,
                init_shift_px=shift,
                ref=config_am.ref,
            )
        else:
            param = StereoPixelParam(
                (h, w), seeds, c, config_am.color_mode, config_am.init_std
            )
        param = param.to(device)

        opt = torch.optim.Adam(param.parameters(), lr=config_am.lr)

        rng = JobRNG(seeds)
        identity = config_am.jitter == 0 and tuple(config_am.scale_range) == (1.0, 1.0)

        steps, acts, scores, train = [], [], [], []
        best = dict(
            score=torch.full((B,), -float("inf"), device=device),  # [B]
            act=torch.zeros(B, device=device),  # [B]
            step=torch.zeros(B, dtype=torch.long, device=device),  # [B]
            img=None,  # [B, 2, C, H, W]
        )

        def record(step: int, image: Tensor, act: Tensor):
            # The UNtransformed pair is what we report, so only it selects the best step.
            # image: [B, 2, C, H, W]
            # act:   [B]
            # score: [B]

            score = config_am.sign * act - image_penalty(image, config_am)
            if not torch.isfinite(score).all():
                raise FloatingPointError("Non-finite activation/objective")

            steps.append(step)
            acts.append(act)
            scores.append(score)

            if best["img"] is None:
                best["img"] = torch.zeros_like(image)
            better = score > best["score"]
            best["img"] = torch.where(better.view(-1, 1, 1, 1, 1), image, best["img"])
            best["score"] = torch.where(better, score, best["score"])
            best["act"] = torch.where(better, act, best["act"])
            best["step"] = torch.where(
                better, torch.full_like(best["step"], step), best["step"]
            )

        for step in range(config_am.iterations + 1):

            # synthesize image from image generator (StereoFFTParam or StereoPixelParam)
            image = param()  # [B, 2, C, H, W]

            final = step == config_am.iterations
            evaluate = final or step % config_am.eval_every == 0

            # Evaluate clean untransformed image
            if evaluate and (final or not identity):
                with torch.no_grad():
                    record(step, image.detach(), target(image.detach(), ch))
            if final:
                break

            # Apply spatial jitter to synthesized image during training
            x = (
                image if identity else apply_jitter(image, rng, config_am)
            )  # [B, 2, C, H, W]

            # compute the activation of neuron/group of neurons in channels
            act = target(x, ch)  # [B]

            if evaluate and identity:  # no transform: reuse this forward pass
                record(step, image.detach(), act.detach())

            # clear gradient
            opt.zero_grad(set_to_none=True)

            # compute the objective function
            # d loss / d params = -sign * (d x/d params)^T d act/d x, via plain autograd
            (-config_am.sign * act + image_penalty(image, config_am)).sum().backward()
            train.append(act.detach())

            # update the generator (image synthesizer) parameters
            opt.step()

        trace, score_trace = (
            torch.stack(acts).cpu(),
            torch.stack(scores).cpu(),
        )  # [n_eval,B]
        train_trace = torch.stack(train).cpu() if train else torch.empty(0, B)
        runs = []
        for j in range(B):
            runs.append(
                dict(
                    img=best["img"][j].cpu(),  # [2, C, H, W]
                    act=float(best["act"][j]),
                    score=float(best["score"][j]),
                    step=int(best["step"][j]),
                    trace=trace[:, j],  # [n_eval]
                    score_trace=score_trace[:, j],  # [n_eval]
                    trace_steps=torch.tensor(steps),  # [n_eval]
                    train_trace=train_trace[:, j],  # [iterations]
                    seed=int(seeds[j]),
                )
            )
        return runs

    @torch.no_grad()
    def eye_ablation(
        self,
        target: Union[CenterTarget, FastDecoderConvTarget],
        images: Float[Tensor, "U 2 C H W"],
        channels: Sequence[int],
        baseline: float,
        chunk: int,
        device: torch.device,
    ) -> Tuple[Float[Tensor, "4 U"], Float[Tensor, "U"]]:
        """
        The eye_ablation function evaluates how neurons integrate visual information
        across both eyes by measuring their activations under four controlled
        occlusion conditions, mirroring monocular occlusion paradigms from
        classical visual electrophysiology.

        Quantifies ocular dominance and non-linear binocular interaction by testing the neuron
            under 4 controlled visual occlusion states:
                1. Both eyes:     (x_L, x_R)           -> R_both
                2. Left eye only: (x_L, baseline)      -> R_L
                3. Right eye only:(baseline, x_R)      -> R_R
                4. Blank gray:    (baseline, baseline) -> R_base

            Computes interaction synergy: Interaction = R_both - R_L - R_R + R_base.
                - ~ 0: Linear monocular summation (independent eye processing).
                - > 0: Facilitation / true disparity-tuned binocular matching.
                - < 0: Binocular suppression, rivalry, or sum-difference suppression.

        Args:
            images:      [U, 2, C, H, W] (optimized stereo pairs on CPU for U channels)
            channels:    Sequence[int] of length U
            Output:
                values:      [4, U] on CPU
                            Row 0: both eyes driven
                            Row 1: left eye driven only
                            Row 2: right eye driven only
                            Row 3: baseline gray
                interaction: [U] on CPU - Binocular synergy metric

        NOTE: for post-BatchNorm3d targets the blank eye also
        changes the volume-wide statistics, which confounds the interaction term.
        """

        out = []
        for a in range(0, len(images), chunk):
            x = images[a : a + chunk].to(device)  # [chunk_B, 2, C, H, W]
            ch = torch.tensor(channels[a : a + chunk], device=device, dtype=torch.long)
            blank = torch.full_like(x, baseline)  # [chunk_B, 2, C, H, W]
            left, right = blank.clone(), blank.clone()
            left[:, 0] = x[:, 0]  # Left eye active, right blank:  [chunk_B, 2, C, H, W]
            right[:, 1] = x[
                :, 1
            ]  # Right eye active, left blank:  [chunk_B, 2, C, H, W]

            # Stack responses across 4 visual conditions: [4, chunk_B]
            out.append(
                torch.stack([target(v, ch) for v in (x, left, right, blank)]).cpu()
            )

        values = torch.cat(out, dim=1)  # [4, U]
        both, left_only, right_only, blank = values  # Each: [U]
        interaction = both - left_only - right_only + blank  # [U]

        return values, interaction

    @torch.no_grad()
    def disparity_profile(self, generic_target, images, W, chunk, device):
        """
        Objective of the FIXED optimised pair at every disparity index: [U, D].
        """
        out = []
        for a in range(0, len(images), chunk):
            out.append(
                generic_target.profile(
                    images[a : a + chunk].to(device), W[a : a + chunk].to(device)
                ).cpu()
            )
        return torch.cat(out)

    def interocular_xcorr(self, img, max_shift):
        """
        Normalised circular cross-correlation of the optimised pair along x.
        c(s) = mean_x L(x) R(x+s) / (std_L std_R): a peak at s>0 means a feature at x in the left
        image sits at x+s in the right image.  Property of the STIMULUS, independent of the net.
        img [2,C,H,W] -> (shifts [2S+1], corr [2S+1]).
        """
        L = img[0] - img[0].mean((-2, -1), keepdim=True)
        R = img[1] - img[1].mean((-2, -1), keepdim=True)
        w = L.shape[-1]
        S = min(int(max_shift), w // 2 - 1)
        cc = torch.fft.irfft(
            torch.fft.rfft(L, dim=-1).conj() * torch.fft.rfft(R, dim=-1), n=w, dim=-1
        ).mean((0, 1))
        cc = cc / (w * L.std() * R.std() + 1e-8)
        return torch.arange(-S, S + 1), torch.cat([cc[w - S :], cc[: S + 1]])

    def input_gradient(self, target, x, channels):
        """
        True raw-pixel gradient of each job's activation: [B, 2, C, H, W], activation [B].
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            act = target(x, channels)
            (g,) = torch.autograd.grad(act.sum(), x)
        return g.detach(), act.detach()

    def mass_bbox(self, energy, mass=0.95):
        """
        Independent eye rectangle with >= requested fraction of squared gradient.
        Returns (y0, y1, x0, x1), exclusive upper bounds.
        None means zero sensitivity.
        Bounding marginal tails at (1-mass)/4 gives a union-bound guarantee.
        """

        energy = energy.detach().double().cpu()
        total = energy.sum()
        if not torch.isfinite(energy).all() or (energy < 0).any():
            raise ValueError("Energy must be finite and nonnegative")
        if total <= 0:
            return None
        q = (1 - mass) / 4

        def extent(p):
            cum = p.cumsum(0) / p.sum()
            lo = min(
                int(
                    torch.searchsorted(
                        cum, torch.tensor(q, dtype=cum.dtype), right=True
                    )
                ),
                len(p) - 1,
            )
            hi = min(
                int(torch.searchsorted(cum, torch.tensor(1 - q, dtype=cum.dtype))),
                len(p) - 1,
            )
            return lo, hi + 1

        y0, y1 = extent(energy.sum(1))
        x0, x1 = extent(energy.sum(0))
        return y0, y1, x0, x1

    def empirical_rf(
        self,
        target,
        images,
        channels,
        n_samples,
        mass,
        seeds,
        chunk,
        device,
        keep_gradient=False,
    ):
        """
        locates where and how widely each eye influences the neuron in physical pixel space,
        revealing its empirical receptive field (RF).

        Per-neuron, per-eye squared TRUE gradient (mean over colour), averaged over the
        optimised image + n_samples noise probes.

        Each probe map is normalised to unit total energy first, so no single probe dominates;
        probes with zero gradient (inactive unit) are skipped.

        A sampled sensitivity map, not a structural RF.
        Because BatchNorm statistics are image-wide, every pixel has a tiny nonzero
        gradient; the mass box therefore measures where the sensitivity concentrates.
        """

        results = []
        for a in range(0, len(images), chunk):

            x0 = images[a : a + chunk].to(device)
            ch = torch.tensor(channels[a : a + chunk], device=device, dtype=torch.long)
            gens = [
                torch.Generator().manual_seed(int(sd)) for sd in seeds[a : a + chunk]
            ]
            acc, cnt, winner = 0.0, 0.0, None

            for sample in range(n_samples + 1):
                if sample == 0:
                    x = x0
                else:
                    noise = torch.stack(
                        [torch.randn(x0.shape[1:], generator=g) for g in gens]
                    )
                    x = (0.5 + 0.2 * noise.to(device)).clamp(0, 1)

                g, _ = self.input_gradient(target, x, ch)
                if sample == 0:
                    winner = g.cpu()

                e = g.square().mean(2)  # [b, 2, H, W]
                tot = e.flatten(1).sum(1)
                valid = (tot > 0).to(e.dtype)
                e = e / tot.clamp_min(torch.finfo(e.dtype).tiny).view(-1, 1, 1, 1)
                acc = acc + e * valid.view(-1, 1, 1, 1)
                cnt = cnt + valid

            energy = (acc / cnt.clamp_min(1).view(-1, 1, 1, 1)).cpu()

            for u in range(energy.shape[0]):
                boxes = [self.mass_bbox(energy[u, e], mass) for e in range(2)]
                captured = []

                for e, box in enumerate(boxes):
                    if box is None:
                        captured.append(None)
                    else:
                        y0, y1, xa, xb = box
                        captured.append(
                            float(energy[u, e, y0:y1, xa:xb].sum() / energy[u, e].sum())
                        )

                results.append(
                    dict(
                        rf_map=energy[u],
                        rf_boxes=boxes,
                        rf_captured_mass=captured,
                        gradient=winner[u] if keep_gradient else None,
                    )
                )

        return results

    def maximize_layer(
        self,
        layer: nn.Module,
        channels,
        normalizer=None,
        call_index=0,
        layer_name="",
    ):

        # validating parameters for activation optimization
        self.config_am.validate()

        channels = list(channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channels must be nonempty and unique")

        device = self.config.device

        if next(self.model.parameters()).device != device:
            raise ValueError("Move model to device before maximizing")

        if normalizer is not None:
            normalizer = normalizer.to(device)

        with analysis_mode(self.model):  # freeze model temporarily
            target = make_target(
                self.model, layer, self.config_am, normalizer, call_index
            )
            bs = self.config_am.batch_size or target.default_batch
            jobs = [(ch, r) for ch in channels for r in range(self.config_am.restarts)]
            runs = {ch: [] for ch in channels}

            # Redirected ReLU (opt-in) only for the optimisation; ablation and
            # sensitivity maps below always use the TRUE gradient / forward.
            with redirected_relu_mode(
                self.model, self.config_am.sign, self.config_am.redirected_relu
            ) as n_patched:
                if self.config_am.redirected_relu and n_patched == 0:
                    warnings.warn(
                        "redirected_relu=True but no nn.ReLU modules found "
                        "(model may use F.relu); gradients are unchanged"
                    )
                for a in range(0, len(jobs), bs):
                    chunk = jobs[a : a + bs]
                    seeds = [self.config_am.seed + ch * 100003 + r for ch, r in chunk]
                    out = self.optimize_batch(
                        target, self.config_am, [ch for ch, _ in chunk], seeds, device
                    )
                    for (ch, _), run in zip(chunk, out):
                        runs[ch].append(run)
                    print(
                        f"{layer_name} call={call_index} [{target.kind}]: "
                        f"{min(a + bs, len(jobs))}/{len(jobs)} jobs",
                        flush=True,
                    )

            best_idx = {
                ch: max(range(len(runs[ch])), key=lambda r: runs[ch][r]["score"])
                for ch in channels
            }
            images = torch.stack([runs[ch][best_idx[ch]]["img"] for ch in channels])
            abl, interaction = self.eye_ablation(
                target, images, channels, self.config_am.baseline, bs, device
            )
            rfs = self.empirical_rf(
                target,
                images,
                channels,
                self.config_am.rf_samples,
                self.config_am.rf_mass,
                [self.config_am.seed + ch * 100003 + 50000 for ch in channels],
                bs,
                device,
                self.config_am.store_gradient,
            )
            shape, d_index = target.output_shape, target.d_index

        implied_disp = None
        if (
            self.config_am.max_disp is not None
            and shape is not None
            and len(shape) == 5
        ):
            implied_disp = index_to_disparity_px(
                d_index, shape[2], self.config_am.max_disp
            )

        units = []
        for u, ch in enumerate(channels):
            rs, bi = runs[ch], best_idx[ch]
            best = rs[bi]
            if all(
                float(r["score_trace"].max()) <= float(r["score_trace"][0]) + 1e-8
                for r in rs
            ):
                warnings.warn(
                    f"{layer_name}, channel {ch}: no objective improvement; "
                    "inspect gradients and restart traces (inactive unit? try "
                    "a pre-activation target or redirected_relu=True)"
                )

            a_u = abl[:, u]
            unit = dict(
                channel=ch,
                img=best["img"],
                act=best["act"],
                score=best["score"],
                best_restart=bi,
                best_step=best["step"],
                trace=best["trace"],
                trace_steps=best["trace_steps"],
                score_trace=best["score_trace"],
                abl=a_u,
                interaction=interaction[u],
                baseline_relative=a_u[:3] - a_u[3],
                output_shape=shape,
                disp_index=d_index,
                # An eye with exactly zero gradient is not driven (e.g. the other eye of
                # a shared monocular encoder call): its pixels are just the random init.
                driven=[b is not None for b in rfs[u]["rf_boxes"]],
                implied_disparity_px=implied_disp,
                restart_traces=torch.stack([r["trace"] for r in rs]),
                train_traces=torch.stack([r["train_trace"] for r in rs]),
                restart_scores=torch.stack([r["score_trace"] for r in rs]),
                restart_best_scores=[r["score"] for r in rs],
                restart_best_steps=[r["step"] for r in rs],
                restart_seeds=[r["seed"] for r in rs],
                **rfs[u],
            )
            if self.config_am.store_restart_images:
                unit["restart_best_images"] = torch.stack([r["img"] for r in rs])
            units.append(unit)
            print(
                f"{layer_name} call={call_index}, channel={ch}: activation="
                f'{best["act"]:.5g}, restart={bi}, step={best["step"]}',
                flush=True,
            )
        return dict(
            channels=channels,
            config_am=asdict(self.config_am),
            layer_name=layer_name,
            call_index=call_index,
            units=units,
            img=torch.stack([u["img"] for u in units]),
            act=torch.tensor([u["act"] for u in units]),
            trace=torch.stack([u["trace"] for u in units], dim=1),
            trace_steps=units[0]["trace_steps"],
            abl=torch.stack([u["abl"] for u in units], dim=1),
            normalization=(
                None
                if normalizer is None
                else {k: v.detach().cpu() for k, v in normalizer.state_dict().items()}
            ),
            ablation_order=["both", "left_only", "right_only", "blank"],
            baseline_relative_order=[
                "both_minus_blank",
                "left_minus_blank",
                "right_minus_blank",
            ],
        )

    @contextmanager
    def _eval_restored(self):
        flags = [(m, m.training) for m in self.model.modules()]
        try:
            self.model.eval()
            yield
        finally:
            for m, flag in flags:
                m.training = flag

    def sanity_checks(
        self,
        layer: nn.Module,
        normalizer: RGBNormalize,
        config: ConfigAM,
        device: torch.device,
        n=2,
        channels=None,
        disp_pxs=None,
        tol=1e-4,
    ):
        """
        1. per-sample norm == original BatchNorm behaviour (B=1);
        2. batched == one-by-one (generic AND fast target: no cross-sample leakage);
        3. fast path == generic hook path (value and input gradient), at several
        disparities including the first depth slice (zero padding along disparity);
        4. optional: Captum NeuronGradient == autograd.
        Relative errors above `tol` are flagged FAIL.
        """
        c, h, w = config.input_size
        n_out = getattr(layer, "out_channels", 1)
        ch_list = (
            list(channels) if channels is not None else [i % n_out for i in range(n)]
        )
        n = len(ch_list)
        g = torch.Generator().manual_seed(1234)
        x = torch.rand(n, 2, c, h, w, generator=g).to(device)
        ch = torch.tensor(ch_list, device=device)
        rep = {}
        ref = x.new_full((1,), float(config.ref))

        def rel(a, b):
            return float((a - b).abs().max()) / (float(b.abs().max()) + 1e-12)

        def full(i):
            l, r = _split_normalize(x[i : i + 1], normalizer)
            return self.model(NestedTensor(left=l, right=r, ref=ref))

        with torch.no_grad(), self._eval_restored():
            orig = [full(i) for i in range(n)]
            with analysis_mode(self.model):
                patched = [full(i) for i in range(n)]
        rep["instance_norm_vs_batchnorm"] = max(
            rel(b, a) for a, b in zip(orig, patched)
        )

        is_vol = isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d))
        if disp_pxs is None:
            m = config.max_disp
            disp_pxs = (0, -m // 4, -m // 2) if (is_vol and m) else (None,)

        with analysis_mode(self.model):
            for k, dpx in enumerate(disp_pxs):
                tag = f"[d={dpx}]"
                base = replace(config, disp_idx=None, disp_px=dpx)
                generic = make_target(
                    self.model, layer, replace(base, fast_path=False), normalizer, 0
                )
                with torch.no_grad():
                    batched = generic(x, ch)
                    single = torch.cat(
                        [generic(x[i : i + 1], ch[i : i + 1]) for i in range(n)]
                    )
                rep[f"generic_batched_vs_single{tag}"] = rel(batched, single)
                gg, aa = self.input_gradient(generic, x, ch)
                fast = make_target(self.model, layer, base, normalizer, 0)
                if fast.kind == "fast":
                    with torch.no_grad():
                        fb = fast(x, ch)
                        fs = torch.cat(
                            [fast(x[i : i + 1], ch[i : i + 1]) for i in range(n)]
                        )
                    rep[f"fast_batched_vs_single{tag}"] = rel(fb, fs)
                    gf, af = self.input_gradient(fast, x, ch)
                    rep[f"fast_vs_generic_act{tag}"] = rel(af, aa)
                    rep[f"fast_vs_generic_grad{tag}"] = rel(gf, gg)
                if k == 0:
                    try:
                        from captum.attr import NeuronGradient

                        class _Wrap(nn.Module):
                            def __init__(self, t):
                                super().__init__()
                                self.t, self.tap = t, nn.Identity()

                            def forward(self, im):
                                return self.tap(
                                    self.t(im, ch[: im.shape[0]]).view(-1, 1)
                                )

                        wrap = _Wrap(generic)
                        xc = x[:1].clone().requires_grad_(True)
                        gc = NeuronGradient(wrap, wrap.tap).attribute(
                            xc, neuron_selector=0
                        )
                        rep["captum_vs_autograd"] = rel(gc, gg[:1])
                    except ImportError:
                        pass
            print(
                f"sanity: target kind for this layer = "
                f"{make_target(self.model, layer, config, normalizer, 0).kind}"
            )
        for key, v in rep.items():
            print(f"sanity {key:44s} {v:.3e}  {'ok' if v <= tol else 'FAIL'}")
        return rep


# %%
config = ConfigGCNet()
config.load_state = True
config.compile_mode = None  # prevent unexpected results
config_am = ConfigAM()
engine = EngineGCNet(config)

am = ActivationMaximization(engine, config_am)
for spec in am.target_layers:

    layer = am.model.get_submodule(spec["module"])
    channels = spec["channels"]
    if channels is None:
        if not hasattr(layer, "out_channels"):
            raise ValueError(
                "Set channels explicitly for composite/nonconvolution modules"
            )
        channels = list(range(layer.out_channels))
    config_am = replace(
        config_am,
        disp_idx=spec.get("disp_idx"),
        disp_px=spec.get("disp_px"),
    )
    result = maximize_layer(
        layer,
        channels,
        config_am,
        normalizer,
        call_index=spec["call"],
        layer_name=spec["module"],
    )
    result["target_spec"] = spec
    if SAVE_RESULTS:
        torch.save(result, am.am_dir / (spec["name"] + ".pt"))
    # per-unit pages: curve | left | right | sensitivity crops | eye ablation
    am.plot_layer(
        spec["name"], result, am.am_dir / (spec["name"] + ".png"), show=SHOW_FIGURES
    )
    # one population overview of the eye ablation for this layer
    am.plot_ablation_summary(
        spec["name"],
        result,
        am.am_dir / (spec["name"] + "_ablation_summary.png"),
        show=SHOW_FIGURES,
    )
