# %% Editable configuration — no argparse
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import NeuronGradient

from config.config_bnn import ConfigBNN
from engine.engine_bnn import EngineBNN
from utilities.misc import NestedTensor

# %%


@dataclass
class AMSettings:
    input_size: tuple = (3, 64, 64)  # match training dimensions
    restarts: int = 3
    iterations: int = 250
    lr: float = 0.01
    parameterization: str = "fft"  # fft or pixel (both use sigmoid)
    color_mode: str = "rgb"  # rgb or gray
    decay: float = 1.0  # FFT optimization bias; compare 0 and 1
    init_std: float = 0.05
    backend: str = "captum"  # or autograd; mathematically equivalent
    sign: int = 1  # -1 minimizes raw cost / activation
    l2: float = 0.0  # optional penalty around raw gray .5
    tv: float = 0.0  # optional squared spatial differences
    ref: int = 1  # +1 physical left reference, -1 right
    disp_idx: int | None = None  # explicit depth INDEX, default D//2
    max_disp: int | None = None  # only used to REPORT the implied disparity (px)
    window: int = 1  # 1 is one neuron; odd >1 is a region mean
    # --- robustness transforms (identical for both eyes; jitter=0 & scale=(1,1) disables)
    jitter: float = 4.0  # px, random translation
    scale_range: tuple = (0.97, 1.03)  # random isotropic zoom
    eval_every: int = 10  # evaluate the UNtransformed pair every k steps
    # --- opt-in: modifies the BACKWARD pass of nn.ReLU (not the true gradient)
    redirected_relu: bool = False
    seed: int = 0
    rf_samples: int = 4  # additional independent random probes
    rf_mass: float = 0.95
    baseline: float = 0.5  # raw RGB baseline used for ablation

    def validate(self):
        c, h, w = self.input_size
        if c != 3 or min(h, w) < 4 or h % 2 or w % 2:
            raise ValueError("Use RGB and even H,W >=4 for this BNN")
        if self.restarts < 1 or self.iterations < 0 or self.lr <= 0:
            raise ValueError("restarts>=1, iterations>=0, lr>0 required")
        if self.window < 1 or self.window % 2 != 1:
            raise ValueError("window must be a positive odd integer")
        if self.sign not in (-1, 1) or self.ref not in (-1, 1):
            raise ValueError("sign and ref must be +1 or -1")
        if self.backend not in ("captum", "autograd"):
            raise ValueError("backend must be captum or autograd")
        if self.parameterization not in ("fft", "pixel"):
            raise ValueError("parameterization must be fft or pixel")
        if self.color_mode not in ("rgb", "gray"):
            raise ValueError("color_mode must be rgb or gray")
        if min(self.l2, self.tv, self.decay, self.rf_samples) < 0:
            raise ValueError("penalties, decay, and rf_samples must be nonnegative")
        if (
            self.init_std <= 0
            or not 0 < self.rf_mass <= 1
            or not 0 <= self.baseline <= 1
        ):
            raise ValueError("invalid init_std, rf_mass, or baseline")
        if self.jitter < 0 or not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("jitter>=0 and 0<scale_range[0]<=scale_range[1] required")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")


class RGBNormalize(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class StereoFFTParam(nn.Module):
    """Independent eye spectra. Fourier scaling is an optimization prior,
    not a guarantee of natural, non-adversarial, or Gabor-like stimuli.
    Uses the original script's rFFT convention; redundant boundary degrees
    of freedom are harmless for the generated real image.
    """

    def __init__(
        self, size, channels=3, color_mode="rgb", decay=1.0, init_std=0.05, seed=0
    ):
        super().__init__()
        h, w = size
        self.size, self.channels, self.gray = size, channels, color_mode == "gray"
        c = 1 if self.gray else channels
        generator = torch.Generator().manual_seed(seed)
        self.spec = nn.Parameter(
            torch.randn(1, 2, c, h, w // 2 + 1, 2, generator=generator) * init_std
        )
        fy, fx = torch.fft.fftfreq(h)[:, None], torch.fft.rfftfreq(w)[None, :]
        freq = (fx.square() + fy.square()).sqrt().clamp_min(1 / max(h, w))
        self.register_buffer("scale", freq.pow(-decay))

    def forward(self):
        spectrum = torch.view_as_complex(self.spec.contiguous()) * self.scale
        raw = torch.fft.irfft2(spectrum, s=self.size, norm="ortho") / 4
        if self.gray:
            raw = raw.expand(-1, -1, self.channels, -1, -1)
        return raw.sigmoid()


class StereoPixelParam(nn.Module):
    def __init__(self, size, channels=3, color_mode="rgb", init_std=0.05, seed=0):
        super().__init__()
        self.gray, self.channels = color_mode == "gray", channels
        g = torch.Generator().manual_seed(seed)
        self.logits = nn.Parameter(
            torch.randn(1, 2, 1 if self.gray else channels, *size, generator=g)
            * init_std
        )

    def forward(self):
        x = self.logits
        if self.gray:
            x = x.expand(-1, -1, self.channels, -1, -1)
        return x.sigmoid()


def random_affine(x, max_shift_px=4.0, scale=(0.97, 1.03)):
    """Random translation + isotropic zoom on x: [1, 2*C, H, W] (left/right channels
    STACKED, so both eyes get exactly the same transform; a per-eye shift would
    destroy the disparity). No rotation on purpose (it creates vertical disparity).
    Differentiable w.r.t. x."""
    if max_shift_px == 0 and tuple(scale) == (1.0, 1.0):
        return x
    n, _, h, w = x.shape
    dev = x.device
    s = torch.empty(n, device=dev).uniform_(*scale)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (2 * max_shift_px / w)
    ty = (torch.rand(n, device=dev) * 2 - 1) * (2 * max_shift_px / h)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def center_slice(fm, disp_idx=None, window=1):
    if window < 1 or window % 2 != 1:
        raise ValueError("window must be odd and positive")
    if fm.ndim not in (4, 5):
        raise ValueError(f"Expected [N,C,H,W] or [N,C,D,H,W], got {tuple(fm.shape)}")
    h, w = fm.shape[-2:]
    half = window // 2
    y0, x0 = h // 2 - half, w // 2 - half
    if y0 < 0 or x0 < 0 or y0 + window > h or x0 + window > w:
        raise ValueError("window does not fit; reduce it rather than silently clipping")
    if fm.ndim == 5:
        d = fm.shape[2] // 2 if disp_idx is None else disp_idx
        if not 0 <= d < fm.shape[2]:
            raise ValueError(f"disparity index {d} outside [0,{fm.shape[2]})")
        fm = fm[:, :, d]
    elif disp_idx is not None:
        raise ValueError("disp_idx is only meaningful for a 5D target")
    return fm[..., y0 : y0 + window, x0 : x0 + window].mean((-2, -1))


class _StopForward(Exception):
    pass


class CenterTarget(nn.Module):
    """Expose one call of one layer as a [1,1] scalar Captum target.

    Shared encoder: call_index=0 is left, 1 is right in the inspected BNN.
    Decoder layers: call_index=0. No implicit sum over eyes or calls.
    Early exit avoids downstream work and in-place mutation of hooked output.
    Hooks are removed on every forward, including failures.
    The activation of the last forward pass is cached in `last_act` so callers
    do not need a second forward pass just to read it.
    """

    def __init__(
        self,
        model,
        layer,
        channel,
        normalizer=None,
        call_index=0,
        ref=1,
        disp_idx=None,
        window=1,
    ):
        super().__init__()
        if call_index < 0:
            raise ValueError("call_index must be nonnegative")
        self.model = model
        # Bypass registration of a second alias to the same layer.
        object.__setattr__(self, "layer", layer)
        self.channel, self.call_index = channel, call_index
        self.normalizer, self.ref = normalizer, ref
        self.disp_idx, self.window = disp_idx, window
        self.tap = nn.Identity()
        self.output_shape = None
        self.last_act = None

    def forward(self, image):
        if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 2:
            raise ValueError("Input must be exactly one stereo pair [1,2,C,H,W]")
        found, count = [], 0

        def hook(module, inputs, out):
            nonlocal count
            current = count
            count += 1
            if current == self.call_index:
                if not isinstance(out, torch.Tensor):
                    raise TypeError("Select a layer whose output is a single tensor")
                self.output_shape = tuple(out.shape)
                found.append(out.clone())  # preserve preactivation values and graph
                raise _StopForward()

        handle = self.layer.register_forward_hook(hook)
        try:
            left, right = image[:, 0], image[:, 1]
            if self.normalizer is not None:
                left, right = self.normalizer(left), self.normalizer(right)
            self.model(
                NestedTensor(left=left, right=right, ref=image.new_full((1,), self.ref))
            )
        except _StopForward:
            pass
        finally:
            handle.remove()
        if len(found) != 1:
            raise ValueError(
                f"Target call {self.call_index} was not reached (observed {count} calls)"
            )
        value = center_slice(found[0], self.disp_idx, self.window)
        if not 0 <= self.channel < value.shape[1]:
            raise ValueError(f"channel {self.channel} outside [0,{value.shape[1]})")
        out = self.tap(value[:, self.channel : self.channel + 1])
        self.last_act = out.detach()
        return out


@contextmanager
def analysis_mode(model):
    """Freeze model temporarily without changing its normalization definition."""
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    relus = [(m, m.inplace) for m in model.modules() if isinstance(m, nn.ReLU)]
    try:
        model.eval().requires_grad_(False)
        for m, _ in relus:
            m.inplace = False
        yield
    finally:
        for m, flag in modes:
            m.training = flag
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, flag in relus:
            m.inplace = flag


class _RedirectedReLUFn(torch.autograd.Function):
    """Forward = ReLU. Backward additionally lets gradient into INACTIVE units when
    it would move the objective in the wanted direction. Not a true gradient."""

    @staticmethod
    def forward(ctx, x, sign):
        ctx.save_for_backward(x)
        ctx.sign = sign
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        # g = d(activation)/d(relu output); sign*g > 0 means "raising this output
        # moves the objective in the wanted direction".
        return g * ((x > 0) | (ctx.sign * g > 0)).to(g.dtype), None


@contextmanager
def redirected_relu_mode(model, sign=1, enabled=False):
    """Patch forward of every nn.ReLU module (values unchanged, backward changed).
    Does NOT affect functional F.relu calls. Use only for post-ReLU targets."""
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
            del m.forward  # fall back to the class forward


def input_gradient(target, image, backend="captum"):
    """Returns (true raw-pixel gradient of the target activation, activation).

    Gradient and activation come from ONE forward pass (the activation is cached by
    CenterTarget), the gradient is detached; the optimizer applies the chain rule
    through the transform, FFT and sigmoid with a vector-Jacobian product below.
    """
    with torch.enable_grad():
        x = image.detach().requires_grad_(True)
        if backend == "captum":
            g = NeuronGradient(target, target.tap).attribute(x, neuron_selector=0)
            act = target.last_act.reshape(())
        elif backend == "autograd":
            out = target(x)
            (g,) = torch.autograd.grad(out.sum(), x)
            act = out.detach().reshape(())
        else:
            raise ValueError(backend)
    return g.detach(), act


def image_penalty(image, settings):
    return settings.l2 * (image - 0.5).square().mean() + settings.tv * (
        (image[..., 1:, :] - image[..., :-1, :]).square().mean()
        + (image[..., :, 1:] - image[..., :, :-1]).square().mean()
    )


def _optimize_one(target, settings, device, seed):
    c, h, w = settings.input_size
    kwargs = dict(
        size=(h, w),
        channels=c,
        color_mode=settings.color_mode,
        init_std=settings.init_std,
        seed=seed,
    )
    if settings.parameterization == "fft":
        param = StereoFFTParam(**kwargs, decay=settings.decay).to(device)
    else:
        param = StereoPixelParam(**kwargs).to(device)
    optimizer = torch.optim.Adam(param.parameters(), lr=settings.lr)
    steps, trace, scores, train_trace = [], [], [], []
    best = None
    for step in range(settings.iterations + 1):
        image = param()
        # The UNtransformed pair is what we report, so only it selects the best step.
        if step % settings.eval_every == 0 or step == settings.iterations:
            with torch.no_grad():
                act = target(image).item()
                score = settings.sign * act - image_penalty(image, settings).item()
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite activation/objective")
            steps.append(step)
            trace.append(act)
            scores.append(score)
            if best is None or score > best["score"]:
                best = dict(img=image.detach().clone(), act=act, score=score, step=step)
        if step == settings.iterations:
            break
        # Same random shift/zoom for both eyes (eyes stacked on the channel axis).
        x = random_affine(image.flatten(1, 2), settings.jitter, settings.scale_range)
        x = x.reshape(image.shape)
        gradient, act_t = input_gradient(target, x, settings.backend)
        train_trace.append(act_t)
        optimizer.zero_grad(set_to_none=True)
        # d[-sign*(x*g_detached).sum()]/d_parameters is exactly
        # -sign * (d x/d parameters)^T * d activation/d x.
        loss = -settings.sign * (x * gradient).sum() + image_penalty(image, settings)
        loss.backward()
        optimizer.step()
    best.update(
        trace=torch.tensor(trace),
        score_trace=torch.tensor(scores),
        trace_steps=torch.tensor(steps),
        train_trace=(torch.stack(train_trace).cpu() if train_trace else torch.empty(0)),
        seed=seed,
    )
    return best


@torch.no_grad()
def eye_ablation(target, image, baseline=0.5):
    blank = torch.full_like(image, baseline)
    left = blank.clone()
    left[:, 0] = image[:, 0]
    right = blank.clone()
    right[:, 1] = image[:, 1]
    values = torch.stack([target(x).squeeze() for x in (image, left, right, blank)])
    both, lo, ro, base = values
    return values.cpu(), (both - lo - ro + base).cpu()


def mass_bbox(energy, mass=0.95):
    """Independent eye rectangle with >= requested fraction of squared gradient.
    Returns (y0,y1,x0,x1), exclusive upper bounds. None means zero sensitivity.
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
            int(torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype), right=True)),
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


def empirical_rf(target, image, n_samples=4, mass=0.95, seed=0, backend="captum"):
    """Per-neuron, per-eye squared TRUE gradient, averaged across inputs.
    Includes the optimized image plus independent noise probes (batch size 1).
    This is a sampled sensitivity map, not a structural receptive-field bound.
    """
    generator = torch.Generator(device=image.device).manual_seed(seed)
    maps, winner_gradient = [], None
    for sample in range(n_samples + 1):
        if sample == 0:
            x = image
        else:
            x = (
                0.5
                + 0.2
                * torch.randn(image.shape, device=image.device, generator=generator)
            ).clamp(0, 1)
        gradient, _ = input_gradient(target, x, backend)
        if sample == 0:
            winner_gradient = gradient[0].cpu()
        maps.append(
            gradient[0].square().mean(1).cpu()
        )  # [eye,H,W], square before averaging
    energy = torch.stack(maps).mean(0)
    boxes = [mass_bbox(energy[e], mass) for e in range(2)]
    captured = []
    for e, box in enumerate(boxes):
        if box is None:
            captured.append(None)
        else:
            y0, y1, x0, x1 = box
            captured.append(float(energy[e, y0:y1, x0:x1].sum() / energy[e].sum()))
    return dict(
        rf_map=energy,
        rf_boxes=boxes,
        rf_captured_mass=captured,
        gradient=winner_gradient,
    )


def list_candidate_layers(model):
    kinds = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    for name, module in model.named_modules():
        if isinstance(module, kinds):
            print(
                f"{name:48} {type(module).__name__:18} channels={module.out_channels}"
            )


def maximize_layer(
    model,
    layer,
    channels,
    settings=None,
    normalizer=None,
    call_index=0,
    layer_name="",
    device=None,
):
    settings = AMSettings() if settings is None else settings
    settings.validate()
    channels = list(channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must be nonempty and unique")
    device = next(model.parameters()).device if device is None else torch.device(device)
    if next(model.parameters()).device != device:
        raise ValueError("Move model to device before maximizing")
    if normalizer is not None:
        normalizer = normalizer.to(device)
    results = []
    with analysis_mode(model):
        for channel in channels:
            target = CenterTarget(
                model,
                layer,
                channel,
                normalizer,
                call_index,
                settings.ref,
                settings.disp_idx,
                settings.window,
            )
            runs = []
            # Redirected ReLU (opt-in) only for the optimisation; ablation and
            # sensitivity maps below always use the TRUE gradient / forward.
            with redirected_relu_mode(
                model, settings.sign, settings.redirected_relu
            ) as n_patched:
                if settings.redirected_relu and n_patched == 0:
                    warnings.warn(
                        "redirected_relu=True but no nn.ReLU modules found "
                        "(model may use F.relu); gradients are unchanged"
                    )
                for restart in range(settings.restarts):
                    seed = settings.seed + channel * 100003 + restart
                    run = _optimize_one(target, settings, device, seed)
                    run["img"] = run["img"].cpu()  # only one pair resident on GPU
                    runs.append(run)
            best_index = max(range(len(runs)), key=lambda r: runs[r]["score"])
            best = runs[best_index]
            image = best["img"].to(device)
            abl, interaction = eye_ablation(target, image, settings.baseline)
            sensitivity = empirical_rf(
                target,
                image,
                settings.rf_samples,
                settings.rf_mass,
                settings.seed + channel * 100003 + 50000,
                settings.backend,
            )
            # An eye with exactly zero gradient is not driven (e.g. the other eye of a
            # shared monocular encoder call): its pixels are just the random init.
            driven = [b is not None for b in sensitivity["rf_boxes"]]
            implied_disp = None
            shape = target.output_shape
            if settings.max_disp is not None and shape is not None and len(shape) == 5:
                d_count = shape[2]
                d_used = (
                    d_count // 2 if settings.disp_idx is None else settings.disp_idx
                )
                # ASSUMPTION: index i <-> i * max_disp / D px (uniform, starting at 0).
                implied_disp = d_used * settings.max_disp / d_count
            if all(
                float(r["score_trace"].max()) <= float(r["score_trace"][0]) + 1e-8
                for r in runs
            ):
                warnings.warn(
                    f"{layer_name}, channel {channel}: no objective improvement; "
                    "inspect gradients and restart traces (inactive unit? try "
                    "a pre-activation target or redirected_relu=True)"
                )
            results.append(
                dict(
                    channel=channel,
                    img=best["img"][0],
                    act=best["act"],
                    score=best["score"],
                    best_restart=best_index,
                    best_step=best["step"],
                    trace=best["trace"],
                    trace_steps=best["trace_steps"],
                    score_trace=best["score_trace"],
                    abl=abl,
                    interaction=interaction,
                    baseline_relative=abl[:3] - abl[3],
                    output_shape=target.output_shape,
                    driven=driven,
                    implied_disparity_px=implied_disp,
                    restart_traces=torch.stack([r["trace"] for r in runs]),
                    train_traces=torch.stack([r["train_trace"] for r in runs]),
                    restart_scores=torch.stack([r["score_trace"] for r in runs]),
                    restart_best_images=torch.stack([r["img"][0] for r in runs]),
                    restart_best_scores=[r["score"] for r in runs],
                    restart_best_steps=[r["step"] for r in runs],
                    restart_seeds=[r["seed"] for r in runs],
                    **sensitivity,
                )
            )
            print(
                f"{layer_name} call={call_index}, channel={channel}: "
                f'activation={best["act"]:.5g}, restart={best_index}, step={best["step"]}',
                flush=True,
            )
    # Preserve familiar top-level plotting/data keys, but add explicit semantics.
    return dict(
        channels=channels,
        settings=asdict(settings),
        layer_name=layer_name,
        call_index=call_index,
        units=results,
        img=torch.stack([r["img"] for r in results]),
        act=torch.tensor([r["act"] for r in results]),
        trace=torch.stack([r["trace"] for r in results], dim=1),
        trace_steps=results[0]["trace_steps"],
        abl=torch.stack([r["abl"] for r in results], dim=1),
        normalization=(
            None
            if normalizer is None
            else {k: v.detach().cpu() for k, v in normalizer.state_dict().items()}
        ),
        ablation_order=["both", "left_only", "right_only", "both_baseline"],
        baseline_relative_order=[
            "both_minus_baseline",
            "left_minus_baseline",
            "right_minus_baseline",
        ],
    )


def plot_layer(
    name, result, path=None, rows_per_page=6, show=False, stretch=False, dpi=600
):
    import matplotlib.pyplot as plt

    if rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")
    units = result["units"]
    saved = []
    for page, start in enumerate(range(0, len(units), rows_per_page)):
        subset = units[start : start + rows_per_page]
        fig, axes = plt.subplots(
            len(subset), 5, figsize=(16, 3.3 * len(subset)), squeeze=False
        )
        for row, u in enumerate(subset):
            ax = axes[row]
            steps = u["trace_steps"].numpy()
            for train, ident in zip(u["train_traces"], u["restart_traces"]):
                ax[0].plot(train.numpy(), color="0.8", lw=0.8)
                ax[0].plot(steps, ident.numpy(), lw=1.2)
            ax[0].scatter([u["best_step"]], [u["act"]], color="black", s=14, zorder=3)
            a, l, r, b = u["abl"].tolist()
            disp = (
                ""
                if u["implied_disparity_px"] is None
                else f"; disp≈{u['implied_disparity_px']:.0f}px"
            )
            ax[0].set_title(
                f'ch {u["channel"]}: A={a:.3g}; baseline={b:.3g}{disp}\n'
                f'L={l:.3g}; R={r:.3g}; interaction={u["interaction"]:.3g}',
                fontsize=9,
            )
            ax[0].set(
                xlabel="Step (gray: jittered input; colour: untransformed pair)",
                ylabel="Raw activation",
            )
            img = u["img"]
            if stretch:
                sel = [img[e] for e in range(2) if u["driven"][e]]
                if sel:
                    lo = min(float(s.min()) for s in sel)
                    hi = max(float(s.max()) for s in sel)
                    img = ((img - lo) / (hi - lo + 1e-8)).clamp(0, 1)
            for eye, label in enumerate(("Physical left", "Physical right")):
                image = img[eye].permute(1, 2, 0).numpy()
                if u["driven"][eye]:
                    ax[1 + eye].imshow(image, vmin=0, vmax=1)
                else:
                    ax[1 + eye].text(
                        0.5,
                        0.5,
                        "not driven\n(zero gradient)",
                        ha="center",
                        va="center",
                        transform=ax[1 + eye].transAxes,
                    )
                ax[1 + eye].set_title(
                    label + (" [stretched]" if stretch else " [raw 0–1]"), fontsize=9
                )
                box = u["rf_boxes"][eye]
                if box is None:
                    ax[3 + eye].text(
                        0.5,
                        0.5,
                        "No sampled sensitivity",
                        ha="center",
                        va="center",
                        transform=ax[3 + eye].transAxes,
                    )
                else:
                    y0, y1, x0, x1 = box
                    ax[3 + eye].imshow(image[y0:y1, x0:x1], vmin=0, vmax=1)
                    ax[3 + eye].set_title(
                        f"{label} sensitivity crop\ny[{y0}:{y1}] x[{x0}:{x1}]",
                        fontsize=9,
                    )
                ax[1 + eye].axis("off")
                ax[3 + eye].axis("off")
        fig.suptitle(
            f'{name}: call {result["call_index"]}; per-eye sensitivity crops'
            + ("; joint contrast stretch" if stretch else "; no contrast stretching")
        )
        fig.tight_layout()
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            filename = path.with_name(path.stem + f"_p{page}" + (path.suffix or ".png"))
            fig.savefig(filename, dpi=dpi)
            saved.append(filename)
        if show:
            plt.show()
        plt.close(fig)
    return saved


# %%


MAX_DISP = 192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SETTINGS = AMSettings(
    input_size=(3, 64, 64),  # repository training crop; don't shrink silently
    restarts=3,
    iterations=250,
    lr=0.01,
    parameterization="fft",
    color_mode="rgb",  # gray is a separate, constrained experiment
    decay=1.0,
    backend="captum",  # "captum" or "autograd" change to autograd for the equivalence control
    ref=1,
    window=1,
    max_disp=MAX_DISP,  # only for reporting the implied disparity of disp_idx
    jitter=4.0,  # 0 reproduces the old (no-transform) behaviour
    scale_range=(0.97, 1.03),
    eval_every=10,
    redirected_relu=False,  # opt-in, for post-ReLU targets with dead units
    seed=0,
    rf_samples=4,
)

# Each entry targets ONE invocation of a module. channels=None means all.
# Shared encoder call 0=physical left, call 1=physical right (the other eye is then
# "not driven" and is shown as such). Start with the same layer as before.
LAYER_TARGETS = [
    dict(
        name="decoder_layer3_pre",
        module="decoder.layer3.0",
        call=0,
        channels=None,
        disp_idx=48,  # sweep this for decoder layers (None = D//2, NOT zero disparity)
        sign=1,
    ),
    # dict(name='left_conv1_pre', module='encoder.in_conv.0', call=0, channels=None, disp_idx=None, sign=1),
    # dict(name='right_conv1_pre', module='encoder.in_conv.0', call=1, channels=None, disp_idx=None, sign=1),
    # dict(name='left_layer1_post', module='encoder.in_conv', call=0, channels=list(range(32)), disp_idx=None, sign=1),
    # dict(name='right_layer1_post', module='encoder.in_conv', call=1, channels=list(range(32)), disp_idx=None, sign=1),
    # dict(name='left_layer2', module='encoder.layer2.0', call=0, channels=None, disp_idx=None, sign=1),
    # dict(name='right_layer2', module='encoder.layer2.0', call=1, channels=None, disp_idx=None, sign=1),
    # dict(name='decoder_layer3_post', module='decoder.layer3', call=0, channels=list(range(32)), disp_idx=None, sign=1),
    # The final layer has one channel. +1 maximizes raw COST; -1 minimizes cost.
    # dict(name='decoder_layer4_cost', module='decoder.layer4', call=0, channels=[0], disp_idx=None, sign=1),
]
SHOW_FIGURES = False
SAVE_RESULTS = True


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# More reproducible kernels where available; warnings expose unsupported ops.
torch.use_deterministic_algorithms(True, warn_only=True)

# %% set up DNN
config = ConfigBNN()
config.load_state = True
engine = EngineBNN(config)

# Unwrap a torch.compile'd model: hooks that raise (early exit) are not compile-safe.
model = engine.model._orig_mod if hasattr(engine.model, "_orig_mod") else engine.model
model.to(DEVICE)
normalizer = RGBNormalize().to(DEVICE)  # must match the BNN's training preprocessing
list_candidate_layers(model)

OUTPUT_DIR = Path(engine.save_dir) / "activation_maximization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% feature visualization
for spec in LAYER_TARGETS:
    layer = model.get_submodule(spec["module"])
    channels = spec["channels"]
    if channels is None:
        if not hasattr(layer, "out_channels"):
            raise ValueError(
                "Set channels explicitly for composite/nonconvolution modules"
            )
        channels = list(range(layer.out_channels))
    settings = replace(SETTINGS, disp_idx=spec["disp_idx"], sign=spec["sign"])
    result = maximize_layer(
        model,
        layer,
        channels,
        settings,
        normalizer,
        call_index=spec["call"],
        layer_name=spec["module"],
    )

    result["target_spec"] = spec
    if SAVE_RESULTS:
        torch.save(result, OUTPUT_DIR / (spec["name"] + ".pt"))
    plot_layer(
        spec["name"],
        result,
        OUTPUT_DIR / (spec["name"] + ".png"),
        show=SHOW_FIGURES,
    )

# %%
