"""Captum NeuronGradient for wundari/rev_depth_networks BNN.

Run from the repository root using the project's Python environment:
  pip install captum matplotlib pillow
  PYTHONPATH=. python /path/to/bnn_neuron_gradient.py \
      --checkpoint model.pth.tar --left left.png --right right.png \
      --layer conv1 --channel 0 --out neuron_gradient

Inputs are an already rectified/aligned pair of RGB files at equal resolution.
No resizing, eye swapping, disparity shifting, or data augmentation is applied.
All derivatives are with respect to RGB pixels scaled to [0,1], BEFORE the
training normalization. Run one pair at a time to preserve input-dependent BN.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from PIL import Image
from captum.attr import NeuronGradient

from BNN.modules.bnn import BNN
from utilities.misc import NestedTensor

LAYERS = ("conv1", "block1", "layer2", "layer3_conv", "layer3", "layer4")
ENCODER_LAYERS = ("conv1", "block1", "layer2")


def normalize(rgb):
    """Match dataset/stereo_albumentation.py for RGB inputs in [0,1]."""
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (rgb - mean) / std


class NeuronView(nn.Module):
    """Expose a single, unambiguous feature tensor to Captum.

    The Identity tap runs once, AFTER the selected operation. This avoids
    hooking an encoder module called twice, or a pre-ReLU tensor subsequently
    modified by an in-place ReLU. Computation stops at the selected layer.
    """

    def __init__(self, model, layer, eye="left", ref=1):
        super().__init__()
        if layer not in LAYERS or eye not in ("left", "right") or ref not in (-1, 1):
            raise ValueError("Invalid layer, eye, or reference")
        self.model, self.layer, self.eye, self.ref = model, layer, eye, ref
        self.tap = nn.Identity()

    def forward(self, left, right):
        if self.layer in ENCODER_LAYERS:
            selected, unused = (left, right) if self.eye == "left" else (right, left)
            x = normalize(selected)
            if self.layer == "conv1":
                a = self.model.encoder.in_conv[0](x)  # before BN/ReLU
            else:
                a = self.model.encoder.in_conv(x)
                if self.layer == "layer2":
                    a = self.model.encoder.layer2(a)  # signed; no ReLU
            # Explicit zero connection lets Captum return a zero map for the
            # unused eye instead of failing on an unused autograd input.
            a = a + 0.0 * unused.sum(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        else:
            reference = left.new_full((left.shape[0],), self.ref)
            features = self.model.encoder(
                NestedTensor(normalize(left), normalize(right), ref=reference)
            )
            volume = self.model.decoder.build_costVol(*features, reference)
            if self.layer == "layer3_conv":
                a = self.model.decoder.layer3[0](volume)  # before BN/ReLU
            else:
                a = self.model.decoder.layer3(volume)
                if self.layer == "layer4":
                    a = self.model.decoder.layer4(a)  # cost BEFORE -cost softmax
        return self.tap(a)


def neuron_gradient(
    model,
    left,
    right,
    layer="conv1",
    channel=0,
    eye="left",
    disparity_index=None,
    y=None,
    x=None,
    ref=1,
):
    """Return ((grad_left, grad_right), metadata) for ONE selected neuron.

    Input/output gradients: [1,3,H,W]. Inputs: finite RGB tensors in [0,1].
    Coordinates y/x are FEATURE-MAP indices, not input-image coordinates.
    disparity_index is a VOLUME INDEX, not disparity in pixels.
    The caller should put model and inputs on the same device.
    """
    if left.shape != right.shape or left.ndim != 4 or left.shape[:2] != (1, 3):
        raise ValueError("Expected matching left/right tensors of shape [1,3,H,W]")
    for rgb in (left, right):
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Inputs must be finite RGB values in [0,1]")
    if min(left.shape[-2:]) < 8:
        raise ValueError("Use images at least 8 pixels high and wide")
    if layer in ENCODER_LAYERS and disparity_index is not None:
        raise ValueError("Encoder activations have no disparity dimension")
    model.eval()
    view = NeuronView(model, layer, eye, ref).eval()
    left = left.detach().clone().requires_grad_(True)
    right = right.detach().clone().requires_grad_(True)
    with torch.no_grad():
        a = view(left, right)
    row = a.shape[-2] // 2 if y is None else y
    col = a.shape[-1] // 2 if x is None else x
    selector = (channel, row, col)
    if a.ndim == 5:
        d = a.shape[2] // 2 if disparity_index is None else disparity_index
        selector = (channel, d, row, col)
    if any(not 0 <= index < size for index, size in zip(selector, a.shape[1:])):
        raise ValueError(
            f"Selector {selector} is outside feature shape {tuple(a.shape)}"
        )

    # Main Captum call: gradients of the tap's selected neuron w.r.t. BOTH inputs.
    # No absolute value, baseline, input multiplication, or noise is applied.
    gradients = NeuronGradient(view, view.tap).attribute(
        inputs=(left, right), neuron_selector=selector
    )
    gradients = tuple(g.detach().cpu() for g in gradients)
    if not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Non-finite gradients; inspect checkpoint/input values")
    metadata = dict(
        layer=layer,
        eye=eye if layer in ENCODER_LAYERS else "both",
        ref=ref,
        selector=list(selector),
        feature_shape=list(a.shape),
        activation=a[(0,) + selector].item(),
        input_shape=list(left.shape),
        derivative_units="per RGB unit in [0,1]",
        gradient_l1=[g.abs().sum().item() for g in gradients],
    )
    return gradients, metadata


def load_model(checkpoint_path, device="cpu", interaction=None):
    """Load project checkpoints or bare state_dicts, without the training engine."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = dict(checkpoint.get("state_dict", checkpoint))
    # Handle wrapped training checkpoints without changing tensor values.
    while any(k.startswith(("module.", "_orig_mod.")) for k in state):
        for prefix in ("module.", "_orig_mod."):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state, prefix
            )
    first = state["encoder.in_conv.0.weight"]
    config = dict(
        in_channels=first.shape[1],
        base_channels=first.shape[0],
        max_disp=int(state["disp_indices"].numel()),
    )
    saved_config = checkpoint.get("config", {})
    mode = interaction or saved_config.get("binocular_interaction")
    if mode is None:
        raise ValueError(
            "Checkpoint has no interaction config. Pass --interaction with "
            "the mode actually used during training."
        )
    if mode not in ("default", "bem", "cmm", "sum_diff"):
        raise ValueError(f"Unknown binocular interaction {mode}")
    config["binocular_interaction"] = mode
    if config["in_channels"] != 3:
        raise ValueError("This example requires an RGB BNN")
    model = BNN(SimpleNamespace(**config))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def read_rgb(path, device="cpu"):
    with Image.open(path) as image:
        a = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def plot_gradients(left, right, gradients, metadata, path, radius=None):
    """Signed RGB and equal-RGB-direction maps with a shared symmetric scale.

    Sum across RGB is d a / d epsilon for adding epsilon to all three channels;
    it can cancel color-opponent responses. Always inspect individual RGB maps.
    Zooms center on each eye's gradient-energy peak independently; coordinates
    are labeled. Neither absolute-value maps nor per-eye rescaling are used.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = [g[0].numpy() for g in gradients]
    maps = [[g.sum(axis=0), *g] for g in arrays]
    scale = max(float(np.abs(m).max()) for row in maps for m in row)
    scale = scale if scale > 0 else 1.0
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for i, (rgb, g, row) in enumerate(zip((left, right), arrays, maps)):
        image = rgb.detach().cpu()[0].permute(1, 2, 0).numpy()
        h, w = image.shape[:2]
        y0, y1, x0, x1 = 0, h, 0, w
        if radius is not None:
            cy, cx = np.unravel_index(np.abs(g).sum(axis=0).argmax(), (h, w))
            if not np.any(g):
                cy, cx = h // 2, w // 2
            y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        eye = ("Left", "Right")[i]
        axes[i, 0].imshow(image[y0:y1, x0:x1], interpolation="nearest")
        axes[i, 0].set_title(f"{eye} RGB\nx=[{x0},{x1}), y=[{y0},{y1})")
        for j, (m, name) in enumerate(
            zip(
                row,
                (
                    "equal-RGB derivative",
                    "R derivative",
                    "G derivative",
                    "B derivative",
                ),
            ),
            1,
        ):
            im = axes[i, j].imshow(
                m[y0:y1, x0:x1],
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
                interpolation="nearest",
            )
            axes[i, j].set_title(name)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(
        im,
        ax=axes[:, 1:].ravel().tolist(),
        shrink=0.8,
        label="signed derivative (red positive / blue negative)",
    )
    title = (
        f"{metadata['layer']} neuron {metadata['selector']} | "
        f"activation={metadata['activation']:.5g} | "
    )
    fig.suptitle(
        title
        + ("independently centered zooms" if radius is not None else "full images")
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--layer", choices=LAYERS, default="conv1")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--eye", choices=("left", "right"), default="left")
    p.add_argument("--disparity-index", type=int)
    p.add_argument("--y", type=int)
    p.add_argument("--x", type=int)
    p.add_argument("--ref", type=int, choices=(-1, 1), default=1)
    p.add_argument("--interaction", choices=("default", "bem", "cmm", "sum_diff"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--zoom-radius", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("neuron_gradient"))
    args = p.parse_args()
    if args.zoom_radius < 0:
        p.error("--zoom-radius must be nonnegative")
    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, args.device, args.interaction)
    left, right = read_rgb(args.left, args.device), read_rgb(args.right, args.device)
    gradients, metadata = neuron_gradient(
        model,
        left,
        right,
        args.layer,
        args.channel,
        args.eye,
        args.disparity_index,
        args.y,
        args.x,
        args.ref,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    metadata.update(
        model_config=config,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    np.savez_compressed(
        args.out / "gradients.npz",
        left=gradients[0].numpy(),
        right=gradients[1].numpy(),
    )
    with (args.out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    plot_gradients(left, right, gradients, metadata, args.out / "full.png")
    plot_gradients(
        left, right, gradients, metadata, args.out / "zoom.png", args.zoom_radius
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved signed derivatives and plots to {args.out.resolve()}")
    if any(v == 0 for v in metadata["gradient_l1"]):
        print(
            "A zero eye map is expected for encoder targets. For decoder targets, "
            "check inactive ReLUs, weights, and the selected unit/input."
        )


if __name__ == "__main__":
    main()
