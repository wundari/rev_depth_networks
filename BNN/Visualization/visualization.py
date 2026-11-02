# %%
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

from utilities.misc import NestedTensor

# Load necessary modules from your repository
from config.config_bnn import ConfigBNN
from engine.engine_bnn import EngineBNN


# --------------------------------------------------------------------------- #
# Image Normalization
# --------------------------------------------------------------------------- #
class RGBNormalize(nn.Module):
    """
    Applies channel-specific mean/std normalization inside the forward pass.
    Adjust these values to match your dataset's exact preprocessing stats.
    """

    def __init__(
        self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), device="cuda"
    ):
        super().__init__()
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def center_activation(fm, disp_idx=None, window_size=3):
    """Calculates the mean activation over a small center spatial window."""
    half_win = window_size // 2

    if fm.dim() == 4:
        h, w = fm.shape[-2:]
        ch, cw = h // 2, w // 2
        return fm[
            0, 0, ch - half_win : ch + half_win + 1, cw - half_win : cw + half_win + 1
        ].mean()

    if fm.dim() == 5:
        d, h, w = fm.shape[-3:]
        di = d // 2 if disp_idx is None else disp_idx
        ch_idx, cw_idx = h // 2, w // 2
        return fm[
            0,
            0,
            di,
            ch_idx - half_win : ch_idx + half_win + 1,
            cw_idx - half_win : cw_idx + half_win + 1,
        ].mean()

    raise ValueError(f"Unexpected feature map shape: {tuple(fm.shape)}")


def tv_loss(img):
    tv_h = torch.abs(img[..., 1:, :] - img[..., :-1, :]).mean()
    tv_w = torch.abs(img[..., :, 1:] - img[..., :, :-1]).mean()
    return tv_h + tv_w


# --------------------------------------------------------------------------- #
# Core Optimization (Independent per Channel)
# --------------------------------------------------------------------------- #
def maximize_single_channel(
    model,
    target_layer,
    channel_idx,
    input_size=(3, 64, 64),
    iterations=300,
    lr=0.05,
    tv_weight=0.1,
    l2_weight=0.01,
    restarts=3,
    disp_idx=None,
    device="cuda",
    normalizer=None,
):
    """
    Optimizes a single channel independently with multiple random restarts.
    """
    best_act = -float("inf")
    best_img = None
    best_trace = []

    captured = {}
    handle = target_layer.register_forward_hook(
        lambda m, i, o: captured.__setitem__(
            "out", o[:, channel_idx : channel_idx + 1, ...]
        )
    )

    ref = torch.ones(1, device=device)

    try:
        for r in range(restarts):
            # Parameterize in inverse-sigmoid space so images naturally bound to [0, 1]
            w = torch.randn(2, 1, *input_size, device=device) * 0.1
            w.requires_grad_(True)
            optimizer = optim.Adam([w], lr=lr)

            trace = []

            for it in range(iterations):
                optimizer.zero_grad(set_to_none=True)

                # Constrain to [0, 1] pixel space
                img = torch.sigmoid(w)

                # Apply dataset normalization
                norm_img = normalizer(img) if normalizer else img

                model(NestedTensor(left=norm_img[0], right=norm_img[1], ref=ref))
                act = center_activation(captured["out"], disp_idx)

                loss = -act
                if tv_weight > 0:
                    loss = loss + tv_weight * tv_loss(img)
                if l2_weight > 0:
                    loss = loss + l2_weight * img.pow(2).mean()

                loss.backward()
                optimizer.step()

                trace.append(act.item())

            final_act = trace[-1]
            if final_act > best_act:
                best_act = final_act
                best_img = torch.sigmoid(w).detach().cpu()
                best_trace = trace
    finally:
        handle.remove()

    return best_img[0, 0], best_img[1, 0], best_act, best_trace


# --------------------------------------------------------------------------- #
# Plotting with Crops and Traces
# --------------------------------------------------------------------------- #
def plot_channel_results(name, results_dict, crop_size=24, path=None):
    """
    Plots [Trace | Left Full | Right Full | Left Crop | Right Crop]
    with fixed display scaling [0, 1].
    """
    n = len(results_dict)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = [axes]

    for idx, (ch, (left, right, act, trace)) in enumerate(results_dict.items()):
        ax_trace, ax_lf, ax_rf, ax_lc, ax_rc = axes[idx]

        # 1. Convergence Trace
        ax_trace.plot(trace, color="blue", linewidth=2)
        ax_trace.set_title(f"Ch {ch} Trace\nFinal Act: {act:.2f}")
        ax_trace.set_xlabel("Iterations")
        ax_trace.set_ylabel("Activation")

        # Images are strictly [0, 1] -> permute to (H, W, C)
        l_img = left.permute(1, 2, 0).numpy()
        r_img = right.permute(1, 2, 0).numpy()

        # Center coordinates for cropping
        h, w = l_img.shape[:2]
        ch_y, ch_x = h // 2, w // 2
        hw = crop_size // 2

        l_crop = l_img[ch_y - hw : ch_y + hw, ch_x - hw : ch_x + hw, :]
        r_crop = r_img[ch_y - hw : ch_y + hw, ch_x - hw : ch_x + hw, :]

        # 2 & 3. Full Images
        ax_lf.imshow(l_img, vmin=0, vmax=1)
        ax_lf.set_title("Left (Full)")
        ax_lf.axis("off")

        ax_rf.imshow(r_img, vmin=0, vmax=1)
        ax_rf.set_title("Right (Full)")
        ax_rf.axis("off")

        # 4 & 5. Enlarged Crops
        ax_lc.imshow(l_crop, vmin=0, vmax=1)
        ax_lc.set_title(f"Left Center Crop")
        ax_lc.axis("off")

        ax_rc.imshow(r_crop, vmin=0, vmax=1)
        ax_rc.set_title(f"Right Center Crop")
        ax_rc.axis("off")

    fig.suptitle(
        f"{name} Activation Maximization (Independent & Restarted)", fontsize=14
    )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    plt.show()


# %% set up DNN
config = ConfigBNN()
config.load_state = True
engine = EngineBNN(config)

# %% feature visualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = (
    engine.model._orig_mod if hasattr(engine.model, "_orig_mod") else engine.model
)
base_model.to(device)
base_model.eval()

# FIX 2: Freeze all model weights. We only want to calculate gradients for the input image.
for param in base_model.parameters():
    param.requires_grad = False

single_eye_input_size = (3, 64, 64)

# Init the Normalizer based on your dataset
normalizer = RGBNormalize(device=device)

layers_to_probe = {"decoder.layer3": base_model.decoder.layer3[0]}

# We will probe a subset of channels for demonstration (e.g., first 3)
channels_to_probe = [0, 1, 2]

for layer_name, layer in layers_to_probe.items():
    print(f"Optimizing {layer_name} ...")

    results = {}
    for ch in range(32):
        print(f"  -> Processing Channel {ch}...")
        left, right, act, trace = maximize_single_channel(
            model=base_model,
            target_layer=layer,
            channel_idx=ch,
            input_size=single_eye_input_size,
            iterations=250,
            restarts=3,  # Run 3 times, keep the highest activation
            tv_weight=0.25,  # Regularization
            l2_weight=0.01,
            device=device,
            normalizer=normalizer,
        )
        results[ch] = (left, right, act, trace)

    plot_channel_results(
        layer_name, results, crop_size=24, path=f"actmax_{layer_name}_revised.png"
    )
