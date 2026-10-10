# %%
"""
Temporal integration analysis for RDSs.

Main quantities
---------------
X_t:
    binocular RDS at frame t

Y_t:
    reference eye
        +1 = left
        -1 = right

Z_t:
    network disparity in reference-relative coordinates

Canonical disparity:
    Z_tilde = Y_t * Z_t

For P(Y=left) = P(Y=right) = 0.5:
    E[Z_tilde | X_t]
        = 0.5 * E[Z | X_t, left]
        - 0.5 * E[Z | X_t, right]

Temporal mean:
    E[Z_bar_T | X_1:T]
        = (1 / T) * sum_t E[Z_tilde | X_t]
"""

from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utilities.misc import NestedTensor
from jaxtyping import Float


# ============================================================
# Basic utilities
# ============================================================
def unwrap_model(model):
    """
    Handle torch.compile models.
    """
    return getattr(model, "_orig_mod", model)


def expectation_from_posterior(
    posterior: Float[Tensor, "B D H W"] | Float[Tensor, "B D"],
    disparity_values: Float[Tensor, "D"],
) -> Float[Tensor, "B H W"] | Float[Tensor, "B"]:
    """
    Parameters
    ----------
        posterior: [B, D, H, W] | [B, D]
        disparity_values: [D]

    Returns
    -------
        expected disparity: [B, H, W] | [B]
    """

    shape = [1, -1] + [1] * (posterior.ndim - 2)
    d = disparity_values.view(*shape).to(
        posterior.device,
        posterior.dtype,
    )

    return torch.sum(posterior * d, dim=1)


def spatial_mean(
    x: Float[Tensor, "B H W"],
    roi=None,
) -> Float[Tensor, "B"]:
    """
    Spatially average a disparity map.

    x: [B, H, W]
    roi: (y0, y1, x0, x1)

    Returns
    -------
    [B]
    """

    if roi is not None:
        y0, y1, x0, x1 = roi
        x = x[..., y0:y1, x0:x1]

    return x.mean(dim=(-2, -1))


def posterior_spatial_mean(
    posterior: Float[Tensor, "B D H W"],
    roi=None,
) -> Tensor:
    """
    Spatially pool a disparity posterior.

    posterior: [B, D, H, W]

    Returns
    -------
    [B, D]
    """

    if roi is not None:
        y0, y1, x0, x1 = roi
        posterior = posterior[..., y0:y1, x0:x1]

    posterior = posterior.mean(dim=(-2, -1))

    # Numerical safety.
    posterior = posterior / posterior.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    return posterior


# ============================================================
# Reference-eye canonicalization
# ============================================================
def canonicalize_reference_posteriors(
    posterior_left: Float[Tensor, "B D H W"],
    posterior_right: Float[Tensor, "B D H W"],
    disparity_values: Float[Tensor, "D"],
):
    """
    Convert left- and right-reference distributions to one
    common physical disparity convention.

    Assume:
        Z_left  = x_left  - x_right
        Z_right = x_right - x_left

    Therefore:
        Z_common = Z_left
        Z_common = -Z_right

    IMPORTANT:
    GC-Net currently has disparities:
        [-96, ..., 95]

    A simple torch.flip() is therefore not an exact sign
    reversal, because +96 does not exist in the original grid.

    We instead construct the union:
        [-96, ..., 96]

    Parameters
    ----------
    posterior_left:
        [B, D, H, W]

    posterior_right:
        [B, D, H, W]

    disparity_values:
        [D]

    Returns
    -------
    p_left_common:
        [B, D_common, H, W]

    p_right_common:
        [B, D_common, H, W]

    common_disp:
        [D_common]
    """

    if posterior_left.shape != posterior_right.shape:
        raise ValueError("Left/right posterior shapes must match")

    d = disparity_values.flatten()

    if not torch.allclose(d, torch.round(d)):
        raise ValueError("This implementation assumes integer disparity bins")

    d = d.round().long()

    # Left-reference support.
    d_left = d

    # Right-reference distribution is reflected:
    # d_common = -d_right
    d_right_common = -d

    min_disp = min(int(d_left.min()), int(d_right_common.min()))
    max_disp = max(int(d_left.max()), int(d_right_common.max()))

    common_disp = torch.arange(
        min_disp,
        max_disp + 1,
        device=posterior_left.device,
        dtype=posterior_left.dtype,
    )

    n_common = common_disp.numel()

    output_shape = (
        posterior_left.shape[0],
        n_common,
        *posterior_left.shape[2:],
    )

    p_left_common = torch.zeros(
        output_shape,
        device=posterior_left.device,
        dtype=posterior_left.dtype,
    )

    p_right_common = torch.zeros_like(p_left_common)

    # Map original bins into common coordinates.
    idx_left = (d_left - min_disp).long()
    idx_right = (d_right_common - min_disp).long()

    p_left_common.index_copy_(1, idx_left, posterior_left)
    p_right_common.index_copy_(1, idx_right, posterior_right)

    return (
        p_left_common,
        p_right_common,
        common_disp,
    )


# ============================================================
# GC-Net inference for BOTH eye references
# ============================================================
@torch.inference_mode()
def predict_both_references(
    model,
    left: Float[Tensor, "B C H W"],
    right: Float[Tensor, "B C H W"],
):
    """
    Run the SAME RDS through GC-Net twice:

        Y = +1 -> left reference
        Y = -1 -> right reference

    Both are evaluated in one forward pass.

    Parameters
    ----------
    left, right:
        [B, C, H, W]

    Returns
    -------
    dict with:
        posterior_left
        posterior_right
        mu_left
        mu_right
        disparity_values
    """

    net = unwrap_model(model)
    device = next(net.parameters()).device

    left = left.to(device, non_blocking=True)
    right = right.to(device, non_blocking=True)
    B = left.shape[0]

    # --------------------------------------------
    # Duplicate each image pair.
    # first B samples: left reference
    # second B samples: right reference
    # --------------------------------------------
    left_all = torch.cat([left, left], dim=0)
    right_all = torch.cat([right, right], dim=0)
    ref = torch.cat(
        [
            torch.ones(
                B,
                device=device,
                dtype=torch.long,
            ),
            -torch.ones(
                B,
                device=device,
                dtype=torch.long,
            ),
        ],
        dim=0,
    )
    x = NestedTensor(left=left_all, right=right_all, ref=ref)

    was_training = net.training
    net.eval()
    try:
        # Encoder
        feat_left, feat_right = net.encoder(x)

        # Decoder directly returns:
        # p(d | X, Y)
        posterior = net.decoder(feat_left, feat_right, ref)

    finally:
        net.train(was_training)

    posterior_left = posterior[:B]
    posterior_right = posterior[B:]
    disparity_values = net.disp_indices.reshape(-1).to(
        device=device,
        dtype=posterior.dtype,
    )

    mu_left = expectation_from_posterior(posterior_left, disparity_values)
    mu_right = expectation_from_posterior(posterior_right, disparity_values)

    return {
        "posterior_left": posterior_left,
        "posterior_right": posterior_right,
        "mu_left": mu_left,
        "mu_right": mu_right,
        "disparity_values": disparity_values,
    }


# ============================================================
# Temporal integration
# ============================================================
@torch.inference_mode()
def analyze_temporal_batch(
    model,
    left_sequence: Float[Tensor, "B T C H W"],
    right_sequence: Float[Tensor, "B T C H W"],
    reference_mode: Literal[
        "marginalize",
        "sample",
        "left",
        "right",
    ] = "marginalize",
    p_left: float = 0.5,
    roi=None,
    random_seed: int = 3407,
    return_final_map: bool = False,
    eps: float = 1e-10,
):
    """
    Temporal analysis for one batch of RDS sequences.

    Parameters
    ----------
    left_sequence, right_sequence:
        [B, T, C, H, W]

    reference_mode:

        "marginalize"
            Exact expectation over reference:
                p(z | X)
                = p_left * p(z | X, left)
                + (1-p_left) * p(z | X, right)

            RECOMMENDED for computing E_Y[Z].

        "sample"
            At each frame/sample, randomly choose one
            reference eye.

            This implements the Monte-Carlo model proposed
            in the temporal-reference hypothesis.

        "left"
            Always left reference.

        "right"
            Always right reference.

    roi:
        Optional (y0, y1, x0, x1).
        For your central RDS region with 256x512 images:
            roi = (64, 192, 128, 384)

    Returns
    -------
    Dictionary containing framewise and cumulative quantities.
    """

    if left_sequence.shape != right_sequence.shape:
        raise ValueError("Left/right sequence shapes must match")

    if left_sequence.ndim != 5:
        raise ValueError("Expected [B, T, C, H, W]")

    if not 0.0 <= p_left <= 1.0:
        raise ValueError("p_left must lie in [0,1]")

    B, T, _, _, _ = left_sequence.shape
    device = next(unwrap_model(model).parameters()).device

    # CPU RNG makes sampled eye-reference sequences
    # reproducible independently of CUDA.
    rng = torch.Generator()
    rng.manual_seed(random_seed)

    # --------------------------------------------------------
    # Quantities saved frame-by-frame
    # --------------------------------------------------------
    mu_left_raw_all = []
    mu_right_raw_all = []
    mu_left_common_all = []
    mu_right_common_all = []
    mu_ref_integrated_all = []
    reference_symmetry_error_all = []
    sampled_refs_all = []

    # --------------------------------------------------------
    # Running temporal accumulators
    # --------------------------------------------------------
    running_mu_roi = None
    running_posterior = None
    running_log_evidence = None
    running_map = None

    temporal_mean_curve = []
    posterior_mix_curve = []
    posterior_product_curve = []

    final_common_disp = None

    # ========================================================
    # TIME LOOP
    # ========================================================
    for t in range(T):

        left_t = left_sequence[:, t]
        right_t = right_sequence[:, t]
        pred = predict_both_references(model, left_t, right_t)

        p_left_raw = pred["posterior_left"]
        p_right_raw = pred["posterior_right"]
        d_raw = pred["disparity_values"]

        # ----------------------------------------------------
        # Raw expectations
        # ----------------------------------------------------
        mu_left_raw = pred["mu_left"]
        mu_right_raw = pred["mu_right"]

        # ----------------------------------------------------
        # Canonicalize reference eye.
        # Left: d_common = d_left
        # Right: d_common = -d_right
        # ----------------------------------------------------
        (
            p_left_common,
            p_right_common,
            common_disp,
        ) = canonicalize_reference_posteriors(
            p_left_raw,
            p_right_raw,
            d_raw,
        )

        final_common_disp = common_disp

        mu_left_common = expectation_from_posterior(
            p_left_common,
            common_disp,
        )
        mu_right_common = expectation_from_posterior(
            p_right_common,
            common_disp,
        )

        # ----------------------------------------------------
        # Reference-eye integration
        # ----------------------------------------------------
        if reference_mode == "marginalize":
            p_t = p_left * p_left_common + (1.0 - p_left) * p_right_common
            sampled_ref = None

        elif reference_mode == "sample":
            choose_left_cpu = torch.rand(B, generator=rng) < p_left
            choose_left = choose_left_cpu.to(device=device).view(B, 1, 1, 1)
            p_t = torch.where(choose_left, p_left_common, p_right_common)

            sampled_ref = torch.where(
                choose_left_cpu,
                torch.ones(
                    B,
                    dtype=torch.int64,
                ),
                -torch.ones(
                    B,
                    dtype=torch.int64,
                ),
            )

            sampled_refs_all.append(sampled_ref)

        elif reference_mode == "left":
            p_t = p_left_common
            sampled_ref = None

        elif reference_mode == "right":
            p_t = p_right_common
            sampled_ref = None

        else:
            raise ValueError(f"Unknown reference_mode: " f"{reference_mode}")

        # ----------------------------------------------------
        # E[Z_tilde_t | X_t]
        # ----------------------------------------------------
        mu_t = expectation_from_posterior(p_t, common_disp)

        # ----------------------------------------------------
        # ROI values
        # ----------------------------------------------------
        mu_left_raw_roi = spatial_mean(mu_left_raw, roi)
        mu_right_raw_roi = spatial_mean(mu_right_raw, roi)
        mu_left_common_roi = spatial_mean(mu_left_common, roi)
        mu_right_common_roi = spatial_mean(mu_right_common, roi)
        mu_t_roi = spatial_mean(mu_t, roi)

        # If perfect reference anti-symmetry holds:
        # mu_left_raw = -mu_right_raw

        # Equivalently, after canonicalization:
        # mu_left_common = mu_right_common
        ref_symmetry_error = spatial_mean(
            torch.abs(mu_left_common - mu_right_common),
            roi,
        )

        mu_left_raw_all.append(mu_left_raw_roi.cpu())
        mu_right_raw_all.append(mu_right_raw_roi.cpu())
        mu_left_common_all.append(mu_left_common_roi.cpu())
        mu_right_common_all.append(mu_right_common_roi.cpu())
        mu_ref_integrated_all.append(mu_t_roi.cpu())

        reference_symmetry_error_all.append(ref_symmetry_error.cpu())

        # ----------------------------------------------------
        # Pool posterior spatially: [B,D,H,W] -> [B,D]
        # This keeps temporal probability integration cheap.
        # ----------------------------------------------------
        p_t_roi = posterior_spatial_mean(p_t, roi)

        # ====================================================
        # Temporal integration method A: arithmetic mean of expected disparity
        # 1/t sum E[Z_t]
        # ====================================================
        if running_mu_roi is None:
            running_mu_roi = mu_t_roi
        else:
            running_mu_roi = running_mu_roi + mu_t_roi

        mean_t = running_mu_roi / float(t + 1)
        temporal_mean_curve.append(mean_t.cpu())

        # ====================================================
        # Temporal integration method B: arithmetic mixture of posteriors
        # p_T(z) = 1/T sum p_t(z)
        # ====================================================
        if running_posterior is None:
            running_posterior = p_t_roi.clone()
        else:
            running_posterior += p_t_roi

        posterior_mix = running_posterior / float(t + 1)
        posterior_mix_mean = expectation_from_posterior(
            posterior_mix,
            common_disp,
        )
        posterior_mix_curve.append(posterior_mix_mean.cpu())

        # ====================================================
        # Temporal integration method C: product of evidence
        #
        # log p_T(z)
        #     proportional to
        #     sum_t log p_t(z)
        #
        # NOTE:
        # treat this as an evidence-accumulation model,
        # not as ordinary temporal averaging.
        # ====================================================

        log_p = torch.log(p_t_roi.clamp_min(eps))

        if running_log_evidence is None:
            running_log_evidence = log_p.clone()
        else:
            running_log_evidence += log_p

        product_posterior = torch.softmax(
            running_log_evidence,
            dim=1,
        )
        product_mean = expectation_from_posterior(
            product_posterior,
            common_disp,
        )
        posterior_product_curve.append(product_mean.cpu())

        # ----------------------------------------------------
        # Optional complete temporally averaged disparity map
        # ----------------------------------------------------
        if return_final_map:

            if running_map is None:
                running_map = mu_t.clone()
            else:
                running_map += mu_t

    # ========================================================
    # Assemble results
    # ========================================================
    result = {
        # Raw reference-specific expectation, [B,T]
        "mu_left_raw": torch.stack(mu_left_raw_all, dim=1),
        "mu_right_raw": torch.stack(mu_right_raw_all, dim=1),
        # After common-coordinate transformation, [B,T]
        "mu_left_common": torch.stack(mu_left_common_all, dim=1),
        "mu_right_common": torch.stack(mu_right_common_all, dim=1),
        # E_Y[Z | X_t], [B,T]
        "mu_reference_integrated": torch.stack(mu_ref_integrated_all, dim=1),
        # |left_common - right_common|, [B,T]
        "reference_symmetry_error": torch.stack(
            reference_symmetry_error_all,
            dim=1,
        ),
        # Cumulative temporal expectation, [B,T]
        "temporal_mean_curve": torch.stack(temporal_mean_curve, dim=1),
        # Should be numerically almost identical to
        # temporal_mean_curve because expectation is linear.
        # [B,T]
        "posterior_mix_curve": torch.stack(posterior_mix_curve, dim=1),
        # Nonlinear evidence accumulator, [B,T]
        "posterior_product_curve": torch.stack(posterior_product_curve, dim=1),
        # Final distributions: [B,D_common]
        "posterior_mix_final": (posterior_mix.cpu()),
        "posterior_product_final": (product_posterior.cpu()),
        # [D_common]
        "common_disparity": (final_common_disp.cpu()),
    }

    if sampled_refs_all:
        result["sampled_reference"] = torch.stack(
            sampled_refs_all,
            dim=1,
        )

    if return_final_map:
        result["temporal_mean_map"] = (running_map / float(T)).cpu()

    return result


# ============================================================
# Construct temporal sequences from your existing RDS bank
# ============================================================
class TemporalRDSSequenceDataset(Dataset):
    """
    Wrap your existing DatasetRDS as temporal sequences.

    The current RDS bank is disparity-major:

        disparity 1:
            trial 1
            trial 2
            ...
            trial N

        disparity 2:
            trial 1
            ...
            trial N

    Dynamic RDS:
        use T different independently generated trials.

    Static RDS:
        repeat the same RDS frame T times.

    This creates matched static/dynamic sequence counts.
    """

    def __init__(
        self,
        base_dataset,
        n_rds_each_disp: int,
        n_disparities: int,
        T: int,
        stride: int | None = None,
        dynamic: bool = True,
    ):
        self.base_dataset = base_dataset
        self.n_rds_each_disp = n_rds_each_disp
        self.n_disparities = n_disparities
        self.T = T
        self.dynamic = dynamic

        if T <= 0:
            raise ValueError("T must be positive")

        if T > n_rds_each_disp:
            raise ValueError("T cannot exceed n_rds_each_disp")

        # Non-overlapping windows by default.
        self.stride = T if stride is None else stride

        self.n_seq_per_disp = 1 + (n_rds_each_disp - T) // self.stride

    def __len__(self):
        return self.n_disparities * self.n_seq_per_disp

    def __getitem__(self, idx):

        disp_idx = idx // self.n_seq_per_disp
        seq_idx = idx % self.n_seq_per_disp
        start = disp_idx * self.n_rds_each_disp + seq_idx * self.stride

        if self.dynamic:
            indices = [start + t for t in range(self.T)]

        else:
            # static RDSs: same RDS frame repeated over time.
            indices = [start for _ in range(self.T)]

        left_frames = []
        right_frames = []
        labels = []

        for i in indices:

            left, right, label = self.base_dataset[i]

            left_frames.append(left)
            right_frames.append(right)
            labels.append(int(label))

        if len(set(labels)) != 1:
            raise RuntimeError(
                "Temporal sequence crossed a " "disparity-condition boundary"
            )

        return (
            torch.stack(left_frames, dim=0),
            torch.stack(right_frames, dim=0),
            torch.tensor(labels[0], dtype=torch.long),
        )


# ============================================================
# Run one cRDS / hmRDS / aRDS condition
# ============================================================
def run_temporal_rds_condition(
    analysis,
    dot_match: float,
    dot_density: float,
    T: int,
    dynamic: bool,
    reference_mode="marginalize",
    p_left: float = 0.5,
    sequence_batch_size: int = 1,
    roi=None,
    random_seed: int = 3407,
    return_final_map: bool = False,
    rds_bank=None,
):
    """
    Uses RDS_LayerAct._generate_rds_loader() to generate RDSs.
    """

    base_loader = analysis._generate_rds_loader(
        dot_match,
        dot_density,
        analysis.background_flag,
        analysis.pedestal_flag,
        rds_bank=rds_bank,
    )

    sequence_dataset = TemporalRDSSequenceDataset(
        base_dataset=base_loader.dataset,
        n_rds_each_disp=(analysis.n_rds_each_disp),
        n_disparities=len(analysis.disp_ct_pix_list),
        T=T,
        dynamic=dynamic,
    )

    sequence_loader = DataLoader(
        sequence_dataset,
        batch_size=sequence_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(torch.device(analysis.device).type == "cuda"),
    )

    batch_results = {}
    labels_all = []

    common_disp = None

    for left_sequence, right_sequence, labels in sequence_loader:

        result = analyze_temporal_batch(
            model=analysis.model,
            left_sequence=left_sequence,
            right_sequence=right_sequence,
            reference_mode=reference_mode,
            p_left=p_left,
            roi=roi,
            random_seed=random_seed,
            return_final_map=(return_final_map),
        )

        labels_all.append(labels)
        common_disp = result["common_disparity"]

        for key, value in result.items():
            if key == "common_disparity":
                continue
            batch_results.setdefault(key, []).append(value)

    output = {
        key: torch.cat(
            values,
            dim=0,
        ).numpy()
        for key, values in batch_results.items()
    }

    output["label"] = torch.cat(labels_all, dim=0).numpy()
    output["common_disparity"] = common_disp.numpy()
    output["T"] = T
    output["dynamic"] = dynamic
    output["dot_match"] = dot_match
    output["dot_density"] = dot_density
    output["reference_mode"] = reference_mode

    return output
