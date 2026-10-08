"""Expected signed disparity over a specified observation window.

Let X_t = (I_left,t, I_right,t), Y in {left, right}, and C be RDS type.
Define signed disparity at corresponding visible points as
    d_left = x_left - x_right; d_right = x_right - x_left.
An eye-equivariant prediction distribution would satisfy p_left(z)=p_right(-z),
NOT p_left(z)=-p_right(z). This is a hypothesis for a learned model.

For fixed images and deterministic network/batch context, predicted Z=f(X,Y)
is deterministic. A decoder's q(d|X,Y) is instead a model distribution over
disparity hypotheses; the regression output is sum_d d*q(d|X,Y).

The time-window estimand is E[bar_Z_T | C,d,rho,Y], where
    bar_Z_T = integral_T Z_t dt / duration(T).
First average time within each independent trial, then average trials.
The bootstrap resamples whole paired trial trajectories, never pixels/frames.
Its interval measures stimulus-trial sampling uncertainty, not calibration of q.

NPZ input contract (no pickled objects):
    z: [trial, time, 2], signed ROI means, eyes in order left then right
    eyes: ['left', 'right']
    times: [time], strictly increasing frame onset/observation coordinates
    time_unit: scalar string, e.g. 'seconds', 'frame_index', 'checkpoint_step'
    time_end: scalar endpoint of final frame, required for method='hold'
    rds_type, dot_density, target_disparity: [trial], constant within each trial
    metadata_json: optional scalar JSON string describing model/ROI/provenance

Each row must be an independent stimulus trial/sequence for its condition,
with the SAME sequence evaluated in both eyes. Do not concatenate dependent
observations, checkpoints, or repetitions of the identical image as new trials.
ROIs must refer to the same target surface in each reference view; dense maps
need correspondence alignment before interpreting pixelwise eye symmetry.
Legacy pred_disp_*.npy files lack eye/time metadata and cannot meet this contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _finite(values, name):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def cost_to_pmf(costs, *, axis=-1, temperature=1.0):
    """Compute softmax(-cost/temperature); smaller costs are more likely."""
    costs = _finite(costs, "costs")
    if costs.ndim == 0 or costs.shape[axis] == 0:
        raise ValueError("costs must have a nonempty disparity axis")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    with np.errstate(over="ignore"):
        scores = -(costs - costs.min(axis=axis, keepdims=True)) / temperature
    weights = np.exp(scores)
    return weights / weights.sum(axis=axis, keepdims=True)


def pmf_moments(probabilities, disparity_values, *, axis=-1):
    """Return model hypothesis mean and variance on the supplied signed bins.

    Supply bins matching the decoder/output convention, including any explicit
    ref sign multiplication. Softmax normalization alone does not establish a
    calibrated posterior. Already-regressed disparity maps are not PMFs.
    """
    p = _finite(probabilities, "probabilities")
    bins = _finite(disparity_values, "disparity_values")
    if p.ndim == 0 or bins.ndim != 1 or bins.size == 0:
        raise ValueError("Expected a PMF array and a nonempty vector of bins")
    if p.shape[axis] != bins.size or np.unique(bins).size != bins.size:
        raise ValueError("Disparity axis must match distinct supplied bins")
    totals = p.sum(axis=axis, keepdims=True)
    if np.any(p < 0) or not np.allclose(totals, 1.0, rtol=0, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    p = p / totals  # only correct accepted floating-point normalization error
    shape = [1] * p.ndim
    shape[axis] = bins.size
    bins = bins.reshape(shape)
    mean = (p * bins).sum(axis=axis)
    variance = (p * (bins - np.expand_dims(mean, axis)) ** 2).sum(axis=axis)
    return mean, variance


def window_weights(times, window, *, method="hold", time_end=None):
    """Normalized weights for the explicitly chosen window [start, stop).

    hold: each prediction persists from its frame onset to the next onset;
          the final frame persists to the explicit time_end. Weights are frame
          interval overlaps with the window, divided by its duration.
    linear: integrate piecewise-linear interpolation of observations, clipping
            segments at both window boundaries. No extrapolation is allowed.
    samples: equal weights on observed coordinates in [start, stop); this is
             a sample mean, not a duration-weighted mean for irregular times.
    """
    times = _finite(times, "times")
    bounds = _finite(window, "window")
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a nonempty, strictly increasing vector")
    if bounds.shape != (2,) or bounds[1] <= bounds[0]:
        raise ValueError("window must be (start, stop) with stop > start")
    start, stop = bounds
    if method == "samples":
        weights = ((times >= start) & (times < stop)).astype(float)
        if not weights.any():
            raise ValueError("No observations in the requested sample window")
        return weights / weights.sum()
    if method == "hold":
        if time_end is None or np.ndim(time_end) != 0:
            raise ValueError("hold requires the explicit scalar final frame time_end")
        time_end = float(_finite(time_end, "time_end"))
        if time_end <= times[-1]:
            raise ValueError("time_end must be after the last frame onset")
        ends = np.r_[times[1:], time_end]
        if start < times[0] or stop > time_end:
            raise ValueError("Window extends beyond recorded frame intervals")
        weights = np.maximum(0.0, np.minimum(ends, stop) - np.maximum(times, start))
    elif method == "linear":
        if times.size < 2 or start < times[0] or stop > times[-1]:
            raise ValueError("linear requires at least two times covering the whole window")
        weights = np.zeros(times.size)
        for i, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            a, b = max(t0, start), min(t1, stop)
            if b > a:
                # Integral of each endpoint's linear basis over [a,b].
                right_area = (b - a) * ((a - t0) + (b - t0)) / (2 * (t1 - t0))
                weights[i] += (b - a) - right_area
                weights[i + 1] += right_area
    else:
        raise ValueError("method must be hold, linear, or samples")
    return weights / (stop - start)


def window_trial_means(z, times, window, *, method="hold", time_end=None):
    """Reduce time, retaining independent trials and both measured eye outputs."""
    z = _finite(z, "z")
    if z.ndim != 3 or z.shape[0] == 0 or z.shape[1:] != (len(times), 2):
        raise ValueError("z must have shape [nonempty trial, time, 2 eyes]")
    weights = window_weights(times, window, method=method, time_end=time_end)
    return np.einsum("nte,t->ne", z, weights)


def _paired_statistics(values, rng, n_bootstrap, confidence):
    """Bootstrap every derived statistic using the SAME paired trial draws."""
    # The residual is an eye-symmetry diagnostic; it is never imposed as zero.
    quantities = np.column_stack(
        (values[:, 0], values[:, 1], (values[:, 0] - values[:, 1]) / 2,
         values[:, 0] + values[:, 1])
    )
    names = ("left", "right", "aligned_mean", "eye_sum")
    means = quantities.mean(axis=0)
    n = len(values)
    if n > 1:
        std = quantities.std(axis=0, ddof=1)
        boot = np.empty((n_bootstrap, len(names)))
        # Bound temporary index-array memory for large trial counts.
        chunk_size = max(1, min(256, 1_000_000 // n))
        for offset in range(0, n_bootstrap, chunk_size):
            count = min(chunk_size, n_bootstrap - offset)
            indices = rng.integers(0, n, size=(count, n))
            boot[offset:offset + count] = quantities[indices].mean(axis=1)
        alpha = (1 - confidence) / 2
        intervals = np.quantile(boot, [alpha, 1 - alpha], axis=0).T
    return {
        name: {
            "mean_px": float(means[i]),
            "trial_sd_px": float(std[i]) if n > 1 else None,
            "standard_error_px": float(std[i] / np.sqrt(n)) if n > 1 else None,
            "ci_px": intervals[i].tolist() if n > 1 else None,
        }
        for i, name in enumerate(names)
    }


def analyze_paired(
    z, times, rds_type, dot_density, target_disparity, window, *,
    time_unit, method="hold", time_end=None, n_bootstrap=2000,
    confidence=0.95, seed=3407,
):
    """Estimate E[bar_Z_T | C, target disparity, dot density, eye].

    Rows are equally weighted independent trials within each condition. Pixel
    means must already be reduced over declared corresponding target ROIs.
    aligned_mean=(left-right)/2 expresses both eyes in the left-sign convention;
    eye_sum=left+right measures departure from opposite expected values.
    """
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("Declare a nonempty time_unit; timestamps are never inferred")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    values = window_trial_means(z, times, window, method=method, time_end=time_end)
    n = len(values)
    types = np.asarray(rds_type, dtype=str)
    density = _finite(dot_density, "dot_density")
    targets = _finite(target_disparity, "target_disparity")
    if any(a.shape != (n,) for a in (types, density, targets)):
        raise ValueError("Condition metadata must contain one value per trial")
    if not np.isin(types, ["crds", "hmrds", "ards"]).all():
        raise ValueError("rds_type must use crds, hmrds, or ards")
    if np.any((density <= 0) | (density > 1)):
        raise ValueError("dot_density must be in (0,1]")
    keys = sorted(set(zip(types.tolist(), density.tolist(), targets.tolist())))
    rng = np.random.default_rng(seed)
    groups = []
    for kind, rho, d in keys:
        selected = (types == kind) & (density == rho) & (targets == d)
        statistics = _paired_statistics(values[selected], rng, n_bootstrap, confidence)
        groups.append({
            "rds_type": kind, "dot_density": rho, "target_disparity_px": d,
            "n_trials": int(selected.sum()), **statistics,
            "aligned_gain": statistics["aligned_mean"]["mean_px"] / d if d != 0 else None,
        })
    result = {
        "estimand": "sample-coordinate mean" if method == "samples" else "time-window mean",
        "window": list(map(float, window)), "time_unit": time_unit, "method": method,
        "time_weights": window_weights(times, window, method=method, time_end=time_end).tolist(),
        "sign_convention": "left: x_left-x_right; right: x_right-x_left",
        "bootstrap": {"unit": "independent paired trial/sequence", "replicates": n_bootstrap,
                      "confidence": confidence, "seed": seed},
        "groups": groups,
    }
    return result, values


def load_paired_archive(path):
    """Require measured eye/time metadata; do not infer it from old map files."""
    required = {"z", "eyes", "times", "time_unit", "rds_type", "dot_density", "target_disparity"}
    loaded = np.load(Path(path), allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Use a paired NPZ archive, not legacy disparity-map NPY files")
    with loaded as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing paired analysis fields: {sorted(missing)}")
        data = {name: archive[name] for name in required}
        if data["eyes"].shape != (2,) or data["eyes"].tolist() != ["left", "right"]:
            raise ValueError("eyes must explicitly be ['left', 'right'] in that order")
        data["time_unit"] = str(data["time_unit"].item())
        data["time_end"] = float(archive["time_end"].item()) if "time_end" in archive.files else None
        data["metadata"] = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive.files else {}
    return data
