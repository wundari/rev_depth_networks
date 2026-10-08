"""Collect/analyze paired RDS disparities over stimulus presentation time.

Run from the repository root (uses NumPy; collect additionally uses PyTorch and
the existing RDS generator):

    .venv/bin/python scripts/analysis/run_expected_disparity.py demo --output /tmp/rds_demo.npz
    .venv/bin/python scripts/analysis/run_expected_disparity.py analyze /tmp/rds_demo.npz \
        --window 0 1 --output /tmp/rds_summary
    .venv/bin/python scripts/analysis/run_expected_disparity.py collect \
        --checkpoint PATH_TO_CHECKPOINT --model GC_Net_LR --interaction sum_diff \
        --fps 30 --frames 30 --trials 32 --dot-density 0.3 --disparities -10 10 \
        --output /tmp/paired_rds.npz

Each collection trial is an independent dynamic RDS sequence, with newly drawn
dots per frame. The model predicts frames independently; presentation timestamps
are a prescribed stimulus schedule, not model execution times or biological
response latencies. Analyze any subwindow within the recorded presentation.

The generator's raw displacement s satisfies x_right=x_left+s; we pass s=-d
so the saved target d follows the declared convention x_left-x_right=d.
No disparity pedestal is applied. Background is correlated and zero-disparity.

Existing physical-eye-aware GC_Net_LR and GC_NetGN_LR models are supported.
Specify the architecture/interaction matching the checkpoint. Model defaults
come from its existing config; no EngineBase construction or training occurs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np

# Support direct invocation without requiring a package install/PYTHONPATH.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RDS_analysis.expected_disparity import analyze_paired, load_paired_archive


def save_archive(path, z, conditions, fps, metadata):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Prediction output must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, z=z, eyes=np.array(["left", "right"]),
        times=np.arange(z.shape[1], dtype=float) / fps,
        time_end=np.array(z.shape[1] / fps), time_unit=np.array("seconds"),
        rds_type=np.array([c[0] for c in conditions]),
        dot_density=np.array([c[1] for c in conditions]),
        target_disparity=np.array([c[2] for c in conditions]),
        metadata_json=np.array(json.dumps(metadata)),
    )
    print(f"Saved {z.shape[0]} paired sequences, {z.shape[1]} frames at {fps:g} Hz: {path}")


def demo(args):
    """An explicit synthetic example, not predictions from a trained model."""
    conditions = [
        (kind, density, target)
        for kind, density, target, _ in itertools.product(
            ("ards", "hmrds", "crds"), (0.2, 0.6), (-10.0, 10.0), range(args.trials)
        )
    ]
    rng = np.random.default_rng(args.seed)
    gains = {"ards": -0.4, "hmrds": 0.5, "crds": 1.0}
    target_response = np.array([gains[c[0]] * c[2] for c in conditions])[:, None]
    offsets = rng.normal(0, 0.6, (len(conditions), 1))
    # Serial dependence is included to exercise whole-sequence resampling.
    noise = rng.normal(0, 0.3, (len(conditions), args.frames)).cumsum(axis=1)
    left = target_response + offsets + noise
    right = -target_response - offsets - noise + 0.5
    save_archive(args.output, np.stack((left, right), axis=-1), conditions, args.fps,
                 {"synthetic": True, "seed": args.seed, "fps": args.fps,
                  "known_eye_sum_px": 0.5,
                  "note": "Illustrative responses only; gains are not scientific predictions."})


def analyze(args):
    data = load_paired_archive(args.input)
    result, trial_means = analyze_paired(
        data["z"], data["times"], data["rds_type"], data["dot_density"],
        data["target_disparity"], args.window, time_unit=data["time_unit"],
        method=args.method, time_end=data["time_end"], n_bootstrap=args.bootstrap,
        confidence=args.confidence, seed=args.seed,
    )
    result["source"] = str(Path(args.input).resolve())
    result["provenance"] = data["metadata"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        output / "trial_means.npz", window_mean=trial_means,
        eyes=data["eyes"], rds_type=data["rds_type"], dot_density=data["dot_density"],
        target_disparity=data["target_disparity"], window=np.array(args.window),
        time_unit=np.array(data["time_unit"]),
    )
    print("RDS       density  target   trials     E[left]    E[right]    eye sum (pixels)")
    for group in result["groups"]:
        print(f"{group['rds_type']:8}  {group['dot_density']:6.2f}  "
              f"{group['target_disparity_px']:6.1f}  {group['n_trials']:7d}  "
              f"{group['left']['mean_px']:10.3f}  {group['right']['mean_px']:10.3f}  "
              f"{group['eye_sum']['mean_px']:10.3f}")
    print(f"Saved window estimates: {output / 'summary.json'}")


def collect(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.batch_size != 1:
        raise ValueError("Use batch-size 1: untracked BatchNorm otherwise couples trials, "
                         "invalidating the independent-trial confidence intervals")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if len(set(args.dot_density)) != len(args.dot_density) or any(
        not np.isfinite(rho) or not 0 < rho <= 1 for rho in args.dot_density
    ):
        raise ValueError("dot-density values must be distinct and in (0,1]")
    if len(set(args.disparities)) != len(args.disparities):
        raise ValueError("disparities must be distinct")

    import torch
    from joblib import parallel_config
    from RDS.DataHandler_RDS import RDS_Handler, H_BG, W_BG, R_DOT
    from RDS_analysis.paired_disparity import normalize_rds, predict_paired_roi

    modules = {
        "GC_Net_LR": ("config.config_gcnet_lr", "GC_Net_LR.modules.gcnet"),
        "GC_NetGN_LR": ("config.config_gcnet_gn_lr", "GC_NetGN_LR.modules.gcnet"),
    }
    config_module, model_module = modules[args.model]
    config = importlib.import_module(config_module).ConfigGCNet(
        binocular_interaction=args.interaction, seed=args.seed,
        compile_mode=None, load_state=False, device=args.device,
    )
    if (config.img_height, config.img_width) != (H_BG, W_BG):
        raise ValueError("Model dimensions must match the existing RDS generator")
    if any(abs(d) >= config.max_disp // 2 or abs(d) > 127 for d in args.disparities):
        raise ValueError("Targets must be inside the signed disparity support and int8 generator labels")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu")
    torch.manual_seed(args.seed)
    model = importlib.import_module(model_module).build_gcnet(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    # Match EngineBase's handling of compiled checkpoints, including DDP prefix.
    state = dict(state)
    for prefix in ("module.", "_orig_mod."):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, prefix)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    margin = max(map(abs, args.disparities)) + R_DOT + 1
    roi_left = tuple(args.roi) if args.roi else (
        H_BG // 4 + margin, 3 * H_BG // 4 - margin,
        W_BG // 4 + margin, 3 * W_BG // 4 - margin,
    )
    y0, y1, x0, x1 = roi_left
    if not (H_BG // 4 + R_DOT < y0 < y1 < 3 * H_BG // 4 - R_DOT
            and W_BG // 4 + R_DOT < x0 < x1 < 3 * W_BG // 4 - R_DOT):
        raise ValueError("ROI must be inside the central target, excluding its dot-radius boundary")
    if any(x0 - d < 0 or x1 - d > W_BG for d in args.disparities):
        raise ValueError("Corresponding right-eye ROIs must fit in the image")

    conditions = []
    predictions = []
    amp_dtype = None if args.amp == "float32" else getattr(torch, args.amp)
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This device does not support bfloat16; select another --amp")
    numpy_state = np.random.get_state()
    try:
        # Serial generator calls use per-frame seeds and do not spawn workers.
        with parallel_config(backend="sequential"):
            for kind, match in (("ards", 0.0), ("hmrds", 0.5), ("crds", 1.0)):
                for rho, target in itertools.product(args.dot_density, args.disparities):
                    values = np.empty((args.trials, args.frames, 2), dtype=np.float32)
                    roi_right = (y0, y1, x0 - target, x1 - target)
                    for frame in range(args.frames):
                        for start in range(0, args.trials, args.batch_size):
                            stop = min(start + args.batch_size, args.trials)
                            left, right = [], []
                            for trial in range(start, stop):
                                child = np.random.SeedSequence(
                                    [args.seed, int(match * 10000), int(round(rho * 1e9)),
                                     target + 128, trial, frame]
                                )
                                np.random.seed(int(child.generate_state(1)[0]))
                                with redirect_stdout(io.StringIO()):
                                    image_left, image_right, _ = RDS_Handler.generate_rds(
                                        match, rho, [-target], 1, True, False,
                                    )
                                left.append(normalize_rds(image_left[0]))
                                right.append(normalize_rds(image_right[0]))
                            values[start:stop, frame] = predict_paired_roi(
                                model, torch.stack(left).to(device), torch.stack(right).to(device),
                                roi_left, roi_right=roi_right, amp_dtype=amp_dtype,
                            )
                        if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == args.frames:
                            print(f"{kind}, density={rho:g}, d={target:+d}: frame {frame + 1}/{args.frames}", flush=True)
                    conditions.extend([(kind, rho, target)] * args.trials)
                    predictions.append(values)
    finally:
        np.random.set_state(numpy_state)
    save_archive(args.output, np.concatenate(predictions), conditions, args.fps, {
        "synthetic": False, "checkpoint": str(checkpoint_path.resolve()),
        "model": args.model, "interaction": args.interaction, "seed": args.seed,
        "fps": args.fps, "device": args.device, "amp": args.amp,
        "roi_left": roi_left, "roi_right_rule": "x bounds = left x bounds - target disparity",
        "generator_shift_rule": "raw generator displacement = -target disparity",
        "background": "zero-disparity cRDS", "pedestal": False,
        "batch_size": args.batch_size,
        "batch_composition": "same condition/frame, consecutive trials, identical in both eye calls",
        "note": "Framewise model; timestamps describe scheduled presentation. "
                "Single-trial batches prevent cross-trial coupling through untracked BatchNorm.",
    })


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    create_demo = commands.add_parser("demo", help="Create clearly labeled synthetic paired data")
    create_demo.add_argument("--output", required=True)
    create_demo.set_defaults(action=demo)
    collector = commands.add_parser("collect", help="Evaluate a checkpoint on 30 Hz dynamic RDS sequences")
    collector.add_argument("--checkpoint", required=True)
    collector.add_argument("--model", choices=["GC_Net_LR", "GC_NetGN_LR"], default="GC_Net_LR")
    collector.add_argument("--interaction", choices=["default", "bem", "cmm", "sum_diff"], default="sum_diff")
    collector.add_argument("--dot-density", nargs="+", type=float, default=[0.3])
    collector.add_argument("--disparities", nargs="+", type=int, default=[-10, 10])
    # Instance inference avoids cross-trial BatchNorm coupling in these models.
    collector.add_argument("--batch-size", type=int, choices=[1], default=1,
                           help="Single-trial batches keep stimulus-bootstrap trials independent")
    collector.add_argument("--device", default="cpu", help="Use cuda when available")
    collector.add_argument("--amp", choices=["float32", "float16", "bfloat16"], default="float32")
    collector.add_argument("--roi", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"))
    collector.add_argument("--output", required=True)
    collector.set_defaults(action=collect)
    for command in (create_demo, collector):
        command.add_argument("--fps", type=float, default=30.0)
        command.add_argument("--frames", type=int, default=30, help="30 frames span 1 second at 30 Hz")
        command.add_argument("--trials", type=int, default=32, help="Independent sequences per condition")
        command.add_argument("--seed", type=int, default=3407)
    analyzer = commands.add_parser("analyze", help="Estimate eye-specific window means and paired intervals")
    analyzer.add_argument("input")
    analyzer.add_argument("--window", nargs=2, type=float, required=True, metavar=("START", "STOP"))
    analyzer.add_argument("--method", choices=["hold", "linear", "samples"], default="hold")
    analyzer.add_argument("--bootstrap", type=int, default=2000)
    analyzer.add_argument("--confidence", type=float, default=0.95)
    analyzer.add_argument("--seed", type=int, default=3407)
    analyzer.add_argument("--output", required=True, help="Directory for summary.json and trial_means.npz")
    analyzer.set_defaults(action=analyze)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        if args.command in {"demo", "collect"}:
            if not np.isfinite(args.fps) or args.fps <= 0 or args.frames < 1 or args.trials < 1:
                raise ValueError("fps, frames and trials must be positive")
            if args.seed < 0:
                raise ValueError("seed must be nonnegative")
        args.action(args)
    except (ValueError, OSError) as error:
        arguments.error(str(error))


if __name__ == "__main__":
    main()
