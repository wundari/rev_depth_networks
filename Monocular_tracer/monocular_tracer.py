# %% load necessary modules
"""Monocular response profiling and controlled stereo-network ablation.

Public workflow: MonocularTracer(engine, config_mt), profile_layer(), ablate_layer().
The tensor operations are collected below the analysis class, in # %% sections.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from config.config_monocular_tracer import ConfigMonocularTracer

if TYPE_CHECKING:
    from engine.engine_bnn import EngineBNN
    from engine.engine_gcnet_lr import EngineGCNet


# %%
class MonocularTracer:
    """Measure monocular responsiveness, then lesion fixed units during stereo tests.

    Args:
        engine: Your existing EngineBNN or EngineGCNet with a loaded checkpoint.
        config_mt: Analysis settings from ConfigMonocularTracer.

    The engine owns the model and checkpoint loading. This class owns the
    analysis, target layers, data loaders, result files and population plots.
    """

    def __init__(
        self,
        engine: "EngineBNN | EngineGCNet",
        config_mt: ConfigMonocularTracer,
    ):
        self.engine = engine
        self.config = engine.config
        self.config_mt = config_mt
        self.config_mt.validate()

        self.model = engine.model
        self.model_name = engine.model_name
        self.interaction = self.config.binocular_interaction
        self.device = torch.device(self.config.device)

        if getattr(self.config, "compile_mode", None) is not None:
            raise ValueError("Set config.compile_mode = None before creating the engine")
        if not getattr(self.config, "preserve_physical_eye_identity", False):
            raise ValueError("Use config.preserve_physical_eye_identity = True")

        self.model_fingerprint = self._model_fingerprint()
        self.baseline_rows = None
        self._baseline_loader = None
        self.make_dirs()
        self.set_target_layers()
        self.print_target_layers()
        self.save_metadata()

    # %% directories, target layers and experiment metadata
    def make_dirs(self):
        """Use a new directory for each run to keep different settings separate."""
        if self.config_mt.output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            directory = Path(self.engine.experiment_dir) / "monocular_tracer" / stamp
        else:
            directory = Path(self.config_mt.output_dir)

        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("Analysis output directory must be empty; choose a new directory")
        self.mt_dir = str(directory)

    def set_target_layers(self):
        """Layer dictionaries follow the format used in ActivationMaximization."""
        if self.config_mt.target_layers:
            self.target_layers = deepcopy(self.config_mt.target_layers)
        elif self.model_name == "BNN":
            self.target_layers = [
                dict(
                    name="cost_volume",
                    module="decoder.monocular_cost_volume",
                    call=0,
                ),
                dict(
                    name="layer3",
                    module="decoder.layer3",
                    call=0,
                ),
                dict(
                    name="layer4",
                    module="decoder.layer4",
                    call=0,
                ),
            ]
        elif self.model_name == "GC_Net_LR":
            self.target_layers = [
                dict(
                    name="cost_volume",
                    module="decoder.monocular_cost_volume",
                    call=0,
                ),
                dict(
                    name="layer19",
                    module="decoder.layer19",
                    call=0,
                ),
                dict(
                    name="layer32",
                    module="decoder.layer32",
                    call=0,
                ),
                dict(
                    name="layer36a",
                    module="decoder.layer36a",
                    call=0,
                ),
            ]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        names = []
        for layer in self.target_layers:
            if not all(key in layer for key in ("name", "module", "call")):
                raise ValueError("Each layer needs name, module, and call")
            if Path(layer["name"]).name != layer["name"] or layer["name"] in {".", ".."}:
                raise ValueError("Layer name must be a filename without directory components")
            Site(layer["module"], layer["call"])
            names.append(layer["name"])
        if len(set(names)) != len(names):
            raise ValueError("Layer names must be unique")

    def print_target_layers(self):
        print(f"Monocular tracer: {self.model_name}, interaction={self.interaction}")
        for layer in self.target_layers:
            print(f"  {layer['name']:20} {layer['module']:36} call={layer['call']}")
        print(f"Unit: {self.config_mt.unit}; output: {self.mt_dir}")

    def _model_fingerprint(self) -> str:
        """Identify actual parameters and buffers, not just the checkpoint filename."""
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        return digest.hexdigest()

    def save_metadata(self):
        metadata = dict(
            model_name=self.model_name,
            interaction=self.interaction,
            model_fingerprint=self.model_fingerprint,
            checkpoint=getattr(self.config, "model_pretrained", None),
            experiment_dir=str(self.engine.experiment_dir),
            model_config=vars(self.config),
            tracer_config=asdict(self.config_mt),
            target_layers=self.target_layers,
            torch_version=str(torch.__version__),
        )
        filename = Path(self.mt_dir) / "metadata.json"
        filename.write_text(json.dumps(metadata, indent=2, default=str))

    # %% validation and test datasets
    def make_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Use the repository's validation/test partitions without training transforms.

        Validation sequences calibrate coupling scores. Separate test sequences
        measure lesion effects. Both loaders preserve order and use batch size 1.
        """
        from dataset.scene_flow import SceneFlowFlyingThingsDataset, SceneFlowMonkaaDataset

        if self.config.dataset == "sceneflow_flying":
            dataset_class = SceneFlowFlyingThingsDataset
        elif self.config.dataset == "sceneflow_monkaa":
            dataset_class = SceneFlowMonkaaDataset
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        # Copy config so analysis never changes the engine's training settings.
        dataset_config = deepcopy(self.config)
        calibration_dataset = dataset_class(self.engine.datadir, dataset_config, split="validation")
        test_dataset = dataset_class(self.engine.datadir, dataset_config, split="test")

        calibration_dataset = AnalysisDataset(
            calibration_dataset, self.config_mt.max_calibration_samples,
        )
        test_dataset = AnalysisDataset(test_dataset, self.config_mt.max_test_samples)
        self._check_split_overlap(calibration_dataset.records, test_dataset.records)

        loader_options = dict(
            batch_size=1,
            shuffle=False,
            num_workers=self.config_mt.num_workers,
            collate_fn=_collate_one_sample,
            pin_memory=self.device.type == "cuda",
        )
        if self.config_mt.num_workers:
            loader_options.update(
                persistent_workers=self.config_mt.persistent_workers,
                prefetch_factor=self.config_mt.prefetch_factor,
            )
        calibration_loader = DataLoader(calibration_dataset, **loader_options)
        test_loader = DataLoader(test_dataset, **loader_options)

        filename = Path(self.mt_dir) / "split_manifest.json"
        filename.write_text(json.dumps(dict(
            calibration=calibration_dataset.records,
            test=test_dataset.records,
        ), indent=2))
        return calibration_loader, test_loader

    @staticmethod
    def _check_split_overlap(calibration_records: list, test_records: list):
        for key in ("id", "group"):
            calibration_values = {record[key] for record in calibration_records}
            test_values = {record[key] for record in test_records}
            if calibration_values & test_values:
                raise ValueError(f"Calibration and test overlap in {key}")

    # %% image inputs and model forward pass
    def make_gray_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert raw gray to normalized RGB; image shape is [1, 3, H, W]."""
        normalized_gray = [
            (self.config_mt.baseline - mean) / std
            for mean, std in zip(self.config_mt.mean, self.config_mt.std)
        ]
        gray = image.new_tensor(normalized_gray).view(1, 3, 1, 1)
        return gray.expand_as(image).clone()

    def _forward(self, model: nn.Module, sample: dict) -> torch.Tensor:
        """Keep physical eyes, signed GT and reference sign in repository format."""
        from utilities.misc import NestedTensor

        inputs = NestedTensor(
            left=sample["left"].to(self.device, non_blocking=True),
            right=sample["right"].to(self.device, non_blocking=True),
            disp=sample["disp"].to(self.device, non_blocking=True),
            ref=sample["ref"].to(self.device, non_blocking=True),
        )
        # inputs.left/right: [1, 3, H, W]
        # disparity prediction: [1, H, W]
        precision = self.config_mt.amp_dtype
        if self.device.type != "cuda" and precision != "float32":
            raise ValueError("AMP is supported only for CUDA in this analysis")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
        with torch.autocast(
            device_type=self.device.type, dtype=dtype,
            enabled=precision != "float32",
        ):
            return model(inputs)

    def _profile_settings(self) -> dict:
        settings = self.config_mt
        return dict(unit=settings.unit, amp_dtype=settings.amp_dtype, baseline=settings.baseline,
                    mean=tuple(settings.mean), std=tuple(settings.std),
                    activity_floor=settings.activity_floor,
                    weak_threshold=settings.weak_threshold,
                    strong_threshold=settings.strong_threshold)

    # %% four-condition profiling
    def profile_layer(self, layer: dict, calibration_loader) -> dict:
        """Measure LR, L0, 0R and 00 responses at one layer/invocation.

        Conditions:
            LR: both physical eyes see their original images.
            L0: left original, right gray.
            0R: left gray, right original.
            00: both eyes gray.

        B, L, R are mean absolute responses after subtracting the 00 activation.
        Coupling = (B - max(L, R)) / (B + max(L, R) + epsilon).
        Higher scores indicate stronger dependence on binocular stimulation.

        Returns responses and labels with the selected unit shape, e.g. [C] or
        [C, D]. The result also includes per-sample responses and sequence IDs.
        """
        print(f"Profiling {layer['name']}", flush=True)
        site = Site(layer["module"], layer["call"])
        settings = self.config_mt

        with _cost_volume_probe(self.model):
            result = _measure_responses(
                model=self.model,
                loader=calibration_loader,
                sites=[site],
                forward=self._forward,
                gray=self.make_gray_image,
                unit=settings.unit,
                activity_floor=settings.activity_floor,
                weak_threshold=settings.weak_threshold,
                strong_threshold=settings.strong_threshold,
                batch_size=settings.inference_batch_size,
            )

        return dict(
            layer=deepcopy(layer),
            responses=result["sites"][str(site)],
            calibration_ids=result["ids"],
            calibration_groups=result["groups"],
            calibration_refs=result["refs"],
            model_fingerprint=self.model_fingerprint,
            profile_settings=self._profile_settings(),
        )

    # %% unablated stereo baseline
    def evaluate_baseline(self, test_loader) -> list[dict]:
        """Evaluate the normal stereo network on the same images used for lesions."""
        print("Evaluating stereo baseline", flush=True)
        self.baseline_rows = _evaluate_stereo(
            self.model, test_loader, self._forward, self.config_mt.disp_range,
        )
        self._baseline_loader = test_loader
        self._baseline_range = self.config_mt.disp_range
        self._baseline_precision = self.config_mt.amp_dtype
        return self.baseline_rows

    # %% layerwise ablation experiment
    def ablate_layer(self, layer: dict, profile_result: dict, test_loader) -> dict:
        """Freeze calibration masks, then measure paired stereo-test error changes.

        weak/strong use continuous score rankings among stereo-active units;
        the threshold-defined weak/strong labels are retained in profile_result.
        """
        self.config_mt.validate()
        if profile_result["layer"] != layer:
            raise ValueError("Profile and ablation target different layers")
        if profile_result["model_fingerprint"] != self.model_fingerprint:
            raise ValueError("Profile was measured with a different model")
        if profile_result["profile_settings"] != self._profile_settings():
            raise ValueError("Profile settings changed; measure the layer again")

        if (self.baseline_rows is None or self._baseline_loader is not test_loader
                or self._baseline_range != self.config_mt.disp_range
                or self._baseline_precision != self.config_mt.amp_dtype):
            self.evaluate_baseline(test_loader)

        calibration_ids = set(profile_result["calibration_ids"])
        calibration_groups = set(profile_result["calibration_groups"])
        for row in self.baseline_rows:
            if row["id"] in calibration_ids or row["group"] in calibration_groups:
                raise ValueError("Calibration and test must use disjoint sequences")

        data = profile_result["responses"]
        settings = self.config_mt
        site = Site(layer["module"], layer["call"])
        replacement = dict(zero=None, mean=data["stereo_mean"], blank=data["blank_mean"])[
            settings.replacement
        ]
        runs = []
        unique_masks = []
        mask_indices = {}

        # Build every experiment first. Identical masks share one evaluation;
        # zero-dose experiments reuse the unablated baseline exactly.
        for fraction in settings.fractions:
            weak_mask = _select_units(data, fraction, kind="weak")
            plans = [("weak", 0), ("strong", 0)]
            plans += [(kind, repeat) for kind in ("random", "matched")
                      for repeat in range(settings.n_controls)]

            for kind, repeat in plans:
                run = dict(selection=kind, fraction=fraction, repeat=repeat)
                if kind == "matched" and 2 * int(weak_mask.sum()) > int(data["active"].sum()):
                    run.update(status="skipped", reason="Too few disjoint active units to match this dose")
                    runs.append(run)
                    continue

                mask = _select_units(
                    data=data, fraction=fraction, kind=kind,
                    seed=settings.seed + repeat,
                    match_to=weak_mask if kind == "matched" else None,
                )
                n_selected = int(mask.sum())
                job_index = None
                if n_selected:
                    key = np.packbits(mask.numpy().reshape(-1)).tobytes()
                    if key not in mask_indices:
                        mask_indices[key] = len(unique_masks)
                        unique_masks.append(mask)
                    job_index = mask_indices[key]

                run.update(
                    status="complete", _job_index=job_index,
                    n_selected=n_selected, n_active=int(data["active"].sum()),
                    selected_indices=torch.nonzero(mask).tolist(),
                    mean_selected_B=float(data["B"][mask].mean()) if n_selected else None,
                    mean_selected_stereo_abs=float(data["stereo_abs"][mask].mean()) if n_selected else None,
                    mean_selected_coupling=float(data["coupling"][mask].mean()) if n_selected else None,
                )
                if kind == "matched" and n_selected:
                    run["target_mean_stereo_abs"] = float(data["stereo_abs"][weak_mask].mean())
                runs.append(run)

        print(
            f"{layer['name']}: {len(unique_masks)} distinct nonzero lesions; "
            f"GPU jobs per batch={settings.inference_batch_size}", flush=True,
        )
        with _cost_volume_probe(self.model):
            evaluated_rows = _evaluate_ablation_jobs(
                model=self.model, loader=test_loader, forward=self._forward,
                site=site, masks=unique_masks, unit=settings.unit,
                replacement=replacement, batch_size=settings.inference_batch_size,
                device=self.device, disp_range=settings.disp_range,
                reuse_encoder=settings.reuse_encoder_features,
            )

        # Reuse both predictions and bootstrap results for identical experiments.
        effect_cache = {}
        for run in runs:
            if run["status"] == "skipped":
                continue
            job_index = run.pop("_job_index")
            rows = self.baseline_rows if job_index is None else evaluated_rows[job_index]
            if job_index not in effect_cache:
                effect_cache[job_index] = {
                    metric: _paired_effect(
                        self.baseline_rows, rows, metric,
                        seed=settings.seed, n_boot=settings.n_bootstrap,
                    )
                    for metric in ("epe", "bad3", "d1")
                }
            run.update(test_rows=rows, effects=effect_cache[job_index])

        return dict(
            layer=deepcopy(layer),
            profile_settings=profile_result["profile_settings"],
            ablation_settings=asdict(settings),
            model_fingerprint=self.model_fingerprint,
            baseline_rows=self.baseline_rows,
            runs=runs,
        )

    # %% save results
    def save_result(self, layer: dict, result: dict, analysis: str) -> str:
        """Save one layer's profile or ablation result with a familiar filename."""
        if analysis not in {"profile", "ablation"}:
            raise ValueError("analysis must be profile or ablation")
        filename = Path(self.mt_dir) / f"{layer['name']}_{analysis}.pt"
        torch.save(result, filename)
        return str(filename)

    # %% population plots (optional matplotlib dependency)
    def plot_layer(self, layer: dict, profile_result: dict, filename=None):
        """Show monocular/stereo responses and the coupling-score distribution."""
        import matplotlib.pyplot as plt

        data = profile_result["responses"]
        active = data["active"].numpy().reshape(-1)
        binocular = data["B"].numpy().reshape(-1)
        monocular = torch.maximum(data["L"], data["R"]).numpy().reshape(-1)
        coupling = data["coupling"].numpy().reshape(-1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        axes[0].scatter(binocular[active], monocular[active], s=10, alpha=0.5)
        upper = max(float(binocular[active].max()), float(monocular[active].max()), 1e-12) if active.any() else 1
        axes[0].plot([0, upper], [0, upper], color="gray", linestyle="--")
        axes[0].set(xlabel="Stereo response B", ylabel="Preferred-eye response M")
        axes[1].hist(coupling[active], bins=40, range=(-1, 1), color="slateblue")
        axes[1].axvline(self.config_mt.weak_threshold, color="orange", linestyle="--", label="Weak boundary")
        axes[1].axvline(self.config_mt.strong_threshold, color="green", linestyle="--", label="Strong boundary")
        axes[1].set(xlabel="Coupling score", ylabel="Number of stereo-active units")
        axes[1].legend()
        fig.suptitle(f"{self.model_name}, {self.interaction}: {layer['name']}")
        filename = filename or Path(self.mt_dir) / f"{layer['name']}_profile.pdf"
        fig.savefig(filename)
        plt.close(fig)

    def plot_ablation_summary(self, layer: dict, result: dict, filename=None):
        """Plot EPE effects; random-control bars show spread across control draws."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for kind in ("weak", "strong", "random", "matched"):
            fractions, means, deviations = [], [], []
            for fraction in self.config_mt.fractions:
                values = [row["effects"]["epe"]["delta"] for row in result["runs"]
                          if row["status"] == "complete" and row["selection"] == kind
                          and row["fraction"] == fraction]
                if values:
                    fractions.append(fraction)
                    means.append(np.mean(values))
                    deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            ax.errorbar(fractions, means, yerr=deviations, marker="o", label=kind, capsize=3)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set(xlabel="Fraction of stereo-active units ablated", ylabel="Change in EPE (pixels)",
               title=f"{self.model_name}, {self.interaction}: {layer['name']}")
        ax.legend()
        fig.text(0.5, -0.04, "Bars: SD across control draws; paired sequence CIs are saved in the result.",
                 ha="center", fontsize=8)
        filename = filename or Path(self.mt_dir) / f"{layer['name']}_ablation_summary.pdf"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


# %% dataset identifiers and a readable singleton-batch collator
class AnalysisDataset(Dataset):
    """Add stable frame/sequence identifiers to the repository SceneFlow dataset."""

    def __init__(self, dataset: Dataset, limit: int | None = None):
        self.dataset = dataset
        if getattr(dataset, "transformation", None) is not None:
            raise ValueError("Analysis datasets must have training augmentation disabled")
        self.n_samples = len(dataset) if limit is None else min(limit, len(dataset))
        if not self.n_samples:
            raise ValueError("Empty analysis dataset")
        self.records = []
        for filename in dataset.left_data[:self.n_samples]:
            path = Path(filename).resolve()
            self.records.append(dict(id=str(path), group=str(path.parent.parent)))
        if len({record["id"] for record in self.records}) != self.n_samples:
            raise ValueError("Duplicate input paths in analysis dataset")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample.update(self.records[index])
        return sample


def _collate_one_sample(samples: list[dict]) -> dict:
    """Images [3,H,W] -> [1,3,H,W]; GT [H,W] -> [1,H,W]; IDs stay strings."""
    if len(samples) != 1:
        raise ValueError("Use batch_size=1 for this analysis")
    sample = samples[0]
    result = dict(id=str(sample["id"]), group=str(sample["group"]))
    for key in ("left", "right", "disp", "ref", "valid"):
        if key not in sample:
            if key == "valid":
                continue
            raise ValueError(f"Missing input field: {key}")
        value = sample[key]
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
        value = torch.as_tensor(value)
        result[key] = value.reshape(1) if key == "ref" else value.unsqueeze(0)
    if result["left"].ndim != 4 or result["left"].shape[1] != 3:
        raise ValueError("Expected normalized RGB images with shape [3,H,W]")
    if result["left"].shape != result["right"].shape:
        raise ValueError("Left and right images must have matching shapes")
    if any(not result[key].is_floating_point() or not torch.isfinite(result[key]).all()
           for key in ("left", "right")):
        raise ValueError("Images must be finite, already-normalized floating-point tensors")
    if result["disp"].shape != (1, *result["left"].shape[-2:]):
        raise ValueError("Disparity must match the image shape")
    if result["ref"].item() not in (-1, 1):
        raise ValueError("ref must be +1 or -1")
    if "valid" in result and result["valid"].dtype != torch.bool:
        raise ValueError("valid must be a boolean mask")
    return result


# %% target module and invocation
@dataclass(frozen=True)
class Site:
    """Module output and zero-based invocation (shared encoder: 0=left, 1=right)."""
    module: str
    call: int = 0

    def __post_init__(self):
        if not self.module or self.call < 0:
            raise ValueError('Need a module name and nonnegative invocation')

    @classmethod
    def parse(cls, text):
        name, sep, index = text.partition('@')
        return cls(name, int(index) if sep else 0)

    def __str__(self):
        return f'{self.module}@{self.call}'


# %% unit definitions and tensor shapes
def _keep(ndim, unit):
    if ndim not in (4, 5):
        raise ValueError(f'Expected BCHW or BCDHW, got rank {ndim}')
    if unit == 'channel':
        return (1,)
    if unit == 'feature':
        return (1, 2) if ndim == 5 else (1,)
    if unit == 'neuron':
        return tuple(range(1, ndim))
    raise ValueError('unit must be channel, feature or neuron')


def _reduce(x, unit):
    keep = _keep(x.ndim, unit)
    axes = tuple(i for i in range(1, x.ndim) if i not in keep)
    return x.mean(dim=axes) if axes else x


def _broadcast(x, output, unit):
    keep = _keep(output.ndim, unit)
    expected = tuple(output.shape[i] for i in keep)
    if tuple(x.shape) != expected:
        raise ValueError(f'Unit shape changed: {tuple(x.shape)} != {expected}')
    shape = [1] * output.ndim
    for i, size in zip(keep, x.shape):
        shape[i] = size
    return x.to(device=output.device).view(shape)


# %% temporary evaluation mode and forward hooks
@contextmanager
def _evaluation(model):
    # Restore every submodule's original mode, including mixed train/eval states.
    states = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, mode in states:
            module.training = mode


@contextmanager
def _cost_volume_probe(model):
    """Expose build_costVol output without editing repository source or parameters."""
    decoder = model.decoder
    name = 'monocular_cost_volume'
    if hasattr(decoder, name):
        raise ValueError(f'{name} already exists')
    had_local = 'build_costVol' in decoder.__dict__
    local = decoder.__dict__.get('build_costVol')
    original = decoder.build_costVol
    decoder.add_module(name, nn.Identity())
    decoder.build_costVol = lambda *a, **kw: getattr(decoder, name)(original(*a, **kw))
    try:
        yield f'decoder.{name}'
    finally:
        if had_local:
            decoder.build_costVol = local
        else:
            del decoder.build_costVol
        delattr(decoder, name)


class _CaptureComplete(Exception):
    """Internal signal: profiling needs no computation beyond the last target."""


@contextmanager
def _capture(model, sites, stop_after_capture=False):
    """Clone at hook time: downstream in-place ReLUs cannot change recorded values."""
    sites = list(sites)
    if len(set(sites)) != len(sites):
        raise ValueError('Duplicate sites')
    values, counts, handles = {}, {}, []
    modules = dict(model.named_modules())
    def hook_for(name):
        def hook(_module, _inputs, output):
            index = counts.get(name, 0)
            counts[name] = index + 1
            site = Site(name, index)
            if site in sites:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f'{site} does not return a tensor; choose a child module')
                if not torch.isfinite(output).all():
                    raise ValueError(f'Nonfinite activation at {site}')
                values[site] = output.detach().float().clone()
                if stop_after_capture and len(values) == len(sites):
                    raise _CaptureComplete()
        return hook
    try:
        for name in dict.fromkeys(s.module for s in sites):
            if name not in modules:
                raise ValueError(f'Unknown module {name}')
            handles.append(modules[name].register_forward_hook(hook_for(name)))
        def reset():
            values.clear()
            counts.clear()
        yield values, reset
    finally:
        for handle in handles:
            handle.remove()


def _forward_capture(model, forward, sample, values, reset, sites):
    reset()
    try:
        forward(model, sample)
    except _CaptureComplete:
        pass
    missing = set(sites) - values.keys()
    if missing:
        raise ValueError(f'Unexecuted sites: {missing}')
    return dict(values)


# %% GPU microbatches with independent normalization
@contextmanager
def _independent_batchnorm(model):
    """Match separate batch-size-one calls when packing independent jobs together.

    Repository BatchNorm2d has no running statistics. Ordinary batching mixes
    conditions/lesions. Temporarily normalize each sample/channel independently,
    using the same eps and affine parameters. Tracked eval BatchNorm is unchanged.
    Parameter tensors, buffers, module identities and hooks are preserved.
    """
    changed = []

    def make_forward(module, original):
        def forward(inputs):
            if module.training:
                raise ValueError("Independent normalization requires model.eval()")
            if inputs.shape[0] == 1:
                return original(inputs)
            if inputs.ndim < 3:
                raise ValueError("Independent normalization needs spatial dimensions")
            return torch.nn.functional.instance_norm(
                inputs, running_mean=None, running_var=None,
                weight=module.weight, bias=module.bias,
                use_input_stats=True, momentum=0.0, eps=module.eps,
            )
        return forward

    try:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.running_mean is None and module.running_var is None:
                    had_local = "forward" in module.__dict__
                    local = module.__dict__.get("forward")
                    changed.append((module, had_local, local))
                    module.forward = make_forward(module, module.forward)
        yield
    finally:
        for module, had_local, local in reversed(changed):
            if had_local:
                module.forward = local
            else:
                del module.forward


def _stack_samples(samples):
    """Pack independent singleton inputs: N x [1,C,H,W] -> [N,C,H,W]."""
    batch = dict(samples[0])
    for key in ("left", "right", "disp", "ref", "valid"):
        if key in batch:
            batch[key] = torch.cat([sample[key] for sample in samples], dim=0)
    return batch


@contextmanager
def _ablate_microbatch(model, site, masks, unit, replacement=None):
    """Different fixed lesion per batch row; shared-eye call selection is preserved."""
    count = 0
    seen = False

    def reset(_module, _inputs):
        nonlocal count, seen
        count, seen = 0, False

    def hook(_module, _inputs, output):
        nonlocal count, seen
        invocation, count = count, count + 1
        if invocation != site.call:
            return
        seen = True
        if output.shape[0] != masks.shape[0]:
            raise ValueError("Mask count must match the independent job batch")
        keep = _keep(output.ndim, unit)
        if tuple(masks.shape[1:]) != tuple(output.shape[i] for i in keep):
            raise ValueError("Activation unit shape differs from the calibration mask")
        shape = [output.shape[0]] + [1] * (output.ndim - 1)
        for axis, size in zip(keep, masks.shape[1:]):
            shape[axis] = size
        selected = masks.to(output.device).reshape(shape)
        value = 0 if replacement is None else _broadcast(replacement, output, unit)
        return torch.where(selected, value, output)

    def verify(_module, _inputs, _output):
        if not seen:
            raise ValueError(f"Ablation site was not executed: {site}")

    handles = []
    try:
        module = model.get_submodule(site.module)
        handles.append(model.register_forward_pre_hook(reset))
        handles.append(module.register_forward_hook(hook))
        handles.append(model.register_forward_hook(verify))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _prediction_rows(prediction, sample, disp_range=None):
    """Reduce all job predictions on the device; copy only small metric vectors.

    prediction: [J,H,W]; sample contains one image's [1,H,W] signed GT.
    Each returned row describes the same test image under a different lesion.
    """
    prediction = prediction.detach().float()
    ground_truth = sample["disp"].to(prediction.device).float()
    if (prediction.ndim != 3 or ground_truth.ndim != 3
            or ground_truth.shape[0] != 1 or prediction.shape[1:] != ground_truth.shape[1:]):
        raise ValueError("Prediction must be [J,H,W] and ground truth [1,H,W]")
    valid = torch.isfinite(ground_truth)
    if "valid" in sample:
        if sample["valid"].shape != ground_truth.shape:
            raise ValueError("Valid mask shape differs from ground truth")
        valid &= sample["valid"].to(prediction.device).bool()
    if disp_range is not None:
        valid &= (ground_truth >= disp_range[0]) & (ground_truth < disp_range[1])

    error = (prediction - ground_truth).abs()
    invalid_prediction = (valid & ~torch.isfinite(prediction)).flatten(1).any(1)
    n_valid = valid.sum().expand(prediction.shape[0])
    denominator = n_valid.clamp_min(1)
    epe = torch.where(valid, error, 0).flatten(1).sum(1) / denominator
    bad3 = ((error > 3) & valid).flatten(1).sum(1) / denominator
    d1 = ((error > 3) & (error > .05 * ground_truth.abs()) & valid).flatten(1).sum(1) / denominator
    packed = torch.stack([n_valid, epe, bad3, d1, invalid_prediction], dim=1).cpu().tolist()

    ref = int(torch.as_tensor(sample["ref"]).item())
    rows = []
    for count, epe_value, bad3_value, d1_value, invalid in packed:
        if not count:
            raise ValueError(f"No valid pixels for {sample['id']}")
        if invalid:
            raise ValueError("Nonfinite prediction at a valid pixel")
        rows.append(dict(
            id=str(sample["id"]), group=str(sample.get("group", sample["id"])),
            ref=ref, n_valid=int(count), epe=epe_value, bad3=bad3_value, d1=d1_value,
        ))
    return rows


@contextmanager
def _reuse_encoder_features(model, enabled):
    """Cache identical upstream features across decoder lesions of one image.

    The first microbatch computes encoder features. Subsequent microbatches
    broadcast its first (identical) input's features. Never used for encoder
    lesions, and never shared across different test images.
    """
    if not enabled or not hasattr(model, "encoder"):
        yield
        return
    encoder = model.encoder
    original = encoder.forward
    had_local = "forward" in encoder.__dict__
    local = encoder.__dict__.get("forward")
    cached = None

    def forward(inputs):
        nonlocal cached
        if not hasattr(inputs, "left"):
            return original(inputs)
        if cached is None:
            result = original(inputs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("Expected the repository encoder's left/right feature tuple")
            cached = tuple(features[:1].clone() for features in result)
            return result
        return tuple(features.expand(inputs.left.shape[0], *features.shape[1:])
                     for features in cached)

    encoder.forward = forward
    try:
        yield
    finally:
        if had_local:
            encoder.forward = local
        else:
            del encoder.forward


def _evaluate_ablation_jobs(model, loader, forward, site, masks, unit,
                            replacement, batch_size, device, disp_range=None, reuse_encoder=True):
    """Image-major evaluation: decode each test image once per layer, pack lesions.

    The loader is read once for all masks. GPU memory grows with batch_size,
    rather than the total number of lesion experiments. No concurrent threads
    mutate hooks or normalization state on the shared model.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results = [[] for _ in masks]
    if not masks:
        return results
    n_images = 0
    replacement_device = None if replacement is None else replacement.to(device)
    with _evaluation(model), _independent_batchnorm(model):
        for sample in loader:
            if sample["left"].shape[0] != 1:
                raise ValueError("Input loader must yield one test image at a time")
            sample = {
                key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
                for key, value in sample.items()
            }
            with _reuse_encoder_features(
                model, enabled=reuse_encoder and site.module.startswith("decoder."),
            ):
                for start in range(0, len(masks), batch_size):
                    chunk = masks[start:start + batch_size]
                    mask_batch = torch.stack(chunk).to(device, non_blocking=True)
                    batch = _stack_samples([sample] * len(chunk))
                    with _ablate_microbatch(model, site, mask_batch, unit, replacement_device):
                        predictions = forward(model, batch)
                    rows = _prediction_rows(predictions, sample, disp_range)
                    for offset, row in enumerate(rows):
                        results[start + offset].append(row)
            n_images += 1
    if not n_images:
        raise ValueError("Empty evaluation set")
    return results


# %% four-condition responses and coupling scores
def _measure_responses(model, loader, sites, forward, gray, unit='channel', activity_floor=1e-6,
            weak_threshold=0.2, strong_threshold=0.6, batch_size=1):
    """Independent LR, L0, 0R, 00 jobs; mean(abs(a_condition-a_00)).

    Loader yields singleton-batch images. Up to batch_size conditions share a
    forward call, with live BatchNorm isolated per sample. Profiling exits as
    soon as every requested target activation is captured. Gray is already in
    normalized input coordinates. Each score still describes singleton inference.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sites = list(sites)
    if not sites or not -1 <= weak_threshold < strong_threshold <= 1 or activity_floor < 0:
        raise ValueError('Invalid sites, thresholds or activity floor')
    rows = {str(s): {} for s in sites}
    ids, groups, refs = [], [], []
    with _evaluation(model), _independent_batchnorm(model), _capture(model, sites, stop_after_capture=True) as (values, reset):
        for sample in loader:
            if sample['left'].shape[0] != 1 or sample['right'].shape[0] != 1:
                raise ValueError('Use batch_size=1 for profiling')
            left, right = sample['left'], sample['right']
            gl, gr = gray(left), gray(right)
            acts = {}
            conditions = [('00', gl, gr), ('L0', left, gr),
                          ('0R', gl, right), ('LR', left, right)]
            # Pack independent conditions without mixing normalization statistics.
            for start in range(0, len(conditions), batch_size):
                chunk = conditions[start:start + batch_size]
                batch = _stack_samples([
                    {**sample, 'left': l, 'right': r} for _, l, r in chunk
                ])
                captured = _forward_capture(model, forward, batch, values, reset, sites)
                for index, (condition, _, _) in enumerate(chunk):
                    acts[condition] = {site: activation[index:index + 1]
                                       for site, activation in captured.items()}
            for site in sites:
                a0 = acts['00'][site]
                if any(a[site].shape != a0.shape for a in acts.values()):
                    raise ValueError('Activation shape differs across conditions')
                measurements = {
                    'B': _reduce((acts['LR'][site] - a0).abs(), unit),
                    'L': _reduce((acts['L0'][site] - a0).abs(), unit),
                    'R': _reduce((acts['0R'][site] - a0).abs(), unit),
                    'interaction': _reduce((acts['LR'][site] - acts['L0'][site]
                                           - acts['0R'][site] + a0).abs(), unit),
                    'blank_mean': _reduce(a0, unit),
                    'stereo_mean': _reduce(acts['LR'][site], unit),
                    'stereo_abs': _reduce(acts['LR'][site].abs(), unit),
                }
                # One device-to-host transfer for all seven reduced responses.
                packed = torch.stack(list(measurements.values()), dim=0)[:, 0].cpu()
                for index, key in enumerate(measurements):
                    rows[str(site)].setdefault(key, []).append(packed[index])
            ids.append(str(sample['id']))
            groups.append(str(sample.get('group', sample['id'])))
            refs.append(int(torch.as_tensor(sample['ref']).item()))
    if not ids:
        raise ValueError('Empty calibration set')
    for data in rows.values():
        for key in list(data):
            data[key + '_samples'] = torch.stack(data[key])
            data[key] = data[key + '_samples'].mean(0)
        b, l, r = data['B'], data['L'], data['R']
        m = torch.maximum(l, r)
        data['coupling'] = (b-m) / (b+m).clamp_min(1e-12)
        data['ocular_dominance'] = (l-r) / (l+r).clamp_min(1e-12)
        data['active'] = b > activity_floor
        data['weak'] = data['active'] & (data['coupling'] <= weak_threshold)
        data['strong'] = data['active'] & (data['coupling'] >= strong_threshold)
        data['mono_only'] = ~data['active'] & (m > activity_floor)
    return dict(sites=rows, ids=ids, groups=groups, refs=refs, unit=unit,
                activity_floor=activity_floor, weak_threshold=weak_threshold,
                strong_threshold=strong_threshold)


# %% equal-dose unit selection and activity-matched controls
def _select_units(data, fraction, kind='weak', seed=0, match_to=None, match_metric='stereo_abs'):
    """Fraction of *stereo-active* units; raise rather than silently change dose.

    weak/strong are continuous rankings (not threshold-defined class selection).
    random draws from the same active population. matched draws distinct units
    outside match_to, greedily matching stereo activation magnitude in log space. Use match_metric='B'
    to instead match blank-corrected response strength.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must lie in [0,1]')
    pool = torch.where(data['active'].reshape(-1))[0]
    k = math.ceil(len(pool) * fraction)
    if not len(pool) and fraction > 0:
        raise ValueError('No stereo-active units at this site')
    rng = torch.Generator().manual_seed(seed)
    scores = data['coupling'].reshape(-1)
    if kind in ('weak', 'strong'):
        order = torch.argsort(scores[pool], descending=kind == 'strong', stable=True)
        chosen = pool[order[:k]]
    elif kind == 'random':
        chosen = pool[torch.randperm(len(pool), generator=rng)[:k]]
    elif kind == 'matched':
        if match_to is None or tuple(match_to.shape) != tuple(data['active'].shape):
            raise ValueError('matched requires a target mask with matching shape')
        targets = torch.where(match_to.reshape(-1))[0]
        if len(targets) != k:
            raise ValueError('Target count differs from requested dose')
        pool = pool[~match_to.reshape(-1)[pool]]
        if len(pool) < k:
            raise ValueError('Insufficient disjoint active units for matched control')
        # Random tie breaking makes repeated controls meaningful.
        pool = pool[torch.randperm(len(pool), generator=rng)]
        magnitude = data[match_metric].reshape(-1).clamp_min(1e-12).log()
        chosen = []
        for target in targets[torch.randperm(k, generator=rng)]:
            j = (magnitude[pool] - magnitude[target]).abs().argmin()
            chosen.append(pool[j].item())
            pool = torch.cat([pool[:j], pool[j+1:]])
        chosen = torch.tensor(chosen, dtype=torch.long)
    else:
        raise ValueError(f'Unknown selection {kind}')
    mask = torch.zeros_like(data['active']).reshape(-1)
    mask[chosen] = True
    return mask.reshape(data['active'].shape)


# %% reversible unit ablation
@contextmanager
def _ablate_units(model, site, mask, unit='channel', replacement=None):
    """Ablate one invocation; fresh tensor preserves sibling residual branches.

    replacement=None means literal zero, otherwise supply per-unit calibration
    stereo_mean or blank_mean. No weights or normalizer buffers are changed.
    """
    count = 0
    seen = False
    def reset(_m, _a):
        nonlocal count, seen
        count, seen = 0, False
    def hook(_m, _a, output):
        nonlocal count, seen
        index, count = count, count + 1
        if index != site.call:
            return
        seen = True
        selected = _broadcast(mask, output, unit)
        value = 0 if replacement is None else _broadcast(replacement, output, unit)
        return torch.where(selected, value, output)
    modules = dict(model.named_modules())
    if site.module not in modules:
        raise ValueError(f'Unknown module {site.module}')
    h0 = model.register_forward_pre_hook(reset)
    h1 = modules[site.module].register_forward_hook(hook)
    def verify(_m, _a, _o):
        if not seen:
            raise ValueError(f'Ablation site was not executed: {site}')
    h2 = model.register_forward_hook(verify)
    try:
        yield
    finally:
        for h in (h0, h1, h2):
            h.remove()


# %% stereo metrics and paired sequence bootstrap
def _evaluate_stereo(model, loader, forward, disp_range=None):
    """Per-image metrics. Validity comes from explicit mask and finite GT.

    Signed and zero disparities are valid. D1 = error>3 AND error/abs(GT)>0.05.
    Invalid predictions at valid pixels raise; they are never silently masked.
    """
    rows = []
    with _evaluation(model):
        for sample in loader:
            prediction = forward(model, sample)
            if prediction.shape[0] != 1:
                raise ValueError("Serial evaluation expects singleton predictions")
            rows.extend(_prediction_rows(prediction, sample, disp_range))
    if not rows:
        raise ValueError("Empty evaluation set")
    return rows


def _paired_effect(baseline, intervention, metric='epe', seed=0, n_boot=2000):
    """Image-weighted paired delta, cluster bootstrap over whole sequence groups."""
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    if len(baseline) != len(intervention) or not baseline:
        raise ValueError('Evaluation sets differ or are empty')
    clusters = {}
    for a, b in zip(baseline, intervention):
        if any(a[k] != b[k] for k in ('id', 'group', 'ref', 'n_valid')):
            raise ValueError('Pairing, reference, or validity changed between evaluations')
        clusters.setdefault(a['group'], []).append(b[metric]-a[metric])
    values = [np.asarray(v, dtype=float) for v in clusters.values()]
    delta = float(np.concatenate(values).mean())
    if len(values) < 2:
        ci = [None, None]
    else:
        rng = np.random.default_rng(seed)
        sums = np.asarray([value.sum() for value in values])
        counts = np.asarray([len(value) for value in values])
        draws = np.empty(n_boot, dtype=float)
        # Same cluster draws as the serial algorithm, without concatenating images.
        for start in range(0, n_boot, 128):
            stop = min(start + 128, n_boot)
            indices = rng.integers(len(values), size=(stop - start, len(values)))
            draws[start:stop] = sums[indices].sum(1) / counts[indices].sum(1)
        ci = np.quantile(draws, [.025, .975]).tolist()
    return dict(delta=delta, ci95=ci, n_images=len(baseline), n_groups=len(values),
                baseline=float(np.mean([x[metric] for x in baseline])),
                intervention=float(np.mean([x[metric] for x in intervention])))
