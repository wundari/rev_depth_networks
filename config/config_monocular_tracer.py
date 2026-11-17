# %% load necessary modules
from dataclasses import dataclass, field


# %%
@dataclass
class ConfigMonocularTracer:
    """Settings for monocular-response profiling and stereo-performance ablation.

    Model architecture, checkpoint, dataset and training seed remain in your
    existing ConfigBNN or ConfigGCNet. This class contains analysis settings only.
    """

    # unit to analyse
    unit: str = "channel"  # "channel", "feature", or "neuron"
    # channel: [C], average over disparity and spatial positions
    # feature: [C, D] for a 3D layer; [C] for a 2D layer
    # neuron:  [C, D, H, W] for a 3D layer; [C, H, W] for a 2D layer

    # blank-eye image, in raw RGB coordinates before normalization
    baseline: float = 0.5
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # operational coupling categories; these are adjustable hypotheses
    activity_floor: float = 1e-6
    weak_threshold: float = 0.2
    strong_threshold: float = 0.6

    # fractions of ALL stereo-active units, ranked by coupling score
    fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    n_controls: int = 10  # independent random/activity-matched control draws
    replacement: str = "zero"  # "zero", "mean", or "blank"
    # zero: literal zero; mean: calibration stereo mean; blank: calibration 00 mean

    # evaluation and uncertainty
    disp_range: tuple[float, float] | None = None  # optional [minimum, maximum)
    n_bootstrap: int = 1000  # resample whole sequences, keeping image pairs together
    seed: int = 1618  # randomness for unit controls and bootstrap samples

    # full datasets by default; limits take the first N samples for a pilot
    max_calibration_samples: int | None = None
    max_test_samples: int | None = None
    num_workers: int = 4  # parallel image decoding; set 0 for notebook/debugging
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # independent GPU jobs per forward call; start at 2 for full-resolution GCNet
    inference_batch_size: int = 4  # try 4, then 8 if memory permits
    reuse_encoder_features: bool = True  # decoder lesions of the same test image
    amp_dtype: str = "bfloat16"  # optional "bfloat16" or "float16" on CUDA
    # AMP changes numerical precision: check against float32 before scientific use.
    # The input loader still yields singleton images. Normalization is isolated
    # per job, matching the original single-image analysis mathematically.

    # None -> engine.experiment_dir/monocular_tracer/<timestamp>
    output_dir: str | None = None
    save_figures: bool = True  # requires matplotlib; only imported by plot methods

    # Optional explicit list of target-layer dictionaries.
    # Empty list -> sensible defaults for BNN or GC_Net_LR.
    target_layers: list[dict] = field(default_factory=list)
    # Example:
    # [dict(name="layer19", module="decoder.layer19", call=0)]
    # Shared encoder modules: call=0 is physical left, call=1 physical right.
    # Both-gray responses are used as the activation baseline at every layer.

    def validate(self):
        if self.amp_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("amp_dtype must be float32, bfloat16, or float16")
        if (
            self.inference_batch_size < 1
            or int(self.inference_batch_size) != self.inference_batch_size
        ):
            raise ValueError("inference_batch_size must be a positive integer")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        if self.unit not in {"channel", "feature", "neuron"}:
            raise ValueError("unit must be channel, feature, or neuron")
        if not 0 <= self.baseline <= 1:
            raise ValueError("baseline must be a raw RGB value in [0, 1]")
        if len(self.mean) != 3 or len(self.std) != 3 or min(self.std) <= 0:
            raise ValueError("Use three RGB means and positive standard deviations")
        if not -1 <= self.weak_threshold < self.strong_threshold <= 1:
            raise ValueError(
                "Coupling thresholds must satisfy -1 <= weak < strong <= 1"
            )
        if self.activity_floor < 0:
            raise ValueError("activity_floor must be nonnegative")
        if not self.fractions or any(not 0 <= value <= 1 for value in self.fractions):
            raise ValueError("fractions must be nonempty and between 0 and 1")
        if self.n_controls < 1 or self.n_bootstrap < 1 or self.num_workers < 0:
            raise ValueError(
                "Use positive control/bootstrap counts and nonnegative workers"
            )
        if self.replacement not in {"zero", "mean", "blank"}:
            raise ValueError("replacement must be zero, mean, or blank")
        if self.disp_range is not None:
            if len(self.disp_range) != 2 or self.disp_range[0] >= self.disp_range[1]:
                raise ValueError("disp_range must be an increasing (minimum, maximum)")
        for value in (self.max_calibration_samples, self.max_test_samples):
            if value is not None and value < 1:
                raise ValueError("Sample limits must be positive or None")
