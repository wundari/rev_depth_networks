# %%
import torch

# from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


# %%
@dataclass_json
@dataclass
class Config:

    # learning params
    lr: float = 6e-4
    # lr_decay_rate: float = 0.99
    max_lr: float = 6e-4  # 6e-4
    min_lr: float = 3e-5
    warmup_steps: int = 250

    # dataset parameter
    dataset: str = "sceneflow_monkaa"
    dataset_directory: str = ""
    validation: str = "validation"
    checkpoint: str = "dev"
    c_disp_shift: float = 2.0  # a multiplier for shifting disparity map
    validation_fraction: float = 0.1  # fraction of dataset used for validation
    test_fraction: float = 0.1  # fraction of dataset used for testing
    split_seed: int = 42  # fixed across all model seeds/interactions

    # training params
    batch_size: int = 4
    batch_size_val: int = 8
    num_workers: int = 4  # the number of cpu cores for train dataloader
    eval_num_workers: int = 4  # the number of cpu cores for val dataloader
    persistent_workers: bool = True
    prefetch_factor: int = 2
    loader_timeout: int = 120
    amp_dtype: str = "bfloat16"  # "float16" or "float32" also supported
    deterministic: bool = True
    weight_decay: float = 1e-4
    start_epoch: int = 0
    if dataset == "sceneflow_monkaa":
        epochs: int = 10
    elif dataset == "sceneflow_flying":
        epochs: int = 5

    log_interval: int = 50  # interval for tqdm logging
    eval_interval: int = 500  # interval for calculating validation error
    n_iter_eval: int = 400  # the number of iterations for validation
    save_snapshot: bool = True  # whether to save predicted disparity map
    snapshot_interval: int = (
        1000  # interval for visualizing the predicted disparity map
    )
    clip_max_norm: float = 1.0  # gradient clipping max norm
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    compile_mode: str | None = "reduce-overhead"  # "reduce-overhead", "max-autotune"

    # Loss
    loss: str = "smooth_l1"
    px_error_threshold: int = 3  # Number of pixels for error computation (default 3 px)
    validation_max_disp: int = -1

    # network parameters
    img_height: int = 256  # crop height
    img_width: int = 512  # crop width
    in_channels: int = 3  # input channels, RGB
    base_channels: int = 32
    n_resBlocks: int = 8  # the number of residual blocks
    max_disp: int = 192  # disparity range
    interactions: list[str] = field(
        default_factory=lambda: ["default", "bem", "cmm", "sum_diff"]
    )

    # rds analysis
    seed_to_analyse: list[int] = field(
        default_factory=lambda: [
            1618,
            11364,
            16476,
            27829,
            35154,
            35744,
            36675,
            43798,
            55826,
            59035,
            65190,
            82220,
            94750,
        ]
    )
