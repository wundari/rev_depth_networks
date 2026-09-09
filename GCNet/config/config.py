# %%
import torch

# from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


# %%
@dataclass_json
@dataclass
class GCNetconfig:

    model_name: str = "GC-Net"
    # learning params
    lr: float = 6e-4
    # lr_decay_rate: float = 0.99
    max_lr: float | None = None
    min_lr: float | None = None
    warmup_steps: int = 250  # 1000
    # max_steps_lr: int = 20000  # max steps for learning rate function

    # dataset parameter
    dataset: str = "sceneflow_monkaa"
    dataset_directory: str = ""
    validation: str = "validation"
    checkpoint: str = "dev"
    c_disp_shift: float = 2.0  # a multiplier for shifting disparity map

    # training params
    batch_size: int = 4
    batch_size_val: int = 4
    num_workers: int = 2
    eval_num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 2
    loader_timeout: int = 120
    log_interval: int = 20
    amp_dtype: str = "bfloat16"  # "float16" or "float32" also supported
    deterministic: bool = True
    split_seed: int = 42  # fixed across all model seeds/interactions
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    weight_decay: float = 1e-4
    start_epoch: int = 0
    if dataset == "sceneflow_monkaa":
        epochs: int = 10
    elif dataset == "sceneflow_flying":
        epochs: int = 5
    eval_interval: int = 100  # interval for calculating validation error
    eval_iter: int = 200  # the number of iterations for validation
    clip_max_norm: float = 0.1  # gradient clipping max norm
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    compile_mode: str | None = "max-autotune"  # "reduce-overhead", "max-autotune"
    seed: int = 11364
    loss: str = "smooth_l1"

    # GCNet
    img_height: int = 256  # crop height
    img_width: int = 512  # crop width
    in_channels: int = 3  # input channels, RGB
    base_channels: int = 32
    n_resBlocks: int = 8  # the number of residual blocks
    max_disp: int = 192  # disparity range
    binocular_interaction: str = "default"
    interactions: list[str] = field(
        default_factory=lambda: ["default", "bem", "cmm", "sum_diff"]
    )

    # Loss
    px_error_threshold: int = 3  # Number of pixels for error computation (default 3 px)
    validation_max_disp: int = -1

    # resume from checkpoint
    load_state: bool = True
    if load_state:
        compile_mode = None
    experiment_id: int = 6  # experiment id for loading pretrained DNN
    epoch_to_load = 8
    iter_to_load = 15600
    resume: str = (
        f"epoch_{epoch_to_load}_iter_{iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
    )
    # resume = f"epoch_{epoch_to_load}_model.pth.tar"

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

    # pre-trained model, sorted in order of seed_to_analyse
    # bino_interaction: default, batch size 4, v1
    # epoch_iter_to_load_default: list[tuple[int, int]] = field(
    #     default_factory=lambda: [
    #         (8, 18100),  # seed 1618
    #         (9, 20400),  # seed 11364
    #         (9, 21400),  # seed 16476
    #         (7, 16500),  # seed 27829
    #         (7, 17300),  # seed 35154
    #         (8, 19300),  # seed 35744
    #         (9, 20900),  # seed 36675
    #         (6, 14300),  # seed 43798
    #         (6, 14200),  # seed 55826
    #         (5, 12700),  # seed 59035
    #         (7, 16000),  # seed 65190
    #         (9, 21000),  # seed 82220
    #         (9, 21000),  # seed 94750
    #     ]
    # )

    # bino_interaction: default, batch size 4
    epoch_iter_to_load_default: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (8, 18100),  # seed 1618
            (9, 20400),  # seed 11364
            (8, 18100),  # seed 16476
            (7, 16500),  # seed 27829
            (7, 17300),  # seed 35154
            (8, 19300),  # seed 35744
            (9, 20900),  # seed 36675
            (9, 19800),  # seed 43798
            (6, 14200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (9, 21000),  # seed 82220
            (9, 21000),  # seed 94750
        ]
    )

    # bino_interaction: bem, batch size 4
    epoch_iter_to_load_bem: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (7, 15500),  # seed 1618
            (9, 20200),  # seed 11364
            (9, 21500),  # seed 16476
            (9, 19500),  # seed 27829
            (8, 18300),  # seed 35154
            (8, 19300),  # seed 35744
            (9, 20900),  # seed 36675
            (6, 14300),  # seed 43798
            (8, 18400),  # seed 55826
            (7, 16400),  # seed 59035
            (7, 16000),  # seed 65190
            (9, 21000),  # seed 82220
            (7, 15800),  # seed 94750
        ]
    )

    # bino_interaction: cmm, batch size 4
    epoch_iter_to_load_cmm: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (8, 18100),  # seed 1618
            (9, 21300),  # seed 11364
            (8, 18100),  # seed 16476
            (7, 16500),  # seed 27829
            (7, 17300),  # seed 35154
            (6, 14800),  # seed 35744
            (9, 20900),  # seed 36675
            (6, 14300),  # seed 43798
            (6, 14200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (9, 21000),  # seed 82220
            (9, 21100),  # seed 94750
        ]
    )

    # bino_interaction: sum_diff, batch size 4
    epoch_iter_to_load_sum_diff: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (8, 18100),  # seed 1618
            (7, 15800),  # seed 11364
            (8, 18100),  # seed 16476
            (7, 16500),  # seed 27829
            (7, 17300),  # seed 35154
            (8, 19300),  # seed 35744
            (9, 20900),  # seed 36675
            (6, 14300),  # seed 43798
            (6, 14200),  # seed 55826
            (8, 17700),  # seed 59035
            (7, 16000),  # seed 65190
            (9, 21000),  # seed 82220
            (9, 21000),  # seed 94750
        ]
    )
    # bino_interaction: default, batch size 16
    # epoch_iter_to_load_default: list[tuple[int, int]] = field(
    #     default_factory=lambda: [
    #         (7, 3900),  # seed 1618
    #         (9, 4900),  # seed 11364
    #         (9, 5100),  # seed 16476
    #         (7, 4200),  # seed 27829
    #         (9, 5400),  # seed 35154
    #         (9, 5100),  # seed 35744
    #         (8, 4500),  # seed 36675
    #         (8, 4700),  # seed 43798
    #         (8, 4800),  # seed 55826
    #         (8, 4800),  # seed 59035
    #         (9, 5100),  # seed 65190
    #         (9, 5200),  # seed 82220
    #         (9, 5200),  # seed 94750
    #     ]
    # )

    def __post_init__(self):
        if self.max_lr is None:
            self.max_lr = self.lr
        if self.min_lr is None:
            self.min_lr = self.max_lr * 0.1
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0:
            raise ValueError("Require 0 <= min_lr <= max_lr and max_lr > 0")
