# %%
import torch
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


# %%
@dataclass_json
@dataclass
class BNNconfig:
    # learning params
    lr: float = 6e-4
    # lr_decay_rate: float = 0.99
    max_lr: float = lr
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 250  # 1000
    # max_steps_lr: int = 20000  # max steps for learning rate function

    # dataset parameter
    dataset: str = "sceneflow_monkaa"  # "sceneflow_flying" or "sceneflow_monkka"
    dataset_directory: str = ""
    validation: str = "validation"
    checkpoint: str = "dev"
    c_disp_shift: float = 2  # a multiplier for shifting disparity map

    # training params
    batch_size: int = 4
    batch_size_val: int = 4
    num_workers: int = 2
    eval_num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 2
    log_interval: int = 20
    weight_decay: float = 1e-4
    start_epoch: int = 0
    if dataset == "sceneflow_monkaa":
        epochs: int = 10  # 20
    elif dataset == "sceneflow_flying":
        epochs: int = 5
    eval_interval: int = 100  # interval for calculating validation error
    eval_iter: int = 200  # the number of iterations for validation
    clip_max_norm: float = 0.1  # gradient clipping max norm
    device = "cuda" if torch.cuda.is_available() else "mps"
    compile_mode: str | None = "reduce-overhead"
    seed: int = 11364

    # network parameters
    img_height: int = 256  # crop height
    img_width: int = 512  # crop width
    in_channels: int = 3  # input channels, RGB
    base_channels: int = 32
    n_resBlocks: int = 8  # the number of residual blocks
    max_disp: int = 192  # disparity range
    binocular_interaction: str = "bem"  # "default", "bem", "cmm", "sum_diff"
    interactions: list[str] = field(
        default_factory=lambda: ["default", "bem", "cmm", "sum_diff"]
    )

    # Loss
    px_error_threshold: int = 3  # Number of pixels for error computation (default 3 px)
    validation_max_disp: int = -1

    # resume from checkpoint
    load_state = False
    if load_state:
        compile_mode = None
    experiment_id = 4  # experiment id for loading pretrained DNN
    epoch_to_load = 9
    iter_to_load = 19600
    resume = f"epoch_{epoch_to_load}_iter_{iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
    # resume = f"epoch_{epoch_to_load}_model.pth.tar"

    # group analysis
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
    # bino_interaction: default
    epoch_iter_to_load_default: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 19600),  # seed 1618
            (5, 12100),  # seed 11364
            (8, 18100),  # seed 16476
            (7, 16500),  # seed 27829
            (7, 17300),  # seed 35154
            (5, 11400),  # seed 35744
            (5, 12400),  # seed 36675
            (6, 14300),  # seed 43798
            (6, 14200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (6, 13600),  # seed 82220
            (8, 19000),  # seed 94750
        ]
    )

    # bino_interaction: bem
    epoch_iter_to_load_bem: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (8, 18100),  # seed 1618
            (5, 12100),  # seed 11364
            (8, 17500),  # seed 16476
            (9, 21400),  # seed 27829
            (7, 17300),  # seed 35154
            (5, 11400),  # seed 35744
            (4, 9900),  # seed 36675
            (6, 14300),  # seed 43798
            (9, 21200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (6, 13600),  # seed 82220
            (8, 19000),  # seed 94750
        ]
    )

    # bino_interaction: cmm
    epoch_iter_to_load_cmm: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 19600),  # seed 1618
            (5, 12100),  # seed 11364
            (8, 18100),  # seed 16476
            (5, 12400),  # seed 27829
            (7, 17300),  # seed 35154
            (5, 11400),  # seed 35744
            (7, 16000),  # seed 36675
            (6, 14300),  # seed 43798
            (6, 14200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (6, 13600),  # seed 82220
            (8, 19000),  # seed 94750
        ]
    )

    # bino_interaction: sum_diff
    epoch_iter_to_load_sum_diff: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 19600),  # seed 1618
            (5, 12100),  # seed 11364
            (8, 18100),  # seed 16476
            (5, 12400),  # seed 27829
            (7, 17300),  # seed 35154
            (5, 11400),  # seed 35744
            (5, 12400),  # seed 36675
            (6, 14300),  # seed 43798
            (6, 14200),  # seed 55826
            (5, 12700),  # seed 59035
            (7, 16000),  # seed 65190
            (6, 13600),  # seed 82220
            (8, 19000),  # seed 94750
        ]
    )

    pretrained_models: list[str] = field(
        default_factory=lambda: [
            "epoch_9_iter_19600_model_best.pth.tar",
        ]
    )
