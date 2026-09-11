from config.config import Config
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


# %%
@dataclass_json
@dataclass
class ConfigBNN(Config):

    model_name: str = "BNN"
    binocular_interaction: str = "cmm"
    seed: int = 1618

    # resume from checkpoint
    load_state: bool = False
    if load_state:
        compile_mode = None
    experiment_id: int = 12  # experiment id for loading pretrained DNN
    epoch_to_load = 9
    iter_to_load = 17000
    resume: str = (
        f"epoch_{epoch_to_load}_iter_{iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
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
            (9, 16000),  # seed 1618
            (7, 13500),  # seed 11364
            (9, 17000),  # seed 16476
            (8, 15000),  # seed 27829
            (9, 17589),  # seed 35154
            (9, 16500),  # seed 35744
            (9, 16000),  # seed 36675
            (8, 15000),  # seed 43798
            (7, 13500),  # seed 55826
            (9, 17500),  # seed 59035
            (9, 17500),  # seed 65190
            (7, 12500),  # seed 82220
            (9, 17000),  # seed 94750
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
