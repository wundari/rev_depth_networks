from config.config import Config
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConfigGCNet(Config):

    model_name: str = "GC_Net"
    binocular_interaction: str = "default"
    seed: int = 1618

    # resume from checkpoint
    load_state: bool = False
    if load_state:
        compile_mode = None
    experiment_id: int = 0  # experiment id for loading pretrained DNN
    epoch_to_load = 8
    iter_to_load = 15400
    resume: str = (
        f"epoch_{epoch_to_load}_iter_{iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
    )

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
