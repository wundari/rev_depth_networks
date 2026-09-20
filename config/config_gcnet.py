from config.config import Config
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConfigGCNet(Config):

    model_name: str = "GC_Net"
    binocular_interaction: str = "bem"
    seed: int = 1618

    # RDS parameters
    batch_size_rds: int = 8
    assert (
        Config.n_rds_each_disp % batch_size_rds == 0
    ), "n_rds_each_disp must be divisible by batch_size_rds"

    # resume from checkpoint
    load_state: bool = True
    if load_state:
        compile_mode = None
    experiment_id: int = seed  # experiment id for loading pretrained DNN
    epoch_to_load = 7
    iter_to_load = 12800
    model_pretrained: str = (
        f"epoch_{epoch_to_load}_iter_{iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
    )

    # bino_interaction: default, batch size 4
    epoch_iter_to_load_default: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 16500),  # seed 1618
            (9, 16000),  # seed 11364
            (9, 16500),  # seed 16476
            (9, 17000),  # seed 27829
            (9, 17000),  # seed 35154
            (9, 16000),  # seed 35744
            (9, 17589),  # seed 36675
            (9, 17500),  # seed 43798
            (9, 17589),  # seed 55826
            (9, 16500),  # seed 59035
            (9, 16500),  # seed 65190
            (9, 16500),  # seed 82220
            (9, 16500),  # seed 94750
        ]
    )

    # bino_interaction: bem, batch size 4
    epoch_iter_to_load_bem: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 16500),  # seed 1618
            (9, 17500),  # seed 11364
            (9, 16500),  # seed 16476
            (9, 17000),  # seed 27829
            (9, 17500),  # seed 35154
            (9, 16000),  # seed 35744
            (7, 14000),  # seed 36675
            (9, 16500),  # seed 43798
            (9, 17589),  # seed 55826
            (9, 16500),  # seed 59035
            (9, 17000),  # seed 65190
            (8, 15500),  # seed 82220
            (8, 15500),  # seed 94750
        ]
    )

    # bino_interaction: cmm, batch size 4
    epoch_iter_to_load_cmm: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 16000),  # seed 1618
            (9, 17500),  # seed 11364
            (8, 15500),  # seed 16476
            (9, 17500),  # seed 27829
            (9, 16500),  # seed 35154
            (9, 17589),  # seed 35744
            (9, 17589),  # seed 36675
            (9, 17000),  # seed 43798
            (9, 17500),  # seed 55826
            (9, 16500),  # seed 59035
            (7, 14000),  # seed 65190
            (9, 17589),  # seed 82220
            (9, 17500),  # seed 94750
        ]
    )

    # bino_interaction: sum_diff, batch size 4
    epoch_iter_to_load_sum_diff: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (9, 16500),  # seed 1618
            (8, 14500),  # seed 11364
            (9, 16000),  # seed 16476
            (8, 15500),  # seed 27829
            (9, 17000),  # seed 35154
            (9, 16000),  # seed 35744
            (9, 17589),  # seed 36675
            (9, 17500),  # seed 43798
            (9, 17500),  # seed 55826
            (9, 16500),  # seed 59035
            (8, 15500),  # seed 65190
            (9, 17589),  # seed 82220
            (9, 17500),  # seed 94750
        ]
    )
