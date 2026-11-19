import torch
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConfigBI:

    dnn_type: str = "BNN"
    if dnn_type == "BNN":
        target_layer_indices: list[int] = field(
            default_factory=lambda: list(range(3, 5))
        )
    else:
        target_layer_indices: list[int] = field(
            default_factory=lambda: list(range(19, 38))
        )

    amp_dtype: torch.dtype = torch.bfloat16
    early_exit: bool = True
