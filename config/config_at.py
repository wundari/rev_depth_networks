from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConfigAT:

    target_layer_indices: list[int] = field(default_factory=lambda: list(range(19, 24)))
