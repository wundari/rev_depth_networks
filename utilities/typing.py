from typing import Dict, Optional, Callable
from torch import Tensor
from torch.nn import Module

ModuleOutputMapping = Dict[Module, Optional[Tensor]]
LossFunction = Callable[[ModuleOutputMapping], Tensor]
