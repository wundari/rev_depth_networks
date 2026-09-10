# %% load necessary modules
"""
GCNet training/evaluation engine.
"""

from GC_Net.config.config import ConfigGCNet
from GC_Net.modules.gcnet import build_gcnet

from engine.engine_base import EngineBase


class EngineGCNet(EngineBase):

    MODEL_LABEL = "GCNet"

    def __init__(self, config: ConfigGCNet) -> None:
        super().__init__(config)

    def _build_model(self, config: ConfigGCNet):
        return build_gcnet(config)
