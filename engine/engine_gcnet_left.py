# %% load necessary modules
"""
GCNet training/evaluation engine.
"""

from config.config_gcnet_left import ConfigGCNet
from GC_Net_L.modules.gcnet import build_gcnet

from engine.engine_base import EngineBase


class EngineGCNet(EngineBase):

    MODEL_LABEL = "GCNet_L"

    def __init__(self, config: ConfigGCNet) -> None:
        super().__init__(config)

    def _build_model(self, config: ConfigGCNet):
        return build_gcnet(config)
