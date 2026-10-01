# %% load necessary modules
"""
GCNet training/evaluation engine.
"""

from config.config_gcnet_gn_lr import ConfigGCNet
from GC_NetGN_LR.modules.gcnet import build_gcnet

from engine.engine_base import EngineBase


class EngineGCNet(EngineBase):

    MODEL_LABEL = "GCNetGN_LR"

    def __init__(self, config: ConfigGCNet) -> None:
        super().__init__(config)

    def _build_model(self, config: ConfigGCNet):
        return build_gcnet(config)
