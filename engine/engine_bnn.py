# %% load necessary modules
"""
BNN training/evaluation engine.

"""

from config.config_bnn import ConfigBNN
from BNN.modules.bnn import build_bnn

from engine.engine_base import EngineBase


class EngineBNN(EngineBase):

    MODEL_LABEL = "BNN"

    def __init__(self, config: ConfigBNN) -> None:
        super().__init__(config)

    def _build_model(self, config: ConfigBNN):
        return build_bnn(config)
