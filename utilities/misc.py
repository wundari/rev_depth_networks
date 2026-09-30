# %%
import numpy as np
from pathlib import Path


class NestedTensor:
    def __init__(self, left, right, disp=None, ref=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.ref = ref


def load_layer_activation(
    layer_act_dir: str,
    activation_metadata: dict,
    dotDens: float,
):
    data, labels = {}, {}
    for rds_cond, dotMatch in (("ards", 0.0), ("hmrds", 0.5), ("crds", 1.0)):
        suffix = f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
        data[rds_cond] = np.load(
            Path(layer_act_dir) / ("act_rds" + suffix),
            allow_pickle=True,
        ).item()
        if data[rds_cond].get("_metadata") != activation_metadata:
            raise ValueError(
                "Legacy or incompatible activation files; regenerate all conditions"
            )
        labels[rds_cond] = np.load(Path(layer_act_dir) / ("targetDisp_rds" + suffix))

    return data, labels
