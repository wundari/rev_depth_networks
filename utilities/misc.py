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
    """

    Args:
        layer_act_dir (str): _description_
        activation_metadata (dict): _description_
        dotDens (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        data = {"ards": {layer19: [n_samples_per_rds_type, n_feat],
                        layer20: [n_samples_per_rds_type, n_feat],
                        ...},
                "hmrds": {layer19: [n_samples_per_rds_type, n_feat],
                        layer20: [n_samples_per_rds_type, n_feat],
                        ...},
                ...}

        labels = {"ards": [n_samples_per_rds_type],
                  "hmrds": [n_samples_per_rds_type],
                  "crds": [n_samples_per_rds_type]}

        ps: n_samples_per_rds_cond = len(self.disp_ct_pix_list) * self.n_rds_each_disp
            n_feat = n_feature_channels * n_disp_channels

    """
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
