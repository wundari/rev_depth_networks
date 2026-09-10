# %%
import numpy as np

from config.config import ConfigGCNet
from GroupAnalysis.group_analysis_superposition import GA_Superposition

# %%
config = ConfigGCNet()
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 256,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 2,
}
config = ConfigGCNet()
ga_sup = GA_Superposition(config, params_rds)

# %%
interactions = ["default", "bem", "cmm", "sum_diff"]
conv_layer_names, _ = ga_sup.get_conv_names_and_weights()
save_flag = True
for interaction in interactions:

    # feature dimensionality analysis
    feat_dim_interaction = ga_sup.feature_dimensionality_interaction(interaction)

    # plot violin feature dimensionality for a given interaction
    ga_sup.plotViolin_feat_dim_interaction(
        feat_dim_interaction, conv_layer_names, save_flag
    )

# %% statistical test
ga_sup.stat_feat_dim()

# %%
