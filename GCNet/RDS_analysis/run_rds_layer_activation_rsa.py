# %%
from RDS_analysis.rds_layer_activation_rsa import RDS_LayerAct_RSA
import numpy as np

from config.config import GCNetconfig

# %% load GCNet config
config = GCNetconfig()
# config.compile_mode = "default"

# rds parameters
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 128,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 4,
    "n_bootstrap": 1000,
}

seed_number = config.seed
rdsl = RDS_LayerAct_RSA(config, params_rds)


# %% compute layer activation for a specific dot density and dot match
dotDens = 0.9
dotMatch_list = [0.0, 0.5, 1.0]
for dotMatch in dotMatch_list:
    rdsl.compute_layer_act_rds(dotMatch, dotDens, rdsl.background_flag)

# %% compute correlation between layer activation in response to RDSs
corr_crds_ards, corr_crds_hmrds, corr_crds_crds = rdsl.compute_corr_activation(dotDens)

# %% plot
save_flag = 0
rdsl.plotLine_rsa(corr_crds_ards, corr_crds_hmrds, corr_crds_crds, dotDens, save_flag)

# %%
