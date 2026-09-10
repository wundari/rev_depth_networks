# %% load necessary modules
from config.config import ConfigGCNet
from GroupAnalysis.group_analysis_rds import GA_RDS

import numpy as np

# %%
config = ConfigGCNet()
# rds parameters
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 512,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 8,
}

ga_rds = GA_RDS(config, params_rds)

# %% compute disparity map
# n_bootstrap = 1000
# interactions = ["default", "bem", "cmm", "sum_diff"]
# for interaction in interactions:
#     ga_rds.compute_disp_map_all_seeds(interaction, n_bootstrap)

# %% plot depth performance averaged across all seeds
save_flag = True
interactions = ["default", "bem", "cmm", "sum_diff"]
for interaction in interactions:
    ga_rds.plotLine_xDecode_all_seeds(interaction, save_flag)

# %% plot depth performance for every seeds

for interaction in interactions:
    for s, seed in enumerate(ga_rds.config.seed_to_analyse):

        if interaction == "default":
            epoch, iter = ga_rds.config.epoch_iter_to_load_default[s]
        elif interaction == "bem":
            epoch, iter = ga_rds.config.epoch_iter_to_load_bem[s]
        elif interaction == "cmm":
            epoch, iter = ga_rds.config.epoch_iter_to_load_cmm[s]
        else:  # sum_diff
            epoch, iter = ga_rds.config.epoch_iter_to_load_sum_diff[s]

        # update network configuration and directory addresses
        ga_rds.update_network_config(interaction, seed, epoch, iter)

        # plot cross-decoding performance
        ga_rds.plotLine_xDecode(save_flag)

        # plot predicted disparity map for a single bootstrap
        ga_rds.plotHeat_dispMap(save_flag)

        # plot predicted disparity map averaged across all bootstrap
        ga_rds.plotHeat_dispMap_avg(save_flag)

# %%
