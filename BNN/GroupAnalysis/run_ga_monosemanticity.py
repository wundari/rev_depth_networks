# %% load necessary modules
from config.config import BNNconfig
from GroupAnalysis.group_analysis_monosemanticity import GA_Monosemanticity

import numpy as np

# %%
config = BNNconfig()
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

ga_mono = GA_Monosemanticity(config, params_rds)

# %% compute layer activation to RDSs
n_bootstrap = 1000
save_flag = True  # set to True to save the plot

for interaction in ga_mono.interactions:
    ga_mono.compute_layer_act_rds_all_seeds(interaction)

# %% Plot monosemanticity spectrum in each layer
save_flag = True  # set to True to save the plot
for interaction in ga_mono.interactions:
    for s, seed in enumerate(ga_mono.config.seed_to_analyse):

        if interaction == "default":
            epoch, iter = ga_mono.config.epoch_iter_to_load_default[s]
        elif interaction == "bem":
            epoch, iter = ga_mono.config.epoch_iter_to_load_bem[s]
        elif interaction == "cmm":
            epoch, iter = ga_mono.config.epoch_iter_to_load_cmm[s]
        else:  # sum_diff
            epoch, iter = ga_mono.config.epoch_iter_to_load_sum_diff[s]

        # update network configuration and directory addresses
        ga_mono.update_network_config(interaction, seed, epoch, iter)

        # update model
        ga_mono.load_model()

        for i in range(len(ga_mono.layer_name)):
            layer_name = ga_mono.layer_name[i]
            monosemanticity = ga_mono.compute_monosemanticity(layer_name)
            ga_mono.plot_monosemanticity_spectrum_in_layer(
                monosemanticity, layer_name, save_flag
            )

        # ga_mono.plotLine_n_mono_vs_layer(save_flag=save_flag)

# %% plot n_mono per features for all interactions
ga_mono.plotLine_n_mono_vs_layer_all_interactions(
    threshold=0.9, n_features=27, save_flag=True
)

# %%
