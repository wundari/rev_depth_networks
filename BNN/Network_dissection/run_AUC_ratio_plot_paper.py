#%%
# Script for plotting Figure 4d, 4e, 5a
# working dir: Codes/Python/gcnet
# %%
from Network_dissection.AUC_ratio import AUCRatio

# %% prepare GC-net
# network parameters
params_network = {
    "h_bg": 256,
    "w_bg": 512,
    "maxdisp": 192,
}

# training parameters
sceneflow_type = "monkaa"  # dataset on which GC-Net was trained
epoch_to_load = 7
iter_to_load = 22601
target_disp = 10
target_dotDens = 0.25
c_disp_shift = 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
batch_size = 1
params_train = {
    "sceneflow_type": sceneflow_type,
    "c_disp_shift": c_disp_shift,
    "epoch_to_load": epoch_to_load,
    "iter_to_load": iter_to_load,
    "flip_input": flip_input,
    "target_disp": target_disp,
    "target_dotDens": target_dotDens,
    "batch_size": batch_size,
}

aucr = AUCRatio(params_network, params_train)

# %% compute disparity tuning

# rds_type = "grad"
# dotMatch_list = np.arange(0, 1.1, 0.1)
# dotDens_list = [0.25]
# background_flag = True  # with cRDS background

# for dotDens in dotDens_list:
#     aucr.target_dotDens = dotDens
#     for dotMatch in dotMatch_list:
#         aucr.compute_disp_tuning(dotMatch, dotDens, background_flag, rds_type)

# %%
aucr.target_dotDens = 0.25
aucr.load_disp_tuning_layers()  # load disparity tuning for all layers
auc_ratio_all_layer = aucr.compute_corr_auc_ratio_all_layer()

# %%
save_flag = 0
aucr.plotHist_auc_ratio(auc_ratio_all_layer, save_flag) # Figure 4d
aucr.plotLine_auc_ratio(auc_ratio_all_layer, save_flag) # Figure 4e
# aucr.plotLine_auc_ratio_all_dotDens(save_flag)
# aucr.plotLine_auc_as_func_dot_corr(auc_ratio_all_layer, save_flag)

# interocular pref inputs all units
aucr.PlotBox_neuron_pref_input_interocular_corr(save_flag) # Figure 5a, left panel
aucr.plotScatter_neuron_pref_input_interocular_corr_vs_auc_ratio(save_flag) # Figure 5a, right panel
