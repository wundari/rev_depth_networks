#%%
# Script for plotting Figure 5b
# working dir: Codes/Python/gcnet
# %% load necessary modules

from Network_dissection.network_dissection import NetworkDissection
from utils.utils import *

import numpy as np
import torch

from Visualization.network_visualization import NetworkVisualization

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


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
c_disp_shift = 1.5
flip_input = 0  # set to 1 if flip the input (right2left), use right disparity image as ground truth
batch_size = 2
params_train = {
    "sceneflow_type": sceneflow_type,
    "c_disp_shift": c_disp_shift,
    "epoch_to_load": epoch_to_load,
    "iter_to_load": iter_to_load,
    "flip_input": flip_input,
    "target_disp": target_disp,
    "batch_size": batch_size,
}
nd = NetworkDissection(params_network, params_train)
nv = NetworkVisualization(params_train)

# %% optimize single neuron (an example)
feature_channel_index = 4
disp_channel_index = 40  # disparity channel index
neuron_row_index = 64  # neuron coordinate
neuron_col_index = 128  # neuron coordinate
# target_layer = list(nd.model.layer19[0]) + list(nd.model.layer20[0])
target_layer = nd.model.layer20[0]
n_iters = 128
img_left_optimized, img_right_optimized, loss_history = nd.optimize_neuron_activation(
    target_layer,
    feature_channel_index,
    disp_channel_index,
    neuron_row_index,
    neuron_col_index,
    n_iters,
)

# visualize
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))
img_left = img_left_optimized.cpu().numpy() * 255
axes[0, 0].imshow(img_left.astype(np.uint8))
axes[0, 0].set_title("Left image")
axes[0, 0].axis("off")

img_right = img_right_optimized.cpu().numpy() * 255
axes[0, 1].imshow(img_right.astype(np.uint8))
axes[0, 1].set_title("Right image")
axes[0, 1].axis("off")

axes[1, 0].plot(loss_history.cpu().numpy())
axes[1, 0].set_title("Loss")


# %% activation maximization
for i in range(1, len(nd.target_list)):
    target_layer_name = nd.target_name[i]
    target_layer = nd.target_list[i]
    print(target_layer_name)
    (
        img_left_layer,
        img_right_layer,
        loss_history,
    ) = nd.optimize_all_neurons_in_single_layer(target_layer)

    # save
    # np.save(f"{nd.neuron_dir}/img_left_{target_layer_name}_conv.npy", img_left_layer)
    # np.save(f"{nd.neuron_dir}/img_right_{target_layer_name}_conv.npy", img_right_layer)
    # np.save(f"{nd.neuron_dir}/loss_history_{target_layer_name}_conv.npy", loss_history)

# %% visualize neuron activation
target_layer_name = "layer19"
img_left_layer = np.load(f"{nd.neuron_dir}/img_left_{target_layer_name}_conv.npy")
img_right_layer = np.load(f"{nd.neuron_dir}/img_right_{target_layer_name}_conv.npy")

rgb_flag = True
save_flag = False
n_img_row = 1
nv.visualize_neuron_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag, save_flag
)

rgb_flag = False
nv.visualize_neuron_activation_in_single_disp_channel(
    img_left_layer, img_right_layer, target_layer_name, n_img_row, rgb_flag, save_flag
)

# %% visualize neuron activation max, near-zero, and min interocular correlation (Fig. 5b)
save_flag = 0
nv.visualize_neuron_activation_mzm_interocular_corr(save_flag)
# %%
