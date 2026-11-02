# %% load necessary modules
from Network_dissection.network_dissection import NetworkDissection

import numpy as np
import os

from Common.network_correlation import *

import matplotlib.pyplot as plt

# %% compute correlation between left and right preferred inputs
sceneflow_type = "monkaa"  # dataset on which GC-Net was trained
epoch_to_load = 7
iter_to_load = 22201
disp_mag = 10
c_disp_shift = 1.5
nd = NetworkDissection(
    sceneflow_type, epoch_to_load, iter_to_load, disp_mag, c_disp_shift
)


# %% channel correlation
channel_dir = nd.network_diss_dir + "/channel_optimization"
if not os.path.exists(channel_dir):
    os.mkdir(channel_dir)

target_name = [
    "layer19",
    "layer20",
    "layer21",
    "layer22",
    "layer23",
    "layer24",
    "layer25",
    "layer26",
    "layer27",
    "layer28",
    "layer29",
    "layer30",
    "layer31",
    "layer32",
    "layer33a",
    "layer34a",
    "layer35a",
    "layer36a",
]


mean = 2.5
size = 25
filt = gauss2d(mean, size)

# %%
t = 0
for t in range(len(target_name)):
    target_layer_name = target_name[t]
    print(f"processing {target_layer_name}")
    img_left_layer = np.load(f"{channel_dir}/img_left_channel_{target_layer_name}.npy")
    img_right_layer = np.load(
        f"{channel_dir}/img_right_channel_{target_layer_name}.npy"
    )

    # convert to grayscale
    # img_left_layer = img_left_layer.mean(axis=-1)
    # img_right_layer = img_right_layer.mean(axis=-1)

    # get a color channel
    img_left_layer = img_left_layer[:, :, :, :, 2]
    img_right_layer = img_right_layer[:, :, :, :, 2]

    corr_channel = compute_corr_channel_pref_input(
        img_left_layer, img_right_layer, filt
    )

    # save
    np.save(
        f"{channel_dir}/channel_corr_{target_layer_name}_blue_channel.npy", corr_channel
    )


# %%
corr_channel_center_disp = np.zeros(len(target_name), dtype=np.float32)
var = np.zeros(len(target_name), dtype=np.float32)

#
for t in range(len(target_name)):
    target_layer_name = target_name[t]
    temp = corr_channel_all[target_layer_name]
    n_disp_channel = temp.shape[1]

    signal = temp[:, n_disp_channel // 2]
    corr_channel_center_disp[t] = np.median(signal)
    var[t] = np.std(signal)

plt.errorbar(x=np.arange(len(target_name)), y=corr_channel_center_disp, yerr=var)


# %%
t = 18
target_layer_name = target_name[t]
corr_channel = np.load(f"{channel_dir}/channel_corr_{target_layer_name}.npy")

fig, axes = plt.subplots(1, 2)
axes[0].plot(corr_channel.mean(axis=0))
axes[1].plot(corr_channel.mean(axis=1))
# %%


# %% neuron correlation

neuron_dir = nd.network_diss_dir + "/neuron_optimization"
if not os.path.exists(neuron_dir):
    os.mkdir(neuron_dir)

# %%
for t in range(len(target_name)):
    target_layer_name = target_name[t]
    print(f"processing {target_layer_name}")
    img_left_layer = np.load(f"{neuron_dir}/img_left_{target_layer_name}.npy")
    img_right_layer = np.load(f"{neuron_dir}/img_right_{target_layer_name}.npy")

    # convert to grayscale
    img_left_layer = img_left_layer.mean(axis=-1)
    img_right_layer = img_right_layer.mean(axis=-1)

    corr_neuron = compute_corr_channel_pref_input(img_left_layer, img_right_layer, filt)

    # save
    np.save(f"{neuron_dir}/neuron_corr_{target_layer_name}.npy", corr_neuron)

# %%
corr_neuron_mean = np.zeros(len(target_name), dtype=np.float32)
for t in range(len(target_name)):
    target_layer_name = target_name[t]
    corr_neuron = np.load(f"{neuron_dir}/neuron_corr_{target_layer_name}.npy")

    corr_neuron_mean[t] = np.median(corr_neuron)
# %%
plt.plot(corr_neuron_mean)

# %%
