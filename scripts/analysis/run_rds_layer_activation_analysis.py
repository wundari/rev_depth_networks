# %%
from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
import numpy as np

from config.config import ConfigGCNet

# %% load GCNet config
config = ConfigGCNet()
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
rdsl = RDS_LayerAct(config, params_rds)

# %% compute layer activation for a specific dot density and dot match
dotDens = 0.9
dotMatch_list = [0.0, 0.5, 1.0]
for dotMatch in dotMatch_list:
    rdsl.compute_layer_act_rds(dotMatch, dotDens, rdsl.background_flag)

# compute layer activation for all dot density
rdsl.compute_layer_act_rds_all(rdsl.background_flag)

# %% bootstrap
split_train = 0.8

for dotDens in rdsl.dotDens_list:
    # dotDens = 0.5
    rdsl.xDecode(dotDens, split_train, rdsl.n_bootstrap)

# %% plot
save_flag = 1

# all dot density in one image
rdsl.plotLine_xDecode_across_layers(save_flag)

# each dot density
for dotDens in rdsl.dotDens_list:
    rdsl.plotLine_xDecode_across_layers_at_dotDens(dotDens, save_flag)

rdsl.plotHeat_xDecode(save_flag)

# %% debug
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utilities.misc import NestedTensor
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS

dotMatch = 0.0
dotDens = 0.9
background_flag = 1
pedestal_flag = 0
rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
    dotMatch,
    dotDens,
    rdsl.disp_ct_pix_list,
    rdsl.n_rds_each_disp,
    background_flag,
    pedestal_flag,
)

rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=rdsl.transform_data)

rds_loader = DataLoader(
    rds_data,
    batch_size=2,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=1,
)
n_samples = 2 * rdsl.n_rds_each_disp

inputs_left, inputs_right, disps = next(iter(rds_loader))
# generate disparity direction
ref = disps / 10.0
print(ref)

# %%
# build nested tensor
if ref.mean() > 0:
    input_data = NestedTensor(
        left=inputs_left.pin_memory().to(rdsl.config.device, non_blocking=True),
        right=inputs_right.pin_memory().to(rdsl.config.device, non_blocking=True),
        ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
    )
else:
    input_data = NestedTensor(
        left=inputs_right.pin_memory().to(rdsl.config.device, non_blocking=True),
        right=inputs_left.pin_memory().to(rdsl.config.device, non_blocking=True),
        ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
    )

# compute activation
module_outputs = rdsl.compute_layer_activations(input_data, rdsl.target_list)
model_out = rdsl.model(input_data)

# get layer activation and average across feature channels
layer = rdsl.target_list[0]
out = module_outputs[layer].mean(dim=1)  # [batch, disp_channel, h, w]

# compute the prob (normalized across disp_channel)
out = F.softmax(-out, dim=1)  # [batch, disp_channel, h, w]

# create disparity multiplier tensor
# h_layer = out.shape[-2]
# w_layer = out.shape[-1]
# n_disp_channel = out.shape[-3]
# disp_mul = rdsl.create_disp_indices(
#     h_layer, w_layer, n_disp_channel
# )  # [disp_channel, h, w]
# compute the expected activation across disp channels
# layer_act = torch.sum(out.mul(disp_mul), dim=1)  # [batch, h, w]
layer_act = torch.sum(out * rdsl.model.disp_indices, dim=1)
# layer_act = torch.sum(
#     out * rdsl.model.disp_indices * input_data.ref.view(-1, 1, 1, 1), dim=1
# )

# %% visualize the disparity map
import matplotlib.pyplot as plt

i = 0
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(model_out[i].cpu().detach().numpy())
ax[0, 1].imshow(layer_act[i].cpu().detach().numpy())
ax[1, 0].plot(np.arange(512), model_out[i].mean(dim=0).cpu().detach().numpy())
ax[1, 1].plot(np.arange(512), layer_act[i].mean(dim=0).cpu().detach().numpy())

# %%
import matplotlib.pyplot as plt

i = 1
fig, ax = plt.subplots(1, 2)
ax[0].imshow(model_out[i].cpu().detach().numpy())
ax[1].plot(np.arange(512), model_out[i].mean(dim=0).cpu().detach().numpy())

# %% debug cross-decoding
import torch
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import svm
import numpy as np

n_bootstrap = 1000
split_train = 0.8
n_samples = (
    2 * rdsl.n_rds_each_disp
)  # number of rds in total, the "2" comes from near and far disp
n_train = int(split_train * n_samples)  # number of training dataset

score_ards = np.empty(
    (n_bootstrap, len(rdsl.layer_name)),
    dtype=np.float32,
)
score_hmrds = np.empty(
    (n_bootstrap, len(rdsl.layer_name)),
    dtype=np.float32,
)
score_crds = np.empty(
    (n_bootstrap, len(rdsl.layer_name)),
    dtype=np.float32,
)

# load layer activation data and target disparity label
dotDens = 0.9
# ards
dotMatch = 0.0
layer_act_ards = torch.load(
    f"{rdsl.layer_act_dir}/act_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)
targetDisp_ards = torch.load(
    f"{rdsl.layer_act_dir}/targetDisp_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)

# hmrds
dotMatch = 0.5
layer_act_hmrds = torch.load(
    f"{rdsl.layer_act_dir}/act_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)
targetDisp_hmrds = torch.load(
    f"{rdsl.layer_act_dir}/targetDisp_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)

# crds
dotMatch = 1.0
layer_act_crds = torch.load(
    f"{rdsl.layer_act_dir}/act_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)
targetDisp_crds = torch.load(
    f"{rdsl.layer_act_dir}/targetDisp_rds"
    + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
)

for i, layer in enumerate(rdsl.layer_name):

    i = 6
    layer = rdsl.layer_name[i]

    # compute mean and std for normalization across batch with crds
    x = layer_act_crds[layer]  # [batch, h, w]
    # average across row
    x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
    # get mean & std across batch
    x_mean = x.mean(axis=0, keepdim=True)  # [1, w]
    x_std = x.std(axis=0, keepdim=True)  # [1, w]

    # define classifier
    clf = svm.SVC(kernel="linear", cache_size=1000)
    for i_bootstrap in range(n_bootstrap):
        print(
            f"Cross-decoding for rds dotDens {dotDens:.2f}, "
            + f"{layer}, bootstrap: {i_bootstrap}/{n_bootstrap}"
        )

        # generate random numbers for splitting train and test dataset
        idx = np.random.choice(n_samples, size=n_samples, replace=False)

        # build training dataset with crds
        x = layer_act_crds[layer][idx[0:n_train]]  # [n_train, h, w]
        # average across row
        x = x.mean(axis=-2)  # [n_train, h, w] => [n_train, w]
        # standardize
        x_train = (x - x_mean) / x_std
        y_train = targetDisp_crds[idx[0:n_train]]

        # train classifier
        clf.fit(x_train.view(n_train, -1).numpy(), y_train)

        # ards
        # prepare ards test dataset
        x = layer_act_ards[layer]
        # average across row
        x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
        # standardize
        x_test = (x - x_mean) / x_std
        y_test = targetDisp_ards
        # fit
        score_ards[i_bootstrap, i] = clf.score(
            x_test.view(n_samples, -1).numpy(), y_test
        )

        # hmrds
        x = layer_act_hmrds[layer]
        # average across row
        x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
        # standardize
        x_test = (x - x_mean) / x_std
        y_test = targetDisp_hmrds
        # fit
        score_hmrds[i_bootstrap, i] = clf.score(
            x_test.view(n_samples, -1).numpy(), y_test
        )

        # crds
        x = layer_act_crds[layer][idx[n_train:]]
        # average across row
        x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
        # standardize
        x_test = (x - x_mean) / x_std
        y_test = targetDisp_crds[idx[n_train:]]
        # fit
        score_crds[i_bootstrap, i] = clf.score(
            x_test.view((n_samples - n_train), -1).numpy(), y_test
        )
