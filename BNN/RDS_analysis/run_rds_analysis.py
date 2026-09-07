# %%

from config.config import BNNconfig

from RDS_analysis.rds_analysis import RDSAnalysis
import numpy as np

# %%
config = BNNconfig()

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
}

rdsa = RDSAnalysis(config, params_rds)
rdsa.model.eval()

# %% compute model responses to RDSs
rdsa.compute_disp_map_rds_group(
    rdsa.dotDens_list, rdsa.background_flag, rdsa.pedestal_flag
)
# %% cross-decoding analysis with SVM
n_bootstrap = 1000
rdsa.xDecode(rdsa.dotDens_list, n_bootstrap, rdsa.background_flag)

# %% plot cross-decoding performance
# plot performance at a target dot density
save_flag = 1
dotDens = 0.2
rdsa.plotLine_xDecode_at_dotDens(dotDens, save_flag)

# plot performance as a function of dot density
rdsa.plotLine_xDecode(save_flag)
# %% plot disparity map
rdsa.plotHeat_dispMap(save_flag)
rdsa.plotHeat_dispMap_avg(save_flag)

# %% debug
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
import matplotlib.pyplot as plt
import random

from config.config import GCNetconfig

from RDS_analysis.rds_analysis import RDSAnalysis
import numpy as np

config = GCNetconfig()

# rds parameters
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 64,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 2,
}

rdsa = RDSAnalysis(config, params_rds)
rdsa.model.eval()

seed_number = 35154


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # print out seed number for each worker
    # np_seed = np.random.get_state()[1][0]
    # py_seed = random.getstate()[1][0]

    # print(f"{worker_id} seed pytorch: {worker_seed}\n")
    # print(f"{worker_id} seed numpy: {np_seed}\n")
    # print(f"{worker_id} seed python: {py_seed}\n")


g = torch.Generator()
g.manual_seed(seed_number)

dotMatch = 1.0
dotDens = 0.25
background_flag = 1
pedestal_flag = 0
rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
    dotMatch,
    dotDens,
    rdsa.disp_ct_pix_list,
    rdsa.n_rds_each_disp,
    background_flag,
    pedestal_flag,
)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_data = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t + 1.0) / 2.0),
        transforms.Normalize(mean, std),
    ]
)

rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
rds_loader = DataLoader(
    rds_data,
    batch_size=rdsa.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# %%
from tqdm import tqdm
from utilities.misc import NestedTensor

tepoch = tqdm(rds_loader)
for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
    # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

    # generate disparity direction
    ref = disps / 10.0

    print(ref)

(inputs_left, inputs_right, disps) = next(iter(rds_loader))
print(disps)

# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].imshow(rds_left[0], cmap="gray", vmin=-1, vmax=1)
# axes[1].imshow(rds_right[0], cmap="gray", vmin=-1, vmax=1)


# %%
# generate disparity direction
ref = disps / 10.0

# build nested tensor
if ref.mean() > 0:
    input_data = NestedTensor(
        left=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
        right=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
        ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
    )
else:
    input_data = NestedTensor(
        left=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
        right=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
        ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
    )
feat_left, feat_right = rdsa.model.encoder(input_data)
logits = rdsa.model.decoder(feat_left, feat_right)

# %% visualize logit
x = np.arange(-rdsa.config.max_disp // 2, rdsa.config.max_disp // 2, 1)
i = 1
# plt.hist(logits[i, :, 128, 256].detach().cpu().numpy())
plt.plot(x, logits[i, :, 128, 256].detach().cpu().numpy())
plt.xlabel("Disparity (pixel)")
plt.ylabel("Prob.")
plt.title("Decoder output")
plt.savefig("logits_crds_disp_pos.png", dpi=600)

# logits_sum = logits.sum(dim=1).detach().cpu().numpy()
# plt.imshow(logits_sum[i])
# plt.plot(logits_sum[i, 128, :])


# %%
disp_indices = (
    torch.arange(rdsa.config.max_disp // 2, -rdsa.config.max_disp // 2, -1)
    .view(1, -1, 1, 1)
    .to(device=rdsa.config.device)
)
disp_pred = torch.sum(logits * rdsa.model.disp_indices, dim=1)
disp_pred2 = torch.sum(logits * disp_indices, dim=1)
# print(disp_pred[i, 64:192, 128:384].mean())

disp_pred_mean = disp_pred[i].mean(dim=0).detach().cpu().numpy()
disp_pred_mean2 = disp_pred2[i].mean(dim=0).detach().cpu().numpy()

plt.plot(disp_pred_mean)
plt.xlabel("X (Pixel)")
plt.ylabel("Disparity (pixel)")
plt.title("Mean disparity - cRDS")
plt.savefig("cRDS_disp_neg_mean.png", dpi=600)

plt.plot(disp_pred_mean2)
print(disp_pred_mean[128:384].mean())
print(disp_pred_mean2[128:384].mean())

# %%
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0]
plt.imshow(disp_pred[i].detach().cpu().numpy(), cmap="jet", vmin=-10, vmax=10)
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("crds_disp_neg.png")

plt.imshow(disp_pred2[i].detach().cpu().numpy())
plt.plot(disp_pred[i, 128, :].detach().cpu().numpy())
plt.plot(disp_pred2[i, 128, :].detach().cpu().numpy())

# %%log
# dotDens = 0.25
# disp = -10 -> crds_mean = 3.3, hmrds_mean = 1.1, ards_mean = -8.3
# disp = 10 -> crds_mean = -6.23, hmrds_mean = -6.5, ards_mean = -9.5

# dotDens = 0.9
# disp = -10 -> crds_mean = 3.8, hmrds_mean = -12.4, ards_mean = -25.8
# disp = 10 -> crds_mean = -6, hmrds_mean = -15.3, ards_mean = -27.9
