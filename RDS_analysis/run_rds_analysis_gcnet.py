# %%
# script for in-silico analysis using RDS for GCNet
# working dir: rev_depth_networks


# %%
def main():
    from config.config_gcnet import ConfigGCNet
    from RDS_analysis.rds_analysis import RDSAnalysis

    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    rdsa = RDSAnalysis(config)

    # compute model responses to RDSs
    rdsa.compute_disp_map_rds_group(
        rdsa.dotDens_list, rdsa.background_flag, rdsa.pedestal_flag
    )
    # cross-decoding analysis with SVM
    rdsa.xDecode(rdsa.dotDens_list, rdsa.n_bootstrap, rdsa.background_flag)

    # plot cross-decoding performance
    # plot performance at a target dot density
    save_flag = 1
    dotDens = 0.3
    rdsa.plotLine_xDecode_at_dotDens(dotDens, save_flag)

    # plot performance as a function of dot density
    rdsa.plotLine_xDecode(save_flag)

    # plot disparity map
    rdsa.plotHeat_dispMap(save_flag)
    rdsa.plotHeat_dispMap_avg(save_flag)


if __name__ == "__main__":
    main()

# %% average across seed
# dataset_name = "sceneflow_monkaa"
# rdsa.plotLine_xDecode_avg_seed(dataset_name, save_flag)

# # %% debug
# import torch
# from torch.utils.data import DataLoader
# from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
# import matplotlib.pyplot as plt
# import random
# from config.config_gcnet import ConfigGCNet
# from utilities.misc import NestedTensor

# from RDS_analysis.rds_analysis import RDSAnalysis
# import numpy as np

# config = ConfigGCNet()

# # rds parameters
# params_rds = {
#     "target_disp": 10,  # RDS target disparity (pix) to be analyzed
#     "n_rds_each_disp": 64,  # n_rds for each disparity magnitude in disp_ct_pix
#     "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
#     "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
#     "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
#     "background_flag": 1,  # 1: with cRDS background
#     "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
#     "batch_size_rds": 2,
#     "n_bootstrap": 1000,
# }

# rdsa = RDSAnalysis(config, params_rds)
# rdsa.model.eval()

# seed_number = config.seed


# # initialize random seed number for dataloader
# def seed_worker(worker_id):
#     worker_seed = seed_number  # torch.initial_seed()  % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

#     # print out seed number for each worker
#     # np_seed = np.random.get_state()[1][0]
#     # py_seed = random.getstate()[1][0]

#     # print(f"{worker_id} seed pytorch: {worker_seed}\n")
#     # print(f"{worker_id} seed numpy: {np_seed}\n")
#     # print(f"{worker_id} seed python: {py_seed}\n")


# g = torch.Generator()
# g.manual_seed(seed_number)

# dotMatch = 1.0
# dotDens = 0.25
# background_flag = 1
# pedestal_flag = 0
# rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
#     dotMatch,
#     dotDens,
#     rdsa.disp_ct_pix_list,
#     rdsa.n_rds_each_disp,
#     background_flag,
#     pedestal_flag,
# )


# class NormalizeRDS:
#     """Normalize signed [-1,1] RGB arrays; no uint8 ToTensor ambiguity/lambda."""

#     # def __init__(self):

#     # mean = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
#     # std = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
#     # mean = (0.485, 0.456, 0.406)
#     # std = (0.229, 0.224, 0.225)
#     # mean = np.array([0.5, 0.5, 0.5])
#     # std = np.array([0.5, 0.5, 0.5])

#     _mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
#     _std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

#     def __call__(self, image):
#         x = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32).permute(
#             2, 0, 1
#         )
#         if not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
#             raise ValueError("RDS pixels must be finite in [-1,1]")

#         # mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
#         #         # std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

#         return ((x + 1) / 2 - self._mean) / self._std


# transform_data = NormalizeRDS()

# rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
# rds_loader = DataLoader(
#     rds_data,
#     batch_size=rdsa.batch_size_rds,
#     shuffle=True,
#     pin_memory=True,
#     drop_last=True,
#     num_workers=0,
#     worker_init_fn=seed_worker,
#     generator=g,
# )


# pred_disp = torch.empty(
#     (len(rdsa.disp_ct_pix_list) * rdsa.n_rds_each_disp, rdsa.h_bg, rdsa.w_bg),
#     dtype=torch.float32,
# )
# pred_disp_labels = np.empty(
#     (len(rdsa.disp_ct_pix_list) * rdsa.n_rds_each_disp), dtype=np.int8
# )

# # predict disparity map
# rdsa.model.eval()
# # tepoch = tqdm(rds_loader)
# # for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
# for i in range(len(rds_loader)):
#     inputs_left, inputs_right, disps = next(iter(rds_loader))

#     # generate disparity direction
#     ref = disps / 10.0

#     # build nested tensor
#     # input_data = NestedTensor(
#     #     left=inputs_left.pin_memory().to(self.config.device, non_blocking=True),
#     #     right=inputs_right.pin_memory().to(
#     #         self.config.device, non_blocking=True
#     #     ),
#     #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
#     # )
#     if ref.mean() > 0:
#         input_data = NestedTensor(
#             left=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
#             right=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
#             ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
#         )
#     else:
#         input_data = NestedTensor(
#             left=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
#             right=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
#             ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
#         )

#     # model output
#     with torch.autocast(device_type=rdsa.config.device, dtype=torch.bfloat16):
#         disp_pred = rdsa.model(input_data)  # [batch, h, w]
#         # module_outputs = self.compute_layer_activations(
#         #     input_data, self.target_list
#         # )
#         # layer = self.target_list[-1]
#         # # [batch, feat_channel, disp_channel, h, w] => [batch, disp_channel, h, w]
#         # disp_pred = module_outputs[layer].mean(dim=1)
#         # disp_pred = F.softmax(-disp_pred, dim=1)  # [batch, disp_channel, h, w]
#         # # disp_pred = torch.sum(
#         # #     disp_pred * self.model.disp_indices, dim=1
#         # # )  # [batch, h, w]
#         # disp_pred = torch.sum(
#         #     disp_pred
#         #     * self.model.disp_indices
#         #     * input_data.ref.view(-1, 1, 1, 1),
#         #     dim=1,
#         # )

#     id_start = i * rdsa.batch_size_rds
#     id_end = id_start + rdsa.batch_size_rds
#     pred_disp_labels[id_start:id_end] = disps
#     pred_disp[id_start:id_end] = disp_pred.detach().float().cpu()

#     # tepoch.set_description(
#     #     f"RDS dotMatch: {dotMatch:.2f}, "
#     #     + f"dotDens: {dotDens:.2f}, "
#     #     + f"iter: {i+1}/{len(rds_loader)}"
#     # )

# for dm, dotMatch in enumerate(rdsa.dotMatch_list):
#     pred_disp = torch.empty(
#         (
#             len(rdsa.dotDens_list),
#             len(rdsa.disp_ct_pix_list) * rdsa.n_rds_each_disp,
#             rdsa.h_bg,
#             rdsa.w_bg,
#         ),
#         dtype=torch.float32,
#     )
#     pred_disp_labels = np.empty(
#         (len(rdsa.dotDens_list), len(rdsa.disp_ct_pix_list) * rdsa.n_rds_each_disp),
#         dtype=np.int8,
#     )
#     for dd, dotDens in enumerate(rdsa.dotDens_list):
#         a, b = rdsa.compute_disp_map_rds(
#             dotMatch, dotDens, background_flag, pedestal_flag
#         )

#         pred_disp[dd], pred_disp_labels[dd] = rdsa.compute_disp_map_rds(
#             dotMatch, dotDens, background_flag, pedestal_flag
#         )

#     np.save(
#         f"{self.xDecode_dir}/pred_disp_{self.rds_type[dm]}.npy",
#         pred_disp.cpu().detach().numpy(),
#     )
#     np.save(
#         f"{self.xDecode_dir}/pred_disp_labels_{self.rds_type[dm]}.npy",
#         pred_disp_labels,
#     )

# # %%
# from tqdm import tqdm
# from utilities.misc import NestedTensor

# tepoch = tqdm(rds_loader)
# for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
#     # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

#     # generate disparity direction
#     ref = disps / 10.0

#     print(ref)

# inputs_left, inputs_right, disps = next(iter(rds_loader))
# print(disps)

# # visualize rds
# img_left = (rds_left[0] * 128 + 127).astype(np.int32)
# img_right = (rds_right[0] * 128 + 127).astype(np.int32)

# fig, axes = plt.subplots(nrows=1, ncols=2)
# fig.text(
#     0.5,
#     0.7,
#     f"RDS, dotMatch: {dotMatch}, dotDens: {dotDens}",
#     horizontalalignment="center",
# )
# axes[0].imshow(img_left, cmap="gray", vmin=-1, vmax=1)
# axes[1].imshow(img_right, cmap="gray", vmin=-1, vmax=1)
# axes[0].set_title("Left")
# axes[1].set_title("Right")

# for axes in axes.ravel():
#     axes.set_axis_off()

# plt.savefig(
#     f"rds_sample_images/RDS_dotMatch{dotMatch}.pdf", dpi=600, bbox_inches="tight"
# )


# # %%
# # generate disparity direction
# ref = disps / 10.0

# # build nested tensor
# if ref.mean() > 0:
#     input_data = NestedTensor(
#         left=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
#         right=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
#     )
# else:
#     input_data = NestedTensor(
#         left=inputs_right.pin_memory().to(rdsa.config.device, non_blocking=True),
#         right=inputs_left.pin_memory().to(rdsa.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsa.config.device, non_blocking=True),
#     )
# feat_left, feat_right = rdsa.model.encoder(input_data)
# logits = rdsa.model.decoder(feat_left, feat_right)

# # %% visualize logit
# x = np.arange(-rdsa.config.max_disp // 2, rdsa.config.max_disp // 2, 1)
# i = 1
# # plt.hist(logits[i, :, 128, 256].detach().cpu().numpy())
# plt.plot(x, logits[i, :, 128, 256].detach().cpu().numpy())
# plt.xlabel("Disparity (pixel)")
# plt.ylabel("Prob.")
# plt.title("Decoder output")
# plt.savefig("logits_crds_disp_pos.png", dpi=600)

# # logits_sum = logits.sum(dim=1).detach().cpu().numpy()
# # plt.imshow(logits_sum[i])
# # plt.plot(logits_sum[i, 128, :])


# # %%
# disp_indices = (
#     torch.arange(rdsa.config.max_disp // 2, -rdsa.config.max_disp // 2, -1)
#     .view(1, -1, 1, 1)
#     .to(device=rdsa.config.device)
# )
# disp_pred = torch.sum(logits * rdsa.model.disp_indices, dim=1)
# disp_pred2 = torch.sum(logits * disp_indices, dim=1)
# # print(disp_pred[i, 64:192, 128:384].mean())

# disp_pred_mean = disp_pred[i].mean(dim=0).detach().cpu().numpy()
# disp_pred_mean2 = disp_pred2[i].mean(dim=0).detach().cpu().numpy()

# plt.plot(disp_pred_mean)
# plt.xlabel("X (Pixel)")
# plt.ylabel("Disparity (pixel)")
# plt.title("Mean disparity - cRDS")
# plt.savefig("cRDS_disp_neg_mean.png", dpi=600)

# plt.plot(disp_pred_mean2)
# print(disp_pred_mean[128:384].mean())
# print(disp_pred_mean2[128:384].mean())

# # %%
# fig, axes = plt.subplots(nrows=2, ncols=2)
# axes[0, 0]
# plt.imshow(disp_pred[i].detach().cpu().numpy(), cmap="jet", vmin=-10, vmax=10)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("crds_disp_neg.png")

# plt.imshow(disp_pred2[i].detach().cpu().numpy())
# plt.plot(disp_pred[i, 128, :].detach().cpu().numpy())
# plt.plot(disp_pred2[i, 128, :].detach().cpu().numpy())

# # %%log
# # dotDens = 0.25
# # disp = -10 -> crds_mean = 3.3, hmrds_mean = 1.1, ards_mean = -8.3
# # disp = 10 -> crds_mean = -6.23, hmrds_mean = -6.5, ards_mean = -9.5

# # dotDens = 0.9
# # disp = -10 -> crds_mean = 3.8, hmrds_mean = -12.4, ards_mean = -25.8
# # disp = 10 -> crds_mean = -6, hmrds_mean = -15.3, ards_mean = -27.9
