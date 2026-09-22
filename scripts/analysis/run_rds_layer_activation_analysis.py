# %%
def main():
    from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
    from config.config_gcnet import ConfigGCNet

    # %% load GCNet config
    config = ConfigGCNet()
    config.load_state = True
    config.compile_mode = None
    config.binocular_interaction = "bem"
    config.seed = config.experiment_id = 1618
    config.epoch_to_load = 9
    config.iter_to_load = 17100
    config.model_pretrained = f"epoch_{config.epoch_to_load}_iter_{config.iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar

    config.n_rds_each_disp = 128
    config.batch_size_rds = 8
    config.n_bootstrap = 100
    rdsl = RDS_LayerAct(config)

    # compute layer activation for all dot density
    # rdsl.compute_layer_act_rds_all(rdsl.background_flag)

    # # Cross-decoding
    # split_train = 0.8
    # for dotDens in rdsl.dotDens_list:
    #     # dotDens = 0.5
    #     rdsl.xDecode_layer_activation(dotDens, split_train, rdsl.n_bootstrap)

    # Cosine-similarity
    split_train = 0.8
    for dotDens in rdsl.dotDens_list:
        rdsl.compute_cosine_similarity(dotDens, split_train, rdsl.n_bootstrap)

    # plot
    save_flag = 1

    # all dot density in one image
    rdsl.plotLine_xDecode_across_layers(save_flag)

    # each dot density
    for dotDens in rdsl.dotDens_list:
        rdsl.plotLine_xDecode_across_layers_at_dotDens(dotDens, save_flag)

    rdsl.plotHeat_xDecode(save_flag)


# %%
if __name__ == "__main__":
    main()

# %%
# import numpy as np
# from pathlib import Path
# from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
# from config.config_gcnet import ConfigGCNet

# from utilities.misc import NestedTensor

# # load GCNet config
# config = ConfigGCNet()
# config.load_state = True
# config.compile_mode = None
# config.binocular_interaction = "bem"
# config.seed = config.experiment_id = 1618
# config.epoch_to_load = 9
# config.iter_to_load = 17100
# config.model_pretrained = f"epoch_{config.epoch_to_load}_iter_{config.iter_to_load}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
# config.n_rds_each_disp = 128
# config.batch_size_rds = 8
# config.n_bootstrap = 100

# rdsl = RDS_LayerAct(config)

# # %% create dataloader for RDS
# dotMatch = 0.0
# dotDens = 0.5
# rds_loader = rdsl._generate_rds_loader(
#     dotMatch, dotDens, rdsl.background_flag, rdsl.pedestal_flag
# )

# inputs_left, inputs_right, disps = next(iter(rds_loader))
# i = 0
# id_start = i * rdsl.batch_size_rds
# id_end = id_start + rdsl.batch_size_rds

# # generate disparity direction
# ref = disps / 10.0

# # build nested tensor
# # input_data = NestedTensor(
# #     left=inputs_left.to(self.config.device, non_blocking=True),
# #     right=inputs_right.to(self.config.device, non_blocking=True),
# #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
# # )

# # swap left and right inputs if ref < 0
# if ref.mean() > 0:
#     input_data = NestedTensor(
#         left=inputs_left.to(rdsl.config.device, non_blocking=True),
#         right=inputs_right.to(rdsl.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
#     )
# else:
#     input_data = NestedTensor(
#         left=inputs_right.to(rdsl.config.device, non_blocking=True),
#         right=inputs_left.to(rdsl.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
#     )

# # debug compute_layer_activations
# # compute activation for whole layers
# targets = (
#     list(rdsl.target_list)
#     if isinstance(rdsl.target_list, (list, tuple))
#     else [rdsl.target_list]
# )
# outputs = {module: [] for module in targets}
# handles = []
# modes = {module: module.training for module in rdsl.model.modules()}


# captured = rdsl.compute_layer_activations(
#     input_data, rdsl.target_list, pooled=True, include_prediction=True
# )

# # fetch each layer activation
# features = {}
# n_samples = 2 * rdsl.n_rds_each_disp
# for name, module in zip(rdsl.layer_name, rdsl.target_list, strict=True):
#     # module = rdsl.target_list[0]
#     act = captured[module].numpy()

#     if name not in features:
#         features[name] = np.empty((n_samples, act.shape[1]), dtype=np.float32)

#     features[name][id_start:id_end] = act

# ###
# # cosine similarity
# ###
# from sklearn.metrics.pairwise import cosine_similarity

# # load layer activation
# dotDens = 0.5
# data, labels = {}, {}
# for condition, dotMatch in (("ards", 0.0), ("hmrds", 0.5), ("crds", 1.0)):
#     suffix = f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
#     data[condition] = np.load(
#         Path(rdsl.layer_act_dir) / ("act_rds" + suffix),
#         allow_pickle=True,
#     ).item()
#     if data[condition].get("_metadata") != rdsl._metadata():
#         raise ValueError(
#             "Legacy or incompatible activation files; regenerate all conditions"
#         )
#     labels[condition] = np.load(Path(rdsl.layer_act_dir) / ("targetDisp_rds" + suffix))


# def disparity_axis(features, labels):
#     near = features[labels > 0]  # .mean(axis=0)
#     far = features[labels < 0]  # .mean(axis=0)
#     return near - far


# layer_name = "layer19"
# feat_crds = data["crds"][layer_name]
# feat_hmrds = data["hmrds"][layer_name]
# feat_ards = data["ards"][layer_name]
# label_crds = labels["crds"]
# label_hmrds = labels["hmrds"]
# label_ards = labels["ards"]

# delta_crds = disparity_axis(feat_crds, label_crds)
# delta_ards = disparity_axis(feat_ards, label_ards)
# delta_hmrds = disparity_axis(feat_hmrds, label_hmrds)

# cosine_dist = cosine_similarity(delta_crds, delta_ards)
# plt.imshow(cosine_dist)
# plt.hist(cosine_dist.flatten())


# denominator = (
#     np.linalg.norm(delta_crds, axis=1, keepdims=True)
#     * np.linalg.norm(delta_ards, axis=1, keepdims=True).T
# )
# alignment = np.dot(delta_crds, delta_ards.T) / denominator

# from sklearn.model_selection import GroupShuffleSplit

# n_bootstrap = 1000
# split_train = 0.8
# split_seed = 3407
# y = labels["crds"]
# groups = rdsl._split_groups(len(y))
# splits = list(
#     GroupShuffleSplit(
#         n_splits=n_bootstrap,
#         train_size=split_train,
#         random_state=split_seed,
#     ).split(np.zeros(len(y)), y, groups)
# )

# # %% debug
# import torch
# from torch.nn import functional as F
# from utilities.misc import NestedTensor

# dotMatch = 0.0
# dotDens = 0.1
# background_flag = 1
# pedestal_flag = 0
# rds_loader = rdsl._generate_rds_loader(
#     dotMatch,
#     dotDens,
#     background_flag,
#     pedestal_flag,
# )

# n_samples = 2 * rdsl.n_rds_each_disp

# inputs_left, inputs_right, disps = next(iter(rds_loader))
# # generate disparity direction
# ref = disps / 10.0
# print(ref)

# # %%
# # build nested tensor
# if ref.mean() > 0:
#     input_data = NestedTensor(
#         left=inputs_left.pin_memory().to(rdsl.config.device, non_blocking=True),
#         right=inputs_right.pin_memory().to(rdsl.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
#     )
# else:
#     input_data = NestedTensor(
#         left=inputs_right.pin_memory().to(rdsl.config.device, non_blocking=True),
#         right=inputs_left.pin_memory().to(rdsl.config.device, non_blocking=True),
#         ref=ref.pin_memory().to(rdsl.config.device, non_blocking=True),
#     )

# # compute activation
# module_outputs = rdsl.compute_layer_activations(input_data, rdsl.target_list)
# model_out = rdsl.model(input_data)

# # get layer activation and average across feature channels
# layer = rdsl.target_list[0]
# out = module_outputs[layer].mean(dim=1)  # [batch, disp_channel, h, w]

# # compute the prob (normalized across disp_channel)
# out = F.softmax(-out, dim=1)  # [batch, disp_channel, h, w]

# # create disparity multiplier tensor
# # h_layer = out.shape[-2]
# # w_layer = out.shape[-1]
# # n_disp_channel = out.shape[-3]
# # disp_mul = rdsl.create_disp_indices(
# #     h_layer, w_layer, n_disp_channel
# # )  # [disp_channel, h, w]
# # compute the expected activation across disp channels
# # layer_act = torch.sum(out.mul(disp_mul), dim=1)  # [batch, h, w]
# layer_act = torch.sum(out * rdsl.model.disp_indices, dim=1)
# # layer_act = torch.sum(
# #     out * rdsl.model.disp_indices * input_data.ref.view(-1, 1, 1, 1), dim=1
# # )

# # %% visualize the disparity map
# import matplotlib.pyplot as plt

# i = 0
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(model_out[i].cpu().detach().numpy())
# ax[0, 1].imshow(layer_act[i].cpu().detach().numpy())
# ax[1, 0].plot(np.arange(512), model_out[i].mean(dim=0).cpu().detach().numpy())
# ax[1, 1].plot(np.arange(512), layer_act[i].mean(dim=0).cpu().detach().numpy())

# # %%
# import matplotlib.pyplot as plt

# i = 1
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(model_out[i].cpu().detach().numpy())
# ax[1].plot(np.arange(512), model_out[i].mean(dim=0).cpu().detach().numpy())

# # %% debug cross-decoding
# import torch
# from sklearnex import patch_sklearn

# patch_sklearn()
# from sklearn import svm
# import numpy as np

# n_bootstrap = 1000
# split_train = 0.8
# n_samples = (
#     2 * rdsl.n_rds_each_disp
# )  # number of rds in total, the "2" comes from near and far disp
# n_train = int(split_train * n_samples)  # number of training dataset

# score_ards = np.empty(
#     (n_bootstrap, len(rdsl.layer_name)),
#     dtype=np.float32,
# )
# score_hmrds = np.empty(
#     (n_bootstrap, len(rdsl.layer_name)),
#     dtype=np.float32,
# )
# score_crds = np.empty(
#     (n_bootstrap, len(rdsl.layer_name)),
#     dtype=np.float32,
# )

# # load layer activation data and target disparity label
# dotDens = 0.9
# # ards
# dotMatch = 0.0
# layer_act_ards = torch.load(
#     f"{rdsl.layer_act_dir}/act_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )
# targetDisp_ards = torch.load(
#     f"{rdsl.layer_act_dir}/targetDisp_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )

# # hmrds
# dotMatch = 0.5
# layer_act_hmrds = torch.load(
#     f"{rdsl.layer_act_dir}/act_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )
# targetDisp_hmrds = torch.load(
#     f"{rdsl.layer_act_dir}/targetDisp_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )

# # crds
# dotMatch = 1.0
# layer_act_crds = torch.load(
#     f"{rdsl.layer_act_dir}/act_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )
# targetDisp_crds = torch.load(
#     f"{rdsl.layer_act_dir}/targetDisp_rds"
#     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.pt"
# )

# for i, layer in enumerate(rdsl.layer_name):

#     i = 6
#     layer = rdsl.layer_name[i]

#     # compute mean and std for normalization across batch with crds
#     x = layer_act_crds[layer]  # [batch, h, w]
#     # average across row
#     x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
#     # get mean & std across batch
#     x_mean = x.mean(axis=0, keepdim=True)  # [1, w]
#     x_std = x.std(axis=0, keepdim=True)  # [1, w]

#     # define classifier
#     clf = svm.SVC(kernel="linear", cache_size=1000)
#     for i_bootstrap in range(n_bootstrap):
#         print(
#             f"Cross-decoding for rds dotDens {dotDens:.2f}, "
#             + f"{layer}, bootstrap: {i_bootstrap}/{n_bootstrap}"
#         )

#         # generate random numbers for splitting train and test dataset
#         idx = np.random.choice(n_samples, size=n_samples, replace=False)

#         # build training dataset with crds
#         x = layer_act_crds[layer][idx[0:n_train]]  # [n_train, h, w]
#         # average across row
#         x = x.mean(axis=-2)  # [n_train, h, w] => [n_train, w]
#         # standardize
#         x_train = (x - x_mean) / x_std
#         y_train = targetDisp_crds[idx[0:n_train]]

#         # train classifier
#         clf.fit(x_train.view(n_train, -1).numpy(), y_train)

#         # ards
#         # prepare ards test dataset
#         x = layer_act_ards[layer]
#         # average across row
#         x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
#         # standardize
#         x_test = (x - x_mean) / x_std
#         y_test = targetDisp_ards
#         # fit
#         score_ards[i_bootstrap, i] = clf.score(
#             x_test.view(n_samples, -1).numpy(), y_test
#         )

#         # hmrds
#         x = layer_act_hmrds[layer]
#         # average across row
#         x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
#         # standardize
#         x_test = (x - x_mean) / x_std
#         y_test = targetDisp_hmrds
#         # fit
#         score_hmrds[i_bootstrap, i] = clf.score(
#             x_test.view(n_samples, -1).numpy(), y_test
#         )

#         # crds
#         x = layer_act_crds[layer][idx[n_train:]]
#         # average across row
#         x = x.mean(axis=-2)  # [batch, h, w] => [batch, w]
#         # standardize
#         x_test = (x - x_mean) / x_std
#         y_test = targetDisp_crds[idx[n_train:]]
#         # fit
#         score_crds[i_bootstrap, i] = clf.score(
#             x_test.view((n_samples - n_train), -1).numpy(), y_test
#         )

# %%
