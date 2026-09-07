# %%

import numpy as np

from RDS.DataHandler_RDS import *
from RDS.RDS_v3 import RDS

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

device = "cuda" if torch.cuda.is_available() else "mps"

SAVE_PATH = "results/sceneflow/layer_activation"
n_monolayers = 10  # number of layers for monocular layers
n_binolayers = 21  # number of layers for binocular layers

monolayer_names = [
    "layer1",
    "res_block_0",
    "res_block_1",
    "res_block_2",
    "res_block_3",
    "res_block_4",
    "res_block_5",
    "res_block_6",
    "res_block_7",
    "layer18",
]
binolayer_names = [
    "cost_vol",
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
    "layer33",
    "layer34",
    "layer35",
    "layer36",
    "layer37",
    "final",
]

layer_names_all = monolayer_names + monolayer_names + binolayer_names
# %% parameters
h = 256
w = 512
maxdisp = 192

# %% generate rds
# rds params
W_CT = w // 2
H_CT = h // 2
rDot = 5  # dot radius in pixel
overlap_flag = 0  # 0: dots are not allowed to overlap
batch_test = 1

disp_ct_pix_list = [-30, 30]
dotDens_list = 0.1 * np.arange(1, 10)

rds_type = ["ards", "hmrds", "crds"]
# ards: 0, crds: 1, hmrds: 0.5, urds: -1
dotMatch_list = [0.0, 0.5, 1.0]


def make_rds(n_rds_each_disp, dotMatch_ct, dotDens, rDot, overlap_flag):
    n_rds = n_rds_each_disp * len(disp_ct_pix_list)

    print(
        "generating rds, dot-match: {}, dot-density: {:.1f}".format(
            dotMatch_ct, dotDens
        )
    )

    # cosntruct rds
    rds = RDS(n_rds_each_disp, w, h, W_CT, H_CT, dotDens, rDot, overlap_flag)

    # [n_rds_each_disp, len(disp_ct_pix), w, h]
    rds_batch_left, rds_batch_right = rds.create_rds_batch(
        disp_ct_pix_list, dotMatch_ct
    )

    # remapping rds into [n_rds, h_bg, w_bg]
    rds_left = np.zeros(
        (n_rds, 3, rds.h_bg, rds.w_bg), dtype=np.float32  # rgb channels
    )
    rds_right = np.zeros(
        (n_rds, 3, rds.h_bg, rds.w_bg), dtype=np.float32  # rgb channels
    )
    rds_label = np.zeros(n_rds, dtype=np.int8)
    count = 0
    for d in range(len(disp_ct_pix_list)):
        for t in range(n_rds_each_disp):
            temp = rds_batch_left[t, d]
            # temp = np.roll(
            #     temp, disp_ct_pix[1] // 2, axis=1
            # )  # shift the whole rds to set near disp at 0 disp
            rds_left[count, 0] = temp
            rds_left[count, 1] = temp
            rds_left[count, 2] = temp

            temp = rds_batch_right[t, d]
            # temp = np.roll(temp, -disp_ct_pix[1] // 2, axis=1)
            rds_right[count, 0] = temp
            rds_right[count, 1] = temp
            rds_right[count, 2] = temp

            rds_label[count] = disp_ct_pix_list[d]

            count += 1

    return (
        rds_left,
        rds_right,
        rds_label,
    )


# %% compute mono layer activation
def compute_monolayer_activation(model, input_left, input_right):
    # batch_size = input_left.shape[0]
    # loss_mul_list = []
    # for d in range(maxdisp):
    #     loss_mul_temp = (
    #         Variable(torch.Tensor(np.ones([batch_size, 1, h, w]) * d))
    #         .pin_memory()
    #         .to(device, non_blocking=True)
    #     )
    #     loss_mul_list.append(loss_mul_temp)
    # loss_mul = torch.cat(loss_mul_list, 1)

    # get model's layers
    model_children = list(model.children())
    childrens_name = [name for name, _ in model.named_children()]

    outputs_left = []
    outputs_right = []
    # outputs_bino = []
    layers_name = []

    # compute layer activations before the left-right inputs concatenate
    out_left = input_left
    out_right = input_right
    for i in range(3):
        layer = model_children[i]

        if i == 1:  # residual blocks
            for j in range(len(layer)):
                sublayer = layer[j]
                layer_name = (
                    childrens_name[i]
                    + "_"
                    + str(sublayer).partition("(")[0]
                    + "_"
                    + str(j)
                )
                layers_name.append(layer_name)

                # print(
                #     "Epoch: {}/{} - activating layer: {}".format(
                #         epoch, n_epochs, layer_name
                #     )
                # )

                out_left = sublayer(out_left)
                outputs_left.append(out_left.data.cpu().numpy())

                out_right = sublayer(out_right)
                outputs_right.append(out_right.data.cpu().numpy())

        else:
            # print(
            #     "Epoch: {}/{} - activating layer: {}".format(
            #         epoch, n_epochs, childrens_name[i]
            #     )
            # )

            out_left = layer(out_left)
            outputs_left.append(out_left.data.cpu().numpy())

            out_right = layer(out_right)
            outputs_right.append(out_right.data.cpu().numpy())

            layers_name.append(childrens_name[i])

    return outputs_left, outputs_right, layers_name


# %% define binolayer activation
def compute_binolayer_activation(model, input_left, input_right):
    # get model's layers
    model_children = list(model.children())
    childrens_name = [name for name, _ in model.named_children()]

    outputs_left = []
    outputs_right = []
    outputs_bino = []
    layers_name = []

    # compute layer activations before the left-right inputs concatenate
    out_left = input_left
    out_right = input_right
    for i in range(3):
        layer = model_children[i]

        if i == 1:  # residual blocks
            for j in range(len(layer)):
                sublayer = layer[j]
                layer_name = (
                    childrens_name[i]
                    + "_"
                    + str(sublayer).partition("(")[0]
                    + "_"
                    + str(j)
                )
                layers_name.append(layer_name)

                # print(
                #     "Epoch: {}/{} - activating layer: {}".format(
                #         epoch, n_epochs, layer_name
                #     )
                # )

                out_left = sublayer(out_left)
                outputs_left.append(out_left.data.cpu().numpy())

                out_right = sublayer(out_right)
                outputs_right.append(out_right.data.cpu().numpy())

        else:
            # print(
            #     "Epoch: {}/{} - activating layer: {}".format(
            #         epoch, n_epochs, childrens_name[i]
            #     )
            # )

            out_left = layer(out_left)
            outputs_left.append(out_left.data.cpu().numpy())

            out_right = layer(out_right)
            outputs_right.append(out_right.data.cpu().numpy())

            layers_name.append(childrens_name[i])

    # cost-volume layers
    # output_dim: [n_batch, n_features_left_and_right, max_disp/2, img_height/2, img_width/2]
    # output_dim: [n_batch, 64, 96, 128, 256]
    # print("Epoch: {}/{} - activating layer: cost-volume".format(epoch, n_epochs))
    cost_vol = model.cost_volume(out_left, out_right)
    outputs_bino.append(cost_vol.data.cpu().numpy())
    layers_name.append("cost_vol")

    # layer 19
    # print("Epoch: {}/{} - activating layer: layer 19".format(epoch, n_epochs))
    # output_dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
    # output_dim: [n_batch, 32, 96, 128, 256]
    out_19 = model.layer19(cost_vol)
    outputs_bino.append(out_19.data.cpu().numpy())
    layers_name.append("layer19")

    # layer 20
    # print("Epoch: {}/{} - activating layer: layer 20".format(epoch, n_epochs))
    # output dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
    # output dim: [n_batch, 32, 96, 128, 256]
    out_20 = model.layer20(out_19)
    outputs_bino.append(out_20.data.cpu().numpy())
    layers_name.append("layer20")

    # # layer 21
    # # print("Epoch: {}/{} - activating layer: layer 21".format(epoch, n_epochs))
    # out_21 = model.layer21(cost_vol)
    # outputs_bino.append(out_21.data.cpu().numpy())
    # layers_name.append("layer21")

    # # layer 22
    # # print("Epoch: {}/{} - activating layer: layer 22".format(epoch, n_epochs))
    # out_22 = model.layer22(out_21)
    # outputs_bino.append(out_22.data.cpu().numpy())
    # layers_name.append("layer22")

    # # layer 23
    # # print("Epoch: {}/{} - activating layer: layer 23".format(epoch, n_epochs))
    # out_23 = model.layer23(out_22)
    # outputs_bino.append(out_23.data.cpu().numpy())
    # layers_name.append("layer23")

    # # layer 24
    # # print("Epoch: {}/{} - activating layer: layer 24".format(epoch, n_epochs))
    # out_24 = model.layer24(out_21)
    # outputs_bino.append(out_24.data.cpu().numpy())
    # layers_name.append("layer24")

    # # layer 25
    # # print("Epoch: {}/{} - activating layer: layer 25".format(epoch, n_epochs))
    # out_25 = model.layer25(out_24)
    # outputs_bino.append(out_25.data.cpu().numpy())
    # layers_name.append("layer25")

    # # layer 26
    # # print("Epoch: {}/{} - activating layer: layer 26".format(epoch, n_epochs))
    # out_26 = model.layer26(out_25)
    # outputs_bino.append(out_26.data.cpu().numpy())
    # layers_name.append("layer26")

    # # layer 27
    # # print("Epoch: {}/{} - activating layer: layer 27".format(epoch, n_epochs))
    # out_27 = model.layer27(out_24)
    # outputs_bino.append(out_27.data.cpu().numpy())
    # layers_name.append("layer27")

    # # layer 28
    # # print("Epoch: {}/{} - activating layer: layer 28".format(epoch, n_epochs))
    # out_28 = model.layer28(out_27)
    # outputs_bino.append(out_28.data.cpu().numpy())
    # layers_name.append("layer28")

    # # layer 29
    # # print("Epoch: {}/{} - activating layer: layer 29".format(epoch, n_epochs))
    # out_29 = model.layer29(out_28)
    # outputs_bino.append(out_29.data.cpu().numpy())
    # layers_name.append("layer29")

    # # layer 30
    # # print("Epoch: {}/{} - activating layer: layer 30".format(epoch, n_epochs))
    # out_30 = model.layer30(out_27)
    # outputs_bino.append(out_30.data.cpu().numpy())
    # layers_name.append("layer30")

    # # layer 31
    # # print("Epoch: {}/{} - activating layer: layer 31".format(epoch, n_epochs))
    # out_31 = model.layer31(out_30)
    # outputs_bino.append(out_31.data.cpu().numpy())
    # layers_name.append("layer31")

    # # layer 32
    # # print("Epoch: {}/{} - activating layer: layer 32".format(epoch, n_epochs))
    # out_32 = model.layer32(out_31)
    # outputs_bino.append(out_32.data.cpu().numpy())
    # layers_name.append("layer32")

    # # layer 33a
    # out_33a = model.layer33a(out_32)

    # # layer 33b
    # # print("Epoch: {}/{} - activating layer: layer 33".format(epoch, n_epochs))
    # out_33b = out_33a + out_29
    # outputs_bino.append(out_33b.data.cpu().numpy())
    # layers_name.append("layer33")

    # # layer 34a
    # out_34a = model.layer34a(out_33b)

    # # layer 34b
    # # print("Epoch: {}/{} - activating layer: layer 34".format(epoch, n_epochs))
    # out_34b = out_34a + out_26
    # outputs_bino.append(out_34b.data.cpu().numpy())
    # layers_name.append("layer34")

    # # layer 35a
    # out_35a = model.layer35a(out_34b)

    # # layer 35b
    # # print("Epoch: {}/{} - activating layer: layer 35".format(epoch, n_epochs))
    # out_35b = out_35a + out_23
    # outputs_bino.append(out_35b.data.cpu().numpy())
    # layers_name.append("layer35")

    # # layer 36a
    # out_36a = model.layer36a(out_35b)

    # # layer 36b
    # # print("Epoch: {}/{} - activating layer: layer 36".format(epoch, n_epochs))
    # out_36b = out_36a + out_20
    # outputs_bino.append(out_36b.data.cpu().numpy())
    # layers_name.append("layer36")

    # # layer 37
    # # print("Epoch: {}/{} - activating layer: layer 37".format(epoch, n_epochs))
    # out_37 = model.layer37(out_36b)
    # outputs_bino.append(out_37.data.cpu().numpy())
    # layers_name.append("layer37")

    # # final layer
    # out = out_37.view(len(out_37), model.max_disp, model.img_height, model.img_width)
    # out = F.softmax(-out, 1)
    # out = torch.sum(out.mul(loss_mul), 1)
    # outputs_bino.append(out.data.cpu().numpy())
    # layers_name.append("final")

    return outputs_bino, layers_name


# %% define layer activation
def compute_layer_activation(model, input_left, input_right):
    batch_size = input_left.shape[0]
    loss_mul_list = []
    for d in range(maxdisp):
        loss_mul_temp = (
            Variable(torch.Tensor(np.ones([batch_size, 1, h, w]) * d))
            .pin_memory()
            .to(device, non_blocking=True)
        )
        loss_mul_list.append(loss_mul_temp)
    loss_mul = torch.cat(loss_mul_list, 1)

    # get model's layers
    model_children = list(model.children())
    childrens_name = [name for name, _ in model.named_children()]

    outputs_left = []
    outputs_right = []
    outputs_bino = []
    layers_name = []

    # compute layer activations before the left-right inputs concatenate
    out_left = input_left
    out_right = input_right
    for i in range(3):
        layer = model_children[i]

        if i == 1:  # residual blocks
            for j in range(len(layer)):
                sublayer = layer[j]
                layer_name = (
                    childrens_name[i]
                    + "_"
                    + str(sublayer).partition("(")[0]
                    + "_"
                    + str(j)
                )
                layers_name.append(layer_name)

                # print(
                #     "Epoch: {}/{} - activating layer: {}".format(
                #         epoch, n_epochs, layer_name
                #     )
                # )

                out_left = sublayer(out_left)
                outputs_left.append(out_left.data.cpu().numpy())

                out_right = sublayer(out_right)
                outputs_right.append(out_right.data.cpu().numpy())

        else:
            # print(
            #     "Epoch: {}/{} - activating layer: {}".format(
            #         epoch, n_epochs, childrens_name[i]
            #     )
            # )

            out_left = layer(out_left)
            outputs_left.append(out_left.data.cpu().numpy())

            out_right = layer(out_right)
            outputs_right.append(out_right.data.cpu().numpy())

            layers_name.append(childrens_name[i])

    # cost-volume layers
    # output_dim: [n_batch, n_features_left_and_right, max_disp/2, img_height/2, img_width/2]
    # output_dim: [n_batch, 64, 96, 128, 256]
    # print("Epoch: {}/{} - activating layer: cost-volume".format(epoch, n_epochs))
    cost_vol = model.cost_volume(out_left, out_right)
    outputs_bino.append(cost_vol.data.cpu().numpy())
    layers_name.append("cost_vol")

    # layer 19
    # print("Epoch: {}/{} - activating layer: layer 19".format(epoch, n_epochs))
    out_19 = model.layer19(cost_vol)
    outputs_bino.append(out_19.data.cpu().numpy())
    layers_name.append("layer19")

    # layer 20
    # print("Epoch: {}/{} - activating layer: layer 20".format(epoch, n_epochs))
    out_20 = model.layer20(out_19)
    outputs_bino.append(out_20.data.cpu().numpy())
    layers_name.append("layer20")

    # layer 21
    # print("Epoch: {}/{} - activating layer: layer 21".format(epoch, n_epochs))
    out_21 = model.layer21(cost_vol)
    outputs_bino.append(out_21.data.cpu().numpy())
    layers_name.append("layer21")

    # layer 22
    # print("Epoch: {}/{} - activating layer: layer 22".format(epoch, n_epochs))
    out_22 = model.layer22(out_21)
    outputs_bino.append(out_22.data.cpu().numpy())
    layers_name.append("layer22")

    # layer 23
    # print("Epoch: {}/{} - activating layer: layer 23".format(epoch, n_epochs))
    out_23 = model.layer23(out_22)
    outputs_bino.append(out_23.data.cpu().numpy())
    layers_name.append("layer23")

    # layer 24
    # print("Epoch: {}/{} - activating layer: layer 24".format(epoch, n_epochs))
    out_24 = model.layer24(out_21)
    outputs_bino.append(out_24.data.cpu().numpy())
    layers_name.append("layer24")

    # layer 25
    # print("Epoch: {}/{} - activating layer: layer 25".format(epoch, n_epochs))
    out_25 = model.layer25(out_24)
    outputs_bino.append(out_25.data.cpu().numpy())
    layers_name.append("layer25")

    # layer 26
    # print("Epoch: {}/{} - activating layer: layer 26".format(epoch, n_epochs))
    out_26 = model.layer26(out_25)
    outputs_bino.append(out_26.data.cpu().numpy())
    layers_name.append("layer26")

    # layer 27
    # print("Epoch: {}/{} - activating layer: layer 27".format(epoch, n_epochs))
    out_27 = model.layer27(out_24)
    outputs_bino.append(out_27.data.cpu().numpy())
    layers_name.append("layer27")

    # layer 28
    # print("Epoch: {}/{} - activating layer: layer 28".format(epoch, n_epochs))
    out_28 = model.layer28(out_27)
    outputs_bino.append(out_28.data.cpu().numpy())
    layers_name.append("layer28")

    # layer 29
    # print("Epoch: {}/{} - activating layer: layer 29".format(epoch, n_epochs))
    out_29 = model.layer29(out_28)
    outputs_bino.append(out_29.data.cpu().numpy())
    layers_name.append("layer29")

    # layer 30
    # print("Epoch: {}/{} - activating layer: layer 30".format(epoch, n_epochs))
    out_30 = model.layer30(out_27)
    outputs_bino.append(out_30.data.cpu().numpy())
    layers_name.append("layer30")

    # layer 31
    # print("Epoch: {}/{} - activating layer: layer 31".format(epoch, n_epochs))
    out_31 = model.layer31(out_30)
    outputs_bino.append(out_31.data.cpu().numpy())
    layers_name.append("layer31")

    # layer 32
    # print("Epoch: {}/{} - activating layer: layer 32".format(epoch, n_epochs))
    out_32 = model.layer32(out_31)
    outputs_bino.append(out_32.data.cpu().numpy())
    layers_name.append("layer32")

    # layer 33a
    out_33a = model.layer33a(out_32)

    # layer 33b
    # print("Epoch: {}/{} - activating layer: layer 33".format(epoch, n_epochs))
    out_33b = out_33a + out_29
    outputs_bino.append(out_33b.data.cpu().numpy())
    layers_name.append("layer33")

    # layer 34a
    out_34a = model.layer34a(out_33b)

    # layer 34b
    # print("Epoch: {}/{} - activating layer: layer 34".format(epoch, n_epochs))
    out_34b = out_34a + out_26
    outputs_bino.append(out_34b.data.cpu().numpy())
    layers_name.append("layer34")

    # layer 35a
    out_35a = model.layer35a(out_34b)

    # layer 35b
    # print("Epoch: {}/{} - activating layer: layer 35".format(epoch, n_epochs))
    out_35b = out_35a + out_23
    outputs_bino.append(out_35b.data.cpu().numpy())
    layers_name.append("layer35")

    # layer 36a
    out_36a = model.layer36a(out_35b)

    # layer 36b
    # print("Epoch: {}/{} - activating layer: layer 36".format(epoch, n_epochs))
    out_36b = out_36a + out_20
    outputs_bino.append(out_36b.data.cpu().numpy())
    layers_name.append("layer36")

    # layer 37
    # print("Epoch: {}/{} - activating layer: layer 37".format(epoch, n_epochs))
    out_37 = model.layer37(out_36b)
    outputs_bino.append(out_37.data.cpu().numpy())
    layers_name.append("layer37")

    # final layer
    out = out_37.view(len(out_37), model.max_disp, model.img_height, model.img_width)
    out = F.softmax(-out, 1)
    out = torch.sum(out.mul(loss_mul), 1)
    outputs_bino.append(out.data.cpu().numpy())
    layers_name.append("final")

    return outputs_left, outputs_right, outputs_bino, layers_name


# %% compute layer activation for RDSs
def compute_layer_activation_rds(model, rds_left, rds_right, batch_size):
    n_rds_each_disp = rds_left.shape[0]

    output_layers_left = {
        "layer1": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_0": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_1": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_2": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_3": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_4": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_5": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_6": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_7": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "layer18": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
    }

    output_layers_right = {
        "layer1": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_0": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_1": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_2": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_3": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_4": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_5": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_6": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "res_block_7": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "layer18": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
    }

    # output_layers_bino = {
    #     "cost_vol": np.zeros((n_rds_each_disp, 64, 96, 128, 256), dtype=np.float32),
    #     "layer19": np.zeros((n_rds_each_disp, 32, 96, 128, 256), dtype=np.float32),
    #     "layer20": np.zeros((n_rds_each_disp, 32, 96, 128, 256), dtype=np.float32),
    #     "layer21": np.zeros((n_rds_each_disp, 64, 48, 64, 128), dtype=np.float32),
    #     "layer22": np.zeros((n_rds_each_disp, 64, 48, 64, 128), dtype=np.float32),
    #     "layer23": np.zeros((n_rds_each_disp, 64, 48, 64, 128), dtype=np.float32),
    #     "layer24": np.zeros((n_rds_each_disp, 64, 24, 32, 64), dtype=np.float32),
    #     "layer25": np.zeros((n_rds_each_disp, 64, 24, 32, 64), dtype=np.float32),
    #     "layer26": np.zeros((n_rds_each_disp, 64, 24, 32, 64), dtype=np.float32),
    #     "layer27": np.zeros((n_rds_each_disp, 64, 12, 16, 32), dtype=np.float32),
    #     "layer28": np.zeros((n_rds_each_disp, 64, 12, 16, 32), dtype=np.float32),
    #     "layer29": np.zeros((n_rds_each_disp, 64, 12, 16, 32), dtype=np.float32),
    #     "layer30": np.zeros((n_rds_each_disp, 128, 6, 8, 16), dtype=np.float32),
    #     "layer31": np.zeros((n_rds_each_disp, 128, 6, 8, 16), dtype=np.float32),
    #     "layer32": np.zeros((n_rds_each_disp, 128, 6, 8, 16), dtype=np.float32),
    #     "layer33": np.zeros((n_rds_each_disp, 64, 12, 16, 32), dtype=np.float32),
    #     "layer34": np.zeros((n_rds_each_disp, 64, 24, 32, 64), dtype=np.float32),
    #     "layer35": np.zeros((n_rds_each_disp, 64, 48, 64, 128), dtype=np.float32),
    #     "layer36": np.zeros((n_rds_each_disp, 32, 96, 128, 256), dtype=np.float32),
    #     "layer37": np.zeros((n_rds_each_disp, 1, 192, 256, 512), dtype=np.float32),
    #     "final": np.zeros((n_rds_each_disp, 256, 512), dtype=np.float32),
    # }
    output_layers_bino = {
        "cost_vol": np.zeros((n_rds_each_disp, 64, 128, 256), dtype=np.float32),
        "layer19": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "layer20": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "layer21": np.zeros((n_rds_each_disp, 64, 64, 128), dtype=np.float32),
        "layer22": np.zeros((n_rds_each_disp, 64, 64, 128), dtype=np.float32),
        "layer23": np.zeros((n_rds_each_disp, 64, 64, 128), dtype=np.float32),
        "layer24": np.zeros((n_rds_each_disp, 64, 32, 64), dtype=np.float32),
        "layer25": np.zeros((n_rds_each_disp, 64, 32, 64), dtype=np.float32),
        "layer26": np.zeros((n_rds_each_disp, 64, 32, 64), dtype=np.float32),
        "layer27": np.zeros((n_rds_each_disp, 64, 16, 32), dtype=np.float32),
        "layer28": np.zeros((n_rds_each_disp, 64, 16, 32), dtype=np.float32),
        "layer29": np.zeros((n_rds_each_disp, 64, 16, 32), dtype=np.float32),
        "layer30": np.zeros((n_rds_each_disp, 128, 8, 16), dtype=np.float32),
        "layer31": np.zeros((n_rds_each_disp, 128, 8, 16), dtype=np.float32),
        "layer32": np.zeros((n_rds_each_disp, 128, 8, 16), dtype=np.float32),
        "layer33": np.zeros((n_rds_each_disp, 64, 16, 32), dtype=np.float32),
        "layer34": np.zeros((n_rds_each_disp, 64, 32, 64), dtype=np.float32),
        "layer35": np.zeros((n_rds_each_disp, 64, 64, 128), dtype=np.float32),
        "layer36": np.zeros((n_rds_each_disp, 32, 128, 256), dtype=np.float32),
        "layer37": np.zeros((n_rds_each_disp, 1, 256, 512), dtype=np.float32),
        "final": np.zeros((n_rds_each_disp, 256, 512), dtype=np.float32),
    }

    n_epochs = n_rds_each_disp // batch_size
    for i in range(n_epochs):
        id_start = i * batch_size
        id_end = id_start + batch_size

        ## layers before left-right inputs concatenate
        ## crossed disparity
        # rds_left = ards_left_crossed[id_start:id_end]
        # rds_right = ards_right_crossed[id_start:id_end]
        input_left = torch.tensor(rds_left[id_start:id_end])
        input_right = torch.tensor(rds_right[id_start:id_end])

        # move to gpu
        input_left = input_left.pin_memory().to(device, non_blocking=True)
        input_right = input_right.pin_memory().to(device, non_blocking=True)
        # input_disp = disps.pin_memory().to(device, non_blocking=True)

        print("iter: {}/{}, compute layer activation".format(i, n_epochs))
        (
            outputs_left,
            outputs_right,
            outputs_bino,
            layers_name,
        ) = compute_layer_activation(model, input_left, input_right)

        ## flatten out the activation in each layer
        print("iter: {}/{}, flattening out monolayer outputs".format(i, n_epochs))
        for j in range(len(outputs_left)):  # iterate over layers
            layer_name = monolayer_names[j]
            # monocular left layers
            temp = outputs_left[j]
            output_layers_left[layer_name][id_start:id_end] = temp

            # monocular right layers
            temp = outputs_right[j]
            output_layers_right[layer_name][id_start:id_end] = temp

        # bino layers
        print("iter: {}/{}, flattening out binolayer outputs".format(i, n_epochs))
        for j in range(len(outputs_bino)):
            layer_name = binolayer_names[j]
            temp = outputs_bino[j]
            if j < len(outputs_bino) - 1:
                # average along disparity axis
                temp = temp.mean(axis=2)
                output_layers_bino[layer_name][id_start:id_end] = temp
            else:  # final layer
                output_layers_bino[layer_name][id_start:id_end] = temp

    return (output_layers_left, output_layers_right, output_layers_bino)


# %% define rdm computation
def compute_layer_activation_all_rds(
    model,
    ards_left_crossed,
    ards_left_uncrossed,
    ards_right_crossed,
    ards_right_uncrossed,
    hmrds_left_crossed,
    hmrds_left_uncrossed,
    hmrds_right_crossed,
    hmrds_right_uncrossed,
    crds_left_crossed,
    crds_left_uncrossed,
    crds_right_crossed,
    crds_right_uncrossed,
    dotDens,
    batch_size,
):
    ##################################### ards #########################################
    print("compute activation for ards-crossed, dotDens: {:.1f}".format(dotDens))
    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, ards_left_crossed, ards_right_crossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_ards_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_ards_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_ards_crossed_binolayers_dotDens_{:.1f}.pkl".format(SAVE_PATH, dotDens),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")

    print("compute activation for ards-uncrossed, dotDens: {:.1f}".format(dotDens))
    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, ards_left_uncrossed, ards_right_uncrossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_ards_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_ards_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_ards_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")

    ##################################### hmrds #########################################
    print("compute activation for hmrds-crossed, dotDens: {:.1f}".format(dotDens))

    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, hmrds_left_crossed, hmrds_right_crossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_hmrds_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_hmrds_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_hmrds_crossed_binolayers_dotDens_{:.1f}.pkl".format(SAVE_PATH, dotDens),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")

    print("compute activation for hmrds-uncrossed, dotDens: {:.1f}".format(dotDens))
    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, hmrds_left_uncrossed, hmrds_right_uncrossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_hmrds_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_hmrds_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_hmrds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")

    ##################################### crds #########################################
    print("compute activation for crds-crossed, dotDens: {:.1f}".format(dotDens))
    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, crds_left_crossed, crds_right_crossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_crds_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_crds_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_crds_crossed_binolayers_dotDens_{:.1f}.pkl".format(SAVE_PATH, dotDens),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")

    print("compute activation for crds-uncrossed, dotDens: {:.1f}".format(dotDens))
    (
        act_rds_left,
        act_rds_right,
        act_rds_bino,
    ) = compute_layer_activation_rds(
        model, crds_left_uncrossed, crds_right_uncrossed, batch_size
    )

    ## save file
    print("saving files")
    with open(
        "{}/act_crds_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_left, fp)
        print("saving file successfully")
    with open(
        "{}/act_crds_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_right, fp)
        print("saving file successfully")
    with open(
        "{}/act_crds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
            SAVE_PATH, dotDens
        ),
        "wb",
    ) as fp:
        pickle.dump(act_rds_bino, fp)
        print("saving file successfully")


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))


def rdm_single_layer(
    act_ards_crossed,
    act_ards_uncrossed,
    act_hmrds_crossed,
    act_hmrds_uncrossed,
    act_crds_crossed,
    act_crds_uncrossed,
    layer_names,
    n_rds_each_disp,
    i,
):
    layer_name = layer_names[i]
    print("compute rdm for layer: {}".format(layer_name))
    activation = [
        act_ards_crossed[layer_name],
        act_ards_uncrossed[layer_name],
        act_hmrds_crossed[layer_name],
        act_hmrds_uncrossed[layer_name],
        act_crds_crossed[layer_name],
        act_crds_uncrossed[layer_name],
    ]

    rdm = np.zeros(
        (n_rds_each_disp, len(activation), len(activation)),
        dtype=np.float32,
    )
    for cond1 in range(len(activation)):
        for cond2 in range(cond1 + 1, len(activation)):
            temp1 = activation[cond1].reshape(n_rds_each_disp, -1)
            temp2 = activation[cond2].reshape(n_rds_each_disp, -1)

            dist = euclidean_dist(temp1, temp2)
            rdm[:, cond1, cond2] = dist

            # copy to lower triangle
            rdm[:, cond2, cond1] = dist

    return rdm


def compute_rdm(dotDens):
    ########################## monolayer left #######################################
    file_name = "{}/act_ards_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_crossed = pickle.load(fp)

    file_name = "{}/act_ards_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_uncrossed = pickle.load(fp)

    file_name = "{}/act_hmrds_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_crossed = pickle.load(fp)

    file_name = "{}/act_hmrds_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_uncrossed = pickle.load(fp)

    file_name = "{}/act_crds_crossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_crossed = pickle.load(fp)

    file_name = "{}/act_crds_uncrossed_monolayers_left_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_uncrossed = pickle.load(fp)

    n_rds_each_disp = len(act_ards_crossed["layer1"])
    rdm_all_layers = np.zeros(
        (2 * n_monolayers + n_binolayers, n_rds_each_disp, 6, 6), dtype=np.float32
    )
    ## compute the rdm
    count_layer = 0

    rdm_layers = []
    rdm_layers.append(
        Parallel(n_jobs=n_monolayers)(
            delayed(rdm_single_layer)(
                act_ards_crossed,
                act_ards_uncrossed,
                act_hmrds_crossed,
                act_hmrds_uncrossed,
                act_crds_crossed,
                act_crds_uncrossed,
                monolayer_names,
                n_rds_each_disp,
                i,
            )
            for i in range(n_monolayers)
        )
    )

    # unpacking
    for i in range(n_monolayers):
        temp = rdm_layers[0][i]
        rdm_all_layers[count_layer] = temp
        count_layer += 1

    ########################## monolayer right #######################################
    file_name = "{}/act_ards_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_crossed = pickle.load(fp)

    file_name = "{}/act_ards_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_uncrossed = pickle.load(fp)

    file_name = "{}/act_hmrds_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_crossed = pickle.load(fp)

    file_name = "{}/act_hmrds_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_uncrossed = pickle.load(fp)

    file_name = "{}/act_crds_crossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_crossed = pickle.load(fp)

    file_name = "{}/act_crds_uncrossed_monolayers_right_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_uncrossed = pickle.load(fp)

    ## compute the rdm
    rdm_layers = []
    rdm_layers.append(
        Parallel(n_jobs=n_monolayers)(
            delayed(rdm_single_layer)(
                act_ards_crossed,
                act_ards_uncrossed,
                act_hmrds_crossed,
                act_hmrds_uncrossed,
                act_crds_crossed,
                act_crds_uncrossed,
                monolayer_names,
                n_rds_each_disp,
                i,
            )
            for i in range(n_monolayers)
        )
    )

    # unpacking
    for i in range(n_monolayers):
        temp = rdm_layers[0][i]
        rdm_all_layers[count_layer] = temp
        count_layer += 1

    ########################## binolayer #######################################
    file_name = "{}/act_ards_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_crossed = pickle.load(fp)

    file_name = "{}/act_ards_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_uncrossed = pickle.load(fp)

    file_name = "{}/act_hmrds_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_crossed = pickle.load(fp)

    file_name = "{}/act_hmrds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_uncrossed = pickle.load(fp)

    file_name = "{}/act_crds_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_crossed = pickle.load(fp)

    file_name = "{}/act_crds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_uncrossed = pickle.load(fp)

    ## compute the rdm
    rdm_layers = []
    rdm_layers.append(
        Parallel(n_jobs=n_binolayers)(
            delayed(rdm_single_layer)(
                act_ards_crossed,
                act_ards_uncrossed,
                act_hmrds_crossed,
                act_hmrds_uncrossed,
                act_crds_crossed,
                act_crds_uncrossed,
                binolayer_names,
                n_rds_each_disp,
                i,
            )
            for i in range(n_binolayers)
        )
    )

    # unpacking
    for i in range(n_binolayers):
        temp = rdm_layers[0][i]
        rdm_all_layers[count_layer] = temp
        count_layer += 1

    ## save file
    np.save(
        "{}/rdm_all_layers_dotDens_{}.npy".format(SAVE_PATH, dotDens),
        rdm_all_layers,
    )

    return rdm_all_layers


# %% plot


def plotHeat_rdm(rdm_all_layers, dotDens, save_flag):
    n_layers = 2 * len(monolayer_names) + len(binolayer_names)

    # average across examples
    rdm_avg = rdm_all_layers.mean(axis=1)

    ## reconstruct rdm
    rdm_reconstruct = np.zeros((n_layers, 6, 6), dtype=np.float32)
    for layer in range(n_layers):
        # for i in range(n_rds):
        ## get upper triangle elements
        temp = rdm_avg[layer]

        # normalize to (0, 1)
        num = temp.max() - temp.min()
        rdm_norm = (temp - temp.min()) / num

        rdm_reconstruct[layer] = rdm_norm

    conds = [
        "aRDS_cross",
        "aRDS_uncross",
        "hmRDS_cross",
        "hmRDS_uncross",
        "cRDS_cross",
        "cRDS_uncross",
    ]

    # start plotting
    sns.set_theme()
    sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

    # estimate v_min and v_max for cbar
    v_min = 0
    v_max = 1.0
    # v_min = np.round(np.min(rdm_reconstruct), 2)
    # v_max = np.round(np.max(rdm_reconstruct), 2)

    figsize = (12, 15)
    n_row = 9
    n_col = 5

    fig, axes = plt.subplots(
        nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
    )

    fig.text(
        0.5,
        1.02,
        "RDM for each layer in GC-Net, dotDens: {:.1f}".format(dotDens),
        ha="center",
    )
    fig.text(-0.15, 0.5, "Conditions", va="center", rotation=90)
    fig.text(0.5, -0.15, "Conditions", ha="center")
    # fig.text(0.5, -0.5, "pearson_r = {}".format(str(np.round(rdm_r, 2))), ha="center")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    cmap = "jet"
    for layer in range(n_layers):
        id_row = layer // n_col
        id_col = layer % n_col

        sns.heatmap(
            rdm_reconstruct[layer],
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=conds,
            yticklabels=conds,
            ax=axes[id_row, id_col],
        )

        axes[id_row, id_col].set_title(layer_names_all[layer], fontsize=15)

    ## save
    if save_flag == 1:
        fig.savefig(
            "{}/plotHeat_rdm_all_layers_dotDens_{:.1f}.pdf".format(SAVE_PATH, dotDens),
            dpi=600,
            bbox_inches="tight",
        )


# %% Merge RDM of all conditions for each layer


def load_activation(act_name, dotDens_list):
    """
    load layer activation files associated with act_name including all dotDensity.


    Args:
        act_name (string): activation filename
        for example: act_name = "act_ards_crossed_monolayers_left"

        dotDens_list (np.array): a list containing dot density
        for ex: 0.1 * np.arange(1, 10)

    Returns:
        activation_dict [dict]: a dict containing all activation data of a rds_type
                    with all dot density
    """
    #
    activation_dict = {}

    for d in range(len(dotDens_list)):
        dotDens = dotDens_list[d]
        file_name = "{}/{}_dotDens_{:.1f}.pkl".format(SAVE_PATH, act_name, dotDens)
        with open(file_name, "rb") as fp:
            act = pickle.load(fp)
        temp = "{}_{:.1f}".format(act_name, dotDens)
        activation_dict[temp] = act

    return activation_dict


def rdm_all_conditions_single_layer(
    activation_dict1,
    activation_dict2,
    act_name1,
    act_name2,
    layer_names,
    n_rds_each_disp,
    dotDens_list,
    layer,
):
    layer_name = layer_names[layer]

    # compute rdm
    rdm = np.zeros(
        (n_rds_each_disp, len(dotDens_list), len(dotDens_list)),
        dtype=np.float32,
    )
    for cond1 in range(len(dotDens_list)):
        cond_name1 = "{}_{:.1f}".format(act_name1, dotDens_list[cond1])
        for cond2 in range(len(dotDens_list)):
            cond_name2 = "{}_{:.1f}".format(act_name2, dotDens_list[cond2])
            print(
                "{} => euclidean dist: {} vs. {}".format(
                    layer_name, cond_name1, cond_name2
                )
            )

            temp1 = activation_dict1[cond_name1][layer_name].reshape(
                n_rds_each_disp, -1
            )
            temp2 = activation_dict2[cond_name2][layer_name].reshape(
                n_rds_each_disp, -1
            )

            dist = euclidean_dist(temp1, temp2)
            rdm[:, cond1, cond2] = dist

    return rdm


def rdm_all_conditions(dotDens_list, n_rds_each_disp):
    ########################## monolayer left #######################################

    n_conds = 6 * len(dotDens_list)
    rdm_all_layers = np.zeros(
        (2 * n_monolayers + n_binolayers, n_rds_each_disp, n_conds, n_conds),
        dtype=np.float32,
    )

    act_name_list = [
        "act_ards_crossed_monolayers_left",
        "act_ards_uncrossed_monolayers_left",
        "act_hmrds_crossed_monolayers_left",
        "act_hmrds_uncrossed_monolayers_left",
        "act_crds_crossed_monolayers_left",
        "act_crds_uncrossed_monolayers_left",
    ]

    for i in range(len(act_name_list)):
        act_name1 = act_name_list[i]
        activation_dict1 = load_activation(act_name1, dotDens_list)

        row_start = i * len(dotDens_list)
        row_end = row_start + len(dotDens_list)

        for j in range(i, len(act_name_list)):
            act_name2 = act_name_list[j]
            activation_dict2 = load_activation(act_name2, dotDens_list)

            ## compute the rdm
            rdm_layers = []
            rdm_layers.append(
                Parallel(n_jobs=1)(
                    delayed(rdm_all_conditions_single_layer)(
                        activation_dict1,
                        activation_dict2,
                        act_name1,
                        act_name2,
                        monolayer_names,
                        n_rds_each_disp,
                        dotDens_list,
                        layer,
                    )
                    for layer in range(n_monolayers)
                )
            )

            # unpacking
            count_layer = 0
            col_start = j * len(dotDens_list)
            col_end = col_start + len(dotDens_list)
            for layer in range(n_monolayers):
                temp = rdm_layers[0][layer]
                rdm_all_layers[
                    count_layer, :, row_start:row_end, col_start:col_end
                ] = temp

                count_layer += 1

    ########################## monolayer right #######################################

    act_name_list = [
        "act_ards_crossed_monolayers_right",
        "act_ards_uncrossed_monolayers_right",
        "act_hmrds_crossed_monolayers_right",
        "act_hmrds_uncrossed_monolayers_right",
        "act_crds_crossed_monolayers_right",
        "act_crds_uncrossed_monolayers_right",
    ]

    for i in range(len(act_name_list)):
        act_name1 = act_name_list[i]
        activation_dict1 = load_activation(act_name1, dotDens_list)

        row_start = i * len(dotDens_list)
        row_end = row_start + len(dotDens_list)

        for j in range(i, len(act_name_list)):
            act_name2 = act_name_list[j]
            activation_dict2 = load_activation(act_name2, dotDens_list)

            ## compute the rdm
            rdm_layers = []
            rdm_layers.append(
                Parallel(n_jobs=1)(
                    delayed(rdm_all_conditions_single_layer)(
                        activation_dict1,
                        activation_dict2,
                        act_name1,
                        act_name2,
                        monolayer_names,
                        n_rds_each_disp,
                        dotDens_list,
                        layer,
                    )
                    for layer in range(n_monolayers)
                )
            )

            # unpacking
            count_layer = n_monolayers
            col_start = j * len(dotDens_list)
            col_end = col_start + len(dotDens_list)
            for layer in range(n_monolayers):
                temp = rdm_layers[0][layer]
                rdm_all_layers[
                    count_layer, :, row_start:row_end, col_start:col_end
                ] = temp

                count_layer += 1

    ########################## binolayer #######################################

    act_name_list = [
        "act_ards_crossed_binolayers",
        "act_ards_uncrossed_binolayers",
        "act_hmrds_crossed_binolayers",
        "act_hmrds_uncrossed_binolayers",
        "act_crds_crossed_binolayers",
        "act_crds_uncrossed_binolayers",
    ]

    for i in range(len(act_name_list)):
        act_name1 = act_name_list[i]
        activation_dict1 = load_activation(act_name1, dotDens_list)

        row_start = i * len(dotDens_list)
        row_end = row_start + len(dotDens_list)

        for j in range(i, len(act_name_list)):
            act_name2 = act_name_list[j]
            activation_dict2 = load_activation(act_name2, dotDens_list)

            ## compute the rdm
            rdm_layers = []
            rdm_layers.append(
                Parallel(n_jobs=1)(
                    delayed(rdm_all_conditions_single_layer)(
                        activation_dict1,
                        activation_dict2,
                        act_name1,
                        act_name2,
                        binolayer_names,
                        n_rds_each_disp,
                        dotDens_list,
                        layer,
                    )
                    for layer in range(n_binolayers)
                )
            )

            # unpacking
            count_layer = 2 * n_monolayers
            col_start = j * len(dotDens_list)
            col_end = col_start + len(dotDens_list)
            for layer in range(n_binolayers):
                temp = rdm_layers[0][layer]
                rdm_all_layers[
                    count_layer, :, row_start:row_end, col_start:col_end
                ] = temp

                count_layer += 1

    # copy to lower triangle
    rdm_all_layers = np.triu(rdm_all_layers)
    rdm_all_layers = rdm_all_layers + rdm_all_layers.transpose(0, 1, 3, 2)

    # save file
    np.save("{}/rdm_all_layers_all_conds.npy".format(SAVE_PATH), rdm_all_layers)

    return rdm_all_layers


def plotHeat_rdm_all_conds(rdm_all_layers, save_flag):
    n_layers = 2 * len(monolayer_names) + len(binolayer_names)

    # average across examples
    rdm_avg = rdm_all_layers.mean(axis=1)

    ## reconstruct rdm
    rdm_reconstruct = np.zeros((n_layers, 54, 54), dtype=np.float32)
    for layer in range(n_layers):
        # for i in range(n_rds):
        temp = rdm_avg[layer]

        # normalize to (0, 1)
        num = temp.max() - temp.min()
        rdm_norm = (temp - temp.min()) / num

        rdm_reconstruct[layer] = rdm_norm

    conds = [
        "aRDS_cross_0.1",
        "aRDS_cross_0.2",
        "aRDS_cross_0.3",
        "aRDS_cross_0.4",
        "aRDS_cross_0.5",
        "aRDS_cross_0.6",
        "aRDS_cross_0.7",
        "aRDS_cross_0.8",
        "aRDS_cross_0.9",
        "aRDS_uncross_0.1",
        "aRDS_uncross_0.2",
        "aRDS_uncross_0.3",
        "aRDS_uncross_0.4",
        "aRDS_uncross_0.5",
        "aRDS_uncross_0.6",
        "aRDS_uncross_0.7",
        "aRDS_uncross_0.8",
        "aRDS_uncross_0.9",
        "hmRDS_cross_0.1",
        "hmRDS_cross_0.2",
        "hmRDS_cross_0.3",
        "hmRDS_cross_0.4",
        "hmRDS_cross_0.5",
        "hmRDS_cross_0.6",
        "hmRDS_cross_0.7",
        "hmRDS_cross_0.8",
        "hmRDS_cross_0.9",
        "hmRDS_uncross_0.1",
        "hmRDS_uncross_0.2",
        "hmRDS_uncross_0.3",
        "hmRDS_uncross_0.4",
        "hmRDS_uncross_0.5",
        "hmRDS_uncross_0.6",
        "hmRDS_uncross_0.7",
        "hmRDS_uncross_0.8",
        "hmRDS_uncross_0.9",
        "cRDS_cross_0.1",
        "cRDS_cross_0.2",
        "cRDS_cross_0.3",
        "cRDS_cross_0.4",
        "cRDS_cross_0.5",
        "cRDS_cross_0.6",
        "cRDS_cross_0.7",
        "cRDS_cross_0.8",
        "cRDS_cross_0.9",
        "cRDS_uncross_0.1",
        "cRDS_uncross_0.2",
        "cRDS_uncross_0.3",
        "cRDS_uncross_0.4",
        "cRDS_uncross_0.5",
        "cRDS_uncross_0.6",
        "cRDS_uncross_0.7",
        "cRDS_uncross_0.8",
        "cRDS_uncross_0.9",
    ]

    # %%start plotting
    sns.set()
    sns.set(context="paper", style="white", font_scale=2, palette="deep")

    # estimate v_min and v_max for cbar
    v_min = 0
    v_max = 1.0
    # v_min = np.round(np.min(rdm_reconstruct), 2)
    # v_max = np.round(np.max(rdm_reconstruct), 2)

    figsize = (12, 15)
    n_row = 9
    n_col = 5

    fig, axes = plt.subplots(
        nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
    )

    fig.text(
        0.5,
        1.02,
        "RDM for each layer in GC-Net all conds",
        ha="center",
    )
    fig.text(-0.15, 0.5, "Conditions", va="center", rotation=90)
    fig.text(-0.08, 0.12, "aRDS_crossed", ha="left", va="center", fontsize=12)
    fig.text(-0.08, 0.105, "aRDS_uncrossed", ha="left", va="center", fontsize=12)
    fig.text(-0.08, 0.09, "hmRDS_crossed", ha="left", va="center", fontsize=12)
    fig.text(-0.08, 0.075, "hmRDS_uncrossed", ha="left", va="center", fontsize=12)
    fig.text(-0.08, 0.06, "cRDS_crossed", ha="left", va="center", fontsize=12)
    fig.text(-0.08, 0.045, "cRDS_uncrossed", ha="left", va="center", fontsize=12)

    fig.text(0.5, -0.15, "Conditions", ha="center")
    fig.text(0.07, -0.07, "aRDS_crossed", va="baseline", fontsize=12, rotation=90)
    fig.text(0.09, -0.07, "aRDS_uncrossed", va="baseline", fontsize=12, rotation=90)
    fig.text(0.11, -0.07, "hmRDS_crossed", va="baseline", fontsize=12, rotation=90)
    fig.text(0.13, -0.07, "hmRDS_uncrossed", va="baseline", fontsize=12, rotation=90)
    fig.text(0.15, -0.07, "cRDS_crossed", va="baseline", fontsize=12, rotation=90)
    fig.text(0.17, -0.07, "cRDS_uncrossed", va="baseline", fontsize=12, rotation=90)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    cmap = "jet"
    for layer in range(n_layers):
        id_row = layer // n_col
        id_col = layer % n_col

        sns.heatmap(
            rdm_reconstruct[layer],
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            xticklabels=[],
            yticklabels=[],
            ax=axes[id_row, id_col],
        )

        axes[id_row, id_col].set_title(layer_names_all[layer], fontsize=15)

    # %% save
    if save_flag == 1:
        fig.savefig(
            "{}/plotHeat_rdm_all_layers_all_conds.pdf".format(SAVE_PATH),
            dpi=600,
            bbox_inches="tight",
        )


# %% SVM analysis


def load_binolayer_activation(dotDens):
    ########################## binolayer #######################################
    file_name = "{}/act_ards_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_crossed = pickle.load(fp)

    file_name = "{}/act_ards_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_ards_uncrossed = pickle.load(fp)

    file_name = "{}/act_hmrds_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_crossed = pickle.load(fp)

    file_name = "{}/act_hmrds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_hmrds_uncrossed = pickle.load(fp)

    file_name = "{}/act_crds_crossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_crossed = pickle.load(fp)

    file_name = "{}/act_crds_uncrossed_binolayers_dotDens_{:.1f}.pkl".format(
        SAVE_PATH, dotDens
    )
    with open(file_name, "rb") as fp:
        act_crds_uncrossed = pickle.load(fp)

    return (
        act_ards_crossed,
        act_ards_uncrossed,
        act_hmrds_crossed,
        act_hmrds_uncrossed,
        act_crds_crossed,
        act_crds_uncrossed,
    )


def svm_single_layer(layer_names, dotDens_list, layer):
    layer_name = layer_names[layer]
    score_ards = np.zeros(len(dotDens_list), dtype=np.float32)
    score_hmrds = np.zeros(len(dotDens_list), dtype=np.float32)
    for d in range(len(dotDens_list)):
        dotDens = dotDens_list[d]

        print("svm, dotDens: {:.1f}, layer: {}".format(dotDens, layer_name))

        (
            act_ards_crossed,
            act_ards_uncrossed,
            act_hmrds_crossed,
            act_hmrds_uncrossed,
            act_crds_crossed,
            act_crds_uncrossed,
        ) = load_binolayer_activation(dotDens)

        ## compute the svm
        # build training dataset (using crds)

        if layer_name != "final":
            temp = act_crds_crossed[layer_name]
            n_rds_each_disp = temp.shape[0]
            x_crossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            temp = act_crds_uncrossed[layer_name]
            x_uncrossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            X_train = np.concatenate([x_crossed, x_uncrossed])
            x_mean = X_train.mean()
            x_std = X_train.std()
            X_train = (X_train - x_mean) / x_std
            Y_train = np.concatenate([np.zeros(temp.shape[0]), np.ones(temp.shape[0])])

            # build test dataset
            # ards
            temp = act_ards_crossed[layer_name]
            x_crossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            temp = act_ards_uncrossed[layer_name]
            x_uncrossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            X_ards = np.concatenate([x_crossed, x_uncrossed])
            X_ards = (X_ards - x_mean) / x_std
            Y_ards = np.concatenate(
                [np.zeros(n_rds_each_disp), np.ones(n_rds_each_disp)]
            )

            # hmrds
            temp = act_hmrds_crossed[layer_name]
            x_crossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            temp = act_hmrds_uncrossed[layer_name]
            x_uncrossed = temp.mean(axis=1).reshape(
                n_rds_each_disp, -1
            )  # average across features and flatten
            X_hmrds = np.concatenate([x_crossed, x_uncrossed])
            X_hmrds = (X_hmrds - x_mean) / x_std
            Y_hmrds = np.concatenate(
                [np.zeros(n_rds_each_disp), np.ones(n_rds_each_disp)]
            )

        else:  # in the final layer, no need averaging across features
            temp = act_crds_crossed[layer_name]
            n_rds_each_disp = temp.shape[0]
            x_crossed = temp.reshape(n_rds_each_disp, -1)
            temp = act_crds_uncrossed[layer_name]
            x_uncrossed = temp.reshape(n_rds_each_disp, -1)
            X_train = np.concatenate([x_crossed, x_uncrossed])
            x_mean = X_train.mean()
            x_std = X_train.std()
            X_train = (X_train - x_mean) / x_std
            Y_train = np.concatenate([np.zeros(temp.shape[0]), np.ones(temp.shape[0])])

            # build test dataset
            # ards
            temp = act_ards_crossed[layer_name]
            x_crossed = temp.reshape(n_rds_each_disp, -1)
            temp = act_ards_uncrossed[layer_name]
            x_uncrossed = temp.reshape(n_rds_each_disp, -1)
            X_ards = np.concatenate([x_crossed, x_uncrossed])
            X_ards = (X_ards - x_mean) / x_std
            Y_ards = np.concatenate(
                [np.zeros(n_rds_each_disp), np.ones(n_rds_each_disp)]
            )

            # hmrds
            temp = act_hmrds_crossed[layer_name]
            x_crossed = temp.reshape(n_rds_each_disp, -1)
            temp = act_hmrds_uncrossed[layer_name]
            x_uncrossed = temp.reshape(n_rds_each_disp, -1)
            X_hmrds = np.concatenate([x_crossed, x_uncrossed])
            X_hmrds = (X_hmrds - x_mean) / x_std
            Y_hmrds = np.concatenate(
                [np.zeros(n_rds_each_disp), np.ones(n_rds_each_disp)]
            )

        ## classifying rds with SVM

        clf = SVC(kernel="linear")
        clf.fit(X_train, Y_train)
        # print(clf.score(X_train, Y_train))

        score_ards[d] = clf.score(X_ards, Y_ards)
        score_hmrds[d] = clf.score(X_hmrds, Y_hmrds)

    return score_ards, score_hmrds


def compute_svm(dotDens_list):
    svm_layers = []
    svm_layers.append(
        Parallel(n_jobs=2)(
            delayed(svm_single_layer)(binolayer_names, dotDens_list, layer)
            for layer in range(n_binolayers)
        )
    )

    # unpacking
    score_ards = np.zeros((n_binolayers, len(dotDens_list)), dtype=np.float32)
    score_hmrds = np.zeros((n_binolayers, len(dotDens_list)), dtype=np.float32)
    for layer in range(n_binolayers):
        temp = svm_layers[0][layer]

        score_ards[layer] = temp[0]
        score_hmrds[layer] = temp[1]

    # save files
    np.save("{}/score_ards.npy".format(SAVE_PATH), score_ards)
    np.save("{}/score_hmrds.npy".format(SAVE_PATH), score_hmrds)


def plotHeat_svm(score_ards, score_hmrds, save_flag):
    # %% start plotting
    sns.set_theme()
    sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

    # estimate v_min and v_max for cbar
    v_min = 0
    v_max = 1.0
    # v_min = np.round(np.min(rdm_reconstruct), 2)
    # v_max = np.round(np.max(rdm_reconstruct), 2)

    figsize = (15, 8)
    n_row = 1
    n_col = 2

    fig, axes = plt.subplots(
        nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
    )

    fig.text(
        0.5,
        1.02,
        "SVM for each binolayers in GC-Net",
        ha="center",
    )
    fig.text(-0.06, 0.5, "Layers", va="center", rotation=90)
    fig.text(0.5, -0.05, "Dot density", ha="center")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    cmap = "jet"

    x = np.round(dotDens_list, 1)
    sns.heatmap(
        score_ards,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
        xticklabels=x,
        yticklabels=binolayer_names,
        ax=axes[0],
    )
    axes[0].set_title("aRDS")

    sns.heatmap(
        score_hmrds,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
        xticklabels=x,
        yticklabels=binolayer_names,
        ax=axes[1],
    )
    axes[1].set_title("hmRDS")

    # %% ## save
    if save_flag == 1:
        fig.savefig(
            "{}/plotHeat_svm_all_layers.pdf".format(SAVE_PATH),
            dpi=600,
            bbox_inches="tight",
        )
