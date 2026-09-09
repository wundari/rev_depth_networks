# %% import necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# %%


class StereoDecoder(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.img_height = config.img_height
        self.img_width = config.img_width
        self.max_disp = config.max_disp
        self.batch_size = config.batch_size
        self.binocular_interaction = config.binocular_interaction

        #### layers for processing cost volume
        # input: cost_vol -> concatenate left and right features
        # output_dim: [b, c, max_disp/2, h/2, w/2]
        # output_dim: [b, 32, 96, 128, 256]
        self.layer19 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 20
        # input layer: layer 19
        # output dim: [b, c, max_disp/2, h/2, h/2]
        # output dim: [b, 32, 96, 128, 256]
        self.layer20 = nn.Sequential(
            nn.Conv3d(
                config.base_channels,
                config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 21
        # input layer: cost_vol
        # output dim: [b, 2xc, max_disp/4, h/4, h/4]
        # output dim: [b, 64, 48, 64, 128]
        self.layer21 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 22
        # input layer: layer 21
        # output dim: [b, 2xc, max_disp/4, h/4, w/4]
        # output dim: [b, 64, 48, 64, 128]
        self.layer22 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 23
        # input layer: layer 22
        # output dim: [b, 2xc, max_disp/4, h/4, w/4]
        # output dim: [b, 64, 48, 64, 128]
        self.layer23 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 24
        # input layer: layer 21
        # [b, 2xc, max_disp/8, h/8, w/8]
        # [b, 64, 24, 32, 64]
        self.layer24 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 25
        # input layer: layer 24
        # [b, 2xc, max_disp/8, h/8, w/8]
        # [b, 64, 24, 32, 64]
        self.layer25 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 26
        # input layer: layer 25
        # [b, 2xc, max_disp/8, h/8, w/8]
        # [b, 64, 24, 32, 64]
        self.layer26 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 27
        # input layer: layer 24
        # [b, 2xc, max_disp/16, h/16, w/16]
        # [b, 64, 12, 16, 32]
        self.layer27 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 28
        # input layer: layer 27
        # [b, 2xc, max_disp/16, h/16, w/16]
        # [b, 64, 12, 16, 32]
        self.layer28 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 29
        # input layer: layer 28
        # [b, 2xc, max_disp/16, h/16, w/16]
        # [b, 64, 12, 16, 32]
        self.layer29 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                2 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 30
        # input layer: layer 27
        # [b, 2x2xc, max_disp/32, h/32, w/32]
        # [b, 128, 6, 8, 16]
        self.layer30 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                4 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(4 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 31
        # input layer: layer 30
        # [b, 2*2xc, max_disp/32, h/32, w/32]
        # [b, 128, 6, 8, 16]
        self.layer31 = nn.Sequential(
            nn.Conv3d(
                4 * config.base_channels,
                4 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(4 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 32
        # input layer: layer 31
        # [b, 2*2xc, max_disp/32, h/32, w/32]
        # [b, 128, 6, 8, 16]
        self.layer32 = nn.Sequential(
            nn.Conv3d(
                4 * config.base_channels,
                4 * config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(4 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        #### end of layers for encoders ####

        #### layers for deconvolution 3D ####
        ## layer 33a
        # input layer: layer 32
        # output dim: [b, 2x2xc, max_disp/16, h/16, w/16]
        # output dim: [b, 64, 12, 16, 32]
        self.layer33a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=4 * config.base_channels,
                out_channels=2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 34a
        # input layer: layer 33b
        # [b, 2xc, max_disp/8, h/8, w/8]
        # [b, 64, 24, 32, 64]
        self.layer34a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=2 * config.base_channels,
                out_channels=2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 35a
        # input layer: layer 34b
        # [b, 2xc, max_disp/4, h/4, w/4]
        # [b, 64, 48, 64, 128]
        self.layer35a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=2 * config.base_channels,
                out_channels=2 * config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(2 * config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 36a
        # input layer: layer 35b
        # [b, c, max_disp/2, h/2, w/2]
        # [b, 32, 96, 128, 256]
        self.layer36a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=2 * config.base_channels,
                out_channels=config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## layer 37
        # input layer: layer 36b
        # [b, 1, max_disp, h, w]
        # [b, 1, 192, 256, 512]
        self.layer37 = nn.ConvTranspose3d(
            in_channels=config.base_channels,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    # def build_costVol(self, feat_left: Tensor, feat_right: Tensor):
    #     # def build_costVol(self, feat_right: Tensor, feat_left: Tensor):
    #     """
    #     construct disparity cost volume 4D tensor
    #     input_left and input_right are concatenated along the feature axis

    #      Args:
    #          input_left ([b, c, h, w] tensor):
    #              input left
    #          input_right ([b, c, h, w] tensor):
    #              input right
    #     """

    #     b, c, h, w = feat_left.size()

    #     D = self.max_disp // 2
    #     # padding the left and right side of the feat_left with zeros
    #     padded_left = F.pad(feat_left, (D // 2, D // 2))
    #     # [b=2, c=32, h=128, w + disp=352]

    #     # concatenate input_right along the feature axis with the input_left
    #     cost_vol_list = []
    #     for d in range(D):
    #         # padding the left and right side of the input_right with zeros
    #         padded_right = F.pad(feat_right, (d, D - d))
    #         # [b=2, c=32, h=128, (d|w|D-d)=352]

    #         # concatenate along the feature axis
    #         temp = torch.cat(
    #             (padded_left, padded_right), dim=1
    #         )  # [b=2, 2xc=64, h=128, (d|w|D-d)=352]
    #         cost_vol_list.append(temp)

    #     # merge all along the feature axis,
    #     # [b, D x 2 x c, h, w+D]
    #     costVol = torch.cat(cost_vol_list, dim=1)

    #     # reshape, [b, D, 2 x c, h, w+D]
    #     costVol = costVol.view(
    #         b,
    #         D,
    #         2 * c,
    #         h,
    #         w + D,
    #     )

    #     # swap axis, [b, 2xc, D, h, w+D]
    #     costVol = costVol.permute(0, 2, 1, 3, 4)

    #     # crop the image, [b, 2xc, D, h, w]
    #     costVol = costVol[:, :, :, :, D // 2 : w + D // 2]

    #     return costVol

    def build_costVol(self, feat_left: Tensor, feat_right: Tensor):
        if feat_left.shape != feat_right.shape:
            raise ValueError("Left/right feature shapes must match")
        b, c, h, w = feat_left.shape
        D = self.max_disp // 2
        volumes = []
        for d in range(D):
            right = F.pad(feat_right, (d, D - d))[..., D // 2:D // 2 + w]
            left = feat_left
            if self.binocular_interaction == "default":
                channels = (left, right)
            elif self.binocular_interaction == "bem":
                channels = (left * right / c, (left.square() + right.square()) / c)
            elif self.binocular_interaction == "cmm":
                product = left * right / c
                channels = (product, F.relu(product))
            elif self.binocular_interaction == "sum_diff":
                channels = (left + right, left - right)
            else:
                raise ValueError(f"Unknown interaction: {self.binocular_interaction}")
            volumes.append(torch.cat(channels, dim=1))
        return torch.stack(volumes, dim=2)

    def forward(self, feat_left: Tensor, feat_right: Tensor):
        # input_left: [n_batch, n_features, img_height, img_width]
        # [n_batch, n_features, 256, 512]

        # build cost volume
        # concatenate the left and right features to form the cost volume tensor
        # [n_batch, n_features_left_and_right, max_disp, img_height/2, img_width/2]
        # [n_batch, 2*32, 192, 128, 256]
        costVol = self.build_costVol(feat_left, feat_right)

        ## layer 19
        # input layer: cost_vol
        # output dim: [n_batch, n_features, max_disp/2, h/2, w/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        out_19 = self.layer19(costVol)

        ## layer 20
        # input layer: layer 19
        # output dim: [n_batch, n_features, max_disp/2, h/2, w/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        out_20 = self.layer20(out_19)

        ## layer 21
        # input layer: cost_vol
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, h/4, w/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_21 = self.layer21(costVol)

        ## layer 22
        # input layer: layer 21
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, h/4, w/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_22 = self.layer22(out_21)

        ## layer 23
        # input layer: layer 22
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, h/4, w/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_23 = self.layer23(out_22)

        ## layer 24
        # input layer: layer 21
        # [n_batch, n_features_left_and_right, max_disp/8, h/8, w/8]
        # [n_batch, 64, 24, 32, 64]
        out_24 = self.layer24(out_21)

        ## layer 25
        # input layer: layer 24
        # [n_batch, n_features_left_and_right, max_disp/8, h/8, w/8]
        # [n_batch, 64, 24, 32, 64]
        out_25 = self.layer25(out_24)

        ## layer 26
        # input layer: layer 25
        # [n_batch, n_features_left_and_right, max_disp/8, h/8, w/8]
        # [n_batch, 64, 24, 32, 64]
        out_26 = self.layer26(out_25)

        ## layer 27
        # input layer: 24
        # [n_batch, n_features_left_and_right, max_disp/16, h/16, w/16]
        # [n_batch, 64, 12, 16, 32]
        out_27 = self.layer27(out_24)

        ## layer 28
        # input layer: layer 27
        # [n_batch, n_features_left_and_right, max_disp/16, h/16, w/16]
        # [n_batch, 64, 12, 16, 32]
        out_28 = self.layer28(out_27)

        ## layer 29
        # input layer: layer 28
        # [n_batch, n_features_left_and_right, max_disp/16, h/16, w/16]
        # [n_batch, 64, 12, 16, 32]
        out_29 = self.layer29(out_28)

        ## layer 30
        # input layer: layer 27
        # [n_batch, 2*n_features_left_and_right, max_disp/32, h/32, w/32]
        # [n_batch, 128, 6, 8, 16]
        out_30 = self.layer30(out_27)

        ## layer 31
        # input layer: layer 30
        # [n_batch, 2*n_features_left_and_right, max_disp/32, h/32, w/32]
        # [n_batch, 128, 6, 8, 16]
        out_31 = self.layer31(out_30)

        ## layer 32
        # input layer: layer 31
        # [n_batch, 2*n_features_left_and_right, max_disp/32, h/32, w/32]
        # [n_batch, 128, 6, 8, 16]
        out_32 = self.layer32(out_31)

        ## layer 33a
        # input layer: layer 32
        # output dim: [n_batch, n_features_left_and_right, max_disp/16, h/16, w/16]
        # output dim: [n_batch, 64, 12, 16, 32]
        out_33a = self.layer33a(out_32)

        ## layer 33b
        # input layer: layer 33a + layer 29 (residual connection)
        # output dim: [n_batch, n_features_left_and_right, max_disp/16, h/16, w/16]
        # output dim: [n_batch, 64, 12, 16, 32]
        # out_33b = F.relu(out_33a + out_29)
        out_33b = out_33a + out_29

        ## layer 34a
        # input layer: layer 33b
        # [n_batch, n_features_left_and_right, max_disp/8, h/8, w/8]
        # [n_batch, 64, 24, 32, 64]
        out_34a = self.layer34a(out_33b)

        ## layer 34b
        # input layer: layer 34a + layer 26 (residual connection)
        # output dim: [n_batch, n_features_left_and_right, max_disp/8, h/8, w/8]
        # output dim: [n_batch, 64, 24, 32, 64]
        # out_34b = F.relu(out_34a + out_26)
        out_34b = out_34a + out_26

        ## layer 35a
        # input layer: layer 34b
        # [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # [n_batch, 64, 48, 64, 128]
        out_35a = self.layer35a(out_34b)

        ## layer 35b
        # input layer: layer 35a + layer 23 (residual connection)
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        # out_35b = F.relu(out_35a + out_23)
        out_35b = out_35a + out_23

        ## layer 36a
        # input layer: layer 35b
        # [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # [n_batch, 32, 96, 128, 256]
        out_36a = self.layer36a(out_35b)

        ## layer 36b
        # input layer: layer 36a + layer 20 (residual connection)
        # output dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        # out_36b = F.relu(out_36a + out_20)
        out_36b = out_36a + out_20

        ## layer 37
        # input layer: layer 36b
        # [n_batch, n_features, max_disp, img_height, img_width]
        # [n_batch, 1, 192, 256, 512]
        out_37 = self.layer37(out_36b).squeeze(1)

        # squeeze
        # [n_batch, max_disp, img_height, img_width]
        # [n_batch, 192, 256, 512]
        # out = out_37.view(
        #     self.batch_size, self.max_disp, self.img_height, self.img_width
        # )

        # out_37 *= self.max_disp**-0.5  # for numerical stability

        # compute probability
        # [n_batch, max_disp, img_height, img_width]
        # [n_batch, 192, 256, 512]
        logits = F.softmax(-out_37, dim=1, dtype=torch.float32)

        return logits


def build_decoder(config):
    return StereoDecoder(config)
