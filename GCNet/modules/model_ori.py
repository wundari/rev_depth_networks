"""
File: /media/wundari/WD_Black/Learning_deeplearning/gc-net/GC_Net_v2.py
Project: /media/wundari/WD_Black/Learning_deeplearning/gc-net
Created Date: 2023-03-02 17:02:06
Author: Bayu G. Wundari
-----
Last Modified: 2023-03-07 14:57:39
Modified By: Bayu G. Wundari
-----
Copyright (c) 2023 National Institute of Information and Communications Technology (NICT)

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
"""

# %% load necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

# %%
# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # conv block 1
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # conv block 2
        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # # get the residual connection
        # residual = x

        # # conv block 1
        # out = self.conv2d_block1(x)

        # # conv block 2
        # out = self.conv2d_block2(out)

        # # add the residual connection (the input)
        # out += residual

        # out = F.relu(x + self.conv2d_block2(self.conv2d_block1(x)))
        out = x + F.relu(
            self.conv2d_block2(self.conv2d_block1(x))
        )  # pre-activation connection

        # # final output
        # out = F.relu(out)

        return out


class GC_Net(nn.Module):
    def __init__(
        self,
        ResBlock,
        n_resBlocks,  # 8
        img_height,
        img_width,
        max_disp,
    ):
        super().__init__()

        self.block = ResBlock
        self.n_resBlocks = n_resBlocks

        self.img_height = img_height
        self.img_width = img_width
        self.max_disp = max_disp

        #### feature extractor layers, layer 1-18 ####
        ## layer 1, [n_batch, n_features, img_height/2, img_width/2]
        # [n_batch, 32, 128, 256]
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=5, stride=2, padding=2, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        ## residual block, layer 2-17
        # [n_batch, n_features, img_height/2, img_width/2]
        # [n_batch, 32, 128, 256]
        self.res_block = self.make_layer(
            self.block,
            in_channels=32,
            out_channels=32,
            n_blocks=self.n_resBlocks,
            stride=1,
        )

        ## layer 18
        # [n_batch, n_features, img_height/2, img_width/2]
        # [n_batch, 32, 128, 256]
        self.layer18 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        #### end of feature extractor layers ####

        #### layers for processing cost volume
        # input: cost_vol -> concatenate left and right features
        # output_dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # output_dim: [n_batch, 32, 96, 128, 256]
        self.layer19 = nn.Sequential(
            nn.Conv3d(
                64, 32, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        ## layer 20
        # input layer: layer 19
        # output dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        self.layer20 = nn.Sequential(
            nn.Conv3d(
                32, 32, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        ## layer 21
        # input layer: cost_vol
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        self.layer21 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 22
        # input layer: layer 21
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        self.layer22 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 23
        # input layer: layer 22
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        self.layer23 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 24
        # input layer: layer 21
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        self.layer24 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 25
        # input layer: layer 24
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        self.layer25 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 26
        # input layer: layer 25
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        self.layer26 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 27
        # input layer: layer 24
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        self.layer27 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 28
        # input layer: layer 27
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        self.layer28 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 29
        # input layer: layer 28
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        self.layer29 = nn.Sequential(
            nn.Conv3d(
                64, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 30
        # input layer: layer 27
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        self.layer30 = nn.Sequential(
            nn.Conv3d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        ## layer 31
        # input layer: layer 30
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        self.layer31 = nn.Sequential(
            nn.Conv3d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        ## layer 32
        # input layer: layer 31
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        self.layer32 = nn.Sequential(
            nn.Conv3d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False
            ),  # bias=False when use BatchNorm
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        #### end of layers for encoders ####

        #### layers for deconvolution 3D ####
        ## layer 33a
        # input layer: layer 32
        # output dim: [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # output dim: [n_batch, 64, 12, 16, 32]
        self.layer33a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 34a
        # input layer: layer 33b
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        self.layer34a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 35a
        # input layer: layer 34b
        # [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # [n_batch, 64, 48, 64, 128]
        self.layer35a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        ## layer 36a
        # input layer: layer 35b
        # [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # [n_batch, 32, 96, 128, 256]
        self.layer36a = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        ## layer 37
        # input layer: layer 36b
        # [n_batch, n_features, max_disp, img_height, img_width]
        # [n_batch, 1, 192, 256, 512]
        self.layer37 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, input_left, input_right):
        # input_left: [n_batch, n_features, img_height, img_width]
        # [n_batch, n_features, 256, 512]

        ## layer 1
        # input layer: layer_left and layer_right
        # output dim: [n_batch, n_features, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 128, 256]
        out_left = self.layer1(input_left)
        out_right = self.layer1(input_right)

        ## layer resBlock, layer 2-17
        # input layer: layer 1
        # output dim: [n_batch, n_features, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 128, 256]
        out_left = self.res_block(out_left)
        out_right = self.res_block(out_right)

        ## layer 18
        # input layer: layer resBlock
        # output dim: [n_batch, n_features, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 128, 256]
        out_left = self.layer18(out_left)
        out_right = self.layer18(out_right)

        # concatenate the left and right features to form the cost volume tensor
        # [n_batch, n_features_left_and_right, max_disp, img_height/2, img_width/2]
        # [n_batch, 2*32, 192, 128, 256]
        cost_vol = self.cost_volume(out_left, out_right)

        ## layer 19
        # input layer: cost_vol
        # output dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        out_19 = self.layer19(cost_vol)

        ## layer 20
        # input layer: layer 19
        # output dim: [n_batch, n_features, max_disp/2, img_height/2, img_width/2]
        # output dim: [n_batch, 32, 96, 128, 256]
        out_20 = self.layer20(out_19)

        ## layer 21
        # input layer: cost_vol
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_21 = self.layer21(cost_vol)

        ## layer 22
        # input layer: layer 21
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_22 = self.layer22(out_21)

        ## layer 23
        # input layer: layer 22
        # output dim: [n_batch, n_features_left_and_right, max_disp/4, img_height/4, img_width/4]
        # output dim: [n_batch, 64, 48, 64, 128]
        out_23 = self.layer23(out_22)

        ## layer 24
        # input layer: layer 21
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        out_24 = self.layer24(out_21)

        ## layer 25
        # input layer: layer 24
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        out_25 = self.layer25(out_24)

        ## layer 26
        # input layer: layer 25
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        out_26 = self.layer26(out_25)

        ## layer 27
        # input layer: 24
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        out_27 = self.layer27(out_24)

        ## layer 28
        # input layer: layer 27
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        out_28 = self.layer28(out_27)

        ## layer 29
        # input layer: layer 28
        # [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # [n_batch, 64, 12, 16, 32]
        out_29 = self.layer29(out_28)

        ## layer 30
        # input layer: layer 27
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        out_30 = self.layer30(out_27)

        ## layer 31
        # input layer: layer 30
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        out_31 = self.layer31(out_30)

        ## layer 32
        # input layer: layer 31
        # [n_batch, 2*n_features_left_and_right, max_disp/32, img_height/32, img_width/32]
        # [n_batch, 128, 6, 8, 16]
        out_32 = self.layer32(out_31)

        ## layer 33a
        # input layer: layer 32
        # output dim: [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # output dim: [n_batch, 64, 12, 16, 32]
        out_33a = self.layer33a(out_32)

        ## layer 33b
        # input layer: layer 33a + layer 29 (residual connection)
        # output dim: [n_batch, n_features_left_and_right, max_disp/16, img_height/16, img_width/16]
        # output dim: [n_batch, 64, 12, 16, 32]
        # out_33b = F.relu(out_33a + out_29)
        out_33b = out_33a + out_29

        ## layer 34a
        # input layer: layer 33b
        # [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
        # [n_batch, 64, 24, 32, 64]
        out_34a = self.layer34a(out_33b)

        ## layer 34b
        # input layer: layer 34a + layer 26 (residual connection)
        # output dim: [n_batch, n_features_left_and_right, max_disp/8, img_height/8, img_width/8]
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
        out_37 = self.layer37(out_36b)

        # squeeze
        # [n_batch, max_disp, img_height, img_width]
        # [n_batch, 192, 256, 512]
        out = out_37.view(len(out_37), self.max_disp, self.img_height, self.img_width)

        # compute probability
        # [n_batch, max_disp, img_height, img_width]
        # [n_batch, 192, 256, 512]
        prob = F.softmax(-out, 1)

        return prob

    def make_layer(self, block, in_channels, out_channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))

        return nn.Sequential(*layers)

    # def cost_volume(self, input_left, input_right):
    #     """
    #     input_left and input_right are concatenated along the feature axis

    #      Args:
    #          input_left ([n_batch, n_feature, h, w] tensor):
    #              input left
    #          input_right ([n_batch, n_feature, h, w] tensor):
    #              input right
    #     """

    #     disp = self.max_disp // 2
    #     # padding the right side of the input_left with zeros
    #     pad_operator = nn.ZeroPad2d((0, disp, 0, 0))  # [left, right, top, bottom]
    #     padded_left = pad_operator(
    #         input_left
    #     )  # [n_batch=2, n_features=32, h/2=128, w/2 + disp=352]

    #     # concatenate input_right along the feature axis with the input_left
    #     cost_vol_list = []
    #     for d in range(disp):
    #         # padding the left and right side of the input_right with zeros
    #         pad_operator = nn.ZeroPad2d(
    #             (d, disp - d, 0, 0)
    #         )  # [left, right, top, bottom]
    #         padded_right = pad_operator(
    #             input_right
    #         )  # [n_batch=2, n_features=32, h/2=128, w/2 + disp - d=352]

    #         # concatenate along the feature axis
    #         temp = torch.cat(
    #             (padded_left, padded_right), dim=1
    #         )  # [n_batch=2, 2 * n_featues=64, h/2=128, w/2 + disp - d=352]
    #         cost_vol_list.append(temp)

    #     # merge all along the feature axis,
    #     # [n_batch, disp*2*n_features_left_right, h/2, w/2+disp]
    #     cost_vol = torch.cat(cost_vol_list, dim=1)

    #     # reshape, [n_batch, disp, n_features_left_right, h/2, w/2+disp]
    #     cost_vol = cost_vol.view(
    #         len(cost_vol),
    #         disp,
    #         64,
    #         self.img_height // 2,
    #         self.img_width // 2 + disp,
    #     )

    #     # change axis, [n_batch, n_features_left_right, disp, h/2, w/2+disp]
    #     cost_vol = cost_vol.permute(0, 2, 1, 3, 4)

    #     # crop the image, [n_batch, n_features_left_right, disp, h/2, w/2]
    #     cost_vol = cost_vol[:, :, :, :, : self.img_width // 2]

    #     return cost_vol

    def cost_volume(self, input_left, input_right):
        """
        input_left and input_right are concatenated along the feature axis

         Args:
             input_left ([n_batch, n_feature, h, w] tensor):
                 input left
             input_right ([n_batch, n_feature, h, w] tensor):
                 input right
        """

        disp = self.max_disp // 2
        # padding the left and right side of the input_left with zeros
        pad_operator = nn.ZeroPad2d(
            (disp // 2, disp // 2, 0, 0)
        )  # [left, right, top, bottom]
        padded_left = pad_operator(
            input_left
        )  # [n_batch=2, n_features=32, h/2=128, w/2 + disp=352]

        # concatenate input_right along the feature axis with the input_left
        cost_vol_list = []
        for d in range(disp):
            # padding the left and right side of the input_right with zeros
            pad_operator = nn.ZeroPad2d(
                (d, disp - d, 0, 0)
            )  # [left, right, top, bottom]
            padded_right = pad_operator(
                input_right
            )  # [n_batch=2, n_features=32, h/2=128, w/2 + disp - d=352]

            # concatenate along the feature axis
            temp = torch.cat(
                (padded_left, padded_right), dim=1
            )  # [n_batch=2, 2 * n_features=64, h/2=128, w/2 + disp - d=352]
            cost_vol_list.append(temp)

        # merge all along the feature axis,
        # [n_batch, disp*2*n_features_left_right, h/2, w/2+disp]
        cost_vol = torch.cat(cost_vol_list, dim=1)

        # reshape, [n_batch, disp, n_features_left_right, h/2, w/2+disp]
        cost_vol = cost_vol.view(
            len(cost_vol),
            disp,
            64,
            self.img_height // 2,
            self.img_width // 2 + disp,
        )

        # change axis, [n_batch, n_features_left_right, disp, h/2, w/2+disp]
        cost_vol = cost_vol.permute(0, 2, 1, 3, 4)

        # crop the image, [n_batch, n_features_left_right, disp, h/2, w/2]
        cost_vol = cost_vol[:, :, :, :, disp // 2 : self.img_width // 2 + disp // 2]

        return cost_vol

    def predict(self, input_left, input_right):
        n_batches = len(input_left)
        loss_mul_list_test = []
        for d in range(self.max_disp):
            loss_mul_temp = Variable(
                torch.Tensor(
                    np.ones([n_batches, 1, self.img_height, self.img_width]) * d
                )
            ).to(device)
            loss_mul_list_test.append(loss_mul_temp)
        # [n_batch, n_features, max_disp, img_height, img_width]
        # [n_batch, 1, 160, 256, 512]
        loss_mul_test = torch.cat(loss_mul_list_test, 1)

        prob_vol = self.forward(input_left, input_right)

        # compute the expected value of disparity (eq. 1 in the paper)
        predict_disp = torch.sum(prob_vol.mul(loss_mul_test), 1)

        return predict_disp


def GCNet(img_height, img_width, max_disp):
    return GC_Net(
        ResidualBlock,
        8,
        img_height,
        img_width,
        max_disp,
    )
