# %% load necessary modules
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utilities.misc import NestedTensor

# %%


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

    def forward(self, x: Tensor):

        out = x + F.relu(
            self.conv2d_block2(self.conv2d_block1(x))
        )  # pre-activation connection

        return out


def build_resBlock(
    in_channels: int,
    out_channels: int,
    n_resBlocks: int,
    stride: int,
):
    strides = [stride] + [1] * (n_resBlocks - 1)
    layers = []
    for stride in strides:
        layers.append(ResidualBlock(in_channels, out_channels, stride))

    return nn.Sequential(*layers)


class StereoEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        #### feature extractor layers, layer 1-18 ####
        ## layer 1, [b, c, h/2, w/2]
        # [b, 32, 128, 256]
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.base_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),  # bias=False when use BatchNorm
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        ## build residual block, layer 2-17
        # [b, c, h/2, w/2]
        # [b, 32, 128, 256]
        self.res_block = build_resBlock(
            in_channels=config.base_channels,
            out_channels=config.base_channels,
            n_resBlocks=config.n_resBlocks,
            stride=1,
        )

        ## layer 18
        # [b, c, h/2, w/2]
        # [b, 32, 128, 256]
        self.layer18 = nn.Conv2d(
            config.base_channels,
            config.base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        #### end of feature extractor layers ####

    def forward(self, x: NestedTensor):

        feat_left = self.layer18(self.res_block(self.layer1(x.left)))
        feat_right = self.layer18(self.res_block(self.layer1(x.right)))

        return feat_left, feat_right


def build_encoder(config):

    return StereoEncoder(config)
