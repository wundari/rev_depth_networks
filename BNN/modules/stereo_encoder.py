# %% load necessary module
import torch
from torch import nn, Tensor


# %%
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                config.base_channels,
                config.base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(config.base_channels * 2),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor):

        # concatenate left and right inputs along batch channel (dim=0)
        # out = torch.cat([x.left, x.right], dim=1)

        # out = self.in_conv(x)
        out = self.layer2(self.in_conv(x))

        return out


def build_feat_extractor(config):
    return FeatureExtractor(config)
