# %% load necessary modules
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# %%
class StereoDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.binocular_interaction = config.binocular_interaction

        self.max_disp = (
            config.max_disp // 2
        )  # Adjusted for downsampling in feature extractor
        # base_channels = config.base_channels

        D = self.max_disp // 2  # Total number of disparity levels
        self.D = D

        # Disparity range tensor, for computing disparity map
        self.register_buffer(
            "disp_indices",
            torch.arange(-self.max_disp, self.max_disp).view(1, -1, 1, 1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(
                2 * config.base_channels,
                config.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.ConvTranspose3d(
            config.base_channels,
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # elif config.binocular_interaction == "sum_diff":
        # bino-interaction: sum-diff channels
        # self.cost_aggregation = nn.Sequential(
        #     nn.Conv2d(
        #         config.base_channels * 2 * 2,
        #         config.base_channels * 2,
        #         kernel_size=3,
        #         padding=1,
        #     ),
        #     nn.BatchNorm2d(config.base_channels * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         config.base_channels * 2, config.base_channels, kernel_size=3, padding=1
        #     ),
        #     nn.BatchNorm2d(config.base_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(config.base_channels, 1, kernel_size=3, padding=1),
        # )

        # self.upsample = nn.ConvTranspose2d(
        #     self.max_disp,
        #     self.max_disp * 2,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     output_padding=1,
        # )

    def build_cost_volume(self, feat_left: Tensor, feat_right: Tensor):
        b, c, h, w = feat_left.size()
        D = self.max_disp

        # pad feat_left
        padded_left = F.pad(feat_left, (D // 2, D // 2))

        # Build cost volume
        cost_vol_list = []

        for d in range(D):

            padded_right = F.pad(feat_right, (d, D - d))

            if self.binocular_interaction == "bem":
                # bino-interaction: based on binocular energy model
                cost = torch.cat(
                    (
                        (padded_left**2 + padded_right**2) / c,
                        padded_left * padded_right / c,
                    ),
                    dim=1,
                )

            elif self.binocular_interaction == "sum_diff":
                # interaction: sum-diff channels
                cost = torch.cat(
                    (padded_left + padded_right, padded_left - padded_right), dim=1
                )

            elif self.binocular_interaction == "cmm":
                # interaction: xcorr and xmatch

                cost = torch.cat(
                    (
                        padded_left * padded_right / c,
                        torch.relu(padded_left * padded_right / c),
                    ),
                    dim=1,
                )

            else:
                # default
                cost = torch.cat((padded_left, padded_right), dim=1)

            cost_vol_list.append(cost)

            # crop to original size
            # cost = cost[:, :, :, D // 2 : w + D // 2]

            # # Aggregate cost
            # cost = self.cost_aggregation(cost)  # [b, 1, h, w]
            # cost = cost_aggregation(cost)  # [b, 1, h, w]
            # cost_volume.append(cost.squeeze(1))  # [b, h, w]

        # Stack cost volume
        # cost_vol_list = torch.stack(cost_vol_list, dim=1)  # [b, D, h, w]

        # # upsample
        # cost_vol_list = self.upsample(cost_vol_list)  # [b, 2*D, h, w]
        # # cost_volume2 = convT(cost_volume)

        # merge all along the feature axis
        # [b, D x 2 x c, h, w]
        costVol = torch.cat(cost_vol_list, dim=1)

        # reshape, [b, D, 2 x c, h, w]
        costVol = costVol.view(b, D, 2 * c, h, w + D)

        # swap axis [b, 2xc, D, h, w]
        costVol = costVol.permute(0, 2, 1, 3, 4)

        # crop the image [b, 2xc, D, h, w]
        costVol = costVol[:, :, :, :, D // 2 : w + D // 2]

        return costVol

    def forward(self, feat_left: Tensor, feat_right: Tensor):

        # construct cost volume
        cost_volume = self.build_cost_volume(feat_left, feat_right)

        # conv3d-convT3d
        out = self.layer4(self.layer3(cost_volume)).squeeze(1)

        # Apply softmax to get probability volume
        logits = F.softmax(-out, dim=1)  # [b, 2*D, h, w]

        # disparity regression
        # Compute disparity map
        # disparity_map = torch.sum(prob_volume * self.disp_range, dim=1)  # [b, h, w]

        return logits


def build_decoder(config):
    return StereoDecoder(config)
