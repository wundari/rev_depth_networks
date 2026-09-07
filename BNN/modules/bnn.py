# %% load necessary modules
import torch
from torch import nn

from modules.stereo_encoder import build_feat_extractor
from modules.stereo_decoder import build_decoder

from utilities.misc import NestedTensor


# %%
class BNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_disp = config.max_disp

        self.encoder = build_feat_extractor(config)
        self.decoder = build_decoder(config)

        # Disparity range tensor, for computing disparity map
        self.register_buffer(
            "disp_indices",
            torch.arange(-self.max_disp // 2, self.max_disp // 2).view(1, -1, 1, 1),
        )

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()

    def _reset_parameters(self):
        """
        xavier init
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable BatchNorm tracking stats to reduce dependency on dataset
        (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x: NestedTensor):

        # extract features
        feat_left = self.encoder(x.left)
        feat_right = self.encoder(x.right)

        # concatenate left and right features
        # feat = torch.cat([feat_left, feat_right], dim=1)
        # feat = feat_left * feat_right

        logits = self.decoder(feat_left, feat_right)

        # regress disparity
        out = torch.sum(logits * self.disp_indices * x.ref.view(-1, 1, 1, 1), dim=1)

        return out


def build_bnn(config):
    return BNN(config)
