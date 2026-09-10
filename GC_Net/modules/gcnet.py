# load necessary modules
import torch
import torch.nn as nn

from GC_Net.modules.stereo_encoder_relu import build_encoder
from GC_Net.modules.stereo_decoder_relu import build_decoder

from utilities.misc import NestedTensor


# %%
class GCNet(nn.Module):

    def __init__(self, config):

        super().__init__()
        if any(
            value <= 0 or value % 32
            for value in (config.max_disp, config.img_height, config.img_width)
        ):
            raise ValueError(
                "GCNet disparity range and crop dimensions must be positive multiples of 32"
            )
        self.max_disp = config.max_disp

        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

        # Disparity range tensor, for computing disparity map
        self.register_buffer(
            "disp_indices",
            torch.arange(-self.max_disp // 2, self.max_disp // 2).view(1, -1, 1, 1),
        )
        # self.register_buffer(
        #     "disp_indices",
        #     torch.arange(self.max_disp // 2, -self.max_disp // 2, -1).view(1, -1, 1, 1),
        # ) # this indices make the network overfit the training data, thus reversed doesn't occur

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()
        self._gelu_approximate()

    def _reset_parameters(self):
        """
        xavier initialize all params
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
        disable Batchnorm tracking stats to reduce dependency on dataset
        (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def _gelu_approximate(self):
        """
        make all GELU approximate
        """
        for m in self.modules():
            if isinstance(m, nn.GELU):
                m.approximate = "tanh"

    def forward(self, x: NestedTensor):

        # extract features
        feat_left, feat_right = self.encoder(x)

        # compute disparity map
        # p_swap_left2right = random.random()  # determine the direction of disparity
        # if p_swap_left2right <= 0.5:
        #     # use disp_left as ground truth,
        #     # use img_left as a reference, enhance negative disparity
        #     logits = self.decoder(feat_left, feat_right)
        #     disp_pred = torch.sum(logits * self.disp_indices, dim=1)
        # else:
        #     # swap left and right inputs
        #     # use disp_right as ground truth
        #     # use img_right as a reference, enhance positive disparity
        #     logits = self.decoder(feat_right, feat_left)
        #     disp_pred = torch.sum(logits * self.disp_indices * -1, dim=1)

        logits = self.decoder(feat_left, feat_right)
        disp_pred = torch.sum(
            logits * self.disp_indices * x.ref.view(-1, 1, 1, 1), dim=1
        )

        # logits = self.decoder(feat_right, feat_left)
        # logits = self.decoder(feat_left, feat_right)
        # disp_pred = torch.sum(logits * self.disp_indices, dim=1)

        return disp_pred


def build_gcnet(config) -> GCNet:
    return GCNet(config)


# %% log: implemented disp_left and disp_map, using x.ref.view(-1, 1, 1, 1)
# bino_interaction: default
# monkaa-experiment28, c = 1.5
# monkaa-experiment29, c = 0.5
# monkaa-experiment31, c = 1
# monkaa-experiment42, c = 2 -> ards 0.1 too positive; hmrds 0.8-0.9 too reversed; otherwise awesome
#                           model_9 is the best so far
# monkaa-experiment43, c = 1.75
# monkaa-experiment44, c = 2.25 (batch_train=16, CinetCluster)
# monkaa-experiment45, c = 1.9
# monkaa-experiment46, c = 2.5
# monkaa-experiment48, c = 2 (batch train=16, CiNetCluster) -> almost perfect!!
# monkaa-experiment49, c = 2 (epoch7_iter15200 seems the final candidate, though not perfect)

# it seems that by introducing image augmentation, hmRDS's performance start
# to decline as dot density increases

# flying-experiment2, c = 1
# flying-experiment3, c = 1.5
# flying-experiment4, c = 0.5
# flying-experiment11, c = 2

# %% log: implemented disp_left and disp_map,
# using logits = self.decoder(feat_left, feat_right)
# disp_right is multiplied by -1
# monkaa-experiment35, c = 1.5, performance similar to flying-experiment 6
# monkaa-experiment36, c = 0.5
# monkaa-experiment37, c = 1
# monkaa-experiment32, c = 0
# monkaa-experiment38, c = 2 -> works, but ards is too reversed, hmrds not so declining
# monkaa-experiment39, c = 2.25 -> similar to experiment38
# monkaa-experiment40, c = 1.75
# monkaa-experiment41, c = 1.9

# flying-experiment5, c = 0.5
# flying-experiment6, c = 1.5 (batch_train=16, CinetCluster) -> seems promising
# flying-experiment7, c = 1 (batch_train=16, CinetCluster)

# not tested yet
# flying-experiment8, c = 2.25 (batch_train=16, CinetCluster)
# flying-experiment9, c = 2 (batch_train=16, CinetCluster)
# flying-experiment10, c = 1.9 (batch_train=16, CinetCluster)
