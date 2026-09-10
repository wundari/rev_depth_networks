# %%
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from engine.engine_base import EngineBase
from config.config import ConfigGCNet

import numpy as np
from jaxtyping import Float
import os

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem

from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utilities.misc import NestedTensor


# %%
class SuperpositionAnalysis(EngineBase):

    def __init__(self, config: ConfigGCNet, params_rds: dict):
        super().__init__(config)

        self.config = config

        # folders for superosition analysis
        self.superposition_dir = (
            f"{self.save_dir}/"
            + f"experiment_{self.config.experiment_id}/"
            + f"superposition_epoch_{self.config.epoch_to_load}"
            + f"_iter_{self.config.iter_to_load}_seed_{self.config.seed}"
        )
        if not os.path.exists(self.superposition_dir):
            os.makedirs(self.superposition_dir)

        self.layer_name = [
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
            "layer33a",
            "layer34a",
            "layer35a",
            "layer36a",
            "layer37",
        ]

        # target_layer, only convolutional layers
        self.target_layer = [
            self.model.decoder.layer19[0],
            self.model.decoder.layer20[0],
            self.model.decoder.layer21[0],
            self.model.decoder.layer22[0],
            self.model.decoder.layer23[0],
            self.model.decoder.layer24[0],
            self.model.decoder.layer25[0],
            self.model.decoder.layer26[0],
            self.model.decoder.layer27[0],
            self.model.decoder.layer28[0],
            self.model.decoder.layer29[0],
            self.model.decoder.layer30[0],
            self.model.decoder.layer31[0],
            self.model.decoder.layer32[0],
            self.model.decoder.layer33a[0],
            self.model.decoder.layer34a[0],
            self.model.decoder.layer35a[0],
            self.model.decoder.layer36a[0],
            self.model.decoder.layer37,
        ]

        # rds parameters
        self.target_disp = params_rds[
            "target_disp"
        ]  # RDS target disparity (pix) to be analyzed
        self.n_rds_each_disp = params_rds[
            "n_rds_each_disp"
        ]  # n_rds for each disparity magnitude in disp_ct_pix
        self.dotDens_list = params_rds["dotDens_list"]  # dot density
        self.rds_type = params_rds[
            "rds_type"
        ]  # ards: 0.0, crds: 1.0, hmrds: 0.5, urds: -1.0
        self.dotMatch_list = params_rds["dotMatch_list"]  # [0.0, 0.5, 1.0]  # dot match
        self.background_flag = params_rds["background_flag"]  # 1: with cRDS background
        self.pedestal_flag = params_rds[
            "pedestal_flag"
        ]  # 1: use pedestal to ensure rds disparity > 0
        self.batch_size_rds = params_rds[
            "batch_size_rds"
        ]  # batch size for RDS generation
        self.disp_ct_pix_list = [self.target_disp, -self.target_disp]

        self.device = config.device

    def get_conv_names_and_weights(self) -> list[Tensor]:

        names = []
        params = []
        for name, param in self.model.named_parameters():
            names.append(name)
            params.append(param)

        # get index for convolutional layers
        layer_idx = []
        for i in range(len(names)):
            if ".0.weight" in names[i] or "layer18.weight" in names[i]:
                layer_idx.append(i)

        # get convolutional layer names and weights
        conv_layer_names = []
        conv_layer_weights = []
        for i in layer_idx:
            conv_layer_names.append(names[i])
            conv_layer_weights.append(params[i])

        return (conv_layer_names, conv_layer_weights)

    def compute_superposition_index(self, w: Float[Tensor, "n_hidden n_feat"]):
        """
        Compute representation strength and superposition index.

        Representation strength: how strong a feature is represented

                ||Wi|| = sqrt(sum(Wi^2))
                where Wi is the i-th feature vector.
        Superposition index: how much a feature shares its dimension
                with other features.

                The idea is to sum the projection of all other features onto
                the direction vector W_i. If the projection is 0, it means
                that the feature W_i is orthogonal to all other features (W_i is then
                nearly monosemantic).
                On the other hand, if the sum of projection >=1, it means that
                the feature W_i can also activate other group of features (superposition
                takes place, thus W_i is polysemantic).

                sum(Wi^2) / (||Wi||^2 + tol)
                where tol is a small value to avoid division by zero.

        input args:
            w <torch.Tensor, [n_hidden, n_feat]: weight matrix after spatial averaging.

                the original weight matrix is of shape [C_out, C_in, h, w]
                (for 2D convolution) or [C_out, C_in, d, h, w] for 3D convolution.

                the input dimension (C_in) is associated with the number of features (n_feat)
                and the layer dimension (C_out) is associated with the number of neurons (n_hidden).
                Thus, the dimension becomes [n_hidden, n_feat].

        """

        n_feat = w.shape[-1]  # number of features
        # compute representation strength: how strong a feature is represented
        rep_strength = torch.sum(w**2, dim=0) ** 0.5  # [n_feat]

        # compute superposition index: how much a feature shares its dimension
        # with other features.
        # The idea is to sum the projection of all other features onto
        # the direction vector W_i. If the projection is 0, it means
        # that the feature W_i is orthogonal to all other features (W_i is then
        # nearly monosemantic).
        # On the other hand, if the sum of projection >=1, it means that
        # the feature W_i can also activate other group of features (superposition
        # takes place, thus W_i is polysemantic).

        tol = 1e-6
        mask = torch.ones((n_feat, n_feat)) - torch.eye(n_feat)  # [n_feat, n_feat]
        ww = torch.einsum("nj, nk -> jk", w, w)  # [n_feat, n_feat]
        superposition_index = torch.sum((ww * mask) ** 2, dim=0)  # [n_feat]
        superposition_index = superposition_index / (rep_strength**2 + tol)

        return rep_strength.detach().cpu(), superposition_index.detach().cpu()

    def feature_capacity(
        self,
        rep_strength: Float[Tensor, "n_feat"],
        superposition_index: Float[Tensor, "n_feat"],
    ) -> Float[Tensor, "n_feat"]:
        """
        Compute feature capacity (feature dimensionality):
                            fraction of embedding dimensions used for representing
                            a feature.
        """

        num = rep_strength**2  # [n_feat]
        den = (rep_strength**2) + superposition_index  # [n_feat]
        feat_dimensionality = num / den  # [n_feat]

        return feat_dimensionality

    def feature_capacity_layers(
        self, conv_layer_weights: list[Tensor]
    ) -> list[Float[Tensor, "n_feat"]]:

        feat_capacity_all_layers = []
        for i, w in enumerate(conv_layer_weights):
            # average weights across spatial dimensions
            if len(w.shape) == 4:
                w = w.mean(dim=(2, 3)).cpu()  # [n_hidden, n_feat]
            else:
                w = w.mean(dim=(2, 3, 4)).cpu()  # [n_hidden, n_feat]
            # w_norm = (w / torch.norm(w, dim=0)).cpu()  # [n_hidden, n_feat]
            rep_strength, superposition_index = sa.compute_superposition_index(w)

            fd = self.feature_capacity(rep_strength, superposition_index)
            feat_capacity_all_layers.append(fd)

        return feat_capacity_all_layers

    def dimensions_per_feature(self, w: Float[Tensor, "n_hidden n_feat"]) -> float:
        """Compute dimensions per feature, i.e. hidden_dim divided by Frobenius norm of matrix

        input:
            w [n_hidden, n_feat]: weight matrix (hidden_dim, input features)

        output
            frobenius_norm: float
        """
        hidden_dim = w.size(0)  # n_hidden
        w_frob = torch.norm(w, p="fro") ** 2
        w_frob = w_frob.item()
        return hidden_dim / w_frob

    def dimension_per_feature_layers(self, conv_layer_weights: list[Tensor]):
        """
        Compute the number of dimensions per feature.
        """

        dim_per_feat_layers = torch.empty(len(conv_layer_weights))
        for i, w in enumerate(conv_layer_weights):

            # average weights across spatial dimensions
            if len(w.shape) == 4:
                w = w.mean(dim=(2, 3)).cpu()
            else:
                w = w.mean(dim=(2, 3, 4)).cpu()

            # frobenius norm: how many features each layer can represent
            dim_per_feat = self.dimensions_per_feature(w)
            dim_per_feat_layers[i] = dim_per_feat

        return dim_per_feat_layers

    @torch.no_grad()
    def compute_layer_act_rds(self, dotMatch: float, dotDens: float):
        """
        compute the RDS activation for each layer for a given dotMatch and dotDens
        """

        # set the model to evaluation mode
        self.model.eval()  # inference mode (turns off dropout, etc.)

        # Create a dict to store the output
        activations = {}

        # Define a forward hook to capture the output of the target layer
        def get_activation(name):
            def hook(module, input, output):
                # output is still on GPU if the model is; detach & move to CPU if needed
                activations[name] = output.detach()

            return hook

        # Register the forward hook
        hook_handles = []
        for layer in self.target_layer:
            hook_handle = layer.register_forward_hook(get_activation(layer))
            hook_handles.append(hook_handle)

        # create dataloader
        rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
            dotMatch,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            self.background_flag,
            self.pedestal_flag,
        )
        # rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
        #     dotMatch,
        #     dotDens,
        #     sa.disp_ct_pix_list,
        #     sa.n_rds_each_disp,
        #     sa.background_flag,
        #     sa.pedestal_flag,
        # )

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t + 1.0) / 2.0),
                transforms.Normalize(mean, std),
            ]
        )
        rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
        batch_size = params_rds["batch_size_rds"]
        rds_loader = DataLoader(
            rds_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=1,
        )

        n_samples = len(rds_loader.dataset)
        layer_act_dict = {
            "layer19": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer20": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer21": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer22": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer23": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer24": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer25": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer26": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer27": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer28": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer29": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer30": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer31": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer32": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer33a": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer34a": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer35a": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer36a": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer37": torch.empty((n_samples, 192), dtype=torch.float32),
        }

        disp_record = torch.empty(n_samples, dtype=torch.int8)
        for i, (inputs_left, inputs_right, disps) in enumerate(rds_loader):

            id_start = i * batch_size
            id_end = id_start + batch_size

            # record the disparity
            disp_record[id_start:id_end] = disps.cpu()

            # build nested tensor
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))
            ref = disps / 10.0  # disparity direction
            input_data = NestedTensor(
                left=inputs_left.pin_memory().to(self.device, non_blocking=True),
                right=inputs_right.pin_memory().to(self.device, non_blocking=True),
                disp=disps.pin_memory().to(self.device, non_blocking=True),
                ref=ref.pin_memory().to(self.device, non_blocking=True),
            )

            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _ = self.model(input_data)

            # retrieve the activations
            for j, layer in enumerate(self.target_layer):
                out = activations[layer]  # shape: (B, C_out, H_out, W_out)
                # print(f"{target_layer[i]} output:", out.size())

                # average the output across 2D spatial dimensions
                out_avg = out.mean(dim=(-2, -1)).cpu().detach()
                if len(out.shape) == 4:
                    layer_act_dict[self.layer_name[j]][id_start:id_end] = out_avg
                else:
                    out_avg = out_avg.view(out_avg.size(0), -1)
                    layer_act_dict[self.layer_name[j]][id_start:id_end] = out_avg

        # clean up
        for h in hook_handles:
            h.remove()

        # save file
        # for layer in self.layer_name:
        torch.save(
            layer_act_dict,
            f"{self.superposition_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

        torch.save(
            disp_record,
            f"{self.superposition_dir}/disp_record_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

        # return (layer_act_dict, disp_record)

    def compute_layer_act_rds_all(self):
        """
        compute the RDS activation for each layer for all dotMatch and dotDens
        """

        for dotDens in self.dotDens_list:
            for dotMatch in self.dotMatch_list:

                print(
                    f"Computing RDS activation: "
                    + f"dotDens={dotDens:.2f}, "
                    + f"dotMatch={dotMatch:.2f}"
                )

                # compute layer activation for RDS
                self.compute_layer_act_rds(dotMatch, dotDens)

    def compute_monosemanticity(self, layer_name: str) -> Float[Tensor, "n_hidden"]:
        """
        Compute the monosemanticity score for each neuron in a given layer.

        The monosemnaticity is defined as in eq. 7 in https://arxiv.org/abs/2211.09169

        Specifically, the monosemanticity score is defined as the ratio of the
        activation of neuron i to its most strongly activating feature and the
        sum of its activations over all features.
        """

        # load layer activations
        # layer_name = sa.layer_name[-2]

        # load the activation only for getting the shape
        dotDens = 0.3
        dotMatch = 1.0
        temp = torch.load(
            f"{sa.superposition_dir}/"
            + f"act_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )[layer_name]
        n_sample = temp.size(0) // 2
        n_feat = len(sa.dotMatch_list) * len(sa.dotDens_list) * 2
        n_hidden = temp.size(1)
        act_all = torch.empty((n_sample, n_feat, n_hidden), dtype=torch.float32)

        # gather the activation for all features (RDSs)
        for dm, dotMatch in enumerate(sa.dotMatch_list):
            for dd, dotDens in enumerate(sa.dotDens_list):

                # dotDens = 0.3
                # dotMatch = 1.0
                act_rds = torch.load(
                    f"{sa.superposition_dir}/"
                    + f"act_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )[
                    layer_name
                ]  # [n_sample, n_hidden]

                disp_record = torch.load(
                    f"{sa.superposition_dir}/"
                    + f"disp_record_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )

                # store the activation for near and far disparity
                count = (dm * len(sa.dotDens_list) * 2) + (dd * 2)

                # get near/far disparity activation
                act_all[:, count] = act_rds[disp_record == sa.disp_ct_pix_list[0]]
                act_all[:, count + 1] = act_rds[disp_record == sa.disp_ct_pix_list[1]]

        # average across batch
        tol = 1e-6
        act_all_avg = act_all.mean(dim=0)  # [n_feat, n_hidden]
        num = torch.max(F.relu(act_all_avg), dim=0)[0]  # [n_hidden]
        den = torch.sum(F.relu(act_all_avg), dim=0)  # [n_hidden]
        monosemanticity = num / (den + tol)

        return monosemanticity

    def plotBar_superposition(
        self,
        rep_strength: Float[Tensor, "n_feat"],
        superposition_index: Float[Tensor, "n_feat"],
        layer_name: str,
        save_flag: bool = False,
    ):
        """
        plot bar plot of representation strength and superposition index
        """

        n_feat = rep_strength.shape[-1]
        features = range(n_feat)
        bars = rep_strength.numpy()
        color_values = superposition_index.numpy()
        color_values /= np.max(color_values)
        cmap = mlp.colormaps["cividis"]
        bar_colors = [cmap(value) for value in color_values]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (7, 14)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Superposition Analysis ({layer_name})",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Features", va="center", rotation=90)
        fig.text(0.5, -0.01, "||Wi||", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        axes.invert_yaxis()
        axes.barh(features, bars, color=bar_colors)
        axes.set_yticklabels("")
        axes.set_xlabel("||Wi||")
        axes.set_ylabel("Features")

        # vertical line at 1
        axes.axvline(x=1, color="black", linestyle="--", linewidth=2)
        axes.set_box_aspect(2)
        # axes.set_axis_off()

        # color bar
        norm = mlp.colors.Normalize(vmin=0, vmax=1)
        sm = mlp.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation="horizontal")
        cbar.set_label("Superposition Index", labelpad=5)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.superposition_dir}/superposition_analysis_{layer_name}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotStem_monosemanticity(
        self,
        monosemanticity: Float[Tensor, "n_hidden"],
        layer_name: str,
        save_flag: bool = False,
    ):

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 5)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Monosemanticity spectrum ({layer_name})",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        axes.stem(monosemanticity, linefmt="black", basefmt="k-")

        y_up = 1.1
        y_low = 0.0
        y_step = 0.2
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Neuron")
        axes.set_ylabel("Monosemanticity")

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.superposition_dir}/monosemanticity_{layer_name}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_monosemanticity_vs_layer(
        self, threshold: float = 0.5, n_features: float = 54, save_flag: bool = False
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer.

        input args:
            threshold: float, threshold for monosemanticity
                (default: 0.5)

            n_features: int, number of features per layer
                (default: 54 -> n_rds_types * n_dotDens * 2
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)
                          the 2 for near and far disparity

            save_flag: bool, whether to save the figure (default: False)
        """

        mono_per_feat = torch.empty(len(self.layer_name))
        for i in range(len(self.layer_name)):
            layer_name = sa.layer_name[i]
            monosemanticity = self.compute_monosemanticity(layer_name)

            # threshold the monosemanticity
            mono_thresh = torch.where(monosemanticity > threshold, 1.0, 0.0)
            mono_per_feat[i] = mono_thresh.sum() / n_features

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (12, 5)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Monosemantic neurons per feature across Layer",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        axes.plot(mono_per_feat, linewidth=2, marker="o", color="black")

        y_up = 1.1
        y_low = 0.0
        y_step = 0.2
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Layer")
        axes.set_ylabel("n_mono/n_feat")
        axes.set_xticks(np.arange(len(self.layer_name)))
        axes.set_xticklabels(self.layer_name, rotation=45, ha="right")
        axes.set_xlim(-0.5, len(self.layer_name) - 0.5)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.superposition_dir}/monosemanticity_vs_layer.pdf",
                dpi=600,
                bbox_inches="tight",
            )


# %%
params_rds = {
    "target_disp": 10,  # RDS target disparity (pix) to be analyzed
    "n_rds_each_disp": 64,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 2,
}
config = ConfigGCNet()
sa = SuperpositionAnalysis(config, params_rds)
# %%
conv_layer_names, conv_layer_weights = sa.get_conv_names_and_weights()

for i, w in enumerate(conv_layer_weights):

    # average weights across spatial dimensions
    if len(w.shape) == 4:
        w = w.mean(dim=(2, 3)).cpu()  # [n_hidden, n_feat]
    else:
        w = w.mean(dim=(2, 3, 4)).cpu()  # [n_hidden, n_feat]
    # w_norm = (w / torch.norm(w, dim=0)).cpu()
    rep_strength, superposition_index = sa.compute_superposition_index(w)

    layer_name = conv_layer_names[i]
    sa.plotBar_superposition(rep_strength, superposition_index, layer_name, save_flag=1)

# %%
feat_capacity_layers = sa.feature_capacity_layers(conv_layer_weights)

feat_cap_avg = torch.empty(len(feat_capacity_layers))
feat_cap_sem = torch.empty(len(feat_capacity_layers))
for i in range(len(feat_capacity_layers)):
    feat_cap_avg[i] = feat_capacity_layers[i].mean()
    feat_cap_sem[i] = sem(feat_capacity_layers[i].numpy())

plt.plot(feat_cap_avg, "o-")

# %%
dim_per_feat_layers = sa.dimension_per_feature_layers(conv_layer_weights)

plt.plot(dim_per_feat_layers / dim_per_feat_layers.max(), "o-")
plt.plot(feat_cap_avg, "o-")


# %% plot weights
i = -2
w_pre = conv_layer_weights[i]
w_post = conv_layer_weights[i + 1]
plt.imshow(
    w_pre.mean(dim=(2, 3, 4)).T.detach().cpu(), cmap="coolwarm", interpolation="nearest"
)
plt.imshow(
    w_post.mean(dim=(2, 3, 4)).T.detach().cpu(),
    cmap="coolwarm",
    interpolation="nearest",
)

# %%
w = w_post.mean(dim=(2, 3, 4)).detach().cpu()
w_norm = w / torch.norm(w, dim=0)
ww = w_norm.T @ w_norm
plt.imshow(ww, cmap="coolwarm", interpolation="nearest")

# %% compute layer activation for all RDSs
sa.compute_layer_act_rds_all()

# %%
layer_name = sa.layer_name[-1]

# load the activation only for getting the shape
dotDens = 0.3
dotMatch = 1.0
temp = torch.load(
    f"{sa.superposition_dir}/"
    + f"act_rds_dotDens_{dotDens:.2f}_"
    + f"dotMatch_{dotMatch:.2f}.pt"
)[layer_name]
n_sample = temp.size(0) // 2
n_feat = len(sa.dotMatch_list) * len(sa.dotDens_list) * 2
n_hidden = temp.size(1)
act_all = torch.empty((n_sample, n_feat, n_hidden), dtype=torch.float32)

for dm, dotMatch in enumerate(sa.dotMatch_list):
    for dd, dotDens in enumerate(sa.dotDens_list):

        # dotDens = 0.3
        # dotMatch = 1.0
        act_rds = torch.load(
            f"{sa.superposition_dir}/"
            + f"act_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )[
            layer_name
        ]  # [n_sample, n_hidden]

        disp_record = torch.load(
            f"{sa.superposition_dir}/"
            + f"disp_record_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )

        # store the activation for near and far disparity
        count = (dm * len(sa.dotDens_list) * 2) + (dd * 2)
        act_all[:, count] = act_rds[disp_record == sa.disp_ct_pix_list[0]]
        act_all[:, count + 1] = act_rds[disp_record == sa.disp_ct_pix_list[1]]

# %%
# normalize the activation
act_norm = act_all / act_all.max()
rep_strength, superposition_index = sa.compute_superposition_index(
    act_norm.mean(dim=0).T
)
feat_cap = sa.feature_capacity(
    rep_strength,
    superposition_index,
)
sa.plotBar_superposition(
    rep_strength,
    superposition_index,
    f"{layer_name}_far",
    save_flag=False,
)
# %% monosemanticity
save_flag = True  # set to True to save the plot
for i in range(len(sa.layer_name)):
    layer_name = sa.layer_name[i]
    monosemanticity = sa.compute_monosemanticity(layer_name)
    sa.plotStem_monosemanticity(monosemanticity, layer_name, save_flag)
# %%
save_flag = True
sa.plotLine_monosemanticity_vs_layer(save_flag=save_flag)
# %%
