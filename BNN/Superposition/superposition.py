# %%
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from engine.engine_base import Engine
from config.config import BNNconfig
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utilities.misc import NestedTensor

import pandas as pd
import numpy as np
from jaxtyping import Float
import os

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# %%
class SuperpositionAnalysis(Engine):

    def __init__(self, config: BNNconfig, params_rds: dict):
        super().__init__(config)

        self.config = config

        # folders for superposition analysis
        self.superposition_dir = (
            f"{self.save_dir}/"
            + f"experiment_{self.config.experiment_id}/"
            + f"superposition_epoch_{self.config.epoch_to_load}"
            + f"_iter_{self.config.iter_to_load}_seed_{self.config.seed}"
        )
        if not os.path.exists(self.superposition_dir):
            os.makedirs(self.superposition_dir)

        self.layer_name = [
            "encoder.in_conv",
            "encoder.layer2",
            "decoder.layer3",
            "decoder.layer4",
        ]

        # target_layer = sa.model.encoder.in_conv[0]
        self.target_layer = [
            self.model.encoder.in_conv[0],
            self.model.encoder.layer2[0],
            self.model.decoder.layer3[0],
            self.model.decoder.layer4,
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

    def get_conv_names_and_weights(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the names and weights of convolutional layers in the model.
        Returns:
            conv_layer_names (list): List of names of convolutional layers.
            conv_layer_weights (list): List of weights of convolutional layers.
        """

        names = []
        params = []
        for name, param in self.model.named_parameters():
            names.append(name)
            params.append(param)

        # get index for convolutional layers, excepth the last one
        layer_idx = []
        for i in range(len(names)):  # gather the first 3 conv layers in [names]
            if ".0.weight" in names[i] or "layer4.weight" in names[i]:
                layer_idx.append(i)

        # get convolutional layer names and weights
        conv_layer_names = []
        conv_layer_weights = []
        for i in layer_idx:
            conv_layer_names.append(names[i])
            conv_layer_weights.append(params[i])

        return (conv_layer_names, conv_layer_weights)

    def compute_superposition_index(self, w: Float[Tensor, "n_neuron n_feat"]):
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
                the feature W_i can also activate other group of features
                (superposition takes place, thus W_i is polysemantic).

                sum(Wi^2) / (||Wi||^2 + tol)
                where tol is a small value to avoid division by zero.

        input args:
            w <torch.Tensor, [n_neuron, n_feat]: weight matrix after spatial
                averaging (average across the width and height).

                the original weight matrix is of shape [C_out, C_in, h, w]
                (for 2D convolution) or [C_out, C_in, d, h, w] for
                3D convolution.

                the input dimension (C_in) is associated with the number of
                features (n_feat)
                and the layer dimension (C_out) is associated with the number
                of neurons (n_neuron).
                Thus, the dimension becomes [n_neuron, n_feat].

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

        return rep_strength, superposition_index

    def feature_dimensionality(
        self,
        rep_strength: Float[Tensor, "n_feat"],
        superposition_index: Float[Tensor, "n_feat"],
    ) -> Float[Tensor, "n_feat"]:
        """
        Compute feature dimensionality for a single layer.
        Feature dimensionality is defined as the ratio of
        representation strength to the superposition index.

        Feature dimensionality measures the fraction of embedding dimensions
        within a layer used for representing individual features
        (Elhage et al., 2022). The values are bounded within a range of 0 and 1.
        A lower value signifies that a small fraction of the layer’s
        dimensions are used to represent the features, suggesting a stronger
        superposition, where multiple features are encoded within the same
        embedding dimensions. For example, a feature dimensionality with a
        value of 3/4 can be geometrically visualized as 4 features being
        represented in 3 dimensions (forming a tetrahedron); a value of 2/3
        can be visualized as three features being represented in 2 dimensions
        (forming a triangle); see Elhage et al., 2022 for more examples.

        Args:
            rep_strength (Float[Tensor, "n_feat"]): Representation strength of
                features in the layer.
            superposition_index (Float[Tensor, "n_feat"]): Superposition index
                of features in the layer.

        Returns:
            Float[Tensor, "n_feat"]: Feature dimensionality for each feature
                in the layer.
        """

        num = rep_strength**2  # [n_feat]
        den = (rep_strength**2) + superposition_index  # [n_feat]
        feat_dimensionality = num / den  # [n_feat]

        return feat_dimensionality

    def feature_dimensionality_layers(
        self, conv_layer_weights: list[Tensor]
    ) -> list[Float[Tensor, "n_feat"]]:
        """
        Compute feature dimensionality for all convolutional layers.

        Args:
            conv_layer_weights (list[Tensor]): List of weights for each
                convolutional layer, where each weight is a tensor of shape
                [n_inst, n_feat, h, w] for 2D convolution or
                [n_inst, n_feat, d, h, w] for 3D convolution.

        Returns:
            list[Float[Tensor, "n_feat"]]: List of feature dimensionality
                tensors for each convolutional layer, where each tensor has
                shape [n_feat]. Each tensor represents the feature dimensionality
                for the corresponding layer, indicating how many dimensions are
                used to represent the features in that layer.
        """

        feat_dimensionality_all_layers = []
        for i, w in enumerate(conv_layer_weights):
            # for i in range(len(conv_layer_weights) - 1):  # exclude the last layer
            # w = conv_layer_weights[i]  # [n_neuron, n_feat]

            # average weights across spatial dimensions
            if len(w.shape) == 4:
                w = w.mean(dim=(2, 3)).cpu()  # [n_inst, n_feat]
            else:
                w = w.mean(dim=(2, 3, 4)).cpu()
            rep_strength, superposition_index = self.compute_superposition_index(w)

            fd = self.feature_dimensionality(rep_strength, superposition_index)
            feat_dimensionality_all_layers.append(fd)

        # save data
        torch.save(
            feat_dimensionality_all_layers,
            f"{self.superposition_dir}/feat_dimensionality_all_layers.pt",
        )

        return feat_dimensionality_all_layers

    def dimensions_per_feature(self, w: Float[Tensor, "n_neuron n_feat"]) -> float:
        """
        Compute dimensions per feature, i.e.
        hidden_dim divided by Frobenius norm of matrix

        input:
            w [n_neuron, n_feat]: weight matrix (hidden_dim, input features)

        output
            frobenius_norm: float
        """
        hidden_dim = w.size(0)  # n_neuron
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
                w = w.mean(dim=(-2, -1)).cpu()
            else:
                w = w.mean(dim=(-3, -2, -1)).cpu()

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
        # layer_act_dict = {
        #     key: torch.empty((n_samples, 32), dtype=torch.float32)
        #     for key in sa.layer_name
        # }
        layer_act_dict = {
            "encoder.in_conv": torch.empty((n_samples, 32), dtype=torch.float32),
            "encoder.layer2": torch.empty((n_samples, 32), dtype=torch.float32),
            "decoder.layer3": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "decoder.layer4": torch.empty((n_samples, 192), dtype=torch.float32),
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
                    "Computing RDS activation: "
                    + f"dotDens={dotDens:.2f}, "
                    + f"dotMatch={dotMatch:.2f}"
                )

                # compute layer activation for RDS
                self.compute_layer_act_rds(dotMatch, dotDens)

    def compute_monosemanticity(self, layer_name: str) -> Float[Tensor, "n_neuron"]:
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
            f"{self.superposition_dir}/"
            + f"act_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )[layer_name]
        n_sample = temp.size(0) // 2
        n_feat = len(self.dotMatch_list) * len(self.dotDens_list) * 2
        n_neuron = temp.size(1)
        act_all = torch.empty((n_sample, n_feat, n_neuron), dtype=torch.float32)

        # gather the activation for all features (RDSs)
        for dm, dotMatch in enumerate(self.dotMatch_list):
            for dd, dotDens in enumerate(self.dotDens_list):

                # dotDens = 0.3
                # dotMatch = 1.0
                act_rds = torch.load(
                    f"{self.superposition_dir}/"
                    + f"act_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )[
                    layer_name
                ]  # [n_sample, n_neuron]

                disp_record = torch.load(
                    f"{self.superposition_dir}/"
                    + f"disp_record_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )

                # store the activation for near and far disparity
                count = (dm * len(self.dotDens_list) * 2) + (dd * 2)

                # get near/far disparity activation
                act_all[:, count] = act_rds[disp_record == self.disp_ct_pix_list[0]]
                act_all[:, count + 1] = act_rds[disp_record == self.disp_ct_pix_list[1]]

        # average across batch
        tol = 1e-6
        act_all_avg = act_all.mean(dim=0)  # [n_feat, n_neuron]
        num = torch.max(F.relu(act_all_avg), dim=0)[0]  # [n_neuron]
        den = torch.sum(F.relu(act_all_avg), dim=0)  # [n_neuron]
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
        plot representation strength and superposition index
        """

        n_feat = rep_strength.shape[-1]
        features = range(n_feat)
        bars = rep_strength.detach().cpu().numpy()
        color_values = superposition_index.detach().cpu().numpy()
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

    def plot_featDimensionality(
        self, feat_dimensionality_layers: list[Tensor], save_flag: bool = False
    ):

        # average across features
        feat_dimensionality_avg = torch.empty(len(feat_dimensionality_layers))
        feat_dimensionality_sem = torch.empty(len(feat_dimensionality_layers))
        for i in range(len(feat_dimensionality_layers)):
            feat_dimensionality_avg[i] = feat_dimensionality_layers[i].mean().detach()
            feat_dimensionality_sem[i] = sem(
                feat_dimensionality_layers[i].detach().numpy()
            )

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (9, 9)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Feat. dimensionality ({self.config.binocular_interaction})",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Feat. dimensionality", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot the one standard error
        x = np.arange(len(feat_dimensionality_avg))  # [0, 1, 2, ...]
        y = np.array(feat_dimensionality_avg)
        y_err = np.array(feat_dimensionality_sem)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="black", ls="-", capsize=7)

        # plot the marker
        markersize = 12
        axes.plot(x, y, "o", markersize=markersize, c="black")

        # plot the horizontal line at 0.5
        axes.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

        x_low = 0.0
        x_up = len(feat_dimensionality_avg)
        x_step = 1.0
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        # layer_name = self.layer_name[:-1]  # exclude the last layer (decoder.layer4)
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(self.layer_name, rotation=45)
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.2, x_up - 0.5)
        axes.set_ylim(y_low - 0.05, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            fig.savefig(
                f"{self.superposition_dir}/PlotLine_featDimensionality.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plot_featDimensionality_all_interactions(self, save_flag: bool = False):
        """
        plot the feature dimensionality for all binocular interactions.
        """

        sp_dir_avg = "run/sceneflow_monkaa/bino_interaction_avg/"
        sp_dir_default = f"{sp_dir_avg}/superposition_default_{self.config.seed}"
        sp_dir_bem = f"{sp_dir_avg}/superposition_bem_{self.config.seed}"
        sp_dir_cmm = f"{sp_dir_avg}/superposition_cmm_{self.config.seed}"
        sp_dir_sum_diff = f"{sp_dir_avg}/superposition_sum_diff_{self.config.seed}"

        sp_dirs = [sp_dir_default, sp_dir_bem, sp_dir_cmm, sp_dir_sum_diff]

        # Define group‐names and layer‐names in the same order as rows/columns:
        groups = ["default", "bem", "cmm", "sum_diff"]
        n_layers = 2  # only the last 2 layers: layer 3 & 4

        # Build a list of dicts (one row per (group, layer) combination):
        # load data for obtaining the shape
        dr = sp_dirs[0]
        temp = torch.load(f"{dr}/feat_dimensionality_all_layers.pt")
        n_rows_per_layer = np.array([len(f) for f in temp])
        n_rows = n_rows_per_layer[2:].sum() * len(groups)  # only layer 3 & 4
        records = np.zeros((n_rows, 3), dtype=np.float32)  # [interaction, layer, unit]

        feat_dim_avg_all = torch.empty((len(sp_dirs), 4))
        feat_dim_sem_all = torch.empty((len(sp_dirs), 4))
        for d in range(len(sp_dirs)):

            # load data
            # d = 0
            dr = sp_dirs[d]
            feat_dimensionality_layers = torch.load(
                f"{dr}/feat_dimensionality_all_layers.pt"
            )

            for j in range(n_layers):

                n_units = len(feat_dimensionality_layers[j + 2])  # start from layer 3
                id_start = (d * n_rows_per_layer[2:].sum()) + n_rows_per_layer[
                    2 : j + 2
                ].sum()
                id_end = id_start + n_units
                # print(f"{id_start} - {id_end}")

                records[id_start:id_end, 0] = (
                    d  # bino interaction: 1: default, 2: bem, 3: cmm, 4: sum_diff
                )
                records[id_start:id_end, 1] = j + 3  # layer, start from layer 3
                records[id_start:id_end, 2] = (
                    feat_dimensionality_layers[j + 2].detach().numpy()
                )

            # average across layers
            feat_dimensionality_avg = torch.empty(len(feat_dimensionality_layers))
            feat_dimensionality_sem = torch.empty(len(feat_dimensionality_layers))
            for i in range(len(feat_dimensionality_layers)):
                feat_dimensionality_avg[i] = (
                    feat_dimensionality_layers[i].mean().detach()
                )
                feat_dimensionality_sem[i] = sem(feat_dimensionality_layers[i].detach())

            feat_dim_avg_all[d] = feat_dimensionality_avg
            feat_dim_sem_all[d] = feat_dimensionality_sem

        # create pandas df
        df = pd.DataFrame(records, columns=["bino", "layer", "feat_dim"])
        # 2way ANOVA
        aov2 = pg.anova(data=df, dv="feat_dim", between=["bino", "layer"])
        # save 2-way anova to csv
        aov2.to_csv(f"{sp_dir_avg}/feature_dimensionality_anova2way.csv", index=False)

        # Tukey test
        df_layer3 = df[df.layer == 3]
        tukey = pairwise_tukeyhsd(
            endog=df_layer3["feat_dim"], groups=df_layer3["bino"], alpha=0.05
        )
        print(tukey)
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        # save
        tukey_df.to_csv(
            f"{sp_dir_avg}/feature_dimensionality_layer3_tukey.csv", index=False
        )

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (9, 9)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Feat. dimensionality all interactions",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Feat. dimensionality", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot the one standard error
        line_colors = ["black", "blue", "magenta", "darkorange"]
        x = [0, 1]  # only layer 3 and 4
        for f in range(len(feat_dim_avg_all)):
            y = np.array(feat_dim_avg_all[f, 2:])  # only layer 3 and 4
            y_err = np.array(feat_dim_sem_all[f, 2:])  # only layer 3 and 4
            axes.errorbar(x, y, yerr=y_err, lw=3, c=line_colors[f], ls="-", capsize=7)

            # # plot the marker
            # markersize = 12
            # axes.plot(x, y, "o", markersize=markersize, c="black")

        # legend
        axes.legend(["Concat", "BEM", "CMM", "Sum-diff"], loc="lower right")

        # plot the horizontal line at 0.5
        axes.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

        x_low = 0.0
        x_up = 2  # only layer 3 and 4
        x_step = 1.0
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        # layer_name = self.layer_name[:-1]  # exclude the last layer (decoder.layer4)
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 1))
        axes.set_xticklabels(self.layer_name[2:], rotation=45)
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.2, x_up - 0.5)
        axes.set_ylim(y_low - 0.05, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            fig.savefig(
                f"{sp_dir_avg}/PlotLine_featDimensionality_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plot_monosemanticity_spectrum_in_layer(
        self,
        monosemanticity: Float[Tensor, "n_neuron"],
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

    def plotLine_n_mono_vs_layer(
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
            layer_name = self.layer_name[i]
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
    "n_rds_each_disp": 256,  # n_rds for each disparity magnitude in disp_ct_pix
    "dotDens_list": 0.1 * np.arange(1, 10),  # dot density
    "rds_type": ["ards", "hmrds", "crds"],  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
    "dotMatch_list": [0.0, 0.5, 1.0],  # dot match
    "background_flag": 1,  # 1: with cRDS background
    "pedestal_flag": 0,  # 1: use pedestal to ensure rds disparity > 0
    "batch_size_rds": 2,
}
config = BNNconfig()
sa = SuperpositionAnalysis(config, params_rds)

# %%
conv_layer_names, conv_layer_weights = sa.get_conv_names_and_weights()

for i, w in enumerate(conv_layer_weights):

    # average weights across spatial dimensions
    if len(w.shape) == 4:
        w = w.mean(dim=(-2, -1)).cpu()  # [n_inst, n_feat]
    else:
        w = w.mean(dim=(2, 3, 4)).cpu()
    rep_strength, superposition_index = sa.compute_superposition_index(w)

    layer_name = conv_layer_names[i]
    sa.plotBar_superposition(rep_strength, superposition_index, layer_name, save_flag=1)

# %%
feat_dimensionality_layers = sa.feature_dimensionality_layers(conv_layer_weights)
sa.plot_featDimensionality(feat_dimensionality_layers, save_flag=True)
sa.plot_featDimensionality_all_interactions(save_flag=True)

# %%
dim_per_feat_layers = sa.dimension_per_feature_layers(conv_layer_weights)
print(dim_per_feat_layers)


# %% plot weights
i = 1
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
layer_name = sa.layer_name[2]

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
n_neuron = temp.size(1)
act_all = torch.empty((n_sample, n_feat, n_neuron), dtype=torch.float32)

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
        ]  # [n_sample, n_neuron]

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
    sa.plot_monosemanticity_spectrum_in_layer(monosemanticity, layer_name, save_flag)
# %%
save_flag = True
sa.plotLine_n_mono_vs_layer(save_flag=save_flag)

# %%
