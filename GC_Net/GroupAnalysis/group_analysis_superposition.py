# %% load necessary modules
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from modules.gcnet import build_gcnet
from config.config import ConfigGCNet

import numpy as np
import pandas as pd
from jaxtyping import Float
import os

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import sem
import pingouin as pg
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multicomp import pairwise_tukeyhsd


from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utilities.misc import NestedTensor


# %%
class GA_Superposition:

    def __init__(self, config: ConfigGCNet, params_rds: dict):

        self.config = config
        self.dataset = config.dataset
        self.binocular_interaction = config.binocular_interaction
        self.interactions = config.interactions
        self.seed = config.seed
        self.epoch = config.epoch_to_load
        self.iter = config.iter_to_load

        # set up experiment directory
        self.experiment_dir = (
            f"run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"{self.seed}"
        )

        # create folder for saving plots of a given interaction
        # (all seeds are combined)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # create folder for saving group analysis
        # (all interactions are combined)
        self.group_dir = f"run/{self.dataset}/bino_interaction_group"
        if not os.path.exists(self.group_dir):
            os.makedirs(self.group_dir)

        # create folders for superposition analysis
        self.superposition_dir = (
            f"{self.experiment_dir}/"
            + f"superposition_epoch_{self.epoch}"
            + f"_iter_{self.iter}"
        )
        if not os.path.exists(self.superposition_dir):
            os.makedirs(self.superposition_dir)

        # create folder for saving group analysis plots
        # (all interactions are combined)
        self.group_plot_dir = f"{self.group_dir}/plots"
        if not os.path.exists(self.group_plot_dir):
            os.makedirs(self.group_plot_dir)

        # create folder for saving group analysis statistical tests
        # (all interactions are combined)
        self.group_stat_dir = f"{self.group_dir}/stats"
        if not os.path.exists(self.group_stat_dir):
            os.makedirs(self.group_stat_dir)

        # load model
        self.model = build_gcnet(config)

        self.layer_name = [
            "encoder.in_conv",
            "encoder.layer2",
            "decoder.layer3",
            "decoder.layer4",
        ]

        # target_layer, only convolutional layers
        # target_layer, only convolutional layers
        self.target_layer = [
            self.model.encoder.layer1[0],
            self.model.encoder.res_block[0].conv2d_block1[0],
            self.model.encoder.res_block[0].conv2d_block2[0],
            self.model.encoder.res_block[1].conv2d_block1[0],
            self.model.encoder.res_block[1].conv2d_block2[0],
            self.model.encoder.res_block[2].conv2d_block1[0],
            self.model.encoder.res_block[2].conv2d_block2[0],
            self.model.encoder.res_block[3].conv2d_block1[0],
            self.model.encoder.res_block[3].conv2d_block2[0],
            self.model.encoder.res_block[4].conv2d_block1[0],
            self.model.encoder.res_block[4].conv2d_block2[0],
            self.model.encoder.res_block[5].conv2d_block1[0],
            self.model.encoder.res_block[5].conv2d_block2[0],
            self.model.encoder.res_block[6].conv2d_block1[0],
            self.model.encoder.res_block[6].conv2d_block2[0],
            self.model.encoder.res_block[7].conv2d_block1[0],
            self.model.encoder.res_block[7].conv2d_block2[0],
            self.model.encoder.layer18,
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
            self.model.decoder.layer33a[0],  # convT3d
            self.model.decoder.layer34a[0],  # convT3d
            self.model.decoder.layer35a[0],  # convT3d
            self.model.decoder.layer36a[0],  # convT3d
            self.model.decoder.layer37,  # convT3d
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
        self.dotMatch_list = params_rds["dotMatch_list"]  # [0.0, 0.5, 1.0] dot match
        self.background_flag = params_rds["background_flag"]  # 1: with cRDS background
        self.pedestal_flag = params_rds[
            "pedestal_flag"
        ]  # 1: use pedestal to ensure rds disparity > 0
        self.batch_size_rds = params_rds[
            "batch_size_rds"
        ]  # batch size for RDS generation
        self.disp_ct_pix_list = [self.target_disp, -self.target_disp]

        self.device = config.device

        # print out network configuration
        self.__getconfig__()

    def __getconfig__(self) -> None:
        """
        print out the network configuration
        """

        print(
            "Network config\n"
            + f"binocular interaction: {self.config.binocular_interaction}\n"
            + f"seed: {self.config.seed}\n"
            + f"epoch: {self.config.epoch_to_load}\n"
            + f"iter: {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
            + f"superposition_dir: {self.superposition_dir}\n"
        )

    def update_network_config(
        self, interaction: str, seed: int, epoch: int, iter: int
    ) -> None:
        """
        Update the binocular interaction in the config and experiment
        directory.

        Args:
            interaction (str): The new binocular interaction to set.
        """

        # old config, for printing purposes
        interaction_old = self.binocular_interaction
        seed_old = self.seed
        epoch_old = self.epoch
        iter_old = self.iter

        # update binocular_interaction, seed, epoch, iter in
        # the class and config
        self.binocular_interaction = interaction
        self.config.binocular_interaction = interaction
        self.seed = seed
        self.config.seed = seed
        self.epoch = epoch
        self.config.epoch_to_load = epoch
        self.iter = iter
        self.config.iter_to_load = iter

        # update the experiment directories based on the new interaction
        self.experiment_dir = (
            f"run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"{self.seed}"
        )

        # update directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # update superposition_dir
        self.superposition_dir = (
            f"{self.experiment_dir}/" + f"superposition_epoch_{epoch}" + f"_iter_{iter}"
        )
        if not os.path.exists(self.superposition_dir):
            os.makedirs(self.superposition_dir)

        print(
            "Updating network config\n"
            + f"binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"seed: {seed_old} => {self.config.seed}\n"
            + f"epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
            + f"superposition_dir: {self.superposition_dir}\n"
        )

    def load_model(self) -> None:
        """
        Load the model state from a checkpoint (pre-trained model).
        """

        # build model
        self.model = build_gcnet(self.config)

        # load model state from checkpoint
        resume = f"epoch_{self.epoch}_iter_{self.iter}_model_best.pth.tar"
        resume_path = os.path.join(self.experiment_dir, resume)
        checkpoint = torch.load(f"{resume_path}", map_location=self.device)
        pretrained_dict = checkpoint["state_dict"]

        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(pretrained_dict.items()):
            if k.startswith(unwanted_prefix):
                pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)

        self.model.load_state_dict(pretrained_dict)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"GC-Net was successfully loaded to {self.device}. \n"
            + f"GC-Net model: {resume_path}\n"
            + f"binocular interaction: {self.binocular_interaction}\n"
            + f"compile mode: {self.config.compile_mode}\n"
            + f"experiment dir: {self.experiment_dir}\n"
        )

    def get_conv_names_and_weights(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the names and weights of convolutional layers in the model.

        the original weight matrix is of shape [C_out, C_in, h, w]
        (for 2D convolution) or [C_out, C_in, d, h, w] for
        3D convolution. See:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

        HOWEVER, in the case of convTranspose3D such
        as in decoder.layer4, the weight matrix is of shape
        [C_in, C_out, d, h, w] (please see this documentation:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)]

        the input dimension (C_in) is associated with the number
        of features (n_feat),
        and the layer dimension (C_out) is associated with the number
        of neurons (n_neuron).
        Thus, the dimension becomes [n_neuron, n_feat].

        Returns:
            conv_layer_names (list): List of names of convolutional layers.
            conv_layer_weights (list): List of weights of convolutional layers.
        """

        names = []
        params = []
        for name, param in self.model.named_parameters():
            names.append(name)
            params.append(param)

        # get index for convolutional layers
        layer_idx = []
        for i in range(len(names)):  # gather the first 3 conv layers in [names]
            if ".0.weight" in names[i] or "layer4.weight" in names[i]:
                layer_idx.append(i)

        # get convolutional layer names and weights
        conv_layer_names = []
        conv_layer_weights = []
        for i in layer_idx:
            conv_layer_names.append(names[i])

            # if layer4, transpose the weights because it is ConvTranspose3D
            # whose original shape is [C_in, C_out, d, h, w]
            # see: https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
            if "layer4" in names[i]:
                conv_layer_weights.append(params[i].transpose(0, 1))
            else:
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
                that the feature W_i is orthogonal to all other features
                (W_i is then nearly monosemantic).
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
                3D convolution. See:
                https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

                HOWEVER, in the case of convTranspose3D such
                as in decoder.layer4, the weight matrix is of shape
                [C_in, C_out, d, h, w] (please see this documentation:
                https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)]

                the input dimension (C_in) is associated with the number
                of features (n_feat),
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
        self, conv_layer_names: list[str], conv_layer_weights: list[Tensor]
    ) -> dict[str, Float[Tensor, "n_feat"]]:
        """
        Compute feature dimensionality for all convolutional layers.

        Args:
            conv_layer_names <list[str]>: list of names of convolutional layers
                in GC-Net:
                ["encoder.in_conv.0.weight",
                "encoder.layer2.0.weight",
                "decoder.layer3.0.weight",
                "decoder.layer4.weight"]

            conv_layer_weights (list[Tensor]): List of weights for each
                convolutional layer, where each weight is a tensor of shape
                [n_inst, n_feat, h, w] for 2D convolution or
                [n_inst, n_feat, d, h, w] for 3D convolution.

        Returns:
            dict[layer_name:str, Float[Tensor, "n_feat"]]: dict of feature dimensionality
                tensors for each convolutional layer, where each tensor has
                shape [n_feat]. Each tensor represents the feature dimensionality
                for the corresponding layer, indicating how many dimensions are
                used to represent the features in that layer.
        """

        feat_dim_all_layers = {}
        for i, w in enumerate(conv_layer_weights):

            # average weights across spatial dimensions
            if len(w.shape) == 4:
                w = w.mean(dim=(2, 3)).cpu()  # [n_inst, n_feat]
            else:
                w = w.mean(dim=(2, 3, 4)).cpu()
            rep_strength, superposition_index = self.compute_superposition_index(w)

            fd = self.feature_dimensionality(rep_strength, superposition_index)

            layer_name = conv_layer_names[i]
            feat_dim_all_layers[layer_name] = fd.detach()

        # save data
        torch.save(
            feat_dim_all_layers,
            f"{self.superposition_dir}/feat_dimensionality_all_layers.pt",
        )

        return feat_dim_all_layers

    def feature_dimensionality_interaction(
        self, interaction: str
    ) -> list[dict[str, Float[Tensor, "n_feat"]]]:
        """
        Compute feature dimensionality for a given interaction type
        across all seeds.

        Args:
            interaction <str>: the type of binocular interaction
                to analyse.

        Returns:
            feat_dim_interaction <list[dict[str, Float[Tensor, "n_feat"]]]>:
                List of dictionaries where each dictionary contains
                feature dimensionality tensors for a given seed number and
                binocular interaction type.
                Each key in the dictionary refers to each convolutional layer,
                (layer names) and values as tensors of shape [n_feat].
        """

        feat_dim_interaction = []

        for s, seed in enumerate(self.config.seed_to_analyse):

            # load pre-trained model
            if interaction == "default":
                epoch, iter = self.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = self.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

            # update network config and update directories
            self.update_network_config(interaction, seed, epoch, iter)

            # load new model
            self.load_model()

            # feature dimensionality analysis
            conv_layer_names, conv_layer_weights = self.get_conv_names_and_weights()
            feat_dim_layers = self.feature_dimensionality_layers(
                conv_layer_names, conv_layer_weights
            )  # dict[layer_name:str, Float[Tensor, "n_feat"]]

            # append to the list
            feat_dim_interaction.append(feat_dim_layers)

        return feat_dim_interaction

    def stat_feat_dim(self):
        """
        Perform statistical tests on feature dimensionality across
        different interaction types.
        """

        # gather all feat_dim and store it into records
        # calculate the # rows
        temp = torch.load(f"{self.superposition_dir}/feat_dimensionality_all_layers.pt")
        conv_layer_names = list(temp.keys())
        n_rows_each_layer = np.array([len(temp[l]) for l in conv_layer_names])
        n_rows_per_seed = n_rows_each_layer.sum()
        n_rows = (
            len(self.interactions) * len(self.config.seed_to_analyse) * n_rows_per_seed
        )
        records = np.empty(
            (n_rows, 4), dtype=np.float32
        )  # [interaction, seed, layer, feat_dim]

        # count = 0
        for i, interaction in enumerate(self.interactions):
            for s, seed in enumerate(self.config.seed_to_analyse):

                # load pre-trained model
                if interaction == "default":
                    epoch, iter = self.config.epoch_iter_to_load_default[s]
                elif interaction == "bem":
                    epoch, iter = self.config.epoch_iter_to_load_bem[s]
                elif interaction == "cmm":
                    epoch, iter = self.config.epoch_iter_to_load_cmm[s]
                else:  # sum_diff
                    epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

                # update network config and update directories
                self.update_network_config(interaction, seed, epoch, iter)

                # load feat_dimensionality of a given seed (all layers)
                # dict[layer_name:str, Float[Tensor, "n_feat"]]
                feat_dim_layers = torch.load(
                    f"{self.superposition_dir}/feat_dimensionality_all_layers.pt"
                )

                id_start = 0
                for ly, layer_name in enumerate(conv_layer_names):

                    id_start = (
                        i * len(self.config.seed_to_analyse) * n_rows_per_seed
                        + s * n_rows_per_seed
                        + n_rows_each_layer[:ly].sum()
                    )

                    id_end = id_start + n_rows_each_layer[ly]

                    print(id_start, id_end)

                    records[id_start:id_end, 0] = i  # interaction
                    records[id_start:id_end, 1] = seed  # seed
                    records[id_start:id_end, 2] = ly  # layer
                    records[id_start:id_end, 3] = feat_dim_layers[layer_name].numpy()

                    # count += n_rows_each_layer[l]

        # print(count)

        # create pandas df
        df = pd.DataFrame(records, columns=["interaction", "seed", "layer", "feat_dim"])

        # 2way ANOVA
        aov2 = pg.anova(data=df, dv="feat_dim", between=["interaction", "layer"])
        print(aov2)

        # save 2-way anova to csv
        aov2.to_csv(
            f"{self.group_stat_dir}/feat_dim_anova2way.csv",
            index=False,
        )

        # 2-way ANOVA with statsmodel
        # import statsmodels.api as sm
        # from statsmodels.formula.api import ols

        # # Fit a fixed-effects model
        # model_fe = ols("feat_dim ~ C(interaction) * C(layer)", data=df).fit()

        # # Get ANOVA table with F-values
        # anova_table = sm.stats.anova_lm(model_fe, typ=2)  # or typ=3
        # print(anova_table)

        # # mixed model
        # import statsmodels.formula.api as smf
        # mixed_model = smf.mixedlm(
        #     "feat_dim ~ C(interaction) * C(layer)", data=df, groups=df["seed"]
        # )
        # results = mixed_model.fit()
        # print(results.summary())

        # post-hoc with statsmodel
        comp = mc.MultiComparison(df["feat_dim"], df["interaction"])
        post_hocs = comp.tukeyhsd()
        print(post_hocs)

        # save
        # Convert TukeyHSD results to DataFrame
        posthoc_df = pd.DataFrame(
            data=post_hocs._results_table.data[1:],  # skip header row
            columns=post_hocs._results_table.data[0],  # use header row
        )

        # Save to CSV
        posthoc_df.to_csv(
            f"{self.group_stat_dir}/feat_dim_post_hoc_tukey.csv", index=False
        )

    def plotViolin_feat_dim_interaction(
        self,
        feat_dim_interaction: list[dict[str, Float[Tensor, "n_feat"]]],
        conv_layer_names: list[str],
        save_flag: bool = False,
    ) -> None:
        """
        Violin plot the average of feature dimensionality across seed number
        for a given interaction type.

        Args:
            feat_dim_interaction <list[dict[str, Float[Tensor, "n_feat"]]]>:
                List of dictionaries where each dictionary contains
                feature dimensionality tensors for a given seed number and
                binocular interaction type.
                Each key in the dictionary refers to each convolutional layer,
                (layer names) and values as tensors of shape [n_feat].

            conv_layer_names <list[str]>: List of names of convolutional layers
                in BNN:
                ["encoder.in_conv.0.weight",
                "encoder.layer2.0.weight",
                "decoder.layer3.0.weight",
                "decoder.layer4.weight"]

            save_flag <bool>: If True, save the plot to a file.
                Default is False.
        """

        # allocate an array to store feature dimensionality for each seed and layer
        feat_dim_layers = {}
        for i, layer_name in enumerate(conv_layer_names):

            feat_dim_seeds = []
            for s in range(len(self.config.seed_to_analyse)):
                feat_dim = feat_dim_interaction[s][layer_name]  # [n_feat]

                # average across features and append to the list
                feat_dim_seeds.append(feat_dim)  # [n_feat]

            # average across seeds for each feature in the layer
            feat_dim_layers[layer_name] = torch.stack(feat_dim_seeds).mean(
                dim=0
            )  # [n_feat]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (16, 4)
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
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # Layer positions on the x-axis
        positions = np.arange(1, len(conv_layer_names) + 1)

        # Violin plot
        data = []
        for layer_name in conv_layer_names:
            data.append(feat_dim_layers[layer_name])

        # Compute a width for each violin (e.g. proportion of max count)
        counts = np.array([len(d) for d in data])
        widths = counts / counts.max()

        parts = axes.violinplot(
            data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            widths=widths,
        )

        # Style the violins
        for pc in parts["bodies"]:
            pc.set_facecolor("#333333")
            # pc.set_edgecolor("black")
            pc.set_alpha(0.25)

        # set face and edge also for the mean/median lines
        # for line in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        for line in ["cmeans", "cbars", "cmaxes", "cmins"]:
            parts[line].set_edgecolor("#000000")
            parts[line].set_linewidth(1)

        # Overlay raw data points with a slight horizontal jitter
        for i, layer in enumerate(data):
            x = np.random.normal(positions[i], 0.04, size=layer.shape)
            axes.scatter(x, layer, s=12, alpha=1.0, color="#000000")

        # Labels and title
        # axes.set_xticks(positions)
        axes.set_xlabel("Convolutional Layer")
        axes.set_ylabel("Feat. dimensionality")

        # Grid on y-axis
        axes.yaxis.grid(True, linestyle="--", alpha=0.6)
        axes.set_axisbelow(True)

        # set y limits
        axes.set_ylim(-0.02, 1.02)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.plot_dir}/plotViolin_feat_dim_{self.config.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )


# %%
