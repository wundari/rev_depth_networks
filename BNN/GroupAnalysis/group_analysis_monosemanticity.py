# %% load necessary modules

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from scipy.stats import sem

from GroupAnalysis.group_analysis import GA

# from modules.bnn import build_bnn
from config.config import BNNconfig
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utilities.misc import NestedTensor

from jaxtyping import Float
import os


# %%
class GA_Monosemanticity(GA):

    def __init__(self, config: BNNconfig, params_rds: dict):

        super().__init__(config, params_rds)

        # create folders for monosemanticity analysis
        self.monosemanticity_dir = (
            f"{self.experiment_dir}/"
            + f"monosemanticity_epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}"
        )
        if not os.path.exists(self.monosemanticity_dir):
            os.makedirs(self.monosemanticity_dir)

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
            + f"monosemanticity_dir: {self.monosemanticity_dir}\n"
        )

    def update_network_config(
        self, interaction: str, seed: int, epoch_to_load: int, iter_to_load: int
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
        epoch_old = self.epoch_to_load
        iter_old = self.iter_to_load

        # update binocular_interaction, seed, epoch, iter in
        # the class and config
        self.binocular_interaction = interaction
        self.config.binocular_interaction = interaction
        self.seed = seed
        self.config.seed = seed
        self.epoch_to_load = epoch_to_load
        self.config.epoch_to_load = epoch_to_load
        self.iter_to_load = iter_to_load
        self.config.iter_to_load = iter_to_load

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

        # update monosemanticity_dir
        self.monosemanticity_dir = (
            f"{self.experiment_dir}/"
            + f"monosemanticity_epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}"
        )
        if not os.path.exists(self.monosemanticity_dir):
            os.makedirs(self.monosemanticity_dir)

        print(
            "Updating network config\n"
            + f"binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"seed: {seed_old} => {self.config.seed}\n"
            + f"epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
            + f"monosemanticity_dir: {self.monosemanticity_dir}\n"
        )

    @torch.no_grad()
    def compute_layer_act_rds(self, dotMatch: float, dotDens: float):
        """
        compute the RDS activation for each layer for a given dotMatch
        and dotDens
        """

        # set up seed
        seed_number = self.seed
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        random.seed(seed_number)
        np.random.seed(seed_number)
        os.environ["PYTHONHASHSEED"] = str(seed_number)

        # initialize random seed number for dataloader
        def seed_worker(worker_id):
            worker_seed = seed_number  # torch.initial_seed()  % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

            # print out seed number for each worker
            # np_seed = np.random.get_state()[1][0]
            # py_seed = random.getstate()[1][0]

            # print(f"{worker_id} seed pytorch: {worker_seed}\n")
            # print(f"{worker_id} seed numpy: {np_seed}\n")
            # print(f"{worker_id} seed python: {py_seed}\n")

        g = torch.Generator()
        g.manual_seed(seed_number)

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
        # dotMatch = 1.0
        # dotDens = 0.3
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
        batch_size = self.batch_size_rds
        rds_loader = DataLoader(
            rds_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g,
        )

        n_samples = len(rds_loader.dataset)
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
            if ref.mean() > 0:
                input_data = NestedTensor(
                    left=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            else:
                input_data = NestedTensor(
                    left=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            # input_data = NestedTensor(
            #     left=inputs_left.pin_memory().to(ga_mono.device, non_blocking=True),
            #     right=inputs_right.pin_memory().to(ga_mono.device, non_blocking=True),
            #     disp=disps.pin_memory().to(ga_mono.device, non_blocking=True),
            #     ref=ref.pin_memory().to(ga_mono.device, non_blocking=True),
            # )

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
            f"{self.monosemanticity_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

        torch.save(
            disp_record,
            f"{self.monosemanticity_dir}/disp_record_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

    def compute_layer_act_all_rds(self):
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

    def compute_layer_act_rds_all_seeds(self, interaction: str):

        for s, seed in enumerate(self.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = self.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = self.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            self.update_network_config(interaction, seed, epoch, iter)

            # update model
            self.load_model()

            # compute layer activation for all RDSs
            self.compute_layer_act_all_rds()

    def compute_monosemanticity(self, layer_name: str) -> Float[Tensor, "n_neuron"]:

        # load the activation only for getting the shape
        dotDens = 0.3
        dotMatch = 1.0
        temp = torch.load(
            f"{self.monosemanticity_dir}/"
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

                act_rds = torch.load(
                    f"{self.monosemanticity_dir}/"
                    + f"act_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )[
                    layer_name
                ]  # [n_sample, n_neuron]

                disp_record = torch.load(
                    f"{self.monosemanticity_dir}/"
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

    def plot_monosemanticity_spectrum_in_layer(
        self,
        monosemanticity: Float[Tensor, "n_neuron"],
        layer_name: str,
        save_flag: bool = False,
    ):
        """
        Plot the monosemanticity spectrum for a given layer.

        input args:
            monosemanticity: Float[Tensor, "n_neuron"], monosemanticity values
                for each neuron in the layer

            layer_name: str, name of the layer to plot

            save_flag: bool, whether to save the figure (default: False)
        """

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
        axes.set_xlabel("Neuron_index")
        axes.set_ylabel("Monosemanticity")

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.monosemanticity_dir}/monosemanticity_{layer_name}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

    def plotLine_n_mono_vs_layer(
        self, threshold: float = 0.5, n_features: float = 27, save_flag: bool = True
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer.

        input args:
            threshold: float, threshold for monosemanticity
                (default: 0.5)

            n_features: int, number of features per layer
                (default: 27 -> n_rds_types * n_dotDens
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)

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
            "Monosemantic neurons per feature across Layer\n"
            + f"({self.binocular_interaction}, seed={self.seed})",
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
                f"{self.monosemanticity_dir}/monosemanticity_vs_layer.pdf",
                dpi=600,
                bbox_inches="tight",
            )

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

    def plotLine_n_mono_vs_layer_all_interactions(
        self,
        threshold: float = 0.9,
        n_features: float = 27,
        save_flag: bool = True,
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer,
        foir all binocular interactions.

        input args:
            interaction: str, binocular interaction to analyze
                (e.g., "default", "bem", "cmm", "sum_diff")

            threshold: float, threshold for monosemanticity
                (default: 0.9)

            n_features: int, number of features per layer
                (default: 27 -> n_rds_types * n_dotDens
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)

            save_flag: bool, whether to save the figure (default: False)
        """

        interactions = ["default", "bem", "cmm", "sum_diff"]
        mono_per_feat_all_interactions = torch.empty(
            (
                len(interactions),
                len(self.config.seed_to_analyse),
                len(self.layer_name),
            ),
            dtype=torch.float32,
        )

        for i, interaction in enumerate(interactions):
            for s, seed in enumerate(self.config.seed_to_analyse):
                if interaction == "default":
                    epoch, iter = self.config.epoch_iter_to_load_default[s]
                elif interaction == "bem":
                    epoch, iter = self.config.epoch_iter_to_load_bem[s]
                elif interaction == "cmm":
                    epoch, iter = self.config.epoch_iter_to_load_cmm[s]
                else:  # sum_diff
                    epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

                # update network configuration and directory addresses
                self.update_network_config(interaction, seed, epoch, iter)

                # update model
                self.load_model()

                for j in range(len(self.layer_name)):
                    layer_name = self.layer_name[j]
                    monosemanticity = self.compute_monosemanticity(layer_name)

                    # threshold the monosemanticity
                    mono_thresh = torch.where(monosemanticity > threshold, 1.0, 0.0)
                    mono_per_feat = mono_thresh.sum() / n_features
                    mono_per_feat_all_interactions[i, s, j] = mono_per_feat

        # average across seeds
        mono_avg = mono_per_feat_all_interactions.mean(dim=1)  # [interaction, n_layers]
        mono_sem = sem(
            mono_per_feat_all_interactions, axis=1
        )  # [interaction, n_layers]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (8, 8)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Monosemantic neurons per feature across Layer\navg across seeds",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]
        for i in range(len(interactions)):
            axes.plot(
                mono_avg[i], linewidth=2, marker="o", color=colors[i], label=labels[i]
            )
            axes.fill_between(
                np.arange(len(mono_avg[i])),
                mono_avg[i] - mono_sem[i],
                mono_avg[i] + mono_sem[i],
                color=colors[i],
                alpha=0.2,
            )

        y_up = 0.41
        y_low = 0.0
        y_step = 0.1
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Layer")
        axes.set_ylabel("n_mono/n_feat")
        axes.set_xticks(np.arange(len(self.layer_name)))
        axes.set_xticklabels(self.layer_name, rotation=45, ha="right")
        axes.set_xlim(-0.5, len(self.layer_name) - 0.5)

        plt.legend(
            loc="upper left",
            fontsize=20,
            frameon=False,
        )

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.group_dir}/plots/monosemanticity_vs_layer_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
