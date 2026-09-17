"""
Script for running a group analysis of rds_analysis results.

working directory: BNN
"""

# %% load necessary modules
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

from config.config_bnn import ConfigBNN
from config.config_gcnet import ConfigGCNet
from RDS_analysis.rds_analysis import RDSAnalysis


# %%
class GA_RDS(RDSAnalysis):

    def __init__(self, config: ConfigBNN | ConfigGCNet) -> None:

        super().__init__(config)

        self.dataset = config.dataset
        self.binocular_interaction = config.binocular_interaction
        self.seed = config.seed
        self.epoch = config.epoch_to_load
        self.iter = config.iter_to_load
        self.device = config.device

        # directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # print out network configuration
        self.__getconfig__()
        self.__getconfig_rds___()

    def update_network_config(
        self, interaction: str, seed: int, epoch: int, iter: int
    ) -> None:
        """
        Update the network configuration and directories for storing
        the results
        """

        # old config, for printing purposes
        interaction_old = self.binocular_interaction
        seed_old = self.seed
        epoch_old = self.epoch
        iter_old = self.iter

        # update binocular_interaction, seed, epoch, iter, and model_pretrained in
        # the class and config
        self.binocular_interaction = interaction
        self.config.binocular_interaction = interaction
        self.seed = seed
        self.config.seed = seed
        self.config.experiment_id = seed
        self.epoch = epoch
        self.config.epoch_to_load = epoch
        self.iter = iter
        self.config.iter_to_load = iter
        self.model_pretrained = f"epoch_{self.epoch}_iter_{self.iter}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
        self.config.model_pretrained = self.model_pretrained

        # update the experiment directories based on the new interaction
        self.experiment_dir = (
            f"{self.model_name}/run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"experiment_{self.seed}"
        )

        # update directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # update folders for rds analysis
        self.make_rds_dirs()

        print(
            "==============================================================\n"
            + f"Updating {self.model_name} config:\n"
            + "==============================================================\n"
            + f"Binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"Seed: {seed_old} => {self.config.seed}\n"
            + f"Epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"Iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"Experiment directory: {self.experiment_dir}\n"
            + f"RDS directory: {self.rds_dir}\n"
            + f"Cross-decoding directory: {self.xDecode_dir}\n"
            + "==============================================================\n"
        )

    def compute_disp_map_all_seeds(
        self, interaction: str, n_bootstrap: int = 1000
    ) -> None:
        """
        Predict RDS disparity maps for all seeds and a binocular interaction.

        """

        for s, seed in enumerate(self.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = self.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = self.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = self.config.epoch_iter_to_load_cmm[s]
            elif interaction == "sum_diff":
                epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]
            else:
                raise ValueError(
                    f"Invalid binocular interaction: {interaction}!\n"
                    + "Only one of these interactions are allowed: [default, bem, cmm, sum_diff]"
                )

            # update network configuration and directory addresses
            self.update_network_config(interaction, seed, epoch, iter)

            # update model
            self._load_pretrained_model()
            self.model.to(self.device)

            # compute model responses to RDSs
            self.compute_disp_map_rds_group(
                self.dotDens_list, self.background_flag, self.pedestal_flag
            )

            # cross-decoding analysis with SVM
            self.xDecode(self.dotDens_list, n_bootstrap, self.background_flag)

    def plotLine_xDecode_all_seeds(self, interaction: str, save_flag: bool = False):

        # gather data all seeds
        score_ards_all = []
        score_hmrds_all = []
        score_crds_all = []
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

            # load cross-decoding data of a seed
            # [n_bootstrap, len(dotDens_list)]
            score_ards_bootstrap = np.load(
                f"{self.xDecode_dir}/score_ards_bootstrap.npy"
            )
            score_hmrds_bootstrap = np.load(
                f"{self.xDecode_dir}/score_hmrds_bootstrap.npy"
            )
            score_crds_bootstrap = np.load(
                f"{self.xDecode_dir}/score_crds_bootstrap.npy"
            )

            score_ards_all.append(score_ards_bootstrap)
            score_hmrds_all.append(score_hmrds_bootstrap)
            score_crds_all.append(score_crds_bootstrap)

        # stack and average across bootstrap
        score_ards = np.stack(score_ards_all).mean(
            axis=1
        )  # [n_seed, n_bootstrap, n_dotDens]
        score_hmrds = np.stack(score_hmrds_all).mean(axis=1)
        score_crds = np.stack(score_crds_all).mean(axis=1)

        # average across seeds
        score_ards_avg = np.mean(score_ards, axis=0)
        score_ards_sem = sem(score_ards, axis=0)
        score_hmrds_avg = np.mean(score_hmrds, axis=0)
        score_hmrds_sem = sem(score_hmrds, axis=0)
        score_crds_avg = np.mean(score_crds, axis=0)
        score_crds_sem = sem(score_crds, axis=0)

        # plotting
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
            "BNN depth performance all seeds \n"
            + f"({self.config.binocular_interaction})",
            ha="center",
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        colors = ["#6a5acd", "#00CED1", "#333333"]
        x = np.arange(10, 100, 10)
        # ards
        axes.plot(
            x, score_ards_avg, linewidth=2, color=colors[0], label="cRDS vs. aRDS"
        )
        axes.plot(x, score_ards_avg, "o", markersize=12, color=colors[0])
        axes.fill_between(
            x,
            score_ards_avg - score_ards_sem,
            score_ards_avg + score_ards_sem,
            color=colors[0],
            alpha=0.2,
        )

        # hmrds
        axes.plot(
            x, score_hmrds_avg, linewidth=2, color=colors[1], label="cRDS vs. hmRDS"
        )
        axes.plot(x, score_hmrds_avg, "o", markersize=12, color=colors[1])
        axes.fill_between(
            x,
            score_hmrds_avg - score_hmrds_sem,
            score_hmrds_avg + score_hmrds_sem,
            color=colors[1],
            alpha=0.2,
        )

        # crds
        axes.plot(x, score_crds_avg, linewidth=2, color=colors[2], label="cRDS")
        axes.plot(x, score_crds_avg, "o", markersize=12, color=colors[2])
        axes.fill_between(
            x,
            score_crds_avg - score_crds_sem,
            score_crds_avg + score_crds_sem,
            color=colors[2],
            alpha=0.2,
        )

        # plot chance level
        axes.hlines([0.5], xmin=0, xmax=100, colors="red", linestyles="--", linewidth=3)

        x_low = 0
        x_up = 105
        x_step = 20
        y_low = 0.0
        y_up = 1.1
        y_step = 0.2

        axes.set_xlabel("Dot density (%)")
        axes.set_ylabel("Prediction acc.")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="lower right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.plot_dir}/plotLine_{self.model_name}_rds_xDecode_{interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
