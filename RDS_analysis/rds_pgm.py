# %% load necessary packages
import torch
from torch import Tensor
from torch.utils.data import SequentialSampler

import os
import gc
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from jaxtyping import Float, Int

from config.config_gcnet_lr import ConfigGCNet
from RDS_analysis.rds_analysis_v2 import RDSAnalysis, RDSBankDataset

# from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
from utilities.misc import NestedTensor


# %%
class PGM(RDSAnalysis):
    """
    Probabilistic graphical model (PGM) for analysing the behavioral responses of GCNet
    that includes variance and expected value of model's output as a function of dot
    density.

    The model based on PGM is relatively simple for this case.
    There are 3 random variables (RVs):
    1. X: input images: {cRDS, hmRDS, aRDS}
    2. Y: eye reference: {Left, Right}
    3. Z: disparity

    Essentially we want to compute P(Z|X):
    P(Z|X) = P((Z|X), Y=Left) + P((Z|X), Y=Right)
           = P(Z|X, Y=Left) * P(Y=Left|X) + P(Z|X, Y=Right) * P(Y=Right|X)
    """

    def __init__(self, config: ConfigGCNet) -> None:

        super().__init__(config)

        # create folders for pgm analysis
        self._create_pgm_dir()

    def _create_pgm_dir(self):

        # create folders for pgm data
        self.pgm_dir = os.path.join(
            self.experiment_dir, f"pgm_iter_{self.config.iter_to_load}"
        )
        if not os.path.exists(self.pgm_dir):
            os.makedirs(self.pgm_dir)

        # create folders for pgm plots
        self.pgm_plot_dir = os.path.join(self.pgm_dir, "Plots")
        if not os.path.exists(self.pgm_plot_dir):
            os.makedirs(self.pgm_plot_dir)

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
        # batch_size_rds_old = self.batch_size_rds

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

        # update folders for pgm
        self._create_pgm_dir()

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
            + f"PGM directory: {self.pgm_dir}\n"
            + "==============================================================\n"
        )

    @torch.inference_mode()
    def compute_ref_stats(
        self,
        rds_bank: RDSBankDataset,
        roi: Int[np.ndarray, "4"] = (
            64,  # y_start
            192,  # y_end
            128,  # x_start
            384,  # x_end
        ),
    ):
        """
        Compute some statistics (mean and variance) under both eye references

        args:
            rds_bank: RDS_bank dataloader
                RDS bank structure: [dotMatch dotDens disp_magnitude n_rds_each_disp]
                For example:
                    [0.0 0.1 10 rds_1
                                rds_2
                                .
                                .
                                .rds_(n_rds_each_disp)
                            -10 rds_1
                                rds_2
                                .
                                .
                                .rds_(n_rds_each_disp)
                    0.0 0.2 10 rds_1
                               rds_2
                               .
                               .
                               .rds_(n_rds_each_disp)
                    ]

            roi: region of interest in the center of RDSs
                [y_start, y_end, x_start, x_end]

        Returns:
            outputs{"roi_mean": [2, n_types, n_densities, n_trials],
                    "spatial_var": [2, n_types, n_densities, n_trials],
                    "posterior_var": [2, n_types, n_densities, n_trials],
                    "labels"}

                note: Axis 0 denotes the left/right reference:
                    0 = left reference (+1)
                    1 = right reference (-1)
        """

        # model = getattr(analysis.model, "_orig_mod", analysis.model)
        self.model.eval()
        device = next(self.model.parameters()).device

        dataset = rds_bank.dataset
        if not isinstance(dataset, RDSBankDataset):
            raise TypeError("Use the DataLoader returned by create_rds_bank")
        if not isinstance(rds_bank.sampler, SequentialSampler) or rds_bank.drop_last:
            raise ValueError(
                "The RDS bank must use sequential sampling and drop_last=False"
            )
        condition_shape = dataset.condition_shape
        n_samples = len(dataset)

        # allocate buffer
        ref_stats = {
            name: np.empty(
                (2, n_samples),
                dtype=np.float32,
            )
            for name in (
                "roi_mean",
                "spatial_var",
                "posterior_var",
            )
        }

        labels = np.empty(n_samples, dtype=np.int16)

        y0, y1, x0, x1 = roi
        offset = 0
        tepoch = tqdm(rds_bank, desc="Predicting RDS")
        for img_left, img_right, target_disp in tepoch:

            img_left = img_left.to(device, non_blocking=True)
            img_right = img_right.to(device, non_blocking=True)

            B = img_left.shape[0]
            labels[offset : offset + B] = target_disp.cpu().numpy()

            # iterate over left and right references
            for index, ref_value in enumerate((1, -1)):

                # Preserve physical left/right identity.
                ref = torch.full((B,), ref_value, dtype=torch.long, device=device)
                x = NestedTensor(left=img_left, right=img_right, ref=ref)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat_left, feat_right = self.model.encoder(x)

                    # posterior P(Z|X) [B, D, H, W]
                    posterior = self.model.decoder(feat_left, feat_right, ref)
                posterior = posterior.float()

                # Expected disparity map E[Z] = SUM(z * P(Z=z|X)), [B, H, W]
                mu = (posterior * self.model.disp_indices).sum(dim=1)

                # Posterior variance map, [B, H, W]
                posterior_variance = (
                    posterior * (self.model.disp_indices - mu.unsqueeze(1)).square()
                ).sum(dim=1)
                mu_roi = mu[:, y0:y1, x0:x1]
                pv_roi = posterior_variance[:, y0:y1, x0:x1]

                # ROI-mean disparity for each trial
                mean_per_trial = mu_roi.mean(dim=(-2, -1))

                # Spatial variance within each trial
                spatial_per_trial = mu_roi.var(dim=(-2, -1), unbiased=False)

                # Mean posterior variance within ROI
                posterior_per_trial = pv_roi.mean(dim=(-2, -1))

                sl = slice(offset, offset + B)
                ref_stats["roi_mean"][index, sl] = mean_per_trial.cpu().numpy()
                ref_stats["spatial_var"][index, sl] = spatial_per_trial.cpu().numpy()
                ref_stats["posterior_var"][
                    index, sl
                ] = posterior_per_trial.cpu().numpy()

            offset += B

        assert offset == n_samples

        ref_stats = {
            key: values.reshape(2, *condition_shape)
            for key, values in ref_stats.items()
        }
        ref_stats["labels"] = labels.reshape(*condition_shape)

        return ref_stats

    def compute_monocular_weight(self, ref_stats):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)

        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # variance across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        var_near = roi_mean[..., :n].var(axis=-1, ddof=1)
        var_far = roi_mean[..., n:].var(axis=-1, ddof=1)

        # compute weight
        eps = 1e-8

        # Each has shape [RDS type, density]
        v_left_near = var_near[0]
        v_right_near = var_near[1]

        v_left_far = var_far[0]
        v_right_far = var_far[1]

        # weight for left-reference
        # [dotMatch, dotDens]
        w_left_near = v_right_near / (v_left_near + v_right_near + eps)
        w_left_far = v_right_far / (v_left_far + v_right_far + eps)

        # weight for right-reference
        # [dotMatch, dotDens]
        w_right_near = 1.0 - w_left_near
        w_right_far = 1.0 - w_left_far

        w_near = np.asarray([w_left_near, w_right_near])
        w_far = np.asarray([w_left_far, w_right_far])

        return w_near, w_far

    def compute_expected_disp(self, ref_stats, w_near, w_far):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)
        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # average across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        mean_near = roi_mean[..., :n].mean(axis=-1)
        mean_far = roi_mean[..., n:].mean(axis=-1)

        # compute expected disparity, [dotMatch, dotDens]
        # rho = np.repeat([[0.0], [0.5], [1.0]], 9, axis=1)
        # E_near = w_left_near * mean_near[0] + w_right_near * mean_far[1]
        # E_far = w_left_far * mean_far[0] + w_right_far * mean_near[1]
        E_near = w_near[0] * mean_near[0] + w_near[1] * mean_far[1]
        E_far = w_far[0] * mean_far[0] + w_far[1] * mean_near[1]

        return E_near, E_far

    def plot_expected_disp(self, E_near, E_far, save_flag: bool = True):

        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharey=True)

        dotDens = np.asarray(self.dotDens_list)
        rds_names = ["aRDS", "hmRDS", "cRDS"]

        for ax, (condition, values) in zip(
            axes,
            [("Near", E_near), ("Far", E_far)],
        ):
            for c, name in enumerate(rds_names):

                ax.plot(
                    dotDens,
                    values[c],
                    marker="o",
                    label=f"{name}",
                )

            ax.set_xlabel("Dot density")
            ax.set_title(condition)
            ax.set_xticks(dotDens)
            # ax.grid(alpha=0.2)

        axes[0].set_ylabel("Expected disparity of ROI (px)")
        axes[1].legend(fontsize=8, ncol=2)

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            plt.savefig(
                f"{self.pgm_plot_dir}/plot_expected_disp.pdf",
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

    def plot_mean_disp_unweighted(self, ref_stats, save_flag: bool = True):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)
        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # average across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        mean_near = roi_mean[..., :n].mean(axis=-1)
        mean_far = roi_mean[..., n:].mean(axis=-1)

        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharey=True)

        dotDens = np.asarray(self.dotDens_list)
        rds_names = ["aRDS", "hmRDS", "cRDS"]

        for ax, (condition, values) in zip(
            axes,
            [("Near", mean_near), ("Far", mean_far)],
        ):
            for c, name in enumerate(rds_names):

                ax.plot(
                    dotDens,
                    values[0, c],
                    marker="o",
                    label=f"{name} - left ref",
                )

                ax.plot(
                    dotDens,
                    values[1, c],
                    marker="s",
                    linestyle="--",
                    label=f"{name} - right ref",
                )

            ax.set_xlabel("Dot density")
            ax.set_title(condition)
            ax.set_xticks(dotDens)
            # ax.grid(alpha=0.2)

        axes[0].set_ylabel("Across-trial average of ROI mean (px²)")
        axes[1].legend(fontsize=8, ncol=2)

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            plt.savefig(
                f"{self.pgm_plot_dir}/plot_mean_disp_unweighted.pdf",
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

    def plot_variance(self, ref_stats, save_flag: bool = True):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)

        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # variance across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        var_near = roi_mean[..., :n].var(axis=-1, ddof=1)
        var_far = roi_mean[..., n:].var(axis=-1, ddof=1)

        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharey=True)

        dotDens = np.asarray(self.dotDens_list)
        rds_names = ["aRDS", "hmRDS", "cRDS"]

        for ax, (condition, values) in zip(
            axes,
            [("Near", var_near), ("Far", var_far)],
        ):
            for c, name in enumerate(rds_names):

                ax.plot(
                    dotDens,
                    values[0, c],
                    marker="o",
                    label=f"{name} - left ref",
                )

                ax.plot(
                    dotDens,
                    values[1, c],
                    marker="s",
                    linestyle="--",
                    label=f"{name} - right ref",
                )

            ax.set_xlabel("Dot density")
            ax.set_title(condition)
            ax.set_xticks(dotDens)
            ax.grid(alpha=0.2)

        axes[0].set_ylabel("Across-trial variance of ROI mean (px²)")
        axes[1].legend(fontsize=8, ncol=2)

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            plt.savefig(
                f"{self.pgm_plot_dir}/plot_variance.pdf",
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

    def plot_monocular_weight(self, w_near, w_far, save_flag: bool = True):

        dotDens = np.asarray(self.dotDens_list)
        rds_names = ["aRDS", "hmRDS", "cRDS"]

        w_left_near = w_near[0]
        w_left_far = w_far[0]

        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharey=True)

        for ax, (condition, weights) in zip(
            axes,
            [
                ("Near", w_left_near),
                ("Far", w_left_far),
            ],
        ):
            for c, name in enumerate(rds_names):
                ax.plot(
                    dotDens,
                    weights[c],
                    marker="o",
                    label=name,
                )

            ax.axhline(0.5, color="red", linestyle="--")

            ax.set_title(condition)
            ax.set_xlabel("Dot density")
            ax.set_ylim(0, 1)
            # ax.grid(alpha=0.2)

        axes[0].set_ylabel("Monocular weight (left-reference)")
        axes[1].legend()

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag:
            plt.savefig(
                f"{self.pgm_plot_dir}/plot_monocular_weight.pdf",
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
