# %% load necessary packages
import torch
from torch import Tensor
from torch.utils.data import SequentialSampler

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from pathlib import Path
from tqdm import tqdm
from scipy.stats import sem
from scipy.special import expit
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
        self.save_dir = (
            f"{self.model_name}/run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
        )
        self.experiment_dir = f"{self.save_dir}/" + f"experiment_{self.seed}"

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
            outputs{"roi_mean": [2, dotMatch, dotDens, 2*n_rds_each_disp],
                    "spatial_var": [2, dotMatch, dotDens, 2*n_rds_each_disp],
                    "posterior_var": [2, dotMatch, dotDens, 2*n_rds_each_disp],
                    "posterior_near": [2, dotMatch, dotDens, 2*n_rds_each_disp, h_roi, w_roi],
                    "posterior_far": [2, dotMatch, dotDens, 2*n_rds_each_disp, h_roi, w_roi],
                    "labels": [dotMatch, dotDens, 2*n_rds_each_disp]}

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
        condition_shape = (
            dataset.condition_shape
        )  # [dotMatch, dotDens, 2*n_rds_each_disp]
        n_samples = len(dataset)

        # allocate buffer
        y0, y1, x0, x1 = roi
        # h_roi = y1 - y0
        # w_roi = x1 - x0
        ref_stats = {
            name: np.empty(
                (2, n_samples),
                dtype=np.float32,
            )
            for name in (
                "roi_mean",
                "spatial_var",
                "posterior_near",
                "posterior_far",
                "posterior_var",
            )
        }
        ref_stats["posterior_roi"] = np.empty(
            (2, n_samples, self.model.disp_indices.shape[1]), dtype=np.float32
        )
        # ref_stats["posterior_near"] = np.empty((2, n_samples), dtype=np.float32)
        # ref_stats["posterior_far"] = np.empty((2, n_samples), dtype=np.float32)
        labels = np.empty(n_samples, dtype=np.int16)

        # disparity bins: [1, D, 1, 1] buffer, plus 1-D version for masking
        # disp_indices = self.model.disp_indices  # [1, D, 1, 1]
        disp_1d = self.model.disp_indices.flatten()  # [D]
        bins_near = disp_1d > 0
        bins_far = disp_1d < 0

        y0, y1, x0, x1 = roi
        offset = 0
        tepoch = tqdm(rds_bank, desc="Predicting RDS")
        for img_left, img_right, target_disp in tepoch:

            img_left = img_left.to(device, non_blocking=True)
            img_right = img_right.to(device, non_blocking=True)

            B = img_left.shape[0]
            labels[offset : offset + B] = target_disp.cpu().numpy()

            # iterate over left and right references
            for ref_idx, ref_value in enumerate((1, -1)):

                # Preserve physical left/right identity.
                ref = torch.full((B,), ref_value, dtype=torch.long, device=device)
                if ref_value > 0:
                    x = NestedTensor(
                        left=img_left,
                        right=img_right,
                        ref=ref,
                    )
                else:  # swap inputs for negative reference because for right-reference: near -> far, far -> near
                    x = NestedTensor(
                        left=img_right,
                        right=img_left,
                        ref=ref,
                    )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat_left, feat_right = self.model.encoder(x)

                    # logits or posterior P(Z|X) [B, D, H, W]
                    posterior = self.model.decoder(feat_left, feat_right, ref)

                # select pixels within ROI, [B, D, H_roi, W_roi]
                posterior = posterior[:, :, y0:y1, x0:x1].float()

                # average posterior within roi, [B, D]
                posterior_roi = posterior.mean(dim=(-2, -1))

                # Expected disparity map E[Z] = SUM(z * P(Z=z|X)), [B, H_roi, W_roi]
                mu_roi = (posterior * self.model.disp_indices).sum(dim=1)

                # Posterior variance map within ROI, [B, H_roi, W_roi]
                posterior_var_roi = (
                    posterior * (self.model.disp_indices - mu_roi.unsqueeze(1)).square()
                ).sum(dim=1)

                # ROI-mean (spatial-average) disparity for each trial, [B]
                mean_per_trial = mu_roi.mean(dim=(-2, -1))

                # Spatial variance within each trial, [B]
                spatial_per_trial = mu_roi.var(dim=(-2, -1), unbiased=False)

                # Mean posterior variance within ROI, [B]
                posterior_per_trial = posterior_var_roi.mean(dim=(-2, -1))

                # For each pixel at (x, y) in each RDS image, find
                # posterior_near Q(posterior(Z > 0|X, dotMatch, dotDens)) and
                # posterior_far Q(posterior(Z < 0|X, dotMatch, dotDens)) then
                # sum the posterior probability over disparity-axis
                # [B, H_roi, W_roi]
                # p_near_roi = posterior[:, bins_near].sum(dim=1)
                # p_far_roi = posterior[:, bins_far].sum(dim=1)

                # spatially-averaged within roi, [B]
                # p_near = p_near_roi.mean(dim=(-2, -1))
                # p_far = p_far_roi.mean(dim=(-2, -1))

                # # pixelwise log-odds: log_post_near - log_post_far, [B, H_row * W_row]
                # # this measures how strong the network believes that particular image
                # # is near
                # log_odds = torch.log(p_near_roi) - torch.log(p_far_roi)
                # # spatially average within roi log_odds
                # lambda_log = log_odds.mean(dim=(-2, -1))  # [B]

                sl = slice(offset, offset + B)
                ref_stats["roi_mean"][ref_idx, sl] = mean_per_trial.cpu().numpy()
                ref_stats["spatial_var"][ref_idx, sl] = spatial_per_trial.cpu().numpy()
                ref_stats["posterior_roi"][ref_idx, sl] = posterior_roi.cpu().numpy()
                # ref_stats["posterior_near"][ref_idx, sl] = p_near.cpu().numpy()
                # ref_stats["posterior_far"][ref_idx, sl] = p_far.cpu().numpy()
                ref_stats["posterior_var"][
                    ref_idx, sl
                ] = posterior_per_trial.cpu().numpy()
                # ref_stats["lambda_log"][ref_idx, sl] = lambda_log.cpu().numpy()

            offset += B

        assert offset == n_samples

        ref_stats = {
            key: values.reshape(2, *condition_shape, *values.shape[2:])
            for key, values in ref_stats.items()
        }  # [ref, dotMatch, dotDens, 2*n_rds_each_disp]
        ref_stats["labels"] = labels.reshape(
            *condition_shape
        )  # [dotMatch, dotDens, 2*n_rds_each_disp]

        # save (dict is pickled: load with np.load(path, allow_pickle=True).item())
        suffix = ".npy"
        path = Path(self.pgm_dir) / ("ref_stats" + suffix)
        np.save(path, ref_stats)

        return ref_stats

    def compute_reference_weight(self, ref_stats):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)

        roi_mean = ref_stats[
            "roi_mean"
        ]  # [reference, dotMatch, dotDens, 2*n_rds_each_disp]

        # variance across n_RDSs (including near and far)
        # [reference, dotMatch, dotDens]
        var = roi_mean.var(axis=-1, ddof=1)

        # compute weight
        eps = 1e-8

        # Each has shape [RDS type, density]
        var_left = var[0]
        var_right = var[1]

        # weight for left-reference
        # [dotMatch, dotDens]
        w_left = var_right / (var_left + var_right + eps)

        # weight for right-reference
        # [dotMatch, dotDens]
        w_right = 1.0 - w_left

        w_ref = np.asarray([w_left, w_right])  # [2, dotMatch, dotDens]

        # save
        suffix = ".npy"
        path = Path(self.pgm_dir) / ("reference_weight" + suffix)
        np.save(path, w_ref)

        return w_ref

    def weighted_posterior(
        self,
        posterior: Float[np.ndarray, "ref dotMatch dotDens n_rds disp"],
        w_ref: Float[np.ndarray, "ref dotMatch dotDens"],
        disp_indices: Int[np.ndarray, "disp"],
    ):
        """
        weighted-average posterior_near and posterior_far based on eye-reference weights.

        Args:
            posterior [ref, dotMatch, dotDens, 2*n_rds_each_disp, disp]: posterior from gcnet
            w_ref [ref, dotMatch, dotDens]: weights based on eye-reference
            disp_indices [disp]: np.array([-96, -95, ..., 95])

        Returns:
            post_weighted [dotMatch, dotDens, 2*n_rds_each_disp, disp]
            post_near_weighted [dotMatch, dotDens, 2*n_rds_each_disp]
            post_far_weighted [dotMatch, dotDens, 2*n_rds_each_disp]
        """

        # weighted-sum posterior based on left and right reference
        post_weighted = np.einsum(
            "rmd, rmdnp -> mdnp", w_ref, posterior, optimize=True
        )  # [dotMatch, dotDens, n_rds, disp]

        # get mask for near and far disparities
        mask_near = disp_indices > 0
        mask_far = disp_indices < 0

        # get near and far posterior then sum along disparity axis
        post_near_weighted = post_weighted[:, :, :, mask_near].sum(
            axis=-1
        )  # [dotMatch, dotDens, n_rd]
        post_far_weighted = post_weighted[:, :, :, mask_far].sum(
            axis=-1
        )  # [dotMatch, dotDens, n_rds]

        return post_weighted, post_near_weighted, post_far_weighted

    def compute_expected_disp(self, ref_stats, w_ref):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)
        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # average across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        mean_far = roi_mean[..., :n].mean(axis=-1)
        mean_near = roi_mean[..., n:].mean(axis=-1)

        # compute expected disparity, [dotMatch, dotDens]
        # rho = np.repeat([[0.0], [0.5], [1.0]], 9, axis=1)
        # E_near = w_ref[0] * mean_near[0] + w_ref[1] * mean_far[1]
        # E_far = w_ref[0] * mean_far[0] + w_ref[1] * mean_near[1]
        E_near = w_ref[0] * mean_near[0] + w_ref[1] * mean_near[1]
        E_far = w_ref[0] * mean_far[0] + w_ref[1] * mean_far[1]
        # E_near = w_ref[0] * mean_near[0] + w_ref[1] * rho * mean_near[1]
        # E_far = w_ref[0] * rho * mean_far[0] + w_ref[1] * mean_far[1]

        # save
        suffix = ".npy"
        path = Path(self.pgm_dir) / ("expected_disp_near" + suffix)
        np.save(path, E_near)
        path = Path(self.pgm_dir) / ("expected_disp_far" + suffix)
        np.save(path, E_far)

        return E_near, E_far

    def compute_deltaLog(self, ref_stats, w_ref):

        # get roi-averaged posterior
        posterior = ref_stats[
            "posterior_roi"
        ]  # [ref, dotMatch, dotDens, 2*n_rds_each_disp, disp]

        # compute weighted-sum of posterior based on reference-weight
        disp_indices = self.model.disp_indices.flatten().cpu().numpy()
        post_weighted, post_near_weighted, post_far_weighted = self.weighted_posterior(
            posterior, w_ref, disp_indices
        )  # [dotMatch, dotDens, 2*n_rds_each_disp]

        # compute delta log
        eps = 1e-12
        deltaLog_odd = np.log(np.maximum(post_near_weighted, eps)) - np.log(
            np.maximum(post_far_weighted, eps)
        )  # [dotMatch, dotDens, 2*n_rds_each_disp]

        return deltaLog_odd

    def compute_prob_x1_x2(
        self,
        deltaLog_x1: Float[np.ndarray, "n_samples none"],
        deltaLog_x2: Float[np.ndarray, "none n_samples"],
        lapse: float = 0.01,
        beta: float = -0.75,
        pos_bias: float = -2,
    ) -> Float[np.ndarray, "n_samples n_samples"]:

        delta = deltaLog_x1 - deltaLog_x2
        prob_state = lapse + (1 - lapse) * expit(beta * delta + pos_bias)

        return prob_state

    def simulate_2afc(
        self, ref_stats, w_ref: Float[np.ndarray, "ref dotMatch dotDens"]
    ):

        # compute deltaLog, [dotMatch, dotDens, 2*n_rds_each_disp]
        deltaLog_odd = self.compute_deltaLog(ref_stats, w_ref)

        disp_labels = ref_stats["labels"]  # [dotMatch, dotDens, 2*n_rds_each_disp]

        n_dotMatch, n_dotDens, _ = deltaLog_odd.shape
        p_corrects = np.empty((n_dotMatch, n_dotDens), dtype=np.float32)
        p_correct_std = np.empty_like(p_corrects)
        for dm in range(n_dotMatch):
            for dd in range(n_dotDens):

                idx_near = np.flatnonzero(disp_labels[dm, dd] > 0)
                idx_far = np.flatnonzero(disp_labels[dm, dd] < 0)

                deltaLog_near = deltaLog_odd[dm, dd, idx_near]  # [n_rds]
                deltaLog_far = deltaLog_odd[dm, dd, idx_far]  # [n_rds]

                # state 1: x1 = near, x2 = far (x1 is the correct response)
                p_correct = self.compute_prob_x1_x2(
                    deltaLog_near[:, None], deltaLog_far[None, :]
                )  # [n_rds, n_rds]

                use = ~np.eye(len(idx_near), dtype=bool)
                p_corrects[dm, dd] = p_correct[use].mean()
                p_correct_std[dm, dd] = p_correct[use].std()

        # save data
        suffix = ".npy"
        path = Path(self.pgm_dir) / ("p_2afc" + suffix)
        np.save(path, p_corrects)
        path = Path(self.pgm_dir) / ("p_2afc_std" + suffix)
        np.save(path, p_correct_std)

        return p_corrects, p_correct_std

    def plot_expected_disp(
        self,
        E_near: Float[np.ndarray, "dotMatch dotDens"],
        E_far: Float[np.ndarray, "dotMatch dotDens"],
        save_flag: bool = True,
    ):

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
        # plt.cla()
        # # Clear the current figure.
        # plt.clf()
        # # Closes all the figure windows.
        # plt.close("all")
        # plt.close(fig)
        # gc.collect()

    def plot_expected_disp_seeds(self, save_flag: bool = True):
        """
        Plot expected_disp averaged across all seed numbers for a given
        binocular interaction

        """

        E_near_seeds = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.config.dotMatch_list),
                len(self.config.dotDens_list),
            ),
            dtype=np.float32,
        )
        E_far_seeds = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.config.dotMatch_list),
                len(self.config.dotDens_list),
            ),
            dtype=np.float32,
        )

        # load expected_disp_near and far all seeds
        for s, seed in enumerate(self.config.seed_to_analyse):

            # interaction = "sum_diff"
            if self.binocular_interaction == "default":
                epoch2load, iter2load = self.config.epoch_iter_to_load_default[s]
            elif self.binocular_interaction == "bem":
                epoch2load, iter2load = self.config.epoch_iter_to_load_bem[s]
            elif self.binocular_interaction == "cmm":
                epoch2load, iter2load = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch2load, iter2load = self.config.epoch_iter_to_load_sum_diff[s]
            # s = 0

            path = Path(self.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
            E_near = np.load(f"{path}/expected_disp_near.npy")  # [dotMatch, dotDens]
            E_far = np.load(f"{path}/expected_disp_far.npy")

            # store to buffer
            E_near_seeds[s] = E_near
            E_far_seeds[s] = E_far

        # average across seeds, [dotMatch, dotDens]
        E_near_mean = E_near_seeds.mean(axis=0)
        E_far_mean = E_far_seeds.mean(axis=0)
        E_near_sem = sem(E_near_seeds, axis=0)
        E_far_sem = sem(E_far_seeds, axis=0)

        # start plotting
        stim_labels = ["aRDS", "hmRDS", "cRDS"]
        markersize = 8

        sns.set_theme(context="paper", style="ticks", font_scale=2, palette="deep")

        figsize = (9, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Expected disparity (interaction: {self.binocular_interaction})",
            ha="center",
            fontweight="semibold",
        )
        fig.text(-0.05, 0.5, "Expected disp [px]", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot E_near
        colors = ["#6a5acd", "#00CED1", "#333333"]
        x = np.array(self.dotDens_list)
        for i in range(len(self.config.dotMatch_list)):
            y = E_near_mean[i]
            y_err = E_near_sem[i]

            axes.plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="-",
            )

            # plot the marker
            axes.plot(x, y, "o", markersize=markersize, c=colors[i])

            axes.fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        ## plot E_far
        for i in range(len(self.config.dotMatch_list)):
            y = E_far_mean[i]
            y_err = E_far_sem[i]

            axes.plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="--",
            )

            # plot the marker
            axes.plot(x, y, "o", markersize=markersize, c=colors[i])

            axes.fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        # plot zero disparity
        axes.hlines(
            y=0.0, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2
        )

        x_low = 0.0
        x_up = 1.05
        x_step = 0.1
        y_low = -15.0
        y_up = 15.5
        y_step = 5

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        ## legend
        # one legend explaining colour (stimulus) and one explaining line style
        stim_handles = [
            Line2D(
                [0],
                [0],
                color=c,
                lw=3,
                marker="o",
                markersize=markersize,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            for c in colors
        ]
        style_handles = [
            Line2D([0], [0], color="gray", lw=2.5, ls="-"),
            Line2D([0], [0], color="gray", lw=2.5, ls="--"),
        ]

        leg1 = axes.legend(
            stim_handles,
            stim_labels,
            title="Stimulus",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            alignment="left",
        )
        axes.add_artist(leg1)
        axes.legend(
            style_handles,
            ["Near", "Far"],
            title="Disparity",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.55),
            alignment="left",
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        if save_flag == 1:
            if not os.path.exists(f"{self.save_dir}/plots"):
                os.mkdir(f"{self.save_dir}/plots")
            fig.savefig(
                f"{self.save_dir}/plots/PlotLine_expected_disp_avg_seeds_{self.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plot_mean_disp_unweighted(self, ref_stats, save_flag: bool = True):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)
        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # average across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        mean_far = roi_mean[..., :n].mean(axis=-1)
        mean_near = roi_mean[..., n:].mean(axis=-1)

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

        # # Clear the current axes.
        # plt.cla()
        # # Clear the current figure.
        # plt.clf()
        # # Closes all the figure windows.
        # plt.close("all")
        # plt.close(fig)
        # gc.collect()

    def plot_pred_disp_variance(self, ref_stats, save_flag: bool = True):

        n = self.config.n_rds_each_disp
        # ensure that disparity labels are non-shuffled
        assert np.all(ref_stats["labels"][..., :n] > 0)
        assert np.all(ref_stats["labels"][..., n:] < 0)

        roi_mean = ref_stats["roi_mean"]  # [reference, RDS type, density, trial]

        # variance across trials (across n_rds_each_disp)
        # [reference, dotMatch, dotDens]
        var_far = roi_mean[..., :n].var(axis=-1, ddof=1)
        var_near = roi_mean[..., n:].var(axis=-1, ddof=1)

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
            # ax.grid(alpha=0.2)

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

    def plot_reference_weight(
        self, w_ref: Float[np.ndarray, "ref dotMatch dotDens"], save_flag: bool = True
    ):

        dotDens = np.asarray(self.dotDens_list)
        rds_names = ["aRDS", "hmRDS", "cRDS"]

        w_left = w_ref[0]
        w_right = w_ref[1]

        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharey=True)

        for ax, (condition, weights) in zip(
            axes,
            [
                ("Left-reference", w_left),
                ("Right-reference", w_right),
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
                f"{self.pgm_plot_dir}/plot_reference_weight.pdf",
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

    def plot_reference_weights_seeds(self, save_flag: bool = True):
        """
        Plot monocular weights averaged across all seed numbers for a given
        binocular interaction

        """

        w_left_seeds = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.config.dotMatch_list),
                len(self.config.dotDens_list),
            ),
            dtype=np.float32,
        )
        w_right_seeds = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.config.dotMatch_list),
                len(self.config.dotDens_list),
            ),
            dtype=np.float32,
        )

        # load expected_disp_near and far all seeds
        for s, seed in enumerate(self.config.seed_to_analyse):

            # interaction = "sum_diff"
            if self.binocular_interaction == "default":
                epoch2load, iter2load = self.config.epoch_iter_to_load_default[s]
            elif self.binocular_interaction == "bem":
                epoch2load, iter2load = self.config.epoch_iter_to_load_bem[s]
            elif self.binocular_interaction == "cmm":
                epoch2load, iter2load = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch2load, iter2load = self.config.epoch_iter_to_load_sum_diff[s]
            # s = 0

            # load reference weights
            path = Path(self.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
            w_ref = np.load(f"{path}/reference_weight.npy")  # [2, dotMatch, dotDens]

            # store to buffer
            w_left_seeds[s] = w_ref[0]
            w_right_seeds[s] = w_ref[1]

        # average across seeds, [dotMatch, dotDens]
        w_left_mean = w_left_seeds.mean(axis=0)
        w_right_mean = w_right_seeds.mean(axis=0)
        w_left_sem = sem(w_left_seeds, axis=0)
        w_right_sem = sem(w_right_seeds, axis=0)

        # start plotting
        stim_labels = ["aRDS", "hmRDS", "cRDS"]
        markersize = 8

        sns.set_theme(context="paper", style="ticks", font_scale=2, palette="deep")

        figsize = (15, 5)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Reference weight ({self.model_name}, interaction: {self.binocular_interaction})",
            ha="center",
            fontweight="semibold",
        )
        fig.text(-0.03, 0.5, "w_ref", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot w_left
        colors = ["#6a5acd", "#00CED1", "#333333"]
        x = np.array(self.dotDens_list)
        for i in range(len(self.config.dotMatch_list)):
            y = w_left_mean[i]
            y_err = w_left_sem[i]

            axes[0].plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="-",
            )

            # plot the marker
            axes[0].plot(x, y, "o", markersize=markersize, c=colors[i])

            axes[0].fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        # plot middle weight
        axes[0].hlines(
            y=0.5, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2
        )

        x_low = 0.0
        x_up = 1.05
        x_step = 0.1
        y_low = 0.3
        y_up = 0.65
        y_step = 0.1

        axes[0].set_title("Left-reference")
        axes[0].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes[0].set_xlim(x_low, x_up)
        axes[0].set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")

        ## plot w_right
        for i in range(len(self.config.dotMatch_list)):
            y = w_right_mean[i]
            y_err = w_right_sem[i]

            axes[1].plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="-",
            )

            # plot the marker
            axes[1].plot(x, y, "o", markersize=markersize, c=colors[i])

            axes[1].fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        # plot middle weight
        axes[1].hlines(
            y=0.5, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2
        )

        axes[1].set_title("Right-reference")
        axes[1].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes[1].set_xlim(x_low, x_up)
        axes[1].set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")

        ## legend
        # one legend explaining colour (stimulus) and one explaining line style
        stim_handles = [
            Line2D(
                [0],
                [0],
                color=c,
                lw=3,
                marker="o",
                markersize=markersize,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            for c in colors
        ]
        leg1 = axes[1].legend(
            stim_handles,
            stim_labels,
            frameon=False,
            loc="lower right",
            alignment="left",
        )
        axes[1].add_artist(leg1)

        if save_flag == 1:
            if not os.path.exists(f"{self.save_dir}/plots"):
                os.mkdir(f"{self.save_dir}/plots")
            fig.savefig(
                f"{self.save_dir}/plots/PlotLine_ref_weights_avg_seeds_{self.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_2afc_performance(
        self,
        p_2afc: Float[np.ndarray, "dotMatch dotDens"],
        p_2afc_std: Float[np.ndarray, "dotMatch dotDens"],
        save_flag: bool = True,
    ):
        # start plotting
        stim_labels = ["aRDS", "hmRDS", "cRDS"]
        markersize = 8

        sns.set_theme(context="paper", style="ticks", font_scale=2, palette="deep")

        figsize = (9, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"2-AFC performance ({self.model_name}, interaction: {self.binocular_interaction} / seed: {self.seed})",
            ha="center",
            fontweight="semibold",
        )
        fig.text(-0.05, 0.5, "Prediction accuracy", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot E_near
        colors = ["#6a5acd", "#00CED1", "#333333"]
        x = np.array(self.dotDens_list)
        for i in range(len(self.config.dotMatch_list)):
            y = p_2afc[i]
            y_err = p_2afc_std[i]

            axes.plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="-",
            )

            # plot the marker
            axes.plot(x, y, "o", markersize=markersize, c=colors[i])

            axes.fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        # plot chance level
        axes.hlines(
            y=0.5, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2
        )

        x_low = 0.0
        x_up = 1.05
        x_step = 0.1
        y_low = 0.0
        y_up = 1.05
        y_step = 0.1

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        ## legend
        # one legend explaining colour (stimulus) and one explaining line style
        stim_handles = [
            Line2D(
                [0],
                [0],
                color=c,
                lw=3,
                marker="o",
                markersize=markersize,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            for c in colors
        ]

        leg1 = axes.legend(
            stim_handles,
            stim_labels,
            title="Stimulus",
            frameon=False,
            loc="lower left",
            alignment="left",
        )
        axes.add_artist(leg1)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        if save_flag == 1:
            fig.savefig(
                f"{self.pgm_plot_dir}/PlotLine_2afc_{self.binocular_interaction}_{self.seed}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_2afc_performance_seeds(self, save_flag: bool = True):
        """
        Plot 2-AFC performance averaged across all seed numbers for a given
        binocular interaction

        """

        p_2afc_seeds = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.config.dotMatch_list),
                len(self.config.dotDens_list),
            ),
            dtype=np.float32,
        )

        # load expected_disp_near and far all seeds
        for s, seed in enumerate(self.config.seed_to_analyse):

            # interaction = "sum_diff"
            if self.binocular_interaction == "default":
                epoch2load, iter2load = self.config.epoch_iter_to_load_default[s]
            elif self.binocular_interaction == "bem":
                epoch2load, iter2load = self.config.epoch_iter_to_load_bem[s]
            elif self.binocular_interaction == "cmm":
                epoch2load, iter2load = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch2load, iter2load = self.config.epoch_iter_to_load_sum_diff[s]
            # s = 0

            path = Path(self.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
            p_2afc = np.load(f"{path}/p_2afc.npy")  # [dotMatch, dotDens]

            # store to buffer
            p_2afc_seeds[s] = p_2afc

        # average across seeds, [dotMatch, dotDens]
        p_2afc_mean = p_2afc_seeds.mean(axis=0)
        p_2afc_sem = sem(p_2afc_seeds, axis=0)

        # start plotting
        stim_labels = ["aRDS", "hmRDS", "cRDS"]
        markersize = 8

        sns.set_theme(context="paper", style="ticks", font_scale=2, palette="deep")

        figsize = (9, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"2-AFC performance ({self.model_name}, interaction: {self.binocular_interaction})",
            ha="center",
            fontweight="semibold",
        )
        fig.text(-0.05, 0.5, "Expected disp [px]", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        ## plot E_near
        colors = ["#6a5acd", "#00CED1", "#333333"]
        x = np.array(self.dotDens_list)
        for i in range(len(self.config.dotMatch_list)):
            y = p_2afc_mean[i]
            y_err = p_2afc_sem[i]

            axes.plot(
                x,
                y,
                lw=2,
                c=colors[i],
                ls="-",
            )

            # plot the marker
            axes.plot(x, y, "o", markersize=markersize, c=colors[i])

            axes.fill_between(
                x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
            )

        # plot chance level
        axes.hlines(
            y=0.5, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2
        )

        x_low = 0.0
        x_up = 1.05
        x_step = 0.1
        y_low = 0.0
        y_up = 1.05
        y_step = 0.1

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        ## legend
        # one legend explaining colour (stimulus) and one explaining line style
        stim_handles = [
            Line2D(
                [0],
                [0],
                color=c,
                lw=3,
                marker="o",
                markersize=markersize,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            for c in colors
        ]
        leg1 = axes.legend(
            stim_handles,
            stim_labels,
            title="Stimulus",
            frameon=False,
            loc="lower left",
            alignment="left",
        )
        axes.add_artist(leg1)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        if save_flag == 1:
            if not os.path.exists(f"{self.save_dir}/plots"):
                os.mkdir(f"{self.save_dir}/plots")
            fig.savefig(
                f"{self.save_dir}/plots/PlotLine_2afc_avg_seeds_{self.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
