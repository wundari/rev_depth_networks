# %% load necessary modules
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import sem

import os
import glob
from pathlib import Path

from engine.engine_base import Engine
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from SVM.svm_analysis_v4 import *

from utilities.misc import NestedTensor
from utilities.output_hook import ModuleOutputsHook


# %%
class NormalizeRDS:
    """Normalize signed [-1,1] RGB arrays; no uint8 ToTensor ambiguity/lambda."""

    # mean = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
    # std = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])

    _mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    _std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __call__(self, image):
        x = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32).permute(
            2, 0, 1
        )
        if not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
            raise ValueError("RDS pixels must be finite in [-1,1]")

        return ((x + 1) / 2 - self._mean) / self._std


class RDSAnalysis(Engine):

    def __init__(self, config, params_rds: dict) -> None:

        super().__init__(config)

        if not config.load_state:
            raise ValueError(
                "Set config.load_state=True to analyze a trained checkpoint; random-model controls require allow_random_model=True"
            )
        if config.compile_mode is not None:
            raise ValueError("Use compile_mode=None for RDS analysis and layer hooks")

        # rds parameters
        self.params_rds = params_rds
        self.h_bg = 256  # rds height
        self.w_bg = 512  # rds width

        self.rds_type = params_rds["rds_type"]
        self.batch_size = params_rds["batch_size_rds"]
        self.n_rds_each_disp = params_rds["n_rds_each_disp"]
        self.dotDens_list = params_rds["dotDens_list"]
        self.dotMatch_list = params_rds["dotMatch_list"]
        self.background_flag = params_rds["background_flag"]
        self.target_disp = params_rds["target_disp"]
        self.pedestal_flag = params_rds["pedestal_flag"]
        self.disp_ct_pix_list = [
            self.target_disp,
            -self.target_disp,
        ]  # disparity magnitude (near, far).
        self.n_bootstrap = params_rds["n_bootstrap"]

        # check if n_rds is divisible by batch_size_rds
        n_total = len(self.disp_ct_pix_list) * self.n_rds_each_disp
        assert n_total % self.batch_size == 0, (
            f"batch_size={self.batch_size} must evenly divide n_total={n_total} "
            "when drop_last=True, or samples will be silently dropped."
        )

        # transform rds to tensor and in range [0, 1]
        self.transform_data = NormalizeRDS()

        # dirs for rds analysis
        self.rds_dir = os.path.join(
            self.experiment_dir,
            f"rds_analysis_{config.resume[:-8]}",
            f"target_disp_{self.target_disp}px",
        )
        if not os.path.exists(self.rds_dir):
            os.makedirs(self.rds_dir)

        if self.pedestal_flag:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_with_pedestal"
        else:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_wo_pedestal"
        if not os.path.exists(self.xDecode_dir):
            os.mkdir(self.xDecode_dir)

        self.target_list = [
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

    @torch.no_grad()
    def compute_layer_activations(
        self,
        input_data: NestedTensor,
        target: nn.Module,
    ) -> dict:
        """
        compute layer activation in respond to left and right inputs.

        Args:
            image_left_gpu (_type_): _description_
            image_right_gpu (_type_): _description_
            target (nn.Module): the target layer to be computed.
                it can be either in this form:
                - target = model.layer35a
                - target = model.layer35a[0] # the convolutional output
                - target = [model.layer35a]
                - target = [model.layer19, model.layer20, ...] # many layers


        Returns:
            module_outputs (list): a list containing the target layer
                    activations.

        """

        # get layer activation to get layer dimension
        # define hook
        if isinstance(target, list):
            hook = ModuleOutputsHook(target)
        else:
            hook = ModuleOutputsHook([target])
            # hook = ModuleOutputsHook([model.layer36a])

        modes = {m: m.training for m in self.model.modules()}
        self.model.eval()
        try:
            self.model(input_data)
            return {
                m: value.detach().clone() for m, value in hook.consume_outputs().items()
            }
        finally:
            hook.remove_hooks()
            for module, mode in modes.items():
                module.training = mode

    @torch.no_grad()
    def compute_disp_map_rds(self, dotMatch, dotDens, background_flag, pedestal_flag):
        """
        generate disparity map specifically for rds for a given dot Match and dotDens.

        Args:
            dotMatch (float): dot match level; between 0 (ards) to 1(crds)

            dotDens (float): dot density level; between 0.1 to 0.9

            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)

            pedestal_flag (binary): a flag indicating with or without pedestal.
                pedestal here means that the whole RDSs are shifted such that
                the smallest disparity = 0.
                0: without pedestal
                1: with pedestal

        Returns:
            pred_disp [len(disp_ct_pix_list) * n_rds_each_disp, h_bg, w_bg)] float32:
                    predicted disparity map

            pred_disp_labels [len(disp_ct_pix_list) * n_rds_each_disp] int8:
                the label (near (+) or far(-)) of the predicted disparity map.
        """

        print(f"disp map RDS dotMatch: {dotMatch:.2f}, dotDens: {dotDens:.2f}")

        # create dataloader for RDS
        # [len(disp_ct_pix) * batch_size, h, w, n_channels]
        rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
            dotMatch,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            background_flag,
            pedestal_flag,
        )

        rds_data = DatasetRDS(
            rds_left, rds_right, rds_label, transform=self.transform_data
        )
        rds_loader = DataLoader(
            rds_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
        )

        pred_disp = torch.empty(
            (len(self.disp_ct_pix_list) * self.n_rds_each_disp, self.h_bg, self.w_bg),
            dtype=torch.float32,
        )
        pred_disp_labels = np.empty(
            (len(self.disp_ct_pix_list) * self.n_rds_each_disp), dtype=np.int8
        )

        # predict disparity map
        self.model.eval()
        tepoch = tqdm(rds_loader)
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):

            # inputs_left, inputs_right, disps = next(iter(rds_loader))

            # generate disparity direction
            ref = disps / 10.0

            # build nested tensor
            # input_data = NestedTensor(
            #     left=inputs_left.pin_memory().to(self.config.device, non_blocking=True),
            #     right=inputs_right.pin_memory().to(
            #         self.config.device, non_blocking=True
            #     ),
            #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            # )
            if ref.mean() >= 0:
                input_data = NestedTensor(
                    left=inputs_left.to(self.config.device, non_blocking=True),
                    right=inputs_right.to(self.config.device, non_blocking=True),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            else:
                input_data = NestedTensor(
                    left=inputs_right.to(self.config.device, non_blocking=True),
                    right=inputs_left.to(self.config.device, non_blocking=True),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )

            # model output
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                disp_pred = self.model(input_data)  # [batch, h, w]
                # module_outputs = self.compute_layer_activations(
                #     input_data, self.target_list
                # )
                # layer = self.target_list[-1]
                # # [batch, feat_channel, disp_channel, h, w] => [batch, disp_channel, h, w]
                # disp_pred = module_outputs[layer].mean(dim=1)
                # disp_pred = F.softmax(-disp_pred, dim=1)  # [batch, disp_channel, h, w]
                # # disp_pred = torch.sum(
                # #     disp_pred * self.model.disp_indices, dim=1
                # # )  # [batch, h, w]
                # disp_pred = torch.sum(
                #     disp_pred
                #     * self.model.disp_indices
                #     * input_data.ref.view(-1, 1, 1, 1),
                #     dim=1,
                # )

            id_start = i * self.batch_size
            id_end = id_start + self.batch_size
            pred_disp_labels[id_start:id_end] = disps
            pred_disp[id_start:id_end] = disp_pred.detach().float().cpu()

            tepoch.set_description(
                f"RDS dotMatch: {dotMatch:.2f}, "
                + f"dotDens: {dotDens:.2f}, "
                + f"iter: {i+1}/{len(rds_loader)}"
            )

        return pred_disp, pred_disp_labels

    def compute_disp_map_rds_group(
        self, dotDens_list: list, background_flag: bool, pedestal_flag: bool
    ) -> None:
        """
        generate disparity map for rds for each dot density in dotDens_list

        Args:
            dotDens_list ([list]): a list containing dot densities
            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)
        """

        for dm, dotMatch in enumerate(self.dotMatch_list):
            pred_disp = torch.empty(
                (
                    len(dotDens_list),
                    len(self.disp_ct_pix_list) * self.n_rds_each_disp,
                    self.h_bg,
                    self.w_bg,
                ),
                dtype=torch.float32,
            )
            pred_disp_labels = np.empty(
                (len(dotDens_list), len(self.disp_ct_pix_list) * self.n_rds_each_disp),
                dtype=np.int8,
            )
            for dd, dotDens in enumerate(dotDens_list):

                pred_disp[dd], pred_disp_labels[dd] = self.compute_disp_map_rds(
                    dotMatch, dotDens, background_flag, pedestal_flag
                )

            np.save(
                f"{self.xDecode_dir}/pred_disp_{self.rds_type[dm]}.npy",
                pred_disp.cpu().detach().numpy(),
            )
            np.save(
                f"{self.xDecode_dir}/pred_disp_labels_{self.rds_type[dm]}.npy",
                pred_disp_labels,
            )

        # return pred_disp, pred_disp_labels

    def xDecode(
        self, dotDens_list: list, n_bootstrap: int, background_flag: bool
    ) -> None:
        """
        Perform cross-decoding: cRDS vs aRDS and cRDS vs hmRDS.

        Args:
            dotDens_list ([list]): a list containing dot densities

            n_bootstrap (int): the number of bootstrap iteration

            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)

        """
        # build training dataset (using crds)
        X_train, Y_train, x_mean, x_std = load_train_data(
            self.xDecode_dir, background_flag
        )
        # X_train, Y_train, x_mean, x_std = load_train_data(rdsa.xDecode_dir, rdsa.background_flag)

        # build test dataset
        X_ards, Y_ards, X_hmrds, Y_hmrds = load_test_data(
            self.xDecode_dir, x_mean, x_std, background_flag
        )
        # X_ards, Y_ards, X_hmrds, Y_hmrds = load_test_data(
        #     rdsa.xDecode_dir, x_mean, x_std, rdsa.background_flag
        # )

        # classifying rds with SVM
        split_train_ratio = 0.8
        # n_bootstrap = 50  # 1000
        (
            score_ards_bootstrap,  # [n_bootstrap, len(dotDens_list)]
            predict_ards_bootstrap,
            score_hmrds_bootstrap,
            predict_hmrds_bootstrap,
            score_crds_bootstrap,
            predict_crds_bootstrap,
        ) = xDecode_bootstrap(
            X_train,
            Y_train,
            X_ards,
            Y_ards,
            X_hmrds,
            Y_hmrds,
            split_train_ratio,
            n_bootstrap,
            dotDens_list,
        )

        # save file
        np.save(f"{self.xDecode_dir}/score_ards_bootstrap.npy", score_ards_bootstrap)
        np.save(f"{self.xDecode_dir}/score_hmrds_bootstrap.npy", score_hmrds_bootstrap)
        np.save(f"{self.xDecode_dir}/score_crds_bootstrap.npy", score_crds_bootstrap)
        np.save(
            f"{self.xDecode_dir}/predict_ards_bootstrap.npy", predict_ards_bootstrap
        )
        np.save(
            f"{self.xDecode_dir}/predict_hmrds_bootstrap.npy", predict_hmrds_bootstrap
        )
        np.save(
            f"{self.xDecode_dir}/predict_crds_bootstrap.npy", predict_crds_bootstrap
        )

        print("score ards: ", score_ards_bootstrap.mean(axis=0))
        print("score hmrds: ", score_hmrds_bootstrap.mean(axis=0))
        print("score crds: ", score_crds_bootstrap.mean(axis=0))

    def plotLine_xDecode_at_dotDens(self, dotDens, save_flag):
        """
        plot cross-decoding performance at target dot density:
                cRDS vs aRDS, cRDS vs hmRDS, and cRDS vs cRDS

        Args:
            dotDens (float): dot density
        """

        dotDens_idx = np.where(np.round(self.dotDens_list, 2) == dotDens)[0][0]

        # load cross-decoding data
        # [n_bootstrap, len(dotDens_list)]
        score_ards_bootstrap = np.load(f"{self.xDecode_dir}/score_ards_bootstrap.npy")
        score_hmrds_bootstrap = np.load(f"{self.xDecode_dir}/score_hmrds_bootstrap.npy")
        score_crds_bootstrap = np.load(f"{self.xDecode_dir}/score_crds_bootstrap.npy")

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (4.5, 4.5)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"SVM prediction on RDS with background at dotDens {dotDens:.2f}",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Prediction acc.", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot correlation", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        score_ards_mean = score_ards_bootstrap.mean(axis=0)[dotDens_idx]
        score_ards_std = score_ards_bootstrap.std(axis=0)[dotDens_idx]
        score_hmrds_mean = score_hmrds_bootstrap.mean(axis=0)[dotDens_idx]
        score_hmrds_std = score_hmrds_bootstrap.std(axis=0)[dotDens_idx]
        score_crds_mean = score_crds_bootstrap.mean(axis=0)[dotDens_idx]
        score_crds_std = score_crds_bootstrap.std(axis=0)[dotDens_idx]

        ## plot
        x = np.array(self.dotMatch_list)
        y = np.array([score_ards_mean, score_hmrds_mean, score_crds_mean])
        y_err = np.array([score_ards_std, score_hmrds_std, score_crds_std])
        axes.errorbar(x, y, yerr=y_err, lw=2, c="k", ls="-", capsize=7)

        # plot the marker
        markersize = 8
        axes.plot(x, y, "o", markersize=markersize, c="k")

        # plot chance level
        axes.plot([-0.1, 1], [0.5, 0.5], ls="--", lw=2, c="r")

        x_low = 0.0
        x_up = 1.05
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        axes.set_xticks(x)
        axes.set_xticklabels([-1.0, 0.0, 1.0])
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 0.05, x_up)
        axes.set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        if save_flag == 1:
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.mkdir(f"{self.xDecode_dir}/Plots")
            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotLine_disp_map_svm_at_dotDens_{dotDens:.1f}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_xDecode(self, save_flag):
        """
        Plot cross-decoding performance as a function of dot density.
        """

        # load cross-decoding data
        # [n_bootstrap, len(dotDens_list)]
        score_ards_bootstrap = np.load(f"{self.xDecode_dir}/score_ards_bootstrap.npy")
        score_hmrds_bootstrap = np.load(f"{self.xDecode_dir}/score_hmrds_bootstrap.npy")
        score_crds_bootstrap = np.load(f"{self.xDecode_dir}/score_crds_bootstrap.npy")

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
            "Cross-decoding performance\n"
            + f"target disp: {self.target_disp}, with cRDS background",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Prediction acc.", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        score_ards_mean = score_ards_bootstrap.mean(axis=0)
        score_ards_std = score_ards_bootstrap.std(axis=0)
        score_hmrds_mean = score_hmrds_bootstrap.mean(axis=0)
        score_hmrds_std = score_hmrds_bootstrap.std(axis=0)
        score_crds_mean = score_crds_bootstrap.mean(axis=0)
        score_crds_std = score_crds_bootstrap.std(axis=0)

        ## plot the one standard deviation for cRDS vs aRDS
        x = np.array(self.dotDens_list)
        y = np.array(score_ards_mean)
        y_err = np.array(score_ards_std)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="red", ls="-", capsize=7)

        # plot the marker
        markersize = 12
        axes.plot(x, y, "o", markersize=markersize, c="red")

        ## plot the error bar for cRDS vs hmRDS
        y = np.array(score_hmrds_mean)
        y_err = np.array(score_hmrds_std)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="green", ls="-", capsize=7)

        # plot the marker
        axes.plot(x, y, "o", markersize=markersize, c="green")

        ## plot the one standard deviation for cRDS
        y = np.array(score_crds_mean)
        y_err = np.array(score_crds_std)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="blue", ls="-", capsize=7)

        # plot the marker
        axes.plot(x, y, "o", markersize=markersize, c="blue")

        # plot chance level
        axes.plot([0, 1], [0.5, 0.5], "k--", linewidth=3)

        x_low = 0.0
        x_up = 1.05
        x_step = 0.2
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low - 0.05, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        plt.legend(["cRDS vs. aRDS", "cRDS vs. hmRDS", "cRDS"], fontsize=20)
        # bbox_to_anchor=(0.525, 0.95))

        if save_flag == 1:
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.mkdir(f"{self.xDecode_dir}/Plots")

            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotScatter_xDecode.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_xDecode_avg_seed(self, dataset_name, save_flag):
        """
        Average cross-decoding performance across seed numbers and plot it.

        """
        # set folder location
        rds_seed_dir = (
            Path.cwd()
            / "run"
            / dataset_name
            / "bino_interaction_default_avg_batchsize16_v2"
        )

        # load cross-decoding data for every seed
        score_ards_all = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.dotDens_list),
            ),
            dtype=np.float32,
        )
        score_hmrds_all = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.dotDens_list),
            ),
            dtype=np.float32,
        )
        score_crds_all = np.empty(
            (
                len(self.config.seed_to_analyse),
                len(self.dotDens_list),
            ),
            dtype=np.float32,
        )

        for i in range(len(self.config.seed_to_analyse)):

            seed = self.config.seed_to_analyse[i]
            seed_dir = glob.glob(
                f"{str(rds_seed_dir)}/{seed}/rds_analysis_epoch_*", recursive=True
            )[0]
            xDecode_dir = f"{seed_dir}/target_disp_{self.target_disp}px/xDecode_analysis_wo_pedestal"

            # [n_bootstrap, len(dotDens_list)]
            print(xDecode_dir)
            score_ards_bootstrap = np.load(f"{xDecode_dir}/score_ards_bootstrap.npy")
            score_hmrds_bootstrap = np.load(f"{xDecode_dir}/score_hmrds_bootstrap.npy")
            score_crds_bootstrap = np.load(f"{xDecode_dir}/score_crds_bootstrap.npy")

            score_ards_all[i] = score_ards_bootstrap.mean(axis=0)
            score_hmrds_all[i] = score_hmrds_bootstrap.mean(axis=0)
            score_crds_all[i] = score_crds_bootstrap.mean(axis=0)

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
            "Cross-decoding performance\n"
            + f"target disp: {self.target_disp}, with cRDS background",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Prediction acc.", va="center", rotation=90)
        fig.text(0.5, -0.04, "Dot density", ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        score_ards_mean = score_ards_all.mean(axis=0)
        score_ards_sem = sem(score_ards_all)
        score_hmrds_mean = score_hmrds_all.mean(axis=0)
        score_hmrds_sem = sem(score_hmrds_all)
        score_crds_mean = score_crds_all.mean(axis=0)
        score_crds_sem = sem(score_crds_all)

        ## plot the one standard deviation for cRDS vs aRDS
        x = np.array(self.dotDens_list)
        y = np.array(score_ards_mean)
        y_err = np.array(score_ards_sem)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="red", ls="-", capsize=7)

        # plot the marker
        markersize = 12
        axes.plot(x, y, "o", markersize=markersize, c="red")

        ## plot the error bar for cRDS vs hmRDS
        y = np.array(score_hmrds_mean)
        y_err = np.array(score_hmrds_sem)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="green", ls="-", capsize=7)

        # plot the marker
        axes.plot(x, y, "o", markersize=markersize, c="green")

        ## plot the one standard deviation for cRDS
        y = np.array(score_crds_mean)
        y_err = np.array(score_crds_sem)
        axes.errorbar(x, y, yerr=y_err, lw=3, c="blue", ls="-", capsize=7)

        # plot the marker
        axes.plot(x, y, "o", markersize=markersize, c="blue")

        # plot chance level
        axes.plot([0, 1], [0.5, 0.5], "k--", linewidth=3)

        x_low = 0.0
        x_up = 1.05
        x_step = 0.2
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")
        # axes.tick_params(direction='in', length=4, width=1)

        plt.legend(["cRDS vs. aRDS", "cRDS vs. hmRDS", "cRDS"], fontsize=20)
        # bbox_to_anchor=(0.525, 0.95))

        if save_flag == 1:

            fig.savefig(
                f"{rds_seed_dir}/PlotScatter_xDecode_avg_seed_{dataset_name}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def _plot_disp_row(self, axes_row, dd, dotDens, panels, v_min, v_max, cmap, avg):
        """
        panels: list of (disp_map, labels, title_suffix) for ards/hmrds/crds
        """

        def _make_grid(near, far):
            h, w = near.shape
            grid = np.zeros((h, 2 * w), dtype=np.float32)
            grid[:, :w], grid[:, w:] = near, far
            return grid

        for col, (disp_map, labels, name) in enumerate(panels):
            near_idx = np.where(labels[dd] > 0)[0]
            far_idx = np.where(labels[dd] < 0)[0]
            if avg:
                near, far = disp_map[dd, near_idx].mean(axis=0), disp_map[
                    dd, far_idx
                ].mean(axis=0)
            else:
                near, far = disp_map[dd, near_idx[0]], disp_map[dd, far_idx[0]]

            grid = _make_grid(near, far)
            im = axes_row[col].imshow(
                grid, vmin=v_min, vmax=v_max, cmap=cmap, interpolation="nearest"
            )
            axes_row[col].axis("off")
            axes_row[col].set_title(f"dotDens: {dotDens:.1f}, {name}: near//far")
            axes_row[col].plot(
                [near.shape[1]] * 2, [0, near.shape[0]], color="k", linewidth=3
            )
            plt.colorbar(im, fraction=0.02, pad=0.05)

    def plotHeat_dispMap(self, save_flag):
        """
        plot the heat map of the predicted disparity map for a single trial

        Args:
            save_flag (1/0 binary): save picture (1) or not (0)
        """

        # load data
        disp_map_ards = np.load(f"{self.xDecode_dir}/pred_disp_ards.npy")
        disp_map_ards_labels = np.load(f"{self.xDecode_dir}/pred_disp_labels_ards.npy")
        disp_map_hmrds = np.load(f"{self.xDecode_dir}/pred_disp_hmrds.npy")
        disp_map_hmrds_labels = np.load(
            f"{self.xDecode_dir}/pred_disp_labels_hmrds.npy"
        )
        disp_map_crds = np.load(f"{self.xDecode_dir}/pred_disp_crds.npy")
        disp_map_crds_labels = np.load(f"{self.xDecode_dir}/pred_disp_labels_crds.npy")

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (20, 25)
        n_row = 9
        n_col = 3
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        fig.text(
            0.5,
            1.0,
            f"Predicted disparity map, target disparity: {self.target_disp} pixel",
            ha="center",
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.0)

        if self.pedestal_flag:
            v_min = -4 * self.target_disp
            v_max = 4 * self.target_disp
        else:
            c = 2
            v_min = -c * self.target_disp
            v_max = c * self.target_disp

        cmap = "coolwarm"
        for dd, dotDens in enumerate(self.dotDens_list):
            panels = [
                (disp_map_ards, disp_map_ards_labels, "aRDS"),
                (disp_map_hmrds, disp_map_hmrds_labels, "hmRDS"),
                (disp_map_crds, disp_map_crds_labels, "cRDS"),
            ]
            self._plot_disp_row(
                axes[dd], dd, dotDens, panels, v_min, v_max, cmap, avg=False
            )

        if save_flag:
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.mkdir(f"{self.xDecode_dir}/Plots")

            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotHeat_dispMap.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotHeat_dispMap_avg(self, save_flag):
        """
        plot the heat map of the predicted disparity map, averaged across
        trials

        Args:
            save_flag (1/0 binary): save picture (1) or not (0)

        """

        # load data
        disp_map_ards = np.load(f"{self.xDecode_dir}/pred_disp_ards.npy")
        disp_map_ards_labels = np.load(f"{self.xDecode_dir}/pred_disp_labels_ards.npy")
        disp_map_hmrds = np.load(f"{self.xDecode_dir}/pred_disp_hmrds.npy")
        disp_map_hmrds_labels = np.load(
            f"{self.xDecode_dir}/pred_disp_labels_hmrds.npy"
        )
        disp_map_crds = np.load(f"{self.xDecode_dir}/pred_disp_crds.npy")
        disp_map_crds_labels = np.load(f"{self.xDecode_dir}/pred_disp_labels_crds.npy")

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (20, 25)
        n_row = 9
        n_col = 3
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        fig.text(
            0.5,
            1.0,
            f"Predicted disparity map (avg across trials), "
            + f"target disparity: {self.target_disp} pixel",
            ha="center",
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.0)

        if self.pedestal_flag:
            v_min = 0
            v_max = 2 * self.target_disp
        else:
            c = 2
            v_min = -c * self.target_disp
            v_max = c * self.target_disp

        cmap = "coolwarm"
        for dd, dotDens in enumerate(self.dotDens_list):
            panels = [
                (disp_map_ards, disp_map_ards_labels, "aRDS"),
                (disp_map_hmrds, disp_map_hmrds_labels, "hmRDS"),
                (disp_map_crds, disp_map_crds_labels, "cRDS"),
            ]
            self._plot_disp_row(
                axes[dd], dd, dotDens, panels, v_min, v_max, cmap, avg=True
            )

        if save_flag:
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.mkdir(f"{self.xDecode_dir}/Plots")

            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotHeat_dispMap_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )
