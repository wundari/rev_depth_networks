# %% load necessary modules
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

from engine.engine_base import Engine
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from SVM.svm_analysis_v3 import *

from utilities.misc import NestedTensor

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch._dynamo.config.suppress_errors = True

# reproducibility
import random
from config.config import BNNconfig

cfg = BNNconfig()
seed_number = cfg.seed
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


# %%
class RDSAnalysis(Engine):

    def __init__(self, config, params_rds: dict) -> None:

        super().__init__(config)

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

        # transform rds to tensor and in range [0, 1]
        # self.transform_data = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
        # )
        # mean = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
        # std = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # mean = np.array([0.5, 0.5, 0.5])
        # std = np.array([0.5, 0.5, 0.5])
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t + 1.0) / 2.0),
                transforms.Normalize(mean, std),
            ]
        )

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
            num_workers=self.batch_size,
            worker_init_fn=seed_worker,
            generator=g,
        )

        pred_disp = torch.zeros(
            (len(self.disp_ct_pix_list) * self.n_rds_each_disp, self.h_bg, self.w_bg),
            dtype=torch.float32,
        )
        pred_disp_labels = np.zeros(
            (len(self.disp_ct_pix_list) * self.n_rds_each_disp), dtype=np.int8
        )

        # predict disparity map
        tepoch = tqdm(rds_loader)
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

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

            # model output
            with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                disp_pred = self.model(input_data)

            id_start = i * self.batch_size
            id_end = id_start + self.batch_size
            pred_disp_labels[id_start:id_end] = disps
            pred_disp[id_start:id_end] = disp_pred

            tepoch.set_description(
                f"RDS dotMatch: {dotMatch:.2f}, "
                + f"dotDens: {dotDens:.2f}, "
                + f"iter: {i+1}/{len(rds_loader)}"
            )

        return pred_disp, pred_disp_labels

    @torch.no_grad()
    def compute_disp_map_rds_group(self, dotDens_list, background_flag, pedestal_flag):
        """
        generate disparity map for rds for each dot density in dotDens_list

        Args:
            dotDens_list ([list]): a list containing dot densities
            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)
        """

        for dm, dotMatch in enumerate(self.dotMatch_list):
            pred_disp = torch.zeros(
                (
                    len(dotDens_list),
                    len(self.disp_ct_pix_list) * self.n_rds_each_disp,
                    self.h_bg,
                    self.w_bg,
                ),
                dtype=torch.float32,
            )
            pred_disp_labels = np.zeros(
                (len(dotDens_list), len(self.disp_ct_pix_list) * self.n_rds_each_disp),
                dtype=np.int8,
            )
            for dd, dotDens in enumerate(dotDens_list):

                pred_disp[dd], pred_disp_labels[dd] = self.compute_disp_map_rds(
                    dotMatch, dotDens, background_flag, pedestal_flag
                )

            np.save(
                f"{self.xDecode_dir}/pred_disp_{self.rds_type[dm]}.npy",
                pred_disp.numpy(),
            )
            np.save(
                f"{self.xDecode_dir}/pred_disp_labels_{self.rds_type[dm]}.npy",
                pred_disp_labels,
            )

        # return pred_disp, pred_disp_labels

    def xDecode(self, dotDens_list, n_bootstrap, background_flag):
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
        # X_train, Y_train, x_mean, x_std = load_train_data(rdsa.svm_dir, background_flag)

        # build test dataset
        X_ards, Y_ards, X_hmrds, Y_hmrds = load_test_data(
            self.xDecode_dir, x_mean, x_std, background_flag
        )
        # X_ards, Y_ards, X_hmrds, Y_hmrds = load_test_data(
        #     rdsa.svm_dir, x_mean, x_std, background_flag
        # )

        # classifying rds with SVM
        n_samples = int(0.8 * len(X_train))
        # n_bootstrap = 50  # 1000
        (
            score_ards_bootstrap,
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
            n_samples,
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

        dotDens_idx = np.where(self.dotDens_list == dotDens)[0][0]

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

    def plotHeat_dispMap(self, save_flag):
        """
        plot the heat map of the predicted disparity map for a single trial

        Args:
            save_flag (1/0 binary): save picture (1) or not (0)
        """

        def _make_grid(dispMap_near, dispMap_far):

            h_img, w_img = dispMap_near.shape
            img_grid = np.zeros((h_img, 2 * w_img), dtype=np.float32)

            img_grid[:, 0:w_img] = dispMap_near
            img_grid[:, w_img:] = dispMap_far

            return img_grid

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
        for dd in range(len(self.dotDens_list)):
            dotDens = self.dotDens_list[dd]

            # ards
            # ards_near
            disp_id = np.where(disp_map_ards_labels[dd] > 0)[0][0]
            disp_near = disp_map_ards[dd, disp_id]
            # ards_far
            disp_id = np.where(disp_map_ards_labels[dd] < 0)[0][0]
            disp_far = disp_map_ards[dd, disp_id]
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            # plot
            im = axes[dd, 0].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 0].axis("off")
            axes[dd, 0].set_title(f"dotDens: {dotDens:.1f}, aRDS: near//far")
            # make a vertical boundary line
            axes[dd, 0].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

            # hmrds
            # hmrds near
            disp_id = np.where(disp_map_hmrds_labels[dd] > 0)[0][0]
            disp_near = disp_map_hmrds[dd, disp_id]
            # hmrds far
            disp_id = np.where(disp_map_hmrds_labels[dd] < 0)[0][0]
            disp_far = disp_map_hmrds[dd, disp_id]
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            # plot
            im = axes[dd, 1].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 1].axis("off")
            axes[dd, 1].set_title(f"dotDens: {dotDens:.1f}, hmRDS: near//far")
            # make a vertical boundary line
            axes[dd, 1].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

            # crds
            # crds near
            disp_id = np.where(disp_map_crds_labels[dd] > 0)[0][0]
            disp_near = disp_map_crds[dd, disp_id]
            # crds far
            disp_id = np.where(disp_map_crds_labels[dd] < 0)[0][0]
            disp_far = disp_map_crds[dd, disp_id]
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            im = axes[dd, 2].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 2].axis("off")
            axes[dd, 2].set_title(f"dotDens: {dotDens:.1f}, cRDS: near//far")
            # make a vertical boundary line
            axes[dd, 2].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

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

        def _make_grid(dispMap_near, dispMap_far):

            h_img, w_img = dispMap_near.shape
            img_grid = np.zeros((h_img, 2 * w_img), dtype=np.float32)

            img_grid[:, 0:w_img] = dispMap_near
            img_grid[:, w_img:] = dispMap_far

            return img_grid

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
        for dd in range(len(self.dotDens_list)):
            dotDens = self.dotDens_list[dd]

            # ards
            # ards_near
            disp_id = np.where(disp_map_ards_labels[dd] > 0)[0]
            disp_near = disp_map_ards[dd, disp_id].mean(axis=0)
            # ards_far
            disp_id = np.where(disp_map_ards_labels[dd] < 0)[0]
            disp_far = disp_map_ards[dd, disp_id].mean(axis=0)
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            # plot
            im = axes[dd, 0].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 0].axis("off")
            axes[dd, 0].set_title(f"dotDens: {dotDens:.1f}, aRDS: near//far")
            # make a vertical boundary line
            axes[dd, 0].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

            # hmrds
            # hmrds near
            disp_id = np.where(disp_map_hmrds_labels[dd] > 0)[0]
            disp_near = disp_map_hmrds[dd, disp_id].mean(axis=0)
            # hmrds far
            disp_id = np.where(disp_map_hmrds_labels[dd] < 0)[0]
            disp_far = disp_map_hmrds[dd, disp_id].mean(axis=0)
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            # plot
            im = axes[dd, 1].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 1].axis("off")
            axes[dd, 1].set_title(f"dotDens: {dotDens:.1f}, hmRDS: near//far")
            # make a vertical boundary line
            axes[dd, 1].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

            # crds
            # crds near
            disp_id = np.where(disp_map_crds_labels[dd] > 0)[0]
            disp_near = disp_map_crds[dd, disp_id].mean(axis=0)
            # crds far
            disp_id = np.where(disp_map_crds_labels[dd] < 0)[0]
            disp_far = disp_map_crds[dd, disp_id].mean(axis=0)
            # create a grid
            disp_grid = _make_grid(disp_near, disp_far)
            im = axes[dd, 2].imshow(
                disp_grid,
                vmin=v_min,
                vmax=v_max,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[dd, 2].axis("off")
            axes[dd, 2].set_title(f"dotDens: {dotDens:.1f}, cRDS: near//far")
            # make a vertical boundary line
            axes[dd, 2].plot(
                [disp_near.shape[1], disp_near.shape[1]],
                [0, disp_near.shape[0]],
                color="k",
                linewidth=3,
            )
            # color bar
            plt.colorbar(im, fraction=0.02, pad=0.05)

        if save_flag:
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.mkdir(f"{self.xDecode_dir}/Plots")

            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotHeat_dispMap_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )
