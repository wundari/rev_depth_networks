"""
Script for running a group analysis of rds_analysis results.

working directory: gc-net_v2
"""

# %% load necessary modules
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config.config import ConfigGCNet

# from engine.engine_base import Engine
from modules.gcnet import build_gcnet

# from RDS_analysis.rds_analysis import RDSAnalysis
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from SVM.svm_analysis_v4 import load_train_data, load_test_data, xDecode_bootstrap
from utilities.misc import NestedTensor

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import sem

# reproducibility
import random

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


class GA_RDS:

    def __init__(self, config: ConfigGCNet, params_rds: dict) -> None:

        self.config = config
        self.dataset = config.dataset
        self.binocular_interaction = config.binocular_interaction
        self.seed = config.seed
        self.epoch = config.epoch_to_load
        self.iter = config.iter_to_load
        self.device = config.device

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
        self.transform_data = NormalizeRDS()

        # set the experiment directory
        self.experiment_dir = (
            f"run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"{self.seed}"
        )

        # directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.rds_dir = (
            f"{self.experiment_dir}/"
            + f"rds_analysis_epoch_{self.epoch}_"
            + f"iter_{self.iter}/"
            + f"target_disp_{self.target_disp}px"
        )
        if not os.path.exists(self.rds_dir):
            os.makedirs(self.rds_dir)

        if self.pedestal_flag:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_with_pedestal"
        else:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_wo_pedestal"
        if not os.path.exists(self.xDecode_dir):
            os.mkdir(self.xDecode_dir)

        # print out network configuration
        self.__getconfig__()

    def __getconfig__(self) -> None:
        """
        print out the network configuration
        """

        print(
            "Network config\n"
            + f"Binocular interaction: {self.config.binocular_interaction}\n"
            + f"Seed: {self.config.seed}\n"
            + f"Epoch: {self.config.epoch_to_load}\n"
            + f"Iter: {self.config.iter_to_load}\n"
            + f"Experiment_dir: {self.experiment_dir}\n"
            + f"RDS_dir: {self.rds_dir}\n"
            + f"xDecode_dir: {self.xDecode_dir}\n"
        )

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

        # update folders for rds analysis
        self.rds_dir = (
            f"{self.experiment_dir}/"
            + f"rds_analysis_epoch_{self.epoch}"
            + f"_iter_{self.iter}/"
            + f"target_disp_{self.target_disp}px"
        )
        if not os.path.exists(self.rds_dir):
            os.makedirs(self.rds_dir)

        # udpate xDecode_dir
        if self.pedestal_flag:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_with_pedestal"
        else:
            self.xDecode_dir = f"{self.rds_dir}/xDecode_analysis_wo_pedestal"
        if not os.path.exists(self.xDecode_dir):
            os.mkdir(self.xDecode_dir)

        # create xDecode_dir/Plots
        if not os.path.exists(f"{self.xDecode_dir}/Plots"):
            os.makedirs(f"{self.xDecode_dir}/Plots")

        print(
            "Updating network config\n"
            + f"Binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"Seed: {seed_old} => {self.config.seed}\n"
            + f"Epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"Iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"Experiment_dir: {self.experiment_dir}\n"
            + f"RDS_dir: {self.rds_dir}\n"
            + f"xDecode_dir: {self.xDecode_dir}\n"
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

        # compile model
        if self.config.compile_mode is not None:
            self.model = torch.compile(
                self.model, mode=self.config.compile_mode
            )  # use compile_mode = "default" for layer analysis
        self.model.to(self.device)

        print(
            f"GC-Net was successfully loaded to {self.device}. \n"
            + f"GC-Net model: {resume_path}\n"
            + f"binocular interaction: {self.binocular_interaction}\n"
            + f"compile mode: {self.config.compile_mode}\n"
            + f"experiment dir: {self.experiment_dir}\n"
        )

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
            num_workers=4,
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
        split_train_ratio = 0.8
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
            else:  # sum_diff
                epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            self.update_network_config(interaction, seed, epoch, iter)

            # load new model and update directories
            self.load_model()

            # compute model responses to RDSs
            self.compute_disp_map_rds_group(
                self.dotDens_list, self.background_flag, self.pedestal_flag
            )

            # cross-decoding analysis with SVM
            self.xDecode(self.dotDens_list, n_bootstrap, self.background_flag)

    def plotLine_xDecode(self, save_flag: bool = False):
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

        figsize = (8, 8)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "GC-Net depth performance single seed\n"
            + f"({self.config.binocular_interaction})",
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

        colors = ["#6a5acd", "#00CED1", "#333333"]
        ## plot the one standard deviation for cRDS vs aRDS
        x = np.array(self.dotDens_list) * 100
        y = np.array(score_ards_mean)
        axes.plot(x, y, linewidth=2, color=colors[0], label="cRDS vs. aRDS")
        axes.plot(x, y, "o", markersize=12, color=colors[0])
        axes.fill_between(
            x,
            y - score_ards_std,
            y + score_ards_std,
            color=colors[0],
            alpha=0.2,
        )

        ## plot the error bar for cRDS vs hmRDS
        y = np.array(score_hmrds_mean)
        axes.plot(x, y, linewidth=2, color=colors[1], label="cRDS vs. hmRDS")
        axes.plot(x, y, "o", markersize=12, color=colors[1])
        axes.fill_between(
            x,
            y - score_hmrds_std,
            y + score_hmrds_std,
            color=colors[1],
            alpha=0.2,
        )

        ## plot the one standard deviation for cRDS
        y = np.array(score_crds_mean)
        axes.plot(x, y, linewidth=2, color=colors[2], label="cRDS")
        axes.plot(x, y, "o", markersize=12, color=colors[2])
        axes.fill_between(
            x,
            y - score_crds_std,
            y + score_crds_std,
            color=colors[0],
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

        plt.legend(loc="lower right", fontsize=20, frameon=False)
        # bbox_to_anchor=(0.525, 0.95))

        if save_flag == 1:

            fig.savefig(
                f"{self.xDecode_dir}/Plots/PlotScatter_xDecode.pdf",
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
            "GC-Net depth performance all seeds \n"
            + f"({self.config.binocular_interaction})",
            ha="center",
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # colors = ["#FC8D62", "#66C2A5", "#8DA0CB"]
        # colors = ["#D95F02", "#1B9E77", "#7570B3"]
        # colors = ["#D55E00", "#009E73", "#0072B2"]
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
                f"{self.plot_dir}/plotLine_gcnet_rds_xDecode_{interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotHeat_dispMap(self, save_flag: bool = False) -> None:
        """
        plot the heat map of the predicted disparity map for a single trial

        Args:
            save_flag (1/0 binary): save picture (1) or not (0)
        """

        def _make_grid(dispMap_near, dispMap_far):

            h_img, w_img = dispMap_near.shape
            img_grid = np.zeros((h_img, 2 * w_img), dtype=np.float32)

            img_grid[:, :w_img] = dispMap_near
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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

    def plotHeat_dispMap_avg(self, save_flag: bool = False):
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
            "Predicted disparity map (avg across trials), "
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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()
