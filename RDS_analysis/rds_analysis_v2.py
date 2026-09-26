# %% load necessary modules
import torch
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler

import numpy as np
import os
import glob
import gc
import io
import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import redirect_stdout
from pathlib import Path
from tqdm import tqdm
from scipy.stats import sem
from joblib import Parallel, delayed, parallel_config

from engine.engine_base import EngineBase
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from SVM.svm_analysis import *
from BNN.modules.bnn import build_bnn
from GC_Net.modules.gcnet import build_gcnet
from config.config_bnn import ConfigBNN
from config.config_gcnet import ConfigGCNet
from utilities.misc import NestedTensor


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
        x = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(3, -1, -1)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected an HW or HWC RGB image, got {tuple(x.shape)}")
        if not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
            raise ValueError("RDS pixels must be finite in [-1, 1]")

        return ((x + 1) / 2 - self._mean) / self._std


def _generate_rds_condition(
    dot_match,
    dot_density,
    disparities,
    stimulus_seeds,
    background,
    pedestal,
):
    """Generate one condition using one explicit seed per RDS trial."""
    random_state = np.random.get_state()
    try:
        left_trials = []
        right_trials = []
        label_trials = []

        # The outer process handles parallelism. Keep generate_rds's nested
        # joblib call serial and reseed immediately before every RDS trial.
        with parallel_config(backend="sequential"):
            with redirect_stdout(io.StringIO()):
                for seed in stimulus_seeds:
                    np.random.seed(int(seed))
                    left, right, labels = RDS_Handler.generate_rds(
                        dot_match,
                        dot_density,
                        disparities,
                        1,
                        background,
                        pedestal,
                    )
                    # The images are monochrome. One int8 channel reduces the
                    # resident bank size by 12x; NormalizeRDS expands RGB lazily.
                    left_trials.append(left[..., 0].astype(np.int8, copy=False))
                    right_trials.append(right[..., 0].astype(np.int8, copy=False))
                    label_trials.append(labels)

        n_disparities = len(disparities)

        def disparity_major(trials):
            # [trial, disparity, ...] -> [disparity, trial, ...] -> flat
            array = np.stack(trials, axis=0)
            axes = (1, 0, *range(2, array.ndim))
            array = array.transpose(axes)
            return array.reshape(n_disparities * len(stimulus_seeds), *array.shape[2:])

        return (
            disparity_major(left_trials),
            disparity_major(right_trials),
            disparity_major(label_trials),
        )
    finally:
        np.random.set_state(random_state)


class RDSBankDataset(ConcatDataset):
    """Conditions ordered by match, density, then the handler's disparity order."""

    def __init__(
        self, datasets, dotMatch_list, dotDens_list, samples_per_condition, bank_seed
    ):
        super().__init__(datasets)
        self.dotMatch_list = tuple(dotMatch_list)
        self.dotDens_list = tuple(dotDens_list)
        self.bank_seed = bank_seed
        self.condition_shape = (
            len(self.dotMatch_list),
            len(self.dotDens_list),
            samples_per_condition,
        )


class RDSAnalysis(EngineBase):

    def __init__(self, config: ConfigBNN | ConfigGCNet) -> None:

        super().__init__(config)

        self.config = config
        self.model_name = config.model_name
        self.binocular_interaction = config.binocular_interaction
        self.dataset = config.dataset

        self.seed = config.seed
        self.epoch = config.epoch_to_load
        self.iter = config.iter_to_load

        # rds parameters
        self.h_bg = config.img_height  # rds height
        self.w_bg = config.img_width  # rds width
        self.rds_type = config.rds_type  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
        self.batch_size_rds = config.batch_size_rds
        self.n_rds_each_disp = (
            config.n_rds_each_disp
        )  # n_rds for each disparity magnitude in disp_ct_pix
        self.dotDens_list = config.dotDens_list  # dot densities for in-silico analysis
        self.dotMatch_list = config.dotMatch_list  # dot match
        self.background_flag = config.background_flag  # 1: with cRDS background
        self.pedestal_flag = (
            config.pedestal_flag
        )  # 1: use pedestal to ensure rds disparity > 0
        self.n_bootstrap = (
            config.n_bootstrap
        )  # number of bootstrap samples for cosine-similarity analysis
        self.n_bootstrap_xDecode = (
            config.n_bootstrap_xDecode
        )  # number of bootstrap samples for cross-decoding analysis

        self.target_disp = (
            config.target_disp
        )  # RDS target disparity (pix) to be analyzed
        self.disp_ct_pix_list = [
            self.target_disp,
            -self.target_disp,
        ]  # disparity magnitude (near, far)

        # make dirs for saving the rds analysis
        self.make_rds_dirs()

        # print out RDS params for analysis:
        self.__getconfig_rds___()

        # transform rds to tensor and in range [0, 1]
        self.transform_data = NormalizeRDS()

    def __getconfig_rds___(self):
        print(
            "==============================================================\n"
            + "RDS analysis parameters: \n"
            + "==============================================================\n"
            + f"DNN model used for RDS analysis: {self.config.model_pretrained}\n"
            + f"Batch size RDS: {self.batch_size_rds} \n"
            + f"Disparity targets: [-{self.target_disp}, {self.target_disp}] pixels \n"
            + f"Number of RDS for each disparity target: {self.n_rds_each_disp} \n"
            + f"Number of bootstrap for cross-decoding analysis: {self.n_bootstrap} \n"
            + f"RDS directory: {self.rds_dir}\n"
            + f"Cross-decoding directory: {self.xDecode_dir}\n"
            + "==============================================================\n"
        )

    def make_rds_dirs(self) -> None:
        """
        dirs for rds analysis
        """

        self.rds_dir = os.path.join(
            self.experiment_dir,
            f"rds_analysis_{self.config.model_pretrained[:-8]}",
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

    def _build_model(self, config: ConfigBNN | ConfigGCNet) -> None:
        if config.model_name == "BNN":
            return build_bnn(config)
        elif config.model_name == "GC_Net":
            return build_gcnet(config)

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

    def create_rds_bank(
        self,
        background_flag: bool,
        pedestal_flag: bool,
        bank_seed: int = 3407,
        n_jobs: int = 16,
        loader_workers: int = 4,
    ):
        """
        Generate a bank of RDS images that store all rds types for all dotMatch and dotDens.

        RDS bank structure:
        [dotMatch dotDens disp_magnitude n_rds_each_disp]

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
                    .]

        For each rds_cond (each dotMatch and dotDens), there are len(disp_ct_pix_list) * n_rds_each_disp

        Args:
            dotMatch_list: Dot-match levels, for example [0.0, 0.5, 1.0].

            dotDens_list: Dot-density levels, for example [0.1, ..., 0.9].

            background_flag: Whether to surround the RDS with a cRDS background.

            pedestal_flag: Whether to shift each RDS so its minimum disparity is 0.

            bank_seed: Root seed used to derive an explicit seed for every trial.

            n_jobs: Number of processes used for condition-level generation.

            loader_workers: DataLoader worker count. Zero avoids copying the
                multi-gigabyte bank into spawned loader processes.

        Returns:
            A sequential DataLoader whose dataset is ordered by dot match, dot
            density, disparity, then trial.
        """

        dotMatch_list = tuple(float(value) for value in self.dotMatch_list)
        dotDens_list = tuple(float(value) for value in self.dotDens_list)
        if not dotMatch_list or not dotDens_list:
            raise ValueError("dotMatch_list and dotDens_list must be non-empty")
        if len(set(dotMatch_list)) != len(dotMatch_list):
            raise ValueError("dotMatch_list contains duplicate conditions")
        if len(set(dotDens_list)) != len(dotDens_list):
            raise ValueError("dotDens_list contains duplicate conditions")
        if self.n_rds_each_disp <= 0:
            raise ValueError("n_rds_each_disp must be positive")
        if bank_seed < 0:
            raise ValueError("bank_seed must be non-negative")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if loader_workers < 0:
            raise ValueError("loader_workers must be non-negative")

        n_samples = len(self.disp_ct_pix_list) * self.n_rds_each_disp
        conditions = [
            (dotMatch, dotDens)
            for dotMatch in dotMatch_list
            for dotDens in dotDens_list
        ]

        # Derive seeds from condition values instead of condition indices. Thus,
        # reordering or subsetting the requested conditions does not change an
        # existing stimulus. Each submitted job receives all seeds explicitly.
        stimulus_seeds = []
        for dot_match, dot_density in conditions:
            condition_seed = np.random.SeedSequence(
                [
                    bank_seed,
                    int(round((dot_match + 1.0) * 10_000)),
                    int(round(dot_density * 10_000)),
                ]
            )
            stimulus_seeds.append(
                [
                    int(child.generate_state(1, dtype=np.uint32)[0])
                    for child in condition_seed.spawn(self.n_rds_each_disp)
                ]
            )

        datasets = []
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(
                n_jobs=n_jobs, return_as="generator", pre_dispatch="n_jobs"
            )(
                delayed(_generate_rds_condition)(
                    dotMatch,
                    dotDens,
                    self.disp_ct_pix_list,
                    seeds,
                    background_flag,
                    pedestal_flag,
                )
                for (dotMatch, dotDens), seeds in zip(
                    conditions, stimulus_seeds, strict=True
                )
            )

            for rds_left, rds_right, rds_label in tqdm(
                results, total=len(conditions), desc="Generating RDS bank"
            ):
                expected_shape = (n_samples, self.h_bg, self.w_bg)
                if (
                    rds_left.shape != expected_shape
                    or rds_right.shape != expected_shape
                ):
                    raise ValueError(
                        f"Expected RDS shape {expected_shape}, got {rds_left.shape} and {rds_right.shape}"
                    )
                if rds_label.shape != (n_samples,):
                    raise ValueError(
                        "RDS labels do not match the condition sample count"
                    )

                datasets.append(
                    DatasetRDS(
                        rds_left, rds_right, rds_label, transform=self.transform_data
                    )
                )

        dataset = RDSBankDataset(
            datasets, dotMatch_list, dotDens_list, n_samples, bank_seed
        )
        loader_options = {
            "dataset": dataset,
            "batch_size": self.batch_size_rds,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False,
            "num_workers": loader_workers,
        }
        if loader_workers > 0:
            loader_options["prefetch_factor"] = 2
            loader_options["persistent_workers"] = True

        return DataLoader(**loader_options)

    @torch.inference_mode()
    def compute_disp_map_rds(
        self,
        rds_bank: DataLoader,
    ):
        """
        generate disparity map specifically for rds for a given dot Match and dotDens.

        Args:
            rds_bank: DataLoader

        Returns:
            pred_disp [len(disp_ct_pix_list) * n_rds_each_disp, h_bg, w_bg)] float32:
                    predicted disparity map

            pred_disp_labels [len(disp_ct_pix_list) * n_rds_each_disp] int8:
                the label (near (+) or far(-)) of the predicted disparity map.
        """

        dataset = rds_bank.dataset
        if not isinstance(dataset, RDSBankDataset):
            raise TypeError("Use the DataLoader returned by create_rds_bank")
        if not isinstance(rds_bank.sampler, SequentialSampler) or rds_bank.drop_last:
            raise ValueError(
                "The RDS bank must use sequential sampling and drop_last=False"
            )

        n_samples = len(dataset)
        pred_disp = torch.empty(
            (
                n_samples,
                self.h_bg,
                self.w_bg,
            ),
            dtype=torch.float32,
        )
        pred_disp_labels = np.empty(
            n_samples,
            dtype=np.int8,
        )

        # predict disparity map
        self.model.eval()
        tepoch = tqdm(rds_bank, desc="Predicting RDS")
        offset = 0
        for inputs_left, inputs_right, disps in tepoch:
            # for i in range(len(rds_loader)):
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

            # print(f"disp map RDS dotMatch: {dotMatch:.2f}, dotDens: {dotDens:.2f}")

            # Generate disparity direction. Swap left/right per sample rather
            # than per batch, so correctness does not depend on batch boundaries.
            # generate disparity direction
            ref = disps / 10.0

            # build nested tensor
            # input_data = NestedTensor(
            #     left=inputs_left.to(self.config.device, non_blocking=True),
            #     right=inputs_right.to(
            #         self.config.device, non_blocking=True
            #     ),
            #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            # )
            if ref.mean() > 0:
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
                disp_pred = self.model(input_data)

            batch_length = len(disps)
            id_end = offset + batch_length
            pred_disp_labels[offset:id_end] = disps.cpu().numpy()
            pred_disp[offset:id_end].copy_(disp_pred.float().cpu())
            offset = id_end

        if offset != n_samples:
            raise RuntimeError(f"Predicted {offset} samples, expected {n_samples}")

        return (
            pred_disp.reshape(*dataset.condition_shape, self.h_bg, self.w_bg),
            pred_disp_labels.reshape(*dataset.condition_shape),
        )

    def compute_disp_map_rds_group(self, rds_bank: DataLoader):
        """Predict and save disparity maps for every condition in an RDS bank.

        Args:
            rds_bank: The sequential DataLoader returned by create_rds_bank.
        """
        dataset = rds_bank.dataset
        if not isinstance(dataset, RDSBankDataset):
            raise TypeError("Use the DataLoader returned by create_rds_bank")

        print(
            "==============================================================\n"
            + f"Computing model responses to RDSs, batch size: {self.batch_size_rds}\n"
            + "==============================================================\n"
        )
        pred_disp, pred_disp_labels = self.compute_disp_map_rds(rds_bank)
        for dm in range(len(self.dotMatch_list)):
            np.save(
                f"{self.xDecode_dir}/pred_disp_{self.rds_type[dm]}.npy",
                pred_disp[dm].numpy(),
            )
            np.save(
                f"{self.xDecode_dir}/pred_disp_labels_{self.rds_type[dm]}.npy",
                pred_disp_labels[dm],
            )

        # return pred_disp, pred_disp_labels

    def xDecode(self, dotDens_list: list, n_bootstrap: int, background_flag: bool):
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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

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
            f"{self.model_name} depth performance single seed\n"
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
            if not os.path.exists(f"{self.xDecode_dir}/Plots"):
                os.makedirs(f"{self.xDecode_dir}/Plots")

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

    def plotLine_xDecode_avg_seed(self, dataset_name: str, save_flag: bool = True):
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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

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

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()


# %%
