# %% load necessary modules
from turtle import color
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch._dynamo

from Network_dissection.network_dissection import NetworkDissection
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utils.output_hook import ModuleOutputsHook
from Common.network_correlation import extract_activation
from Common.disp_tuning_correlation import (
    _compute_disp_tuning_corr_layer,
    compute_auc_disp_tuning_corr_layer,
    compute_MI_disp_tuning_layer,
)

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import sem
from tqdm import tqdm
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")


# reproducibility
import random

seed_number = 3407  # 12321
torch.manual_seed(seed_number)
np.random.seed(seed_number)


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_number)

# %%
class AUCRatio(NetworkDissection):
    def __init__(self, params_network, params_train) -> None:
        super().__init__(params_network, params_train)

        # network params
        self.target_dotDens = params_train["target_dotDens"]

        # rds params
        self.n_rds_each_disp = 200  # n_rds for each disparity magnitude in disp_ct_pix
        self.rds_type = ["ards", "hmrds", "crds"]
        # ards: 0, crds: 1, hmrds: 0.5, urds: -1
        self.dotMatch_list = np.arange(0, 1.1, 0.1)
        self.dotDens_list = np.arange(0.1, 1.0, 0.1)
        self._disp_mag = 30  # disparity tuning range
        self._disp_step = 2
        self.disp_ct_pix_list = np.arange(
            -self._disp_mag, self._disp_mag + self._disp_step, self._disp_step
        )  # disparity magnitude (neg: near, pos: far)

        self.target_list = [
            self.model.layer19[0],  # only get the conv layer output
            self.model.layer20[0],
            self.model.layer21[0],
            self.model.layer22[0],
            self.model.layer23[0],
            self.model.layer24[0],
            self.model.layer25[0],
            self.model.layer26[0],
            self.model.layer27[0],
            self.model.layer28[0],
            self.model.layer29[0],
            self.model.layer30[0],
            self.model.layer31[0],
            self.model.layer32[0],
            self.model.layer33a[0],
            self.model.layer34a[0],
            self.model.layer35a[0],
            self.model.layer36a[0],
            self.model.layer37,
        ]

        # self.disp_tuning_layers = []

    @property
    def target_dotDens(self):
        return self._target_dotDens

    @target_dotDens.setter
    def target_dotDens(self, target_dotDens_new):
        self._target_dotDens = target_dotDens_new
        # saving dir
        self.dotDens_dir = f"{self.neuron_dir}/dotDens_{self.target_dotDens:.2f}"
        if not os.path.exists(self.dotDens_dir):
            os.makedirs(self.dotDens_dir)

    # @property
    # def disp_tuning_layers(self):
    #     return self._disp_tuning_layers

    # @disp_tuning_layers.setter
    def load_disp_tuning_layers(self):

        print(
            f"loading disparity tuning at dot dens: {self.target_dotDens:.2f} for all layers"
        )
        self.disp_tuning_layers = []
        for dm in range(len(self.dotMatch_list)):
            dotMatch = self.dotMatch_list[dm]
            disp_tuning_layers_dotMatch = np.load(
                f"{self.dotDens_dir}/disp_tuning_dotMatch_{dotMatch:.1f}.npy",
                allow_pickle=True,
            )

            self.disp_tuning_layers.append(disp_tuning_layers_dotMatch.item())

    def _generate_rds(self, dotMatch_ct, dotDens, background_flag):
        # generate rds
        rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
            dotMatch_ct,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            background_flag,
        )
        self._n_rds = rds_left.shape[0]

        # transform rds to tensor and in range [0, 1]
        transform_data = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
        )
        rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
        batch_size = 1
        rds_loader = DataLoader(
            rds_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        return rds_loader

    @torch.no_grad()
    def compute_disp_tuning(
        self, dotMatch_ct: float, dotDens: float, background_flag: bool, rds_type: str
    ) -> None:
        # generate rds_loader
        print(f"generating {rds_type}")
        rds_loader = self._generate_rds(dotMatch_ct, dotDens, background_flag)

        # allocate
        disp_tuning_rds_all = {}
        for t in range(len(self.target_list)):
            target_layer_name = self.target_name[t]
            # get activation shape for this layer
            n_feature_channels, n_disp_channels, _, _ = self.get_activations_shape(
                self.target_list[t]
            )

            # pre-allocate array
            disp_tuning_crds = np.empty(
                (n_feature_channels * n_disp_channels, self._n_rds), dtype=np.float32
            )
            disp_tuning_rds_all[target_layer_name] = disp_tuning_crds

        disp_tuning_axis = np.empty(self._n_rds, dtype=np.int8)

        # define hook
        hook = ModuleOutputsHook(self.target_list)
        tepoch = tqdm(rds_loader, unit="batch")
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
            input_left = inputs_left.pin_memory().to(device, non_blocking=True)
            input_right = inputs_right.pin_memory().to(device, non_blocking=True)
            disp_tuning_axis[i] = disps.item()

            # compute model's output
            logits = self.model(input_left, input_right)

            # consume_outputs return the captured values and resets the hook's state
            # compute module output
            module_outputs = hook.consume_outputs()

            # extract activations for each channel
            for t in range(len(self.target_name)):
                target_layer_name = self.target_name[t]
                # activations = (
                #     module_outputs[self.target_list[t]].cpu().detach().numpy()
                # )  # get conv2d output
                activations = module_outputs[self.target_list[t]]  # get conv2d output

                # get the neuron activations at the center channel
                activation_extracted = extract_activation(activations)
                disp_tuning_rds_all[target_layer_name][:, i] = activation_extracted

        tepoch.close()
        hook.remove_hooks()

        # save data
        np.save(
            f"{self.dotDens_dir}/disp_tuning_dotMatch_{dotMatch_ct:.1f}.npy",
            disp_tuning_rds_all,
        )
        np.save(
            f"{self.dotDens_dir}/disp_tuning_{dotMatch_ct:.1f}_axis.npy",
            disp_tuning_axis,
        )

    def _compute_disp_tuning_corr(self):
        # load data
        disp_tuning_ards_all = np.load(
            f"{self.neuron_dir}/disp_tuning_dotMatch_0.0.npy", allow_pickle=True
        )
        disp_tuning_hmrds_all = np.load(
            f"{self.neuron_dir}/disp_tuning_dotMatch_0.5.npy", allow_pickle=True
        )
        disp_tuning_crds_all = np.load(
            f"{self.neuron_dir}/disp_tuning_dotMatch_1.0.npy", allow_pickle=True
        )

        corr_crds_ards_all = {}
        corr_crds_hmrds_all = {}
        for t, target_layer_name in enumerate(self.target_name):
            # load data
            disp_tuning_ards = disp_tuning_ards_all.item()[target_layer_name]
            disp_tuning_hmrds = disp_tuning_hmrds_all.item()[target_layer_name]
            disp_tuning_crds = disp_tuning_crds_all.item()[target_layer_name]

            # average neural responses across rds_trials at each disparity value
            n_neurons = disp_tuning_ards.shape[0]  # n_neurons in a layer
            disp_tuning_ards_avg = disp_tuning_ards.reshape(
                n_neurons, len(self.disp_ct_pix_list), self.n_rds_each_disp
            ).mean(axis=2)
            disp_tuning_hmrds_avg = disp_tuning_hmrds.reshape(
                n_neurons, len(self.disp_ct_pix_list), self.n_rds_each_disp
            ).mean(axis=2)
            disp_tuning_crds_avg = disp_tuning_crds.reshape(
                n_neurons, len(self.disp_ct_pix_list), self.n_rds_each_disp
            ).mean(axis=2)

            # compute the Pearson correlation (crds, ards) and (crds, hmrds)
            corr_crds_ards = np.array(
                [
                    np.corrcoef(disp_tuning_crds_avg[n], disp_tuning_ards_avg[n])[0, 1]
                    for n in range(n_neurons)
                ]
            )

            corr_crds_hmrds = np.array(
                [
                    np.corrcoef(disp_tuning_crds_avg[n], disp_tuning_hmrds_avg[n])[0, 1]
                    for n in range(n_neurons)
                ]
            )  # type: ignore

            corr_crds_ards_all[target_layer_name] = corr_crds_ards
            corr_crds_hmrds_all[target_layer_name] = corr_crds_hmrds

        return corr_crds_ards_all, corr_crds_hmrds_all

    def get_disp_tuning_layer(self, target_layer_name):
        """
            get disparity tuning data for the target_layer_name

        Args:
            disp_tuning_layers <list>: a list containing disparity tuning in all layers
                and all dotMatch
            target_layer_name (string): the target layer name

        Returns:
            disp_tuning_layer <np.array [n_neurons, n_dotMatch, len(disp_ct_pix_list)]:
                disparity tuning of neurons in target_layer_name for every dotMatch
        """
        # load crds disparity tuning to get n_neurons in target_layer_name
        disp_tuning_crds = self.disp_tuning_layers[-1][target_layer_name]
        # reshape
        n_neurons = disp_tuning_crds.shape[0]
        disp_tuning_crds = disp_tuning_crds.reshape(
            n_neurons, len(self.disp_ct_pix_list), self.n_rds_each_disp
        )

        disp_tuning_layer = np.zeros(
            (
                len(self.dotMatch_list),
                n_neurons,
                len(self.disp_ct_pix_list),
                self.n_rds_each_disp,
            ),
            dtype=np.float32,
        )

        disp_tuning_layer[-1] = disp_tuning_crds
        # load non-crds disparity tuning
        for dm in range(len(self.dotMatch_list) - 1):
            disp_tuning = self.disp_tuning_layers[dm][target_layer_name]
            # reshape
            disp_tuning = disp_tuning.reshape(
                n_neurons, len(self.disp_ct_pix_list), self.n_rds_each_disp
            )

            disp_tuning_layer[dm] = disp_tuning

        # swap axis　to [n_neurons, dotMatch, disp, rds_trial]
        disp_tuning_layer = disp_tuning_layer.transpose(1, 0, 2, 3)

        return disp_tuning_layer

    def compute_disp_tuning_corr_layer(self, target_layer_name):
        """
        For each neuron in a network layer, compute the Pearson correlation
        between non-crds disparity tuning and cRDS disparity tuning.

        Args:
            target_layer_name (string): the name of the target layer

        Returns:
            disp_tuning_corr_layer (np.array [n_neurons, n_dotMatch - 1]):
                the Pearson correlation between non-crds disparity tuning
                and cRDS disparity tuning
        """

        # load disparity tuning data
        disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)
        # average across rds_trials
        disp_tuning_layer_avg = np.mean(disp_tuning_layer, axis=-1)

        # compute the Pearson correlation between non-crds disp_tuning and
        # crds disp_tuning
        disp_tuning_corr_layer = _compute_disp_tuning_corr_layer(disp_tuning_layer_avg)

        return disp_tuning_corr_layer

    def compute_corr_auc_ratio_layer(self, disp_tuning_corr_layer, omit_outlier=True):
        """
        compute the ratio between area under curve (auc) in a target layer

        Args:
        disp_tuning_corr_layer (np.array [n_neurons, n_dotMatch - 1]):
            the Pearson correlation between non-crds disparity tuning
            and cRDS disparity tuning

        Returns:
            area_ratio: _description_
        """
        # compute the trapezoidal width dx for calculating the auc
        dx = self.dotMatch_list[1] - self.dotMatch_list[0]

        # compute the auc
        auc_neg_layer, auc_pos_layer = compute_auc_disp_tuning_corr_layer(
            disp_tuning_corr_layer, dx
        )

        tol = 1e-6
        # auc_ratio = -auc_neg_layer / (auc_pos_layer + tol)
        auc_ratio = np.abs(auc_neg_layer) / (auc_pos_layer + np.abs(auc_neg_layer))

        if omit_outlier:
            # omit data whose auc_ratio > mean + 6 * std
            auc_std = np.std(auc_ratio)
            threshold = np.mean(auc_ratio) + 6 * auc_std  # type: ignore
            auc_ratio = auc_ratio[auc_ratio < threshold]

        return auc_ratio

    def compute_corr_auc_ratio_all_layer(self):
        auc_ratio_layers = {}
        for t, target_layer_name in enumerate(self.target_name):

            print(f"Compute the AUC ratio in {target_layer_name}")

            # compute disparity tuning
            disp_tuning_corr_layer = self.compute_disp_tuning_corr_layer(
                target_layer_name
            )

            # compute auc ratio
            auc_ratio = self.compute_corr_auc_ratio_layer(disp_tuning_corr_layer)

            auc_ratio_layers[target_layer_name] = auc_ratio

        return auc_ratio_layers

    def plotScatter_neuron_pref_input_interocular_corr_vs_auc_ratio(self, save_flag):
        auc_ratio_layers = {}
        neuron_pref_input_corr_layers = {}
        for t, target_layer_name in enumerate(self.target_name):

            ## process preferred inputs
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]
            # reshape into [n_neurons]
            neuron_pref_input_corr = neuron_pref_input_corr.ravel()

            ## process disparity tuning
            # load disparity tuning data [n_neurons, len]
            disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)

            # average across rds_trials [n_neurons, len(disp_ct_list), ]
            disp_tuning_layer_avg = np.mean(disp_tuning_layer, axis=-1)

            # compute the Pearson correlation between non-crds disp_tuning and
            # crds disp_tuning
            disp_tuning_corr_layer = _compute_disp_tuning_corr_layer(
                disp_tuning_layer_avg
            )

            # compute auc ratio
            auc_ratio = self.compute_corr_auc_ratio_layer(
                disp_tuning_corr_layer, omit_outlier=False
            )
            auc_ratio_layers[target_layer_name] = auc_ratio

            ## omit outlier based on auc_ratio
            # process auc_ratio
            # threshold = np.mean(auc_ratio) + 6 * np.std(auc_ratio)  # type: ignore
            threshold = 0.0
            idx_omit = np.where(auc_ratio < threshold)
            auc_ratio_layers[target_layer_name] = np.delete(auc_ratio, idx_omit)

            # process neuron_pref_input_corr
            neuron_pref_input_corr_layers[target_layer_name] = np.delete(
                neuron_pref_input_corr, idx_omit
            )

        ## start plotting
        x_med = np.zeros(len(self.target_name), dtype=np.float32)
        x_std = np.zeros(len(self.target_name), dtype=np.float32)
        y_med = np.zeros(len(self.target_name), dtype=np.float32)
        y_std = np.zeros(len(self.target_name), dtype=np.float32)
        for t, target_layer_name in enumerate(self.target_name):
            x = neuron_pref_input_corr_layers[target_layer_name]
            y = auc_ratio_layers[target_layer_name]

            x_med[t] = np.mean(x)
            x_std[t] = sem(x)  # np.std(x)
            y_med[t] = np.mean(y)
            y_std[t] = sem(x)  # np.std(y)

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (7, 7)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            f"Neuron pref input interocular corr vs AUC ratio (dotDens: {self.target_dotDens:.2f})",
            ha="center",
        )
        # fig.text(0.35, 0.9, "Cross-correlation dominance", ha="center", color="gray")
        # fig.text(0.35, 0.12, "Cross-matching dominance", ha="center", color="gray")

        fig.text(0.5, -0.03, "Pref inputs interocular corr", ha="center")
        fig.text(-0.03, 0.5, "AUC ratio", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # plot marker for legend
        markersize = 400  # 17
        marker_list = ["o", "v", "^", "<", ">", "s", "D", "P", "X", "*", "p", "1"]
        markercolor_list = ["magenta", "lime", "cyan"]
        marker_counter = 0
        id_end_early = 6
        id_end_mid = 13
        # early layers: 19 - 25
        id_start = 0
        id_end = id_end_early
        marker_counter = 0
        for t in range(id_start, id_end):
            # plot the marker
            axes.scatter(
                x_med[t],
                y_med[t],
                marker=marker_list[marker_counter],
                s=markersize,
                c=markercolor_list[0],
                edgecolors="black",
            )
            marker_counter += 1

        # middle layers: 29 - 32
        id_start = id_end
        id_end = id_end_mid
        marker_counter = 0
        for t in range(id_start, id_end):
            # plot the marker
            axes.scatter(
                x_med[t],
                y_med[t],
                marker=marker_list[marker_counter],
                s=markersize,
                c=markercolor_list[1],
                edgecolors="black",
            )
            marker_counter += 1

        # final layers: 33a - 37
        id_start = id_end
        id_end = len(self.target_name)
        marker_counter = 0
        for t in range(id_start, id_end):
            # plot the marker
            axes.scatter(
                x_med[t],
                y_med[t],
                marker=marker_list[marker_counter],
                s=markersize,
                c=markercolor_list[2],
                edgecolors="black",
            )
            marker_counter += 1

        axes.legend(
            self.target_name,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        # plot error bar
        marker_counter = 0
        for t in range(len(self.target_name)):
            # error bar
            axes.errorbar(
                x_med[t],
                y_med[t],
                xerr=x_std[t],
                yerr=y_std[t],
                lw=3,
                ls="-",
                capsize=4,
                color="gray",
            )

            # re-plot marker for clarity
            if t < id_end_early:  # early layers
                # plot the marker
                axes.scatter(
                    x_med[t],
                    y_med[t],
                    marker=marker_list[t],
                    s=markersize,
                    c=markercolor_list[0],
                    edgecolors="black",
                )

            elif t >= id_end_early and t < id_end_mid:  # middle layers
                # plot the marker
                axes.scatter(
                    x_med[t],
                    y_med[t],
                    marker=marker_list[t - id_end_early],
                    s=markersize,
                    c=markercolor_list[1],
                    edgecolors="black",
                )

            else:  # final layers
                # plot the marker
                axes.scatter(
                    x_med[t],
                    y_med[t],
                    marker=marker_list[t - id_end_mid],
                    s=markersize,
                    c=markercolor_list[2],
                    edgecolors="black",
                )

        # upper and lower plot
        # x_low = -0.3
        # x_up = 0.31
        # x_step = 0.1
        # y_low = 0.4
        # y_up = 0.91
        # y_step = 0.1
        x_low = -0.3
        x_up = 0.31
        x_step = 0.1
        y_low = 0.1
        y_up = 0.81
        y_step = 0.1
        # x_low = -0.5
        # x_up = 0.31
        # x_step = 0.1
        # y_low = 0.0
        # y_up = 0.51
        # y_step = 0.1

        # Plot line at 0 correlation
        axes.axvline(0.0, color="r", linestyle="--", linewidth=3)
        # Plot line at 0.5 auc_ratio
        axes.axhline(0.5, color="r", linestyle="--", linewidth=3)

        axes.set_xlim(x_low, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.dotDens_dir}/PlotScatter_neuron_pref_input_interocular_corr_vs_auc_ratio.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_neuron_pref_input_interocular_corr_vs_auc_ratio_earlylayers(
        self, save_flag
    ):
        # analyze the first 3 early layers
        auc_ratio_layers = {}
        neuron_pref_input_corr_layers = {}
        for t, target_layer_name in enumerate(self.target_name):

            ## process preferred inputs
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]
            # reshape into [n_neurons]
            neuron_pref_input_corr = neuron_pref_input_corr.ravel()

            ## process disparity tuning
            # load disparity tuning data [n_neurons, len]
            disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)

            # average across rds_trials [n_neurons, len(disp_ct_list), ]
            disp_tuning_layer_avg = np.mean(disp_tuning_layer, axis=-1)

            # compute the Pearson correlation between non-crds disp_tuning and
            # crds disp_tuning
            disp_tuning_corr_layer = _compute_disp_tuning_corr_layer(
                disp_tuning_layer_avg
            )

            # compute auc ratio
            auc_ratio = self.compute_corr_auc_ratio_layer(
                disp_tuning_corr_layer, omit_outlier=False
            )
            auc_ratio_layers[target_layer_name] = auc_ratio

            ## omit outlier based on auc_ratio
            # process auc_ratio
            threshold = np.mean(auc_ratio) + 6 * np.std(auc_ratio)  # type: ignore
            idx_omit = np.where(auc_ratio > threshold)
            auc_ratio_layers[target_layer_name] = np.delete(auc_ratio, idx_omit)

            # process neuron_pref_input_corr
            neuron_pref_input_corr_layers[target_layer_name] = np.delete(
                neuron_pref_input_corr, idx_omit
            )

        # compute p-val for 2 distributions: auc_ratio layer 19 vs layer 21
        x19 = auc_ratio_layers["layer19"]
        x21 = auc_ratio_layers["layer21"]
        t_val, p_val = stats.ttest_ind(x19, x21)

        ## start plotting
        x_med = np.zeros(len(self.target_name[:3]), dtype=np.float32)
        x_std = np.zeros(len(self.target_name[:3]), dtype=np.float32)
        y_mean = np.zeros(len(self.target_name[:3]), dtype=np.float32)
        y_std = np.zeros(len(self.target_name[:3]), dtype=np.float32)
        for t, target_layer_name in enumerate(self.target_name[:3]):
            x = neuron_pref_input_corr_layers[target_layer_name]
            y = auc_ratio_layers[target_layer_name]

            x_med[t] = np.median(x)
            x_std[t] = sem(x)
            y_mean[t] = np.median(y)
            y_std[t] = sem(y)

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (12, 9)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "Neuron pref input interocular corr vs AUC ratio",
            ha="center",
        )
        fig.text(0.5, -0.03, "Pref inputs interocular corr", ha="center")
        fig.text(-0.03, 0.5, "AUC ratio", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # plot marker for legend
        markersize = 13
        marker_list = ["o", "v", "^"]
        markercolor_list = ["magenta"]
        # first 3 early layers: 19 - 21
        for t in range(len(self.target_name[:3])):
            # plot the marker
            axes.plot(
                x_med[t],
                y_mean[t],
                marker_list[t],
                markersize=markersize,
                c=markercolor_list[0],
            )

        axes.legend(
            self.target_name,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        # print t-test
        fig.text(
            0.3,
            0.7,
            "auc_ratio layer 19 vs. 21\n"
            + f"layer19 (median, sem): {y_mean[0]:.3f}, {y_std[0]:.3f}\n"
            + f"layer21 (median, sem): {y_mean[2]:.3f}, {y_std[2]:.3f}\n"
            + f"p_val = {p_val:.3f}",
            color="red",
        )

        # plot error bar
        for t in range(len(self.target_name[:3])):
            # error bar
            axes.errorbar(
                x_med[t],
                y_mean[t],
                xerr=x_std[t],
                yerr=y_std[t],
                lw=3,
                ls="-",
                capsize=4,
                color="k",
            )

            # re-plot marker for clarity
            axes.plot(
                x_med[t],
                y_mean[t],
                marker_list[t],
                markersize=markersize,
                c=markercolor_list[0],
            )

        # upper and lower plot
        x_low = -0.5
        x_up = 0.1
        x_step = 0.1
        y_low = 0.4
        y_up = 0.75
        y_step = 0.1

        # Plot line at 0 correlation
        axes.axvline(0.5, color="k", linestyle="--", linewidth=3)

        axes.set_xlim(x_low, x_up)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.dotDens_dir}/PlotLine_neuron_pref_input_interocular_corr_vs_auc_ratio_earlylayers.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def PlotBox_neuron_pref_input_interocular_corr(self, save_flag):
        neuron_pref_input_interocular_allunits_layers = (
            {}
        )  # left-right interocular corr

        for t, target_layer_name in enumerate(self.target_name):

            ## process preferred inputs
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]
            # reshape into [n_neurons]
            neuron_pref_input_corr = neuron_pref_input_corr.ravel()

            ## process disparity tuning
            # load disparity tuning data [n_neurons, len]
            disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)

            # average across rds_trials [n_neurons, len(disp_ct_list), ]
            disp_tuning_layer_avg = np.mean(disp_tuning_layer, axis=-1)

            # compute the Pearson correlation between non-crds disp_tuning and
            # crds disp_tuning
            disp_tuning_corr_layer = _compute_disp_tuning_corr_layer(
                disp_tuning_layer_avg
            )

            # compute auc ratio
            auc_ratio = self.compute_corr_auc_ratio_layer(
                disp_tuning_corr_layer, omit_outlier=False
            )

            ## omit outlier based on auc_ratio
            # threshold auc_ratio
            threshold = np.mean(auc_ratio) + 6 * np.std(auc_ratio)  # type: ignore
            idx_omit = np.where(auc_ratio > threshold)

            # threshold neuron_pref_input_corr
            neuron_pref_input_corr_layer = np.delete(neuron_pref_input_corr, idx_omit)

            neuron_pref_input_interocular_allunits_layers[target_layer_name] = (
                neuron_pref_input_corr_layer
            )

        corr_neuron_mean = np.zeros(len(self.target_name), dtype=np.float32)
        corr_error = np.zeros(len(self.target_name), dtype=np.float32)
        x = []
        y = []
        # collect data
        for t, target_layer_name in enumerate(self.target_name):
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = neuron_pref_input_interocular_allunits_layers[
                target_layer_name
            ]  # [n_feature_channels, n_disp_channels]

            y.append(neuron_pref_input_corr.ravel())
            x.append(t * np.ones(len(neuron_pref_input_corr.ravel())))

            corr_neuron_mean[t] = np.mean(neuron_pref_input_corr)
            corr_error[t] = sem(neuron_pref_input_corr.ravel())

        # one-way ANOVA
        data = [
            neuron_pref_input_interocular_allunits_layers[layer]
            for layer in self.target_name
        ]
        dof_between = len(self.target_name) - 1
        # flatten data
        a = [x for row in data for x in row]
        dof_within = len(a) - len(self.target_name)
        F_val, p_val = stats.f_oneway(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
        )

        # post-hoc test
        res = stats.tukey_hsd(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
        )

        # create pandas df for recording the mean and sem interocular corr
        # [layer_name, interocular_corr_mean, interocular_corr_sem]
        temp = {
            "layer_name": self.target_name,
            "interocular_corr_mean": corr_neuron_mean,
            "interocular_corr_sem": corr_error,
        }
        df_mean = pd.DataFrame(temp)

        # create pandas df for the statistical test
        # [stat, p_val, lower_ci, upper_ci]
        n_row = len(self.target_name) ** 2
        tukey_stat = np.empty((n_row, 4), dtype=np.float32)
        col_comp = []
        for i, layer1 in enumerate(self.target_name):
            id_start = i * len(self.target_name)
            id_end = id_start + len(self.target_name)
            tukey_stat[id_start:id_end, 0] = res.statistic[i]  # mean difference
            tukey_stat[id_start:id_end, 1] = res.pvalue[i]  # p_val
            tukey_stat[id_start:id_end, 2] = res.confidence_interval().low[
                i
            ]  # 95% CI lower bound
            tukey_stat[id_start:id_end, 3] = res.confidence_interval().high[
                i
            ]  # 95% CI upper bound

            # comparison
            for j, layer2 in enumerate(self.target_name):
                comp = (layer1, layer2)
                col_comp.append(comp)

        temp = {
            "comparison": col_comp,
            "mean_diff": tukey_stat[:, 0],
            "p_val_tukey": tukey_stat[:, 1],
            "95%_ci_low": tukey_stat[:, 2],
            "95%_ci_up": tukey_stat[:, 3],
            "dof_between": dof_between,
            "dof_within": dof_within,
            "F_1way_anova": F_val,
            "p_1way_anova": p_val,
        }
        df = pd.DataFrame(temp)

        # # linear fit on mean
        step = 2 / len(self.target_name)
        # y = corr_neuron_mean
        x = step * np.arange(len(self.target_name))

        # model = sm.OLS(y, sm.add_constant(x))
        # results = model.fit()
        # b, m = results.params
        # p_val = results.pvalues[1]  # p-val for slope

        ## start plotting x-corr neurons across all layers
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (15, 5)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=False, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "Neuron's preferred inputs interocular correlation across layer",
            ha="center",
        )
        fig.text(0.5, -0.03, "Layer", ha="center")
        fig.text(-0.03, 0.5, "Pearson correlation", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # box plot properties
        bar_width = 0.075
        linewidth = 1.5
        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="D",
            markersize=6,
            markerfacecolor="firebrick",
            markeredgecolor="firebrick",
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        # # plot the linear fit on median
        # x_fit = np.linspace(-bar_width, x[-1] + bar_width)
        # y_fit = b + m * x_fit
        # axes.plot(x_fit, y_fit, "r--", linewidth=3)

        # # plot dummy dots for legend
        # axes.plot(0, 0, "w.")
        # axes.plot(0, 0, "w.")
        # axes.plot(0, 0, "w.")

        # plt.legend(
        #     [
        #         f"Mean-fit",
        #         f"y = {m:.2f}x{b:.2f}",
        #         f"R2: {results.rsquared:.3f}",
        #         f"p_val: {p_val:.3f}",
        #     ],
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.25),
        #     fancybox=True,
        #     shadow=True,
        #     ncol=4,
        # )

        # upper and lower boxplot y-axis
        x_low = x[0] - bar_width
        x_up = x[-1] + bar_width
        y_low = -1.0
        y_up = 1.1
        y_step = 0.5

        # collect data
        y = []
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            y.append(neuron_pref_input_interocular_allunits_layers[target_layer_name])

        axes.boxplot(
            y,
            widths=bar_width,
            patch_artist=True,
            positions=x,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # Plot line at 0 correlation
        axes.axhline(0, color="red", linestyle="--", linewidth=1.5)

        layer_name_list = [
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33a",
            "34a",
            "35a",
            "36a",
            "37",
        ]
        axes.set_xlim(x_low, x_up)
        axes.set_xticks(x)
        axes.set_xticklabels(layer_name_list)
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.neuron_dir}/PlotBox_neuron_pref_input_interocular_corr.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save auc_mean stat
            df_mean.to_csv(f"{self.neuron_dir}/interocular_corr.csv", index=False)

            # save tukey stat
            df.to_csv(f"{self.neuron_dir}/interocular_corr_tukey.csv", index=False)

    def PlotBox_neuron_pref_input_interocular_corr_xcorr(self, save_flag):
        auc_ratio_xcorr_layers = {}
        auc_ratio_xmatch_layers = {}
        neuron_pref_input_interocular_xcorr_layers = {}  # left-right interocular corr
        neuron_pref_input_interocular_xmatch_layers = {}

        for t, target_layer_name in enumerate(self.target_name):

            ## process preferred inputs
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]
            # reshape into [n_neurons]
            neuron_pref_input_corr = neuron_pref_input_corr.ravel()

            ## process disparity tuning
            # load disparity tuning data [n_neurons, len]
            disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)

            # average across rds_trials [n_neurons, len(disp_ct_list), ]
            disp_tuning_layer_avg = np.mean(disp_tuning_layer, axis=-1)

            # compute the Pearson correlation between non-crds disp_tuning and
            # crds disp_tuning
            disp_tuning_corr_layer = _compute_disp_tuning_corr_layer(
                disp_tuning_layer_avg
            )

            # compute auc ratio
            auc_ratio = self.compute_corr_auc_ratio_layer(
                disp_tuning_corr_layer, omit_outlier=False
            )

            ## omit outlier based on auc_ratio
            # threshold auc_ratio
            threshold = np.mean(auc_ratio) + 6 * np.std(auc_ratio)  # type: ignore
            idx_omit = np.where(auc_ratio > threshold)
            auc_ratio_layer = np.delete(auc_ratio, idx_omit)

            # threshold neuron_pref_input_corr
            neuron_pref_input_corr_layer = np.delete(neuron_pref_input_corr, idx_omit)

            # classify xcorr and xmatch neurons based on auc
            # xcorr: auc > 0.5
            # xmatch: auc <= 0.5
            idx_xcorr = np.where(auc_ratio_layer > 0.5)
            idx_xmatch = np.where(auc_ratio_layer <= 0.5)
            auc_ratio_xcorr_layers[target_layer_name] = auc_ratio_layer[idx_xcorr]
            auc_ratio_xmatch_layers[target_layer_name] = auc_ratio_layer[idx_xmatch]

            neuron_pref_input_interocular_xcorr_layers[target_layer_name] = (
                neuron_pref_input_corr_layer[idx_xcorr]
            )

            neuron_pref_input_interocular_xmatch_layers[target_layer_name] = (
                neuron_pref_input_corr_layer[idx_xmatch]
            )

        corr_neuron_median = np.zeros(len(self.target_name), dtype=np.float32)
        corr_error = np.zeros(len(self.target_name), dtype=np.float32)
        x = []
        y = []
        # collect data
        for t, target_layer_name in enumerate(self.target_name):
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = neuron_pref_input_interocular_xcorr_layers[
                target_layer_name
            ]  # [n_feature_channels, n_disp_channels]

            y.append(neuron_pref_input_corr.ravel())
            x.append(t * np.ones(len(neuron_pref_input_corr.ravel())))

            corr_neuron_median[t] = np.median(neuron_pref_input_corr)
            corr_error[t] = np.std(neuron_pref_input_corr.ravel())

        # linear fit on median
        step = 2 / len(self.target_name)
        y = corr_neuron_median
        x = step * np.arange(len(self.target_name))

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()
        b, m = results.params
        p_val = results.pvalues[1]  # p-val for slope

        ## start plotting x-corr neurons across all layers
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=4, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (24, 10)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=False, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "X-corr neuron's preferred inputs interocular correlation across layer",
            ha="center",
        )
        fig.text(0.5, -0.03, "Layer", ha="center")
        fig.text(-0.03, 0.5, "Pearson correlation", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # box plot properties
        bar_width = 0.075
        linewidth = 1.5
        boxprops = dict(
            linewidth=linewidth, color="black", facecolor=(0, 0, 0, 0)
        )  # transparent box
        medianprops = dict(linestyle="-", linewidth=linewidth, color="black")
        meanprops = dict(
            marker="D",
            markersize=6,
            markerfacecolor="firebrick",
            markeredgecolor="firebrick",
        )
        whiskerprops = dict(linewidth=linewidth)
        capprops = dict(linewidth=linewidth)

        # plot the linear fit on median
        x_fit = np.linspace(-bar_width, x[-1] + bar_width)
        y_fit = b + m * x_fit
        axes.plot(x_fit, y_fit, "r--", linewidth=3)

        # plot dummy dots for legend
        axes.plot(0, 0, "w.")
        axes.plot(0, 0, "w.")
        axes.plot(0, 0, "w.")

        plt.legend(
            [
                f"Median-fit",
                f"y = {m:.2f}x{b:.2f}",
                f"R2: {results.rsquared:.3f}",
                f"p_val: {p_val:.3f}",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        # upper and lower boxplot y-axis
        x_low = x[0] - bar_width
        x_up = x[-1] + bar_width
        y_low = -1.0
        y_up = 1.1
        y_step = 0.25

        # collect data
        y = []
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            y.append(neuron_pref_input_interocular_xcorr_layers[target_layer_name])

        axes.boxplot(
            y,
            widths=bar_width,
            patch_artist=True,
            positions=x,
            medianprops=medianprops,
            boxprops=boxprops,
            meanprops=meanprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            showfliers=False,
            showmeans=True,
        )

        # Plot line at 0 correlation
        axes.axhline(0, color="k", linestyle="--", linewidth=1.5)

        layer_name_list = [
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33a",
            "34a",
            "35a",
            "36a",
            "37",
        ]
        axes.set_xlim(x_low, x_up)
        axes.set_xticks(x)
        axes.set_xticklabels(layer_name_list)
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.dotDens_dir}/PlotBox_neuron_pref_input_interocular_corr_xcorr.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def compute_MI_layers(self, bins):
        MI_layers = {}
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]

            print(f"Compute MI in {target_layer_name}")

            # get disparity tuning in target_layer_name
            disp_tuning_layer = self.get_disp_tuning_layer(target_layer_name)

            # compute mutual information between crds_disp_tuning and
            # non-crds disp_tuning
            MI_layer = compute_MI_disp_tuning_layer(disp_tuning_layer, bins)

            MI_layers[target_layer_name] = MI_layer

        return MI_layers

    def plotLine_disp_tuning_corr(self, save_flag):
        # compute disparity tuning
        (
            corr_crds_ards_all,  # dict(layer_name: [n_neurons])
            corr_crds_hmrds_all,
        ) = self._compute_disp_tuning_corr()

        # average across neurons in each layer
        corr_crds_ards_avg = np.empty(len(self.target_name), dtype=np.float32)
        corr_crds_hmrds_avg = np.empty(len(self.target_name), dtype=np.float32)
        corr_crds_ards_err = np.empty(len(self.target_name), dtype=np.float32)
        corr_crds_hmrds_err = np.empty(len(self.target_name), dtype=np.float32)
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]

            corr_crds_ards_avg[t] = corr_crds_ards_all[target_layer_name].mean()
            corr_crds_hmrds_avg[t] = corr_crds_hmrds_all[target_layer_name].mean()

            corr_crds_ards_err[t] = corr_crds_ards_all[target_layer_name].std()
            corr_crds_hmrds_err[t] = corr_crds_hmrds_all[target_layer_name].std()

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (12, 7)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "Pearson correlation between disparity tuning",
            ha="center",
        )
        # fig.text(0.5, -0.03, "Pearson correlation", ha="center")
        fig.text(-0.03, 0.5, "Pearson correlation", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        x = np.arange(len(self.target_name))
        axes.errorbar(
            x,
            corr_crds_ards_avg,
            yerr=corr_crds_ards_err,
            lw=3,
            c="red",
            ls="-",
            capsize=7,
        )
        axes.errorbar(
            x,
            corr_crds_hmrds_avg,
            yerr=corr_crds_hmrds_err,
            lw=3,
            c="blue",
            ls="-",
            capsize=7,
        )
        axes.legend(["cRDS vs. aRDS", "cRDS vs. hmRDS"])

        # plot the marker
        markersize = 12
        axes.plot(x, corr_crds_ards_avg, "o", markersize=markersize, c="red")

        # plot the marker
        axes.plot(x, corr_crds_hmrds_avg, "o", markersize=markersize, c="blue")

        # upper and lower plot
        x_low = 0.0
        x_up = x[-1]
        y_low = -1.0
        y_up = 1.1
        y_step = 0.25

        # Plot line at 0 correlation
        axes.axhline(0, color="k", linestyle="--", linewidth=1.5)

        axes.set_xlim(x_low - 1, x_up + 1)
        axes.set_xticks(x)
        axes.set_xticklabels(self.target_name, rotation=45)
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.neuron_dir}/PlotLine_disp_tuning_correlation.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_disp_tuning_corr(self, save_flag):
        # compute disparity tuning
        (
            corr_crds_ards_all,  # dict(layer_name: [n_neurons])
            corr_crds_hmrds_all,
        ) = self._compute_disp_tuning_corr()

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (20, 20)
        n_row = 5
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "Pearson correlation between disparity tuning",
            ha="center",
        )
        fig.text(0.5, -0.03, "Pearson correlation (cRDS vs. hmRDS)", ha="center")
        fig.text(
            -0.03, 0.5, "Pearson correlation (cRDS vs. aRDS)", va="center", rotation=90
        )
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # upper and lower plot
        x_low = -1.0
        x_up = 1.1
        x_step = 0.5
        y_low = -1.0
        y_up = 1.1
        y_step = 0.5

        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            id_row = t // n_col
            id_col = t % n_col
            axes[id_row, id_col].plot(
                corr_crds_hmrds_all[target_layer_name],
                corr_crds_ards_all[target_layer_name],
                "k.",
            )

            # plot a marker for pure cross-correlation
            axes[id_row, id_col].plot(0, -1, "X", markersize=12, c="red")

            # plot a marker for pure cross-matching
            axes[id_row, id_col].plot(1, 0, "X", markersize=12, c="blue")

            # Plot line at 0 correlation
            axes[id_row, id_col].axhline(0, color="k", linestyle="--", linewidth=1.5)
            axes[id_row, id_col].axvline(0, color="k", linestyle="--", linewidth=1.5)

            axes[id_row, id_col].set_title(target_layer_name)

            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )
            axes[id_row, id_col].set_ylim(y_low - 0.1, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        # plot a dummy for legend
        axes[id_row, id_col + 1].plot(2, 2, "X", markersize=12, c="red")
        # plot a marker for pure cross-matching
        axes[id_row, id_col + 1].plot(2, 2, "X", markersize=12, c="blue")
        axes[id_row, id_col + 1].legend(
            ["pure X-correlation", "pure X-matching"], loc="lower right"
        )
        axes[id_row, id_col + 1].axis("off")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.neuron_dir}/PlotScatter_disp_tuning_correlation.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotHist_auc_ratio(self, auc_ratio_all_layer, save_flag):
        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (18, 18)
        n_row = 5
        n_col = 4
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "AUC between disparity tuning non-cRDS vs cRDS",
            ha="center",
        )
        fig.text(0.5, -0.01, "AUC Ratio", ha="center")
        fig.text(-0.01, 0.5, "Neuronal unit density", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # upper and lower plot
        x_low = 0
        x_up = 1.1
        x_step = 0.5
        y_low = 0
        y_up = 7
        y_step = 2

        cm = plt.cm.get_cmap("seismic")
        layer_name_list = [
            "Layer 19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
        ]

        for t, target_layer_name in enumerate(self.target_name):
            id_row = t // n_col
            id_col = t % n_col

            _, bins, patches = axes[id_row, id_col].hist(
                auc_ratio_all_layer[target_layer_name],
                bins=20,
                range=(x_low, x_up),
                density=True,
            )
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            # maxi = np.abs(bin_centers).max()
            norm = plt.Normalize(0, 1)

            for c, p in zip(bin_centers, patches):
                plt.setp(p, "facecolor", cm(norm(c)))

            axes[id_row, id_col].set_title(layer_name_list[t], y=1, pad=-20)

            axes[id_row, id_col].set_xlim(x_low, x_up)
            axes[id_row, id_col].set_xticks(np.arange(x_low, x_up, x_step))
            axes[id_row, id_col].set_xticklabels(
                np.round(np.arange(x_low, x_up, x_step), 2)
            )
            axes[id_row, id_col].set_ylim(y_low - 0.1, y_up)
            axes[id_row, id_col].set_yticks(np.arange(y_low, y_up, y_step))
            axes[id_row, id_col].set_yticklabels(
                np.round(np.arange(y_low, y_up, y_step), 2)
            )

            # remove top and right frame
            axes[id_row, id_col].spines["top"].set_visible(False)
            axes[id_row, id_col].spines["right"].set_visible(False)

            # show ticks on the left and bottom axis
            axes[id_row, id_col].xaxis.set_ticks_position("bottom")
            axes[id_row, id_col].yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.dotDens_dir}/PlotHist_auc_corr_ratio.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_auc_ratio(self, auc_ratio_all_layer, save_flag):
        """
        plot auc_ratio across all layers

        Args:
            auc_ratio_all_layer (dict): a dict containing the auc_ratio
                from layer 19 to layer 37

                auc_ratio_all_layer[layer_name]: np.array[n_neurons]

            save_flag (_type_): _description_
        """

        auc_avg = np.zeros(len(self.target_name), dtype=np.float32)
        auc_std = np.zeros(len(self.target_name), dtype=np.float32)
        # average across neurons in each layer
        for t, target_layer_name in enumerate(self.target_name):
            auc = auc_ratio_all_layer[target_layer_name]
            auc_avg[t] = np.mean(auc)
            auc_std[t] = sem(auc)  # np.std(auc)  #

        # one-way ANOVA
        data = [auc_ratio_all_layer[layer] for layer in self.target_name]
        dof_between = len(self.target_name) - 1
        # flatten data
        a = [x for row in data for x in row]
        dof_within = len(a) - len(self.target_name)
        F_val, p_val = stats.f_oneway(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
        )

        # post-hoc test
        res = stats.tukey_hsd(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
            data[14],
            data[15],
            data[16],
            data[17],
            data[18],
        )

        # create pandas df for recording the mean and sem AUC ratio
        # [layer_name, auc_ratio_mean, auc_ratio_sem]
        temp = {
            "layer_name": self.target_name,
            "auc_ratio_mean": auc_avg,
            "auc_ratio_sem": auc_std,
        }
        df_mean = pd.DataFrame(temp)

        # create pandas df for the statistical test
        # [stat, p_val, lower_ci, upper_ci]
        n_row = len(self.target_name) ** 2
        tukey_stat = np.empty((n_row, 4), dtype=np.float32)
        col_comp = []
        for i, layer1 in enumerate(self.target_name):
            id_start = i * len(self.target_name)
            id_end = id_start + len(self.target_name)
            tukey_stat[id_start:id_end, 0] = res.statistic[i]  # mean difference
            tukey_stat[id_start:id_end, 1] = res.pvalue[i]  # p_val
            tukey_stat[id_start:id_end, 2] = res.confidence_interval().low[
                i
            ]  # 95% CI lower bound
            tukey_stat[id_start:id_end, 3] = res.confidence_interval().high[
                i
            ]  # 95% CI upper bound

            # comparison
            for j, layer2 in enumerate(self.target_name):
                comp = (layer1, layer2)
                col_comp.append(comp)

        temp = {
            "comparison": col_comp,
            "mean_diff": tukey_stat[:, 0],
            "p_val_tukey": tukey_stat[:, 1],
            "95%_ci_low": tukey_stat[:, 2],
            "95%_ci_up": tukey_stat[:, 3],
            "dof_between": dof_between,
            "dof_within": dof_within,
            "F_1way_anova": F_val,
            "p_1way_anova": p_val,
        }
        df = pd.DataFrame(temp)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (16, 4)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(0.5, -0.05, "Layer", ha="center")
        fig.text(-0.01, 0.5, "AUC ratio", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        x = np.arange(len(self.target_name))
        axes.errorbar(
            x,
            auc_avg,
            yerr=auc_std,
            lw=3,
            c="black",
            ls="-",
            capsize=7,
        )

        # print out one-way ANOVA
        fig.text(
            0.1,
            -0.2,
            f"F({dof_between}, {dof_within}) = {F_val:.3f}; p_val = {p_val:.3f}",
        )

        # plot the marker
        markersize = 12
        axes.plot(x, auc_avg, "o", markersize=markersize, c="black")

        # upper and lower plot
        x_low = 0.0
        x_up = x[-1]
        y_low = 0.1
        y_up = 0.81
        y_step = 0.1
        # y_low = 0.0
        # y_up = 0.51
        # y_step = 0.1

        layer_name_list = [
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
        ]
        axes.set_xlim(x_low - 1, x_up + 1)
        axes.set_xticks(x)
        axes.set_xticklabels(layer_name_list)
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # plot 0.5 threshold
        axes.plot([x_low - 1, x_up + 1], [0.5, 0.5], "r--", linewidth=3)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        # save
        if save_flag == 1:
            fig.savefig(
                f"{self.dotDens_dir}/plotLine_auc_corr_ratio.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save auc_mean stat
            df_mean.to_csv(f"{self.dotDens_dir}/auc_ratio.csv", index=False)

            # save tukey stat
            df.to_csv(f"{self.dotDens_dir}/auc_ratio_tukey.csv", index=False)