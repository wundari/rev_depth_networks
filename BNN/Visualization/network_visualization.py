# %% load necessary modules

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

# %%
class NetworkVisualization:
    def __init__(self, params_train):

        self.epoch_to_load = params_train["epoch_to_load"]
        self.iter_to_load = params_train["iter_to_load"]
        self.batch_size = params_train["batch_size"]  # training batch size
        self.c_disp_shift = params_train["c_disp_shift"]  # disparity shift multiplier
        self.dataset_to_process = params_train["sceneflow_type"]
        self.flip_input = params_train["flip_input"]
        self.target_disp = params_train["target_disp"]

        if self.flip_input:
            self.save_dir = (
                "results/sceneflow/"
                + f"{self.dataset_to_process}"
                + f"/shift_{self.c_disp_shift}_median_wrt_right"
            )
        else:
            self.save_dir = (
                "results/sceneflow/"
                + f"{self.dataset_to_process}"
                + f"/shift_{self.c_disp_shift}_median_wrt_left"
            )

        self.network_diss_dir = (
            f"{self.save_dir}/"
            + f"epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}/"
            + "network_dissection/"
            + f"target_disp_{self.target_disp}px/"
        )
        if not os.path.exists(self.network_diss_dir):
            os.mkdir(self.network_diss_dir)

        self.neuron_dir = self.network_diss_dir + "/neuron_optimization"
        if not os.path.exists(self.neuron_dir):
            os.mkdir(self.neuron_dir)

        self.neuron_vis_dir = self.network_diss_dir + "/neuron_visualization"
        if not os.path.exists(self.neuron_vis_dir):
            os.makedirs(self.neuron_vis_dir)

        self.target_name = [
            "layer19",
            "layer20",
            "layer21",
            "layer22",
            "layer23",
            "layer24",
            "layer25",
            "layer26",
            "layer27",
            "layer28",
            "layer29",
            "layer30",
            "layer31",
            "layer32",
            "layer33a",
            "layer34a",
            "layer35a",
            "layer36a",
            "layer37",
        ]

    @staticmethod
    def make_grid_image(
        img_left_layer, img_right_layer, disp_channel: int, n_img_row: int
    ):
        C, _, H, W, _ = img_left_layer.shape
        pad_within = 8  # distance with between left and right images
        pad_between = 3 * pad_within  # distance width between pairs

        # disp_channel = 0
        # img_per_row = 12
        n_img_col = C // n_img_row
        h_grid = n_img_row * H + (n_img_row - 1) * pad_between
        w_grid = (
            (2 * n_img_col * W)
            + (n_img_col * pad_within)
            + ((n_img_col - 1) * pad_between)
        )
        img_grid = np.zeros((h_grid, w_grid, 3), dtype=np.uint8)
        count = 0
        for row in range(n_img_row):
            id_row_start = row * (H + pad_between)
            id_row_end = id_row_start + H
            for col in range(n_img_col):
                # left image
                id_col_start = 2 * col * W + col * (pad_within + pad_between)
                id_col_end = id_col_start + W

                img_left = (img_left_layer[count, disp_channel] * 255.0).astype(
                    np.uint8
                )
                img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img_left

                # right image
                id_col_start = id_col_end + pad_within
                id_col_end = id_col_start + W

                img_right = (img_right_layer[count, disp_channel] * 255.0).astype(
                    np.uint8
                )
                img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img_right

                count += 1

        return img_grid

    @staticmethod
    def make_grid_image_v2(img_left_layer, img_right_layer, n_img_row: int):
        C, H, W, _ = img_left_layer.shape
        pad_within = 8  # distance with between left and right images
        pad_between = 3 * pad_within  # distance between pairs

        # disp_channel = 0
        # img_per_row = 12
        n_img_col = C // n_img_row
        if (n_img_col * n_img_row) <= C:
            n_img_col += 1
        h_grid = n_img_row * H + (n_img_row - 1) * pad_between
        w_grid = (
            (2 * n_img_col * W)
            + (n_img_col * pad_within)
            + ((n_img_col - 1) * pad_between)
        )
        img_grid = np.zeros((h_grid, w_grid, 3), dtype=np.uint8)
        count = 0
        for row in range(n_img_row):
            id_row_start = row * (H + pad_between)
            id_row_end = id_row_start + H
            for col in range(n_img_col):
                if count < C:
                    # left image
                    id_col_start = 2 * col * W + col * (pad_within + pad_between)
                    id_col_end = id_col_start + W

                    img_left = (img_left_layer[count] * 255.0).astype(np.uint8)
                    img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = (
                        img_left
                    )

                    # right image
                    id_col_start = id_col_end + pad_within
                    id_col_end = id_col_start + W

                    img_right = (img_right_layer[count] * 255.0).astype(np.uint8)
                    img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = (
                        img_right
                    )

                    count += 1
                else:
                    return img_grid

        return img_grid

    def visualize_neuron_activation_in_single_disp_channel(
        self,
        img_left_layer,
        img_right_layer,
        target_layer_name: str,
        n_img_row: int,
        rgb_flag: bool = True,
        save_flag: bool = False,
    ):
        """
        Visualize the optimized neuron activation (the neuron in the center)
        in each feature channel and disparity channel.

        Args:
            img_left_layer (_type_): _description_
            img_right_layer (_type_): _description_
            layer_name (str): _description_
            n_img_row (int): _description_
        """

        layer_dir = f"{self.neuron_vis_dir}/{target_layer_name}"
        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)

        n_disp_channels = img_left_layer.shape[1]

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (10, 10)
        n_row = 1
        n_col = 1

        for d in range(n_disp_channels):
            print(
                f"generate neuron visualization in layer: {target_layer_name}, "
                + f"disp_channel: {d}/{n_disp_channels}"
            )
            fig, axes = plt.subplots(
                nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
            )

            fig.text(
                0.5,
                1.0,
                "Binocular neuron optimized activation\n"
                + f"layer: {target_layer_name}, disp_channel: {d}",
                ha="center",
            )
            # fig.text(0.5, -0.04, "Channel", ha="center")

            fig.tight_layout()

            img_grid = self.make_grid_image(
                img_left_layer, img_right_layer, d, n_img_row
            )
            if rgb_flag:
                axes.imshow(img_grid)
                axes.axis("off")

                if save_flag:
                    fig.savefig(
                        f"{layer_dir}/Neuron_activation_{target_layer_name}_"
                        + f"disp_channel_{d}_rgb.pdf",
                        dpi=600,
                        bbox_inches="tight",
                    )
            else:
                axes.imshow(img_grid.mean(axis=2), cmap="jet")
                axes.axis("off")

                if save_flag:
                    fig.savefig(
                        f"{layer_dir}/Neuron_activation_{target_layer_name}_"
                        + f"disp_channel_{d}.pdf",
                        dpi=600,
                        bbox_inches="tight",
                    )

            plt.close()

    def plotBox_neuron_pref_input_interocular_corr(self, save_flag):
        """
        Plot the distribution of the Pearson correlation between the left and right
        neurons preffered inputs in each DNN layer.
        Also performed the linear fit on the median

        Args:
            save_flag (_type_): _description_
        """
        corr_neuron_median = np.zeros(len(self.target_name), dtype=np.float32)
        neuron_pref_input_corr_all = {}
        corr_error = np.zeros(len(self.target_name), dtype=np.float32)
        x = []
        y = []
        # collect data
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]

            neuron_pref_input_corr_all[target_layer_name] = (
                neuron_pref_input_corr.ravel()
            )

            y.append(neuron_pref_input_corr.ravel())
            x.append(t * np.ones(len(neuron_pref_input_corr.ravel())))

            corr_neuron_median[t] = np.median(neuron_pref_input_corr)
            corr_error[t] = np.std(neuron_pref_input_corr.ravel())

        # one-way ANOVA
        data = [neuron_pref_input_corr_all[layer] for layer in self.target_name]
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
        tukey_df = pd.DataFrame(temp)

        # linear fit on median
        step = 2 / len(self.target_name)
        y = corr_neuron_median
        x = step * np.arange(len(self.target_name))

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()
        b, m = results.params
        # print(results.params)
        # print(results.summary())

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="muted")
        # plt.rcParams["font.family"] = "serif"

        figsize = (18, 8)
        n_row = 1
        n_col = 1
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        fig.text(
            0.5,
            1.02,
            "Neuron's preferred inputs interocular correlation across layer",
            ha="center",
        )
        # fig.text(0.5, -0.03, "Layer", ha="center")
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

        # plot the linear fit on mean
        b, m = results.params  #
        x_fit = np.linspace(-bar_width, x[-1] + bar_width)
        y_fit = b + m * x_fit
        axes.plot(x_fit, y_fit, "r--", linewidth=3)

        # plot dummy dots for legend
        axes.plot(0, 0, "w.")
        axes.plot(0, 0, "w.")

        plt.legend(
            [f"Median-fit", f"y = {m:.2f}x{b:.2f}", f"R2: {results.rsquared:.3f}"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            fancybox=True,
            shadow=True,
            ncol=3,
        )

        # upper and lower boxplot y-axis
        x_low = x[0] - bar_width
        x_up = x[-1] + bar_width
        y_low = -1.0
        y_up = 1.1
        y_step = 0.25

        # collect data ards, hmrds, crds
        y = []
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            y.append(neuron_pref_input_corr_all[target_layer_name])

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

        axes.set_xlim(x_low - bar_width, x_up + bar_width)
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
                f"{self.neuron_dir}/PlotBox_neuron_pref_input_interocular_corr.pdf",
                dpi=600,
                bbox_inches="tight",
            )

            # save tukey test
            tukey_df.to_csv(
                f"{self.neuron_dir}/tukey_neuron_pref_input_interocular_corr.csv",
                index=False,
            )

    def visualize_neuron_activation_mzm_interocular_corr(self, save_flag):
        """
        get neuron preferred inputs with max-zero-min interocular correlation
        """

        img_left_mzm = {}  # img with max-zero-min interocular-corr
        img_right_mzm = {}
        # collect data
        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]

            # load neuron activation [n_feature_channels, n_disp_channels, h, w, rgb]
            img_left = np.load(
                f"{self.neuron_dir}/img_left_{target_layer_name}_conv.npy"
            )
            img_right = np.load(
                f"{self.neuron_dir}/img_right_{target_layer_name}_conv.npy"
            )

            # load the pearson corr between neuron left-right preferred inputs
            neuron_pref_input_corr = np.load(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy"
            )  # [n_feature_channels, n_disp_channels]

            # fetch neurons with max, near-zero, and min interocular-corr
            # left image
            img_l_mzm = []
            img_r_mzm = []
            # max interocular-corr
            id_feature_channel, id_disp_channel = np.where(
                neuron_pref_input_corr == neuron_pref_input_corr.max()
            )
            img_l_mzm.append(img_left[id_feature_channel[0], id_disp_channel[0]])
            img_r_mzm.append(img_right[id_feature_channel[0], id_disp_channel[0]])

            # near-zero interocular-corr
            id_feature_channel, id_disp_channel = np.where(neuron_pref_input_corr >= 0)
            img_l_mzm.append(img_left[id_feature_channel[-1], id_disp_channel[-1]])
            img_r_mzm.append(img_right[id_feature_channel[-1], id_disp_channel[-1]])

            # min interocular-corr
            id_feature_channel, id_disp_channel = np.where(
                neuron_pref_input_corr == neuron_pref_input_corr.min()
            )
            img_l_mzm.append(img_left[id_feature_channel[0], id_disp_channel[0]])
            img_r_mzm.append(img_right[id_feature_channel[0], id_disp_channel[0]])

            img_left_mzm[target_layer_name] = img_l_mzm
            img_right_mzm[target_layer_name] = img_r_mzm

        # make image grid
        pad_within = 12  # distance between left and right images
        pad_between = 3 * pad_within  # distance between image pairs

        img_h, img_w = img_left_mzm["layer19"][0].shape[:2]
        grid_h = (
            len(self.target_name) * img_h + (len(self.target_name) - 1) * pad_between
        )
        grid_w = 6 * img_w + 3 * pad_within + 2 * pad_between
        img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for row in range(len(self.target_name)):

            target_layer_name = self.target_name[row]
            img_left = img_left_mzm[target_layer_name]
            img_right = img_right_mzm[target_layer_name]

            id_row_start = row * (img_h + pad_between)
            id_row_end = id_row_start + img_h

            # max interocular-corr
            # left image
            id_col_start = 0
            id_col_end = id_col_start + img_w
            img = (img_left[0] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img
            # right image
            id_col_start = id_col_end + pad_within
            id_col_end = id_col_start + img_w
            img = (img_right[0] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img

            # near-zero interocular-corr
            # left image
            id_col_start = id_col_end + pad_between
            id_col_end = id_col_start + img_w
            img = (img_left[1] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img
            # right image
            id_col_start = id_col_end + pad_within
            id_col_end = id_col_start + img_w
            img = (img_right[1] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img

            # min interocular-corr
            # left image
            id_col_start = id_col_end + pad_between
            id_col_end = id_col_start + img_w
            img = (img_left[2] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img
            # right image
            id_col_start = id_col_end + pad_within
            id_col_end = id_col_start + img_w
            img = (img_right[2] * 255.0).astype(np.uint8)
            img_grid[id_row_start:id_row_end, id_col_start:id_col_end] = img

        # start plotting
        # rgb image
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 25)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            0.92,
            "Binocular neuron optimized activation",
            ha="center",
        )
        fig.text(
            0.2,
            0.88,
            "Max\ninterocular-corr",
            ha="center",
        )
        fig.text(
            0.5,
            0.88,
            "Near-zero\ninterocular-corr",
            ha="center",
        )
        fig.text(
            0.8,
            0.88,
            "Min\ninterocular-corr",
            ha="center",
        )

        for t, layer_name in enumerate(self.target_name):
            y = 0.86 - t * 0.04
            fig.text(-0.05, y, layer_name, ha="center")

        fig.tight_layout()
        axes.imshow(img_grid)
        axes.axis("off")

        if save_flag:
            fig.savefig(
                f"{self.neuron_vis_dir}/Neuron_activation_mzm_rgb.png",
                dpi=600,
                bbox_inches="tight",
            )

        # heat map
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 25)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            0.92,
            "Binocular neuron optimized activation",
            ha="center",
        )
        fig.text(
            0.2,
            0.88,
            "Max\ninterocular-corr",
            ha="center",
        )
        fig.text(
            0.5,
            0.88,
            "Near-zero\ninterocular-corr",
            ha="center",
        )
        fig.text(
            0.8,
            0.88,
            "Min\ninterocular-corr",
            ha="center",
        )

        for t, layer_name in enumerate(self.target_name):
            y = 0.86 - t * 0.04
            fig.text(-0.05, y, layer_name, ha="center")

        fig.tight_layout()
        axes.imshow(img_grid.mean(axis=2), cmap="jet")
        axes.axis("off")

        if save_flag:
            fig.savefig(
                f"{self.neuron_vis_dir}/Neuron_activation_mzm_heat.png",
                dpi=600,
                bbox_inches="tight",
            )