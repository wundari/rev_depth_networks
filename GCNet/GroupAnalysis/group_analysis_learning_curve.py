# %% load necessary modules
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from config.config import GCNetconfig

import seaborn as sns
import matplotlib.pyplot as plt

import os
import json
from pathlib import Path
from jaxtyping import Float


# %%
class GA_LearningCurve:
    """
    Class for analyzing and plotting learning curves for different binocular
    interactions.

    It gathers training and validation losses and accuracies across multiple
    seeds, and provides methods to plot these metrics.
    """

    def __init__(self, config: GCNetconfig):

        self.config = config

        self.interactions = list(config.interactions)

        # folder location where training loss is stored
        self.experiment_dir = (
            f"run/{config.dataset}/bino_interaction_{config.binocular_interaction}"
        )

        # create folder for saving plots
        self.plot_dir = f"{self.experiment_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # create folder for saving group analysis
        self.group_dir = f"run/{config.dataset}/bino_interaction_group"
        if not os.path.exists(self.group_dir):
            os.makedirs(self.group_dir)

        # create folder for saving group analysis plots
        self.group_plot_dir = f"{self.group_dir}/plots"
        if not os.path.exists(self.group_plot_dir):
            os.makedirs(self.group_plot_dir)

        # create folder for saving group analysis stats
        self.group_stat_dir = f"{self.group_dir}/stats"
        if not os.path.exists(self.group_stat_dir):
            os.makedirs(self.group_stat_dir)

        # save config
        # Analysis must not overwrite training metadata.

    def save_config(self):
        config_dict = self.config.to_dict()
        with open(f"{self.group_stat_dir}/analysis_config.json", "w") as f:
            json.dump(config_dict, f)

    def update_bino_interaction(self, interaction: str) -> None:
        """
        Update the binocular interaction in the config and experiment directory.

        Args:
            interaction (str): The new binocular interaction to set.
        """
        # update the binocular interaction in the config
        previous = self.config.binocular_interaction
        self.config.binocular_interaction = interaction

        # update the experiment and plot directories based on the new interaction
        self.experiment_dir = (
            f"run/{self.config.dataset}/bino_interaction_{interaction}"
        )
        self.plot_dir = f"{self.experiment_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # save config
        # Analysis must not overwrite training metadata.

        print(
            "Update binocular interaction from "
            + f"{previous} to {interaction}.\n"
            + f"Experiment directory: {self.experiment_dir}\n"
            + f"Plot directory: {self.plot_dir}"
        )

    def _load_metric(self, name):
        rows, schedules = [], []
        for seed in self.config.seed_to_analyse:
            root = Path(self.experiment_dir)
            candidates = []
            for folder in [root / str(seed), *sorted(root.glob("experiment_*"))]:
                if not (folder / f"{name}.npy").exists():
                    continue
                metadata = folder / "config.json"
                if not metadata.exists():
                    raise ValueError(f"Missing run metadata: {metadata}")
                cfg = json.loads(metadata.read_text())
                if (cfg.get("seed") == seed and
                    cfg.get("binocular_interaction") == self.config.binocular_interaction and
                    cfg.get("dataset") == self.config.dataset):
                    candidates.append((folder, cfg))
            if len(candidates) != 1:
                raise ValueError(f"Expected one matching run for seed {seed}, found {len(candidates)} in {root}")
            folder, cfg = candidates[0]
            values = np.load(folder / f"{name}.npy")
            if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                raise ValueError(f"Invalid metric array: {folder}/{name}.npy")
            steps_path = folder / "eval_steps.npy"
            steps = np.load(steps_path) if steps_path.exists() else np.arange(len(values)) * cfg["eval_interval"]
            if steps.shape != values.shape or not np.isfinite(steps).all() or np.any(np.diff(steps) <= 0):
                raise ValueError(f"Invalid evaluation steps in {folder}")
            rows.append(values)
            schedules.append(steps)
        if not rows or any(not np.array_equal(schedules[0], x) for x in schedules[1:]):
            raise ValueError("Seeds must have matching evaluation schedules")
        if hasattr(self, "eval_steps") and not np.array_equal(self.eval_steps, schedules[0]):
            raise ValueError("Interactions/metrics must have matching evaluation schedules")
        self.eval_steps = schedules[0]
        return np.stack(rows)

    def get_train_loss(self):
        return self._load_metric("losses_train")

    def get_train_acc(self):
        return self._load_metric("accs_train")

    def get_val_loss(self):
        return self._load_metric("losses_val")

    def get_val_acc(self):
        return self._load_metric("accs_val")

    def _steps(self, values):
        steps = getattr(self, "eval_steps", None)
        if steps is None:
            return np.arange(len(values)) * self.config.eval_interval
        if len(steps) != len(values):
            raise ValueError("Plot values do not match evaluation steps")
        return steps

    def _compute_statistics(
        self, data: Float[np.ndarray, "len_interactions n_seed"], data_name: str
    ):

        # overall non-parametric repeated-measured test
        if data.ndim != 2 or data.shape[0] != len(self.interactions) or data.shape[1] < 2 or not np.isfinite(data).all():
            raise ValueError("Statistics require finite interaction-by-seed data with at least two seeds")
        if len(self.interactions) < 3:
            raise ValueError("Friedman test requires at least three interactions")
        if np.all(data == data[0]):
            friedman = (0.0, 1.0)
        else:
            friedman = stats.friedmanchisquare(*data)
        pd.DataFrame([{"statistic": friedman[0], "p_value": friedman[1]}]).to_csv(
            f"{self.group_stat_dir}/friedman_{data_name}.csv", index=False)

        print(f"Friedman test: {friedman}")

        # post-hoc pairwise Wilcoxon tests
        pairs = list(combinations(range(len(self.interactions)), 2))
        p_vals = []
        rows = []
        for i, j in pairs:
            stat, p = (0.0, 1.0) if np.array_equal(data[i], data[j]) else stats.wilcoxon(data[i], data[j])
            p_vals.append(p)

            rows.append(
                {
                    "data_name": data_name,
                    "model_1": self.interactions[i],
                    "model_2": self.interactions[j],
                    "mean_model_1": np.mean(data[i]),
                    "mean_model_2": np.mean(data[j]),
                    "mean_difference_model_1_minus_model_2": np.mean(data[i] - data[j]),
                    "wilcoxon_statistic": stat,
                    "raw_p_value": p,
                }
            )

        # Holm-Bonferroni correction
        reject, p_val_corrected, _, _ = multipletests(p_vals, method="holm")

        for row, p_corr, significant in zip(rows, p_val_corrected, reject):
            row["holm_corrected_p_val"] = p_corr
            row["significant_after_holm"] = significant

        df_posthoc = pd.DataFrame(rows)
        df_posthoc.to_csv(
            f"{self.group_stat_dir}/posthoc_wilcoxon_results_{data_name}.csv",
            index=False,
        )

        print(df_posthoc)
        print(
            f"Saved to '{self.group_stat_dir}/posthoc_wilcoxon_results_{data_name}.csv'"
        )

    def compute_statistics_all(self):

        train_loss_all_interactions = np.empty(
            (len(self.interactions), len(self.config.seed_to_analyse)), dtype=np.float64
        )
        train_acc_all_interactions = np.empty(
            (len(self.interactions), len(self.config.seed_to_analyse)), dtype=np.float64
        )
        val_loss_all_interactions = np.empty(
            (len(self.interactions), len(self.config.seed_to_analyse)), dtype=np.float64
        )
        val_acc_all_interactions = np.empty(
            (len(self.interactions), len(self.config.seed_to_analyse)), dtype=np.float64
        )

        for i, interaction in enumerate(self.interactions):

            # load train loss for each interaction
            self.update_bino_interaction(interaction)
            train_losses = self.get_train_loss()  # [n_seed, n_step]
            train_accs = self.get_train_acc()  # [n_seed, n_step]
            val_losses = self.get_val_loss()  # [n_seed, n_step]
            val_accs = self.get_val_acc()  # [n_seed, n_step]

            # gather the train losses and accs at the last iter
            train_loss_all_interactions[i] = train_losses[:, -1]
            train_acc_all_interactions[i] = train_accs[:, -1]
            val_loss_all_interactions[i] = val_losses[:, -1]
            val_acc_all_interactions[i] = val_accs[:, -1]

        # overall non-parametric repeated-measured test
        print("### Statistics for train losses ###")
        data_name = "train_loss"
        self._compute_statistics(train_loss_all_interactions, data_name)
        print()

        print("### Statistics for validation losses ###")
        data_name = "val_loss"
        self._compute_statistics(val_loss_all_interactions, data_name)
        print()

        print("### Statistics for train 3-pix accuracies ###")
        data_name = "train_acc"
        self._compute_statistics(train_acc_all_interactions, data_name)
        print()

        print("### Statistics for validation 3-pix accuracies ###")
        data_name = "val_acc"
        self._compute_statistics(val_acc_all_interactions, data_name)
        print()

    def plotLine_loss_all_seeds(
        self,
        train_losses: Float[np.ndarray, "n_seed n_step"],
        val_losses: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training and validation loss across all seeds
        of a given binocular interaction.

        Args:
            train_losses (Float[np.ndarray, "n_seed n_step"]): Training losses
                for each seed.

            val_losses (Float[np.ndarray, "n_seed n_step"]): Validation losses
                for each seed.

            save_flag <bool>: If True, save the plot to a file.
                Default is False.
        Returns:
            None

        """

        # average losses across seeds
        train_loss_avg = np.mean(train_losses, axis=0)  # [n_steps]
        train_loss_std = np.std(train_losses, axis=0)
        val_loss_avg = np.mean(val_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Train and validation losses all seeds \n({self.config.binocular_interaction})",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#00CED1"]
        axes.plot(self._steps(train_loss_avg), train_loss_avg, linewidth=2, color=colors[0], label="Train")
        axes.fill_between(
            self._steps(train_loss_avg),
            train_loss_avg - train_loss_std,
            train_loss_avg + train_loss_std,
            color=colors[0],
            alpha=0.2,
        )
        axes.plot(self._steps(val_loss_avg), val_loss_avg, linewidth=2, color=colors[1], label="Val")
        axes.fill_between(
            self._steps(val_loss_avg),
            val_loss_avg - val_loss_std,
            val_loss_avg + val_loss_std,
            color=colors[1],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel(self.config.loss)




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

        plt.legend(
            loc="upper right",
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
                f"{self.plot_dir}/plotLine_loss_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotLine_train_loss_all_interactions(
        self,
        train_loss_default: Float[np.ndarray, "n_seed n_step"],
        train_loss_bem: Float[np.ndarray, "n_seed n_step"],
        train_loss_cmm: Float[np.ndarray, "n_seed n_step"],
        train_loss_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training losses for all binocular interactions.
        Args:
            train_loss_default (Float[np.ndarray, "n_seed n_step"]): train
                losses for default interaction.

            train_loss_bem (Float[np.ndarray, "n_seed n_step"]): train
                losses for BEM interaction.

            train_loss_cmm (Float[np.ndarray, "n_seed n_step"]): train
                losses for CMM interaction.

            train_loss_sum_diff (Float[np.ndarray, "n_seed n_step"]): train
                losses for sum_diff interaction.

            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        train_loss_default_avg = np.mean(train_loss_default, axis=0)
        train_loss_default_std = np.std(train_loss_default, axis=0)
        train_loss_bem_avg = np.mean(train_loss_bem, axis=0)
        train_loss_bem_std = np.std(train_loss_bem, axis=0)
        train_loss_cmm_avg = np.mean(train_loss_cmm, axis=0)
        train_loss_cmm_std = np.std(train_loss_cmm, axis=0)
        train_loss_sum_diff_avg = np.mean(train_loss_sum_diff, axis=0)
        train_loss_sum_diff_std = np.std(train_loss_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Training losses\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(self._steps(train_loss_default_avg), train_loss_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            self._steps(train_loss_default_avg),
            train_loss_default_avg - train_loss_default_std,
            train_loss_default_avg + train_loss_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(self._steps(train_loss_bem_avg), train_loss_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            self._steps(train_loss_bem_avg),
            train_loss_bem_avg - train_loss_bem_std,
            train_loss_bem_avg + train_loss_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(self._steps(train_loss_cmm_avg), train_loss_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            self._steps(train_loss_cmm_avg),
            train_loss_cmm_avg - train_loss_cmm_std,
            train_loss_cmm_avg + train_loss_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(
            self._steps(train_loss_sum_diff_avg), train_loss_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3]
        )
        axes.fill_between(
            self._steps(train_loss_sum_diff_avg),
            train_loss_sum_diff_avg - train_loss_sum_diff_std,
            train_loss_sum_diff_avg + train_loss_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel(self.config.loss)




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

        plt.legend(
            loc="upper right",
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
                f"{self.group_plot_dir}/plotLine_train_loss_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotLine_val_loss_all_interactions(
        self,
        val_loss_default: Float[np.ndarray, "n_seed n_step"],
        val_loss_bem: Float[np.ndarray, "n_seed n_step"],
        val_loss_cmm: Float[np.ndarray, "n_seed n_step"],
        val_loss_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation losses for all binocular interactions.
        Args:
            val_loss_default (Float[np.ndarray, "n_seed n_step"]): Validation losses for default interaction.
            val_loss_bem (Float[np.ndarray, "n_seed n_step"]): Validation losses for BEM interaction.
            val_loss_cmm (Float[np.ndarray, "n_seed n_step"]): Validation losses for CMM interaction.
            val_loss_sum_diff (Float[np.ndarray, "n_seed n_step"]): Validation losses for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        val_loss_default_avg = np.mean(val_loss_default, axis=0)
        val_loss_default_std = np.std(val_loss_default, axis=0)
        val_loss_bem_avg = np.mean(val_loss_bem, axis=0)
        val_loss_bem_std = np.std(val_loss_bem, axis=0)
        val_loss_cmm_avg = np.mean(val_loss_cmm, axis=0)
        val_loss_cmm_std = np.std(val_loss_cmm, axis=0)
        val_loss_sum_diff_avg = np.mean(val_loss_sum_diff, axis=0)
        val_loss_sum_diff_std = np.std(val_loss_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Validation losses\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(self._steps(val_loss_default_avg), val_loss_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            self._steps(val_loss_default_avg),
            val_loss_default_avg - val_loss_default_std,
            val_loss_default_avg + val_loss_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(self._steps(val_loss_bem_avg), val_loss_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            self._steps(val_loss_bem_avg),
            val_loss_bem_avg - val_loss_bem_std,
            val_loss_bem_avg + val_loss_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(self._steps(val_loss_cmm_avg), val_loss_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            self._steps(val_loss_cmm_avg),
            val_loss_cmm_avg - val_loss_cmm_std,
            val_loss_cmm_avg + val_loss_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(self._steps(val_loss_sum_diff_avg), val_loss_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            self._steps(val_loss_sum_diff_avg),
            val_loss_sum_diff_avg - val_loss_sum_diff_std,
            val_loss_sum_diff_avg + val_loss_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel(self.config.loss)




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

        plt.legend(
            loc="upper right",
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
                f"{self.group_plot_dir}/plotLine_val_loss_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotLine_acc_all_seeds(
        self,
        train_accs: Float[np.ndarray, "n_seed n_step"],
        val_accs: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training and validation accuracy.

        Args:
            train_accs (Float[np.ndarray, "n_seed n_step"]): Training accuracies for each seed.
            val_accs (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for each seed.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average accuracies
        train_acc_avg = np.mean(train_accs, axis=0)
        train_acc_std = np.std(train_accs, axis=0)
        val_acc_avg = np.mean(val_accs, axis=0)
        val_acc_std = np.std(val_accs, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Train and validation accuracies all seeds \n({self.config.binocular_interaction})",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#00CED1"]
        axes.plot(self._steps(train_acc_avg), train_acc_avg, linewidth=2, color=colors[0], label="Train")
        axes.fill_between(
            self._steps(train_acc_avg),
            train_acc_avg - train_acc_std,
            train_acc_avg + train_acc_std,
            color=colors[0],
            alpha=0.2,
        )
        axes.plot(self._steps(val_acc_avg), val_acc_avg, linewidth=2, color=colors[1], label="Val")
        axes.fill_between(
            self._steps(val_acc_avg),
            val_acc_avg - val_acc_std,
            val_acc_avg + val_acc_std,
            color=colors[1],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel("3-pix accuracy")




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

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
                f"{self.plot_dir}/plotLine_acc_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotLine_train_acc_all_interactions(
        self,
        train_acc_default: Float[np.ndarray, "n_seed n_step"],
        train_acc_bem: Float[np.ndarray, "n_seed n_step"],
        train_acc_cmm: Float[np.ndarray, "n_seed n_step"],
        train_acc_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation accuracies for all binocular interactions.
        Args:
            train_acc_default (Float[np.ndarray, "n_seed n_step"]): train accuracies for default interaction.
            train_acc_bem (Float[np.ndarray, "n_seed n_step"]): train accuracies for BEM interaction.
            train_acc_cmm (Float[np.ndarray, "n_seed n_step"]): train accuracies for CMM interaction.
            train_acc_sum_diff (Float[np.ndarray, "n_seed n_step"]): train accuracies for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        train_acc_default_avg = np.mean(train_acc_default, axis=0)
        train_acc_default_std = np.std(train_acc_default, axis=0)
        train_acc_bem_avg = np.mean(train_acc_bem, axis=0)
        train_acc_bem_std = np.std(train_acc_bem, axis=0)
        train_acc_cmm_avg = np.mean(train_acc_cmm, axis=0)
        train_acc_cmm_std = np.std(train_acc_cmm, axis=0)
        train_acc_sum_diff_avg = np.mean(train_acc_sum_diff, axis=0)
        train_acc_sum_diff_std = np.std(train_acc_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Train accuracies\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(self._steps(train_acc_default_avg), train_acc_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            self._steps(train_acc_default_avg),
            train_acc_default_avg - train_acc_default_std,
            train_acc_default_avg + train_acc_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(self._steps(train_acc_bem_avg), train_acc_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            self._steps(train_acc_bem_avg),
            train_acc_bem_avg - train_acc_bem_std,
            train_acc_bem_avg + train_acc_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(self._steps(train_acc_cmm_avg), train_acc_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            self._steps(train_acc_cmm_avg),
            train_acc_cmm_avg - train_acc_cmm_std,
            train_acc_cmm_avg + train_acc_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(self._steps(train_acc_sum_diff_avg), train_acc_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            self._steps(train_acc_sum_diff_avg),
            train_acc_sum_diff_avg - train_acc_sum_diff_std,
            train_acc_sum_diff_avg + train_acc_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel("3-pix accuracy")




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

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
                f"{self.group_plot_dir}/plotLine_train_acc_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plotLine_val_acc_all_interactions(
        self,
        val_acc_default: Float[np.ndarray, "n_seed n_step"],
        val_acc_bem: Float[np.ndarray, "n_seed n_step"],
        val_acc_cmm: Float[np.ndarray, "n_seed n_step"],
        val_acc_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation accuracies for all binocular interactions.
        Args:
            val_acc_default (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for default interaction.
            val_acc_bem (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for BEM interaction.
            val_acc_cmm (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for CMM interaction.
            val_acc_sum_diff (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        val_acc_default_avg = np.mean(val_acc_default, axis=0)
        val_acc_default_std = np.std(val_acc_default, axis=0)
        val_acc_bem_avg = np.mean(val_acc_bem, axis=0)
        val_acc_bem_std = np.std(val_acc_bem, axis=0)
        val_acc_cmm_avg = np.mean(val_acc_cmm, axis=0)
        val_acc_cmm_std = np.std(val_acc_cmm, axis=0)
        val_acc_sum_diff_avg = np.mean(val_acc_sum_diff, axis=0)
        val_acc_sum_diff_std = np.std(val_acc_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Validation accuracies\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(self._steps(val_acc_default_avg), val_acc_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            self._steps(val_acc_default_avg),
            val_acc_default_avg - val_acc_default_std,
            val_acc_default_avg + val_acc_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(self._steps(val_acc_bem_avg), val_acc_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            self._steps(val_acc_bem_avg),
            val_acc_bem_avg - val_acc_bem_std,
            val_acc_bem_avg + val_acc_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(self._steps(val_acc_cmm_avg), val_acc_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            self._steps(val_acc_cmm_avg),
            val_acc_cmm_avg - val_acc_cmm_std,
            val_acc_cmm_avg + val_acc_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(self._steps(val_acc_sum_diff_avg), val_acc_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            self._steps(val_acc_sum_diff_avg),
            val_acc_sum_diff_avg - val_acc_sum_diff_std,
            val_acc_sum_diff_avg + val_acc_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )


        axes.set_xlabel("Optimizer updates")
        axes.set_ylabel("3-pix accuracy")




        axes.autoscale(enable=True, axis="x")
        if "accuracy" in axes.get_ylabel():
            axes.set_ylim(0, 1)
        else:
            axes.autoscale(enable=True, axis="y")

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
                f"{self.group_plot_dir}/plotLine_val_acc_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)
