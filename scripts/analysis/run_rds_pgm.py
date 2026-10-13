# %%
from config.config_gcnet_lr import ConfigGCNet
from RDS_analysis.rds_pgm import PGM

from pathlib import Path
import numpy as np


# %%
def main():
    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    config.n_rds_each_disp = 256
    pgm = PGM(config)

    # create rds_bank
    # RDS bank structure: [dotMatch dotDens disp_magnitude n_rds_each_disp]
    # rds_bank = pgm.create_rds_bank(pgm.background_flag, pgm.pedestal_flag)
    # pred_disp, pred_disp_labels = analysis.compute_disp_map_rds(rds_bank)
    # pred_disp: [dotMatch, dotDens, 2*n_rds_each_disp, h, w]
    # pred_disp_labels = [dotMatch, dotDens, 2*n_rds_each_disp]

    interactions = pgm.config.interactions
    # interactions = ["cmm", "sum_diff"]
    for interaction in interactions:
        for s, seed in enumerate(pgm.config.seed_to_analyse):

            if interaction == "default":
                epoch2load, iter2load = pgm.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch2load, iter2load = pgm.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch2load, iter2load = pgm.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch2load, iter2load = pgm.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            pgm.update_network_config(interaction, seed, epoch2load, iter2load)

            # update model
            pgm._load_pretrained_model()
            pgm.model.to(pgm.device)

            # compute activation stats for each eye reference
            # ref_stats = pgm.compute_ref_stats(rds_bank)
            path = Path(pgm.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
            ref_stats = np.load(f"{path}/ref_stats.npy", allow_pickle=True).item()

            # plot variance
            pgm.plot_variance(ref_stats)

            # compute reference weights
            # w_near, w_far = pgm.compute_monocular_weight(ref_stats)
            # pgm.plot_monocular_weight(w_near, w_far)
            w_ref = pgm.compute_reference_weight(ref_stats)
            pgm.plot_reference_weight(w_ref)

            # compute expected disparity
            # E_near, E_far = pgm.compute_expected_disp(ref_stats, w_near, w_far)
            E_near, E_far = pgm.compute_expected_disp(ref_stats, w_ref)
            pgm.plot_expected_disp(E_near, E_far)

        # plot average across seeds
        pgm.plot_expected_disp_seeds()


if __name__ == "__main__":
    main()

# %%
# from config.config_gcnet_lr import ConfigGCNet
# from RDS_analysis.rds_pgm import PGM
# from pathlib import Path
# import numpy as np

# # set up GCNet model and RDS analysis
# config = ConfigGCNet()
# config.n_rds_each_disp = 256
# pgm = PGM(config)

# # load ref_stats
# seed = 1618
# iter2load = 16500
# path = Path(pgm.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
# ref_stats = np.load(f"{path}/ref_stats.npy", allow_pickle=True).item()

# # load reference weights
# path = Path(pgm.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
# w_ref = np.load(f"{path}/reference_weight.npy")  # [2, dotMatch, dotDens]

# # compute expected disparity
# n = pgm.config.n_rds_each_disp
# # ensure that disparity labels are non-shuffled
# assert np.all(ref_stats["labels"][..., :n] > 0)
# assert np.all(ref_stats["labels"][..., n:] < 0)
# roi_mean = ref_stats["roi_mean"]  # [reference, dotMatch, dotDens, trial]

# # average across trials (across n_rds_each_disp)
# # [reference, dotMatch, dotDens]
# mean_far = roi_mean[..., :n].mean(axis=-1)
# mean_near = roi_mean[..., n:].mean(axis=-1)

# pgm.plot_mean_disp_unweighted(ref_stats, save_flag=False)

# # %% compute expected disparity, [dotMatch, dotDens]
# # rho = np.repeat([[-1.0], [0.0], [1.0]], 9, axis=1)
# # rho = np.repeat([[0.0], [0.5], [1.0]], 9, axis=1)
# # E_near = w_ref[0] * mean_near[0] + w_ref[1] * mean_far[1]
# # E_far = w_ref[0] * mean_far[0] + w_ref[1] * mean_near[1]
# E_near = w_ref[0] * mean_near[0] + w_ref[1] * mean_near[1]
# E_far = w_ref[0] * mean_far[0] + w_ref[1] * mean_far[1]

# pgm.plot_expected_disp(E_near, E_far, save_flag=False)

# # %%
# w_left_near_seeds = np.empty(
#     (
#         len(pgm.config.seed_to_analyse),
#         len(pgm.config.dotMatch_list),
#         len(pgm.config.dotDens_list),
#     ),
#     dtype=np.float32,
# )
# w_left_far_seeds = np.empty(
#     (
#         len(pgm.config.seed_to_analyse),
#         len(pgm.config.dotMatch_list),
#         len(pgm.config.dotDens_list),
#     ),
#     dtype=np.float32,
# )

# # load expected_disp_near and far all seeds
# for s, seed in enumerate(pgm.config.seed_to_analyse):

#     # interaction = "sum_diff"
#     # s = 0
#     # seed = pgm.config.seed_to_analyse[s]
#     if pgm.binocular_interaction == "default":
#         epoch2load, iter2load = pgm.config.epoch_iter_to_load_default[s]
#     elif pgm.binocular_interaction == "bem":
#         epoch2load, iter2load = pgm.config.epoch_iter_to_load_bem[s]
#     elif pgm.binocular_interaction == "cmm":
#         epoch2load, iter2load = pgm.config.epoch_iter_to_load_cmm[s]
#     else:  # sum_diff
#         epoch2load, iter2load = pgm.config.epoch_iter_to_load_sum_diff[s]

#     path = Path(pgm.save_dir, f"experiment_{seed}", f"pgm_iter_{iter2load}")
#     w_near = np.load(f"{path}/monocular_weight_near.npy")  # [2, dotMatch, dotDens]
#     w_far = np.load(f"{path}/monocular_weight_far.npy")

#     # store to buffer
#     w_left_near_seeds[s] = E_near
#     w_left_far_seeds[s] = E_far

# # average across seeds, [dotMatch, dotDens]
# E_near_mean = w_left_near_seeds.mean(axis=0)
# E_far_mean = w_left_far_seeds.mean(axis=0)
# E_near_sem = sem(w_left_near_seeds, axis=0)
# E_far_sem = sem(w_left_far_seeds, axis=0)

# # start plotting
# stim_labels = ["aRDS", "hmRDS", "cRDS"]
# markersize = 8

# sns.set_theme(context="paper", style="ticks", font_scale=2, palette="deep")

# figsize = (9, 6)
# n_row = 1
# n_col = 1

# fig, axes = plt.subplots(
#     nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
# )

# fig.text(
#     0.5,
#     1.0,
#     f"Expected disparity (interaction: {pgm.binocular_interaction})",
#     ha="center",
#     fontweight="semibold",
# )
# fig.text(-0.05, 0.5, "Expected disp [px]", va="center", rotation=90)
# fig.text(0.5, -0.04, "Dot density", ha="center")

# fig.tight_layout()
# plt.subplots_adjust(wspace=0.2, hspace=0.3)

# ## plot E_near
# colors = ["#6a5acd", "#00CED1", "#333333"]
# x = np.array(pgm.dotDens_list)
# for i in range(len(pgm.config.dotMatch_list)):
#     y = E_near_mean[i]
#     y_err = E_near_sem[i]

#     axes.plot(
#         x,
#         y,
#         lw=2,
#         c=colors[i],
#         ls="-",
#     )

#     # plot the marker
#     axes.plot(x, y, "o", markersize=markersize, c=colors[i])

#     axes.fill_between(
#         x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
#     )

# ## plot E_far
# for i in range(len(pgm.config.dotMatch_list)):
#     y = E_far_mean[i]
#     y_err = E_far_sem[i]

#     axes.plot(
#         x,
#         y,
#         lw=2,
#         c=colors[i],
#         ls="--",
#     )

#     # plot the marker
#     axes.plot(x, y, "o", markersize=markersize, c=colors[i])

#     axes.fill_between(
#         x, y - y_err, y + y_err, color=colors[i], alpha=0.2, lw=0, zorder=2
#     )

# # plot zero disparity
# axes.hlines(y=0.0, xmin=0.0, xmax=1.0, colors="red", linestyles="--", linewidth=2)

# x_low = 0.0
# x_up = 1.05
# x_step = 0.1
# y_low = -15.0
# y_up = 15.5
# y_step = 5

# axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
# axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
# axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
# axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

# axes.set_xlim(x_low, x_up)
# axes.set_ylim(y_low, y_up)

# ## legend
# # one legend explaining colour (stimulus) and one explaining line style
# stim_handles = [
#     Line2D(
#         [0],
#         [0],
#         color=c,
#         lw=3,
#         marker="o",
#         markersize=markersize,
#         markeredgecolor="white",
#         markeredgewidth=1.5,
#     )
#     for c in colors
# ]
# style_handles = [
#     Line2D([0], [0], color="gray", lw=2.5, ls="-"),
#     Line2D([0], [0], color="gray", lw=2.5, ls="--"),
# ]

# leg1 = axes.legend(
#     stim_handles,
#     stim_labels,
#     title="Stimulus",
#     frameon=False,
#     loc="upper left",
#     bbox_to_anchor=(1.02, 1.0),
#     alignment="left",
# )
# axes.add_artist(leg1)
# axes.legend(
#     style_handles,
#     ["Near", "Far"],
#     title="Disparity",
#     frameon=False,
#     loc="upper left",
#     bbox_to_anchor=(1.02, 0.55),
#     alignment="left",
# )

# # Hide the right and top spines
# axes.spines["right"].set_visible(False)
# axes.spines["top"].set_visible(False)

# # Only show ticks on the left and bottom spines
# axes.yaxis.set_ticks_position("left")
# axes.xaxis.set_ticks_position("bottom")
