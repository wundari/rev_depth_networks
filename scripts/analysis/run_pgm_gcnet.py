# %%
from config.config_gcnet_lr import ConfigGCNet
from PGM.pgm import PGM

from pathlib import Path
import numpy as np


# %%
def main():
    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    config.load_state = True
    config.n_rds_each_disp = 256
    pgm = PGM(config)

    # create rds_bank
    # RDS bank structure: [dotMatch dotDens disp_magnitude n_rds_each_disp]
    # rds_bank = pgm.create_rds_bank(pgm.background_flag, pgm.pedestal_flag)

    # pred_disp, pred_disp_labels = analysis.compute_disp_map_rds(rds_bank)
    # pred_disp: [dotMatch, dotDens, 2*n_rds_each_disp, h, w]
    # pred_disp_labels = [dotMatch, dotDens, 2*n_rds_each_disp]

    interactions = pgm.config.interactions
    # interactions = ["default"]
    for interaction in interactions:
        for s, seed in enumerate(pgm.config.seed_to_analyse):

            # interaction = "sum_diff"
            # s = 0
            # seed = pgm.config.seed_to_analyse[s]
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

            # plot predicted disparity variance
            # pgm.plot_pred_disp_variance(ref_stats)

            # compute reference weights
            # w_ref = pgm.compute_reference_weight(ref_stats)
            # pgm.plot_reference_weight(w_ref)
            w_ref = np.load(f"{path}/reference_weight.npy")

            # compute expected disparity
            E_near, E_far = pgm.compute_expected_disp(ref_stats, w_ref)
            pgm.plot_expected_disp(E_near, E_far)

            # simulate 2-AFC
            p_2afc, p_2afc_std = pgm.simulate_2afc(ref_stats, w_ref)
            pgm.plotLine_2afc_performance(p_2afc, p_2afc_std)

        # # plot average expected_disp across seeds
        # pgm.plot_expected_disp_seeds()

        # # plot average reference weight across seeds
        # pgm.plot_reference_weights_seeds()

        # plot average p_2afc across seeds
        pgm.plotLine_2afc_performance_seeds()


if __name__ == "__main__":
    main()
