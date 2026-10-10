# %%
from config.config_gcnet_lr import ConfigGCNet
from RDS_analysis.rds_pgm import PGM


# %%
def main():
    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    # config.n_rds_each_disp = 128
    pgm = PGM(config)

    # create rds_bank
    # RDS bank structure: [dotMatch dotDens disp_magnitude n_rds_each_disp]
    rds_bank = pgm.create_rds_bank(pgm.background_flag, pgm.pedestal_flag)
    # pred_disp, pred_disp_labels = analysis.compute_disp_map_rds(rds_bank)
    # pred_disp: [dotMatch, dotDens, 2*n_rds_each_disp, h, w]
    # pred_disp_labels = [dotMatch, dotDens, 2*n_rds_each_disp]

    interactions = pgm.config.interactions
    for interaction in interactions:
        for s, seed in enumerate(pgm.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = pgm.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = pgm.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = pgm.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = pgm.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            pgm.update_network_config(interaction, seed, epoch, iter)

            # update model
            pgm._load_pretrained_model()
            pgm.model.to(pgm.device)

            ref_stats = pgm.compute_ref_stats(rds_bank)

            # plot variance
            pgm.plot_variance(ref_stats)

            # compute monocular weights
            w_near, w_far = pgm.compute_monocular_weight(ref_stats)
            pgm.plot_monocular_weight(w_near, w_far)

            # compute expected disparity
            E_near, E_far = pgm.compute_expected_disp(ref_stats, w_near, w_far)
            pgm.plot_expected_disp(E_near, E_far)


# %%
if __name__ == "__main__":
    main()
