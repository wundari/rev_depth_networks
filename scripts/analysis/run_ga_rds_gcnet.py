def main():

    # load necessary modules
    from config.config_gcnet import ConfigGCNet
    from GroupAnalysis.group_analysis_rds import GA_RDS

    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    ga_rds = GA_RDS(config)

    # generate RDS dataloader
    rds_bank = ga_rds.create_rds_bank(
        ga_rds.background_flag,
        ga_rds.pedestal_flag,
    )

    # compute disparity map
    interactions = ga_rds.config.interactions
    for interaction in interactions:
        if interaction == "bem":
            ga_rds.batch_size_rds = (
                8  # reduce batch size for bem as it consumes more GPU
            )
        ga_rds.compute_disp_map_all_seeds(
            rds_bank, interaction, ga_rds.config.n_bootstrap
        )

    # plot depth performance averaged across all seeds
    save_flag = True
    for interaction in interactions:
        ga_rds.plotLine_xDecode_all_seeds(interaction, save_flag)

    # plot depth performance for every seeds
    for interaction in interactions:
        for s, seed in enumerate(ga_rds.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = ga_rds.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = ga_rds.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = ga_rds.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = ga_rds.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            ga_rds.update_network_config(interaction, seed, epoch, iter)

            # plot cross-decoding performance
            ga_rds.plotLine_xDecode(save_flag)

            # plot predicted disparity map for a single bootstrap
            ga_rds.plotHeat_dispMap(save_flag)

            # plot predicted disparity map averaged across all bootstrap
            ga_rds.plotHeat_dispMap_avg(save_flag)


# %%
if __name__ == "__main__":
    main()
