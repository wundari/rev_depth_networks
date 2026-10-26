def main():

    # load necessary modules
    from config.config_bnn import ConfigBNN
    from GroupAnalysis.group_analysis_rds import GA_RDS

    # set up BNN model and RDS analysis
    config = ConfigBNN()
    config.load_state = True
    ga_rds = GA_RDS(config)

    # generate RDS dataloader
    rds_bank = ga_rds.create_rds_bank(
        ga_rds.background_flag,
        ga_rds.pedestal_flag,
    )

    # compute disparity map
    # interactions = ga_rds.config.interactions
    interactions = ["default", "bem"]
    for interaction in interactions:
        if interaction == "default":
            ga_rds.config.seed_to_analyse = [
                1618,
                # 11364,
                16476,
                27829,
                # 35154,
                35744,
                36675,
                43798,
                55826,
                59035,
                65190,
                82220,
                # 94750,
            ]
            ga_rds.config.epoch_iter_to_load_default = [
                (9, 17500),  # seed 1618
                # (9, 17500),  # seed 11364
                (9, 17500),  # seed 16476
                (9, 17500),  # seed 27829
                # (9, 17000),  # seed 35154
                (9, 17500),  # seed 35744
                (9, 17589),  # seed 36675
                (9, 17000),  # seed 43798
                (8, 15500),  # seed 55826
                (9, 17500),  # seed 59035
                (9, 17500),  # seed 65190
                (9, 17589),  # seed 82220
                # (9, 17000),  # seed 94750]
            ]
        elif interaction == "bem":
            ga_rds.config.seed_to_analyse = [
                # 1618,
                11364,
                16476,
                27829,
                35154,
                35744,
                36675,
                43798,
                55826,
                59035,
                65190,
                82220,
                94750,
            ]
            ga_rds.config.epoch_iter_to_load_bem = [
                # (9, 16000),  # seed 1618
                (9, 17000),  # seed 11364
                (9, 17500),  # seed 16476
                (7, 14000),  # seed 27829
                (9, 17589),  # seed 35154
                (9, 17500),  # seed 35744
                (8, 15000),  # seed 36675
                (9, 17500),  # seed 43798
                (7, 13500),  # seed 55826
                (9, 17000),  # seed 59035
                (9, 17500),  # seed 65190
                (9, 16000),  # seed 82220
                (8, 14500),  # seed 94750
            ]
        ga_rds.compute_disp_map_all_seeds(
            rds_bank, interaction, ga_rds.config.n_bootstrap
        )

    # plot depth performance averaged across all seeds
    save_flag = True
    # for interaction in interactions:
    #     ga_rds.plotLine_xDecode_all_seeds(interaction, save_flag)

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
