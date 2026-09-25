def main():

    # load necessary modules
    from config.config_gcnet import ConfigGCNet
    from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct

    # set up GCNet model and RDS analysis
    config = ConfigGCNet()
    rdsl = RDS_LayerAct(config)

    # generate RDS dataloader
    rds_bank = rdsl.create_rds_bank(
        rdsl.background_flag,
        rdsl.pedestal_flag,
    )

    interactions = rdsl.config.interactions
    # interactions = ["cmm", "default", "sum_diff"]
    save_flag = True

    for interaction in interactions:
        for s, seed in enumerate(rdsl.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = rdsl.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = rdsl.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = rdsl.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = rdsl.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            rdsl.update_network_config(interaction, seed, epoch, iter)

            # update model
            rdsl._load_pretrained_model()
            rdsl.model.to(rdsl.device)

            # compute layer activation for all dot density
            rdsl.compute_layer_act_rds(
                rds_bank, rdsl.background_flag, rdsl.pedestal_flag
            )

            # Cosine-similarity
            for dotDens in rdsl.dotDens_list:
                result = rdsl.compute_cosine_similarity(
                    dotDens=dotDens,
                    split_train=0.8,
                    n_bootstrap=rdsl.n_bootstrap,
                )

            # plot cosine similarity
            rdsl.plot_cosine_similarity(save_flag)

        # plot cosine similarity averaged across all seeds
        rdsl.plot_cosine_similarity_all_seeds(interaction, save_flag)


# %%
if __name__ == "__main__":
    main()
