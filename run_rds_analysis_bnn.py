# %%
# script for in-silico analysis using RDS for BNN
# working dir: rev_depth_networks


# %%
def main():
    from config.config_bnn import ConfigBNN
    from RDS_analysis.rds_analysis import RDSAnalysis

    # set up BNN model and RDS analysis
    config = ConfigBNN()
    rdsa = RDSAnalysis(config)

    # compute model responses to RDSs
    rdsa.compute_disp_map_rds_group(
        rdsa.dotDens_list, rdsa.background_flag, rdsa.pedestal_flag
    )
    # cross-decoding analysis with SVM
    rdsa.xDecode(rdsa.dotDens_list, rdsa.n_bootstrap, rdsa.background_flag)

    # plot cross-decoding performance
    # plot performance at a target dot density
    save_flag = 1
    dotDens = 0.3
    rdsa.plotLine_xDecode_at_dotDens(dotDens, save_flag)

    # plot performance as a function of dot density
    rdsa.plotLine_xDecode(save_flag)

    # plot disparity map
    rdsa.plotHeat_dispMap(save_flag)
    rdsa.plotHeat_dispMap_avg(save_flag)


if __name__ == "__main__":
    main()
