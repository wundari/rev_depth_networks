# %%
from config.config_bnn import ConfigBNN
from GroupAnalysis.group_analysis_superposition import GA_Superposition

# %%
config = ConfigBNN()
ga_sup = GA_Superposition(config)

# %%
# interactions = ["default", "bem", "cmm", "sum_diff"]
conv_layer_names, _ = ga_sup.get_conv_names_and_weights()
save_flag = True
for interaction in ga_sup.interactions:

    # feature dimensionality analysis
    feat_dim_interaction = ga_sup.feature_dimensionality_interaction(interaction)

    # plot violin feature dimensionality for a given interaction
    ga_sup.plotViolin_feat_dim_interaction(
        feat_dim_interaction, conv_layer_names, save_flag
    )

# %% statistical test
ga_sup.stat_feat_dim()

# %%
