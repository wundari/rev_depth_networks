# %% necessary modules

from GroupAnalysis.group_analysis_learning_curve import GA_LearningCurve
from config.config import GCNetconfig

# %% load config and create GA_LearningCurve object
config = GCNetconfig()
ga = GA_LearningCurve(config)

for interaction in ga.interactions:
    # interaction = "bem"
    ga.update_bino_interaction(interaction)
    train_losses = ga.get_train_loss()
    train_accs = ga.get_train_acc()
    val_losses = ga.get_val_loss()
    val_accs = ga.get_val_acc()

    # plot training and validation loss and accuracy, for each interaction
    save_flag = True
    ga.plotLine_loss_all_seeds(train_losses, val_losses, save_flag)
    ga.plotLine_acc_all_seeds(train_accs, val_accs, save_flag)

# %% statistical analysis
ga.compute_statistics_all()

# %% plot combined interactions
interaction = "default"
ga.update_bino_interaction(interaction)
train_loss_default = ga.get_train_loss()
train_acc_default = ga.get_train_acc()
val_loss_default = ga.get_val_loss()
val_acc_default = ga.get_val_acc()

interaction = "bem"
ga.update_bino_interaction(interaction)
train_loss_bem = ga.get_train_loss()
train_acc_bem = ga.get_train_acc()
val_loss_bem = ga.get_val_loss()
val_acc_bem = ga.get_val_acc()

interaction = "cmm"
ga.update_bino_interaction(interaction)
train_loss_cmm = ga.get_train_loss()
train_acc_cmm = ga.get_train_acc()
val_loss_cmm = ga.get_val_loss()
val_acc_cmm = ga.get_val_acc()

interaction = "sum_diff"
ga.update_bino_interaction(interaction)
train_loss_sum_diff = ga.get_train_loss()
train_acc_sum_diff = ga.get_train_acc()
val_loss_sum_diff = ga.get_val_loss()
val_acc_sum_diff = ga.get_val_acc()

ga.plotLine_train_loss_all_interactions(
    train_loss_default, train_loss_bem, train_loss_cmm, train_loss_sum_diff, save_flag
)

ga.plotLine_val_loss_all_interactions(
    val_loss_default, val_loss_bem, val_loss_cmm, val_loss_sum_diff, save_flag
)

ga.plotLine_train_acc_all_interactions(
    train_acc_default, train_acc_bem, train_acc_cmm, train_acc_sum_diff, save_flag
)

ga.plotLine_val_acc_all_interactions(
    val_acc_default, val_acc_bem, val_acc_cmm, val_acc_sum_diff, save_flag
)

# %%
