# %% load necessary modules
import numpy as np
from config.config_gcnet import ConfigGCNet
from RDS_analysis.rds_layer_activation_analysis import RDS_LayerAct
from RDS_analysis.rds_temporal_integration import (
    run_temporal_rds_condition,
)
import matplotlib.pyplot as plt

# %%
# set up GCNet model and RDS analysis
config = ConfigGCNet()
analysis = RDS_LayerAct(config)

T = 16

result_ards = run_temporal_rds_condition(
    analysis=analysis,
    dot_match=0.0,  # aRDS
    dot_density=0.5,
    T=T,
    dynamic=True,
    reference_mode="sample",
    p_left=0.5,
    sequence_batch_size=2,
    # roi=center_roi,
)

result_hmrds = run_temporal_rds_condition(
    analysis=analysis,
    dot_match=0.5,  # hmRDS
    dot_density=0.5,
    T=T,
    dynamic=True,
    reference_mode="sample",
    p_left=0.5,
    sequence_batch_size=1,
    # roi=center_roi,
)

result_crds = run_temporal_rds_condition(
    analysis=analysis,
    dot_match=1.0,  # cRDS
    dot_density=0.5,
    T=T,
    dynamic=True,
    reference_mode="sample",
    p_left=0.5,
    sequence_batch_size=1,
    # roi=center_roi,
)

static_ards = run_temporal_rds_condition(
    analysis,
    dot_match=0.0,
    dot_density=0.5,
    T=T,
    dynamic=False,
    reference_mode="marginalize",
    # roi=center_roi,
)


# %%
def plot_disp(results):
    labels = results["label"]
    near = labels > 0
    far = labels < 0

    near_curve = results["temporal_mean_curve"][near].mean(axis=0)
    far_curve = results["temporal_mean_curve"][far].mean(axis=0)

    frame_rate = 30.0
    time_ms = np.arange(1, T + 1) / frame_rate * 1000
    plt.figure()
    plt.plot(time_ms, near_curve, label="near")
    plt.plot(time_ms, far_curve, label="far")

    plt.axhline(0, linestyle="--")

    plt.xlabel("Integration time (ms)")
    plt.ylabel("Expected disparity (px)")

    plt.legend()
    plt.show()


plot_disp(result_ards)
# plot_disp(static_ards)
# %%
plot_disp(result_hmrds)
# %%
plot_disp(result_crds)
# %%
