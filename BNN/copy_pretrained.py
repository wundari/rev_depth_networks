# %%
import shutil

seed_to_analyse = [
    1618,
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

epoch_iter_to_load_default = [
    (9, 19600),  # seed 1618
    (5, 12100),  # seed 11364
    (8, 18100),  # seed 16476
    (7, 16500),  # seed 27829
    (7, 17300),  # seed 35154
    (5, 11400),  # seed 35744
    (5, 12400),  # seed 36675
    (6, 14300),  # seed 43798
    (6, 14200),  # seed 55826
    (5, 12700),  # seed 59035
    (7, 16000),  # seed 65190
    (6, 13600),  # seed 82220
    (8, 19000),  # seed 94750
]


# bino_interaction: bem
epoch_iter_to_load_bem = [
    (8, 18100),  # seed 1618
    (5, 12100),  # seed 11364
    (8, 17500),  # seed 16476
    (9, 21400),  # seed 27829
    (7, 17300),  # seed 35154
    (5, 11400),  # seed 35744
    (4, 9900),  # seed 36675
    (6, 14300),  # seed 43798
    (9, 21200),  # seed 55826
    (5, 12700),  # seed 59035
    (7, 16000),  # seed 65190
    (6, 13600),  # seed 82220
    (8, 19000),  # seed 94750
]

# bino_interaction: cmm
epoch_iter_to_load_cmm = [
    (9, 19600),  # seed 1618
    (5, 12100),  # seed 11364
    (8, 18100),  # seed 16476
    (5, 12400),  # seed 27829
    (7, 17300),  # seed 35154
    (5, 11400),  # seed 35744
    (7, 16000),  # seed 36675
    (6, 14300),  # seed 43798
    (6, 14200),  # seed 55826
    (5, 12700),  # seed 59035
    (7, 16000),  # seed 65190
    (6, 13600),  # seed 82220
    (8, 19000),  # seed 94750
]


# bino_interaction: sum_diff
epoch_iter_to_load_sum_diff = [
    (9, 19600),  # seed 1618
    (5, 12100),  # seed 11364
    (8, 18100),  # seed 16476
    (5, 12400),  # seed 27829
    (7, 17300),  # seed 35154
    (5, 11400),  # seed 35744
    (5, 12400),  # seed 36675
    (6, 14300),  # seed 43798
    (6, 14200),  # seed 55826
    (5, 12700),  # seed 59035
    (7, 16000),  # seed 65190
    (6, 13600),  # seed 82220
    (8, 19000),  # seed 94750
]

interactions = ["default", "bem", "cmm", "sum_diff"]
source_dir = "/home/wundari/BNN/run/sceneflow_monkaa"
target_dir = "/home/wundari/rev_depth_networks/BNN/run/sceneflow_monkaa"
for interaction in interactions:
    if interaction == "default":
        epoch_iter_to_load = epoch_iter_to_load_default
    elif interaction == "bem":
        epoch_iter_to_load = epoch_iter_to_load_bem
    elif interaction == "cmm":
        epoch_iter_to_load = epoch_iter_to_load_cmm
    elif interaction == "sum_diff":
        epoch_iter_to_load = epoch_iter_to_load_sum_diff
    else:
        raise ValueError(f"Unknown interaction: {interaction}")
    for s, seed in enumerate(seed_to_analyse):
        epoch = epoch_iter_to_load[s][0]
        iter = epoch_iter_to_load[s][1]
        pretrained_file = f"{source_dir}/bino_interaction_{interaction}/{seed}/epoch_{epoch}_iter_{iter}_model_best.pth.tar"
        target_folder = f"{target_dir}/bino_interaction_{interaction}/{seed}"

        shutil.copy(pretrained_file, target_folder)

# %%
