# interactions=("default")
# seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
# for interaction in "${interactions[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sed -i -E \
#             -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
#             -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
#             config/config_bnn.py

#         echo "Training BNN: interaction=$interaction, seed=$seed"
#         python -m scripts.training.train_bnn || break 2
#     done
# done

# interactions=("sum_diff")
# seeds=(16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
# for interaction in "${interactions[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sed -i -E \
#             -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
#             -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
#             config/config_bnn.py

#         echo "Training BNN: interaction=$interaction, seed=$seed"
#         python -m scripts.training.train_bnn || break 2
#     done
# done

# interactions=("default")
# seeds=(27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
# for interaction in "${interactions[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sed -i -E \
#             -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
#             -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
#             config/config_gcnet.py

#         echo "Training GCNet: interaction=$interaction, seed=$seed"
#         python -m scripts.training.train_gcnet || break 2
#     done
# done

# interactions=("cmm" "sum_diff")
# seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
# for interaction in "${interactions[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sed -i -E \
#             -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
#             -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
#             config/config_gcnet.py

#         echo "Training GCNet: interaction=$interaction, seed=$seed"
#         python -m scripts.training.train_gcnet || break 2
#     done
# done

#!/usr/bin/env bash
set -euo pipefail

model_name="GC_Net_LR"
config_file="config_gcnet_lr"
train_file="train_gcnet_lr"
interactions=("bem" "default")
seeds=(1618 11364 16476 27829 35154)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Training $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, train_file=$train_file"
        python -m scripts.training.${train_file}
    done
done

# run rds analysis after all training runs succeed
python -m scripts.analysis.run_ga_rds_gcnet_lr