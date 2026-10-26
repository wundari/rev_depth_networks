#!/usr/bin/env bash
set -euo pipefail

model_name="BNN"
config_file="config_bnn"
train_file="train_bnn"
interactions=("sum_diff")
seeds=(27829 35744)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Training $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, train_file=$train_file"
        if ! python -m scripts.training.${train_file}; then
            echo "FAILED: interaction=$interaction, seed=$seed" >&2
            failed_runs+=("${interaction}/${seed}")
            continue
        fi
    done
done

# interactions=("cmm" "sum_diff")
# seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
# for interaction in "${interactions[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sed -i -E \
#             -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
#             -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
#             config/${config_file}.py

#         echo "Training $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, train_file=$train_file"
#         if ! python -m scripts.training.${train_file}; then
#             echo "FAILED: interaction=$interaction, seed=$seed" >&2
#             failed_runs+=("${interaction}/${seed}")
#             continue
#         fi
#     done
# done

# python -m scripts.analysis.run_ga_rds_bnn