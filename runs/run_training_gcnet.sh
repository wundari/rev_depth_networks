#!/usr/bin/env bash
set -euo pipefail

python -m scripts.analysis.run_rds_pgm

model_name="GC_Net_LR"
config_file="config_gcnet_lr"
train_file="train_gcnet_lr"
interactions=("cmm" "sum_diff")
seeds=(65190 82220 94750)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Training $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, train_file=$train_file"
        python -m scripts.training.${train_file} || break 2
    done
done

# run rds analysis after all training runs succeed
# python -m scripts.analysis.run_ga_rds_gcnet_lr
# python -m scripts.analysis.run_ga_layer_activation_analysis_gcnet_lr