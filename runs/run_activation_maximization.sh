#!/usr/bin/env bash
set -euo pipefail

model_name="GCNet"
config_file="config_gcnet_lr"
am_file="run_activation_maximization"
interactions=("default" "cmm" "sum_diff" "bem")
seeds=(1618)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Activation maximization $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, am_file=$am_file"
        if ! python -m scripts.analysis.${am_file}; then
            echo "FAILED: interaction=$interaction, seed=$seed" >&2
            failed_runs+=("${interaction}/${seed}")
            continue
        fi
    done
done