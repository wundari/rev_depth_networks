#!/usr/bin/env bash
set -euo pipefail

model_name="GCNet"
config_file="config_gcnet_lr"
bi_file="run_bino_integration_gcnet"
interactions=("default" "bem" "cmm" "sum_diff")
seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for seed in "${seeds[@]}"; do
    for interaction in "${interactions[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Binocular integration $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, bi_file=$bi_file"
        if ! python -m scripts.analysis.${bi_file}; then
            echo "FAILED: interaction=$interaction, seed=$seed" >&2
            failed_runs+=("${interaction}/${seed}")
            continue
        fi
    done
done


model_name="BNN"
config_file="config_bnn"
bi_file="run_bino_integration_bnn"
interactions=("default" "bem" "cmm" "sum_diff")
seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for seed in "${seeds[@]}"; do
    for interaction in "${interactions[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/${config_file}.py

        echo "Binocular integration $model_name: interaction=$interaction, seed=$seed, config_file=$config_file, bi_file=$bi_file"
        if ! python -m scripts.analysis.${bi_file}; then
            echo "FAILED: interaction=$interaction, seed=$seed" >&2
            failed_runs+=("${interaction}/${seed}")
            continue
        fi
    done
done

