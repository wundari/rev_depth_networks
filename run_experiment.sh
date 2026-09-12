interactions=("default")
seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/config_bnn.py

        echo "Training BNN: interaction=$interaction, seed=$seed"
        python train_bnn.py || break 2
    done
done

interactions=("sum_diff")
seeds=(16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/config_bnn.py

        echo "Training BNN: interaction=$interaction, seed=$seed"
        python train_bnn.py || break 2
    done
done

interactions=("default")
seeds=(27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/config_gcnet.py

        echo "Training GCNet: interaction=$interaction, seed=$seed"
        python train_gcnet.py || break 2
    done
done

interactions=("bem" "cmm" "sum_diff")
seeds=(1618 11364 16476 27829 35154 35744 36675 43798 55826 59035 65190 82220 94750)
for interaction in "${interactions[@]}"; do
    for seed in "${seeds[@]}"; do
        sed -i -E \
            -e "s/^([[:space:]]*binocular_interaction: str = ).*/\1\"${interaction}\"/" \
            -e "s/^([[:space:]]*seed: int = )[0-9]+/\1${seed}/" \
            config/config_gcnet.py

        echo "Training GCNet: interaction=$interaction, seed=$seed"
        python train_gcnet.py || break 2
    done
done