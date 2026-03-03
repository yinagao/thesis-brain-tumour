#!/bin/bash

#SBATCH -N 1 -c 4
#SBATCH --mem=4096
#SBATCH -t 1-0 ## this requests one day of walltime 
#SBATCH --gres gpu:1

module load python/3.11.3_torch_gpu

## GRID SEARCH FOR 3D BRAIN TUMOR CLASSIFIER

TRAIN_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/train"
VAL_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/val"
TEST_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/test"

BATCH_SIZES=(4)

LEARNING_RATES=(0.0003)

EPOCHS=30

BACKBONES=(
    "tf_efficientnet_b2.in1k"
    "densenet121"
)

RESULTS_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_frozen_results"  

echo "============================================"
echo "   STARTING GRID SEARCH EXPERIMENTS"
echo "============================================"

for BACKBONE in "${BACKBONES[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do

            echo "--------------------------------------------"
            echo "Running experiment:"
            echo " - Backbone:      $BACKBONE"
            echo " - Batch size:    $BATCH"
            echo " - Learning rate: $LR"
            echo " - Epochs:        $EPOCHS"
            echo "--------------------------------------------"

            python train_multiclass_frozen_experiments.py \
                --train_path "$TRAIN_DIR" \
                --val_path "$VAL_DIR" \
                --test_path "$TEST_DIR" \
                --batch_size $BATCH \
                --learning_rate $LR \
                --epochs $EPOCHS \
                --results_dir "$RESULTS_DIR" \
                --backbones "${BACKBONES[@]}"

        done
    done
done

echo "============================================"
echo " GRID SEARCH COMPLETE!"
echo "============================================"
