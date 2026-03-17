#!/bin/bash

#SBATCH -N 1 -c 4
#SBATCH --mem=4096
#SBATCH -t 1-0 ## this requests one day of walltime 
#SBATCH --gres gpu:1

module load python/3.11.3_torch_gpu
module load libffi/3.2.1

## GRID SEARCH FOR 3D BRAIN TUMOR CLASSIFIER

TRAIN_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/train"
VAL_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/val"
TEST_DIR="/hpf/largeprojects/fkhalvati/Yina/multiclass_splits/test"

BATCH_SIZES=(4)

LEARNING_RATES=(0.0001)

EPOCHS=30

POSITIVE_CLASS=1 # dipg, mb, plgg

BACKBONES=(
    "resnet18.a1_in1k"
    "resnext50_32x4d.fb_swsl_ig1b_ft_in1k"
    "tf_efficientnet_b2.in1k"
    "convnext_tiny"
    "densenet121"
)

RESULTS_DIR="/hpf/largeprojects/fkhalvati/Yina/onevsall_weighted/medulloblastoma"  

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
            echo " - Positive class: $POSITIVE_CLASS"
            echo "--------------------------------------------"

            python train_binary_one_vs_all_weighted.py \
                --train_path "$TRAIN_DIR" \
                --val_path "$VAL_DIR" \
                --test_path "$TEST_DIR" \
                --positive_class "$POSITIVE_CLASS" \
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
