#!/bin/bash

export GPUS_PER_NODE=4
######################

export OUTPUT_PATH="output.log"
export LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_processes $GPUS_PER_NODE \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
nohup $LAUNCHER \
        src/train.py --dataset_name all --save_dir models/all_2 --batch_size 4 \
        --image_height 64 --image_width 320 \
        --color_mode grayscale \
        --lora \
    > $OUTPUT_PATH 2>&1 &


# srun python src/train.py --dataset_name all --save_dir models/all --batch_size 4