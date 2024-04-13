#!/bin/bash
#SBATCH --job-name=CS5242_proj
#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:titianrtx:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=CS5242_proj_%j.out
#SBATCH --error=CS5242_proj_%j.error

# Activate your environment
source ~/.bashrc
conda activate CS5242
cd ~/codes/markup2im

export GPUS_PER_NODE=1
######################

# Used for a100mig devices
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep "MIG" | awk '{gsub(/\)/,""); print $6}' | paste -sd "," -)

export LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_processes $GPUS_PER_NODE \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
srun $LAUNCHER \
        src/train.py --dataset_name all --save_dir models/all_2 --batch_size 4 \
        --image_height 64 --image_width 320 \
        --color_mode grayscale


# srun python src/train.py --dataset_name all --save_dir models/all --batch_size 4