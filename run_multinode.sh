#!/bin/bash
#SBATCH --job-name=CS5242_proj
#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=CS5242_proj_%j.out
#SBATCH --error=CS5242_proj_%j.error

# Activate your environment
source ~/.bashrc
conda activate CS5242
cd ~/codes/markup2im

export NCCL_DEBUG=INFO
export GPUS_PER_NODE=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_BLOCKING_WAIT=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=enp59s
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --mixed_precision fp16
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
srun $LAUNCHER \
        src/train.py --dataset_name all --save_dir models/all_2 --batch_size 4 \
        --image_height 64 --image_width 320 \
        --color_mode grayscale


# srun python src/train.py --dataset_name all --save_dir models/all --batch_size 4