#!/bin/bash
#SBATCH --job-name=libero_lora_single_gpu
#SBATCH --output=/coc/flash8/dhe83/openpi/logs/libero_lora_single_gpu.out
#SBATCH --error=/coc/flash8/dhe83/openpi/logs/libero_lora_single_gpu.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --mem-per-gpu=128
#SBATCH --requeue

source ~/.zshrc
source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 # train with 90% of GPU memory

uv run scripts/train.py \
    pi05_libero_lora \
    --exp-name=libero_lora_finetune_single_gpu \
    --overwrite