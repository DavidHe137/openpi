#!/bin/bash
#SBATCH --job-name=finetune_25_act_libero
#SBATCH --output=scripts/finetune_25_act_libero.out
#SBATCH --error=scripts/finetune_25_act_libero.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:4"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64

export PYTHONUNBUFFERED=TRUE
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

source ~/.bashrc

nvidia-smi

srun -u XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_act_horizon_25 --exp-name=pi05_libero_act_horizon_25 --overwrite
