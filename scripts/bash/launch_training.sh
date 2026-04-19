#!/bin/bash
#SBATCH --job-name=throw_legos_exp
#SBATCH --output=/coc/flash7/rbansal66/vvla/openpi-training/scripts/bash/log/output.out
#SBATCH --error=/coc/flash7/rbansal66/vvla/openpi-training/scripts/bash/log/error.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --exclusive
#SBATCH --exclude="samantha"
#SBATCH --mem-per-gpu=48G

cd /coc/flash7/rbansal66/vvla/openpi-training

source ~/.bashrc

nvidia-smi
hostname

# uv run scripts/compute_norm_stats.py --config-name pi05_throw_the_legos
# OOM: lower mem fraction slightly for BFC headroom, or pass e.g. --batch-size 4 (must divide #GPUs).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Default fsdp_devices=2 on 8 GPUs -> (4,2) mesh; XLA then does involuntary full rematerialization (~32GiB/GPU).
# Set --fsdp-devices to match GPU count on this node (must divide jax.device_count(); batch_size must too).
uv run scripts/train.py pi05_throw_the_legos --exp-name=throw_legos_exp --fsdp-devices 8 --overwrite