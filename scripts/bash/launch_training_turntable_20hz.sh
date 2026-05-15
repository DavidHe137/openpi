#!/bin/bash
#SBATCH --job-name=turntable_20hz_finetune
#SBATCH --output=/coc/flash7/rbansal66/vvla/openpi-training/scripts/bash/log/turntable_20hz_finetune.out
#SBATCH --error=/coc/flash7/rbansal66/vvla/openpi-training/scripts/bash/log/turntable_20hz_finetune.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node="a40:4"
#SBATCH --mem-per-gpu=48G
#SBATCH --requeue

cd /coc/flash7/rbansal66/vvla/openpi-training

source ~/.bashrc

nvidia-smi
hostname

# uv run scripts/compute_norm_stats.py --config-name pi05_throw_the_legos
# OOM: lower mem fraction slightly for BFC headroom, or pass e.g. --batch-size 4 (must divide #GPUs).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Default fsdp_devices=2 on 8 GPUs -> (4,2) mesh; XLA then does involuntary full rematerialization (~32GiB/GPU).
# Set --fsdp-devices to match GPU count on this node (must divide jax.device_count(); batch_size must too).

uv run scripts/compute_norm_stats.py --config-name=pi05_turntable_20hz
uv run scripts/train.py pi05_turntable_20hz --exp-name=turntable_20hz_finetune --fsdp-devices 4 --resume