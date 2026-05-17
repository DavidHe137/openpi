#!/bin/bash
#SBATCH --job-name=turntable_20hz_finetune
#SBATCH --output=/home/hice1/rbansal66/scratch/openpi/scripts/bash/log/turntable_20hz_finetune.out
#SBATCH --error=/home/hice1/rbansal66/scratch/openpi/scripts/bash/log/turntable_20hz_finetune.err
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node="h100:2"
#SBATCH --mem-per-gpu=100G
#SBATCH -t8:00:00

cd /home/hice1/rbansal66/scratch/openpi

source ~/.bashrc

nvidia-smi
hostname

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# uv run scripts/compute_norm_stats.py --config-name=pi05_turntable_20hz
uv run scripts/train.py pi05_turntable_20hz --exp-name=turntable_20hz_finetune --fsdp-devices 2 --resume