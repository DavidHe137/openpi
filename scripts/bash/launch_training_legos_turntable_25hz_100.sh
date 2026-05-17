#!/bin/bash
#SBATCH --job-name=legos_turntable_25hz_finetune_act100
#SBATCH --output=/home/hice1/rbansal66/scratch/openpi/scripts/bash/log/legos_turntable_25hz_finetune_act100.out
#SBATCH --error=/home/hice1/rbansal66/scratch/openpi/scripts/bash/log/legos_turntable_25hz_finetune_act100.err
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node="h100:2"
#SBATCH --mem-per-gpu=100G
#SBATCH -c 10
#SBATCH -t8:00:00

cd /home/hice1/rbansal66/scratch/openpi

source ~/.bashrc

nvidia-smi
hostname

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# uv run scripts/compute_norm_stats.py --config-name=pi05_legos_turntable_25hz_act100
uv run scripts/train.py pi05_legos_turntable_25hz_act100 --exp-name=legos_turntable_25hz_finetune_act100 --fsdp-devices 2 --resume