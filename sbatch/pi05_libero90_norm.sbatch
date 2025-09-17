#!/bin/bash
#SBATCH --job-name=libero90_norm
#SBATCH --output=/coc/flash7/zhenyang/logs/sbatch_out/libero90_norm.out
#SBATCH --error=/coc/flash7/zhenyang/logs/sbatch_err/libero90_norm.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy,voltron"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

cd /coc/flash7/zhenyang/openpi  # Change to openpi directory

source $HOME/.local/bin/env # source the uv env

# dataset environment variables
export OPENPI_DATA_HOME=/coc/flash7/zhenyang/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/coc/flash7/zhenyang/data
export HF_DATASETS_CACHE=/coc/flash7/zhenyang/data
# export LEROBOT_HOME=/home/hice1/zchen927/scratch/datasets/lerobot # set the output directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 # train with 90% of GPU memory

uv run scripts/compute_norm_stats.py --config-name pi05_libero90_lora