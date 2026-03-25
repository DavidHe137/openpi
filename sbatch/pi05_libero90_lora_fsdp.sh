#!/bin/bash
#SBATCH --job-name=libero90_lora_fsdp
#SBATCH --output=/coc/flash7/zhenyang/logs/sbatch_out/libero90_lora_fsdp.out
#SBATCH --error=/coc/flash7/zhenyang/logs/sbatch_err/libero90_lora_fsdp.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:8"
#SBATCH --exclude="clippy,voltron"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

cd /coc/flash7/zhenyang/openpi  # Change to openpi directory

## Set clean environment
unset XLA_FLAGS
unset LD_LIBRARY_PATH

export PATH=/nethome/zchen927/.local/bin:/nethome/zchen927/.cursor-server/cli/servers/Stable-9455eaa4c87f2bad91eda3f2bc9b42b16eae1080/server/bin/remote-cli:/srv/rl2-lab/flash7/zhenyang/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/coc/testnvme/admin/tools/skynet-utilities:/opt/slurm/Ubuntu-20.04/current/bin:/coc/testnvme/admin/tools/skynet-utilities:/opt/slurm/Ubuntu-20.04/current/bin:/nethome/zchen927/.cursor-server/extensions/ms-python.debugpy-2025.10.0-linux-x64/bundled/scripts/noConfigScripts

source /nethome/zchen927/.bashrc_sky
conda deactivate

# dataset environment variables
export OPENPI_DATA_HOME=/coc/flash7/zhenyang/openpi/assets # set the cache directory
export HF_LEROBOT_HOME=/coc/flash7/zhenyang/data
export HF_DATASETS_CACHE=/coc/flash7/zhenyang/data
# export LEROBOT_HOME=/home/hice1/zchen927/scratch/datasets/lerobot # set the output directory

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 # train with 90% of GPU memory
# export GOOGLE_APPLICATION_CREDENTIALS=/home/hice1/zchen927/scratch/openpi/assets/openpi-preview.json

# NOTE: multi-gpu with ``--fsdp-devices=8`` batch-size is 256
uv run scripts/train.py \
    pi05_libero90_lora \
    --exp-name=libero90_lora_finetune \
    --fsdp-devices=8 \
    --batch-size=256 \
    --resume