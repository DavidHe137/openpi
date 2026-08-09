#!/bin/bash
#SBATCH --job-name=pi05_legos_turntable_better_full_30hz_rtc_act20_lora_d15
#SBATCH --output=/coc/flash8/dhe83/train-time-rtc/third_party/openpi/scripts/bash/log/pi05_legos_turntable_better_full_30hz_rtc_act20_lora_d15.out
#SBATCH --error=/coc/flash8/dhe83/train-time-rtc/third_party/openpi/scripts/bash/log/pi05_legos_turntable_better_full_30hz_rtc_act20_lora_d15.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node="l40s:2"
#SBATCH --exclude="bishop,zhurong"
#SBATCH --mem-per-gpu=48G
#SBATCH --time=24:00:00
#SBATCH --qos="short"
#SBATCH --requeue

cd /coc/flash8/dhe83/train-time-rtc/third_party/openpi

source ~/.bashrc

nvidia-smi
hostname

# uv run scripts/compute_norm_stats.py --config-name pi05_throw_the_legos
# OOM: lower mem fraction slightly for BFC headroom, or pass e.g. --batch-size 4 (must divide #GPUs).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Default fsdp_devices=2 on 8 GPUs -> (4,2) mesh; XLA then does involuntary full rematerialization (~32GiB/GPU).
# Set --fsdp-devices to match GPU count on this node (must divide jax.device_count(); batch_size must too).

uv run scripts/train.py pi05_legos_turntable_better_full_30hz_rtc_act20_lora_d15 --exp-name=pi05_legos_turntable_better_full_30hz_rtc_act20_lora_d15 --fsdp-devices 2 --resume
