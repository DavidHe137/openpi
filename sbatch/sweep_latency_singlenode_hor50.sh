#!/bin/bash
#SBATCH --job-name=sweep_latency
#SBATCH --output=scripts/log/sweep_latency_%j.out
#SBATCH --error=scripts/log/sweep_latency_%j.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy, xaea-12, cyborg"
#SBATCH --mem-per-gpu=64

# Exit on error
set -e

USE_RTC=$1
scontrol update JobId=${SLURM_JOB_ID} JobName=sweep_latency_${USE_RTC}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running with args.use_rtc=$USE_RTC"
echo "======================================"

# Source shell configuration
source ~/.bashrc

# Trap to ensure cleanup happens on script exit (normal or error)
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$BACKGROUND_PID" ] && kill -0 $BACKGROUND_PID 2>/dev/null; then
        echo "Stopping background process (PID: $BACKGROUND_PID)"
        kill $BACKGROUND_PID
        wait $BACKGROUND_PID 2>/dev/null || true
    fi
    echo "Cleanup complete"
}

# Register the cleanup function to run on EXIT
trap cleanup EXIT INT TERM

# Start the background process based on action horizon
source .venv/bin/activate


echo "Starting background process for horizon $ACTION_HORIZON (default): serve_policy.py --env=LIBERO"
uv run scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi05_libero_lora \
    --policy.dir=checkpoints/pi05_libero_lora/libero_lora_finetune_single_gpu/10000 &

BACKGROUND_PID=$!
echo "Background process started with PID: $BACKGROUND_PID"

# Wait for the background service to initialize
sleep 5

# Run the foreground script
echo "Running libero client..."

source ~/.bashrc
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
if [ "$USE_RTC" = "True" ]; then
    USE_RTC_FLAG="--args.use_rtc"
else
    USE_RTC_FLAG=""
fi
./examples/libero/.venv/bin/python examples/libero/main_rtc_parallel.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 20 \
    --args.latency-ms 150.0 200.0 250.0 300.0 400.0 \
    --args.action-horizon 50 \
    --args.use_rtc

echo "======================================"
echo "Completed run with args.use_rtc=$USE_RTC"
echo "======================================"
