#!/bin/bash
#SBATCH --job-name=sweep_action_horizons
#SBATCH --output=scripts/log/sweep_action_horizons_%j.out
#SBATCH --error=scripts/log/sweep_action_horizons_%j.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy, xaea-12, gundam, crushinator, protocol, sonny, consu"
#SBATCH --mem-per-gpu=64

# Exit on error
set -e

ACTION_HORIZON=$1
scontrol update JobId=${SLURM_JOB_ID} JobName=sweep_action_horizons_${ACTION_HORIZON}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running with args.action_horizon=$ACTION_HORIZON"
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

if [ "$ACTION_HORIZON" = "25" ]; then
    echo "Starting background process for horizon 25: serve_policy.py"
    uv run scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi05_libero_lora \
        --policy.dir=checkpoints/pi05_libero_lora/libero_lora_finetune_single_gpu/10000 &
elif [ "$ACTION_HORIZON" = "50" ]; then
    echo "Starting background process for horizon 50: serve_policy.py"
    uv run scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi05_libero_lora \
        --policy.dir=checkpoints/pi05_libero_lora/libero_lora_finetune_single_gpu/10000 &
elif [ "$ACTION_HORIZON" = "100" ]; then
    echo "Starting background process for horizon $ACTION_HORIZON (default): serve_policy.py --env=LIBERO"
    uv run scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi05_libero_lora \
        --policy.dir=checkpoints/pi05_libero_lora/libero_lora_finetune_single_gpu/10000 &
elif [ "$ACTION_HORIZON" = "10" ]; then
    echo "Starting background process for horizon $ACTION_HORIZON (default): serve_policy.py --env=LIBERO"
    uv run scripts/serve_policy.py --env=LIBERO &
fi

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
./examples/libero/.venv/bin/python examples/libero/main.py \
    --args.task-suite-name=libero_10 \
    --args.num-trials-per-task=20 \
    --args.action-horizon=$ACTION_HORIZON \
    --args.use_rtc

echo "======================================"
echo "Completed run with args.action_horizon=$ACTION_HORIZON"
echo "======================================"