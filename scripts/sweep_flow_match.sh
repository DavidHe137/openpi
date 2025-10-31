#!/bin/bash
#SBATCH --job-name=sweep_flow_match
#SBATCH --output=logs/sweep_flow_match_%A_%a.out
#SBATCH --error=logs/sweep_flow_match_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1

# Exit on error
set -e

# Define the sweep values for args.num_steps
NUM_STEPS_VALUES=(1 2 5 10)

# Get the current value based on SLURM_ARRAY_TASK_ID
NUM_STEPS=${NUM_STEPS_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running with args.num_steps=$NUM_STEPS"
echo "======================================"

# Source shell configuration
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true

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

# Start the background process
echo "Starting background process: serve_policy.py --env=LIBERO --args.num_steps=$NUM_STEPS"
.venv/bin/python scripts/serve_policy.py --env=LIBERO --num_steps=$NUM_STEPS &
BACKGROUND_PID=$!
echo "Background process started with PID: $BACKGROUND_PID"

# Wait for the background service to initialize
sleep 5

# Run the foreground script
echo "Running libero client..."

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=osmesa
python examples/libero/main_parallel.py --args.task-suite-name libero_10 --args.num-workers=6 --args.num-trials-per-task=10 --args.experiment-name=flow_match_num_steps_$NUM_STEPS

echo "======================================"
echo "Completed run with args.num_steps=$NUM_STEPS"
echo "======================================"

