#!/bin/bash
#SBATCH --job-name=benchmark_latency
#SBATCH --output=logs/benchmark_latency_%A_%a.out
#SBATCH --error=logs/benchmark_latency_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# Exit on error
set -e  

NUM_STEPS_VALUES=(1 2 5 10)
NUM_STEPS=${NUM_STEPS_VALUES[$SLURM_ARRAY_TASK_ID]}

port=$((8000 + ${SLURM_ARRAY_TASK_ID:-0}))

echo "======================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running with num_steps=$NUM_STEPS and port=$port"
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
cmd="uv run scripts/serve_policy.py \
    --env=LIBERO \
    --num_steps=$NUM_STEPS \
    --port=$port"

echo "Starting background process: $cmd"
source .venv/bin/activate
$cmd &

BACKGROUND_PID=$!
echo "Background process started with PID: $BACKGROUND_PID"

# Wait for the background service to initialize
sleep 5

# Run the foreground script
echo "Running benchmark..."

# for loop over request rates
REQUEST_RATES=(1 2 3 4 5 6 7 8 9 10 15 20)
for REQUEST_RATE in ${REQUEST_RATES[@]}; do
    uv run scripts/benchmark.py \
        --host localhost \
        --port $port \
        --env libero \
        --num-requests 100 \
        --request-rate $REQUEST_RATE \
        --max-concurrency 100 \
        --metric-percentiles 95,99 \
        --save-result \
        --save-result-dir benchmarks/latency
done