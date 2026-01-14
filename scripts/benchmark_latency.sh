#!/bin/bash
#SBATCH --job-name=benchmark_latency
#SBATCH --output=../openpi/logs/benchmark_latency_%A_%a.out
#SBATCH --error=../openpi/logs/benchmark_latency_%A_%a.err
#SBATCH --array=0-20
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# Exit on error
set -e  

MODELS=("LIBERO_REALTIME" "LIBERO_PI0" "LIBERO_PYTORCH")
BATCH_SIZES=(1 2 4 8 16 32 64)

# Calculate model and batch size indices from array task ID
MODEL_INDEX=$((${SLURM_ARRAY_TASK_ID} / 7))
BATCH_SIZE_INDEX=$((${SLURM_ARRAY_TASK_ID} % 7))

MODEL=${MODELS[$MODEL_INDEX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_SIZE_INDEX]}

port=$((8080 + ${SLURM_ARRAY_TASK_ID:-0}))

echo "======================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running with model=$MODEL, batch_size=$BATCH_SIZE, and port=$port"
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
    --env=$MODEL \
    --batch_size=$BATCH_SIZE \
    --port=$port"

echo "Starting background process: $cmd"
source .venv/bin/activate
$cmd &

BACKGROUND_PID=$!
echo "Background process started with PID: $BACKGROUND_PID"

# Wait for the background service to initialize
sleep 50

# Run the foreground script
echo "Running benchmark..."

# for loop over request rates
REQUEST_RATES=(100)
for REQUEST_RATE in ${REQUEST_RATES[@]}; do
    uv run scripts/benchmark.py \
        --host localhost \
        --port $port \
        --env libero \
        --num-requests 300 \
        --request-rate $REQUEST_RATE \
        --max-concurrency 300 \
        --metric-percentiles 95,99 \
        --save-result \
        --save-result-dir ../openpi/benchmarks/l40s_100_${MODEL}
done
