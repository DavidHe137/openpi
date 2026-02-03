#!/bin/bash
#SBATCH --job-name=benchmark_latency
#SBATCH --output=logs/benchmark_latency_%A_%a.out
#SBATCH --error=logs/benchmark_latency_%A_%a.err
#SBATCH --array=0-41
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude="xaea-12, dynamics"

# Exit on error
set -e  

BATCH_SIZES=(1 2 4 8 16 32 64)
TIMEOUTS=(0 5 10 20 50 100)

NUM_BATCH_SIZES=${#BATCH_SIZES[@]}
NUM_TIMEOUTS=${#TIMEOUTS[@]}

# Decode 2D index (batch_size_idx, timeout_idx) from 1D SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BATCH_IDX=$((TASK_ID % NUM_BATCH_SIZES))
TIMEOUT_IDX=$((TASK_ID / NUM_BATCH_SIZES))

BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
TIMEOUT_MS=${TIMEOUTS[$TIMEOUT_IDX]}

port=$((8080 + ${BATCH_IDX:-0}))

echo "======================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running with batch_size=$BATCH_SIZE, timeout_ms=$TIMEOUT_MS and port=$port"
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

# Request rates to sweep (requests/second)
REQUEST_RATES=(5 10 20 50 100)

RESULT_DIR="benchmarks/latency_batching_timeout_4"

source .venv/bin/activate

echo "--------------------------------------"
echo "Starting server with batch_size=$BATCH_SIZE, timeout_ms=$TIMEOUT_MS"
echo "--------------------------------------"

# Start the background server for this (batch_size, timeout_ms) combination
cmd="uv run scripts/serve_policy.py \
    --env=LIBERO \
    --batch_size=$BATCH_SIZE \
    --batch-timeout-ms=$TIMEOUT_MS \
    --port=$port"

echo "Starting background process: $cmd"
$cmd &
BACKGROUND_PID=$!
echo "Background process started with PID: $BACKGROUND_PID"

# Wait for the background service to initialize
sleep 250

# Run the foreground benchmarks for all request rates
echo "Running benchmark for timeout_ms=$TIMEOUT_MS..."
for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
    # Check if this (batch_size, timeout_ms, request_rate) combination
    # already has a result JSON in RESULT_DIR. If so, skip.
    SHOULD_RUN=$(
RESULT_DIR="$RESULT_DIR" BATCH_SIZE="$BATCH_SIZE" TIMEOUT_MS="$TIMEOUT_MS" REQUEST_RATE="$REQUEST_RATE" python - << 'PY'
import json
import os

result_dir = os.environ["RESULT_DIR"]
batch_size = int(os.environ["BATCH_SIZE"])
request_rate = float(os.environ["REQUEST_RATE"])

exists = False
for fname in os.listdir(result_dir):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(result_dir, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue

    if (
        data.get("batch_size") == batch_size
        and float(data.get("request_rate")) == request_rate
    ):
        exists = True
        break

print("1" if not exists else "0")
PY
    )

    if [ "$SHOULD_RUN" -eq 0 ]; then
        echo "  - request_rate=$REQUEST_RATE (already exists, skipping)"
        continue
    fi

    echo "  - request_rate=$REQUEST_RATE (running)"
    uv run scripts/benchmark.py \
        --host localhost \
        --port $port \
        --env libero \
        --num-requests 300 \
        --request-rate $REQUEST_RATE \
        --max-concurrency 300 \
        --metric-percentiles 95,99 \
        --save-result \
        --save-result-dir benchmarks/latency_batching_timeout_4
done