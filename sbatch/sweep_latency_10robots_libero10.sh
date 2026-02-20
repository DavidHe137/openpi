#!/bin/bash
#SBATCH --job-name=sweep_lat_10robots
#SBATCH --output=sbatch/log/sweep_latency_10robots_%A_%a.out
#SBATCH --error=sbatch/log/sweep_latency_10robots_%A_%a.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --exclude="clippy,xaea-12,dynamics"
#SBATCH --mem-per-gpu=64G
#SBATCH --array=0-6

set -e

LATENCY_VALUES=(0 50 100 150 200 250 300)
LATENCY=${LATENCY_VALUES[$SLURM_ARRAY_TASK_ID]}
NUM_ROBOTS=10
PORT=$((8080 + SLURM_ARRAY_TASK_ID))

scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} JobName=sweep_lat_10robots_${LATENCY}ms

mkdir -p sbatch/log

echo "======================================"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Latency: ${LATENCY}ms, Num robots: ${NUM_ROBOTS}, Port: ${PORT}"
echo "======================================"

# Get the hostnames of the allocated nodes
HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SERVER_NODE=${HOSTS[0]}
CLIENT_NODE=${HOSTS[1]}

echo "Server node: $SERVER_NODE"
echo "Client node: $CLIENT_NODE"

# Ensure cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$SERVER_JOB_PID" ] && kill -0 $SERVER_JOB_PID 2>/dev/null; then
        echo "Stopping server process (PID: $SERVER_JOB_PID)"
        kill $SERVER_JOB_PID
        wait $SERVER_JOB_PID 2>/dev/null || true
    fi
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# Build latency-ms arg: NUM_ROBOTS copies of LATENCY
LATENCY_ARGS=""
for i in $(seq 1 $NUM_ROBOTS); do
    LATENCY_ARGS="$LATENCY_ARGS $LATENCY"
done

OUTPUT_DIR="data/libero/sweep_latency_10robots/latency_${LATENCY}ms_job${SLURM_ARRAY_JOB_ID}"

# --- Step 1: Launch the policy server on the first node ---
echo "Starting server on $SERVER_NODE..."
srun --nodes=1 --ntasks=1 -w $SERVER_NODE bash -c "
    source ~/.bashrc
    source .venv/bin/activate
    uv run scripts/serve_policy.py --env LIBERO --max_batch_size 4 --port $PORT
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID). Waiting for it to initialize..."
sleep 60

# --- Step 2: Run the multi-robot client on the second node ---
echo "Starting client on $CLIENT_NODE with latency=${LATENCY}ms..."
srun --nodes=1 --ntasks=1 -w $CLIENT_NODE bash -c "
    source ~/.bashrc
    source scripts/libero_client.sh
    python examples/libero/main_multi_robot_runtime.py \
        --host $SERVER_NODE \
        --port $PORT \
        --task-suite-name libero_10 \
        --num-robots $NUM_ROBOTS \
        --latency-ms $LATENCY_ARGS \
        --output-dir $OUTPUT_DIR \
        --overwrite
"

echo "======================================"
echo "Completed run with latency=${LATENCY}ms"
echo "======================================"
