#!/bin/bash
#SBATCH --job-name=sweep_schedulers
#SBATCH --output=logs/sweep_schedulers_%A_%a.out
#SBATCH --error=logs/sweep_schedulers_%A_%a.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --gpus-per-node="l40s:2"
#SBATCH --mem-per-gpu=128
#SBATCH --array=0-2

set -e

SCHEDULERS=(greedy round_robin lookahead)
SCHEDULER=${SCHEDULERS[$SLURM_ARRAY_TASK_ID]}
NUM_ROBOTS_LIST=(20 15 10 5)
NUM_TRIALS_PER_TASK=10
PORT=$((8080 + ${SLURM_ARRAY_TASK_ID:-0}))

echo "======================================"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Scheduler: $SCHEDULER  Port: $PORT"
echo "======================================"

HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE=${HOSTS[0]}

cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$SERVER_JOB_PID" ] && kill -0 $SERVER_JOB_PID 2>/dev/null; then
        echo "Stopping server process (PID: $SERVER_JOB_PID)"
        kill $SERVER_JOB_PID 2>/dev/null || true
        sleep 2
        if kill -0 $SERVER_JOB_PID 2>/dev/null; then
            echo "Force killing server process"
            kill -9 $SERVER_JOB_PID 2>/dev/null || true
        fi
    fi
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# --- Step 1: Launch server ---
srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=2 --overlap --exact -w $NODE bash -c "
    echo 'Starting server on $NODE with scheduler=$SCHEDULER port=$PORT'
    source ~/.bashrc
    source .venv/bin/activate
    uv run scripts/serve_policy.py \
        --env LIBERO \
        --max-batch-size 5 \
        --port $PORT \
        --scheduling-algorithm $SCHEDULER \
        --log-dir logs/server
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID). Waiting for it to initialize..."

# --- Step 2: Sweep num_robots ---
for NUM_ROBOTS in "${NUM_ROBOTS_LIST[@]}"; do
    OUTPUT_DIR="data/libero/sweep_schedulers/scheduler_${SCHEDULER}_num_robots_${NUM_ROBOTS}"
    echo "--------------------------------------"
    echo "Running: scheduler=$SCHEDULER  num_robots=$NUM_ROBOTS"
    echo "Output: $OUTPUT_DIR"
    echo "--------------------------------------"

    srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=20 --overlap --exact -w $NODE bash -c "
        echo 'Starting client on $NODE: scheduler=$SCHEDULER num_robots=$NUM_ROBOTS'
        source scripts/libero_client.sh
        ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
            --host $NODE \
            --port $PORT \
            --num-robots $NUM_ROBOTS \
            --task-suite-name libero_10 \
            --num-trials-per-task $NUM_TRIALS_PER_TASK \
            --control-hz 20 \
            --max-steps 1000 \
            --output-dir $OUTPUT_DIR \
            --progress-type logging \
            --log-dir $OUTPUT_DIR \
            --overwrite \
            --action-chunk-broker-type RTC
    "
done

echo "======================================"
echo "All runs completed for scheduler=$SCHEDULER"
cleanup
trap - EXIT
echo "======================================"
