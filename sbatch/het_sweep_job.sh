#!/bin/bash
# Het group 0: server (l40s GPU + 4 CPUs)
#SBATCH --job-name=sweep_schedulers
#SBATCH --output=logs/sweep_schedulers_%j_server.out
#SBATCH --error=logs/sweep_schedulers_%j_server.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:1
#SBATCH hetjob
# Het group 1: client (a40 GPU + 20 CPUs)
#SBATCH --job-name=sweep_schedulers_client
#SBATCH --output=logs/sweep_schedulers_%j_client.out
#SBATCH --error=logs/sweep_schedulers_%j_client.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=a40:1

SCHEDULER=${1:?Usage: sbatch het_sweep_job.sh <scheduler> <max_batch_size>}
MAX_BATCH_SIZE=${2:?Usage: sbatch het_sweep_job.sh <scheduler> <max_batch_size>}

set -e
source sbatch/utils.sh

PORT=$(find_free_port)
SERVER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
NUM_TRIALS_PER_TASK=2

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Scheduler: $SCHEDULER  MaxBatchSize: $MAX_BATCH_SIZE  Port: $PORT"
echo "Server node: $SERVER_NODE"
echo "======================================"

# --- Step 1: Launch server (het group 0) ---
srun --het-group=0 --gpus-per-node=l40s:1 --cpus-per-task=4 bash -c "
    set -e
    echo 'Starting server on $SERVER_NODE: scheduler=$SCHEDULER max_batch=$MAX_BATCH_SIZE port=$PORT'
    source ~/.bashrc
    source .venv/bin/activate
    uv run scripts/serve_policy.py \
        --env LIBERO \
        --max-batch-size $MAX_BATCH_SIZE \
        --port $PORT \
        --scheduling-algorithm $SCHEDULER \
        --log-dir logs/server
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID)."

sleep 10
setup_server_monitor $SERVER_JOB_PID

# --- Step 2: Sweep num_robots (het group 1) ---
NUM_ROBOTS_LIST=(20 15 10 5)
NUM_RUNS=1
for NUM_ROBOTS in "${NUM_ROBOTS_LIST[@]}"; do
    for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
        OUTPUT_DIR="data/libero/batching/scheduler_${SCHEDULER}_max_batch_${MAX_BATCH_SIZE}_num_robots_${NUM_ROBOTS}_run_${RUN_IDX}"
        echo "--------------------------------------"
        echo "Running: scheduler=$SCHEDULER  max_batch=$MAX_BATCH_SIZE  num_robots=$NUM_ROBOTS  run=$RUN_IDX"
        echo "Output: $OUTPUT_DIR"
        echo "--------------------------------------"

        srun --het-group=1 --gpus-per-node=a40:1 --cpus-per-task=20 bash -c "
            set -e
            echo 'Starting client: num_robots=$NUM_ROBOTS run=$RUN_IDX'
            source scripts/bash/libero_client.sh
            ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
                --host $SERVER_NODE \
                --port $PORT \
                --num-robots $NUM_ROBOTS \
                --task-suite-name libero_10 \
                --num-trials-per-task $NUM_TRIALS_PER_TASK \
                --control-hz 20 \
                --max-steps 600 \
                --output-dir $OUTPUT_DIR \
                --progress-type logging \
                --log-dir $OUTPUT_DIR \
                --overwrite \
                --action-chunk-broker-type RTC
        "
    done
done

echo "======================================"
echo "All runs completed for scheduler=$SCHEDULER  max_batch=$MAX_BATCH_SIZE"

cleanup
trap - EXIT
wait $SERVER_JOB_PID 2>/dev/null || true
echo "======================================"
