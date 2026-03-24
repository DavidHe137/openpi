#!/bin/bash
#SBATCH --job-name=sweep_schedulers
#SBATCH --output=logs/sweep_schedulers_%A_%a.out
#SBATCH --error=logs/sweep_schedulers_%A_%a.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gpus-per-node="l40s:2"
#SBATCH --mem-per-gpu=64
#SBATCH --array=0-2
#SBATCH --exclude="dynamics"

set -e

SCHEDULERS=(greedy round_robin lookahead)
SCHEDULER=${SCHEDULERS[$SLURM_ARRAY_TASK_ID]}
NUM_ROBOTS_LIST=(20 15 10 5)
NUM_TRIALS_PER_TASK=10
PORT=$((8080 + ${SLURM_ARRAY_TASK_ID:-0}))
BASE_EXPERIMENT_CONFIG="examples/libero/experiment_config_20_robot_het.jsonc"

echo "======================================"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Scheduler: $SCHEDULER  Port: $PORT"
echo "======================================"

HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE=${HOSTS[0]}

cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$MONITOR_PID" ] && kill -0 $MONITOR_PID 2>/dev/null; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
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

make_experiment_config() {
    local output_path="$1"
    local broker_type="$2"
    local num_robots="$3"
    local trials_per_robot="$4"
    python - "$BASE_EXPERIMENT_CONFIG" "$output_path" "$broker_type" "$num_robots" "$trials_per_robot" <<'PY'
import json
import pathlib
import sys

template_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
broker_type = str(sys.argv[3]).lower()
num_robots = int(sys.argv[4])
trials_per_robot = int(sys.argv[5])

cfg = json.loads(template_path.read_text(encoding="utf-8"))
cfg["experiment"]["action_chunk_broker_type"] = broker_type
cfg["experiment"]["num_robots"] = num_robots
cfg["experiment"]["trials_per_robot"] = trials_per_robot
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
}

# --- Step 1: Launch server ---
srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=4 --overlap --exact -w $NODE bash -c "
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
echo "Server launched (PID $SERVER_JOB_PID)."

# Monitor server in background: terminate the job if it dies unexpectedly
( while sleep 5; do
    if ! kill -0 $SERVER_JOB_PID 2>/dev/null; then
        echo "ERROR: Server process (PID $SERVER_JOB_PID) died unexpectedly - terminating job"
        kill -TERM $$
        break
    fi
  done ) &
MONITOR_PID=$!

# --- Step 2: Sweep num_robots ---
NUM_RUNS=1
for NUM_ROBOTS in "${NUM_ROBOTS_LIST[@]}"; do
    for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
        OUTPUT_DIR="data/libero/sweep_schedulers/scheduler_${SCHEDULER}_num_robots_${NUM_ROBOTS}_run_${RUN_IDX}"
        EXPERIMENT_CONFIG="$OUTPUT_DIR/experiment_config.json"
        make_experiment_config "$EXPERIMENT_CONFIG" "rtc" "$NUM_ROBOTS" "$NUM_TRIALS_PER_TASK"
        echo "--------------------------------------"
        echo "Running: scheduler=$SCHEDULER  num_robots=$NUM_ROBOTS  run=$RUN_IDX"
        echo "Output: $OUTPUT_DIR"
        echo "--------------------------------------"

        srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=22 --overlap --exact -w $NODE bash -c "
            set -e
            echo 'Starting client on $NODE: scheduler=$SCHEDULER num_robots=$NUM_ROBOTS run=$RUN_IDX'
            source scripts/libero_client.sh
            ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
                --host $NODE \
                --port $PORT \
                --task-suite-name libero_10 \
                --control-hz 20 \
                --max-steps 600 \
                --output-dir $OUTPUT_DIR \
                --progress-type logging \
                --log-dir $OUTPUT_DIR \
                --overwrite \
                --experiment-config $EXPERIMENT_CONFIG \
                --toxiproxy-server-bin /coc/flash7/rbansal66/vvla/toxiproxy-server-linux-amd64
        "
    done
done

echo "======================================"
echo "All runs completed for scheduler=$SCHEDULER"

# --- Step 3: Run synchronous client for greedy only ---
if [ "$SCHEDULER" = "greedy" ]; then
    echo "======================================"
    echo "Running synchronous client for scheduler=$SCHEDULER"
    echo "======================================"
    for NUM_ROBOTS in "${NUM_ROBOTS_LIST[@]}"; do
        for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
            OUTPUT_DIR="data/libero/sweep_schedulers/scheduler_${SCHEDULER}_num_robots_${NUM_ROBOTS}_run_${RUN_IDX}_sync"
            EXPERIMENT_CONFIG="$OUTPUT_DIR/experiment_config.json"
            make_experiment_config "$EXPERIMENT_CONFIG" "sync" "$NUM_ROBOTS" "$NUM_TRIALS_PER_TASK"
            echo "--------------------------------------"
            echo "Running: scheduler=$SCHEDULER  num_robots=$NUM_ROBOTS  run=$RUN_IDX  (sync)"
            echo "Output: $OUTPUT_DIR"
            echo "--------------------------------------"

            srun --ntasks=1 --gpus-per-node="l40s:1" --cpus-per-task=22 --overlap --exact -w $NODE bash -c "
                set -e
                echo 'Starting sync client on $NODE: scheduler=$SCHEDULER num_robots=$NUM_ROBOTS run=$RUN_IDX'
                source scripts/libero_client.sh
                ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
                    --host $NODE \
                    --port $PORT \
                    --task-suite-name libero_10 \
                    --control-hz 20 \
                    --max-steps 600 \
                    --output-dir $OUTPUT_DIR \
                    --progress-type logging \
                    --log-dir $OUTPUT_DIR \
                    --overwrite \
                    --experiment-config $EXPERIMENT_CONFIG \
                    --toxiproxy-server-bin /coc/flash7/rbansal66/vvla/toxiproxy-server-linux-amd64
            "
        done
    done
    echo "Synchronous client runs complete for scheduler=$SCHEDULER"
fi

cleanup
trap - EXIT
wait $SERVER_JOB_PID 2>/dev/null || true
echo "======================================"
