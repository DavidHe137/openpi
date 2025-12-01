#!/bin/bash
#SBATCH --job-name=sweep_latency
#SBATCH --output=scripts/log/sweep_latency_%j.out
#SBATCH --error=scripts/log/sweep_latency_%j.err
#SBATCH --partition=overcap
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --exclude="clippy, xaea-12, cyborg"
#SBATCH --mem-per-gpu=64

set -e

USE_RTC=$1
scontrol update JobId=${SLURM_JOB_ID} JobName=sweep_latency_${USE_RTC}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running with args.use_rtc=$USE_RTC"
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

# --- Step 1: Launch the server on the first node ---
echo "Starting server on $SERVER_NODE..."
srun --nodes=1 --ntasks=1 -w $SERVER_NODE bash -c "
    source ~/.bashrc
    source .venv/bin/activate
    uv run scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi05_libero_lora \
        --policy.dir=checkpoints/pi05_libero_lora/libero_lora_finetune_single_gpu/10000
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID). Waiting for it to initialize..."
sleep 10

# --- Step 2: Run the client on the second node ---
echo "Starting client on $CLIENT_NODE..."
srun --nodes=1 --ntasks=1 -w $CLIENT_NODE bash -c "
    source ~/.bashrc
    source examples/libero/.venv/bin/activate
    export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID=0
    if [ \"$USE_RTC\" = \"True\" ]; then
        USE_RTC_FLAG=\"--args.use_rtc\"
    else
        USE_RTC_FLAG=\"\"
    fi
    ./examples/libero/.venv/bin/python examples/libero/main_rtc_parallel.py \
        --args.host $SERVER_NODE \
        --args.task-suite-name libero_10 \
        --args.num-trials-per-task 20 \
        --args.latency-ms 10.0 50.0 100.0 150.0 200.0 250.0 300.0 400.0 \
        --args.action-horizon 50 \
        --args.use_rtc
"

echo "======================================"
echo "Completed run with args.use_rtc=$USE_RTC"
echo "======================================"
