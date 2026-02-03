#!/bin/bash
#SBATCH --job-name=sweep_moving_bowl
#SBATCH --output=scripts/log/sweep_moving_bowl_%j.out
#SBATCH --error=scripts/log/sweep_moving_bowl_%j.err
#SBATCH --partition=overcap
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy, xaea-12"
#SBATCH --mem-per-gpu=64

set -e

USE_RTC=$1
export USE_RTC
scontrol update JobId=${SLURM_JOB_ID} JobName=sweep_moving_bowl_${USE_RTC}

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running moving-bowl sweep with args.use_rtc=$USE_RTC"
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
    uv run scripts/serve_policy.py --env=LIBERO
" &
SERVER_JOB_PID=$!
echo "Server launched (PID $SERVER_JOB_PID). Waiting for it to initialize..."
sleep 15

# --- Step 2: Run the client on the second node ---
echo "Starting client on $CLIENT_NODE..."
srun --nodes=1 --ntasks=1 -w $CLIENT_NODE bash -c "
    set -e
    source ~/.bashrc
    source examples/libero/.venv/bin/activate
    export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID=0

    REPLAN_LIST=\"1 5\"
    N_OBJ_STEPS_LIST=\"1 5\"
    LAT_LIST=\"0 10 50 100 200\"

    if [ \"\$USE_RTC\" = \"True\" ]; then
        echo \"Running RTC=True sweeps (varying n_obj_steps, latency_ms)\"
        for N_OBJ in \$N_OBJ_STEPS_LIST; do
            for LAT in \$LAT_LIST; do
                echo \"======================================================\"
                echo \"RTC=True, n_obj_steps=\$N_OBJ, latency_ms=\$LAT\"
                echo \"======================================================\"
                ./examples/libero/.venv/bin/python examples/libero/main_rtc_moving_bowl.py \\
                    --args.host $SERVER_NODE \\
                    --args.port 8080 \\
                    --args.use_rtc \\
                    --args.n_obj_steps \$N_OBJ \\
                    --args.latency_ms \$LAT
            done
        done
    else
        echo \"Running RTC=False sweeps (varying replan_steps, n_obj_steps, latency_ms)\"
        for REPLAN in \$REPLAN_LIST; do
            for N_OBJ in \$N_OBJ_STEPS_LIST; do
                for LAT in \$LAT_LIST; do
                    echo \"======================================================\"
                    echo \"RTC=False, replan_steps=\$REPLAN, n_obj_steps=\$N_OBJ, latency_ms=\$LAT\"
                    echo \"======================================================\"
                    ./examples/libero/.venv/bin/python examples/libero/main_rtc_moving_bowl.py \\
                        --args.host $SERVER_NODE \\
                        --args.port 8080 \\
                        --args.replan_steps \$REPLAN \\
                        --args.n_obj_steps \$N_OBJ \\
                        --args.latency_ms \$LAT
                done
            done
        done
    fi
"

echo "======================================"
echo "Completed moving-bowl sweep with args.use_rtc=$USE_RTC"
echo "======================================"


