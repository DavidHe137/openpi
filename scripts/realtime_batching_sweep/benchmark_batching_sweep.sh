#!/bin/bash
#SBATCH --job-name=pi0_batch_sweep
#SBATCH --partition=rl2-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --output=logs/batch_sweep_%j.out
#SBATCH --error=logs/batch_sweep_%j.err
#SBATCH --chdir=/srv/rl2-lab/flash8/rbansal66/vvla/openpi
#SBATCH --exclude=xaea-12

# This is the CLIENT script that runs on A40
# The server runs on a separate l40s node

set -e

# Change to workspace directory
cd /srv/rl2-lab/flash8/rbansal66/vvla/openpi

# Activate LIBERO environment
echo "Activating LIBERO environment..."
if [ -f "examples/libero/.venv/bin/activate" ]; then
    source examples/libero/.venv/bin/activate
else
    echo "ERROR: LIBERO venv not found at examples/libero/.venv"
    echo "Please create it first with:"
    echo "  uv venv --python 3.8 examples/libero/.venv"
    echo "  source examples/libero/.venv/bin/activate"
    echo "  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:${PWD}/third_party/libero"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# Configuration
BATCH_SIZES=(32)
NUM_ROBOTS_LIST=(1 2 3 5 10 20)
ENVS=("LIBERO")
ENV_NAMES=("non_realtime")
BASE_OUTPUT_DIR="data/libero/batch_sweep_$(date +%Y%m%d_%H%M%S)"
NUM_TRIALS_PER_ROBOT=10
CONTROL_HZ=20
MAX_STEPS=500

# Server connection details (will be set by the server launcher)
SERVER_HOST="${SERVER_HOST:-bishop}"

# Use shared filesystem instead of /tmp (which is node-local)
CONFIG_DIR="/srv/rl2-lab/flash8/rbansal66/vvla/openpi/tmp"
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/pi0_server_config_${SLURM_JOB_ID}.txt"

echo "============================================"
echo "Starting batch size sweep experiment"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Server host: ${SERVER_HOST}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "============================================"

# Create logs directory
mkdir -p logs
mkdir -p "${BASE_OUTPUT_DIR}"

# Function to wait for server to be ready
wait_for_server() {
    local host=$1
    local port=${2:-8080}
    local max_wait=600  # 10 minutes (server initialization can take time)
    local elapsed=0
    
    echo "Waiting for server at ${host}:${port}..."
    
    while ! timeout 1 bash -c "echo > /dev/tcp/${host}/${port}" 2>/dev/null; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Server failed to start within ${max_wait} seconds"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... still waiting (${elapsed}s)"
        fi
    done
    
    echo "Server is ready!"
    sleep 10  # Extra buffer for full initialization
    return 0
}

# Function to run client experiment
run_client_experiment() {
    local num_robots=$1
    local env_name=$2
    local batch_size=$3
    local output_dir=$4
    
    echo ""
    echo "========================================"
    echo "Running experiment:"
    echo "  Environment: ${env_name}"
    echo "  Batch size: ${batch_size}"
    echo "  Num robots: ${num_robots}"
    echo "  Output dir: ${output_dir}"
    echo "========================================"
    
    # Run the client
    python examples/libero/main_multi_robot_runtime.py \
        --host "${SERVER_HOST}" \
        --num_robots ${num_robots} \
        --num_trials_per_robot ${NUM_TRIALS_PER_ROBOT} \
        --overwrite \
        --control-hz ${CONTROL_HZ} \
        --output-dir "${output_dir}" \
        --max_steps ${MAX_STEPS}
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Experiment completed successfully"
    else
        echo "WARNING: Experiment failed with exit code ${exit_code}"
    fi
    
    echo ""
    return $exit_code
}

# Main sweep loop
for env_idx in "${!ENVS[@]}"; do
    env="${ENVS[$env_idx]}"
    env_name="${ENV_NAMES[$env_idx]}"
    
    echo ""
    echo "###############################################"
    echo "# Environment: ${env} (${env_name})"
    echo "###############################################"
    echo ""
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "==============================================="
        echo "= Batch size: ${batch_size}"
        echo "==============================================="
        echo ""
        
        # Signal server launcher to start server with this config
        # (The server launcher script monitors these marker files)
        echo "${env}:${batch_size}" > "${CONFIG_FILE}"
        
        # Wait for server to be ready
        if ! wait_for_server "${SERVER_HOST}"; then
            echo "ERROR: Failed to connect to server for ${env} batch_size=${batch_size}"
            continue
        fi
        
        # Run experiments for all num_robots
        for num_robots in "${NUM_ROBOTS_LIST[@]}"; do
            output_dir="${BASE_OUTPUT_DIR}/batch_${batch_size}_robots_${num_robots}_${env_name}"
            
            run_client_experiment ${num_robots} ${env_name} ${batch_size} "${output_dir}"
            
            # Brief pause between experiments
            sleep 2
        done
        
        # Signal server to stop
        echo "STOP" > "${CONFIG_FILE}"
        sleep 5
    done
done

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "============================================"

# Cleanup marker file
rm -f "${CONFIG_FILE}"

