#!/bin/bash
#SBATCH --job-name=pi0_server
#SBATCH --partition=rl2-lab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
#SBATCH --exclude=dynamics
#SBATCH --chdir=/srv/rl2-lab/flash8/rbansal66/vvla/openpi

# This is the SERVER script that runs on l40s
# It monitors for configuration changes and restarts the server accordingly

set -e

# Get the client job ID from argument
CLIENT_JOB_ID=$1

if [ -z "${CLIENT_JOB_ID}" ]; then
    echo "ERROR: CLIENT_JOB_ID not provided"
    exit 1
fi

# Use shared filesystem instead of /tmp (which is node-local)
CONFIG_DIR="/srv/rl2-lab/flash8/rbansal66/vvla/openpi/tmp"
mkdir -p "${CONFIG_DIR}"
CONFIG_FILE="${CONFIG_DIR}/pi0_server_config_${CLIENT_JOB_ID}.txt"
SERVER_PID=""

echo "============================================"
echo "Starting policy server launcher"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Client job ID: ${CLIENT_JOB_ID}"
echo "Config file: ${CONFIG_FILE}"
echo "============================================"

# Create logs directory
mkdir -p logs

# Function to start server
start_server() {
    local env=$1
    local batch_size=$2
    
    echo ""
    echo "========================================"
    echo "Starting server:"
    echo "  Environment: ${env}"
    echo "  Batch size: ${batch_size}"
    echo "  Time: $(date)"
    echo "========================================"
    
    # Kill any existing server
    if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "Killing existing server (PID: ${SERVER_PID})"
        kill ${SERVER_PID}
        wait ${SERVER_PID} 2>/dev/null || true
        sleep 5
    fi
    
    # Start new server
    uv run scripts/serve_policy.py --env ${env} --batch_size ${batch_size} > logs/server_${SLURM_JOB_ID}_${env}_${batch_size}.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server started with PID: ${SERVER_PID}"
    echo "  Logs: logs/server_${SLURM_JOB_ID}_${env}_${batch_size}.log"
    
    # Wait for server to initialize
    echo "Waiting for server initialization..."
    sleep 45  # Longer initial wait for model loading
    
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "ERROR: Server process died immediately"
        echo "Check logs: logs/server_${SLURM_JOB_ID}_${env}_${batch_size}.log"
        return 1
    fi
    
    echo "Server is running"
    return 0
}

# Function to stop server
stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 ${SERVER_PID} 2>/dev/null; then
        echo ""
        echo "Stopping server (PID: ${SERVER_PID})"
        kill ${SERVER_PID}
        wait ${SERVER_PID} 2>/dev/null || true
        echo "Server stopped"
        SERVER_PID=""
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    stop_server
    rm -f "${CONFIG_FILE}"
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# Initialize config file
echo "WAIT" > "${CONFIG_FILE}"

# Main loop - monitor config file and start/stop server as needed
CURRENT_CONFIG=""

while true; do
    # Check if config file exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "Config file disappeared, exiting"
        break
    fi
    
    # Read current config
    NEW_CONFIG=$(cat "${CONFIG_FILE}" 2>/dev/null || echo "")
    
    # Check if config changed
    if [ "${NEW_CONFIG}" != "${CURRENT_CONFIG}" ]; then
        echo ""
        echo "Config changed: '${CURRENT_CONFIG}' -> '${NEW_CONFIG}'"
        
        if [ "${NEW_CONFIG}" = "STOP" ]; then
            stop_server
            CURRENT_CONFIG="${NEW_CONFIG}"
            echo "Server stopped, waiting for new config"
            
        elif [ "${NEW_CONFIG}" = "EXIT" ]; then
            echo "Received EXIT signal"
            break
            
        elif [ "${NEW_CONFIG}" = "WAIT" ]; then
            # Just waiting, update current config to avoid spam
            CURRENT_CONFIG="${NEW_CONFIG}"
            echo "Waiting for configuration..."
            
        elif [ -n "${NEW_CONFIG}" ]; then
            # Parse config: ENV:BATCH_SIZE
            IFS=':' read -r env batch_size <<< "${NEW_CONFIG}"
            
            if [ -n "${env}" ] && [ -n "${batch_size}" ]; then
                start_server "${env}" "${batch_size}"
                CURRENT_CONFIG="${NEW_CONFIG}"
            else
                echo "WARNING: Invalid config format: ${NEW_CONFIG}"
                CURRENT_CONFIG="${NEW_CONFIG}"  # Update anyway to avoid spam
            fi
        fi
    fi
    
    # Check if server is still running
    if [ -n "${SERVER_PID}" ] && ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "WARNING: Server process died unexpectedly"
        SERVER_PID=""
    fi
    
    # Sleep before next check
    sleep 5
done

echo ""
echo "============================================"
echo "Server launcher exiting"
echo "============================================"

