#!/bin/bash

# Master script to launch the full batch sweep experiment
# This script submits both server (l40s) and client (a40) jobs

set -e

echo "============================================"
echo "Launching batch sweep experiment"
echo "Date: $(date)"
echo "============================================"

# Create logs directory
mkdir -p logs

# Generate a unique ID for coordinating server/client
SWEEP_ID="sweep_$$_$(date +%s)"
echo "Sweep coordination ID: ${SWEEP_ID}"

# Step 1: Submit the client job first (hold it)
echo ""
echo "Step 1: Submitting client job (a40) in HELD state..."
CLIENT_JOB_ID=$(sbatch --parsable --hold scripts/realtime_batching_sweep/benchmark_batching_sweep.sh)

if [ -z "${CLIENT_JOB_ID}" ]; then
    echo "ERROR: Failed to submit client job"
    exit 1
fi

echo "Client job submitted (held): ${CLIENT_JOB_ID}"

# Step 2: Submit the server job with the client job ID
echo ""
echo "Step 2: Submitting server job (l40s) with client ID..."
SERVER_JOB_ID=$(sbatch --parsable scripts/realtime_batching_sweep/launch_batch_sweep_server.sh ${CLIENT_JOB_ID})

if [ -z "${SERVER_JOB_ID}" ]; then
    echo "ERROR: Failed to submit server job"
    scancel ${CLIENT_JOB_ID}
    exit 1
fi

echo "Server job submitted: ${SERVER_JOB_ID}"

# Step 3: Wait for server to get a node allocation
echo ""
echo "Step 3: Waiting for server to allocate resources..."
MAX_WAIT=300  # 5 minutes
ELAPSED=0
SERVER_NODE=""

while [ $ELAPSED -lt $MAX_WAIT ]; do
    SERVER_NODE=$(squeue -j ${SERVER_JOB_ID} -h -o "%N" 2>/dev/null || echo "")
    if [ -n "${SERVER_NODE}" ] && [ "${SERVER_NODE}" != "(None)" ]; then
        echo "Server allocated to node: ${SERVER_NODE}"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ -z "${SERVER_NODE}" ] || [ "${SERVER_NODE}" = "(None)" ]; then
    echo "ERROR: Server failed to allocate within ${MAX_WAIT}s"
    scancel ${SERVER_JOB_ID} ${CLIENT_JOB_ID}
    exit 1
fi

# Step 4: Release the client job with server info
echo ""
echo "Step 4: Releasing client job with server info..."
scontrol update JobId=${CLIENT_JOB_ID} Dependency= ExcNodeList=${SERVER_NODE}
export SERVER_HOST=${SERVER_NODE}
scontrol update JobId=${CLIENT_JOB_ID} WorkDir=$(pwd)
scontrol release ${CLIENT_JOB_ID}

echo "Client job released: ${CLIENT_JOB_ID}"

echo ""
echo "============================================"
echo "Jobs submitted successfully!"
echo "============================================"
echo "Server job ID: ${SERVER_JOB_ID}"
echo "  Node: ${SERVER_NODE}"
echo "  Partition: rl2-lab (l40s)"
echo ""
echo "Client job ID: ${CLIENT_JOB_ID}"
echo "  Partition: rl2-lab (a40)"
echo "  Excluded node: ${SERVER_NODE}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/server_${SERVER_JOB_ID}.out"
echo "  tail -f logs/batch_sweep_${CLIENT_JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel ${SERVER_JOB_ID} ${CLIENT_JOB_ID}"
echo "============================================"

# Create a status file for easy reference
STATUS_FILE="logs/sweep_status_${SERVER_JOB_ID}_${CLIENT_JOB_ID}.txt"
cat > "${STATUS_FILE}" <<EOF
Batch Sweep Experiment
Started: $(date)

Server Job: ${SERVER_JOB_ID}
  Node: ${SERVER_NODE}
  GPU: l40s
  Log: logs/server_${SERVER_JOB_ID}.out

Client Job: ${CLIENT_JOB_ID}
  GPU: a40
  Log: logs/batch_sweep_${CLIENT_JOB_ID}.out

Cancel command:
  scancel ${SERVER_JOB_ID} ${CLIENT_JOB_ID}
EOF

echo ""
echo "Status saved to: ${STATUS_FILE}"

