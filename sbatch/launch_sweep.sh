#!/bin/bash
# Submit a grid of heterogeneous sweep jobs: 3 schedulers x 5 batch sizes = 15 jobs
# Each job runs server (l40s + 4 CPUs) and client (a40 + 20 CPUs) as separate het components.
#
# Usage: bash sbatch/launch_sweep.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

SCHEDULERS=(fixed-size-greedy greedy-deadline greedy-action)
BATCH_SIZES=(1 2 3 4 5)

mkdir -p "$PROJECT_ROOT/logs"

for SCHEDULER in "${SCHEDULERS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        JOB_ID=$(sbatch --parsable --chdir "$PROJECT_ROOT" "$PROJECT_ROOT/sbatch/het_sweep_job.sh" "$SCHEDULER" "$BATCH_SIZE")
        echo "Submitted job $JOB_ID: scheduler=$SCHEDULER  max_batch=$BATCH_SIZE"
    done
done
