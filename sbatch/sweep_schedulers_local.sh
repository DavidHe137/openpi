#!/bin/bash

set -e

SCHEDULER=greedy
NUM_ROBOTS_LIST=(20 15 10 5)
NUM_TRIALS_PER_TASK=6
PORT=8080
NODE=zhurong

NUM_RUNS=3
for NUM_ROBOTS in "${NUM_ROBOTS_LIST[@]}"; do
    for RUN_IDX in $(seq 0 $((NUM_RUNS - 1))); do
        OUTPUT_DIR="data/libero/sweep_schedulers/scheduler_${SCHEDULER}_num_robots_${NUM_ROBOTS}_run_${RUN_IDX}"
        echo "--------------------------------------"
        echo "Running: scheduler=$SCHEDULER  num_robots=$NUM_ROBOTS  run=$RUN_IDX"
        echo "Output: $OUTPUT_DIR"
        echo "--------------------------------------"

        source scripts/libero_client.sh
        ./examples/libero/.venv/bin/python examples/libero/main_multi_robot_runtime.py \
            --host $NODE \
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
    done
done