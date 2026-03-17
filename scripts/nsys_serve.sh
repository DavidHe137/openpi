#!/bin/bash
# Launch the policy server under nsys for GPU profiling.
#
# Usage:
#   ./scripts/nsys_serve.sh [nsys options] -- [serve_policy.py options]
#
# Examples:
#   # Basic — profile everything, output to profile.nsys-rep
#   ./scripts/nsys_serve.sh
#
#   # Custom output name and extra serve args
#   ./scripts/nsys_serve.sh -o my_profile -- --port 8080 --max_batch_size 8
#
#   # Limit capture duration (seconds) so you don't need to wait for shutdown
#   ./scripts/nsys_serve.sh --duration 60
#
# After profiling:
#   nsys-ui profile.nsys-rep
#
# Notes:
#   - The GPU worker runs in a forked child process.  --trace-fork-before-exec
#     ensures nsys instruments it before its first CUDA call.
#   - NVTX ranges from engine.py and pi0.py are visible under the
#     "NVTX" row in the nsys timeline.
#   - The warmup / batch-profiling phase runs before "GPU worker ready" is
#     logged.  In the timeline, look for the first repeating infer_batch range
#     to find steady-state inference.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
OUTPUT="${NSYS_OUTPUT:-profile}"
SERVE_ARGS="--env LIBERO --max-batch-size 4 --port 8080 --scheduling-algorithm round_robin"

# ── split args on '--' ────────────────────────────────────────────────────────
# Everything before '--' is passed to nsys; everything after is passed to serve_policy.py.
NSYS_EXTRA_ARGS=()
found_sep=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        found_sep=1
        continue
    fi
    if [[ $found_sep -eq 0 ]]; then
        NSYS_EXTRA_ARGS+=("$arg")
    else
        SERVE_ARGS="$SERVE_ARGS $arg"
    fi
done

# ── run ───────────────────────────────────────────────────────────────────────
exec nsys profile \
    --trace=cuda,nvtx,osrt \
    --trace-fork-before-exec=true \
    --output="$OUTPUT" \
    --force-overwrite=true \
    "${NSYS_EXTRA_ARGS[@]}" \
    uv run python scripts/serve_policy.py $SERVE_ARGS
