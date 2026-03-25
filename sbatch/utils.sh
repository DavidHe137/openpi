#!/bin/bash
# sbatch/utils.sh — shared utilities for SLURM batch scripts
#
# Usage: source sbatch/utils.sh

# find_free_port: find a free TCP port via random search
#
#   PORT=$(find_free_port)                  # random in [8000, 9000]
#   PORT=$(find_free_port 8000 9000)        # explicit range
#   PORT=$(find_free_port 8000 9000 42)     # fixed seed (reproducible)
#
# Exits with status 1 and prints an error to stderr if no free port is found.
find_free_port() {
    local lo=${1:-8000}
    local hi=${2:-9000}
    local seed=${3:-$RANDOM}

    python3 - <<EOF
import socket, random, sys

random.seed($seed)
lo, hi = $lo, $hi
candidates = list(range(lo, hi + 1))
random.shuffle(candidates)

for port in candidates:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(('', port))
        s.close()
        print(port)
        sys.exit(0)
    except OSError:
        pass

print(f"ERROR: no free port found in range [{lo}, {hi}]", file=sys.stderr)
sys.exit(1)
EOF
}
