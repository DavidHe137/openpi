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
# setup_server_monitor: register cleanup trap and start a background watchdog
# for a server process. Must be called after the server is launched.
#
#   setup_server_monitor <server_pid>
#
# Sets globals: SERVER_JOB_PID, MONITOR_PID
# Defines:      cleanup (also registered as EXIT/INT/TERM trap)
setup_server_monitor() {
    SERVER_JOB_PID=$1

    cleanup() {
        echo "Cleaning up..."
        if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
            kill "$MONITOR_PID" 2>/dev/null || true
        fi
        if [ -n "$SERVER_JOB_PID" ] && kill -0 "$SERVER_JOB_PID" 2>/dev/null; then
            echo "Stopping server process (PID: $SERVER_JOB_PID)"
            kill "$SERVER_JOB_PID" 2>/dev/null || true
            sleep 2
            if kill -0 "$SERVER_JOB_PID" 2>/dev/null; then
                echo "Force killing server process"
                kill -9 "$SERVER_JOB_PID" 2>/dev/null || true
            fi
        fi
        echo "Cleanup complete"
    }
    trap cleanup EXIT INT TERM

    ( while sleep 5; do
        if ! kill -0 "$SERVER_JOB_PID" 2>/dev/null; then
            echo "ERROR: Server process (PID $SERVER_JOB_PID) died unexpectedly - terminating job"
            kill -TERM $$
            break
        fi
      done ) &
    MONITOR_PID=$!
}

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
