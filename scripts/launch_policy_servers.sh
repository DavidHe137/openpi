#!/bin/bash

# Launch policy servers on GPUs 0-7 for EIC GPUs!
# Each server will run on a different GPU and port

# Configuration
BASE_PORT=8000
ENV_MODE="LIBERO"
LOG_DIR="policy_logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to launch a policy server
launch_server() {
    local gpu_id=$1
    local port=$2
    local log_file="$LOG_DIR/policy_gpu${gpu_id}_port${port}.log"
    
    echo "Launching policy server on GPU $gpu_id, port $port..."
    echo "Log file: $log_file"
    
    # Launch the server in background with GPU assignment
    CUDA_VISIBLE_DEVICES=$gpu_id \
    uv run scripts/serve_policy.py \
        --env $ENV_MODE \
        --port $port \
        > "$log_file" 2>&1 &
    
    # Store the PID
    local pid=$!
    echo "Server PID: $pid"
    echo "$pid" > "$LOG_DIR/policy_gpu${gpu_id}.pid"
    
    # Wait a moment for the server to start
    sleep 2
    
    # Check if the process is still running
    if kill -0 $pid 2>/dev/null; then
        echo "✓ Policy server on GPU $gpu_id (port $port) started successfully"
    else
        echo "✗ Failed to start policy server on GPU $gpu_id"
        return 1
    fi
}

# Function to stop all servers
stop_servers() {
    echo "Stopping all policy servers..."
    for i in {0..7}; do
        pid_file="$LOG_DIR/policy_gpu${i}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "Stopping server on GPU $i (PID: $pid)"
                kill $pid
                # Wait a moment for graceful shutdown
                sleep 1
                # Force kill if still running
                if kill -0 $pid 2>/dev/null; then
                    echo "Force killing server on GPU $i (PID: $pid)"
                    kill -9 $pid
                fi
            fi
            rm -f "$pid_file"
        fi
    done
    echo "All servers stopped."
}

# Function to check server status
check_status() {
    echo "Checking server status..."
    for i in {0..7}; do
        port=$((BASE_PORT + i))
        pid_file="$LOG_DIR/policy_gpu${i}.pid"
        
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo "✓ GPU $i: Running (PID: $pid, Port: $port)"
            else
                echo "✗ GPU $i: Not running (stale PID file)"
            fi
        else
            echo "✗ GPU $i: No PID file found"
        fi
    done
}

# Handle command line arguments
case "${1:-start}" in
    "start")
        echo "Starting policy servers on GPUs 0-7..."
        echo "Environment: $ENV_MODE"
        echo "Base port: $BASE_PORT"
        echo "Log directory: $LOG_DIR"
        echo ""
        
        # Launch servers on GPUs 0-7
        for i in {0..7}; do
            port=$((BASE_PORT + i))
            launch_server $i $port
        done
        
        echo ""
        echo "All servers launched! Check status with: $0 status"
        echo "Stop servers with: $0 stop"
        ;;
    "stop")
        stop_servers
        ;;
    "status")
        check_status
        ;;
    "restart")
        stop_servers
        sleep 2
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Launch policy servers on GPUs 0-7 (default)"
        echo "  stop    - Stop all running policy servers"
        echo "  status  - Check status of all servers"
        echo "  restart - Stop and restart all servers"
        echo ""
        echo "Server configuration:"
        echo "  Environment: $ENV_MODE"
        echo "  Ports: $BASE_PORT-$((BASE_PORT + 7))"
        echo "  Logs: $LOG_DIR/"
        exit 1
        ;;
esac
