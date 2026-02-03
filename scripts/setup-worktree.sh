#!/usr/bin/env bash
set -e

echo "========================================="
echo "OpenPI Worktree Setup Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Step 1: Initialize submodules
echo -e "${BLUE}[1/4] Initializing git submodules...${NC}"
if [ -d "third_party/libero/.git" ] || [ -f "third_party/libero/.git" ]; then
    echo "Submodules already initialized, updating..."
    git submodule update --recursive
else
    echo "Initializing submodules for the first time..."
    git submodule update --init --recursive
fi
echo -e "${GREEN}✓ Submodules ready${NC}"
echo ""

# Step 2: Set up main Python environment
echo -e "${BLUE}[2/4] Setting up main Python environment (Python 3.11+)...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating main virtual environment..."
    uv venv .venv
fi

echo "Syncing dependencies with uv..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "Installing package in editable mode..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo -e "${GREEN}✓ Main environment ready${NC}"
echo ""

# Step 3: Set up libero example environment
echo -e "${BLUE}[3/4] Setting up LIBERO example environment (Python 3.8)...${NC}"
if [ ! -d "examples/libero/.venv" ]; then
    echo "Creating libero virtual environment with Python 3.8..."
    uv venv --python 3.8 examples/libero/.venv
fi

echo "Activating libero environment and installing dependencies..."
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

echo "Installing openpi-client and libero in editable mode..."
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

deactivate
echo -e "${GREEN}✓ LIBERO environment ready${NC}"
echo ""

# Step 4: Summary
echo -e "${BLUE}[4/4] Setup complete!${NC}"
echo ""
echo "========================================="
echo -e "${GREEN}✓ Worktree setup successful!${NC}"
echo "========================================="
echo ""
echo "To use the main environment:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo "To use the LIBERO example environment:"
echo -e "  ${YELLOW}source examples/libero/.venv/bin/activate${NC}"
echo ""
