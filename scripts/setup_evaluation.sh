#!/bin/bash
#
# Setup script for Cogniverse Evaluation Framework
# This script sets up Phoenix and prepares the evaluation environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PHOENIX_PORT=${PHOENIX_PORT:-6006}
PHOENIX_DATA_DIR=${PHOENIX_DATA_DIR:-"./data/phoenix"}
USE_DOCKER=${USE_DOCKER:-false}

echo -e "${GREEN}Cogniverse Evaluation Framework Setup${NC}"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
port_available() {
    ! lsof -i:$1 >/dev/null 2>&1
}

# Step 1: Check prerequisites
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists uv; then
    echo -e "${YELLOW}Warning: uv is not installed. Installing with pip...${NC}"
    pip install uv
fi

echo -e "${GREEN}✓ Prerequisites checked${NC}"

# Step 2: Install dependencies
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"

if [ -f "pyproject.toml" ]; then
    echo "Installing from pyproject.toml..."
    uv sync
else
    echo "Installing individual packages..."
    uv pip install inspect-ai arize-phoenix phoenix-evals opentelemetry-api opentelemetry-sdk
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Create data directories
echo -e "\n${YELLOW}Step 3: Creating data directories...${NC}"

mkdir -p "$PHOENIX_DATA_DIR"/{traces,datasets,experiments,evaluations,logs}
mkdir -p data/eval
mkdir -p outputs/{evaluations,reports,inspect_logs}

echo -e "${GREEN}✓ Data directories created${NC}"

# Step 4: Start Phoenix
echo -e "\n${YELLOW}Step 4: Starting Phoenix server...${NC}"

if [ "$USE_DOCKER" = true ]; then
    # Use Docker
    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    echo "Starting Phoenix with Docker..."
    docker-compose -f docker/phoenix-compose.yml up -d
    
    # Wait for Phoenix to be ready
    echo "Waiting for Phoenix to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:$PHOENIX_PORT/health >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
else
    # Use local Python
    if ! port_available $PHOENIX_PORT; then
        echo -e "${YELLOW}Port $PHOENIX_PORT is already in use. Phoenix might already be running.${NC}"
        echo "Checking Phoenix health..."
        if curl -s http://localhost:$PHOENIX_PORT/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Phoenix is already running${NC}"
        else
            echo -e "${RED}Port $PHOENIX_PORT is in use but Phoenix is not responding${NC}"
            exit 1
        fi
    else
        echo "Starting Phoenix server in background..."
        python scripts/start_phoenix.py start --background --data-dir "$PHOENIX_DATA_DIR" --port $PHOENIX_PORT
        
        # Wait for Phoenix to be ready
        echo "Waiting for Phoenix to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:$PHOENIX_PORT/health >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
    fi
fi

# Verify Phoenix is running
if curl -s http://localhost:$PHOENIX_PORT/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Phoenix is running at http://localhost:$PHOENIX_PORT${NC}"
else
    echo -e "${RED}Error: Phoenix failed to start${NC}"
    exit 1
fi

# Step 5: Create sample evaluation dataset
echo -e "\n${YELLOW}Step 5: Creating sample evaluation dataset...${NC}"

cat > data/eval/sample_queries.json << 'EOF'
[
    {
        "query": "person wearing winter clothes outdoors",
        "expected_videos": ["v_-IMXSEIabMM"],
        "category": "action"
    },
    {
        "query": "industrial machinery with metallic surfaces",
        "expected_videos": ["elephant_dream_clip"],
        "category": "scene"
    },
    {
        "query": "animated character or cartoon style visuals",
        "expected_videos": ["big_buck_bunny_clip"],
        "category": "style"
    }
]
EOF

echo -e "${GREEN}✓ Sample dataset created${NC}"

# Step 6: Run quick test
echo -e "\n${YELLOW}Step 6: Running quick test evaluation...${NC}"

uv run python scripts/run_experiments_with_visualization.py --dataset-name sample_eval --profiles frame_based_colpali --strategies binary_binary --max-queries 3

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test evaluation completed successfully${NC}"
else
    echo -e "${YELLOW}Warning: Test evaluation failed. Please check the logs.${NC}"
fi

# Print summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluation Framework Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Phoenix Dashboard: http://localhost:$PHOENIX_PORT"
echo "Data Directory: $PHOENIX_DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. View Phoenix dashboard: http://localhost:$PHOENIX_PORT"
echo "  2. Run full evaluation: uv run python scripts/run_experiments_with_visualization.py --dataset-name sample_eval"
echo "  3. Check documentation: docs/EVALUATION_FRAMEWORK.md"
echo ""
echo "To stop Phoenix:"
echo "  - Local: python scripts/start_phoenix.py stop"
echo "  - Docker: docker-compose -f docker/phoenix-compose.yml down"