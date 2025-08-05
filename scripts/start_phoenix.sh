#!/bin/bash
#
# Start/stop/restart Phoenix server using Docker
#

set -e

# Configuration
CONTAINER_NAME="phoenix-server"
PORT="${PHOENIX_PORT:-6006}"
DATA_DIR="${PHOENIX_DATA_DIR:-./data/cogniverse/phoenix}"
IMAGE="arizephoenix/phoenix:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure data directory exists
mkdir -p "$DATA_DIR"/{traces,datasets,experiments,evaluations}

# Functions
start_phoenix() {
    echo -e "${GREEN}Starting Phoenix server...${NC}"
    
    # Check if already running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        echo -e "${YELLOW}Phoenix is already running${NC}"
        return
    fi
    
    # Remove old container if exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    
    # Start container with persistent volume
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:6006" \
        -v "$(realpath "$DATA_DIR"):/data" \
        -e PHOENIX_WORKING_DIR=/data \
        -e PHOENIX_ENABLE_PROMETHEUS=true \
        -e PHOENIX_ENABLE_CORS=true \
        -e PHOENIX_MAX_TRACES=100000 \
        --restart unless-stopped \
        "$IMAGE"
    
    echo -e "${GREEN}Phoenix started on http://localhost:${PORT}${NC}"
    echo -e "${GREEN}Data directory: $DATA_DIR${NC}"
}

stop_phoenix() {
    echo -e "${GREEN}Stopping Phoenix server...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || echo -e "${YELLOW}Phoenix not running${NC}"
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}Phoenix stopped${NC}"
}

restart_phoenix() {
    stop_phoenix
    sleep 2
    start_phoenix
}

status_phoenix() {
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        echo -e "${GREEN}Phoenix is running${NC}"
        echo "Container ID: $(docker ps -q -f name="$CONTAINER_NAME")"
        echo "URL: http://localhost:${PORT}"
        echo "Data directory: $DATA_DIR"
        
        # Try to get trace count
        if command -v curl &> /dev/null; then
            TRACES=$(curl -s "http://localhost:${PORT}/api/v1/traces/count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d: -f2)
            [ -n "$TRACES" ] && echo "Traces: $TRACES"
        fi
    else
        echo -e "${RED}Phoenix is not running${NC}"
    fi
}

logs_phoenix() {
    docker logs "$CONTAINER_NAME" "${@:2}"
}

# Main
case "$1" in
    start)
        start_phoenix
        ;;
    stop)
        stop_phoenix
        ;;
    restart)
        restart_phoenix
        ;;
    status)
        status_phoenix
        ;;
    logs)
        logs_phoenix "$@"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Environment variables:"
        echo "  PHOENIX_PORT     - Port to expose (default: 6006)"
        echo "  PHOENIX_DATA_DIR - Data directory (default: ./data/cogniverse/phoenix)"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start Phoenix"
        echo "  $0 stop                     # Stop Phoenix"
        echo "  $0 logs -f                  # Follow logs"
        echo "  PHOENIX_PORT=8080 $0 start  # Start on port 8080"
        exit 1
        ;;
esac