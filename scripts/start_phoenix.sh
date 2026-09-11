#!/bin/bash
#
# Start/stop/restart Phoenix server using Docker
#
# The container this data directory's start launched is recorded by id in
# $DATA_DIR/phoenix.cid; stop, status and logs act on that id only, never on
# another container holding the same name.
#

set -e

# Configuration
CONTAINER_NAME="${PHOENIX_CONTAINER_NAME:-phoenix-server}"
PORT="${PHOENIX_PORT:-6006}"
OTLP_PORT="${PHOENIX_OTLP_PORT:-4317}"
DATA_DIR="${PHOENIX_DATA_DIR:-./data/cogniverse/phoenix}"
IMAGE="${PHOENIX_IMAGE:-arizephoenix/phoenix:latest}"
CID_FILE="$DATA_DIR/phoenix.cid"

LABEL_ARGS=()
for label in ${PHOENIX_LABELS:-}; do
    LABEL_ARGS+=(--label "$label")
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure data directory exists
mkdir -p "$DATA_DIR"/{traces,datasets,experiments,evaluations}

# Functions
recorded_id() {
    local cid
    cid=$(cat "$CID_FILE" 2>/dev/null) || return 1
    [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || return 1
    echo "$cid"
}

recorded_running() {
    local cid
    cid=$(recorded_id) || return 1
    [ "$(docker ps -q --no-trunc -f id="$cid")" = "$cid" ]
}

start_phoenix() {
    echo -e "${GREEN}Starting Phoenix server...${NC}"

    if recorded_running; then
        echo -e "${YELLOW}Phoenix is already running${NC}"
        return
    fi

    # Clear this data directory's own earlier container, if any
    stop_phoenix

    # Start container with persistent volume
    docker run -d \
        --cidfile "$CID_FILE" \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:6006" \
        -p "${OTLP_PORT}:4317" \
        -v "$(realpath "$DATA_DIR"):/data" \
        -e PHOENIX_WORKING_DIR=/data \
        -e PHOENIX_ENABLE_PROMETHEUS=true \
        -e PHOENIX_ENABLE_CORS=true \
        -e PHOENIX_MAX_TRACES=100000 \
        --restart unless-stopped \
        "${LABEL_ARGS[@]}" \
        "$IMAGE" >/dev/null

    echo -e "${GREEN}Phoenix started on http://localhost:${PORT}${NC}"
    echo -e "${GREEN}Data directory: $DATA_DIR${NC}"
}

stop_phoenix() {
    local cid listed
    if ! cid=$(recorded_id); then
        rm -f "$CID_FILE"
        echo -e "${YELLOW}No Phoenix container recorded in $CID_FILE${NC}"
        return
    fi
    listed=$(docker ps -a -q --no-trunc -f id="$cid")
    if [ "$listed" != "$cid" ]; then
        rm -f "$CID_FILE"
        echo -e "${YELLOW}Phoenix container ${cid:0:12} recorded in $CID_FILE no longer exists${NC}"
        return
    fi
    echo -e "${GREEN}Stopping Phoenix server...${NC}"
    docker stop "$cid" >/dev/null
    docker rm "$cid" >/dev/null
    rm -f "$CID_FILE"
    echo -e "${GREEN}Phoenix stopped${NC}"
}

restart_phoenix() {
    stop_phoenix
    sleep 2
    start_phoenix
}

status_phoenix() {
    if recorded_running; then
        echo -e "${GREEN}Phoenix is running${NC}"
        echo "Container ID: $(recorded_id)"
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
    local cid
    if ! cid=$(recorded_id); then
        echo -e "${RED}No Phoenix container recorded in $CID_FILE${NC}"
        exit 1
    fi
    docker logs "$cid" "${@:2}"
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
        echo "  PHOENIX_PORT           - Port to expose (default: 6006)"
        echo "  PHOENIX_OTLP_PORT      - OTLP gRPC port to expose (default: 4317)"
        echo "  PHOENIX_DATA_DIR       - Data directory (default: ./data/cogniverse/phoenix)"
        echo "  PHOENIX_CONTAINER_NAME - Container name (default: phoenix-server)"
        echo "  PHOENIX_IMAGE          - Image (default: arizephoenix/phoenix:latest)"
        echo "  PHOENIX_LABELS         - Space-separated key=value container labels"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start Phoenix"
        echo "  $0 stop                     # Stop the container this data directory started"
        echo "  $0 logs -f                  # Follow logs"
        echo "  PHOENIX_PORT=8080 $0 start  # Start on port 8080"
        exit 1
        ;;
esac
