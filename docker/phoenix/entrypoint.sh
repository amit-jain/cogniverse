#!/bin/bash
set -e

# Initialize Phoenix data directory structure
init_phoenix() {
    echo "Initializing Phoenix data directories..."
    mkdir -p /data/traces /data/datasets /data/experiments /data/evaluations /data/logs
    
    # Create initial configuration if it doesn't exist
    if [ ! -f /data/phoenix_config.json ]; then
        echo '{
            "version": "1.0",
            "settings": {
                "max_traces": 100000,
                "retention_days": 30,
                "enable_prometheus": true,
                "enable_cors": true
            },
            "datasets": [],
            "experiments": []
        }' > /data/phoenix_config.json
        echo "Created initial Phoenix configuration"
    fi
    
    echo "Phoenix initialization complete"
}

# Start Phoenix server
start_phoenix() {
    echo "Starting Phoenix server..."
    echo "Data directory: ${PHOENIX_WORKING_DIR}"
    echo "Port: ${PHOENIX_PORT}"
    echo "Host: ${PHOENIX_HOST}"
    
    exec python -m phoenix.server.main serve \
        --port ${PHOENIX_PORT} \
        --host ${PHOENIX_HOST}
}

# Main entrypoint
main() {
    case "$1" in
        serve)
            init_phoenix
            start_phoenix
            ;;
        init)
            init_phoenix
            ;;
        *)
            echo "Usage: $0 {serve|init}"
            exit 1
            ;;
    esac
}

main "$@"