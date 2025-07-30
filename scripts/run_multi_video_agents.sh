#!/bin/bash

# Script to run multiple video agent instances with different profiles
# This demonstrates how to run multiple video agents for comparison

echo "🚀 Starting Multiple Video Agent Instances"
echo "=" | head -c 60
echo ""

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ jq is required for JSON parsing. Please install it:"
    echo "   brew install jq  # on macOS"
    echo "   apt-get install jq  # on Ubuntu/Debian"
    exit 1
fi

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [profile1:port1] [profile2:port2] ..."
    echo ""
    echo "Example:"
    echo "  $0 frame_based_colpali:8001 direct_video_colqwen:8002"
    echo ""
    echo "Default: Runs two instances with frame_based_colpali on 8001 and direct_video_colqwen on 8002"
    exit 0
fi

# Get logs directory
LOG_DIR=$(uv run python -c "from src.utils.output_manager import get_output_manager; print(get_output_manager().get_logs_dir())" 2>/dev/null || echo "logs")
mkdir -p "$LOG_DIR"

# Default configurations if none provided
if [ $# -eq 0 ]; then
    CONFIGS=("frame_based_colpali:8001" "direct_video_colqwen:8002")
else
    CONFIGS=("$@")
fi

# Array to store PIDs
declare -a PIDS=()

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down all video agent instances..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null
    done
    echo "✅ All agents stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start each agent instance
for config in "${CONFIGS[@]}"; do
    # Split profile:port
    IFS=':' read -r profile port <<< "$config"
    
    if [ -z "$profile" ] || [ -z "$port" ]; then
        echo "❌ Invalid configuration: $config"
        echo "   Expected format: profile:port"
        continue
    fi
    
    echo "🎥 Starting Video Agent:"
    echo "   Profile: $profile"
    echo "   Port: $port"
    echo "   Log: $LOG_DIR/video_agent_${profile}_${port}.log"
    
    # Start the agent
    uv run python src/agents/video_agent_server.py --port "$port" --profile "$profile" > "$LOG_DIR/video_agent_${profile}_${port}.log" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "   PID: $PID"
    echo ""
done

# Wait for agents to start
echo "⏳ Waiting for agents to initialize..."
sleep 5

# Check each agent
echo "🔍 Checking agent status..."
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r profile port <<< "$config"
    
    if curl -s "http://localhost:$port/agent.json" > /dev/null 2>&1; then
        echo "✅ Agent on port $port ($profile) is running"
    else
        echo "❌ Agent on port $port ($profile) failed to start"
        echo "   Check $LOG_DIR/video_agent_${profile}_${port}.log for details"
    fi
done

echo ""
echo "✅ Multi-agent setup complete!"
echo ""
echo "📝 Configuration for composing agent:"
echo "   Update your config.json to point to different agents:"
echo "   - Frame-based agent: http://localhost:8001"
echo "   - Direct video agent: http://localhost:8002"
echo ""
echo "💡 To manually route queries, pass 'preferred_agent' in the query:"
echo "   {\"query\": \"your search\", \"preferred_agent\": \"http://localhost:8002\"}"
echo ""
echo "Press Ctrl+C to stop all agents..."

# Wait indefinitely
while true; do
    sleep 1
done