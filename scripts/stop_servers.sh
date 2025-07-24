#!/bin/bash

# Stop Multi-Agent RAG System Servers
# This script stops all agent servers but keeps Vespa running

echo "🛑 Stopping Multi-Agent RAG System Servers"
echo "=" * 60

# Function to stop a process by name and display status
stop_process() {
    local process_name="$1"
    local display_name="$2"
    
    if pgrep -f "$process_name" > /dev/null; then
        echo "🔄 Stopping $display_name..."
        pkill -f "$process_name"
        sleep 2
        
        # Check if process was stopped
        if ! pgrep -f "$process_name" > /dev/null; then
            echo "✅ $display_name stopped successfully"
        else
            echo "⚠️  $display_name may still be running, attempting force kill..."
            pkill -9 -f "$process_name"
            sleep 1
            if ! pgrep -f "$process_name" > /dev/null; then
                echo "✅ $display_name force-stopped"
            else
                echo "❌ Failed to stop $display_name"
            fi
        fi
    else
        echo "⚪ $display_name is not running"
    fi
}

# Stop agent servers
stop_process "uvicorn src.agents.video_agent_server" "Video Search Agent"
stop_process "python src/tools/video_file_server.py" "Video File Server"
stop_process "uvicorn src.agents.text_agent_server" "Text Search Agent"
stop_process "adk web" "ADK Web Interface"

echo ""
echo "🔍 Checking remaining processes..."

# Check if any agent processes are still running
REMAINING_PROCESSES=$(ps aux | grep -E "(uvicorn.*agent|video_file_server|adk web)" | grep -v grep | wc -l)

if [ "$REMAINING_PROCESSES" -eq 0 ]; then
    echo "✅ All agent servers stopped successfully"
else
    echo "⚠️  Some processes may still be running:"
    ps aux | grep -E "(uvicorn.*agent|video_file_server|adk web)" | grep -v grep
fi

echo ""
echo "🐳 Vespa Status:"
if docker ps --format '{{.Names}}' | grep -q "^vespa$"; then
    echo "✅ Vespa container is still running (as intended)"
    echo "📊 Access Vespa at: http://localhost:8080"
else
    echo "⚠️  Vespa container is not running"
fi

echo ""
echo "🎯 To restart the system: ./scripts/run_servers.sh"
echo "🛑 To stop Vespa: docker stop vespa"
LOG_DIR=$(python -c "from src.utils.output_manager import get_output_manager; print(get_output_manager().get_logs_dir())" 2>/dev/null || echo "logs")
echo "📜 To view logs: ls -la $LOG_DIR/"