#!/bin/bash

# Enhanced Multi-Agent RAG System Server Launcher
# Reads configuration from config.json and validates services

echo "🚀 Starting Multi-Agent RAG System Servers (Enhanced)"
echo "=" | head -c 60
echo ""

# Parse command line arguments
BACKEND=${1:-""}
HELP_FLAG=""
CHECK_INFERENCE=""

for arg in "$@"; do
    case $arg in
        --help|-h)
            HELP_FLAG="true"
            ;;
        --check-inference)
            CHECK_INFERENCE="true"
            ;;
        vespa)
            BACKEND="$arg"
            ;;
    esac
done

if [ "$HELP_FLAG" = "true" ]; then
    echo "Usage: $0 [backend] [options]"
    echo ""
    echo "Backend:"
    echo "  vespa     - Use Vespa backend (default)"
    echo "  (auto)    - Auto-detect from config.json"
    echo ""
    echo "Options:"
    echo "  --help, -h         Show this help message"
    echo "  --check-inference  Check inference service configuration"
    echo ""
    echo "Examples:"
    echo "  $0                  # Auto-detect backend from config"
    echo "  $0 vespa            # Force Vespa backend"
    echo "  $0 --check-inference # Check inference setup"
    exit 0
fi

# Check if jq is installed for JSON parsing
if ! command -v jq &> /dev/null; then
    echo "❌ jq is required for JSON parsing. Please install it:"
    echo "   brew install jq  # on macOS"
    echo "   apt-get install jq  # on Ubuntu/Debian"
    exit 1
fi

# Read configuration from config.json
CONFIG_FILE="configs/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ config.json not found!"
    echo "   Please copy configs/examples/config.example.json to configs/config.json and configure it."
    exit 1
fi

# Auto-detect backend if not specified
if [ -z "$BACKEND" ]; then
    BACKEND=$(jq -r '.search_backend // "vespa"' "$CONFIG_FILE")
    echo "📋 Auto-detected backend from config: $BACKEND"
fi

export SEARCH_BACKEND="$BACKEND"

# Read inference configuration
INFERENCE_PROVIDER=$(jq -r '.inference.provider // "local"' "$CONFIG_FILE")
INFERENCE_ENDPOINT=$(jq -r '.inference.local_endpoint // "http://localhost:11434"' "$CONFIG_FILE")
INFERENCE_MODEL=$(jq -r '.inference.model // "gemma2:2b"' "$CONFIG_FILE")

echo "📋 Inference configuration:"
echo "   Provider: $INFERENCE_PROVIDER"
if [ "$INFERENCE_PROVIDER" = "local" ]; then
    echo "   Endpoint: $INFERENCE_ENDPOINT"
    echo "   Model: $INFERENCE_MODEL"
elif [ "$INFERENCE_PROVIDER" = "modal" ]; then
    MODAL_ENDPOINT=$(jq -r '.inference.modal_endpoint // ""' "$CONFIG_FILE")
    echo "   Endpoint: $MODAL_ENDPOINT"
fi
echo ""

# Check inference service if requested
if [ "$CHECK_INFERENCE" = "true" ]; then
    echo "🔍 Checking inference service..."
    
    if [ "$INFERENCE_PROVIDER" = "local" ]; then
        if curl -s "$INFERENCE_ENDPOINT/api/tags" > /dev/null 2>&1; then
            echo "✅ Ollama is healthy at $INFERENCE_ENDPOINT"
            # Check if model exists
            MODEL_EXISTS=$(curl -s "$INFERENCE_ENDPOINT/api/tags" | jq -r ".models[] | select(.name | startswith(\"$INFERENCE_MODEL\")) | .name" | head -1)
            if [ -n "$MODEL_EXISTS" ]; then
                echo "✅ Model $INFERENCE_MODEL is available"
                # Test a simple generation
                echo "Testing inference with model $INFERENCE_MODEL..."
                RESPONSE=$(curl -s -X POST "$INFERENCE_ENDPOINT/api/generate" \
                    -d "{\"model\": \"$INFERENCE_MODEL\", \"prompt\": \"Respond with just the word 'healthy'\", \"stream\": false}")
                if [ $? -eq 0 ]; then
                    echo "✅ Inference test successful"
                else
                    echo "❌ Inference test failed"
                fi
            else
                echo "❌ Model $INFERENCE_MODEL not found"
                exit 1
            fi
        else
            echo "❌ Ollama is not running at $INFERENCE_ENDPOINT"
            exit 1
        fi
    elif [ "$INFERENCE_PROVIDER" = "modal" ]; then
        MODAL_ENDPOINT=$(jq -r '.inference.modal_endpoint // ""' "$CONFIG_FILE")
        if [ -n "$MODAL_ENDPOINT" ]; then
            # Check if Modal secret exists
            if ! modal secret list 2>/dev/null | grep -q "huggingface-token"; then
                echo "⚠️  Modal secret 'huggingface-token' not found"
                # Try to create it from environment
                if [ -n "$HF_TOKEN" ] || [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
                    HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
                    echo "📝 Creating Modal secret..."
                    if modal secret create huggingface-token HF_TOKEN="$HF_TOKEN" 2>/dev/null; then
                        echo "✅ Modal secret created successfully"
                    else
                        echo "❌ Failed to create Modal secret"
                    fi
                else
                    echo "❌ No HF_TOKEN found in environment"
                fi
            fi
            
            HEALTH_URL=$(echo "$MODAL_ENDPOINT" | sed 's|/generate$|/health|')
            if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
                echo "✅ Modal inference service is healthy"
            else
                echo "❌ Modal inference service not reachable"
                exit 1
            fi
        else
            echo "❌ Modal endpoint not configured"
            exit 1
        fi
    fi
    exit 0
fi

# Backend-specific setup
if [ "$BACKEND" = "vespa" ]; then
    export VESPA_URL=$(jq -r '.vespa_url // "http://localhost"' "$CONFIG_FILE")
    export VESPA_PORT=$(jq -r '.vespa_port // 8080' "$CONFIG_FILE")
    
    # Check if Vespa is running, start if not
    if ! curl -s "$VESPA_URL:$VESPA_PORT/ApplicationStatus" > /dev/null 2>&1; then
        echo "⚠️  Vespa not running, starting with persistent storage..."
        ./scripts/start_vespa.sh
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start Vespa!"
            exit 1
        fi
    fi
    echo "✅ Vespa is running"
    
    # Check if we have ingested data
    DOC_COUNT=$(curl -s "$VESPA_URL:$VESPA_PORT/search/?yql=select%20*%20from%20video_frame%20where%20true&hits=0" | jq -r '.root.fields.totalCount // 0' 2>/dev/null || echo "0")
    if [ "$DOC_COUNT" -eq 0 ]; then
        echo "⚠️  WARNING: No documents in Vespa!"
        echo "   The system will start but search won't return results."
        echo "   To add documents run: python scripts/run_ingestion.py --video_dir data/videos --backend vespa"
    else
        echo "✅ Vespa has $DOC_COUNT documents indexed"
    fi
    
else
    echo "❌ Unknown backend: $BACKEND"
    echo "   Use 'vespa'"
    exit 1
fi

# Check inference service health
echo ""
echo "🔍 Checking inference service health..."
if [ "$INFERENCE_PROVIDER" = "local" ]; then
    # Check if Ollama is running
    if curl -s "$INFERENCE_ENDPOINT/api/tags" > /dev/null 2>&1; then
        echo "✅ Ollama is running at $INFERENCE_ENDPOINT"
        
        # Check if the model is available
        MODEL_EXISTS=$(curl -s "$INFERENCE_ENDPOINT/api/tags" | jq -r ".models[] | select(.name | startswith(\"$INFERENCE_MODEL\")) | .name" | head -1)
        if [ -n "$MODEL_EXISTS" ]; then
            echo "✅ Model $INFERENCE_MODEL is available"
        else
            echo "⚠️  Model $INFERENCE_MODEL not found. Available models:"
            curl -s "$INFERENCE_ENDPOINT/api/tags" | jq -r '.models[].name' | sed 's/^/     - /'
            echo ""
            echo "   To download the model, run:"
            echo "   ollama pull $INFERENCE_MODEL"
        fi
    else
        echo "❌ Ollama is not running at $INFERENCE_ENDPOINT"
        echo "   Please start Ollama first:"
        echo "   ollama serve"
    fi
elif [ "$INFERENCE_PROVIDER" = "modal" ]; then
    MODAL_ENDPOINT=$(jq -r '.inference.modal_endpoint // ""' "$CONFIG_FILE")
    if [ -n "$MODAL_ENDPOINT" ]; then
        # Extract health endpoint from modal endpoint
        HEALTH_URL=$(echo "$MODAL_ENDPOINT" | sed 's|/generate$|/health|')
        if curl -s "$HEALTH_URL" > /dev/null 2>&1; then
            echo "✅ Modal inference service is healthy"
        else
            echo "⚠️  Modal inference service not reachable at $MODAL_ENDPOINT"
            echo "   Deploy it with: modal deploy src/inference/modal_inference_service.py"
        fi
    else
        echo "❌ Modal endpoint not configured in config.json"
    fi
fi

echo ""
echo "Using video search backend: $SEARCH_BACKEND"
echo ""

# Get logs directory from output manager or use default
LOG_DIR=$(uv run python -c "from src.utils.output_manager import get_output_manager; print(get_output_manager().get_logs_dir())" 2>/dev/null || echo "logs")

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start the Video File Server in the background
STATIC_PORT=$(jq -r '.static_server_port // 8888' "$CONFIG_FILE")
echo "📁 Starting Video File Server on port $STATIC_PORT..."
uv run python src/tools/video_file_server.py > "$LOG_DIR/video_file_server.log" 2>&1 &
VIDEO_FILE_SERVER_PID=$!

# Start the Video Search Agent in the background
VIDEO_PORT=$(jq -r '.video_agent_url' "$CONFIG_FILE" | grep -oE '[0-9]+$' || echo "8001")
echo "🎥 Starting Video Search Agent on port $VIDEO_PORT..."
# Get active profile if set
ACTIVE_PROFILE=$(jq -r '.active_profile // ""' "$CONFIG_FILE")
if [ -n "$ACTIVE_PROFILE" ]; then
    echo "   Using profile: $ACTIVE_PROFILE"
    uv run python src/agents/video_agent_server.py --port "$VIDEO_PORT" --profile "$ACTIVE_PROFILE" > "$LOG_DIR/video_search_agent.log" 2>&1 &
else
    uv run python src/agents/video_agent_server.py --port "$VIDEO_PORT" > "$LOG_DIR/video_search_agent.log" 2>&1 &
fi
VIDEO_AGENT_PID=$!

# Text Search Agent - commented out until Elasticsearch is configured
# TEXT_PORT=$(jq -r '.text_agent_url' "$CONFIG_FILE" | grep -oE '[0-9]+$' || echo "8002")
# echo "📄 Starting Text Search Agent on port $TEXT_PORT..."
# uvicorn src.agents.text_agent_server:app --host 0.0.0.0 --port "$TEXT_PORT" > "$LOG_DIR/text_search_agent.log" 2>&1 &
# TEXT_AGENT_PID=$!

# Wait for the servers to initialize
echo "⏳ Waiting for agent servers to start..."
sleep 5

# Check if video file server is responding with retries
echo "🔍 Checking Video File Server..."
for i in {1..5}; do
    if curl -s "http://localhost:$STATIC_PORT/health" > /dev/null 2>&1; then
        echo "✅ Video File Server started successfully"
        break
    else
        if [ $i -eq 5 ]; then
            echo "❌ Video File Server failed to start after 5 attempts"
            echo "   Check $LOG_DIR/video_file_server.log for details"
            kill $VIDEO_FILE_SERVER_PID 2>/dev/null
            exit 1
        fi
        echo "   Attempt $i/5 failed, waiting 2 seconds..."
        sleep 2
    fi
done

# Check if video agent is responding with retries
echo "🔍 Checking Video Search Agent..."
for i in {1..10}; do
    if curl -s "http://localhost:$VIDEO_PORT/agent.json" > /dev/null 2>&1; then
        echo "✅ Video Search Agent started successfully"
        break
    else
        if [ $i -eq 10 ]; then
            echo "❌ Video Search Agent failed to start after 10 attempts"
            echo "   Check $LOG_DIR/video_search_agent.log for details"
            kill $VIDEO_AGENT_PID $VIDEO_FILE_SERVER_PID 2>/dev/null
            exit 1
        fi
        echo "   Attempt $i/10 failed, waiting 3 seconds..."
        sleep 3
    fi
done

# Get composing agent port from config
COMPOSING_PORT=$(jq -r '.composing_agent_port // 8000' "$CONFIG_FILE")

echo ""
echo "🌐 Starting Multi-Agent Web Interface..."
echo "📖 Instructions:"
echo "   1. Navigate to: http://localhost:$COMPOSING_PORT"
echo "   2. Select 'CoordinatorAgent' from the dropdown"
echo "   3. Try queries like:"
echo "      - 'Show me videos about doctors'"
echo "      - 'Find clips about snow shoveling safety'"
echo "      - 'Search for emergency room footage'"
echo ""
echo "🔧 Configuration Summary:"
echo "   - Search Backend: $SEARCH_BACKEND"
echo "   - Inference Provider: $INFERENCE_PROVIDER"
echo "   - Video Agent: http://localhost:$VIDEO_PORT"
echo "   - Static Server: http://localhost:$STATIC_PORT"
echo "   - Composing Agent: http://localhost:$COMPOSING_PORT"
echo "=" | head -c 60
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down background agent servers..."
    kill $VIDEO_AGENT_PID $VIDEO_FILE_SERVER_PID $ADK_PID 2>/dev/null
    # kill $TEXT_AGENT_PID 2>/dev/null  # Uncomment when text agent is enabled
    echo "✅ All servers stopped. Exiting."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Change to the correct directory for ADK
cd "$(dirname "$0")/.."
echo "📁 Working directory for ADK: $(pwd)"

# Start the ADK web interface (this blocks)
echo "🌐 Starting ADK Web Interface on port $COMPOSING_PORT..."
echo "📋 Access the interface at: http://localhost:$COMPOSING_PORT"
echo "📁 Logs are saved to: $LOG_DIR/"
echo ""

# Start ADK web interface with logging - using current directory as agents directory
uv run adk web . --host 0.0.0.0 --port "$COMPOSING_PORT" > "$LOG_DIR/adk_web_interface.log" 2>&1 &
ADK_PID=$!

# Wait for ADK to start
echo "🔍 Checking ADK Web Interface..."
for i in {1..10}; do
    if curl -s "http://localhost:$COMPOSING_PORT/" > /dev/null 2>&1; then
        echo "✅ ADK Web Interface started successfully"
        break
    else
        if [ $i -eq 10 ]; then
            echo "❌ ADK Web Interface failed to start after 10 attempts"
            echo "   Check $LOG_DIR/adk_web_interface.log for details"
            kill $ADK_PID $VIDEO_AGENT_PID $VIDEO_FILE_SERVER_PID 2>/dev/null
            exit 1
        fi
        echo "   Attempt $i/10 failed, waiting 3 seconds..."
        sleep 3
    fi
done

echo ""
echo "✅ All services are running!"
echo "🌐 Access the interface at: http://localhost:$COMPOSING_PORT"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait indefinitely
while true; do
    sleep 1
done

# If we reach here, ADK stopped, so cleanup
cleanup