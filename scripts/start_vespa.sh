#!/bin/bash

# Start Vespa with proper volume mounts for data persistence
# This script ensures that Vespa data persists across container restarts

echo "🚀 Starting Vespa with persistent storage"

# Set up volume paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VESPA_VAR_STORAGE="$PROJECT_ROOT/outputs/vespa/var"
VESPA_LOG_STORAGE="$PROJECT_ROOT/outputs/vespa/logs"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Vespa var storage: $VESPA_VAR_STORAGE"
echo "📁 Vespa log storage: $VESPA_LOG_STORAGE"

# Create directories if they don't exist
mkdir -p "$VESPA_VAR_STORAGE"
mkdir -p "$VESPA_LOG_STORAGE"

# Check if Vespa container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^vespa$"; then
    echo "🛑 Stopping existing Vespa container..."
    docker stop vespa 2>/dev/null || true
    docker rm vespa 2>/dev/null || true
fi

# Start Vespa container with proper volume mounts and increased memory
# JVM heap for main container is configured via services.xml (4GB heap)
echo "🐳 Starting Vespa container with persistent volumes and 8GB memory limit..."
docker run --detach --name vespa \
  --memory="8g" \
  --memory-swap="8g" \
  --volume "$VESPA_VAR_STORAGE:/opt/vespa/var" \
  --volume "$VESPA_LOG_STORAGE:/opt/vespa/logs" \
  --volume "$PROJECT_ROOT/configs:/opt/vespa/conf/custom" \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# Wait for Vespa to start
echo "⏳ Waiting for Vespa to start..."
sleep 5

# Check if Vespa is running and responding
for i in {1..10}; do
    if curl -s "http://localhost:8080/ApplicationStatus" > /dev/null 2>&1; then
        echo "✅ Vespa is running and responding on port 8080"
        break
    else
        if [ $i -eq 10 ]; then
            echo "❌ Vespa failed to start after 10 attempts"
            echo "   Check logs with: docker logs vespa"
            exit 1
        fi
        echo "   Attempt $i/10 failed, waiting 3 seconds..."
        sleep 3
    fi
done

echo ""
echo "🎉 Vespa started successfully!"
echo "📊 Access Vespa at: http://localhost:8080"
echo "🔍 Application status: http://localhost:8080/ApplicationStatus"
echo "📜 View logs: docker logs vespa"
echo "🛑 Stop Vespa: docker stop vespa"