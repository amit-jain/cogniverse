#!/bin/bash

# Stop Vespa container
# This script stops the Vespa container while preserving data (due to volume mounts)

echo "🛑 Stopping Vespa Container"
echo "=" * 60

# Check if Vespa container exists and is running
if docker ps --format '{{.Names}}' | grep -q "^vespa$"; then
    echo "🔄 Stopping Vespa container..."
    docker stop vespa
    
    if [ $? -eq 0 ]; then
        echo "✅ Vespa container stopped successfully"
        
        # Check if we want to remove the container
        echo ""
        echo "🗑️  Remove container? (data will be preserved due to volume mounts)"
        echo "   [y/N]: "
        read -r REMOVE_CONTAINER
        
        if [[ "$REMOVE_CONTAINER" =~ ^[Yy]$ ]]; then
            docker rm vespa
            if [ $? -eq 0 ]; then
                echo "✅ Vespa container removed"
            else
                echo "❌ Failed to remove Vespa container"
            fi
        else
            echo "⚪ Vespa container kept (stopped)"
        fi
    else
        echo "❌ Failed to stop Vespa container"
        exit 1
    fi
else
    echo "⚪ Vespa container is not running"
fi

echo ""
echo "💾 Data Status:"
if [ -d "data/vespa/var" ] && [ -d "data/vespa/logs" ]; then
    echo "✅ Vespa data preserved in data/vespa/"
    echo "📊 Data size: $(du -sh data/vespa/ | cut -f1)"
else
    echo "⚠️  Vespa data directories not found"
fi

echo ""
echo "🎯 To restart Vespa: ./scripts/start_vespa.sh"
echo "🎯 To restart full system: ./scripts/run_servers.sh"