#!/bin/bash

# Run ingestion for all profiles except frame_based_colpali
PROFILES=(
    "direct_video_colqwen"
    "direct_video_frame"
    "direct_video_frame_large"
    "direct_video_global"
    "direct_video_global_large"
)

echo "Starting ingestion for all non-ColPali profiles..."
echo "=========================================="

for profile in "${PROFILES[@]}"; do
    echo ""
    echo "Running ingestion for profile: $profile"
    echo "----------------------------------------"
    
    JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
        --video_dir data/testset/evaluation/test \
        --backend vespa \
        --profile "$profile" \
        --max_videos 1
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ingestion for $profile"
    else
        echo "✗ Failed ingestion for $profile"
    fi
    echo ""
done

echo "=========================================="
echo "Ingestion complete for all profiles!"