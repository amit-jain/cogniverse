#!/bin/bash

# Analyze Vespa embeddings - exports data and runs integrity test
# Usage: ./analyze_vespa_embeddings.sh [num_slices]

NUM_SLICES=${1:-100}
OUTPUT_DIR="outputs/exports/vespa_slices"

echo "🚀 Vespa Embedding Analysis"
echo "================================"
echo "Number of slices: $NUM_SLICES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export each slice
echo "📥 Exporting data from Vespa in $NUM_SLICES slices..."
for slice_id in $(seq 0 $((NUM_SLICES - 1))); do
    printf "  Slice %3d/%d: " "$slice_id" "$((NUM_SLICES - 1))"
    
    # Export the slice
    if vespa visit -t local --slices "$NUM_SLICES" --slice-id "$slice_id" --selection "video_frame" > "$OUTPUT_DIR/slice_${slice_id}.jsonl" 2>&1; then
        lines=$(wc -l < "$OUTPUT_DIR/slice_${slice_id}.jsonl")
        printf "✅ %4d documents\n" "$lines"
    else
        echo "❌ Failed"
        exit 1
    fi
    
    # Small delay to avoid overwhelming the system
    sleep 0.2
done

# Combine all slices
echo ""
echo "🔗 Combining all slices..."
cat "$OUTPUT_DIR"/slice_*.jsonl > "$OUTPUT_DIR/all_documents.jsonl"

total_lines=$(wc -l < "$OUTPUT_DIR/all_documents.jsonl")
echo "✅ Export complete: $total_lines total documents"

# Run the embedding integrity test
echo ""
echo "🧪 Running embedding integrity analysis..."
echo "================================"
uv run python tests/test_embedding_integrity.py --exported-file "$OUTPUT_DIR/all_documents.jsonl"