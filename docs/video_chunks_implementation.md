# Video Chunks Implementation (TwelveLabs-Style)

## Overview

This implementation creates a video search system similar to TwelveLabs' approach, but using VideoPrism embeddings instead of proprietary models. The key innovation is storing entire videos as single documents with multiple chunk embeddings, enabling efficient semantic search across video segments.

## Key Differences from Our Existing Approaches

### Current Approaches:
1. **frame_based_colpali**: Extracts keyframes and creates one document per frame
2. **direct_video_global**: Creates one embedding for the entire video
3. **direct_video_segment**: Processes 30-second segments as separate documents

### New Video Chunks Approach:
- **One document per video** with multiple 6-second chunk embeddings
- **Hybrid search** combining BM25 text search with chunk-level semantic search
- **Efficient storage** - all chunks in a single tensor field
- **Better context** - chunks can overlap for continuity

## Schema Design

```json
{
  "video_id": "string",
  "title": "string (BM25 indexed)",
  "keywords": "string (BM25 indexed)",
  "video_summary": "string (BM25 indexed)",
  "transcript": "string (full video transcript, BM25 indexed)",
  "chunk_embeddings": "tensor<float>(chunk{},x[1024])",
  "chunk_metadata": "tensor<float>(chunk{},meta[4])",
  "chunk_transcripts": "array<string> (per-chunk transcripts)"
}
```

### Chunk Metadata Structure
Each chunk's metadata tensor contains:
- `meta[0]`: start_time_seconds
- `meta[1]`: end_time_seconds
- `meta[2]`: chunk_index
- `meta[3]`: confidence_score

## Configuration

```json
"video_chunks_videoprism": {
  "vespa_schema": "video_chunks",
  "embedding_model": "videoprism_public_v1_large_hf",
  "model_specific": {
    "chunk_duration": 6.0,      // 6-second chunks like TwelveLabs
    "chunk_overlap": 1.0,       // 1-second overlap between chunks
    "sampling_fps": 2.0,        // 2 FPS within each chunk
    "max_frames_per_chunk": 12, // Max 12 frames per chunk
    "store_as_single_doc": true // Store as single document
  }
}
```

## Processing Pipeline

1. **Video Chunking**:
   ```
   Video (5 minutes) → 50 chunks of 6 seconds each (with 1s overlap)
   ```

2. **Embedding Generation**:
   - Each chunk → 12 frames at 2 FPS
   - VideoPrism processes frames → 1024-dim embedding per chunk

3. **Transcript Alignment**:
   - Full video transcript with timestamps
   - Extract text for each chunk's time range
   - Store both full transcript and per-chunk transcripts

4. **Document Creation**:
   ```python
   {
     "video_id": "video_123",
     "title": "Sample Video",
     "transcript": "Full video transcript...",
     "chunk_embeddings": {
       "0": [0.1, 0.2, ...],  # 1024 dims
       "1": [0.3, 0.4, ...],  # 1024 dims
       ...
       "49": [0.5, 0.6, ...]  # 1024 dims
     },
     "chunk_transcripts": [
       "Hello, welcome to...",     # Chunk 0: 0-6s
       "this video about...",      # Chunk 1: 5-11s
       "machine learning and...",  # Chunk 2: 10-16s
       ...
     ]
   }
   ```

## Search Strategies

### 1. Hybrid Search (Recommended)
Combines text and semantic search:
```
Score = BM25_score + 10 * max(chunk_similarities)
```

### 2. Pure Semantic Search
Finds the best matching chunk:
```
Score = max(cosine_similarity(query_embedding, chunk_embeddings))
```

### 3. Text-Only Search
Traditional BM25 across title, keywords, summary, and transcript.

## Advantages

1. **Efficiency**: Single document per video reduces document count
2. **Flexibility**: Can search by text metadata OR video content
3. **Granularity**: 6-second chunks provide good temporal resolution
4. **Context**: Overlapping chunks maintain continuity
5. **Scalability**: Follows Vespa best practices for large-scale deployment
6. **Rich Search**: Chunk-level transcripts enable precise text+video search
7. **Temporal Alignment**: Find exact moments matching text or visual queries

## Query Examples

### Semantic Search with Text Fallback
```python
session.query(
    yql="select * from video_chunks where userQuery() OR ({targetHits:100}nearestNeighbor(chunk_embeddings,q))",
    ranking="hybrid_chunk_search",
    body={"input.query(q)": query_embedding}
)
```

### Find Specific Moments
```python
# "Person waving at camera"
results = search(
    query_text="person waving hello",
    query_embedding=encode("person waving at camera"),
    ranking="hybrid_chunk_search"
)
# Returns videos with:
# - Matching visual content (person waving)
# - Matching audio ("hello" in chunk transcript)
# - Exact timestamps for both
```

### Multi-Modal Search Example
```python
# Find when someone talks about "machine learning" while showing code
results = search(
    query_text="machine learning",
    query_embedding=encode("computer screen showing code"),
    ranking="hybrid_chunk_search"
)
# Matches chunks where:
# - Transcript contains "machine learning"
# - Visual embedding matches "code on screen"
```

## Implementation Status

- [x] Profile configuration added
- [x] Vespa schema created
- [ ] Modify embedding generator for chunk processing
- [ ] Update document builder for single-doc format
- [ ] Test with sample videos
- [ ] Benchmark against existing approaches

## Next Steps

1. Implement chunk-aware video processor in `src/processing/pipeline_steps/`
2. Create chunk-aware document builder
3. Update search backend to handle chunk results
4. Add timestamp extraction from chunk matches