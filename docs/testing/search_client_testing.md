# Search Client Testing Guide

This guide covers testing the Vespa search client with all available ranking strategies.

## Quick Test Command

```bash
# Run comprehensive search client test
python tests/test_search_client.py
```

## Prerequisites

- Vespa running on localhost:8080
- Deployed schema with fieldsets
- Internet connection (for ColPali model download)

## Test Coverage

### Text-Only Search
- **Strategy**: `bm25_only`
- **Requirements**: Text query
- **Tests**: BM25 search across video titles, descriptions, and transcripts

### Visual Search
- **Strategies**: `float_float`, `binary_binary`, `float_binary`, `phased`
- **Requirements**: Visual embeddings
- **Tests**: ColPali-based semantic visual search

### Hybrid Search
- **Strategies**: `hybrid_float_bm25`, `hybrid_binary_bm25`, `hybrid_bm25_binary`, `hybrid_bm25_float`
- **Requirements**: Text query + visual embeddings
- **Tests**: Combined text and visual search

### No-Description Variants
- **Strategies**: `hybrid_float_bm25_no_description`, `hybrid_binary_bm25_no_description`, `hybrid_bm25_binary_no_description`, `hybrid_bm25_float_no_description`
- **Requirements**: Text query + visual embeddings
- **Tests**: Hybrid search excluding frame descriptions

### Input Validation
- **Tests**: Missing embeddings detection, missing text query detection
- **Purpose**: Ensures proper error handling

### Strategy Recommendation
- **Tests**: Automatic strategy selection based on query characteristics
- **Purpose**: Validates recommendation logic

## Test Process

1. **ColPali Model Loading**: Loads `vidore/colsmol-500m` for embedding generation
2. **Text-Only Testing**: Tests BM25 strategies with query validation
3. **Visual Testing**: Tests pure visual strategies with generated embeddings
4. **Hybrid Testing**: Tests combined text + visual strategies
5. **Validation Testing**: Ensures proper error handling for missing inputs
6. **Recommendation Testing**: Validates strategy selection logic

## Expected Output

```
=== Testing Text-Only Strategies ===
✅ bm25_only: Got 3 results
  1. Score: 0.7625, Video: big_buck_bunny_clip

=== Testing Pure Visual Strategies ===
✅ float_float: Got 3 results
  1. Score: 0.8234, Video: big_buck_bunny_clip

=== Testing Hybrid Strategies ===
✅ hybrid_float_bm25: Got 3 results
  1. Score: 0.9156, Video: big_buck_bunny_clip

=== Testing Input Validation ===
✅ Correctly caught missing embeddings
✅ Correctly caught missing text query

=== Testing Strategy Recommendation ===
Text-only query → bm25_only
Visual query with embeddings → hybrid_float_bm25
```

## Manual Testing

### Initialize Client
```python
from src.processing.vespa.vespa_search_client import VespaVideoSearchClient

client = VespaVideoSearchClient()
```

### Test Text Search
```python
results = client.search({
    "query": "buck",
    "ranking": "bm25_only",
    "top_k": 3
})
```

### Test Visual Search
```python
# Requires embeddings
results = client.search({
    "query": "person walking",
    "ranking": "float_float",
    "top_k": 3
}, embeddings=query_embeddings)
```

### Test Hybrid Search
```python
results = client.search({
    "query": "buck",
    "ranking": "hybrid_float_bm25",
    "top_k": 3
}, embeddings=query_embeddings)
```

### Get Strategy Recommendations
```python
recommended = client.recommend_strategy(
    "show me videos about robots",
    has_embeddings=True,
    speed_priority=False
)
```

## Troubleshooting

### Vespa Connection Issues
```bash
# Check Vespa status
curl localhost:8080/ApplicationStatus

# Restart Vespa if needed
./scripts/start_vespa.sh
```

### Test Failures
```bash
# Verify schema deployment
python scripts/deploy_vespa_schema.py

# Check if data is indexed
curl "localhost:8080/search/?yql=select * from video_frame where true&hits=1"
```

### Model Loading Issues
- Ensure 16GB+ RAM available
- Check internet connection for model downloads
- Clear model cache if corrupted

## Performance Metrics

The test provides:
- **Response times** for each strategy
- **Relevance scores** comparison
- **Strategy effectiveness** analysis
- **Speed vs accuracy** trade-offs

### Performance Benchmarks
- **Response Times**: Measures search latency for each strategy
- **Relevance Scores**: Compares scoring effectiveness
- **Strategy Comparison**: Shows speed vs. accuracy trade-offs
- **Memory Usage**: Tracks resource consumption during testing

## Available Ranking Strategies

| Strategy | Type | Speed | Accuracy | Requirements |
|----------|------|--------|----------|-------------|
| `bm25_only` | Text | Fast | Good | Text query |
| `float_float` | Visual | Slow | Highest | Embeddings |
| `binary_binary` | Visual | Fastest | Good | Embeddings |
| `hybrid_float_bm25` | Hybrid | Slow | Highest | Text + embeddings |
| `hybrid_binary_bm25` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_binary` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_float` | Hybrid | Medium | Very Good | Text + embeddings |

## Test Data Requirements

- **Minimum**: Empty Vespa index (test will work with 0 results)
- **Recommended**: Indexed video content for meaningful results
- **Automatic**: ColPali model downloaded on first run