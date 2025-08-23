# Vespa Search Strategies

This document describes the 13 ranking strategies available in the Vespa search backend.

## Strategy Categories

### Text-Only Search
**Strategy**: `bm25_only`
- **Purpose**: Pure text search using BM25 algorithm
- **Fields**: Searches across video_title, frame_description, audio_transcript
- **Requirements**: Text query only
- **Speed**: Fast
- **Use Case**: Text-heavy queries, document search

### Visual Search
**Strategy**: `float_float`
- **Purpose**: Highest accuracy visual search
- **Method**: ColPali float embeddings with direct similarity
- **Requirements**: Visual embeddings
- **Speed**: Slowest
- **Use Case**: Maximum visual precision needed

**Strategy**: `binary_binary`  
- **Purpose**: Fastest visual search
- **Method**: Binary embeddings with Hamming distance
- **Requirements**: Visual embeddings (binary)
- **Speed**: Fastest
- **Use Case**: Speed-critical visual search

**Strategy**: `float_binary`
- **Purpose**: Balanced speed/accuracy
- **Method**: Float query with binary storage using unpack_bits
- **Requirements**: Visual embeddings
- **Speed**: Fast
- **Use Case**: Good balance of speed and accuracy

**Strategy**: `phased`
- **Purpose**: Optimized retrieval with reranking
- **Method**: Binary first phase, float reranking
- **Requirements**: Both binary and float embeddings
- **Speed**: Fast
- **Use Case**: High accuracy with optimized performance

### Hybrid Search (Text + Visual)
**Strategy**: `hybrid_float_bm25`
- **Purpose**: Best overall accuracy
- **Method**: Float ColPali first phase, BM25 reranking
- **Requirements**: Text query + visual embeddings
- **Speed**: Slow
- **Use Case**: Complex queries with both visual and text components

**Strategy**: `hybrid_binary_bm25`
- **Purpose**: Fast hybrid search
- **Method**: Binary ColPali first phase, BM25 reranking
- **Requirements**: Text query + visual embeddings (binary)
- **Speed**: Fast
- **Use Case**: Fast hybrid search for visual+text queries

**Strategy**: `hybrid_bm25_binary`
- **Purpose**: Text-first with visual validation
- **Method**: BM25 first phase, binary visual reranking
- **Requirements**: Text query + visual embeddings (binary)
- **Speed**: Fast
- **Use Case**: Text-heavy queries with visual validation

**Strategy**: `hybrid_bm25_float`
- **Purpose**: Text-first with precise reranking
- **Method**: BM25 first phase, float visual reranking
- **Requirements**: Text query + visual embeddings
- **Speed**: Medium
- **Use Case**: Text-heavy queries with precise visual reranking

### No-Description Variants
**Strategy**: `hybrid_float_bm25_no_description`
- **Purpose**: Hybrid search excluding frame descriptions
- **Method**: Float ColPali + BM25 on video_title and audio_transcript only
- **Requirements**: Text query + visual embeddings
- **Speed**: Slow
- **Use Case**: When frame descriptions are unreliable

**Strategy**: `hybrid_binary_bm25_no_description`
- **Purpose**: Fast hybrid without descriptions
- **Method**: Binary ColPali + BM25 on title and transcript only
- **Requirements**: Text query + visual embeddings (binary)
- **Speed**: Fast
- **Use Case**: Fast hybrid without frame descriptions

**Strategy**: `hybrid_bm25_binary_no_description`
- **Purpose**: Text-first without descriptions
- **Method**: BM25 on title/transcript, binary visual reranking
- **Requirements**: Text query + visual embeddings (binary)
- **Speed**: Fast
- **Use Case**: Text-first search excluding descriptions

**Strategy**: `hybrid_bm25_float_no_description`
- **Purpose**: Text-first without descriptions, precise reranking
- **Method**: BM25 on title/transcript, float visual reranking
- **Requirements**: Text query + visual embeddings
- **Speed**: Medium
- **Use Case**: Text-first with precise visual reranking, no descriptions

## Usage Guidelines

### Query Type Analysis
- **Text-only queries**: Use `bm25_only`
- **Visual queries**: Use `float_float` (accuracy) or `binary_binary` (speed)
- **Combined queries**: Use `hybrid_float_bm25` (accuracy) or `hybrid_binary_bm25` (speed)

### Speed vs Accuracy Trade-offs
- **Maximum accuracy**: `hybrid_float_bm25`, `float_float`
- **Maximum speed**: `binary_binary`, `hybrid_binary_bm25`
- **Balanced**: `float_binary`, `hybrid_bm25_float`

### Field Configuration
All BM25 strategies use fieldsets to search across:
- `video_title`: Video file names and metadata
- `frame_description`: Generated visual descriptions
- `audio_transcript`: Transcribed audio content

## Technical Implementation

### BM25 Fieldsets
- **Fieldset name**: `default`
- **Fields**: `video_title`, `frame_description`, `audio_transcript`
- **Query method**: `userQuery()` with `model.defaultIndex="default"`

### Embedding Types
- **Float embeddings**: `tensor<float>(patch{}, v[128])`
- **Binary embeddings**: `tensor<int8>(patch{}, v[16])`
- **Multi-vector**: Multiple patches per query/document

### Ranking Phases
- **First phase**: Initial candidate selection
- **Second phase**: Reranking top candidates (default: top 100)
- **Hybrid**: Different models for each phase

## Performance Characteristics (Estimated)

<!-- TODO: Benchmark and update with actual measured performance -->

| Strategy | Response Time* | Memory Usage | Accuracy |
|----------|----------------|--------------|----------|
| `bm25_only` | ~50ms | Low | Text-only |
| `binary_binary` | ~100ms | Medium | Good |
| `float_float` | ~300ms | High | Highest |
| `hybrid_float_bm25` | ~350ms | High | Highest |
| `hybrid_binary_bm25` | ~150ms | Medium | Good |

*Estimated values - actual performance varies based on index size, hardware, and query complexity

## Strategy Selection

### Automatic Recommendation
The search client provides automatic strategy selection:
```python
client.recommend_strategy(
    query_text="your query",
    has_embeddings=True,
    speed_priority=False
)
```

### Manual Selection
Choose based on:
1. **Query type**: Text, visual, or hybrid
2. **Speed requirements**: Real-time vs batch processing
3. **Accuracy needs**: Good enough vs maximum precision
4. **Resource constraints**: Memory and compute limitations