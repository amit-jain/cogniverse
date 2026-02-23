# Vespa Search Strategies

**Package**: cogniverse-vespa (Implementation Layer)
**Related**: cogniverse-core (backend configuration)

This document describes the 14 ranking strategies available in the Vespa search backend. These strategies are implemented in the `cogniverse-vespa` package and configured through the backend profile system in `cogniverse-core`.

## Strategy Categories

### Text-Only Search
**Strategy**: `bm25_only`
- **Purpose**: Pure text search using BM25 algorithm
- **Fields**: Searches across video_title, segment_description (ColPali schemas only), audio_transcript
- **Requirements**: Text query only
- **Speed**: Fast
- **Use Case**: Text-heavy queries, document search

**Strategy**: `bm25_no_description` (ColPali/ColQwen schemas only)
- **Purpose**: Pure text search excluding segment descriptions
- **Fields**: Searches across video_title and audio_transcript only
- **Requirements**: Text query only
- **Speed**: Fast
- **Use Case**: When segment descriptions are unreliable or unavailable

### Visual Search
**Strategy**: `float_float`
- **Purpose**: Highest accuracy visual search
- **Method**: Float embeddings with direct similarity
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
- **Method**: Float embedding first phase, BM25 reranking
- **Requirements**: Text query + visual embeddings
- **Speed**: Slow
- **Use Case**: Complex queries with both visual and text components

**Strategy**: `hybrid_binary_bm25`
- **Purpose**: Fast hybrid search
- **Method**: Binary embedding first phase, BM25 reranking
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

### No-Description Variants (ColPali/ColQwen schemas only)
**Strategy**: `hybrid_float_bm25_no_description`
- **Purpose**: Hybrid search excluding frame descriptions
- **Method**: Float embedding + BM25 on video_title and audio_transcript only
- **Requirements**: Text query + visual embeddings
- **Speed**: Slow
- **Use Case**: When frame descriptions are unreliable

**Strategy**: `hybrid_binary_bm25_no_description`
- **Purpose**: Fast hybrid without descriptions
- **Method**: Binary embedding + BM25 on title and transcript only
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

- `segment_description`: Generated visual descriptions (ColPali schemas only - VideoPrism schemas do not include description fields)

- `audio_transcript`: Transcribed audio content

**Note**: Different schemas use different field names and available fields. ColPali schemas use `segment_description` and `segment_id`. VideoPrism schemas use `segment_id` but have no description field. The search client handles these variations automatically.

## Technical Implementation

### BM25 Fieldsets
- **Fieldset name**: `default`
- **Fields**: `video_title`, `segment_description` (ColPali schemas only), `audio_transcript`
- **Query method**: `userQuery()` with `model.defaultIndex="default"`
- **Note**: VideoPrism schemas exclude segment_description from the fieldset as they do not generate descriptions

### Embedding Types
- **Multi-vector float** (ColPali, VideoPrism mv_chunk): `tensor<bfloat16>(patch{}, v[D])` where D is embedding dimension (128 for ColPali, 768 for VideoPrism base, 1024 for VideoPrism large)
- **Multi-vector binary** (ColPali, VideoPrism mv_chunk): `tensor<int8>(patch{}, v[B])` where B = D/8 (16 for ColPali, 96 for VideoPrism base, 128 for VideoPrism large)
- **Single-vector float** (LVT sv_chunk): `tensor<float>(v[D])` where D is 768 (base) or 1024 (large) — no patch dimension
- **Single-vector binary** (LVT sv_chunk): `tensor<int8>(v[B])` where B is 96 (base) or 128 (large) — no patch dimension

### Ranking Phases
- **First phase**: Initial candidate selection
- **Second phase**: Reranking top candidates (default: top 100)
- **Hybrid**: Different models for each phase

## Architecture Integration

### Package Roles
- **cogniverse-vespa** (Implementation Layer): Implements all 14 search strategies and VespaVideoSearchClient
- **cogniverse-core** (Core Layer): Manages backend configuration and profile selection
- **cogniverse-foundation** (Foundation Layer): Provides configuration management and telemetry interfaces

### Configuration
Backend profiles are configured in system config files. Ranking strategies are selected at query time, not in the config:
```json
{
  "backend": {
    "type": "vespa",
    "profiles": {
      "video_colpali_smol500_mv_frame": {
        "type": "video",
        "schema_name": "video_colpali_smol500_mv_frame",
        "embedding_model": "vidore/colsmol-500m",
        "schema_config": {
          "embedding_dim": 128,
          "binary_dim": 16
        }
      }
    }
  }
}
```

Ranking strategy is specified in the search query:
```python
results = client.search({
    "query": "person walking",
    "ranking": "hybrid_float_bm25",
    "top_k": 10
})
```

### Multi-Modal Support
These strategies support all content types:

- VIDEO: Frame-level visual search with temporal context

- AUDIO: Transcript-based text search with BM25

- IMAGE: Visual embedding search (ColPali, VideoPrism)

- DOCUMENT: Text + visual content search

- TEXT: Pure BM25 text search

- DATAFRAME: Structured data with text search

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
from cogniverse_vespa.vespa_search_client import VespaVideoSearchClient

# Initialize client with required parameters
# config_manager must be provided via dependency injection
client = VespaVideoSearchClient(
    backend_url="http://localhost",
    backend_port=8080,
    tenant_id="default",
    config_manager=config_manager
)

# Get recommended strategy based on query characteristics
strategy = client.recommend_strategy(
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
