# Search & Reranking Module Study Guide

**Last Updated:** 2025-10-07
**Purpose:** Comprehensive guide to the search and reranking system for multi-modal result optimization

---

## Table of Contents
1. [Module Overview](#module-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Reranking Strategies](#reranking-strategies)
5. [Usage Examples](#usage-examples)
6. [Production Considerations](#production-considerations)
7. [Testing](#testing)

---

## Module Overview

### Purpose
The Search & Reranking Module provides intelligent post-retrieval result optimization for multi-modal video search. It combines heuristic multi-modal analysis with learned neural reranking to improve result quality and relevance.

### Key Features
- **Multi-Modal Reranking**: Cross-modal relevance, temporal alignment, complementarity analysis
- **Learned Reranking**: LiteLLM-based neural rerankers (Cohere, Jina, Together AI, Ollama)
- **Hybrid Fusion**: Weighted ensemble, cascade, and consensus strategies
- **Configurable Backend**: Pure heuristic, pure learned, or hybrid approaches
- **Search Service**: Unified search orchestration with query encoding and backend integration

### Module Location
```
src/app/search/
├── base.py                    # Base interfaces (SearchResult, SearchBackend)
├── service.py                 # Unified search service
├── multi_modal_reranker.py    # Heuristic multi-modal reranking
├── learned_reranker.py        # LiteLLM neural reranking
└── hybrid_reranker.py         # Hybrid fusion strategies
```

---

## Architecture

### Search Service Architecture

```mermaid
graph TB
    SearchService[SearchService<br/>• Query Encoding Coordination<br/>• Backend Search Orchestration<br/>• Multi-Tenant Telemetry Integration]

    SearchService --> EncoderFactory[Query Encoder Factory<br/>• ColPali Encoder frame-based<br/>• VideoPrism Encoder chunk/global<br/>• Strategy-Aware Encoding]

    EncoderFactory --> Backend[Search Backend Vespa<br/>• Vector Search binary/float<br/>• Hybrid Ranking BM25 + Neural<br/>• Multi-Schema Support]

    Backend --> Results[SearchResult List]

    style SearchService fill:#e1f5ff
    style EncoderFactory fill:#fff4e1
    style Backend fill:#ffe1e1
    style Results fill:#e1ffe1
```

### Reranking Pipeline Architecture

```mermaid
graph TB
    InitialResults[Initial Search Results<br/>• Vespa vector search output top_k = 100<br/>• Multiple modalities: video, image, audio, document]

    InitialResults --> Configurable[ConfigurableMultiModalReranker<br/>• Selects strategy based on config.json<br/>• Routes to appropriate reranker]

    Configurable --> Heuristic[Heuristic Reranker<br/>• Cross-modal<br/>• Temporal<br/>• Complementary<br/>• Diversity]
    Configurable --> Learned[Learned Reranker<br/>• LiteLLM API<br/>• Cohere<br/>• Together AI<br/>• Jina AI<br/>• Ollama local]
    Configurable --> Hybrid[Hybrid Reranker<br/>• Weighted<br/>• Cascade<br/>• Consensus]

    Heuristic --> Reranked[Reranked Results top_n<br/>• Updated relevance scores<br/>• Modality-aware ordering<br/>• Diversity optimization]
    Learned --> Reranked
    Hybrid --> Reranked

    style InitialResults fill:#e1f5ff
    style Configurable fill:#fff4e1
    style Heuristic fill:#ffe1e1
    style Learned fill:#ffe1e1
    style Hybrid fill:#ffe1e1
    style Reranked fill:#e1ffe1
```

### Hybrid Reranking Strategies

```mermaid
graph TB
    subgraph Strategy1[Strategy: Weighted Ensemble]
        Heur1[Heuristic Reranker<br/>score: 0.85<br/>× weight 0.3<br/>= 0.255]
        Learn1[Learned Reranker<br/>score: 0.92<br/>× weight 0.7<br/>= 0.644]
        Heur1 --> Final1[Final Score<br/>0.255 + 0.644<br/>= 0.899]
        Learn1 --> Final1
    end

    subgraph Strategy2[Strategy: Cascade]
        Step1[Step 1: Heuristic Filter<br/>100 results<br/>Filter Fast]
        Step2[Step 2: Learned Rerank<br/>10 results<br/>Neural Rerank slow]
        Step1 -->|Top 50%| Step2
        Info1[Efficient: Reduces expensive<br/>learned model calls]
    end

    subgraph Strategy3[Strategy: Consensus]
        Ranks[Heuristic Ranks: 1, 5, 3, 10, 2<br/>Learned Ranks: 2, 3, 1, 8, 4]
        Borda[Borda Count Scoring<br/>geometric mean]
        Consensus[Consensus: Requires agreement from BOTH methods<br/>Result must rank highly in heuristic AND learned]
        Ranks --> Borda
        Borda --> Consensus
    end

    style Strategy1 fill:#e1f5ff
    style Heur1 fill:#fff4e1
    style Learn1 fill:#fff4e1
    style Final1 fill:#e1ffe1
    style Strategy2 fill:#ffe1e1
    style Step1 fill:#f5f5f5
    style Step2 fill:#f5f5f5
    style Info1 fill:#f0f0f0
    style Strategy3 fill:#fff4e1
    style Ranks fill:#f5f5f5
    style Borda fill:#f5f5f5
    style Consensus fill:#e1ffe1
```

---

## Core Components

### 1. SearchResult (base.py:9-38)

```python
class SearchResult:
    """Represents a search result with document and score"""

    def __init__(
        self,
        document: Document,
        score: float,
        highlights: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize search result

        Args:
            document: Document object containing content
            score: Relevance score (0.0-1.0)
            highlights: Optional highlighted snippets
        """
```

**Key Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to API response format with temporal info

**Attributes:**
- `document`: Document object with id, metadata, content
- `score`: Original search score from backend
- `highlights`: Highlighted text snippets for display

---

### 2. SearchBackend (base.py:41-93)

```python
class SearchBackend(ABC):
    """Abstract base class for search backends"""

    @abstractmethod
    def search(
        self,
        query_embeddings: Optional[np.ndarray],
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        ranking_strategy: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents matching the query

        Args:
            query_embeddings: Optional embeddings (generated if None)
            query_text: Original query text
            top_k: Number of results to return
            filters: Optional filters (date range, etc.)
            ranking_strategy: Ranking strategy override

        Returns:
            List of SearchResult objects
        """
```

**Key Methods:**
- `get_document(document_id: str)`: Retrieve specific document
- `export_embeddings(schema, max_documents, filters)`: Export embeddings for analysis

---

### 3. SearchService (service.py:18-199)

```python
class SearchService:
    """Unified search service for video retrieval"""

    def __init__(self, config: Dict[str, Any], profile: str):
        """
        Initialize search service

        Args:
            config: Configuration dictionary with vespa_url, etc.
            profile: Video processing profile (frame_based_colpali, etc.)
        """
        self.config = config
        self.profile = profile
        self._init_query_encoder()   # Initialize encoder from profile
        self._init_search_backend()  # Initialize Vespa backend
```

**Key Methods:**

```python
def search(
    self,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    ranking_strategy: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> List[SearchResult]:
    """
    Search with multi-tenant telemetry

    Workflow:
    1. Create search span with tenant isolation
    2. Generate query embeddings with encode span
    3. Call backend with backend_search span
    4. Add result details to spans
    5. Return ranked SearchResult list
    """
```

**Integration Points:**
- **Telemetry**: `search_span`, `encode_span`, `backend_search_span` for multi-tenant tracking
- **Query Encoder**: `QueryEncoderFactory` for strategy-aware encoding
- **Backend Registry**: `get_backend_registry()` for Vespa backend instantiation

---

### 4. MultiModalReranker (multi_modal_reranker.py:50-396)

```python
class MultiModalReranker:
    """
    Heuristic reranker considering multiple modalities

    Scoring Components:
    - Cross-modal relevance (0.3 weight)
    - Temporal alignment (0.2 weight)
    - Result complementarity (0.2 weight)
    - Diversity bonus (0.15 weight)
    - Original score (0.15 weight)
    """

    def __init__(
        self,
        cross_modal_weight: float = 0.3,
        temporal_weight: float = 0.2,
        complementarity_weight: float = 0.2,
        diversity_weight: float = 0.15,
        original_score_weight: float = 0.15
    ):
        """Weights must sum to 1.0"""
```

**Key Methods:**

```python
async def rerank_results(
    self,
    results: List[SearchResult],
    query: str,
    modalities: List[QueryModality],
    context: Optional[Dict] = None
) -> List[SearchResult]:
    """
    Rerank using heuristic multi-modal scoring

    Calculates:
    - Cross-modal alignment (modality compatibility matrix)
    - Temporal alignment (time range matching)
    - Complementarity (unique information contribution)
    - Diversity (modality distribution balance)

    Returns results with updated metadata:
    - reranking_score
    - score_components (breakdown)
    """
```

**Scoring Logic:**

1. **Cross-Modal Score** (multi_modal_reranker.py:160-201):
   - Direct modality match: 1.0
   - Compatible modalities (video↔image): 0.7
   - Mixed query accepts all: 0.8
   - Unrelated modalities: 0.3

2. **Temporal Score** (multi_modal_reranker.py:203-248):
   - Inside time range (centered): 0.7-1.0
   - <30 days outside: 0.5
   - 30-90 days: 0.3
   - >365 days: 0.1

3. **Complementarity Score** (multi_modal_reranker.py:250-289):
   - Keyword overlap analysis
   - Low overlap = high complementarity
   - Score = 1.0 - avg_overlap

4. **Diversity Score** (multi_modal_reranker.py:291-324):
   - First result from modality: 1.0
   - Second: 0.8, Third: 0.6, Fourth: 0.4, Fifth+: 0.2

---

### 5. LearnedReranker (learned_reranker.py:25-215)

```python
class LearnedReranker:
    """
    Unified learned reranker using LiteLLM

    Supported Models:
    - Cohere: rerank-english-v3.0
    - Together AI: Llama-Rank (ColBERT-style)
    - Jina AI: jina-reranker-v2-base-multilingual
    - Ollama: bge-reranker-v2-m3, mxbai-rerank-large-v2
    - Any LiteLLM-supported reranker
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize from config.json "reranking" section

        Config structure:
        {
          "reranking": {
            "model": "cohere",
            "supported_models": {
              "cohere": "cohere/rerank-english-v3.0",
              "ollama": "openai/bge-reranker-v2-m3"
            },
            "api_base": "http://localhost:11434"  # For Ollama
          }
        }
        """
```

**Key Methods:**

```python
async def rerank(
    self,
    query: str,
    results: List[SearchResult],
    top_n: Optional[int] = None
) -> List[SearchResult]:
    """
    Rerank using LiteLLM neural model

    Process:
    1. Limit to max_results_to_rerank (default 100)
    2. Prepare documents as "title content" strings
    3. Call LiteLLM arerank API
    4. Map relevance scores back to SearchResult
    5. Add metadata: reranking_score, reranker_model, original_rank

    Returns:
        Reranked results with neural scores
    """
```

**LiteLLM Integration:**
- Unified API for all reranking models
- For Ollama: uses OpenAI-compatible endpoint with custom `api_base`
- Automatic batching and error handling
- Fallback to original results on failure

---

### 6. HybridReranker (hybrid_reranker.py:26-258)

```python
class HybridReranker:
    """
    Combines heuristic and learned reranking

    Strategies:
    - weighted_ensemble: Parallel scoring with weighted combination
    - cascade: Heuristic filter → learned rerank (efficient)
    - consensus: Borda count requiring agreement from both
    """

    def __init__(
        self,
        heuristic_reranker: Optional[MultiModalReranker] = None,
        learned_reranker: Optional[LearnedReranker] = None,
        strategy: Optional[str] = None,
        learned_weight: Optional[float] = None,
        heuristic_weight: Optional[float] = None
    ):
        """
        Loads from config.json if parameters are None

        Config example:
        {
          "reranking": {
            "hybrid_strategy": "weighted_ensemble",
            "learned_weight": 0.7,
            "heuristic_weight": 0.3
          }
        }
        """
```

**Strategy Implementations:**

1. **Weighted Ensemble** (hybrid_reranker.py:128-175):
```python
# Both rerankers run in parallel
heuristic_results = await self.heuristic_reranker.rerank_results(...)
learned_results = await self.learned_reranker.rerank(...)

# Combine scores
final_score = (
    h_score * self.heuristic_weight +
    l_score * self.learned_weight
)

# Metadata includes: heuristic_score, learned_score, fusion_strategy
```

2. **Cascade** (hybrid_reranker.py:177-204):
```python
# Step 1: Heuristic filtering (top 50% or min 10)
heuristic_results = await self.heuristic_reranker.rerank_results(...)
top_k = max(10, len(results) // 2)
filtered = heuristic_results[:top_k]

# Step 2: Learned reranking on filtered set (more efficient)
final_results = await self.learned_reranker.rerank(query, filtered)
```

3. **Consensus** (hybrid_reranker.py:206-258):
```python
# Get both rankings
heuristic_results = await self.heuristic_reranker.rerank_results(...)
learned_results = await self.learned_reranker.rerank(...)

# Create rank maps
heuristic_ranks = {r.id: idx for idx, r in enumerate(heuristic_results)}
learned_ranks = {r.id: idx for idx, r in enumerate(learned_results)}

# Borda count with geometric mean (emphasizes agreement)
h_score = len(results) - h_rank
l_score = len(results) - l_rank
consensus_score = (h_score * l_score) ** 0.5
```

---

### 7. ConfigurableMultiModalReranker (multi_modal_reranker.py:398-516)

```python
class ConfigurableMultiModalReranker:
    """
    Facade pattern for reranking with config-based routing

    Modes:
    - Pure heuristic (model="heuristic")
    - Pure learned (model="cohere", "ollama", etc.)
    - Hybrid (use_hybrid=true)
    """

    def __init__(self):
        """
        Auto-initializes from config.json

        Example config:
        {
          "reranking": {
            "enabled": true,
            "model": "cohere",
            "use_hybrid": true,
            "hybrid_strategy": "weighted_ensemble"
          }
        }
        """
        self.enabled = rerank_config.get("enabled", False)
        self.heuristic_reranker = MultiModalReranker()  # Always available
        self.learned_reranker = LearnedReranker() if model != "heuristic" else None
        self.hybrid_reranker = HybridReranker() if use_hybrid else None
```

**Key Method:**

```python
async def rerank(
    self,
    query: str,
    results: List[SearchResult],
    modalities: List[QueryModality],
    context: Optional[Dict] = None
) -> List[SearchResult]:
    """
    Route to appropriate reranker based on configuration

    Routing logic:
    - If disabled: return original results
    - If hybrid_reranker: use hybrid strategy
    - Elif learned_reranker: use learned model
    - Else: use heuristic multi-modal logic
    """
```

---

## Reranking Strategies

### Strategy Comparison

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **Pure Heuristic** | ⚡⚡⚡ Fast (5ms) | 🎯 Good | No API costs, interpretable |
| **Pure Learned** | 🐌 Slow (200ms) | 🎯🎯🎯 Excellent | Best quality, API costs |
| **Weighted Ensemble** | 🐌 Slow (200ms) | 🎯🎯🎯 Excellent | Balanced, robust |
| **Cascade** | ⚡⚡ Medium (50ms) | 🎯🎯 Very Good | Efficient learned |
| **Consensus** | 🐌 Slow (200ms) | 🎯🎯 Very Good | High precision |

### Configuration Examples

**Pure Heuristic (No API costs)**:
```json
{
  "reranking": {
    "enabled": true,
    "model": "heuristic",
    "use_hybrid": false
  }
}
```

**Pure Learned (Cohere)**:
```json
{
  "reranking": {
    "enabled": true,
    "model": "cohere",
    "supported_models": {
      "cohere": "cohere/rerank-english-v3.0"
    },
    "top_n": 10,
    "max_results_to_rerank": 100,
    "use_hybrid": false
  }
}
```

**Pure Learned (Ollama Local)**:
```json
{
  "reranking": {
    "enabled": true,
    "model": "ollama",
    "supported_models": {
      "ollama": "openai/bge-reranker-v2-m3"
    },
    "api_base": "http://localhost:11434",
    "use_hybrid": false
  }
}
```

**Hybrid Weighted Ensemble**:
```json
{
  "reranking": {
    "enabled": true,
    "model": "cohere",
    "use_hybrid": true,
    "hybrid_strategy": "weighted_ensemble",
    "learned_weight": 0.7,
    "heuristic_weight": 0.3
  }
}
```

**Hybrid Cascade (Efficient)**:
```json
{
  "reranking": {
    "enabled": true,
    "model": "ollama",
    "use_hybrid": true,
    "hybrid_strategy": "cascade"
  }
}
```

---

## Usage Examples

### Example 1: Basic Search with SearchService

```python
from src.app.search.service import SearchService

# Initialize search service
config = {
    "vespa_url": "http://localhost",
    "vespa_port": 8080,
    "search_backend": "vespa"
}
service = SearchService(config, profile="frame_based_colpali")

# Perform search
results = service.search(
    query="Show me videos about quantum computing",
    top_k=10,
    tenant_id="user_123"
)

# Access results
for result in results:
    print(f"Document: {result.document.id}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

### Example 2: Heuristic Multi-Modal Reranking

```python
from src.app.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality
)

# Initialize reranker
reranker = MultiModalReranker(
    cross_modal_weight=0.3,
    temporal_weight=0.2,
    complementarity_weight=0.2,
    diversity_weight=0.15,
    original_score_weight=0.15
)

# Detect query modalities
modalities = [QueryModality.VIDEO, QueryModality.TEXT]

# Add temporal context
context = {
    "temporal": {
        "time_range": (
            datetime(2024, 1, 1),
            datetime(2024, 12, 31)
        )
    }
}

# Rerank results
reranked = await reranker.rerank_results(
    results=search_results,
    query="quantum computing experiments",
    modalities=modalities,
    context=context
)

# Analyze score breakdown
for result in reranked[:5]:
    print(f"Final Score: {result.metadata['reranking_score']:.3f}")
    print(f"Components: {result.metadata['score_components']}")
```

### Example 3: Learned Reranking with Cohere

```python
from src.app.search.learned_reranker import LearnedReranker

# Initialize with Cohere model
reranker = LearnedReranker(model="cohere/rerank-english-v3.0")

# Rerank with neural model
reranked = await reranker.rerank(
    query="quantum computing breakthroughs",
    results=search_results,
    top_n=10
)

# Check model info
info = reranker.get_model_info()
print(f"Model: {info['model']}")
print(f"Max rerank: {info['max_results_to_rerank']}")
```

### Example 4: Hybrid Weighted Ensemble

```python
from src.app.search.hybrid_reranker import HybridReranker
from src.app.search.multi_modal_reranker import MultiModalReranker
from src.app.search.learned_reranker import LearnedReranker

# Initialize hybrid reranker
hybrid = HybridReranker(
    heuristic_reranker=MultiModalReranker(),
    learned_reranker=LearnedReranker(),
    strategy="weighted_ensemble",
    learned_weight=0.7,
    heuristic_weight=0.3
)

# Rerank with hybrid approach
reranked = await hybrid.rerank_hybrid(
    query="machine learning tutorials",
    results=search_results,
    modalities=[QueryModality.VIDEO],
    context={}
)

# Examine fusion details
for result in reranked[:3]:
    metadata = result.metadata
    print(f"Final: {metadata['reranking_score']:.3f}")
    print(f"  Heuristic: {metadata['heuristic_score']:.3f}")
    print(f"  Learned: {metadata['learned_score']:.3f}")
    print(f"  Strategy: {metadata['fusion_strategy']}")
```

### Example 5: Configurable Reranker (Production)

```python
from src.app.search.multi_modal_reranker import (
    ConfigurableMultiModalReranker,
    QueryModality
)

# Auto-initializes from config.json
reranker = ConfigurableMultiModalReranker()

# Check configuration
info = reranker.get_reranker_info()
print(f"Enabled: {info['enabled']}")
print(f"Model: {info['model']}")
print(f"Hybrid: {info['use_hybrid']}")

# Rerank (automatically routes to correct strategy)
if info['enabled']:
    reranked = await reranker.rerank(
        query="deep learning frameworks",
        results=search_results,
        modalities=[QueryModality.TEXT, QueryModality.VIDEO],
        context=None
    )
```

### Example 6: Cascade Strategy for Efficiency

```python
from src.app.search.hybrid_reranker import HybridReranker

# Cascade: Fast heuristic filter → Expensive learned rerank
hybrid = HybridReranker(strategy="cascade")

# This processes 100 results → filters to 50 → reranks 50
# Much faster than reranking all 100 with learned model
reranked = await hybrid.rerank_hybrid(
    query="neural networks",
    results=large_result_set,  # 100 results
    modalities=[QueryModality.TEXT],
    context={}
)

# Only top 50 were sent to expensive learned model
print(f"Reranked {len(reranked)} results efficiently")
```

---

## Production Considerations

### Performance Characteristics

**Heuristic Reranking**:
- Latency: 5-10ms for 100 results
- Memory: Minimal (<10MB)
- CPU: Low overhead
- Cost: Free (no API calls)

**Learned Reranking (Cohere)**:
- Latency: 200-500ms for 100 results
- Memory: Minimal (API call)
- Cost: ~$0.002 per 1000 docs
- Rate limits: API-dependent

**Learned Reranking (Ollama Local)**:
- Latency: 100-300ms for 100 results
- Memory: 2-4GB (model loaded)
- CPU/GPU: Medium-High
- Cost: Free (local inference)

**Hybrid Cascade**:
- Latency: 50-150ms for 100→50 results
- Best balance of speed and quality
- Reduces learned model calls by 50%

### Scalability Strategies

1. **Result Limiting**:
```python
# Only rerank top N from initial search
reranker.max_results_to_rerank = 100  # Don't rerank 1000s
```

2. **Cascade for Efficiency**:
```python
# Use cascade to reduce expensive calls
hybrid_reranker.strategy = "cascade"  # Fast filter → learned rerank
```

3. **Async Batch Processing**:
```python
# Rerank multiple queries concurrently
tasks = [
    reranker.rerank(query, results)
    for query, results in query_result_pairs
]
reranked_batches = await asyncio.gather(*tasks)
```

4. **Caching Reranking Results**:
```python
# Cache reranked results for frequent queries
cache_key = f"rerank:{query_hash}:{result_ids_hash}"
cached = await cache.get(cache_key)
if cached:
    return cached
reranked = await reranker.rerank(query, results)
await cache.set(cache_key, reranked, ttl=3600)
```

### Monitoring and Metrics

```python
# Track reranking performance
from src.app.telemetry.context import rerank_span

with rerank_span(
    tenant_id="user_123",
    strategy="hybrid_weighted",
    num_results=100,
    query=query
) as span:
    reranked = await reranker.rerank(query, results, modalities, context)

    # Add metrics to span
    span.set_attribute("reranked_count", len(reranked))
    span.set_attribute("top_score", reranked[0].metadata['reranking_score'])
```

### Error Handling

```python
try:
    reranked = await learned_reranker.rerank(query, results)
except Exception as e:
    logger.error(f"Learned reranking failed: {e}")
    # Fallback to heuristic
    reranked = await heuristic_reranker.rerank_results(
        results, query, modalities, context
    )
```

### Configuration Best Practices

1. **Start with Heuristic**: No costs, good baseline
2. **Add Learned for Critical Queries**: Use cascade to manage costs
3. **Monitor Quality**: Track click-through rates, user satisfaction
4. **A/B Test Strategies**: Compare weighted_ensemble vs cascade vs consensus
5. **Tune Weights**: Adjust `learned_weight` based on quality metrics

---

## Testing

### Unit Tests

**Location**: `tests/search/unit/`

**Key Test Files**:
- `test_multi_modal_reranker.py`: Heuristic scoring logic
- `test_learned_reranker.py`: LiteLLM integration
- `test_hybrid_reranker.py`: Fusion strategies

**Example Test**:
```python
def test_cross_modal_scoring():
    reranker = MultiModalReranker()

    result = SearchResult(
        id="v1",
        title="Test",
        content="Content",
        modality="video",
        score=0.8,
        metadata={}
    )

    # Direct match
    score = reranker._calculate_cross_modal_score(
        result, "query", [QueryModality.VIDEO]
    )
    assert score == 1.0

    # Compatible match (image for video query)
    score = reranker._calculate_cross_modal_score(
        result, "query", [QueryModality.IMAGE]
    )
    assert score == 0.7
```

### Integration Tests

**Location**: `tests/search/integration/`

**Test Scenarios**:
1. End-to-end search with reranking
2. Hybrid strategy comparison
3. Ollama local model integration
4. Cohere API integration
5. Multi-tenant telemetry integration

**Example**:
```python
@pytest.mark.asyncio
async def test_hybrid_weighted_ensemble():
    # Setup
    search_service = SearchService(config, "frame_based_colpali")
    hybrid_reranker = HybridReranker(strategy="weighted_ensemble")

    # Initial search
    results = search_service.search("quantum computing", top_k=50)

    # Rerank
    reranked = await hybrid_reranker.rerank_hybrid(
        query="quantum computing",
        results=results,
        modalities=[QueryModality.VIDEO],
        context={}
    )

    # Verify
    assert len(reranked) <= 50
    assert all("reranking_score" in r.metadata for r in reranked)
    assert all("fusion_strategy" in r.metadata for r in reranked)

    # Check score monotonicity
    scores = [r.metadata['reranking_score'] for r in reranked]
    assert scores == sorted(scores, reverse=True)
```

### Performance Tests

**Benchmarks**:
```python
import timeit

# Heuristic reranking benchmark
def bench_heuristic():
    reranker = MultiModalReranker()
    asyncio.run(reranker.rerank_results(results, query, modalities, {}))

time = timeit.timeit(bench_heuristic, number=100) / 100
print(f"Heuristic: {time*1000:.1f}ms")  # Expected: 5-10ms

# Learned reranking benchmark
def bench_learned():
    reranker = LearnedReranker()
    asyncio.run(reranker.rerank(query, results))

time = timeit.timeit(bench_learned, number=10) / 10
print(f"Learned: {time*1000:.1f}ms")  # Expected: 200-500ms
```

---

## Related Modules

- **Agents Module** (01): VideoSearchAgent uses SearchService
- **Routing Module** (02): Routes queries to search
- **Backends Module** (04): Vespa provides initial search results
- **Cache Module** (10): Caches reranked results
- **Telemetry Module** (05): Tracks reranking performance

---

**Study Tips:**
1. Start with `SearchService` to understand end-to-end search flow
2. Experiment with `MultiModalReranker` to understand heuristic scoring
3. Try `LearnedReranker` with Ollama for local neural reranking
4. Compare `HybridReranker` strategies with A/B tests
5. Monitor production performance with telemetry spans

---

**Last Updated:** 2025-10-07
**Version:** 1.0
**Total Lines:** ~1000
