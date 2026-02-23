# Search & Reranking Module Study Guide

**Package:** `cogniverse_agents` (Implementation Layer)
**Module Location:** `libs/agents/cogniverse_agents/search/`

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

## Package Structure

```text
libs/agents/cogniverse_agents/search/
├── __init__.py
├── base.py                    # Base interfaces (SearchResult, SearchBackend)
├── service.py                 # Unified search service
├── multi_modal_reranker.py    # Heuristic multi-modal reranking, ConfigurableMultiModalReranker
├── learned_reranker.py        # LiteLLM neural reranking
├── hybrid_reranker.py         # Hybrid fusion strategies
└── rerankers/                 # Reserved for future reranker implementations
    └── __init__.py
```

---

## Architecture

### Search Service Architecture

```mermaid
flowchart TB
    SearchService["<span style='color:#000'>SearchService<br/>• Query Encoding Coordination<br/>• Backend Search Orchestration<br/>• Multi-Tenant Telemetry Integration</span>"]

    SearchService --> EncoderFactory["<span style='color:#000'>Query Encoder Factory<br/>• ColPali Encoder frame-based<br/>• VideoPrism Encoder chunk/global<br/>• Strategy-Aware Encoding</span>"]

    EncoderFactory --> Backend["<span style='color:#000'>Search Backend Vespa<br/>• Vector Search binary/float<br/>• Hybrid Ranking BM25 + Neural<br/>• Multi-Schema Support</span>"]

    Backend --> Results["<span style='color:#000'>SearchResult List</span>"]

    style SearchService fill:#90caf9,stroke:#1565c0,color:#000
    style EncoderFactory fill:#ffcc80,stroke:#ef6c00,color:#000
    style Backend fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Results fill:#a5d6a7,stroke:#388e3c,color:#000
```

### Reranking Pipeline Architecture

```mermaid
flowchart TB
    InitialResults["<span style='color:#000'>Initial Search Results<br/>• Vespa vector search output top_k = 100<br/>• Multiple modalities: video, image, audio, document</span>"]

    InitialResults --> Configurable["<span style='color:#000'>ConfigurableMultiModalReranker<br/>• Selects strategy based on config.json<br/>• Routes to appropriate reranker</span>"]

    Configurable --> Heuristic["<span style='color:#000'>Heuristic Reranker<br/>• Cross-modal<br/>• Temporal<br/>• Complementary<br/>• Diversity</span>"]
    Configurable --> Learned["<span style='color:#000'>Learned Reranker<br/>• LiteLLM API<br/>• Cohere<br/>• Together AI<br/>• Jina AI<br/>• Ollama local</span>"]
    Configurable --> Hybrid["<span style='color:#000'>Hybrid Reranker<br/>• Weighted<br/>• Cascade<br/>• Consensus</span>"]

    Heuristic --> Reranked["<span style='color:#000'>Reranked Results top_n<br/>• Updated relevance scores<br/>• Modality-aware ordering<br/>• Diversity optimization</span>"]
    Learned --> Reranked
    Hybrid --> Reranked

    style InitialResults fill:#90caf9,stroke:#1565c0,color:#000
    style Configurable fill:#ffcc80,stroke:#ef6c00,color:#000
    style Heuristic fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Learned fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Hybrid fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Reranked fill:#a5d6a7,stroke:#388e3c,color:#000
```

### Hybrid Reranking Strategies

```mermaid
flowchart TB
    subgraph Strategy1["<span style='color:#000'>Strategy: Weighted Ensemble</span>"]
        Heur1["<span style='color:#000'>Heuristic Reranker<br/>score: 0.85<br/>× weight 0.3<br/>= 0.255</span>"]
        Learn1["<span style='color:#000'>Learned Reranker<br/>score: 0.92<br/>× weight 0.7<br/>= 0.644</span>"]
        Heur1 --> Final1["<span style='color:#000'>Final Score<br/>0.255 + 0.644<br/>= 0.899</span>"]
        Learn1 --> Final1
    end

    subgraph Strategy2["<span style='color:#000'>Strategy: Cascade</span>"]
        Step1["<span style='color:#000'>Step 1: Heuristic Filter<br/>100 results<br/>Filter Fast</span>"]
        Step2["<span style='color:#000'>Step 2: Learned Rerank<br/>10 results<br/>Neural Rerank slow</span>"]
        Step1 -->|Top 50%| Step2
        Info1["<span style='color:#000'>Efficient: Reduces expensive<br/>learned model calls</span>"]
    end

    subgraph Strategy3["<span style='color:#000'>Strategy: Consensus</span>"]
        Ranks["<span style='color:#000'>Heuristic Ranks: 1, 5, 3, 10, 2<br/>Learned Ranks: 2, 3, 1, 8, 4</span>"]
        Borda["<span style='color:#000'>Borda Count Scoring<br/>geometric mean</span>"]
        Consensus["<span style='color:#000'>Consensus: Requires agreement from BOTH methods<br/>Result must rank highly in heuristic AND learned</span>"]
        Ranks --> Borda
        Borda --> Consensus
    end

    style Strategy1 fill:#90caf9,stroke:#1565c0,color:#000
    style Heur1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Learn1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Final1 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Strategy2 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Step1 fill:#b0bec5,stroke:#546e7a,color:#000
    style Step2 fill:#b0bec5,stroke:#546e7a,color:#000
    style Info1 fill:#b0bec5,stroke:#546e7a,color:#000
    style Strategy3 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Ranks fill:#b0bec5,stroke:#546e7a,color:#000
    style Borda fill:#b0bec5,stroke:#546e7a,color:#000
    style Consensus fill:#a5d6a7,stroke:#388e3c,color:#000
```

---

## Core Components

### 1. SearchResult (base.py:11-49)

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

### 2. SearchBackend (base.py:52-104)

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

- `get_document(document_id: str) -> Optional[Document]`: Retrieve specific document

- `export_embeddings(schema: str, max_documents: Optional[int], filters: Optional[Dict[str, Any]], include_embeddings: bool = True) -> List[Dict[str, Any]]`: Export embeddings for analysis

---

### 3. SearchService (service.py:14-276)

```python
class SearchService:
    """Unified search service for video retrieval"""

    def __init__(
        self,
        config: Dict[str, Any],
        profile: str,
        tenant_id: str = "default",
        config_manager=None,
        schema_loader=None,
    ):
        """
        Initialize search service

        Args:
            config: Configuration dictionary with backend_url, etc.
            profile: Video processing profile (frame_based_colpali, etc.)
            tenant_id: Tenant identifier for multi-tenancy
            config_manager: ConfigManager instance (REQUIRED - raises ValueError if None)
            schema_loader: SchemaLoader instance (REQUIRED - raises ValueError if None)
        """
        if config_manager is None:
            raise ValueError("config_manager is required")
        if schema_loader is None:
            raise ValueError("schema_loader is required")

        self.config = config
        self.profile = profile
        self.tenant_id = tenant_id
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

### 4. MultiModalReranker (multi_modal_reranker.py:53-395)

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

1. **Cross-Modal Score** (multi_modal_reranker.py:163-204):
   - Direct modality match: 1.0
   - Compatible modalities (video↔image): 0.7
   - Mixed query accepts all: 0.8
   - Unrelated modalities: 0.3

2. **Temporal Score** (multi_modal_reranker.py:206-251):
   - Inside time range (centered): 0.7-1.0
   - <30 days outside: 0.5
   - 30-90 days: 0.3
   - >365 days: 0.1

3. **Complementarity Score** (multi_modal_reranker.py:253-292):
   - Keyword overlap analysis
   - Low overlap = high complementarity
   - Score = 1.0 - avg_overlap

4. **Diversity Score** (multi_modal_reranker.py:294-327):
   - First result from modality: 1.0
   - Second: 0.8, Third: 0.6, Fourth: 0.4, Fifth+: 0.2

---

### 5. LearnedReranker (learned_reranker.py:28-239)

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

    def __init__(
        self,
        model: Optional[str] = None,
        tenant_id: str = "default",
        config_manager: "ConfigManager" = None
    ):
        """
        Initialize from config.json "reranking" section

        Args:
            model: Model name (e.g., "cohere/rerank-english-v3.0")
                   If None, loads from config.json
            tenant_id: Tenant identifier for config scoping
            config_manager: ConfigManager instance (required)

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

### 6. HybridReranker (hybrid_reranker.py:29-274)

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

1. **Weighted Ensemble** (hybrid_reranker.py:148-193):
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

2. **Cascade** (hybrid_reranker.py:195-222):
```python
# Step 1: Heuristic filtering (top 50% or min 10)
heuristic_results = await self.heuristic_reranker.rerank_results(...)
top_k = max(10, len(results) // 2)
filtered = heuristic_results[:top_k]

# Step 2: Learned reranking on filtered set (more efficient)
final_results = await self.learned_reranker.rerank(query, filtered)
```

3. **Consensus** (hybrid_reranker.py:224-274):
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

### 7. ConfigurableMultiModalReranker (multi_modal_reranker.py:397-546)

```python
class ConfigurableMultiModalReranker:
    """
    Facade pattern for reranking with config-based routing

    Modes:
    - Pure heuristic (model="heuristic")
    - Pure learned (model="cohere", "ollama", etc.)
    - Hybrid (use_hybrid=true)
    """

    def __init__(
        self,
        tenant_id: str = "default",
        config_manager: "ConfigManager" = None
    ):
        """
        Initialize configurable reranker from config.json

        Args:
            tenant_id: Tenant identifier for config scoping
            config_manager: ConfigManager instance (REQUIRED - raises ValueError if None)

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
        if config_manager is None:
            raise ValueError("config_manager is required")

        self.enabled = rerank_config.get("enabled", False)
        self.heuristic_reranker = MultiModalReranker()  # Always available
        self.learned_reranker = LearnedReranker(config_manager=config_manager) if model != "heuristic" else None
        self.hybrid_reranker = HybridReranker(config_manager=config_manager) if use_hybrid else None
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
from cogniverse_agents.search.service import SearchService

# Initialize search service
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

config = {
    "backend_url": "http://localhost",
    "backend_port": 8080,
    "search_backend": "vespa"
}
config_manager = create_default_config_manager()
# Note: SchemaLoader is an abstract interface; use concrete implementation
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

service = SearchService(
    config,
    profile="frame_based_colpali",
    config_manager=config_manager,
    schema_loader=schema_loader
)

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
    print(f"Metadata: {result.document.metadata}")
```

### Example 2: Heuristic Multi-Modal Reranking

```python
from datetime import datetime
from cogniverse_agents.search.multi_modal_reranker import (
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
from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_foundation.config.utils import create_default_config_manager

# Initialize with Cohere model
config_manager = create_default_config_manager()
# Option 1: Load model from config.json
reranker = LearnedReranker(
    config_manager=config_manager,
    tenant_id="default"
)

# Option 2: Override with specific model
reranker = LearnedReranker(
    model="cohere/rerank-english-v3.0",
    config_manager=config_manager,
    tenant_id="default"
)

# Rerank with neural model
reranked = await reranker.rerank(
    query="quantum computing breakthroughs",
    results=search_results,
    top_n=10
)

# Check model info
info = reranker.get_model_info()  # Returns Dict[str, Any]
print(f"Model: {info['model']}")
print(f"Max rerank: {info['max_results_to_rerank']}")
```

### Example 4: Hybrid Weighted Ensemble

```python
from cogniverse_agents.search.hybrid_reranker import HybridReranker
from cogniverse_agents.search.multi_modal_reranker import MultiModalReranker
from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_foundation.config.utils import create_default_config_manager

# Initialize hybrid reranker
config_manager = create_default_config_manager()
hybrid = HybridReranker(
    heuristic_reranker=MultiModalReranker(),
    learned_reranker=LearnedReranker(
        config_manager=config_manager,
        tenant_id="default"
    ),
    strategy="weighted_ensemble",
    learned_weight=0.7,
    heuristic_weight=0.3,
    tenant_id="default",
    config_manager=config_manager
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
from cogniverse_agents.search.multi_modal_reranker import (
    ConfigurableMultiModalReranker,
    QueryModality
)
from cogniverse_foundation.config.utils import create_default_config_manager

# Initialize config manager (REQUIRED)
config_manager = create_default_config_manager()

# Auto-initializes from config.json
reranker = ConfigurableMultiModalReranker(
    tenant_id="default",
    config_manager=config_manager
)

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
from cogniverse_agents.search.hybrid_reranker import HybridReranker
from cogniverse_foundation.config.utils import create_default_config_manager

# Cascade: Fast heuristic filter → Expensive learned rerank
config_manager = create_default_config_manager()
hybrid = HybridReranker(
    strategy="cascade",
    tenant_id="default",
    config_manager=config_manager
)

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
# Track reranking performance using search_span
from cogniverse_foundation.telemetry.context import search_span

with search_span(
    tenant_id="user_123",
    query=query,
    top_k=100,
    ranking_strategy="hybrid_weighted"
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

**Location**: `tests/routing/unit/` and `tests/evaluation/unit/`

**Key Test Files**:

- `tests/routing/unit/test_multi_modal_reranker.py`: Heuristic scoring logic
- `tests/routing/unit/test_learned_reranker.py`: LiteLLM integration
- `tests/evaluation/unit/test_reranking.py`: Reranking evaluation

**Example Test**:
```python
from cogniverse_agents.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality,
    RerankerSearchResult  # Dataclass for reranking operations
)

# Note: There are TWO distinct result classes:
# 1. cogniverse_agents.search.base.SearchResult (uses Document object, for API responses)
# 2. cogniverse_agents.search.multi_modal_reranker.RerankerSearchResult (dataclass, for reranking)

def test_cross_modal_scoring():
    """Test cross-modal scoring logic (see tests/routing/unit/test_multi_modal_reranker.py)"""
    reranker = MultiModalReranker()

    # Create test result using RerankerSearchResult dataclass
    result = RerankerSearchResult(
        id="v1",
        title="Test",
        content="Content",
        modality="video",
        score=0.8,
        metadata={}
    )

    # Direct modality match
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

**Location**: `tests/routing/integration/`

**Test Scenarios**:
1. End-to-end search with reranking (`test_learned_reranker_integration.py`)
2. Query expansion with reranking (`test_query_expansion_reranking_integration.py`)
3. Production routing integration (`test_production_routing_integration.py`)
4. Modality optimization integration (`test_modality_optimization_integration.py`)

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

**Total Lines:** ~1100
