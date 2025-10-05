# System Architecture

## Overview

Cogniverse is a multi-modal video search system with intelligent multi-agent orchestration, DSPy-powered optimization, and comprehensive observability. Built for multi-tenant deployments with schema-per-tenant isolation.

## Modern Architecture Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                          │
│              (CLI, API, Dashboard, Streamlit)                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestration Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Routing      │  │ Video Search │  │ Summarizer/  │         │
│  │ Agent        │→ │ Agent        │→ │ Report Agent │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│           Agent-to-Agent (A2A) Communication Protocol           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Query Analysis & Routing                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │ QueryAnalysisToolV3 (Intent, Complexity, Context)  │         │
│  └────────────────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ RoutingAgent (DSPy-optimized decisions)    │         │
│  └────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Video Processing Pipeline                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │         ProcessingStrategySet (Orchestrator)     │           │
│  │  Segmentation → Transcription → Embedding         │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                   │
│  Supported Models:                                               │
│  • ColPali (smol500) - 1024 patches/frame, 128 dims             │
│  • ColQwen (omni) - Variable patches, multimodal                │
│  • VideoPrism Base - 4096 patches, 768 dims                     │
│  • VideoPrism Large - 2048 patches, 1024 dims                   │
│  • VideoPrism LVT - Single-vector, temporal                     │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Vector Storage & Search                         │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Vespa Multi-Tenant Backend                        │           │
│  │ • Schema-per-tenant isolation                     │           │
│  │ • 9 ranking strategies (BM25, float, binary, etc) │           │
│  │ • Configurable reranking pipelines                │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Optimization & Learning Layer                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Advanced Routing Optimizer (GEPA/MIPRO/SIMBA)    │           │
│  │ • Experience replay buffer                        │           │
│  │ • Multi-stage optimization pipeline               │           │
│  │ • Confidence calibration                          │           │
│  └──────────────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Workflow Intelligence                             │           │
│  │ • Pattern learning from orchestration outcomes    │           │
│  │ • Bidirectional routing ↔ orchestration feedback │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Observability & Evaluation Layer                    │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Phoenix Telemetry (Multi-tenant project isolation)│           │
│  │ • Span tracing per tenant                         │           │
│  │ • Experiment tracking                             │           │
│  │ • Root cause analysis                             │           │
│  └──────────────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Evaluation Framework (Inspect AI + Custom)        │           │
│  │ • Reference-free scorers                          │           │
│  │ • LLM judges (visual + text)                      │           │
│  │ • Retrieval metrics (MRR, NDCG, P@k)             │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Multi-Agent Orchestration

**Agent Types:**
- **Routing Agent**: Analyzes queries and routes to appropriate agents using DSPy-optimized policies
- **Video Search Agent**: Executes video searches with ColPali/VideoPrism models
- **Summarizer Agent**: Generates summaries from search results
- **Report Agent**: Creates detailed reports combining multiple data sources

**A2A Protocol:**
- Standardized message format for inter-agent communication
- Support for sequential, parallel, and hierarchical workflows
- Error handling and timeout management

See: [Agent Orchestration Documentation](agent-orchestration.md)

### 2. Query Analysis & Routing

**QueryAnalysisToolV3:**
- Intent classification (search, summarize, analyze, etc.)
- Complexity assessment (simple, moderate, complex)
- Entity extraction (topics, temporal, named entities)
- Relationship identification
- Thinking rationale for transparency

**RoutingAgent:**
- DSPy-powered decision making
- Confidence scoring for routing choices
- Multi-stage optimization (Bootstrap → SIMBA → MIPRO → GEPA)
- Experience replay for continuous learning
- Workflow pattern recognition

See: [Optimization System Documentation](optimization-system.md)

### 3. Video Processing Pipeline

**ProcessingStrategySet Architecture:**

Configurable processing strategies for different use cases:

**Segmentation Strategies:**
- `FrameSegmentationStrategy`: Extract keyframes at FPS intervals
- `ChunkSegmentationStrategy`: Split video into temporal chunks
- `SingleVectorStrategy`: Process entire video as one unit

**Transcription Strategies:**
- `AudioTranscriptionStrategy`: Whisper-based audio transcription
- `NoTranscriptionStrategy`: Skip transcription

**Description Strategies:**
- `VLMDescriptionStrategy`: Generate visual descriptions with VLM
- `NoDescriptionStrategy`: Skip descriptions

**Embedding Strategies:**
- `MultiVectorEmbeddingStrategy`: Multiple embeddings per video
- `SingleVectorEmbeddingStrategy`: One embedding per video

### 4. Embedding Models

**ColPali (smol500)**:
- Architecture: Vision-language patch-based
- Patches per frame: 1024 (32x32 grid)
- Embedding dimension: 128
- Use case: Fine-grained visual search

**ColQwen (omni)**:
- Architecture: Multimodal vision-language
- Patches: Variable based on content
- Use case: Complex multimodal queries

**VideoPrism Base**:
- Architecture: Temporal-aware vision
- Patches: 4096 per chunk
- Embedding dimension: 768
- Use case: Motion and temporal queries

**VideoPrism Large**:
- Architecture: High-capacity temporal
- Patches: 2048 per chunk
- Embedding dimension: 1024
- Use case: Detailed temporal analysis

**VideoPrism LVT**:
- Architecture: Single-vector temporal
- Embedding dimension: Variable
- Use case: Whole-video similarity

### 5. Vespa Multi-Tenant Backend

**Schema-Per-Tenant Isolation:**
- Each tenant gets dedicated schema
- Lazy schema creation on first request
- Tenant ID extracted from JWT/headers
- Complete data isolation

**9 Ranking Strategies:**
1. `bm25_only` - Pure text search baseline
2. `float_float` - Float embeddings with MaxSim
3. `binary_binary` - Hamming distance ranking
4. `float_binary` - Float query × binary docs
5. `phased` - Two-phase ranking (fast → precise)
6. `hybrid_float_bm25` - Combined semantic + keyword
7. `binary_bm25` - Binary embeddings + text
8. `bm25_binary_rerank` - Text first-pass, binary rerank
9. `bm25_float_rerank` - Text first-pass, float rerank

**Performance Features:**
- Connection pooling
- Batch document feeding
- Compression (gzip/brotli)
- Configurable timeouts
- Retry logic with exponential backoff

See: [Multi-Tenant System Documentation](multi-tenant-system.md)

### 6. Optimization & Learning System

**Advanced Routing Optimizer:**

**Optimizer Selection Logic:**
- Bootstrap (< 10 examples): Few-shot learning
- SIMBA (10-50 examples): Similarity-based optimization
- MIPRO (50-200 examples): Information-theoretic optimization
- GEPA (200+ examples): Gradient-based policy optimization

**Experience Replay:**
- Stores routing decisions + outcomes
- Computes reward signals from search quality
- Triggers optimization at thresholds
- Maintains temporal ordering

**Workflow Intelligence:**
- Learns agent coordination patterns
- Bidirectional feedback: routing ↔ orchestration
- Pattern extraction from successful workflows
- Confidence calibration

See: [Optimization System Documentation](optimization-system.md)

### 7. Phoenix Telemetry & Evaluation

**Multi-Tenant Telemetry:**
- Tenant-specific tracer providers
- Project isolation per tenant
- Lazy initialization with LRU caching
- Synchronous span export for testing
- Async batch export for production

**Span Hierarchy:**
```
cogniverse.routing (root)
├── query.analysis
├── routing.decision
├── video_search.search
│   ├── vespa.query
│   └── vespa.feed
├── summarizer.generate
└── report.create
```

**Experiment Tracking:**
- Dataset creation from golden sets
- Multi-profile evaluation
- Strategy comparison
- Phoenix dashboard visualization

See: [Phoenix Integration Documentation](phoenix-integration.md)

### 8. Evaluation Framework

**Scorer Types:**

**Reference-Free Scorers:**
- `QualityScorer`: Relevance, diversity, distribution
- `VisualJudgeScorer`: VLM-based quality assessment
- `TextJudgeScorer`: LLM-based text evaluation

**Retrieval Metrics:**
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@k, Recall@k
- F1@k

**Phoenix Integration:**
- Automatic experiment creation
- Real-time metrics tracking
- Dataset version management
- A/B test comparison

## Data Flow

### Video Ingestion Flow

```
1. Video Input → Strategy Selection
2. Segmentation (frames/chunks/single)
3. Parallel Processing:
   ├── Audio Transcription (Whisper)
   ├── Visual Description (VLM)
   └── Embedding Generation (ColPali/VideoPrism)
4. Document Building (multi-doc or single-doc)
5. Vespa Feed (tenant-specific schema)
6. Verification & Indexing
```

### Search Request Flow

```
1. Request → Tenant Context Extraction
2. Query Analysis (V3 tool)
3. Routing Decision (DSPy-optimized)
4. Agent Execution (Video Search)
5. Vespa Query (tenant schema + ranking strategy)
6. Result Processing & Formatting
7. Experience Recording (for optimization)
8. Telemetry Export (Phoenix)
```

See: [System Flows Documentation](system-flows.md) for detailed sequence diagrams

## Technology Stack

**Core Framework:**
- DSPy: Agent prompting and optimization
- FastAPI: REST API server
- Streamlit: Dashboard UI

**Models:**
- ColPali (smol500): Vidore/colsmol-500m
- ColQwen: Vidore/colqwen2-v1.0
- VideoPrism: Google/videoprism-base-1b
- Whisper: OpenAI/whisper-base

**Storage:**
- Vespa: Multi-tenant vector database
- Mem0: Agent memory persistence

**Observability:**
- Phoenix (Arize): Distributed tracing
- OpenTelemetry: Instrumentation

**Evaluation:**
- Inspect AI: LLM evaluation framework
- Custom scorers: Quality metrics

## Configuration

Configuration managed via `ConfigManager` with multi-tenant support:

```python
from src.common.config_manager import get_config_manager

config_manager = get_config_manager()
system_config = config_manager.get_system_config(tenant_id="acme")
```

See: [Configuration System Documentation](CONFIGURATION_SYSTEM.md)

## Deployment Options

**Local Development:**
- Docker Compose for Vespa
- Local model inference
- File-based config

**Production:**
- Modal serverless deployment
- Cloud Vespa hosting
- Distributed telemetry
- Multi-region support

See: [Deployment Documentation](deployment.md)

## Related Documentation

- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [Optimization System](optimization-system.md) - GEPA/MIPRO/SIMBA
- [Multi-Tenant System](multi-tenant-system.md) - Tenant isolation
- [Phoenix Integration](phoenix-integration.md) - Telemetry and evaluation
- [Memory System](memory-system.md) - Mem0 integration
- [System Flows](system-flows.md) - Detailed request traces

**Last Updated**: 2025-10-04
