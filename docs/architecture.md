# System Architecture

## Overview

Cogniverse is a multi-modal video search system built on a modular, plugin-based architecture that supports multiple video understanding models and search strategies.

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
├─────────────────────────────────────────────────────────┤
│                   Query Routing Layer                    │
│         (GLiNER → LLM → Keyword Fallback)               │
├─────────────────────────────────────────────────────────┤
│                  Processing Pipeline                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│   │ Ingestion│  │ Embedding│  │ Strategy Manager │    │
│   │ Pipeline │→ │Generator │→ │  (Pluggable)     │    │
│   └──────────┘  └──────────┘  └──────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                   Storage Backend                        │
│                  (Vespa Vector DB)                      │
├─────────────────────────────────────────────────────────┤
│                 Evaluation Framework                     │
│            (Phoenix + Inspect AI)                       │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Query Routing Layer

**Three-tier routing architecture:**
- **Tier 1 (Fast Path)**: GLiNER entity extraction (<10ms estimated)
- **Tier 2 (Slow Path)**: LLM reasoning (~100ms estimated)
- **Tier 3 (Fallback)**: Keyword matching (<1ms estimated)

**Key Features:**
- Auto-optimization based on query patterns
- Configurable confidence thresholds
- Performance monitoring and statistics

### 2. Ingestion Pipeline

**Strategy-driven processing:**
- Frame-based extraction (ColPali)
- Chunk-based extraction (30s segments)
- Single-vector extraction (entire video)

**Processing Flow:**
1. Video segmentation
2. Audio transcription (parallel)
3. Visual description generation (parallel)
4. Embedding generation
5. Document creation
6. Backend storage

### 3. Embedding Models

**Supported Models:**
- **ColPali**: Vision-language, ~32 patches/frame, 128 dims
- **ColQwen**: Alternative vision-language, variable patches
- **VideoPrism Base**: 4096 patches/frame, 768 dims
- **VideoPrism Large**: 2048 patches/frame, 1024 dims

**Remote Inference Support:**
- Infinity server integration
- Modal serverless deployment
- Custom HTTP endpoints

### 4. Vespa Backend

**9 Ranking Strategies:**
1. `bm25_only` - Text search baseline
2. `float_float` - ColPali float embeddings
3. `binary_binary` - Hamming distance
4. `float_binary` - Float query × unpacked binary
5. `phased` - Two-phase ranking
6. `hybrid_float_bm25` - Combined approach
7. `binary_bm25` - Binary + text
8. `bm25_binary_rerank` - Text → binary rerank
9. `bm25_float_rerank` - Text → float rerank

**Performance Optimization:**
- Configurable feed parameters
- Batch processing
- Connection pooling
- Compression options

### 5. Evaluation Framework

**Multi-layered evaluation:**
- **Quality Metrics**: Relevance, diversity, distribution
- **LLM Judges**: Visual and text evaluation
- **Retrieval Metrics**: MRR, NDCG, Precision@k
- **Phoenix Integration**: Tracing and experiments

## Data Flow

### Ingestion Flow

```
Video File
    ↓
Profile Selection → Strategy Execution → Embedding Generation
    ↓                      ↓                     ↓
Schema Selection → Document Creation → Vespa Feed
```

### Query Flow

```
User Query
    ↓
Router Analysis → Strategy Selection → Backend Query
    ↓                    ↓                  ↓
Entity Extraction → Search Execution → Result Ranking
    ↓
Response Generation
```

## Deployment Architecture

### Development Setup
```yaml
services:
  - Vespa: localhost:8080
  - Phoenix: localhost:6006
  - Ollama: localhost:11434
  - Application: localhost:8000
```

### Production Setup
```yaml
services:
  vespa:
    replicas: 3
    resources:
      memory: 16GB
      cpu: 8 cores
  
  application:
    replicas: 2
    autoscaling:
      min: 2
      max: 10
      target_cpu: 70%
  
  inference:
    provider: modal
    gpu: T4
    autoscaling: true
```

## Plugin Architecture

### Adding New Components

**New Processor:**
1. Create in `processors/` directory
2. Inherit from `BaseProcessor`
3. Auto-discovery includes it

**New Strategy:**
1. Add to `strategies.py`
2. Inherit from `BaseStrategy`
3. Configure in profile

**New Evaluator:**
1. Implement scorer interface
2. Register with framework
3. Add to configuration

## Performance Characteristics (Estimated)

<!-- TODO: Update with actual measured performance metrics -->

| Component | Latency* | Throughput* | Resource Usage |
|-----------|----------|-------------|----------------|
| GLiNER Routing | <10ms | 1000 qps | CPU only |
| LLM Routing | ~100ms | 100 qps | GPU preferred |
| ColPali Embedding | ~500ms/frame | 10 fps | GPU required |
| VideoPrism | ~1s/video | 5 videos/s | GPU required |
| Vespa Query | <50ms | 500 qps | Memory intensive |

*Performance estimates - actual values depend on hardware and configuration

## Security Considerations

- No secrets in code or configs
- Environment variable management
- API key rotation support
- Secure remote inference endpoints
- Input validation at all layers

## Monitoring & Observability

- OpenTelemetry instrumentation
- Phoenix tracing integration
- Prometheus metrics export
- Custom performance dashboards
- Error tracking and alerting

## Scalability

### Horizontal Scaling
- Stateless application servers
- Distributed Vespa cluster
- Load-balanced inference

### Vertical Scaling
- GPU allocation for models
- Memory for vector storage
- CPU for routing layer

## Future Architecture

### Planned Enhancements
- Multi-modal fusion strategies
- Real-time video processing
- Federated search across backends
- Advanced caching layers
- Stream processing support