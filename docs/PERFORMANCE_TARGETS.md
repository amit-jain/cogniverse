# Performance Targets

Performance benchmarks and targets for the Cogniverse multi-agent video search system with DSPy optimization.

## System Architecture Performance

### Multi-Agent Orchestration
| Component | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **OrchestratorAgent** | < 50ms | < 100ms | < 150ms |
| **VideoSearchAgent** | < 200ms | < 500ms | < 750ms |
| **Routing Decision** | < 10ms | < 25ms | < 50ms |
| **Result Aggregation** | < 20ms | < 50ms | < 100ms |

### Backend Performance (Vespa)
| Operation | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **BM25 Search** | < 10ms | < 25ms | < 50ms |
| **Float Embedding** | < 50ms | < 100ms | < 200ms |
| **Binary Embedding** | < 20ms | < 50ms | < 100ms |
| **Hybrid Ranking** | < 75ms | < 150ms | < 300ms |
| **Phased Ranking** | < 100ms | < 200ms | < 400ms |

## Video Processing Pipeline

### Ingestion Performance
| Stage | Single Video | Batch (10) | Batch (100) |
|-------|--------------|------------|-------------|
| **Frame Extraction** | < 5s | < 30s | < 4 min |
| **Transcription** | < 10s | < 60s | < 8 min |
| **ColPali Embedding** | < 3s | < 20s | < 3 min |
| **VideoPrism Embedding** | < 8s | < 50s | < 7 min |
| **Backend Ingestion** | < 2s | < 10s | < 90s |

### Embedding Model Performance
| Model | Dimensions | Inference Time | Memory |
|-------|------------|----------------|--------|
| **ColPali SmolVLM 500M** | Patch-based (1024 patches × 128D) | < 100ms/frame | 2GB |
| **ColQwen2 Omni** | Patch-based (1024 patches × 128D) | < 150ms/frame | 4GB |
| **VideoPrism Base** | 768 (global) | < 200ms/chunk | 3GB |
| **VideoPrism LVT** | 1024 (global) | < 300ms/chunk | 4GB |

## Query Performance

### End-to-End Latency
| Query Type | P50 | P95 | P99 |
|------------|-----|-----|-----|
| **Simple Text** | < 100ms | < 200ms | < 400ms |
| **Complex Multi-Modal** | < 300ms | < 600ms | < 1000ms |
| **With Visual Reranking** | < 500ms | < 1000ms | < 1500ms |
| **With Memory Lookup** | < 400ms | < 800ms | < 1200ms |

### Concurrent Load
| Metric | Target | Peak |
|--------|--------|------|
| **Queries Per Second** | 100 QPS | 200 QPS |
| **Concurrent Users** | 500 | 1000 |
| **Success Rate** | > 99% | > 95% |
| **Timeout Rate** | < 0.1% | < 1% |

## Optimization System Performance

> **Note**: GEPA (Experience-Guided Policy Adaptation) optimizer is implemented in `libs/agents/cogniverse_agents/routing/advanced_optimizer.py` and automatically selected when dataset size >= 200 examples.

### DSPy Optimizer Performance
| Optimizer | Training Time | Memory | Convergence |
|-----------|--------------|--------|-------------|
| **BootstrapFewShot** | < 10 min | 4GB | 10-20 iterations |
| **MIPROv2** | < 60 min | 16GB | 100-200 iterations |
| **COPRO** | < 30 min | 8GB | 50-100 iterations |
| **GEPA** | < 45 min | 12GB | 50-150 iterations |
| **SIMBA** | < 30 min | 8GB | 30-80 iterations |

> **Note**: Available optimizers include: BootstrapFewShot, LabeledFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2 (via `libs/core/cogniverse_core/registries/dspy_registry.py`), plus GEPA and SIMBA (via `libs/agents/cogniverse_agents/routing/advanced_optimizer.py`).

### Query Enhancement Performance
| Component | Training Time | Memory | Description |
|-----------|--------------|--------|-------------|
| **SIMBA Query Enhancer** | < 30 min | 8GB | Query enhancement using SIMBA optimizer with memory-augmented learning |

> **Note**: SIMBA Query Enhancer (`libs/agents/cogniverse_agents/routing/simba_query_enhancer.py`) uses the `dspy.teleprompt.SIMBA` optimizer for similarity-based memory-augmented query enhancement.

### Target Optimization Impact
> **Note**: The following are target improvements for when optimization is fully deployed:

| Metric | Baseline | Target | Target Improvement |
|--------|----------|--------|-------------------|
| **Routing Accuracy** | 75% | 92% | +17% |
| **Query Latency** | 500ms | 350ms | -30% |
| **Cache Hit Rate** | 20% | 45% | +125% |
| **Error Rate** | 5% | 1% | -80% |

## Memory System Performance

> **Note**: Memory system uses custom abstractions. Mem0 is configurable as a backend but not the only option.

### Memory Operations
| Operation | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **Memory Add** | < 50ms | < 100ms | < 200ms |
| **Memory Search** | < 30ms | < 75ms | < 150ms |
| **Memory Update** | < 40ms | < 90ms | < 180ms |
| **Memory Delete** | < 20ms | < 50ms | < 100ms |

> **Implementation**: Memory operations are implemented in `libs/core/cogniverse_core/memory/` with support for multiple backends including Vespa-based storage.

### Memory Storage Targets
| Metric | Target | Maximum |
|--------|--------|---------|
| **Memories per User** | 1000 | 10,000 |
| **Memory Size** | 1KB | 10KB |
| **Total Storage** | 100GB | 1TB |
| **Retention Period** | 90 days | 365 days |

## Multi-Tenant Performance

### Tenant Isolation
| Metric | Target | Description |
|--------|--------|-------------|
| **Tenant Creation** | < 5s | Complete schema deployment |
| **Tenant Switch** | < 1ms | Context switching overhead |
| **Cross-Tenant Isolation** | 100% | Zero data leakage |
| **Concurrent Tenants** | 1000 | Active tenant limit |

### Per-Tenant Limits
| Resource | Default | Maximum |
|----------|---------|---------|
| **Documents** | 100,000 | 1,000,000 |
| **QPS** | 10 | 100 |
| **Storage** | 10GB | 100GB |
| **Memory Usage** | 1GB | 10GB |

## Telemetry Performance (Phoenix)

### Span Collection
| Metric | Target | Description |
|--------|--------|-------------|
| **Span Export** | < 10ms | Async export latency |
| **Batch Size** | 1000 | Spans per batch |
| **Export Interval** | 5s | Batch export frequency |
| **Span Storage** | 30 days | Retention period |

### Experiment Tracking
| Operation | Target | Description |
|-----------|--------|-------------|
| **Experiment Creation** | < 100ms | New experiment setup |
| **Result Recording** | < 50ms | Per-query result storage |
| **Dataset Upload** | < 1s/1000 rows | Bulk data ingestion |
| **Metric Computation** | < 500ms | Aggregate metrics calc |

## Resource Utilization

### System Resources
| Resource | Normal Load | Peak Load | Maximum |
|----------|-------------|-----------|---------|
| **CPU Usage** | < 40% | < 70% | < 90% |
| **Memory Usage** | < 8GB | < 16GB | < 32GB |
| **Disk I/O** | < 100MB/s | < 500MB/s | < 1GB/s |
| **Network I/O** | < 50MB/s | < 200MB/s | < 500MB/s |

### Container Resources
| Service | CPU Request | Memory Request | Replicas |
|---------|-------------|----------------|----------|
| **OrchestratorAgent** | 2 cores | 4GB | 3 |
| **VideoSearchAgent** | 4 cores | 8GB | 5 |
| **Vespa Container** | 8 cores | 16GB | 3 |
| **Vespa Content** | 4 cores | 32GB | 5 |
| **Phoenix** | 2 cores | 4GB | 1 |
| **Memory Service** | 2 cores | 4GB | 2 |

## Evaluation Metrics

### Search Quality
| Metric | Target | Description |
|--------|--------|-------------|
| **MRR@10** | > 0.8 | Mean Reciprocal Rank |
| **NDCG@10** | > 0.85 | Normalized DCG |
| **Precision@5** | > 0.75 | Top-5 precision |
| **Recall@10** | > 0.9 | Top-10 recall |

### Reference-Free Quality
| Metric | Target | Description |
|--------|--------|-------------|
| **Relevance Score** | > 0.8 | Semantic similarity |
| **Diversity Score** | > 0.6 | Result variety |
| **Distribution Score** | > 0.7 | Score separation |

### Visual LLM Evaluation
| Metric | Target | Description |
|--------|--------|-------------|
| **Visual Relevance** | > 0.85 | Frame-query match |
| **Content Quality** | > 0.8 | Information quality |
| **Evaluation Time** | < 2s | Per-result evaluation |

## Monitoring and Alerting

### Critical Alerts
| Alert | Threshold | Action |
|-------|-----------|--------|
| **High Latency** | P95 > 2x target | Scale replicas, check cache |
| **Low Success Rate** | < 95% | Check agent health |
| **Memory Pressure** | > 90% usage | Increase memory, restart |
| **Disk Full** | > 90% usage | Clean logs, expand storage |

### Performance Degradation
| Metric | Warning | Critical |
|--------|---------|----------|
| **Query Latency** | +50% baseline | +100% baseline |
| **Error Rate** | > 2% | > 5% |
| **Cache Hit Rate** | < 30% | < 20% |
| **CPU Usage** | > 80% | > 95% |

## Scaling Targets

### Horizontal Scaling
| Component | Auto-scale Trigger | Min | Max |
|-----------|-------------------|-----|-----|
| **OrchestratorAgent** | CPU > 70% | 2 | 10 |
| **VideoSearchAgent** | CPU > 70% | 3 | 20 |
| **Vespa Container** | QPS > 100 | 3 | 10 |
| **Vespa Content** | Storage > 80% | 3 | 20 |

### Vertical Scaling
| Trigger | Action |
|---------|--------|
| **Memory > 90%** | Double memory allocation |
| **CPU consistently > 80%** | Add 2 more cores |
| **Disk I/O > 80%** | Upgrade to SSD/NVMe |
| **Network saturation** | Upgrade network tier |

## Testing and Validation

### Load Testing

Load testing suite: `tests/routing/integration/test_production_load.py` — covers 100 QPS throughput, concurrent request handling, sustained load, and latency percentile validation.

### Performance Benchmarks

```bash
# Video ingestion - use integration test with timing
JAX_PLATFORM_NAME=cpu uv run pytest tests/ingestion/integration/ -v -k "ingestion" --durations=0

# Query latency - use search tests with timing
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/integration/ -v -k "search" --durations=0
```

### Continuous Performance Monitoring
```bash
# Real-time dashboard (available)
uv run streamlit run scripts/phoenix_dashboard_standalone.py
```

