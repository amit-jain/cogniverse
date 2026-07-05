# Cogniverse Study Guide: Performance & Monitoring

**Module Path:** System-wide

---

## Module Overview

### Purpose
Comprehensive performance monitoring and optimization covering:

- **Performance Targets**: Latency, throughput, resource utilization

- **System Metrics**: Multi-agent orchestration, Vespa backend, embedding models

- **Monitoring Setup**: Phoenix dashboards for telemetry and experiment tracking

- **Optimization**: Batch DSPy compilation (`BootstrapFewShot`) triggered from Phoenix spans, caching strategies

- **Alerting**: Performance degradation patterns and thresholds

### Key Metrics Categories
- **Query Performance**: End-to-end latency, routing decisions
- **Ingestion Performance**: Video processing pipeline, embedding generation
- **Resource Utilization**: CPU, memory, disk, network
- **Search Quality**: MRR, NDCG, precision, recall

---

## Performance Targets

> **Note**: These are aspirational targets. Actual performance depends on hardware, network conditions, embedding model selection, and query complexity. Use Phoenix dashboard to measure actual latencies in your environment.

### Multi-Agent Orchestration

| Component | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **Multi-Agent Orchestrator** | < 50ms | < 100ms | < 150ms |
| **Search Agent** | < 200ms | < 500ms | < 750ms |
| **Routing Decision** | < 10ms | < 25ms | < 50ms |
| **Result Aggregation** | < 20ms | < 50ms | < 100ms |

### Vespa Backend Performance

| Operation | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **BM25 Search** | < 10ms | < 25ms | < 50ms |
| **Float Embedding** | < 50ms | < 100ms | < 200ms |
| **Binary Embedding** | < 20ms | < 50ms | < 100ms |
| **Hybrid Ranking** | < 75ms | < 150ms | < 300ms |
| **Phased Ranking** | < 100ms | < 200ms | < 400ms |

---

## Performance Architecture

```mermaid
flowchart TB
    Query["<span style='color:#000'>User Query</span>"]

    Query --> Routing["<span style='color:#000'>Routing Agent<br/>P50: 10ms<br/>P95: 25ms</span>"]

    Routing --> SearchAgent["<span style='color:#000'>Search Agent<br/>P50: 200ms<br/>P95: 500ms</span>"]

    SearchAgent --> Vespa["<span style='color:#000'>Vespa Search<br/>Binary: 20ms<br/>Float: 50ms</span>"]

    Vespa --> Results["<span style='color:#000'>Results</span>"]

    Results --> Phoenix["<span style='color:#000'>Phoenix Telemetry<br/>Span Export: 10ms</span>"]

    Phoenix --> Metrics["<span style='color:#000'>Metrics Dashboard<br/>Phoenix Analytics</span>"]

    style Query fill:#90caf9,stroke:#1565c0,color:#000
    style Routing fill:#ffcc80,stroke:#ef6c00,color:#000
    style SearchAgent fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Vespa fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Results fill:#a5d6a7,stroke:#388e3c,color:#000
    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
    style Metrics fill:#b0bec5,stroke:#546e7a,color:#000
```

---

## Video Processing Pipeline

### Ingestion Performance

> **Note**: These are approximate targets based on typical hardware (M1/M2 Mac or similar). Actual performance varies significantly based on video duration, resolution, hardware, and selected embedding models.

| Stage | Single Video | Notes |
|-------|--------------|-------|
| **Frame Extraction** | Variable | Depends on video length, resolution, and keyframe extraction strategy |
| **Transcription** | Variable | Depends on audio length and Whisper model size |
| **ColPali Embedding** | Variable | Depends on number of frames/chunks and GPU availability |
| **VideoPrism Embedding** | Variable | Depends on number of chunks and GPU/TPU availability |
| **Vespa Ingestion** | Variable | Depends on batch size, document size, and network latency |

### Embedding Model Performance

| Profile | Embedding Model | Dimensions | Inference Time | Memory |
|---------|-----------------|------------|----------------|--------|
| **video_colpali_smol500_mv_frame** | `TomoroAI/tomoro-colqwen3-embed-4b` | up to 1024 patches × 320-dim (float) / 40-dim (binary) | Variable | 4GB |
| **video_colqwen_omni_mv_chunk_30s** | `TomoroAI/tomoro-colqwen3-embed-4b` | up to 1024 patches × 320-dim (float) / 40-dim (binary) | Variable | 4GB |
| **video_videoprism_base_mv_chunk_30s** | `videoprism_public_v1_base_hf` | 768 | Variable | 3GB |
| **video_videoprism_large_mv_chunk_30s** | `videoprism_public_v1_large_hf` | 1024 | Variable | 4GB |
| **video_videoprism_lvt_base_sv_chunk_6s** | `videoprism_lvt_public_v1_base` | 768 (global, single-vector) | Variable | 3GB |
| **video_videoprism_lvt_large_sv_chunk_6s** | `videoprism_lvt_public_v1_large` | 1024 (global, single-vector) | Variable | 4GB |

> **Note**: Inference times are hardware-dependent and vary based on input size. Both ColPali/ColQwen profiles share the same `TomoroAI/tomoro-colqwen3-embed-4b` model (patch-based multi-vector embeddings, `tensor<bfloat16>(patch{}, v[320])` float / `tensor<int8>(patch{}, v[40])` binary in the Vespa schema); they differ in keyframe-extraction vs 30-second-chunk sampling, not in model architecture. VideoPrism Base/Large produce 768/1024-dim multi-vector chunk embeddings; VideoPrism LVT Base/Large produce 768/1024-dim single-vector (global) embeddings over shorter 6-second chunks.

---

## Query Performance

### End-to-End Latency

```mermaid
sequenceDiagram
    participant User
    participant Router as Routing Agent<br/>P95: 25ms
    participant Search as Search Agent<br/>P95: 500ms
    participant Vespa as Vespa<br/>P95: 100ms
    participant Phoenix as Phoenix<br/>Async

    User->>Router: Query
    Router->>Search: Route Decision
    Search->>Vespa: Search Request
    Vespa-->>Search: Results
    Search-->>User: Response
    Search->>Phoenix: Span Export (async)

    Note over User,Phoenix: Total P95: < 600ms
```

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

---

## Optimization System Performance

### Batch DSPy Optimization Jobs

The Argo-triggered batch jobs in `optimization_cli.py` (query-enhancement
"SIMBA" job, profile-selection optimization, entity-extraction optimization)
all compile their DSPy module the same way, via `_create_teleprompter()`:
`dspy.teleprompt.BootstrapFewShot`, scaled by the number of training
examples pulled from Phoenix spans (plus any approved synthetic demos).

```mermaid
flowchart LR
    Spans["<span style='color:#000'>Phoenix Spans<br/>+ approved synthetic demos</span>"]

    Spans --> Trainset["<span style='color:#000'>Trainset Size Check<br/>_create_teleprompter()</span>"]

    Trainset -->|"< 50 examples"| Small["<span style='color:#000'>BootstrapFewShot<br/>4 demos, 8 labeled, 1 round</span>"]
    Trainset -->|">= 50 examples"| Large["<span style='color:#000'>BootstrapFewShot<br/>8 demos, 16 labeled, 2 rounds</span>"]

    Small --> Artifact["<span style='color:#000'>Compiled Module<br/>saved via ArtifactManager</span>"]
    Large --> Artifact

    style Spans fill:#90caf9,stroke:#1565c0,color:#000
    style Trainset fill:#ffcc80,stroke:#ef6c00,color:#000
    style Small fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Large fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Artifact fill:#a5d6a7,stroke:#388e3c,color:#000
```

| Metric | Target | Description |
|--------|--------|-------------|
| **Optimization Cycle** | Variable | Time to run one Argo batch job (depends on span volume) |
| **Trainset Threshold** | 50 examples | `_create_teleprompter()` scales demo/round counts at this single cutoff |
| **Compile Time** | Variable | `BootstrapFewShot.compile()` time (depends on trainset size and LM latency) |

### DSPy Optimizer Types

`OptimizerType` (`libs/foundation/cogniverse_foundation/config/agent_config.py`)
declares seven optimizer identifiers for `AgentConfig.optimizer_config`:
`bootstrap_few_shot`, `labeled_few_shot`,
`bootstrap_few_shot_with_random_search`, `copro`, `mipro_v2`, `gepa`, `simba`.
`DSPyOptimizerRegistry` (`libs/core/cogniverse_core/common/dspy_module_registry.py`)
maps five of them to real DSPy classes (`dspy.BootstrapFewShot`,
`dspy.LabeledFewShot`, `dspy.BootstrapFewShotWithRandomSearch`,
`dspy.COPRO`, `dspy.MIPROv2`) — `gepa` and `simba` are declared enum values
with no class mapping in that registry. The batch job named "SIMBA"
(`run_simba_optimization`) does not use a SIMBA teleprompter; it compiles
with `BootstrapFewShot` like the other batch jobs. GEPA and true SIMBA
compilation are not currently wired into any automatic dataset-size
selection.

### Optimization Impact

> **Note**: Impact metrics are targets and will vary based on dataset size, quality, and optimization configuration. Actual improvements depend on the amount of routing experience collected.

| Metric | Baseline | Target | Expected Improvement |
|--------|----------|--------|----------------------|
| **Routing Accuracy** | Varies | Improved | Depends on experience |
| **Query Latency** | Varies | Optimized | Caching reduces latency |
| **Cache Hit Rate** | Initial | Improved | Learning improves caching |
| **Error Rate** | Initial | Reduced | Better routing reduces errors |

---

## Memory System Performance

### Mem0 Operations

> **Note**: Mem0 memory system is implemented and uses Vespa backend. Performance depends on Vespa cluster configuration and embedding model (DenseOn, `lightonai/DenseOn`, 768 dimensions).

| Operation | Target | Description |
|-----------|--------|-------------|
| **Memory Add** | Variable | Depends on embedding generation + Vespa write |
| **Memory Search** | Variable | Depends on Vespa vector search performance |
| **Memory Update** | Variable | Depends on memory retrieval + update |
| **Memory Delete** | Variable | Depends on Vespa delete operation |

### Memory Storage

| Metric | Configuration | Description |
|--------|--------------|-------------|
| **Schema per Tenant** | Yes | Each tenant gets dedicated `agent_memories_{tenant_id}` schema |
| **Embedding Model** | DenseOn (`lightonai/DenseOn`) | 768-dimensional embeddings |
| **Backend** | Vespa | Persistent storage with vector search |
| **Isolation** | Per-tenant | Complete memory isolation between tenants |

---

## Multi-Tenant Performance

> **Note**: Multi-tenant architecture is implemented with schema-per-tenant isolation in Vespa. Performance characteristics depend on Vespa cluster size and configuration.

### Tenant Isolation

| Metric | Target | Description |
|--------|--------|-------------|
| **Tenant Creation** | Variable | Complete schema deployment (depends on Vespa) |
| **Tenant Switch** | < 1ms | Context switching overhead (config lookup) |
| **Cross-Tenant Isolation** | 100% | Zero data leakage (enforced by schema isolation) |
| **Concurrent Tenants** | Variable | Limited by Vespa cluster resources |

### Per-Tenant Limits

> **Note**: These are example limits. Actual limits should be configured based on your deployment resources and requirements.

| Resource | Example Default | Example Maximum |
|----------|----------------|-----------------|
| **Documents** | 100,000 | 1,000,000 |
| **QPS** | 10 | 100 |
| **Storage** | 10GB | 100GB |
| **Memory Usage** | 1GB | 10GB |

---

## Phoenix Telemetry Performance

> **Note**: Phoenix telemetry is implemented using OpenTelemetry SDK. Actual performance depends on Phoenix server configuration and network conditions.

### Span Collection

| Metric | Configuration | Description |
|--------|--------------|-------------|
| **Span Export** | Async | Non-blocking async export |
| **Batch Size** | Configurable | Spans batched before export |
| **Export Interval** | Configurable | Batch export frequency |
| **Span Storage** | SQLite DB | Persistent storage in Phoenix |

### Experiment Tracking

> **Note**: Experiment tracking uses Phoenix SDK. Performance depends on dataset size and evaluation complexity.

| Operation | Characteristics | Description |
|-----------|----------------|-------------|
| **Experiment Creation** | Fast | New experiment setup via Phoenix API |
| **Result Recording** | Batched | Per-query result storage |
| **Dataset Upload** | Variable | Depends on dataset size and format |
| **Metric Computation** | Variable | Depends on metric complexity (MRR, NDCG, etc.) |

---

## Resource Utilization

### System Resources

| Resource | Normal Load | Peak Load | Maximum |
|----------|-------------|-----------|---------|
| **CPU Usage** | < 40% | < 70% | < 90% |
| **Memory Usage** | < 8GB | < 16GB | < 32GB |
| **Disk I/O** | < 100MB/s | < 500MB/s | < 1GB/s |
| **Network I/O** | < 50MB/s | < 200MB/s | < 500MB/s |

### Container Resources

Values below are the `charts/cogniverse/values.yaml` defaults. All 23 agents
(routing, search, image search, document, KG/reasoning, federation, research,
coding, etc.) run inside the single `runtime` deployment via the unified
agent dispatcher — there is no per-agent container or replica count.

| Service | CPU Request | Memory Request | CPU Limit | Memory Limit | Replicas |
|---------|-------------|-----------------|-----------|---------------|----------|
| **runtime** (all agents, unified dispatcher) | 2 cores | 4GB | 4 cores | 8GB | 2 (autoscales 2-10) |
| **vespa** | 4 cores | 8GB | 8 cores | 20GB | 1 (static, no HPA) |
| **ingestor** | 1 core | 2GB | 4 cores | 8GB | 2 (static) |
| **dashboard** | 1 core | 2GB | 2 cores | 4GB | 1 (static) |
| **phoenix** | 1 core | 2GB | 2 cores | 4GB | 1 (static) |

> **Note**: Mem0 uses the same Vespa backend, so no separate deployment is needed. Model-inference sidecars (ColPali/ColQwen/VideoPrism/LLM) are configured separately under the `inference` and `llm` chart values and are not shown here.

---

## Search Quality Metrics

> **Note**: These are aspirational targets for search quality. Actual metrics depend on dataset quality, embedding model selection, and evaluation methodology. Use Phoenix experiments to measure actual performance on your queries.

### Reference-Based Evaluation

| Metric | Target | Description |
|--------|--------|-------------|
| **MRR@10** | > 0.8 | Mean Reciprocal Rank (requires ground truth) |
| **NDCG@10** | > 0.85 | Normalized DCG (requires relevance judgments) |
| **Precision@5** | > 0.75 | Top-5 precision (requires ground truth) |
| **Recall@10** | > 0.9 | Top-10 recall (requires complete relevance set) |

### Reference-Free Quality

| Metric | Target | Description |
|--------|--------|-------------|
| **Relevance Score** | > 0.8 | Semantic similarity |
| **Diversity Score** | > 0.6 | Result variety |
| **Distribution Score** | > 0.7 | Score separation |

---

## Monitoring & Alerting

> **Note**: Alerting infrastructure (Prometheus/Grafana) is not currently implemented. Monitoring is done via Phoenix dashboard and manual metrics inspection.

### Performance Degradation Patterns

```mermaid
flowchart TB
    Metrics["<span style='color:#000'>System Metrics<br/>Phoenix Dashboard</span>"]

    Metrics --> Monitor["<span style='color:#000'>Manual Monitoring</span>"]

    Monitor --> Alert1["<span style='color:#000'>High Latency<br/>P95 > 2x target</span>"]
    Monitor --> Alert2["<span style='color:#000'>Low Success Rate<br/>< 95%</span>"]
    Monitor --> Alert3["<span style='color:#000'>Memory Pressure<br/>> 90%</span>"]
    Monitor --> Alert4["<span style='color:#000'>Disk Full<br/>> 90%</span>"]

    Alert1 --> Action1["<span style='color:#000'>Scale replicas<br/>Check cache</span>"]
    Alert2 --> Action2["<span style='color:#000'>Check agent health</span>"]
    Alert3 --> Action3["<span style='color:#000'>Increase memory<br/>Restart</span>"]
    Alert4 --> Action4["<span style='color:#000'>Clean logs<br/>Expand storage</span>"]

    style Metrics fill:#90caf9,stroke:#1565c0,color:#000
    style Monitor fill:#ffcc80,stroke:#ef6c00,color:#000
    style Alert1 fill:#ffcccc,stroke:#c62828,color:#000
    style Alert2 fill:#ffcccc,stroke:#c62828,color:#000
    style Alert3 fill:#ffcccc,stroke:#c62828,color:#000
    style Alert4 fill:#ffcccc,stroke:#c62828,color:#000
    style Action1 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Action2 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Action3 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Action4 fill:#a5d6a7,stroke:#388e3c,color:#000
```

| Issue Pattern | Threshold | Recommended Action |
|---------------|-----------|-------------------|
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

---

## Scaling Targets

### Horizontal Scaling

The Helm chart ships exactly one `HorizontalPodAutoscaler`
(`charts/cogniverse/templates/hpa.yaml`), targeting the `runtime`
deployment — the process that hosts every agent via the unified
dispatcher. Vespa, the dashboard, Phoenix, and the ingestor run with
static `replicaCount` values and are scaled manually.

```mermaid
flowchart LR
    Load["<span style='color:#000'>Increased Load</span>"]

    Load --> Monitor["<span style='color:#000'>runtime HPA<br/>CPU > 70% or Mem > 80%</span>"]

    Monitor --> Scale["<span style='color:#000'>Auto-scale Trigger</span>"]

    Scale --> Runtime["<span style='color:#000'>runtime deployment<br/>2 → 10 replicas</span>"]

    style Load fill:#90caf9,stroke:#1565c0,color:#000
    style Monitor fill:#ffcc80,stroke:#ef6c00,color:#000
    style Scale fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Runtime fill:#a5d6a7,stroke:#388e3c,color:#000
```

| Component | Auto-scale Trigger | Min | Max |
|-----------|-------------------|-----|-----|
| **runtime** (all agents) | CPU > 70% or Memory > 80% | 2 | 10 |
| **vespa** | Not autoscaled — static `replicaCount: 1` | 1 | 1 |
| **ingestor** | Not autoscaled — static `replicaCount: 2` | 2 | 2 |
| **dashboard** / **phoenix** | Not autoscaled — static `replicaCount: 1` | 1 | 1 |

---

## Performance Testing

### Load Testing

There is no dedicated load-testing suite in the repository today. Use
the integration tests under `tests/agents/integration/` and
`tests/ingestion/integration/` with `--durations=0` (below) to measure
per-request latency, or drive concurrent load externally (e.g. `hey`,
`k6`) against the runtime's `/search` and `/ingestion` endpoints.

### Performance Benchmarks

```bash
# Video ingestion - use integration test with timing
JAX_PLATFORM_NAME=cpu uv run pytest tests/ingestion/integration/ -v -k "ingestion" --durations=0

# Query latency - use search tests with timing
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/integration/ -v -k "search" --durations=0
```

### Continuous Performance Monitoring

```bash
# Real-time dashboard
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py
```

---

## Dashboard Setup

### Phoenix Dashboard

Access the Phoenix dashboard for telemetry and experiment tracking:

```bash
# Port-forward if running in Kubernetes
kubectl port-forward -n cogniverse svc/cogniverse-phoenix 6006:6006

# Open dashboard
open http://localhost:6006

# View tenant-specific traces
# Project naming: cogniverse-{tenant_id} (default), or
# cogniverse-{tenant_id}-{service} for a specific service/component
```

**Available Views:**

- Trace timeline and spans

- Request rate and latency metrics

- Error tracking by tenant

- Experiment tracking and evaluation

- Agent routing decisions

- Search quality metrics (MRR, NDCG)

> **Note**: Grafana integration is not currently implemented. All monitoring is done through Phoenix dashboard.

---

**Related Guides:**

- [Architecture Overview](../architecture/overview.md) - System design

- [Instrumentation](../development/instrumentation.md) - Observability patterns

- [Deployment Guide](../modal/deployment_guide.md) - Deployment

---

**Version History:**

| Version | Date | Changes |
|---------|------|---------|
| 3.0 | 2026-02-04 | Updated component names, clarified aspirational vs actual metrics |
| 2.0 | 2025-10-08 | Complete rewrite for multi-agent architecture |
| 1.5 | 2025-09-15 | Added DSPy optimization targets |
| 1.0 | 2025-08-01 | Initial performance targets |
