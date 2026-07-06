# Performance Targets

Performance benchmarks and targets for the Cogniverse multi-agent video search system with DSPy optimization.

## System Architecture Performance

### Multi-Agent Orchestration
| Component | P50 Target | P95 Target | P99 Target |
|-----------|------------|------------|------------|
| **GatewayAgent** (LLM-free GLiNER classify + route) | < 50ms | < 100ms | < 150ms |
| **OrchestratorAgent** (DSPy multi-agent planning) | < 50ms | < 100ms | < 150ms |
| **SearchAgent** (multi-modal Vespa retrieval) | < 200ms | < 500ms | < 750ms |
| **Routing Decision** | < 10ms | < 25ms | < 50ms |
| **Result Aggregation** | < 20ms | < 50ms | < 100ms |

> **Note**: All 23 agents in the roster (`configs/config.json` → `agents.*`) run as
> handlers inside one unified `runtime` deployment (`libs/runtime/cogniverse_runtime/`),
> not as separate per-agent containers — see [Full Agent Roster](#full-agent-roster)
> below and [Container Resources](#container-resources) for the actual deployment
> topology.

### Full Agent Roster

| Agent | Port | Status | Latency Tier |
|-------|------|--------|---------------|
| **gateway_agent** (`GatewayAgent`) | 8000 | enabled | GatewayAgent tier above |
| **orchestrator_agent** (`OrchestratorAgent`) | 8013 | enabled | OrchestratorAgent tier above |
| **search_agent** (`SearchAgent`) | 8002 | enabled | SearchAgent tier above |
| **image_search_agent** (`ImageSearchAgent`) | 8006 | enabled | Simple Text / Complex Multi-Modal (Query Performance) |
| **document_agent** (`DocumentAgent`) | 8008 | enabled | Simple Text / Complex Multi-Modal (Query Performance) |
| **text_analysis_agent** (`TextAnalysisAgent`) | 8003 | enabled | Simple Text / Complex Multi-Modal (Query Performance) |
| **audio_analysis_agent** (`AudioAnalysisAgent`) | 8007 | enabled | Simple Text / Complex Multi-Modal (Query Performance) |
| **summarizer_agent** (`SummarizerAgent`) | 8004 | enabled | Complex Multi-Modal / With Visual Reranking (Query Performance) |
| **detailed_report_agent** (`DetailedReportAgent`) | 8005 | enabled | Complex Multi-Modal / With Visual Reranking (Query Performance) |
| **profile_selection_agent** (`ProfileSelectionAgent`) | 8000 | enabled | Routing Decision tier above |
| **query_enhancement_agent** (`QueryEnhancementAgent`) | 8000 | enabled | See [Query Enhancement Performance](#query-enhancement-performance) |
| **entity_extraction_agent** (`EntityExtractionAgent`) | 8000 | enabled | GatewayAgent tier above (GLiNER/SpaCy fast path) |
| **deep_research_agent** (`DeepResearchAgent`) | 8009 | enabled | Multi-step iterative — seconds, not covered by the ms tiers above |
| **coding_agent** (`CodingAgent`) | 8010 | enabled | Multi-step iterative — seconds, not covered by the ms tiers above |
| **citation_tracing_agent** (`CitationTracingAgent`) | 8019 | disabled | With Memory Lookup (Query Performance) |
| **contradiction_reconciliation_agent** (`ContradictionReconciliationAgent`) | 8020 | disabled | With Memory Lookup (Query Performance) |
| **multi_document_synthesis_agent** (`MultiDocumentSynthesisAgent`) | 8021 | disabled | Complex Multi-Modal (Query Performance) |
| **kg_traversal_agent** (`KnowledgeGraphTraversalAgent`) | 8022 | disabled | With Memory Lookup (Query Performance) |
| **temporal_reasoning_agent** (`TemporalReasoningAgent`) | 8025 | disabled | With Memory Lookup (Query Performance) |
| **knowledge_summarization_agent** (`KnowledgeSummarizationAgent`) | 8026 | disabled | Complex Multi-Modal (Query Performance) |
| **audit_explanation_agent** (`AuditExplanationAgent`) | 8027 | enabled | With Memory Lookup (Query Performance) |
| **cross_tenant_comparison_agent** (`CrossTenantComparisonAgent`) | 8023 | disabled | With Visual Reranking tier (federated multi-tenant reads) |
| **federated_query_agent** (`FederatedQueryAgent`) | 8024 | disabled | With Visual Reranking tier (federated multi-tenant reads) |

> **Source**: agent roster and enabled/disabled flags from `configs/config.json` →
> `agents.*`; class names from `libs/agents/cogniverse_agents/*.py`. "disabled"
> agents ship code and tests but are not started by default
> (`agents.<name>.enabled: false`); their latency tiers are targets for when
> enabled, mapped to the closest existing tier above rather than measured.

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
| **ColPali** (frame-based, `TomoroAI/tomoro-colqwen3-embed-4b`) | Patch-based (1024 patches × 320D) | < 100ms/frame | 2GB |
| **ColQwen2** (chunk-based, 30s chunks, `TomoroAI/tomoro-colqwen3-embed-4b` weights) | Patch-based (1024 patches × 320D) | < 150ms/frame | 4GB |
| **VideoPrism Base** (multi-vector, 30s chunks) | 768 (4096 patches) | < 200ms/chunk | 3GB |
| **VideoPrism Large** (multi-vector, 30s chunks) | 1024 (2048 patches) | < 250ms/chunk | 4GB |
| **VideoPrism LVT Base** (single-vector, 6s chunks) | 768 | < 250ms/chunk | 3GB |
| **VideoPrism LVT Large** (single-vector, 6s chunks) | 1024 | < 300ms/chunk | 4GB |

> **Source**: `configs/config.json` → `video_colpali_smol500_mv_frame`,
> `video_colqwen_omni_mv_chunk_30s`, `video_videoprism_base_mv_chunk_30s`,
> `video_videoprism_large_mv_chunk_30s`, `video_videoprism_lvt_base_sv_chunk_6s`,
> `video_videoprism_lvt_large_sv_chunk_6s` (`schema_config.embedding_dim` /
> `num_patches`). The chunk-based profile's own `schema_config.model_name` is
> `ColQwen2`; both it and the frame-based ColPali profile load
> `TomoroAI/tomoro-colqwen3-embed-4b` weights.

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

> **Note**: GEPA optimizer is registered as `OptimizerType.GEPA` in `libs/foundation/cogniverse_foundation/config/agent_config.py`. Optimizer selection is configured per-tenant; `DSPyAgentOptimizerPipeline` does not auto-select based on dataset size.

### DSPy Optimizer Performance
| Optimizer | Training Time | Memory | Convergence |
|-----------|--------------|--------|-------------|
| **BootstrapFewShot** | < 10 min | 4GB | 10-20 iterations |
| **MIPROv2** | < 60 min | 16GB | 100-200 iterations |
| **COPRO** | < 30 min | 8GB | 50-100 iterations |
| **GEPA** | < 45 min | 12GB | 50-150 iterations |
| **SIMBA** | < 30 min | 8GB | 30-80 iterations |

> **Note**: Available optimizers include: BootstrapFewShot, LabeledFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2 (via `libs/core/cogniverse_core/common/dspy_module_registry.py`), plus GEPA and SIMBA (configured via `libs/foundation/cogniverse_foundation/config/agent_config.py`).

### Query Enhancement Performance
| Component | Training Time | Memory | Description |
|-----------|--------------|--------|-------------|
| **QueryEnhancementAgent** | < 30 min | 8GB | Query enhancement via QueryEnhancementModule (A2A agent) |

> **Note**: SIMBA optimization runs as an Argo batch job (not inline). Real-time enhancement is handled by `QueryEnhancementAgent` (`libs/agents/cogniverse_agents/query_enhancement_agent.py`) using `QueryEnhancementModule`.

### Target Optimization Impact
> **Note**: The following are target improvements for when optimization is fully deployed:

| Metric | Baseline | Target | Target Improvement |
|--------|----------|--------|-------------------|
| **Routing Accuracy** | 75% | 92% | +17% |
| **Query Latency** | 500ms | 350ms | -30% |
| **Cache Hit Rate** | 20% | 45% | +125% |
| **Error Rate** | 5% | 1% | -80% |

## Memory System Performance

> **Note**: Memory system uses Mem0 as the memory framework with Vespa as the vector store backend.

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
| **Batch Size** | 512 | Spans per batch (`BatchExportConfig.max_export_batch_size` default) |
| **Export Interval** | 500ms | Batch export frequency (`BatchExportConfig.schedule_delay_millis` default) |
| **Max Queue Size** | 2048 | In-memory span queue before drops (`BatchExportConfig.max_queue_size` default) |
| **Span Storage** | 30 days | Retention target (not currently enforced by a Phoenix-side TTL) |

> **Source**: `libs/foundation/cogniverse_foundation/telemetry/config.py` →
> `BatchExportConfig` defaults, applied by
> `PhoenixTelemetryProvider.configure_span_export` in
> `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/provider.py`.

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
| Service | CPU Request / Limit | Memory Request / Limit | Replicas |
|---------|---------------------|-------------------------|----------|
| **runtime** (all 23 agents, unified deployment) | 2 / 4 cores | 4Gi / 8Gi | 2 (autoscales 2–10) |
| **vespa** | 4 / 8 cores | 8Gi / 20Gi | 1 |
| **phoenix** | 1 / 2 cores | 2Gi / 4Gi | 1 |

> **Source**: `charts/cogniverse/values.yaml` → `runtime.resources`,
> `runtime.replicaCount`/`runtime.autoscaling`, `vespa.resources`/`vespa.replicaCount`,
> `phoenix.resources`/`phoenix.replicaCount`. There is no per-agent container —
> `libs/runtime/cogniverse_runtime/` hosts every agent from the roster above as
> handlers inside one FastAPI process, and Vespa runs as a single node (no
> separate container/content-cluster split) in this chart. There is no
> standalone "memory service" container; memory operations run in-process via
> `libs/core/cogniverse_core/memory/`.

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
| **runtime** (all 23 agents) | CPU > 70% or memory > 80% | 2 | 10 |
| **vespa** | Not autoscaled — fixed single node | 1 | 1 |
| **phoenix** | Not autoscaled — fixed single replica | 1 | 1 |

> **Source**: `charts/cogniverse/templates/hpa.yaml` (only the `runtime`
> deployment has a `HorizontalPodAutoscaler`, gated on
> `runtime.autoscaling.enabled`) and `charts/cogniverse/values.yaml` →
> `runtime.autoscaling.{minReplicas,maxReplicas,targetCPUUtilizationPercentage,
> targetMemoryUtilizationPercentage}`. `vespa.replicaCount` and
> `phoenix.replicaCount` are fixed values with no HPA in this chart; scaling
> either requires a manual `replicaCount` change and Vespa content
> redistribution.

### Vertical Scaling
| Trigger | Action |
|---------|--------|
| **Memory > 90%** | Double memory allocation |
| **CPU consistently > 80%** | Add 2 more cores |
| **Disk I/O > 80%** | Upgrade to SSD/NVMe |
| **Network saturation** | Upgrade network tier |

## Testing and Validation

### Load Testing

Load testing suite: `tests/routing/integration/` — covers integration scenarios for routing, connectivity, and feature integration. Production-load throughput and latency percentile tests are not yet implemented.

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
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py
```

