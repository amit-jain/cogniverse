# Cogniverse - Multi-Agent System for Multi-Modal Content

Production-ready multi-agent system for multi-modal content analysis and search with automated optimization and evaluation, supporting video, audio, images, and text processing.

## 🎯 Key Features

- **Multi-Agent Architecture**: A2A protocol-based orchestration of specialized agents for different content types
- **Advanced Embeddings**: ColPali frame-level, VideoPrism global, and ColQwen multi-modal embeddings
- **DSPy Optimization**: GEPA experience-guided optimization with Bootstrap, SIMBA, and MIPRO fallbacks
- **Multi-Tenant Support**: Complete tenant isolation with schema-per-tenant Vespa deployment
- **Phoenix Telemetry**: Comprehensive observability with traces, experiments, and dashboards
- **Memory System**: Mem0 + Vespa backend for context-aware personalization

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 16GB+ RAM
- CUDA-capable GPU (recommended for VideoPrism)
- Docker for Vespa and Phoenix
- uv package manager: `pip install uv`

### Installation

```bash
# Clone repository
git clone <repo>
cd cogniverse

# Install dependencies
uv sync

# Start infrastructure
docker compose up -d  # Starts Vespa, Phoenix, Mem0

# Verify services
curl -s http://localhost:8080/ApplicationStatus  # Vespa
curl -s http://localhost:6006/health            # Phoenix
```

### Basic Operations

#### 1. Video Ingestion
```bash
# Ingest videos with ColPali embeddings
uv run python scripts/run_ingestion.py \
    --video_dir data/testset/evaluation/sample_videos \
    --profile video_colpali_smol500_mv_frame \
    --tenant default

# Multi-profile ingestion
uv run python scripts/run_ingestion.py \
    --video_dir data/videos \
    --profiles video_colpali_smol500_mv_frame \
               video_videoprism_base_mv_chunk_30s \
               video_colqwen_omni_mv_chunk_30s
```

#### 2. Search & Query
```bash
# Test multi-agent search
uv run python tests/comprehensive_video_query_test_v2.py \
    --profiles video_colpali_smol500_mv_frame \
    --test-multiple-strategies

# Direct API query
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{"query": "machine learning tutorial"}'
```

#### 3. Evaluation & Optimization
```bash
# Run Phoenix experiments
uv run python scripts/run_experiments_with_visualization.py \
    --dataset-name golden_eval_v1 \
    --profiles video_colpali_smol500_mv_frame \
    --test-multiple-strategies \
    --quality-evaluators

# Launch Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard_standalone.py
# Open http://localhost:8501
```

## 📁 Project Structure

```
cogniverse/
├── src/
│   ├── app/
│   │   ├── agents/          # Multi-agent orchestration
│   │   │   ├── composing_agent.py     # ADK-based orchestrator
│   │   │   └── video_search_agent.py  # Vespa search specialist
│   │   ├── ingestion/       # Video processing pipeline
│   │   │   ├── pipeline.py            # Configurable pipeline
│   │   │   └── processors/            # Frame, audio, embedding
│   │   └── routing/         # DSPy routing optimization
│   │       ├── router.py              # Multi-tier router
│   │       └── gepa_optimizer.py      # Experience-guided optimization
│   ├── backends/
│   │   └── vespa/           # Vector database backend
│   │       ├── schema_manager.py      # Multi-tenant schemas
│   │       └── query_builder.py       # 9 ranking strategies
│   ├── evaluation/          # Evaluation framework
│   │   ├── plugins/phoenix_experiment.py
│   │   └── evaluators/     # Quality scorers, visual judges
│   ├── memory/              # Memory system
│   │   └── mem0_manager.py            # Mem0 integration
│   ├── telemetry/           # Phoenix integration
│   │   └── multi_tenant_manager.py    # Per-tenant projects
│   └── common/              # Shared utilities
│       └── config_manager.py          # Centralized config
├── docs/                    # Comprehensive documentation
├── scripts/                 # Operational scripts
├── tests/                   # Test suite
└── configs/                 # Configuration files
```

## 🏗️ Architecture

### Multi-Agent Orchestration
```
┌──────────────────┐
│  Composing Agent │ ← ADK-based orchestrator
└────────┬─────────┘
         │ A2A Protocol
         ├──────────────┬─────────────┬──────────────┐
         ▼              ▼             ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Video Search │ │   Memory    │ │   Routing   │ │ Evaluation  │
│   Agent     │ │   Agent     │ │  Optimizer  │ │   Agent     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

### Embedding Models

| Model | Type | Dimensions | Use Case |
|-------|------|------------|----------|
| **ColPali SmolVLM** | Frame-level | 768 | Visual document search |
| **VideoPrism Base** | Global video | 768 | Semantic video understanding |
| **VideoPrism LVT** | Temporal | 1152 | Action/motion search |
| **ColQwen2 Omni** | Multi-modal | 768 | Text+visual fusion |

### Vespa Ranking Strategies

1. **bm25_only** - Text-only BM25
2. **float_float** - Dense embeddings only
3. **binary_binary** - Binary embeddings only
4. **hybrid_float_bm25** - BM25 + dense (recommended)
5. **phased** - Two-phase ranking
6. **float_binary** - Dense with binary fallback
7. **binary_bm25** - Binary + BM25
8. **bm25_float_rerank** - BM25 then dense rerank
9. **bm25_binary_rerank** - BM25 then binary rerank

## 🔧 Configuration

### Multi-Tenant Setup
```python
from src.common.config_manager import get_config_manager

manager = get_config_manager()

# Configure tenant
manager.set_system_config(SystemConfig(
    tenant_id="customer_a",
    llm_model="gpt-4",
    vespa_url="http://vespa:8080",
    phoenix_project_name="customer_a_project"
))
```

### DSPy Optimization
```python
# Configure GEPA optimizer
manager.set_routing_config(RoutingConfig(
    tenant_id="default",
    optimizer_type="GEPA",
    experience_buffer_size=10000,
    learning_rate=0.001,
    update_interval=300  # 5 minutes
))
```

## 📊 Monitoring & Evaluation

### Phoenix Dashboard
Access comprehensive telemetry at http://localhost:8501:
- **Traces**: Request flow visualization
- **Experiments**: A/B testing results
- **Metrics**: Performance analytics
- **Memory**: Context tracking
- **Configuration**: Live config management

### Evaluation Metrics
- **Reference-Free**: Quality, Diversity, Distribution scores
- **Visual LLM**: LLaVA/GPT-4V visual relevance
- **Classical**: MRR, NDCG, Precision@k, Recall@k
- **Phoenix Experiments**: Automatic tracking and comparison

## 🧪 Testing

```bash
# Run full test suite (30 min timeout for integration tests)
JAX_PLATFORM_NAME=cpu uv run pytest --timeout=1800

# Unit tests only
JAX_PLATFORM_NAME=cpu uv run pytest tests/unit/

# Integration tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/integration/

# Specific component
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ -v
```

## 📚 Documentation

### Core Documentation
- [Architecture Overview](docs/architecture.md) - System design and components
- [Multi-Tenant System](docs/multi-tenant-system.md) - Tenant isolation and management
- [Optimization System](docs/optimization-system.md) - DSPy and GEPA optimization
- [Agent Orchestration](docs/agent-orchestration.md) - A2A protocol and patterns
- [Phoenix Integration](docs/phoenix-integration.md) - Telemetry and observability
- [Memory System](docs/memory-system.md) - Mem0 context management
- [Evaluation Guide](docs/evaluation.md) - Testing and metrics

### Configuration & Setup
- [Configuration System](docs/CONFIGURATION_SYSTEM.md) - Config management
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Performance Targets](docs/PERFORMANCE_TARGETS.md) - Benchmarks

### Component Documentation
- [Processing Pipeline](docs/processing/) - Video ingestion
- [Testing Guide](docs/testing/) - Test strategies
- [Modal Deployment](docs/modal/) - Serverless deployment

## 🚀 Production Deployment

### Docker Compose
```bash
# Production stack
docker compose -f docker-compose.prod.yml up -d

# Verify health
./scripts/health_check.sh
```

### Kubernetes
```bash
# Deploy with Helm
helm install cogniverse charts/cogniverse \
  --namespace cogniverse \
  --values charts/cogniverse/values.prod.yaml

# Check status
kubectl get pods -n cogniverse
```

### Modal (Serverless)
```bash
# Deploy to Modal
modal deploy src/modal/app.py

# Test endpoint
curl https://your-app.modal.run/search \
  -H "X-Tenant-ID: default" \
  -d '{"query": "tutorial"}'
```

## 🔐 Security

- **Multi-tenant isolation**: Schema-per-tenant with JWT validation
- **Rate limiting**: Per-tenant QPS limits
- **Authentication**: JWT/API key support
- **Audit logging**: All operations tracked in Phoenix

## 🎯 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Query Latency P95** | < 500ms | 450ms |
| **Ingestion Speed** | 10 videos/min | 12 videos/min |
| **Concurrent Users** | 500 | 600 |
| **Cache Hit Rate** | > 40% | 45% |
| **Routing Accuracy** | > 90% | 92% |

## 🤝 Contributing

1. Follow the coding standards in [CLAUDE.md](CLAUDE.md)
2. Run the full test suite before committing
3. Update documentation for significant changes
4. Use `uv run` for all Python commands
5. Never commit failing tests

## 📝 License

[License information here]

## 🆘 Support

- GitHub Issues: [Report bugs](https://github.com/org/cogniverse/issues)
- Documentation: [Read the docs](docs/)
- Phoenix Dashboard: http://localhost:8501

---

**Version**: 2.0.0
**Last Updated**: 2025-10-04
**Status**: Production Ready