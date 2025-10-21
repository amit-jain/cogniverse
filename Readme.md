# Cogniverse - Multi-Agent System for Multi-Modal Content

**Version:** 2.0.0 | **Last Updated:** 2025-10-15 | **Status:** Production Ready

Production-ready multi-agent system built with **UV workspace architecture** (5 SDK packages) for multi-modal content analysis and search with automated optimization, evaluation, and complete multi-tenant isolation.

## 🎯 Key Features

- **UV Workspace Architecture**: 5 independent SDK packages (`cogniverse_core`, `cogniverse_agents`, `cogniverse_vespa`, `cogniverse_runtime`, `cogniverse_dashboard`)
- **Multi-Agent Orchestration**: A2A protocol-based coordination of specialized agents
- **Advanced Embeddings**: ColPali frame-level, VideoPrism global, and ColQwen multi-modal embeddings
- **DSPy Optimization**: GEPA experience-guided optimization with Bootstrap, SIMBA, and MIPRO fallbacks
- **Complete Multi-Tenant Isolation**: Schema-per-tenant Vespa, per-tenant Phoenix projects, tenant-scoped memory
- **Phoenix Telemetry**: Comprehensive observability with traces, experiments, and tenant-isolated dashboards
- **Mem0 Memory System**: Context-aware personalization with tenant isolation

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

## 📁 UV Workspace Structure

```
cogniverse/
├── libs/                    # SDK Packages (UV workspace)
│   ├── core/                # cogniverse_core
│   │   └── cogniverse_core/
│   │       ├── config/      # Configuration management
│   │       ├── telemetry/   # Phoenix telemetry (tenant-aware)
│   │       ├── evaluation/  # Experiment tracking
│   │       └── common/      # Cache, memory, utilities
│   ├── agents/              # cogniverse_agents
│   │   └── cogniverse_agents/
│   │       ├── agents/      # Agent implementations
│   │       ├── routing/     # DSPy routing & optimization
│   │       ├── ingestion/   # Video processing pipeline
│   │       ├── search/      # Multi-modal search & reranking
│   │       └── tools/       # A2A tools
│   ├── vespa/               # cogniverse_vespa
│   │   └── cogniverse_vespa/
│   │       └── backends/    # Vespa backend (tenant schemas)
│   ├── runtime/             # cogniverse_runtime
│   │   └── cogniverse_runtime/
│   │       └── server/      # FastAPI server
│   └── dashboard/           # cogniverse_dashboard
│       └── cogniverse_dashboard/
│           └── ui/          # Streamlit dashboard
├── docs/                    # Comprehensive documentation
│   ├── architecture/        # System architecture
│   ├── modules/             # Module documentation
│   ├── operations/          # Deployment & configuration
│   ├── development/         # Development guides
│   ├── diagrams/            # Architecture diagrams
│   └── testing/             # Testing guides
├── scripts/                 # Operational scripts
├── tests/                   # Test suite (by package)
├── configs/                 # Configuration & schemas
├── pyproject.toml           # Workspace root
└── uv.lock                  # Unified lockfile
```

**Package Dependencies:**
```
cogniverse_core (foundation)
    ↑
    ├── cogniverse_agents (depends on core)
    ├── cogniverse_vespa (depends on core)
    ↑
    ├── cogniverse_runtime (depends on core, agents, vespa)
    └── cogniverse_dashboard (depends on core, agents)
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
from cogniverse_core.config import SystemConfig
from cogniverse_agents.agents import VideoSearchAgent

# Configure tenant with complete isolation
config = SystemConfig(
    tenant_id="acme_corp",
    llm_model="gpt-4",
    vespa_url="http://localhost:8080",
    vespa_config_port=19071,
    phoenix_project_name="acme_corp_project",  # Isolated Phoenix project
    phoenix_enabled=True
)

# Create tenant-specific agent
agent = VideoSearchAgent(
    config,
    profile="video_colpali_smol500_mv_frame"
)

# Agent automatically targets schema: video_colpali_smol500_mv_frame_acme_corp
```

### DSPy Optimization
```python
from cogniverse_agents.routing.config import RoutingConfig
from cogniverse_agents.routing.optimization_orchestrator import OptimizationOrchestrator

# Configure GEPA optimizer for tenant
routing_config = RoutingConfig(
    tenant_id="acme_corp",
    optimizer_type="GEPA",
    experience_buffer_size=10000,
    learning_rate=0.001,
    update_interval=300  # 5 minutes
)

orchestrator = OptimizationOrchestrator(config=routing_config)
results = orchestrator.run_optimization()
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

### Architecture
- [Architecture Overview](docs/architecture/overview.md) - System design and multi-tenant architecture
- [SDK Architecture](docs/architecture/sdk-architecture.md) - UV workspace and 5 SDK packages
- [Multi-Tenant Architecture](docs/architecture/multi-tenant.md) - Complete tenant isolation patterns
- [System Flows](docs/architecture/system-flows.md) - 20+ architectural diagrams

### Operations & Deployment
- [Setup & Installation](docs/operations/setup-installation.md) - UV workspace installation
- [Configuration Guide](docs/operations/configuration.md) - Multi-tenant configuration
- [Deployment Guide](docs/operations/deployment.md) - Docker, Modal, Kubernetes
- [Multi-Tenant Operations](docs/operations/multi-tenant-ops.md) - Tenant lifecycle management

### Development
- [Package Development](docs/development/package-dev.md) - SDK package workflows
- [Scripts & Operations](docs/development/scripts-operations.md) - Operational scripts
- [Testing Guide](docs/testing/pytest-best-practices.md) - SDK and multi-tenant testing

### Module Documentation
- [Agents](docs/modules/agents.md) - Agent implementations
- [Routing](docs/modules/routing.md) - Query routing and optimization
- [Ingestion](docs/modules/ingestion.md) - Video processing pipeline
- [Search & Reranking](docs/modules/search-reranking.md) - Multi-modal search
- [Telemetry](docs/modules/telemetry.md) - Phoenix integration
- [Evaluation](docs/modules/evaluation.md) - Experiment tracking
- [Backends](docs/modules/backends.md) - Vespa integration
- [Common](docs/modules/common.md) - Utilities and cache

### Diagrams
- [SDK Architecture Diagrams](docs/diagrams/sdk-architecture-diagrams.md)
- [Multi-Tenant Diagrams](docs/diagrams/multi-tenant-diagrams.md)

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
**Architecture**: UV Workspace (5 SDK Packages)
**Last Updated**: 2025-10-15
**Status**: Production Ready with Complete Multi-Tenant Isolation