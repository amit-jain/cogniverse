# Cogniverse - Self-Optimizing Content Intelligence Platform

**Experience-Guided Multi-Agent System for Multi-Modal Understanding**

**Version:** 2.0.0 | **Last Updated:** 2025-11-13 | **Status:** Production Ready

Production-ready AI platform that continuously learns and improves from real-world interactions. Processes **all content types** (video, audio, images, documents, text, dataframes) using **experience-guided optimization (GEPA)** that adapts in real-time. Achieves **<500ms P95 latency** at **500+ concurrent users** with multiple state-of-the-art embeddings (ColPali, VideoPrism, ColQwen), multi-agent orchestration, cross-modal intelligence fusion, and complete multi-tenant isolation. Built on professional 10-package architecture with comprehensive Phoenix observability.

## 🎯 What Makes Cogniverse Different

- **🧠 Self-Optimizing**: Learns from every interaction using GEPA (Experience-Guided Policy Adaptation) - routing strategies improve continuously from real usage
- **🎭 Multi-Modal Intelligence**: Process any content type (video, audio, images, documents, text, dataframes) with unified understanding
- **🤖 Multi-Agent Orchestration**: DSPy 3.0 A2A protocol-based coordination of specialized agents working together
- **🔀 Cross-Modal Fusion**: Intelligent combination of insights across different modalities for richer understanding
- **⚡ Production Performance**: <500ms P95 latency at 500+ concurrent users with 9 Vespa ranking strategies
- **🎯 Multiple SOTA Models**: ColPali (frame-level), VideoPrism (global+temporal), ColQwen (multi-modal fusion)
- **🏢 Multi-Tenant Ready**: Complete schema-per-tenant isolation with independent Phoenix projects and memory
- **📊 Full Observability**: Comprehensive Phoenix telemetry with traces, experiments, and real-time dashboards
- **🧪 Evaluation Framework**: Provider-agnostic metrics with reference-free, visual LLM, and classical evaluators
- **🏗️ Professional Architecture**: 10-package layered structure (Foundation → Core → Implementation → Application)

## 🎬 Use Cases

**For Individual Developers:**
- Build intelligent content search applications across any modality
- Experiment with multiple state-of-the-art embedding models
- Learn multi-agent AI architectures with production-quality code
- Use locally with Ollama (no API costs)

**For Researchers:**
- Run experiments with different embedding strategies and evaluate results
- Optimize routing agents with synthetic data generation
- Track all experiments with comprehensive Phoenix telemetry
- Publish reproducible results with full observability

**For Teams & Organizations:**
- Deploy multi-tenant SaaS applications with complete data isolation
- Achieve production-scale performance (<500ms P95 at 500+ users)
- Monitor and optimize with comprehensive dashboards
- Scale from prototype to production with professional architecture

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
docker compose -f deployment/docker-compose.yml up -d  # Starts Vespa, Phoenix, Ollama

# Verify services
curl -s http://localhost:8080/ApplicationStatus  # Vespa
curl -s http://localhost:6006/health            # Phoenix
```

### Basic Operations

#### 1. Content Ingestion (All Modalities)
```bash
# Ingest videos with ColPali embeddings
uv run python scripts/run_ingestion.py \
    --video_dir data/videos \
    --profile video_colpali_smol500_mv_frame \
    --tenant default

# Multi-modal multi-profile ingestion (video, audio, images, documents)
uv run python scripts/run_ingestion.py \
    --content_dir data/content \
    --profiles video_colpali_smol500_mv_frame \
               video_videoprism_base_mv_chunk_30s \
               video_colqwen_omni_mv_chunk_30s \
    --tenant default
```

#### 2. Multi-Modal Search
```bash
# Multi-agent intelligent search across all content
uv run python tests/comprehensive_video_query_test_v2.py \
    --profiles video_colpali_smol500_mv_frame \
    --test-multiple-strategies

# Direct API query (text, image, or multi-modal)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{"query": "machine learning tutorial", "modalities": ["video", "document", "image"]}'
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
├── libs/                         # SDK Packages (UV workspace - 10 packages)
│   ├── sdk/                      # cogniverse_sdk (Foundation Layer)
│   │   └── cogniverse_sdk/
│   │       ├── interfaces/       # Backend interfaces
│   │       └── document.py       # Universal document model
│   ├── foundation/               # cogniverse_foundation (Foundation Layer)
│   │   └── cogniverse_foundation/
│   │       ├── config/           # Configuration base
│   │       └── telemetry/        # Telemetry interfaces
│   ├── core/                     # cogniverse_core (Core Layer)
│   │   └── cogniverse_core/
│   │       ├── agents/           # Agent base classes
│   │       ├── registries/       # Component registries
│   │       └── common/           # Shared utilities
│   ├── evaluation/               # cogniverse_evaluation (Core Layer)
│   │   └── cogniverse_evaluation/
│   │       ├── experiments/      # Experiment management
│   │       ├── metrics/          # Provider-agnostic metrics
│   │       └── datasets/         # Dataset handling
│   ├── telemetry-phoenix/        # cogniverse_telemetry_phoenix (Core Layer - Plugin)
│   │   └── cogniverse_telemetry_phoenix/
│   │       ├── provider.py       # Phoenix telemetry provider
│   │       └── evaluation/       # Phoenix evaluation provider
│   ├── agents/                   # cogniverse_agents (Implementation Layer)
│   │   └── cogniverse_agents/
│   │       ├── routing/          # DSPy routing & optimization
│   │       ├── search/           # Multi-modal search & reranking
│   │       └── tools/            # A2A tools
│   ├── vespa/                    # cogniverse_vespa (Implementation Layer)
│   │   └── cogniverse_vespa/
│   │       ├── backends/         # Vespa backend (tenant schemas)
│   │       └── schema/           # Schema management
│   ├── synthetic/                # cogniverse_synthetic (Implementation Layer)
│   │   └── cogniverse_synthetic/
│   │       ├── generators/       # Synthetic data generators
│   │       └── service.py        # Synthetic data service
│   ├── runtime/                  # cogniverse_runtime (Application Layer)
│   │   └── cogniverse_runtime/
│   │       ├── server/           # FastAPI server
│   │       └── ingestion/        # Video processing pipeline
│   └── dashboard/                # cogniverse_dashboard (Application Layer)
│       └── cogniverse_dashboard/
│           ├── phoenix/          # Phoenix dashboards
│           └── streamlit/        # Streamlit UI
├── docs/                         # Comprehensive documentation
│   ├── architecture/             # System architecture
│   ├── modules/                  # Module documentation
│   ├── operations/               # Deployment & configuration
│   ├── development/              # Development guides
│   ├── diagrams/                 # Architecture diagrams
│   └── testing/                  # Testing guides
├── scripts/                      # Operational scripts
├── tests/                        # Test suite (by package)
├── configs/                      # Configuration & schemas
├── pyproject.toml                # Workspace root
└── uv.lock                       # Unified lockfile
```

**Package Dependencies (Layered Architecture):**
```
Foundation Layer:
  cogniverse_sdk (zero internal dependencies)
    ↓
  cogniverse_foundation (depends on sdk)

Core Layer:
  cogniverse_core (depends on sdk, foundation, evaluation)
  cogniverse_evaluation (depends on sdk, foundation)
  cogniverse_telemetry_phoenix (plugin - depends on core, evaluation)

Implementation Layer:
  cogniverse_agents (depends on core)
  cogniverse_vespa (depends on core)
  cogniverse_synthetic (depends on core)

Application Layer:
  cogniverse_runtime (depends on core, agents, vespa, synthetic)
  cogniverse_dashboard (depends on core, evaluation)
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
| **VideoPrism LVT** | Temporal | 768/1024 | Action/motion search |
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
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Configure tenant with complete isolation
config = SystemConfig(
    tenant_id="acme_corp",
    llm_model="gpt-4",
    backend_url="http://localhost",
    backend_port=8080,
    phoenix_url="http://localhost:6006",
)

# Create agent — profile-agnostic, tenant-agnostic at construction
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Search with per-request profile and tenant_id
# Agent automatically targets schema: video_colpali_smol500_mv_frame_acme_corp
results = agent.search(
    query="machine learning tutorial",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="acme_corp",
    top_k=10,
)
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
- [SDK Architecture](docs/architecture/sdk-architecture.md) - UV workspace and 10-package layered architecture
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
docker compose -f deployment/docker-compose.prod.yml up -d

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

See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for detailed contribution guidelines.

### Quick Reference

**Code Standards:**
- Use type hints for all function signatures
- Add docstrings to public functions (Google style)
- Follow PEP 8 with `ruff` for linting
- Use `uv run` for all Python commands

**Commit Standards:**
- Use imperative mood: `Add`, `Fix`, `Update`, `Refactor`, `Remove`
- Subject line: WHAT changed (under 72 chars)
- Body: WHY the change was needed (for non-trivial changes)

**Pre-Commit Checklist:**
- Run `uv run pytest` and ensure 100% pass rate
- Run `uv run ruff check` with no errors
- Update documentation for significant changes
- Never commit failing tests or skip markers

## 📝 License

[License information here]

## 🆘 Support

- GitHub Issues: [Report bugs](https://github.com/org/cogniverse/issues)
- Documentation: [Read the docs](docs/)
- Phoenix Dashboard: http://localhost:8501

---

**Version**: 2.0.0
**Architecture**: UV Workspace (11 Packages - Layered Architecture)
**Last Updated**: 2026-01-29
**Status**: Production Ready with Complete Multi-Tenant Isolation