# Welcome to Cogniverse

**Version 2.0.0** | **10-Package UV Workspace Architecture** | **Production Ready** | **Last Updated: 2025-11-13**

## Multi-Agent RAG System for Multi-Modal Content Analysis and Search

Cogniverse is a production-ready multi-agent RAG (Retrieval-Augmented Generation) system for intelligent processing and search across multi-modal content with complete multi-tenant isolation.

---

## What is Cogniverse?

Cogniverse is a technical framework for building production multi-agent systems that process and search across multi-modal content including:

- **Video**: Frame-level analysis with ColPali/VideoPrism embeddings, temporal segmentation, keyframe extraction
- **Audio**: Transcription with Whisper, audio embeddings, speaker detection
- **Images**: Visual similarity search, content-based retrieval, image embeddings
- **Documents**: PDF/text processing, semantic search, document chunking
- **Text**: Natural language understanding, entity extraction, BM25 and dense retrieval
- **Dataframes**: Structured data analysis and integration with unstructured content

### Core Capabilities

**Multi-Modal Content Processing**
- Frame-level and chunk-level video segmentation strategies
- Audio transcription with timestamp alignment
- Image and document embedding generation
- Text analysis with entity extraction and relationship detection
- Unified document model across all modalities

**Multi-Agent Orchestration**
- Agent-to-Agent (A2A) protocol for inter-agent communication
- Routing agent with DSPy 3.0 modules for intelligent query routing
- Video, audio, image, document, and text search agents
- Composing agent for multi-step workflows with dependency management
- Parallel agent execution with timeout and circuit breaker patterns

**Experience-Guided Optimization (GEPA)**
- Continuous learning from query routing decisions
- Experience buffer for storing successful routing patterns
- DSPy-based optimization with GEPA, MIPRO, SIMBA, Bootstrap
- Synthetic data generation for training optimizers
- Automated prompt and module optimization

**Multi-Tenant Architecture**
- Schema-per-tenant physical isolation in Vespa
- Tenant-aware configuration and memory systems
- Independent telemetry streams per tenant
- JWT-based tenant authentication and authorization
- Complete data isolation with no cross-tenant leakage

**Production Observability**
- OpenTelemetry integration with Phoenix
- Distributed tracing across all agent interactions
- Experiment tracking with provider-agnostic evaluation framework
- Metrics collection for latency, accuracy, and resource utilization
- UMAP visualization of multi-modal embeddings

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd cogniverse

# Install UV workspace (all 10 packages)
uv sync

# Start infrastructure services
docker compose up -d

# Verify services
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:6006/health             # Phoenix
curl http://localhost:11434/api/tags          # Ollama

# Ingest sample videos with ColPali embeddings
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --profile video_colpali_smol500_mv_frame \
  --tenant default

# Run comprehensive multi-agent test
JAX_PLATFORM_NAME=cpu uv run python tests/comprehensive_video_query_test_v2.py \
  --profiles video_colpali_smol500_mv_frame \
  --test-multiple-strategies
```

## 10-Package Layered Architecture

Cogniverse uses a UV workspace with 10 packages organized in 4 layers:

```
┌─────────────────────────────────────────────┐
│          APPLICATION LAYER                   │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │ cogniverse-     │  │ cogniverse-      │  │
│  │ runtime         │  │ dashboard        │  │
│  │ (FastAPI Server)│  │ (Streamlit UI)   │  │
│  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│        IMPLEMENTATION LAYER                  │
│  ┌─────────┐  ┌────────┐  ┌──────────────┐ │
│  │ agents  │  │ vespa  │  │ synthetic    │ │
│  └─────────┘  └────────┘  └──────────────┘ │
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│             CORE LAYER                       │
│  ┌───────┐ ┌─────────────┐ ┌──────────────┐│
│  │ core  │ │ evaluation  │ │ telemetry-   ││
│  │       │ │             │ │ phoenix      ││
│  └───────┘ └─────────────┘ └──────────────┘│
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│         FOUNDATION LAYER                     │
│  ┌────────────┐    ┌──────────────────────┐ │
│  │ sdk        │    │ foundation           │ │
│  │ (Interfaces)│    │ (Config & Telemetry)│ │
│  └────────────┘    └──────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Package Responsibilities:**

| Package | Layer | Purpose |
|---------|-------|---------|
| **cogniverse-sdk** | Foundation | Backend interfaces, Document model (zero dependencies) |
| **cogniverse-foundation** | Foundation | Config base, telemetry interfaces |
| **cogniverse-core** | Core | Base agent classes, registries, memory management |
| **cogniverse-evaluation** | Core | Provider-agnostic experiments, metrics, datasets |
| **cogniverse-telemetry-phoenix** | Core | Phoenix telemetry provider (plugin via entry points) |
| **cogniverse-agents** | Implementation | Routing, video search, orchestration agents |
| **cogniverse-vespa** | Implementation | Vespa backend, tenant schema management |
| **cogniverse-synthetic** | Implementation | Synthetic data generation for optimizers |
| **cogniverse-runtime** | Application | FastAPI server, ingestion pipeline, middleware |
| **cogniverse-dashboard** | Application | Streamlit UI, Phoenix analytics |

## Technical Use Cases

### Multi-Modal Video Search
- Frame-level embeddings with ColPali (multi-vector) for visual document understanding
- Global video embeddings with VideoPrism for semantic video understanding
- Chunk-level embeddings with ColQwen for temporal segment retrieval
- Audio transcription with Whisper for text-based video search
- Hybrid ranking strategies: BM25 + dense, binary + float, phased ranking

### Intelligent Query Routing
- DSPy 3.0 modules for structured LLM programming
- GLiNER for entity extraction (people, places, concepts)
- Relationship detection between entities for relevance boosting
- Query modality classification (factual, conceptual, visual, temporal)
- GEPA optimizer for continuous learning from routing decisions

### Multi-Agent Orchestration
- A2A protocol for structured agent communication
- Parallel agent execution with timeout handling
- Task dependency graph resolution
- Memory-aware agents with Mem0 integration
- Circuit breaker pattern for fault tolerance

### Multi-Tenant Content Management
- Schema-per-tenant physical isolation in Vespa
- Per-tenant configuration profiles for video processing
- Independent telemetry streams and Phoenix projects
- JWT-based tenant authentication
- Tenant-specific memory and cache instances

## Technical Components

### Agent System (cogniverse-agents)
- **RoutingAgent**: DSPy 3.0-based intelligent query routing with entity extraction, relationship detection, and query enhancement
- **VideoSearchAgent**: Multi-modal video search with ColPali/VideoPrism embeddings and hybrid ranking
- **ComposingAgent**: Multi-agent orchestration with task dependency resolution and parallel execution
- **BaseAgent**: Foundation class with health checks, telemetry, and memory awareness mixins

### Backend System (cogniverse-vespa)
- **TenantSchemaManager**: Schema-per-tenant lifecycle management with lazy creation
- **VespaSearchClient**: Tenant-aware search client with 9 ranking strategies (BM25, dense, binary, hybrid, phased)
- **VespaPyClient**: Batch ingestion with connection pooling and retry logic
- **JSONSchemaParser**: Parses Vespa schema JSON for multi-tenant deployment

### Ingestion Pipeline (cogniverse-runtime)
- **FrameSegmentationStrategy**: Extract keyframes at configurable FPS with scene change detection
- **ChunkSegmentationStrategy**: Temporal chunking for 30-second segments with overlap
- **MultiVectorEmbeddingStrategy**: Generate per-frame embeddings with ColPali
- **GlobalEmbeddingStrategy**: Generate video-level embeddings with VideoPrism
- **Audio transcription**: Whisper integration with timestamp alignment

### Optimization Framework (cogniverse-synthetic + cogniverse-agents)
- **GEPA Optimizer**: Experience-guided prompt and module optimization with continuous learning
- **SyntheticDataService**: Generate training data for GEPA/MIPRO/Bootstrap/SIMBA optimizers
- **ProfileSelector**: LLM-based profile selection from backend content
- **BackendQuerier**: Sample real content from Vespa for synthetic generation

### Evaluation Framework (cogniverse-evaluation + cogniverse-telemetry-phoenix)
- **Provider-agnostic metrics**: Accuracy, MRR, NDCG, Precision@K independent of telemetry backend
- **Phoenix evaluation provider**: Phoenix-specific experiment tracking and annotation
- **Dataset management**: Load and validate evaluation datasets with ground truth
- **Experiment tracking**: Track experiments with metadata, parameters, and results

## Technology Stack

**Core Technologies:**
- **Python**: 3.12+ (3.11+ for sdk and foundation packages)
- **Package Manager**: UV workspace with unified lockfile
- **Build System**: Hatchling for all packages

**AI/ML Stack:**
- **LLM Framework**: DSPy 3.0 with declarative modules and optimizers (GEPA, MIPRO, SIMBA, Bootstrap)
- **Embeddings**: ColPali (vidore/colsmol-500m), VideoPrism (scenic-t5/base, scenic-t5/lvt), ColQwen (vidore/colqwen2-v1.0)
- **Entity Extraction**: GLiNER for zero-shot NER
- **Audio Transcription**: OpenAI Whisper (base, small, medium, large-v3)
- **LLM Inference**: Ollama (local) with Llama 3.2, Llama 3.1, Qwen models

**Search & Storage:**
- **Vector Database**: Vespa 8.x with 9 ranking strategies (BM25, float, binary, hybrid)
- **Memory System**: Mem0 with Vespa backend for context storage
- **Document Model**: Universal SDK document model across all backends

**Observability:**
- **Telemetry**: OpenTelemetry with Phoenix (Arize) collector
- **Tracing**: Distributed traces with span collection
- **Experiments**: Provider-agnostic evaluation framework with Phoenix provider
- **Visualization**: UMAP embeddings, Streamlit dashboards

**Deployment:**
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with Helm charts
- **Serverless**: Modal deployment support
- **CI/CD**: Automated testing and deployment workflows

## Documentation

**For Users:**
- [User Guide](USER_GUIDE.md) - Complete guide for using Cogniverse
- [Setup & Installation](operations/setup-installation.md) - Installation and configuration
- [Configuration Guide](operations/configuration.md) - Multi-tenant configuration
- [Troubleshooting](operations/troubleshooting.md) - Common issues and solutions

**For Developers:**
- [Developer Guide](DEVELOPER_GUIDE.md) - Development workflows and best practices
- [Architecture Overview](architecture/overview.md) - System architecture and design
- [SDK Architecture](architecture/sdk-architecture.md) - UV workspace and package structure
- [Module Documentation](modules/) - Package-specific technical documentation

**For DevOps:**
- [Deployment Guide](operations/deployment.md) - Production deployment options
- [Docker Deployment](operations/docker-deployment.md) - Docker Compose setup
- [Kubernetes Deployment](operations/kubernetes-deployment.md) - K8s with Helm charts
- [Multi-Tenant Operations](operations/multi-tenant-ops.md) - Tenant lifecycle management

---

**Version**: 2.0.0
**Architecture**: UV Workspace (10 Packages - Layered Architecture)
**Last Updated**: 2025-11-13
**Status**: Production Ready