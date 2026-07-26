# Cogniverse Agents

**Last Updated:** 2026-07-26
**Layer:** Implementation
**Dependencies:** cogniverse-sdk, cogniverse-core, cogniverse-synthetic, cogniverse-vespa

Agent implementations for routing, multi-modal search, and orchestration.

## Overview

The Agents package sits in the **Implementation Layer**, providing concrete agent implementations that power the Cogniverse platform. It includes routing agents with DSPy optimization, multi-modal search agents (video, image, document), OrchestratorAgent for A2A orchestration, and A2A (Agent-to-Agent) protocol tools.

Agents build on the base classes from `cogniverse-core`. Search agents use
the configured Vespa backends and query encoders from `cogniverse-vespa`.

## Package Structure

```
cogniverse_agents/
├── __init__.py
├── gateway_agent.py         # A2A gateway entry point
├── orchestrator_agent.py    # Central A2A orchestration (OrchestratorAgent)
├── image_search_agent.py    # Image search with visual embeddings
├── document_agent.py        # Document retrieval agent
├── summarizer_agent.py      # Content summarization
├── audio_analysis_agent.py  # Audio content analysis
├── text_analysis_agent.py   # Text analysis agent
├── detailed_report_agent.py # Report generation
├── search_agent.py          # General search agent
├── memory_aware_mixin.py    # Memory integration (MemoryAwareMixin)
├── graph_bindable.py        # Shared knowledge-graph binding
├── adapter_loader.py        # Adapter (LoRA/fine-tune) loading utilities
├── citation_tracing_agent.py
├── contradiction_reconciliation_agent.py
├── cross_tenant_comparison_agent.py
├── federated_query_agent.py
├── kg_traversal_agent.py
├── knowledge_summarization_agent.py
├── multi_document_synthesis_agent.py
├── temporal_reasoning_agent.py
├── wiki/                   # Wiki schema and manager
├── workflow/               # Workflow state, intelligence, telemetry store
├── graph/                  # Knowledge-graph extraction and management
├── inference/              # RLM inference, instrumentation, evaluation
├── routing/                # Routing module
│   ├── dspy_routing_signatures.py  # DSPy routing signatures
│   ├── dspy_relationship_router.py # Relationship-based routing
│   ├── annotation_agent.py         # Annotation generation
│   ├── annotation_queue.py         # UTC-aware review queue
│   ├── annotation_storage.py       # Phoenix annotation persistence
│   ├── llm_auto_annotator.py       # Auto-annotation
│   ├── profile_performance_optimizer.py  # Profile optimization
│   └── xgboost_meta_models.py      # XGBoost meta-models
├── search/                 # Search and reranking utilities
│   ├── hybrid_reranker.py  # Hybrid reranking
│   ├── multi_modal_reranker.py  # Multi-modal reranking
│   ├── learned_reranker.py      # Learned reranking
│   └── rerankers/          # Reranker implementations
├── optimizer/              # DSPy optimization
│   ├── artifact_manager.py       # Versioned telemetry artifacts
│   ├── dspy_agent_optimizer.py  # DSPyAgentPromptOptimizer, DSPyAgentOptimizerPipeline
│   ├── signature_variants.py    # Served signature variants
│   └── strategy_learner.py      # Learned strategy persistence
├── orchestrator/           # Orchestration utilities
│   └── sufficient_context_signature.py
└── approval/               # Approval workflows
```

## Key Modules

### Gateway Agent (`cogniverse_agents.gateway_agent`)

A2A entry point that classifies incoming queries and dispatches to `orchestrator_agent`:

**Key Classes:**
- `GatewayAgent`: Lightweight A2A gateway; parses messages and delegates

### Orchestrator Agent (`cogniverse_agents.orchestrator_agent`)

Central A2A orchestration entry point with DSPy-based multi-agent planning:

**Key Classes:**
- `OrchestratorAgent`: DSPy planning + multi-agent execution coordination

### Video Search (see `cogniverse-vespa`)

Multi-modal video search with ColPali and VideoPrism is implemented in the `cogniverse-vespa` package. From `cogniverse-agents` the following applies:

**Embedding Models:**
- **ColPali**: Document and video frame embeddings
- **VideoPrism**: Video understanding embeddings
- **Smolv500-MV**: Lightweight multi-view embeddings

**Features:**
- Frame-level video search
- Temporal understanding
- Cross-modal text-to-video search
- Video summarization
- Thumbnail generation

**Profiles:**
- `video_colpali_smol500_mv_frame`: Frame-level search
- `video_videoprism`: Video-level embeddings
- `video_hybrid`: Combined approach

### Image Search Agent (`cogniverse_agents.image_search_agent`)

Visual search with configured ColPali-compatible query encoders:

**Features:**
- ColPali query embeddings
- Visual similarity search
- Text-to-image search
- Visual reranking

### Document Agent (`cogniverse_agents.document_agent`)

Traditional document retrieval:

**Features:**
- BM25 text search
- Dense vector search
- Hybrid search (BM25 + vector)
- Document reranking
- Snippet extraction

### OrchestratorAgent (`cogniverse_agents.orchestrator_agent`)

Central A2A entry point with DSPy-based multi-agent orchestration:

**Features:**
- DSPy planning via `OrchestrationModule`
- Agent discovery via `AgentRegistry`
- Parallel execution groups
- A2A calls to downstream agents
- Graceful error handling and result aggregation

**Use Cases:**
- Complex multi-step workflows
- Parallel agent execution
- Sequential agent pipelines
- Conditional agent routing

### Summarizer Agent (`cogniverse_agents.summarizer_agent`)

Content summarization:

**Features:**
- Extractive summarization
- Abstractive summarization (LLM-based)
- Multi-document summarization
- Key point extraction
- Configurable summary length

### Audio Analysis Agent (`cogniverse_agents.audio_analysis_agent`)

Audio content analysis:

**Features:**
- Speech-to-text transcription
- Audio classification
- Speaker diarization
- Sentiment analysis from audio
- Music and sound detection

### Detailed Report Agent (`cogniverse_agents.detailed_report_agent`)

Report generation from search results:

**Features:**
- Structured report generation
- Citation management
- Multi-source synthesis
- Format customization (Markdown, HTML, PDF)

### Routing Module (`cogniverse_agents.routing`)

Comprehensive routing infrastructure:

**Components:**
- DSPy-based routing signatures (`dspy_routing_signatures.py`)
- Relationship-based routing (`dspy_relationship_router.py`)
- `AnnotationAgent`: Generate training annotations
- `LLMAutoAnnotator`: Automatic annotation via LLM
- `XGBoostMetaModels`: XGBoost-based routing models
- Profile performance optimization (`profile_performance_optimizer.py`)

### Search Module (`cogniverse_agents.search`)

Search and reranking utilities:

**Components:**
- `HybridReranker`: Combine multiple ranking signals
- `MultiModalReranker`: Cross-modal reranking
- `LearnedReranker`: ML-based reranking

## Installation

```bash
uv add cogniverse-agents
```

Or with pip:
```bash
pip install cogniverse-agents
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Backend interfaces
- `cogniverse-core`: Base classes and registries
- `cogniverse-synthetic`: Synthetic data generation
- `cogniverse-vespa`: Vespa backends, schemas, and query encoders

**Optional local inference (`torch-local`):**
- `torch==2.8.0`
- `transformers==4.56.2`
- `colpali-engine==0.3.13`
- `sentence-transformers==5.1.1`
- `gliner==0.2.26`

**External (Optimization):**
- `xgboost==3.2.0`
- `scikit-learn==1.8.0`
- `scipy==1.17.1`

**External (NLP):**
- `spacy==3.8.14`
- `langextract==1.2.1`

**External (Tracking):**
- `mlflow==3.11.1`

## Usage Examples

### Video Search Agent

Video search is implemented in the `cogniverse-vespa` package (the backend-specific implementation layer), not in `cogniverse-agents`. Refer to `libs/vespa/README.md` for the concrete `VideoSearchAgent` and its Vespa-backed search API.

### OrchestratorAgent (Direct Use)

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()

# Create orchestrator with agent registry
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)
orchestrator = OrchestratorAgent(deps=OrchestratorDeps(), registry=registry)

# Execute multi-agent workflow — DSPy plans the pipeline automatically
result = await orchestrator.process(
    OrchestratorInput(
        query="show me videos about quantum computing",
        tenant_id="acme:production",
    )
)
```

### Image Search

```python
from cogniverse_agents.image_search_agent import ImageSearchAgent, ImageSearchDeps, ImageSearchInput

# Initialize image search agent
deps = ImageSearchDeps(vespa_endpoint="http://localhost:8080")
image_agent = ImageSearchAgent(deps=deps)

# Text-to-image search via the typed process() interface
result = await image_agent.process(ImageSearchInput(
    query="red sports car on mountain road",
    limit=20,
))
for r in result.results:
    print(r.image_id, r.relevance_score)
```

### Multi-Agent Orchestration

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()

# Create orchestrator with agent registry
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)
orchestrator = OrchestratorAgent(deps=OrchestratorDeps(), registry=registry)

# Execute multi-agent workflow — DSPy plans the pipeline automatically
result = await orchestrator.process(
    OrchestratorInput(
        query="Find videos about neural networks and create a summary",
        tenant_id="acme:production",
    )
)
```

### A2A (Agent-to-Agent) Communication

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.config import TelemetryConfig

config_manager = create_default_config_manager()
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)

# Initialize orchestrator with A2A registry
deps = OrchestratorDeps(
    telemetry_config=TelemetryConfig(),
    llm_config=LLMEndpointConfig(
        model="ollama/qwen3:4b",
        api_base="http://localhost:11434",
    ),
)
orchestrator = OrchestratorAgent(deps=deps, registry=registry)

# Orchestrator routes and delegates via A2A protocol
result = await orchestrator.process(OrchestratorInput(
    query="Find and summarize videos about AI",
    tenant_id="acme:production",
))
```

### DSPy Optimization

```python
from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentPromptOptimizer,
    DSPyAgentOptimizerPipeline,
)

# DSPyAgentPromptOptimizer: optimizes prompts for individual agent predictors
optimizer = DSPyAgentPromptOptimizer(config={})

# DSPyAgentOptimizerPipeline: end-to-end optimization pipeline
# (See cogniverse_agents/optimizer/dspy_agent_optimizer.py for full API)
pipeline = DSPyAgentOptimizerPipeline()
```

## Multi-Modal Support

The Agents package provides first-class multi-modal capabilities:

### Video Search
- **Frame-Level Search**: Search within video frames
- **Temporal Understanding**: Time-aware retrieval
- **Multi-View Embeddings**: Multiple embedding models
- **Cross-Modal**: Text-to-video search

### Image Search
- **Visual Embeddings**: ColPali-compatible configured encoders
- **Text-to-Image**: Natural language image search
- **Image-to-Image**: Visual similarity search

### Audio Analysis
- **Speech-to-Text**: Transcription
- **Audio Classification**: Sound detection
- **Speaker Diarization**: Multi-speaker detection

### Cross-Modal Routing
- Automatic modality detection
- Cross-modal query understanding
- Modality-aware agent selection

## Agent Architecture

### Hierarchy

```
AgentBase (cogniverse_core.agents.base)
    └── A2AAgent (cogniverse_core.agents.a2a_agent)
        ├── GatewayAgent
        ├── OrchestratorAgent (+ MemoryAwareMixin)
        ├── ImageSearchAgent
        ├── DocumentAgent
        ├── AudioAnalysisAgent
        ├── SummarizerAgent (+ MemoryAwareMixin)
        └── DetailedReportAgent (+ RLMAwareMixin, MemoryAwareMixin)

# TextAnalysisAgent extends A2AEndpointsMixin / DynamicDSPyMixin (not A2AAgent)
```

### A2A Protocol

Agents communicate using the A2A (Agent-to-Agent) protocol:
- **Structured Messaging**: Type-safe message passing
- **Tool Invocation**: Agents as tools for other agents
- **Result Aggregation**: Combine results from multiple agents
- **Error Handling**: Graceful error propagation
- **Context Sharing**: Shared context across agents

## Architecture Position

```
Foundation Layer:
  cogniverse-sdk → cogniverse-foundation
    ↓
Core Layer:
  cogniverse-core, cogniverse-evaluation
    ↓
Implementation Layer:
  cogniverse-agents ← YOU ARE HERE
  cogniverse-vespa (backend implementation)
  cogniverse-synthetic (data generation)
    ↓
Application Layer:
  cogniverse-runtime (FastAPI runtime)
  cogniverse-dashboard (Streamlit UI)
```

## Development

```bash
# Install in editable mode
cd libs/agents
uv pip install -e .

# Run tests
uv run pytest tests/agents/ tests/routing/ -v --tb=long

# Run specific agent tests
uv run pytest tests/agents/unit/test_orchestrator_agent.py -v --tb=long

# Static checks
uv run ruff check libs/agents/cogniverse_agents tests/agents tests/routing
uv run ruff format --check libs/agents/cogniverse_agents tests/agents tests/routing
```

## Testing

The agents package includes:
- Unit tests for individual agents
- Integration tests with Vespa backend
- DSPy optimization tests
- A2A protocol tests
- Multi-agent orchestration tests
- Multi-modal search tests
- Routing accuracy tests

## Performance

Latency and quality depend on the configured language-model and Vespa
endpoints, active search profiles, and deployment resources. Use the
evaluation and telemetry workflows in `docs/modules/evaluation.md` and
`docs/modules/telemetry.md` to measure the deployed system.

## Optimization

### Caching
- Query modality caching
- Embedding caching
- Result caching
- Profile selection caching

### Parallelization
- Concurrent agent execution
- Batch processing
- Async I/O

### Model Optimization
- DSPy prompt optimization
- XGBoost meta-models for routing
- Cross-modal optimization

## License

MIT
