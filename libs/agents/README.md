# Cogniverse Agents

**Last Updated:** 2025-11-13
**Layer:** Implementation
**Dependencies:** cogniverse-sdk, cogniverse-core, cogniverse-synthetic

Agent implementations for routing, multi-modal search, and orchestration.

## Overview

The Agents package sits in the **Implementation Layer**, providing concrete agent implementations that power the Cogniverse platform. It includes routing agents with DSPy optimization, multi-modal search agents (video, image, document), OrchestratorAgent for A2A orchestration, and A2A (Agent-to-Agent) protocol tools.

All agents build on the base classes from `cogniverse-core` and leverage multi-modal embeddings for video and image search.

## Package Structure

```
cogniverse_agents/
├── __init__.py
├── gateway_agent.py         # A2A gateway entry point
├── orchestrator_agent.py    # Central A2A orchestration
├── video_search_agent.py    # Multi-modal video search
├── image_search_agent.py    # Image search with visual embeddings
├── document_agent.py        # Document retrieval agent
├── summarizer_agent.py      # Content summarization
├── audio_analysis_agent.py  # Audio content analysis
├── text_analysis_agent.py   # Text analysis agent
├── detailed_report_agent.py # Report generation
├── multi_agent_orchestrator.py  # Advanced orchestration
├── workflow_intelligence.py     # Workflow management
├── query_analysis_tool_v3.py    # Query analysis utilities
├── query_encoders.py        # Query encoding
├── result_aggregator.py     # Result aggregation
├── result_enhancement_engine.py # Result enhancement
├── dspy_agent_optimizer.py  # DSPy optimization
├── dspy_integration_mixin.py    # DSPy integration
├── memory_aware_mixin.py    # Memory integration
├── agent_registry.py        # Agent registration
├── a2a_gateway.py          # A2A gateway
├── routing/                # Routing module
│   ├── contract.py         # RoutingContext wire type
│   ├── router.py           # Core routing logic
│   ├── modality_cache.py   # Query modality caching
│   ├── parallel_executor.py    # Concurrent execution
│   ├── dspy_routing_signatures.py  # DSPy signatures
│   ├── modality_optimizer.py       # Modality optimization
│   ├── cross_modal_optimizer.py    # Cross-modal optimization
│   ├── profile_performance_optimizer.py  # Profile optimization
│   ├── annotation_agent.py         # Annotation generation
│   ├── llm_auto_annotator.py       # Auto-annotation
│   ├── xgboost_meta_models.py      # XGBoost meta-models
│   └── unified_optimizer.py        # Unified routing optimizer
├── search/                 # Search agents
│   ├── base.py
│   ├── hybrid_reranker.py  # Hybrid reranking
│   ├── multi_modal_reranker.py  # Multi-modal reranking
│   ├── learned_reranker.py      # Learned reranking
│   └── rerankers/          # Reranker implementations
├── tools/                  # Agent tools
│   ├── a2a_utils.py        # A2A utilities
│   ├── temporal_extractor.py    # Temporal extraction
│   ├── video_file_server.py     # Video file serving
│   └── video_player_tool.py     # Video playback
├── optimizer/              # Optimization module
├── orchestrator/           # Orchestration utilities
├── query/                  # Query processing
├── results/                # Result processing
├── workflow/               # Workflow management
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

### Routing Contract (`cogniverse_agents.routing.contract`)

Wire type passed from the routing layer to execution agents:

**Key Classes:**
- `RoutingContext`: Pydantic model with `recommended_agent`, `confidence`, `reasoning`, and query enrichment fields

### Video Search Agent (`cogniverse_agents.video_search_agent`)

Multi-modal video search with ColPali and VideoPrism:

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

Visual search with multi-modal embeddings:

**Features:**
- CLIP-based image embeddings
- Visual similarity search
- Text-to-image search
- Image classification
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

### Multi-Agent Orchestrator (`cogniverse_agents.multi_agent_orchestrator`)

Advanced orchestration capabilities:

**Features:**
- Dynamic agent graph construction
- Parallel and sequential execution
- State management across agents
- Workflow visualization
- Performance optimization

### Routing Module (`cogniverse_agents.routing`)

Comprehensive routing infrastructure:

**Components:**
- `Router`: Core routing logic
- `ModalityCache`: Caches query modality predictions
- `ParallelExecutor`: Execute multiple agents concurrently
- `Optimizer`: Routing optimization algorithms
- `CrossModalOptimizer`: Cross-modal understanding
- `AnnotationAgent`: Generate training annotations
- `LLMAutoAnnotator`: Automatic annotation via LLM
- `XGBoostMetaModels`: XGBoost-based routing models
- `MLflowIntegration`: Experiment tracking

### Search Module (`cogniverse_agents.search`)

Search and reranking utilities:

**Components:**
- `HybridReranker`: Combine multiple ranking signals
- `MultiModalReranker`: Cross-modal reranking
- `LearnedReranker`: ML-based reranking

### Tools Module (`cogniverse_agents.tools`)

Agent utilities and A2A tools:

**Components:**
- `A2AUtils`: Agent-to-Agent communication utilities
- `TemporalExtractor`: Extract temporal information
- `VideoFileServer`: Serve video files
- `VideoPlayerTool`: Video playback integration

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

**External (Machine Learning):**
- `torch>=2.5.0`: Deep learning framework
- `transformers>=4.50.0`: Hugging Face models
- `colpali-engine>=0.3.12`: ColPali embeddings
- `sentence-transformers>=5.1.0`: Sentence embeddings

**External (Optimization):**
- `xgboost>=3.0.5`: ML optimization
- `scikit-learn>=1.3.0`: ML utilities
- `scipy>=1.10.0`: Scientific computing

**External (NLP):**
- `spacy>=3.7.0`: NLP processing
- `gliner>=0.2.21`: Named entity recognition
- `langextract>=1.0.6`: Language detection

**External (Tracking):**
- `mlflow>=3.0.0`: Experiment tracking

## Usage Examples

### Video Search Agent

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize dependencies
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# Create agent — profile-agnostic, tenant-agnostic at construction
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Search — profile and tenant_id are per-request (synchronous, not async)
results = agent.search(
    query="machine learning tutorial for beginners",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="acme",
    top_k=10,
)

# Results are SearchResult objects
for result in results:
    print(f"Video: {result.document_id}")
    print(f"Score: {result.score:.3f}")
    print(f"Timestamps: {result.timestamps}")
```

### OrchestratorAgent (Direct Use)

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_agents.agent_registry import AgentRegistry

# Create orchestrator with agent registry
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)
orchestrator = OrchestratorAgent(deps=OrchestratorDeps(), registry=registry)

# Execute multi-agent workflow — DSPy plans the pipeline automatically
result = await orchestrator._process_impl(
    OrchestratorInput(
        query="show me videos about quantum computing",
        tenant_id="acme:production",
    )
)

print(result["summary"])
```

### Image Search

```python
from cogniverse_agents import ImageSearchAgent

# Initialize image search agent
image_agent = ImageSearchAgent(config=config)

# Text-to-image search
results = await image_agent.search(
    query="red sports car on mountain road",
    top_k=20
)

# Image-to-image search
results = await image_agent.search_by_image(
    image_path="example_car.jpg",
    top_k=10
)
```

### Multi-Agent Orchestration

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_agents.agent_registry import AgentRegistry

# Create orchestrator with agent registry
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)
orchestrator = OrchestratorAgent(deps=OrchestratorDeps(), registry=registry)

# Execute multi-agent workflow — DSPy plans the pipeline automatically
result = await orchestrator._process_impl(
    OrchestratorInput(
        query="Find videos about neural networks and create a summary",
        tenant_id="acme:production",
    )
)

print(result["summary"])
```

### Advanced Orchestration

```python
from cogniverse_agents import MultiAgentOrchestrator

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator(config=config)

# Define complex workflow
workflow = {
    "nodes": [
        {"id": "search_videos", "agent": "video_search", "params": {"top_k": 10}},
        {"id": "search_docs", "agent": "document_search", "params": {"top_k": 20}},
        {"id": "aggregate", "agent": "result_aggregator", "depends_on": ["search_videos", "search_docs"]},
        {"id": "summarize", "agent": "summarizer", "depends_on": ["aggregate"]}
    ]
}

# Execute workflow
result = await orchestrator.execute_workflow(
    query="quantum computing applications",
    workflow=workflow
)
```

### A2A (Agent-to-Agent) Communication

```python
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_agents.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.config import TelemetryConfig

config_manager = create_default_config_manager()
registry = AgentRegistry(config_manager=config_manager)

# Initialize orchestrator with A2A registry
deps = OrchestratorDeps(
    telemetry_config=TelemetryConfig(),
    llm_config=LLMEndpointConfig(
        model="ollama/qwen3:4b",
        api_base="http://localhost:11434",
    ),
)
orchestrator = OrchestratorAgent(deps=deps)

# Orchestrator routes and delegates via A2A protocol
result = await orchestrator.process(OrchestratorInput(
    query="Find and summarize videos about AI",
    tenant_id="acme:production",
))
```

### DSPy Optimization

```python
from cogniverse_agents import DSPyAgentOptimizer
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps
from cogniverse_agents.agent_registry import AgentRegistry

# Initialize orchestrator
orchestrator = OrchestratorAgent(
    deps=OrchestratorDeps(),
    registry=AgentRegistry(tenant_id="acme:production", config_manager=config_manager),
)

# Create optimizer targeting the orchestrator
optimizer = DSPyAgentOptimizer(
    agent=orchestrator,
    optimizer_type="GEPA",
    num_iterations=100,
)

# Optimize on training data
optimized_agent = await optimizer.optimize(
    training_data=golden_dataset,
    validation_data=validation_set,
    metrics=["accuracy", "precision", "recall"],
)
```

## Multi-Modal Support

The Agents package provides first-class multi-modal capabilities:

### Video Search
- **Frame-Level Search**: Search within video frames
- **Temporal Understanding**: Time-aware retrieval
- **Multi-View Embeddings**: Multiple embedding models
- **Cross-Modal**: Text-to-video search

### Image Search
- **Visual Embeddings**: CLIP, ColPali, custom models
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
BaseAgent (from cogniverse_core)
    ├── GatewayAgent (A2A entry point)
    ├── OrchestratorAgent (A2A orchestration, DSPy planning)
    ├── VideoSearchAgent (multi-modal video search)
    ├── ImageSearchAgent (visual search)
    ├── DocumentAgent (document retrieval)
    ├── AudioAnalysisAgent (audio processing)
    ├── OrchestratorAgent (A2A entry point, DSPy planning)
    ├── SummarizerAgent (content summarization)
    ├── DetailedReportAgent (report generation)
    ├── TextAnalysisAgent (text processing)
    └── MultiAgentOrchestrator (advanced orchestration)
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
pytest tests/agents/ tests/routing/

# Run specific agent tests
pytest tests/agents/test_video_search_agent.py
pytest tests/agents/unit/test_orchestrator_agent.py
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

- **Query Routing**: <50ms P95 latency
- **Video Search**: <450ms P95 latency (with embeddings)
- **Image Search**: <200ms P95 latency
- **Document Search**: <100ms P95 latency
- **Concurrent Agents**: Up to 10 agents in parallel
- **Cache Hit Rate**: >90% for modality predictions
- **Routing Accuracy**: >95% on golden dataset

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
