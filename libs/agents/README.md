# Cogniverse Agents

**Last Updated:** 2025-11-13
**Layer:** Implementation
**Dependencies:** cogniverse-sdk, cogniverse-core, cogniverse-synthetic

Agent implementations for routing, multi-modal search, and orchestration.

## Overview

The Agents package sits in the **Implementation Layer**, providing concrete agent implementations that power the Cogniverse platform. It includes routing agents with DSPy optimization, multi-modal search agents (video, image, document), composing agents for orchestration, and A2A (Agent-to-Agent) protocol tools.

All agents build on the base classes from `cogniverse-core` and leverage multi-modal embeddings for video and image search.

## Package Structure

```
cogniverse_agents/
├── __init__.py
├── routing_agent.py         # Main routing agent with DSPy optimization
├── video_search_agent.py    # Multi-modal video search
├── image_search_agent.py    # Image search with visual embeddings
├── document_agent.py        # Document retrieval agent
├── composing_agent.py       # Multi-agent orchestration
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
├── a2a_routing_agent.py    # A2A routing
├── routing/                # Routing module
│   ├── base.py
│   ├── router.py           # Core routing logic
│   ├── modality_cache.py   # Query modality caching
│   ├── parallel_executor.py    # Concurrent execution
│   ├── optimizer.py        # Routing optimization
│   ├── dspy_routing_signatures.py  # DSPy signatures
│   ├── modality_optimizer.py       # Modality optimization
│   ├── cross_modal_optimizer.py    # Cross-modal optimization
│   ├── profile_performance_optimizer.py  # Profile optimization
│   ├── annotation_agent.py         # Annotation generation
│   ├── llm_auto_annotator.py       # Auto-annotation
│   ├── xgboost_meta_models.py      # XGBoost meta-models
│   └── mlflow_integration.py       # MLflow tracking
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

### Routing Agent (`cogniverse_agents.routing_agent`)

Intelligent query routing with DSPy optimization:

**Core Features:**
- Multi-modal query understanding (text, image, video, audio)
- DSPy-powered optimization (GEPA, MIPRO, Bootstrap, SIMBA)
- Profile selection based on query characteristics
- Entity extraction and relationship mapping
- Confidence scoring for routing decisions

**Optimization Strategies:**
- **GEPA**: Experience-Guided Preference Aggregation
- **MIPRO**: Multi-Instruction Prompt Optimization
- **Bootstrap**: Few-shot learning from examples
- **SIMBA**: Simulation-Based Optimization

**Key Classes:**
- `RoutingAgent`: Main routing agent with optimization
- `ModalityCache`: Query modality prediction caching
- `ParallelExecutor`: Concurrent agent execution
- `ProfilePerformanceOptimizer`: Profile optimization

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

### Composing Agent (`cogniverse_agents.composing_agent`)

Multi-agent orchestration and coordination:

**Features:**
- Task decomposition
- Agent selection and invocation
- Dependency management
- Result aggregation
- Error handling and recovery

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
from cogniverse_agents import VideoSearchAgent
from cogniverse_core.config import SystemConfig

# Initialize video search agent
config = SystemConfig(tenant_id="acme")
agent = VideoSearchAgent(
    config=config,
    profile="video_colpali_smol500_mv_frame"
)

# Search for videos
results = await agent.search(
    query="machine learning tutorial for beginners",
    top_k=10,
    filters={"duration": {"$lt": 600}}  # Videos under 10 minutes
)

# Results include frame-level matches
for result in results:
    print(f"Video: {result.title}")
    print(f"Score: {result.score:.3f}")
    print(f"Matched frames: {result.matched_frames}")
    print(f"Timestamps: {result.timestamps}")
```

### Routing Agent with DSPy Optimization

```python
from cogniverse_agents import RoutingAgent
from cogniverse_core.config import SystemConfig

# Initialize routing agent
config = SystemConfig(tenant_id="acme")
routing_agent = RoutingAgent(
    config=config,
    optimizer="GEPA"  # Use GEPA optimizer
)

# Route query to best agent
decision = await routing_agent.route(
    query="show me videos about quantum computing",
    available_agents=["video_search", "document_search", "image_search"]
)

print(f"Selected agent: {decision.chosen_agent}")
print(f"Confidence: {decision.confidence:.3f}")
print(f"Reasoning: {decision.reasoning}")

# Execute routing with the selected agent
result = await routing_agent.execute_with_routing(
    query="show me videos about quantum computing"
)
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

### Multi-Agent Composition

```python
from cogniverse_agents import ComposingAgent, VideoSearchAgent, SummarizerAgent

# Initialize agents
video_agent = VideoSearchAgent(config=config)
summarizer = SummarizerAgent(config=config)
composing_agent = ComposingAgent(config=config)

# Compose multi-agent workflow
result = await composing_agent.execute(
    task="Find videos about neural networks and create a summary",
    agents={
        "video_search": video_agent,
        "summarizer": summarizer
    },
    workflow=[
        {"agent": "video_search", "params": {"query": "neural networks", "top_k": 5}},
        {"agent": "summarizer", "params": {"input_from": "video_search"}}
    ]
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
from cogniverse_agents import A2ARoutingAgent, A2AGateway

# Initialize A2A gateway
gateway = A2AGateway(config=config)

# Register agents
gateway.register_agent("video_search", VideoSearchAgent(config))
gateway.register_agent("summarizer", SummarizerAgent(config))

# Use A2A routing agent
a2a_agent = A2ARoutingAgent(config=config, gateway=gateway)

# Agents can invoke each other
result = await a2a_agent.execute(
    query="Find and summarize videos about AI",
    protocol="a2a"
)
```

### DSPy Optimization

```python
from cogniverse_agents.routing import RoutingAgent
from cogniverse_agents import DSPyAgentOptimizer

# Initialize agent
routing_agent = RoutingAgent(config=config)

# Create optimizer
optimizer = DSPyAgentOptimizer(
    agent=routing_agent,
    optimizer_type="GEPA",
    num_iterations=100
)

# Optimize on training data
optimized_agent = await optimizer.optimize(
    training_data=golden_dataset,
    validation_data=validation_set,
    metrics=["accuracy", "precision", "recall"]
)

# Use optimized agent
result = await optimized_agent.route(query="example query")
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
    ├── RoutingAgent (routing & optimization)
    ├── VideoSearchAgent (multi-modal video search)
    ├── ImageSearchAgent (visual search)
    ├── DocumentAgent (document retrieval)
    ├── AudioAnalysisAgent (audio processing)
    ├── ComposingAgent (multi-agent orchestration)
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
pytest tests/agents/test_routing_agent.py
pytest tests/agents/test_video_search_agent.py
pytest tests/agents/test_composing_agent.py
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
