# Cogniverse Agents

Agent implementations for routing, search, and orchestration.

## Overview

This package provides concrete agent implementations for the Cogniverse platform, including routing agents with DSPy optimization, video search agents with multi-modal embeddings, composing agents for orchestration, and A2A (Agent-to-Agent) protocol tools.

## Key Components

### Routing Agents (`cogniverse_agents.routing`)

Query routing and DSPy optimization:
- `RoutingAgent`: Main routing agent with GEPA/MIPRO optimization
- `ModalityCache`: Query modality caching
- `ParallelExecutor`: Concurrent agent execution
- Entity extraction and relationship mapping
- Profile selection logic

### Search Agents (`cogniverse_agents.search`)

Multi-modal search implementations:
- `VideoSearchAgent`: ColPali/VideoPrism video search
- `DocumentSearchAgent`: Document retrieval
- `MultiModalReranker`: Cross-modal reranking
- Hybrid ranking strategies

### Orchestration (`cogniverse_agents.orchestration`)

Multi-agent coordination:
- `ComposingAgent`: Orchestrates multiple specialized agents
- Dependency management
- Task decomposition
- Result aggregation

### Tools (`cogniverse_agents.tools`)

Agent communication and utilities:
- A2A protocol implementation
- Agent-to-Agent messaging
- Tool definitions for agent interactions

## Installation

```bash
pip install cogniverse-agents
```

## Dependencies

**Internal:**
- `cogniverse-core`: Base classes and registries

**External:**
- `torch>=2.5.0`: Deep learning framework
- `transformers>=4.50.0`: Hugging Face models
- `colpali-engine>=0.3.12`: ColPali embeddings
- `sentence-transformers>=5.1.0`: Sentence embeddings
- `xgboost>=3.0.5`: ML optimization
- `spacy>=3.7.0`: NLP processing
- `gliner>=0.2.21`: Named entity recognition

## Usage

### Video Search Agent

```python
from cogniverse_agents.search import VideoSearchAgent
from cogniverse_core.config import SystemConfig

# Initialize agent
config = SystemConfig(tenant_id="acme")
agent = VideoSearchAgent(
    config=config,
    profile="video_colpali_smol500_mv_frame"
)

# Execute search
results = await agent.search(
    query="machine learning tutorial",
    top_k=10
)
```

### Routing Agent with DSPy Optimization

```python
from cogniverse_agents.routing import RoutingAgent
from cogniverse_core.config import SystemConfig

# Initialize routing agent
config = SystemConfig(tenant_id="acme")
routing_agent = RoutingAgent(config=config)

# Route query to best agent
decision = await routing_agent.route(
    query="show me videos about AI",
    available_agents=["video_search", "document_search"]
)

print(f"Selected agent: {decision.chosen_agent}")
print(f"Confidence: {decision.confidence}")
```

### Composing Agent for Orchestration

```python
from cogniverse_agents.orchestration import ComposingAgent
from cogniverse_core.config import SystemConfig

# Initialize composing agent
config = SystemConfig(tenant_id="acme")
composing_agent = ComposingAgent(config=config)

# Execute multi-agent workflow
result = await composing_agent.execute(
    task="Find videos about quantum computing and summarize findings",
    agents={
        "video_search": VideoSearchAgent(config),
        "summarizer": SummarizerAgent(config)
    }
)
```

## Agent Architecture

### Agent Hierarchy

```
BaseAgent (from cogniverse_core)
    ├── RoutingAgent (routing & optimization)
    ├── VideoSearchAgent (multi-modal video search)
    ├── DocumentSearchAgent (document retrieval)
    ├── ComposingAgent (multi-agent orchestration)
    └── SummarizerAgent (content summarization)
```

### DSPy Optimization

Agents support multiple DSPy optimizers:
- **GEPA**: Experience-guided optimization
- **MIPRO**: Instruction optimization
- **Bootstrap**: Few-shot learning
- **SIMBA**: Simulation-based optimization

### A2A Protocol

Agents communicate using the A2A (Agent-to-Agent) protocol:
- Structured message passing
- Tool invocation
- Result aggregation
- Error handling

## Architecture Position

Agents sits in the **Implementation Layer** of the Cogniverse architecture:

```
Core Layer:
  cogniverse-core, cogniverse-evaluation
    ↓
Implementation Layer:
  cogniverse-agents ← YOU ARE HERE
  cogniverse-vespa
  cogniverse-synthetic
    ↓
Application Layer:
  cogniverse-runtime, cogniverse-dashboard
```

## Development

```bash
# Install in editable mode
cd libs/agents
pip install -e .

# Run tests
pytest tests/agents/ tests/routing/
```

## Testing

The agents package includes:
- Unit tests for individual agents
- Integration tests with Vespa backend
- DSPy optimization tests
- A2A protocol tests
- Multi-agent orchestration tests

## Performance

- **Query Routing**: <50ms P95 latency
- **Video Search**: <450ms P95 latency (with embeddings)
- **Concurrent Agents**: Up to 10 agents in parallel
- **Cache Hit Rate**: >90% for modality predictions

## License

MIT
