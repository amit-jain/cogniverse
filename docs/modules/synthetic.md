# Synthetic Module

**Package:** `cogniverse_synthetic`
**Location:** `libs/synthetic/cogniverse_synthetic/`
**Purpose:** DSPy-driven synthetic data generation for optimizer training
**Last Updated:** 2026-01-13

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Generator Architecture](#generator-architecture)
   - [BaseGenerator](#basegenerator)
   - [Generator Types](#generator-types)
4. [DSPy Integration](#dspy-integration)
   - [DSPy Signatures](#dspy-signatures)
   - [DSPy Modules](#dspy-modules)
5. [Backend Querier](#backend-querier)
6. [Service Layer](#service-layer)
7. [REST API](#rest-api)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [Testing](#testing)

---

## Overview

The Synthetic module provides **training data generation** for DSPy optimizers:

- **DSPy-Driven Generation**: Uses DSPy signatures and modules for LLM-driven query generation
- **Backend-Agnostic Sampling**: Works with any vector database (Vespa, Pinecone, Weaviate)
- **Optimizer Support**: Generates data for Modality, CrossModal, Routing, and Workflow optimizers
- **Validated Output**: Built-in validation with retry logic ensures quality
- **REST API**: FastAPI router for HTTP endpoints

The generated data is used to train and optimize DSPy modules for better routing, search, and workflow decisions.

---

## Package Structure

```
cogniverse_synthetic/
├── __init__.py                      # Package exports
├── schemas.py                       # Pydantic schemas for all optimizer types
├── registry.py                      # Optimizer configuration registry
├── service.py                       # Main orchestrator service
├── api.py                           # FastAPI router
├── profile_selector.py              # LLM/rule-based profile selection
├── backend_querier.py               # Backend-agnostic content sampling
├── dspy_signatures.py               # DSPy signatures for LLM generation
├── dspy_modules.py                  # Validated DSPy modules with retry
├── generators/                      # Generator implementations
│   ├── __init__.py
│   ├── base.py                      # BaseGenerator abstract class
│   ├── modality.py                  # Modality routing generator
│   ├── cross_modal.py               # Cross-modal query generator
│   ├── routing.py                   # Routing strategy generator
│   └── workflow.py                  # Workflow pattern generator
├── approval/                        # HITL approval utilities
│   ├── __init__.py
│   ├── confidence_extractor.py      # Extract confidence scores
│   └── feedback_handler.py          # Handle approval feedback
└── utils/                           # Utilities
    ├── __init__.py
    ├── pattern_extraction.py        # Extract patterns from content
    └── agent_inference.py           # Infer agents from content
```

---

## Generator Architecture

### BaseGenerator

All generators extend the abstract `BaseGenerator` class:

```python
from cogniverse_synthetic.generators.base import BaseGenerator
from typing import List, Dict, Any
from pydantic import BaseModel

class CustomGenerator(BaseGenerator):
    """Custom generator for specific optimizer."""

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
    ) -> List[BaseModel]:
        """Generate synthetic examples from sampled content."""
        self.validate_inputs(sampled_content, target_count)

        examples = []
        for content in sampled_content[:target_count]:
            # Generate example using DSPy module
            example = await self._generate_single(content)
            examples.append(example)

        return examples
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `generate(sampled_content, target_count, **kwargs)` | Generate synthetic examples (abstract) |
| `validate_inputs(sampled_content, target_count)` | Validate generator inputs |
| `get_generator_info()` | Get generator metadata |

### Generator Types

**ModalityGenerator** - Generate modality routing examples:
```python
from cogniverse_synthetic.generators.modality import ModalityGenerator

generator = ModalityGenerator()
examples = await generator.generate(
    sampled_content=documents,
    target_count=100,
    modalities=["video", "image", "pdf"]
)

# Each example has:
# - query: Natural language query
# - modality: Ground truth modality
# - ground_truth: Expected answer
```

**CrossModalGenerator** - Generate cross-modal query examples:
```python
from cogniverse_synthetic.generators.cross_modal import CrossModalGenerator

generator = CrossModalGenerator()
examples = await generator.generate(
    sampled_content=documents,
    target_count=50,
    source_modality="video",
    target_modality="image"
)
```

**RoutingGenerator** - Generate routing strategy examples:
```python
from cogniverse_synthetic.generators.routing import RoutingGenerator

generator = RoutingGenerator()
examples = await generator.generate(
    sampled_content=documents,
    target_count=75,
    route_types=["direct", "fallback", "hybrid"]
)
```

**WorkflowGenerator** - Generate workflow pattern examples:
```python
from cogniverse_synthetic.generators.workflow import WorkflowGenerator

generator = WorkflowGenerator()
examples = await generator.generate(
    sampled_content=documents,
    target_count=100,
    workflow_types=["ingestion", "search", "analysis"]
)
```

---

## DSPy Integration

### DSPy Signatures

Signatures define the input/output contract for LLM-driven generation:

**GenerateModalityQuery** - Generate queries for specific modalities:
```python
import dspy
from cogniverse_synthetic.dspy_signatures import GenerateModalityQuery

class QueryGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateModalityQuery)

    def forward(self, modality: str, topics: str, context: str) -> str:
        result = self.generate(
            modality=modality,
            topics=topics,
            context=context
        )
        return result.query
```

**GenerateEntityQuery** - Generate queries that include specific entities:
```python
from cogniverse_synthetic.dspy_signatures import GenerateEntityQuery

class EntityQueryGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateEntityQuery)

    def forward(self, topics: str, entities: str, entity_types: str):
        result = self.generate(
            topics=topics,
            entities=entities,
            entity_types=entity_types
        )
        return result.query, result.reasoning
```

**InferAgentFromModality** - Infer the correct agent for a query:
```python
from cogniverse_synthetic.dspy_signatures import InferAgentFromModality

class AgentInferrer(dspy.Module):
    def __init__(self):
        self.infer = dspy.ChainOfThought(InferAgentFromModality)

    def forward(self, modality: str, query: str, available_agents: str):
        result = self.infer(
            modality=modality,
            query=query,
            available_agents=available_agents
        )
        return result.agent_name, result.reasoning
```

### DSPy Modules

Modules wrap signatures with validation and retry logic:

```python
from cogniverse_synthetic.dspy_modules import ValidatedQueryGenerator

# Generator with automatic validation and 3 retries
generator = ValidatedQueryGenerator()

# Generates validated query (retries on validation failure)
query = generator.generate(
    document="Machine learning video tutorial",
    modality="video"
)
```

---

## Backend Querier

The `BackendQuerier` abstracts content sampling from any vector database:

```python
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_vespa import VespaBackend

# Initialize with any backend
backend = VespaBackend(config)
querier = BackendQuerier(backend=backend)

# Sample documents
documents = await querier.sample_documents(
    schema_name="video_colpali_mv_frame",
    count=10,
    modality="video"
)

# Sample with filters
documents = await querier.sample_documents(
    schema_name="video_colpali_mv_frame",
    count=10,
    modality="video",
    filters={"topic": "machine_learning"}
)
```

**Supported Backends:**
- VespaBackend
- PineconeBackend
- WeaviateBackend
- ChromaBackend
- Any backend implementing the `Backend` interface

---

## Service Layer

The `SyntheticDataService` orchestrates the entire generation process:

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# Initialize service
backend = VespaBackend(config)
service = SyntheticDataService(backend=backend)

# Generate modality examples
request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "pdf"]
)
response = await service.generate(request)

print(f"Generated {response.count} examples")
print(f"Success rate: {response.metadata['success_rate']}")
print(f"DSPy modules used: {response.metadata['dspy_modules_used']}")
```

**Service Methods:**

| Method | Description |
|--------|-------------|
| `generate(request)` | Generate examples for single optimizer |
| `batch_generate(requests)` | Generate for multiple optimizers |
| `get_optimizer_config(name)` | Get optimizer configuration |
| `list_optimizers()` | List available optimizers |
| `health_check()` | Check service health |

---

## REST API

The module provides a FastAPI router for HTTP integration:

```python
from fastapi import FastAPI
from cogniverse_synthetic.api import router

app = FastAPI()
app.include_router(router, prefix="/synthetic", tags=["synthetic"])
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/synthetic/generate` | POST | Generate training data |
| `/synthetic/batch/generate` | POST | Batch generation |
| `/synthetic/optimizers` | GET | List available optimizers |
| `/synthetic/optimizers/{name}` | GET | Get optimizer config |
| `/synthetic/health` | GET | Health check |

**Example Requests:**

```bash
# Generate modality examples
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "modality",
    "count": 100,
    "modalities": ["video", "image", "pdf"]
  }'

# Batch generation
curl -X POST http://localhost:8000/synthetic/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"optimizer": "modality", "count": 100},
      {"optimizer": "routing", "count": 50}
    ]
  }'

# List optimizers
curl http://localhost:8000/synthetic/optimizers
```

---

## Usage Examples

### Complete Generation Workflow

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_vespa import VespaBackend
from cogniverse_foundation.config.utils import get_config, create_default_config_manager

# 1. Setup configuration
config_manager = create_default_config_manager()
config = get_config(tenant_id="acme", config_manager=config_manager)

# 2. Initialize backend and service
backend = VespaBackend(config)
service = SyntheticDataService(backend=backend)

# 3. Generate examples for modality optimizer
modality_request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "pdf"],
    profile="video_colpali_mv_frame"
)
modality_response = await service.generate(modality_request)

# 4. Generate examples for routing optimizer
routing_request = SyntheticDataRequest(
    optimizer="routing",
    count=75,
    route_types=["direct", "fallback", "hybrid"]
)
routing_response = await service.generate(routing_request)

# 5. Use examples for training
from cogniverse_agents.routing import RoutingOptimizer

optimizer = RoutingOptimizer()
optimizer.train(
    modality_examples=modality_response.examples,
    routing_examples=routing_response.examples
)
```

### Custom Generator

```python
from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.dspy_signatures import GenerateModalityQuery
from typing import List, Dict, Any
from pydantic import BaseModel
import dspy

class CustomQueryExample(BaseModel):
    query: str
    category: str
    confidence: float

class CustomGenerator(BaseGenerator):
    """Custom generator for specific use case."""

    def __init__(self):
        super().__init__()
        self.query_module = dspy.ChainOfThought(GenerateModalityQuery)

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
    ) -> List[CustomQueryExample]:
        self.validate_inputs(sampled_content, target_count)

        examples = []
        for content in sampled_content[:target_count]:
            # Extract topics from content
            topics = content.get("topics", "general")

            # Generate query using DSPy
            result = self.query_module(
                modality=content.get("modality", "video"),
                topics=topics,
                context=content.get("context", "")
            )

            example = CustomQueryExample(
                query=result.query,
                category=content.get("category", "unknown"),
                confidence=0.85
            )
            examples.append(example)

        return examples
```

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LLM_MODEL="gpt-4"

# Backend Configuration
export VESPA_URL="http://localhost:8080"
export TENANT_ID="acme"
```

### Service Configuration

```python
from cogniverse_foundation.config.unified_config import TelemetryConfigUnified

config = TelemetryConfigUnified(
    tenant_id="acme",
    llm_model="gpt-4",
    llm_api_key="sk-...",
    backend="vespa",
    vespa_url="http://localhost:8080"
)
```

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│          cogniverse-runtime │ cogniverse-dashboard              │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                 Implementation Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │         cogniverse-synthetic ◄─── YOU ARE HERE              ││
│  │  DSPy-driven generation, backend-agnostic sampling          ││
│  └─────────────────────────────────────────────────────────────┘│
│         cogniverse-agents │ cogniverse-vespa                    │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                       Core Layer                                 │
│  cogniverse-core │ cogniverse-evaluation │ cogniverse-telemetry │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation Layer                              │
│           cogniverse-foundation │ cogniverse-sdk                │
└─────────────────────────────────────────────────────────────────┘
```

**Dependencies:**
- `cogniverse-core`: Configuration, utilities
- `cogniverse-foundation`: Base configuration and telemetry
- `dspy-ai`: DSPy framework for LLM programs
- `pydantic`: Data validation

**Dependents:**
- `cogniverse-runtime`: Uses synthetic data via REST API
- `cogniverse-agents`: Uses generated data for training

---

## Testing

```bash
# Run all synthetic tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/routing/unit/synthetic/ -v

# Run specific generator tests
uv run pytest tests/routing/unit/synthetic/test_modality_generator.py -v

# Test with coverage
uv run pytest tests/routing/unit/synthetic/ --cov=cogniverse_synthetic --cov-report=html
```

**Test Categories:**
- `tests/routing/unit/synthetic/` - Unit tests for generators, schemas, service

---

## Related Documentation

- [Core Module](./core.md) - Configuration and utilities
- [Foundation Module](./foundation.md) - Configuration system
- [Agents Module](./agents.md) - Uses synthetic data for training
- [Runtime Module](./runtime.md) - REST API integration
- [DSPy Documentation](https://dspy-docs.vercel.app/) - DSPy framework reference

---

**Summary:** The Synthetic module provides DSPy-driven synthetic data generation for optimizer training. It uses DSPy signatures for LLM-driven query generation, supports multiple optimizer types (Modality, CrossModal, Routing, Workflow), and provides backend-agnostic content sampling. The service layer orchestrates generation with validation and retry logic, and the REST API enables HTTP integration.
