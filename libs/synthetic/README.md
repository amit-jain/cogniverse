# Cogniverse Synthetic Data Generation

**Package**: `cogniverse-synthetic`

Generates high-quality training data for all Cogniverse optimizers by sampling real content from backend storage and generating realistic queries using DSPy-driven LLM modules with validation.

## Quick Start

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# Initialize backend (Vespa, Pinecone, etc.)
backend = VespaBackend(config=backend_config)

# Initialize service
service = SyntheticDataService(backend=backend)

# Generate training data with DSPy-driven query generation
request = SyntheticDataRequest(
    optimizer="modality",
    count=100
)

response = await service.generate(request)
print(f"Generated {response.count} examples")
print(f"Used DSPy modules: {response.metadata.get('dspy_modules_used')}")
```

## REST API

```python
from fastapi import FastAPI
from cogniverse_synthetic import router

app = FastAPI()
app.include_router(router)

# Endpoints:
# POST /synthetic/generate
# GET /synthetic/optimizers
# GET /synthetic/optimizers/{name}
# GET /synthetic/health
# POST /synthetic/batch/generate
```

## Package Structure

- `schemas.py` - Pydantic schemas for all optimizer types
- `registry.py` - Optimizer configuration registry
- `service.py` - Main orchestrator service
- `api.py` - FastAPI router
- `profile_selector.py` - LLM/rule-based profile selection
- `backend_querier.py` - Backend-agnostic content sampling (Vespa, Pinecone, etc.)
- `dspy_signatures.py` - **DSPy signatures for LLM-driven query generation**
- `dspy_modules.py` - **Validated DSPy modules with retry logic**
- `generators/` - Concrete generator implementations (Modality, CrossModal, Routing, Workflow)
- `utils/` - Pattern extraction and agent inference utilities

## Key Features

- **DSPy-Driven Generation**: Uses DSPy signatures and modules for LLM-driven query generation instead of static templates
- **Validated Output**: Built-in validation with retry logic (3 attempts) to ensure quality
- **ChainOfThought Reasoning**: LLM reasons about query generation for better quality
- **Backend Agnostic**: Works with any vector database through Backend interface (Vespa, Pinecone, Weaviate, etc.)
- **No Arbitrary Fallbacks**: Raises exceptions when validation fails - no arbitrary defaults
- **Configuration-Driven**: All behavior externalized to Pydantic configuration classes

## Documentation

📖 **Full documentation**: [docs/synthetic-data-generation.md](../../docs/synthetic-data-generation.md)

Includes:
- Architecture diagrams
- DSPy signatures and modules guide
- Backend abstraction details
- API reference
- Integration guides
- Development guide
- Troubleshooting

## Testing

```bash
uv run pytest tests/routing/unit/synthetic/ -v
# 82 tests covering schemas, generators, service, and integration
```

## Installation

```bash
# For development (editable mode)
uv pip install -e libs/synthetic

# Dependencies automatically managed by uv workspace
```
