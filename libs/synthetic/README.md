# Cogniverse Synthetic Data Generation

**Package**: `cogniverse-synthetic`

Generates high-quality training data for all Cogniverse optimizers by sampling real content from Vespa and generating realistic queries and metadata.

## Quick Start

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Initialize service
service = SyntheticDataService()

# Generate training data
request = SyntheticDataRequest(
    optimizer="modality",
    count=100
)

response = await service.generate(request)
print(f"Generated {response.count} examples")
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
- `backend_querier.py` - Vespa content sampling
- `generators/` - Concrete generator implementations (Modality, CrossModal, Routing, Workflow)
- `utils/` - Pattern extraction and agent inference utilities

## Documentation

📖 **Full documentation**: [docs/synthetic-data-generation.md](../../docs/synthetic-data-generation.md)

Includes:
- Architecture diagrams
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
