# Cogniverse Synthetic Data Generation

**Package**: `cogniverse-synthetic`
**Layer**: Implementation Layer (Yellow/Green)
**Version**: 0.1.0

Generates high-quality training data for DSPy optimizers by sampling real content from backend storage and generating realistic queries using DSPy-driven LLM modules with validation.

---

## Purpose

The `cogniverse-synthetic` package provides:
- **DSPy-Driven Generation**: Uses DSPy signatures and modules for LLM-driven query generation
- **Backend-Agnostic Sampling**: Works with any vector database (Vespa, Pinecone, Weaviate)
- **Optimizer Support**: Generates data for Modality, CrossModal, Routing, and Workflow optimizers
- **Validated Output**: Built-in validation with retry logic to ensure quality
- **REST API**: FastAPI router for HTTP endpoints

---

## Architecture

### Position in 10-Package Structure

```
Foundation Layer (Blue)
├── cogniverse-sdk
└── cogniverse-foundation ← cogniverse-synthetic depends on this

Core Layer (Pink)
├── cogniverse-core ← cogniverse-synthetic depends on this
├── cogniverse-evaluation
└── cogniverse-telemetry-phoenix

Implementation Layer (Yellow/Green)
├── cogniverse-agents
├── cogniverse-vespa
└── cogniverse-synthetic ← YOU ARE HERE

Application Layer (Light Blue/Purple)
├── cogniverse-runtime ← Uses synthetic data
└── cogniverse-dashboard
```

### Dependencies

**Workspace Dependencies:**
- `cogniverse-core` (required) - Configuration, agent context, utilities
- `cogniverse-foundation` (transitive) - Base configuration and telemetry

**External Dependencies:**
- `dspy-ai>=2.5.0` - DSPy framework for LLM programs
- `pydantic>=2.0.0` - Data validation and schemas
- `fastapi>=0.104.0` - REST API framework
- `litellm>=1.0.0` - Unified LLM interface

---

## Key Features

### 1. DSPy-Driven Query Generation

Uses DSPy signatures and modules instead of static templates:

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

service = SyntheticDataService(backend=vespa_backend)

# Generate data with DSPy-driven query generation
request = SyntheticDataRequest(
    optimizer="modality",
    count=100
)

response = await service.generate(request)
print(f"Generated {response.count} examples")
print(f"DSPy modules used: {response.metadata.get('dspy_modules_used')}")
```

### 2. Backend-Agnostic Sampling

Works with any vector database through Backend interface:

```python
from cogniverse_vespa import VespaBackend
from cogniverse_synthetic import SyntheticDataService

# Use Vespa backend
vespa_backend = VespaBackend(config)
service = SyntheticDataService(backend=vespa_backend)

# Or use Pinecone, Weaviate, etc.
# pinecone_backend = PineconeBackend(config)
# service = SyntheticDataService(backend=pinecone_backend)
```

### 3. Optimizer Support

Supports multiple DSPy optimizers:

```python
# Modality Optimizer
request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "pdf"]
)

# CrossModal Optimizer
request = SyntheticDataRequest(
    optimizer="crossmodal",
    count=50,
    source_modality="video",
    target_modality="image"
)

# Routing Optimizer
request = SyntheticDataRequest(
    optimizer="routing",
    count=75,
    route_types=["direct", "fallback", "hybrid"]
)

# Workflow Optimizer
request = SyntheticDataRequest(
    optimizer="workflow",
    count=100,
    workflow_types=["ingestion", "search", "analysis"]
)
```

### 4. Validation and Retry Logic

Built-in validation ensures quality:

```python
# Automatic validation with 3 retries
response = await service.generate(request)

# All examples are validated
for example in response.examples:
    assert example.query  # Non-empty query
    assert example.modality  # Valid modality
    assert example.ground_truth  # Expected answer
```

---

## Installation

### Development (Editable Mode)

```bash
# From workspace root
uv sync

# Or install individually
uv pip install -e libs/synthetic
```

### Production

```bash
pip install cogniverse-synthetic

# Automatically installs:
# - cogniverse-core
# - cogniverse-foundation
# - dspy-ai
# - pydantic
# - fastapi
# - litellm
```

---

## Usage

### Basic Setup

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# Initialize backend
backend = VespaBackend(config)

# Initialize service
service = SyntheticDataService(backend=backend)
```

### Generate Training Data

```python
# Generate modality routing examples
request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "pdf"],
    profile="frame_based"  # Optional: specify profile
)

response = await service.generate(request)

print(f"Generated {response.count} examples")
print(f"Success rate: {response.metadata['success_rate']}")

# Use examples for training
for example in response.examples:
    print(f"Query: {example.query}")
    print(f"Modality: {example.modality}")
    print(f"Ground Truth: {example.ground_truth}")
```

### REST API Integration

```python
from fastapi import FastAPI
from cogniverse_synthetic import router

app = FastAPI()
app.include_router(router, prefix="/synthetic", tags=["synthetic"])

# Endpoints available:
# POST /synthetic/generate - Generate training data
# GET /synthetic/optimizers - List available optimizers
# GET /synthetic/optimizers/{name} - Get optimizer config
# GET /synthetic/health - Health check
# POST /synthetic/batch/generate - Batch generation
```

### Using the REST API

```bash
# Generate data
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "modality",
    "count": 100,
    "modalities": ["video", "image", "pdf"]
  }'

# List optimizers
curl http://localhost:8000/synthetic/optimizers

# Get optimizer config
curl http://localhost:8000/synthetic/optimizers/modality

# Health check
curl http://localhost:8000/synthetic/health

# Batch generation
curl -X POST http://localhost:8000/synthetic/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"optimizer": "modality", "count": 100},
      {"optimizer": "routing", "count": 50}
    ]
  }'
```

---

## Package Structure

```
libs/synthetic/cogniverse_synthetic/
├── __init__.py              # Package exports
├── schemas.py               # Pydantic schemas for all optimizer types
├── registry.py              # Optimizer configuration registry
├── service.py               # Main orchestrator service
├── api.py                   # FastAPI router
├── profile_selector.py      # LLM/rule-based profile selection
├── backend_querier.py       # Backend-agnostic content sampling
├── dspy_signatures.py       # DSPy signatures for LLM-driven generation
├── dspy_modules.py          # Validated DSPy modules with retry logic
├── generators/              # Concrete generator implementations
│   ├── __init__.py
│   ├── base.py              # Base generator interface
│   ├── modality.py          # Modality routing generator
│   ├── crossmodal.py        # Cross-modal generator
│   ├── routing.py           # Routing strategy generator
│   └── workflow.py          # Workflow generator
└── utils/                   # Utilities
    ├── __init__.py
    ├── pattern_extraction.py
    └── agent_inference.py
```

---

## Development

### Running Tests

```bash
# Run all synthetic tests
uv run pytest tests/routing/unit/synthetic/ -v

# 82 tests covering:
# - Schemas and validation
# - Generators (modality, crossmodal, routing, workflow)
# - Service orchestration
# - API endpoints
# - Integration tests
```

### Code Style

```bash
# Format code
uv run ruff format libs/synthetic

# Lint code
uv run ruff check libs/synthetic

# Type check
uv run mypy libs/synthetic
```

---

## Configuration

Configuration is provided via `SystemConfig` from `cogniverse-foundation`:

```python
from cogniverse_foundation.config.unified_config import SystemConfig

config = SystemConfig(
    tenant_id="acme_corp",
    llm_model="gpt-4",
    llm_api_key="sk-...",
    search_backend="vespa",
    backend_url="http://localhost",
    backend_port=8080,
)
```

### Environment Variables

```bash
export ROUTER_OPTIMIZER_TEACHER_KEY="your-api-key"  # Works with any LiteLLM-supported provider
export LLM_MODEL="claude-3-5-sonnet-20241022"
export TENANT_ID="acme_corp"
```

---

## DSPy Signatures and Modules

### Query Generation Signature

```python
from cogniverse_synthetic.dspy_signatures import QueryGenerationSignature
import dspy

class QueryGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(self, document, modality):
        result = self.generate(
            document=document,
            modality=modality
        )
        return result.query
```

### Validation and Retry

```python
from cogniverse_synthetic.dspy_modules import ValidatedQueryGenerator

generator = ValidatedQueryGenerator()

# Automatically retries up to 3 times if validation fails
query = generator.generate(
    document="Machine learning video tutorial",
    modality="video"
)
```

---

## Optimizer Registry

The package includes a registry of optimizer configurations:

```python
from cogniverse_synthetic.registry import OptimizerRegistry

registry = OptimizerRegistry()

# Get optimizer config
config = registry.get("modality")
print(config.name)  # "modality"
print(config.description)  # "Modality routing optimizer"
print(config.supported_modalities)  # ["video", "image", "pdf"]

# List all optimizers
optimizers = registry.list_all()
for optimizer in optimizers:
    print(f"{optimizer.name}: {optimizer.description}")
```

---

## Backend Abstraction

The package works with any backend that implements the Backend interface:

```python
from cogniverse_synthetic.backend_querier import BackendQuerier

querier = BackendQuerier(backend=your_backend)

# Sample documents from any backend
documents = await querier.sample_documents(
    schema_name="video_colpali_mv_frame",
    count=10,
    modality="video"
)

# Works with:
# - VespaBackend
# - PineconeBackend
# - WeaviateBackend
# - ChromaBackend
# - Custom backends implementing Backend interface
```

---

## Documentation

- **Full Docs**: [Synthetic Data Generation](../../docs/synthetic-data-generation.md)
- **Architecture**: [10-Package Architecture](../../docs/architecture/10-package-architecture.md)
- **Diagrams**: [SDK Architecture Diagrams](../../docs/diagrams/sdk-architecture-diagrams.md)
- **DSPy Docs**: [DSPy Documentation](https://dspy-docs.vercel.app/)

---

## Troubleshooting

### Common Issues

**1. LLM API Key Not Found**
```bash
export ROUTER_OPTIMIZER_TEACHER_KEY="your-api-key"
```

**2. Validation Fails After 3 Retries**
- Check LLM model quality (try GPT-4 instead of GPT-3.5)
- Verify document quality in backend
- Review DSPy signatures for clarity

**3. Backend Connection Issues**
```python
# Test backend connection
backend = VespaBackend(config)
await backend.ping()
```

**4. No Documents Sampled**
- Verify schema exists: `await backend.list_schemas()`
- Check tenant isolation: schema should include tenant suffix
- Ensure documents exist in schema

---

## Performance

**Generation Speed:**
- Modality: ~10-15 examples/minute (with GPT-4)
- CrossModal: ~8-12 examples/minute
- Routing: ~12-18 examples/minute
- Workflow: ~10-14 examples/minute

**Optimization Tips:**
- Use batch generation for large datasets
- Cache LLM responses when possible
- Use faster models (GPT-3.5) for prototyping
- Parallel generation across multiple workers

---

## Contributing

```bash
# Create feature branch
git checkout -b feature/synthetic-improvement

# Make changes
# ...

# Run tests
uv run pytest tests/routing/unit/synthetic/ -v

# Submit PR
```

---

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Related Packages

- **cogniverse-core**: Multi-agent orchestration (depends on this)
- **cogniverse-agents**: Uses synthetic data for training
- **cogniverse-vespa**: Backend for sampling documents
- **cogniverse-runtime**: FastAPI server integration
