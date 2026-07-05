# Cogniverse Synthetic Data Generation

**Package**: `cogniverse-synthetic`
**Layer**: Implementation Layer (Yellow/Green)
**Version**: 0.1.0

Generates high-quality training data for DSPy optimizers by sampling real content from backend storage and generating realistic queries using DSPy-driven LLM modules with validation.

---

## Purpose

The `cogniverse-synthetic` package provides:
- **DSPy-Driven Generation**: Uses DSPy signatures and modules for LLM-driven query generation
- **Backend-Agnostic Sampling**: Works with any backend implementing the `Backend` interface (VespaBackend ships today)
- **Optimizer Support**: Generates data for the `profile`, `routing`, `workflow`, `unified`, and `cross_modal` optimizers
- **Validated Output**: Built-in validation with retry logic to ensure quality
- **REST API**: FastAPI router for HTTP endpoints

---

## Architecture

### Position in the 13-Package Workspace

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
├── cogniverse-dashboard
├── cogniverse-cli
├── cogniverse-finetuning ← Reuses cogniverse-synthetic
└── cogniverse-messaging
```

### Dependencies

**Workspace Dependencies:**
- `cogniverse-sdk` (required) - Backend interface
- `cogniverse-foundation` (required) - Configuration classes (`BackendConfig`, `SyntheticGeneratorConfig`, etc.)
- `cogniverse-core` (imported at runtime for `SYSTEM_TENANT_ID`, though not declared in `pyproject.toml`)

**External Dependencies:**
- `dspy-ai==3.1.3` - DSPy framework for LLM programs
- `pydantic==2.12.5` - Data validation and schemas
- `fastapi==0.135.3` - REST API framework
- `httpx==0.28.1` - HTTP client

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
    optimizer="profile",
    count=100,
    tenant_id="acme:production",
)

response = await service.generate(request)
print(f"Generated {response.count} examples")
print(f"Backend query strategy: {response.metadata['backend_query_strategy']}")
```

### 2. Backend-Agnostic Sampling

Works with any vector database through Backend interface:

```python
from cogniverse_vespa import VespaBackend
from cogniverse_synthetic import SyntheticDataService

# Use Vespa backend (backend_config, schema_loader, config_manager are all required)
vespa_backend = VespaBackend(
    backend_config=backend_config,
    schema_loader=schema_loader,
    config_manager=config_manager,
)
service = SyntheticDataService(backend=vespa_backend)

# Any other backend implementing cogniverse_sdk.interfaces.backend.Backend
# can be substituted the same way — VespaBackend is the implementation
# that ships today.
```

### 3. Optimizer Support

Supports the optimizers registered in `OPTIMIZER_REGISTRY` (`profile`,
`routing`, `workflow`, `unified`, `cross_modal`):

```python
# Profile Optimizer
request = SyntheticDataRequest(
    optimizer="profile",
    count=100,
    tenant_id="acme:production",
)

# Routing Optimizer
request = SyntheticDataRequest(
    optimizer="routing",
    count=75,
    tenant_id="acme:production",
)

# Workflow Optimizer
request = SyntheticDataRequest(
    optimizer="workflow",
    count=100,
    tenant_id="acme:production",
)

# Cross-Modal Optimizer
request = SyntheticDataRequest(
    optimizer="cross_modal",
    count=50,
    tenant_id="acme:production",
)
```

### 4. Validation and Retry Logic

Built-in validation ensures quality:

```python
# Automatic validation with 3 retries
response = await service.generate(request)

# response.data is a list of dicts conforming to the optimizer's schema
# (ProfileSelectionExampleSchema for "profile", RoutingExperienceSchema for
# "routing", WorkflowExecutionSchema for "workflow"/"unified"/"cross_modal")
for example in response.data:
    assert example["query"]              # Non-empty query
    assert example["modality"]           # Valid modality
    assert example["selected_profile"]   # Chosen backend profile
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
# - cogniverse-sdk
# - cogniverse-foundation
# - dspy-ai
# - pydantic
# - fastapi
# - httpx
```

---

## Usage

### Basic Setup

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# Initialize backend (backend_config, schema_loader, config_manager are all required)
backend = VespaBackend(
    backend_config=backend_config,
    schema_loader=schema_loader,
    config_manager=config_manager,
)

# Initialize service
service = SyntheticDataService(backend=backend)
```

### Generate Training Data

```python
# Generate profile-selection training examples
request = SyntheticDataRequest(
    optimizer="profile",
    count=100,
    vespa_sample_size=200,
    max_profiles=3,
    tenant_id="acme:production",
)

response = await service.generate(request)

print(f"Generated {response.count} examples")
print(f"Sampled content count: {response.metadata['sampled_content_count']}")

# Use examples for training (response.data is a list of dicts)
for example in response.data:
    print(f"Query: {example['query']}")
    print(f"Modality: {example['modality']}")
    print(f"Selected profile: {example['selected_profile']}")
```

### REST API Integration

```python
from fastapi import FastAPI
from cogniverse_synthetic import router

app = FastAPI()
app.include_router(router, tags=["synthetic-data"])  # router already carries prefix="/synthetic"

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
    "optimizer": "profile",
    "count": 100,
    "tenant_id": "acme:production"
  }'

# List optimizers
curl http://localhost:8000/synthetic/optimizers

# Get optimizer config
curl http://localhost:8000/synthetic/optimizers/profile

# Health check
curl http://localhost:8000/synthetic/health

# Batch generation - optimizer/count_per_batch/num_batches/tenant_id are
# query parameters, not a JSON body
curl -X POST "http://localhost:8000/synthetic/batch/generate?optimizer=profile&count_per_batch=100&num_batches=5&tenant_id=acme:production"
```

---

## Package Structure

```
libs/synthetic/cogniverse_synthetic/
├── __init__.py              # Package exports
├── schemas.py               # Pydantic schemas for all optimizer types
├── registry.py              # Optimizer configuration registry (OPTIMIZER_REGISTRY dict)
├── service.py               # Main orchestrator service
├── api.py                   # FastAPI router
├── profile_selector.py      # LLM/rule-based profile selection
├── backend_querier.py       # Backend-agnostic content sampling
├── dspy_signatures.py       # DSPy signatures (GenerateModalityQuery, GenerateEntityQuery, InferAgentFromModality)
├── dspy_modules.py          # Validated DSPy module (ValidatedEntityQueryGenerator)
├── generators/              # Concrete generator implementations
│   ├── __init__.py
│   ├── base.py              # Base generator interface
│   ├── profile.py           # Profile selection generator
│   ├── routing.py           # Routing strategy generator
│   └── workflow.py          # Workflow generator
├── approval/                # Human-in-the-loop approval workflow
│   ├── __init__.py
│   ├── confidence_extractor.py
│   └── feedback_handler.py
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

# 101 tests covering:
# - Schemas and validation
# - Generators (profile, routing, workflow, cross_modal)
# - Backend querying and optimizer registry lookups
# - Service orchestration
# - Approval workflow
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

Configuration is provided via `BackendConfig` and `SyntheticGeneratorConfig`
from `cogniverse-foundation` (both require `tenant_id`):

```python
from cogniverse_foundation.config.unified_config import BackendConfig, SyntheticGeneratorConfig

backend_config = BackendConfig(
    tenant_id="acme_corp",
    backend_type="vespa",
    url="http://localhost",
    port=8080,
)
generator_config = SyntheticGeneratorConfig(tenant_id="acme_corp")

service = SyntheticDataService(
    backend=backend,
    backend_config=backend_config,
    generator_config=generator_config,
)
```

### Environment Variables

```bash
export ROUTER_OPTIMIZER_TEACHER_KEY="your-api-key"  # Works with any LiteLLM-supported provider
export LLM_MODEL="claude-3-5-sonnet-20241022"
```

---

## DSPy Signatures and Modules

### Query Generation Signatures

Three signatures are defined in `dspy_signatures.py`:

- `GenerateModalityQuery` — generates a natural search query for a given content modality
- `GenerateEntityQuery` — generates a query that must mention at least one provided entity
- `InferAgentFromModality` — infers the correct agent for a modality/query pair

```python
from cogniverse_synthetic.dspy_signatures import GenerateModalityQuery, GenerateEntityQuery
import dspy

class ModalityQueryGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateModalityQuery)

    def forward(self, modality, topics, context):
        result = self.generate(
            modality=modality,
            topics=topics,
            context=context
        )
        return result.query
```

### Validation and Retry

```python
from cogniverse_synthetic.dspy_modules import ValidatedEntityQueryGenerator

generator = ValidatedEntityQueryGenerator(max_retries=3)

# Automatically retries up to 3 times if no entity appears in the query
result = generator.forward(
    topics="machine learning, neural networks",
    entities="PyTorch, TensorFlow",
    entity_types="TECHNOLOGY, TECHNOLOGY"
)
print(result.query)
```

---

## Optimizer Registry

The package includes a registry of optimizer configurations. There is no
`OptimizerRegistry` class — the registry is a module-level dict
`OPTIMIZER_REGISTRY` mapping optimizer names to `OptimizerConfig` objects,
with helper functions for lookup and listing.

```python
from cogniverse_synthetic.registry import (
    OPTIMIZER_REGISTRY,
    OptimizerConfig,
    get_optimizer_config,
    list_optimizers,
)

# Get optimizer config by name
config = get_optimizer_config("routing")
print(config.name)         # "routing"
print(config.description)  # "Advanced routing with entity extraction..."

# List all registered optimizers
for name, description in list_optimizers().items():
    print(f"{name}: {description}")

# Direct registry access
print(list(OPTIMIZER_REGISTRY.keys()))
# ['routing', 'workflow', 'profile', 'unified', 'cross_modal']
```

---

## Backend Abstraction

The package works with any backend that implements the Backend interface:

```python
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_foundation.config.unified_config import BackendConfig, FieldMappingConfig

querier = BackendQuerier(
    backend=your_backend,
    backend_config=BackendConfig(tenant_id="acme_corp"),
    field_mappings=FieldMappingConfig(),
)

# Sample documents from any backend
documents = await querier.query_profiles(
    profile_configs=[{"schema_name": "video_colpali_mv_frame"}],
    sample_size=10,
    strategy="diverse",
)

# Works with any backend implementing cogniverse_sdk.interfaces.backend.Backend.
# VespaBackend is the implementation that ships today; other backends can be
# added the same way.
```

---

## Documentation

- **Full Docs**: [Synthetic Data Generation](../../docs/synthetic-data-generation.md)
- **Architecture**: [SDK Architecture](../../docs/architecture/sdk-architecture.md)
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
backend = VespaBackend(
    backend_config=backend_config,
    schema_loader=schema_loader,
    config_manager=config_manager,
)
backend.health_check()
```

**4. No Documents Sampled**
- Verify schema exists: `backend.schema_exists(schema_name, tenant_id=...)`
- Check tenant isolation: schema should include tenant suffix
- Ensure documents exist in schema

---

## Performance

**Generation Speed:**
- Profile: ~10-15 examples/minute (with GPT-4)
- Cross-Modal: ~8-12 examples/minute
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

- **cogniverse-core**: Provides tenant utilities (`SYSTEM_TENANT_ID`) used by this package
- **cogniverse-agents**: Depends on this package to generate training data
- **cogniverse-vespa**: Backend implementation used for sampling documents
- **cogniverse-runtime**: Mounts the synthetic API router
