# Cogniverse Vespa Backend

**Package**: `cogniverse-vespa`
**Layer**: Implementation Layer (Yellow/Green)
**Version**: 0.1.0

Multi-tenant Vespa backend implementation providing schema management, document feeding, and search capabilities for the Cogniverse SDK.

---

## Purpose

The `cogniverse-vespa` package provides:
- **VespaBackend**: Primary interface for Vespa operations (feed, search, delete)
- **Schema Management**: Dynamic schema deployment with tenant isolation
- **Multi-Tenant Support**: Automatic tenant suffix appending to schema names
- **JSON Schema Parsing**: Convert JSON schema definitions to PyVespa objects

---

## Architecture

### Position in 10-Package Structure

```
Foundation Layer (Blue)
├── cogniverse-sdk
└── cogniverse-foundation ← cogniverse-vespa depends on this

Core Layer (Pink)
├── cogniverse-core ← cogniverse-vespa depends on this
├── cogniverse-evaluation
└── cogniverse-telemetry-phoenix

Implementation Layer (Yellow/Green)
├── cogniverse-agents
├── cogniverse-vespa ← YOU ARE HERE
└── cogniverse-synthetic

Application Layer (Light Blue/Purple)
├── cogniverse-runtime
└── cogniverse-dashboard
```

### Dependencies

**Workspace Dependencies:**
- `cogniverse-core` (required) - Configuration, agent context, memory
- `cogniverse-foundation` (transitive) - Base configuration and telemetry

**External Dependencies:**
- `pyvespa>=0.44.0` - Official Vespa Python client
- `requests>=2.31.0` - HTTP requests for Vespa API
- `httpx>=0.25.0` - Async HTTP client

---

## Key Features

### 1. Multi-Tenant Schema Management

Automatically appends tenant suffixes to schema names via `VespaSchemaManager`:

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

# Initialize schema manager (backend_endpoint and backend_port are REQUIRED)
schema_manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
)

# Get tenant-specific schema name (colon → underscore)
schema_name = schema_manager.get_tenant_schema_name(
    tenant_id="acme_corp",
    base_schema_name="video_colpali_smol500_mv_frame"
)
# Returns: "video_colpali_smol500_mv_frame_acme_corp"
# For "acme:production": "video_colpali_smol500_mv_frame_acme_production"
```

### 2. Dynamic Schema Deployment

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from cogniverse_vespa.json_schema_parser import JsonSchemaParser

# Parse JSON schema definition
parser = JsonSchemaParser()
schema = parser.load_schema_from_json_file("configs/schemas/video_colpali_smol500_mv_frame_schema.json")

# Deploy schema — schema_registry required for tenant operations
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))
schema_registry = SchemaRegistry(config_manager=config_manager, backend=None, schema_loader=schema_loader)

manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
    schema_loader=schema_loader,
    schema_registry=schema_registry,
)

# Delete tenant schemas (unregisters from registry, then immediately redeploys to Vespa
# with allow_schema_removal=True so Vespa accepts the content type removal)
deleted = manager.delete_tenant_schemas(tenant_id="acme_corp")
# Returns: ["video_colpali_smol500_mv_frame_acme_corp", ...]
```

### 3. Document Operations

```python
from cogniverse_vespa.backend import VespaBackend
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))

backend = VespaBackend(
    backend_config={"url": "http://localhost", "port": 8080},
    schema_loader=schema_loader,
    config_manager=config_manager,
)

# Feed documents (synchronous)
backend.feed_datapoint(
    doc_id="video1",
    data={"title": "ML Tutorial", "embedding": [0.1, 0.2]},
    schema_name="video_colpali_smol500_mv_frame_acme_corp",
)

# Search documents (synchronous)
results = backend.search(
    query_dict={"query": "machine learning", "schema": "video_colpali_smol500_mv_frame_acme_corp"},
    tenant_id="acme_corp",
)
```

### 4. Tenant Schema Operations

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

# Full initialization with schema_registry for tenant operations
manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
    schema_loader=schema_loader,
    schema_registry=schema_registry,
)

# Check if tenant schema exists
exists = manager.tenant_schema_exists(
    tenant_id="acme_corp",
    base_schema_name="video_colpali_smol500_mv_frame"
)

# Delete tenant schemas (immediately redeploys to Vespa without deleted schemas)
deleted = manager.delete_tenant_schemas(tenant_id="acme_corp")
# Returns: List of deleted schema names
```

---

## Installation

### Development (Editable Mode)

```bash
# From workspace root
uv sync

# Or install individually
uv pip install -e libs/vespa
```

### Production

```bash
pip install cogniverse-vespa

# Automatically installs:
# - cogniverse-core
# - cogniverse-foundation
# - pyvespa
# - requests
# - httpx
```

---

## Usage

### Basic Setup

```python
from cogniverse_vespa.backend import VespaBackend
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize dependencies
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))

# Initialize backend
backend = VespaBackend(
    backend_config={"url": "http://localhost", "port": 8080},
    schema_loader=schema_loader,
    config_manager=config_manager,
)
```

### Feed Documents

```python
backend.feed_datapoint(
    doc_id="video_001",
    data={
        "title": "Machine Learning Basics",
        "embedding": [0.1, 0.2, ...],  # vector matching schema dimension
    },
    schema_name="video_colpali_smol500_mv_frame_acme_corp",
)
```

### Search Documents

```python
# Construct query dict for Vespa YQL
results = backend.search(
    query_dict={
        "query": "machine learning tutorial",
        "schema": "video_colpali_smol500_mv_frame_acme_corp",
        "hits": 10,
    },
    tenant_id="acme_corp",
)

for result in results:
    print(f"Document: {result.get('id')}, Score: {result.get('relevance')}")
```

### Deploy Tenant Schemas

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_core.registries.schema_registry import SchemaRegistry
from pathlib import Path

schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))
schema_registry = SchemaRegistry(
    config_manager=config_manager, backend=None, schema_loader=schema_loader
)

manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
    schema_loader=schema_loader,
    schema_registry=schema_registry,
)

# Delete tenant schemas — immediately redeploys Vespa without the removed schemas
deleted = manager.delete_tenant_schemas(tenant_id="acme_corp")
print(f"Deleted schemas: {deleted}")
```

---

## Development

### Running Tests

```bash
# Run Vespa-specific tests
uv run pytest tests/backends/unit/ -v -k vespa

# Run integration tests (requires Vespa running)
uv run pytest tests/backends/integration/ -v -k vespa
```

### Local Vespa Instance

```bash
# Start Vespa using Docker
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# Wait for Vespa to be ready
curl -s --head http://localhost:19071/ApplicationStatus

# Deploy test schema
uv run python scripts/deploy_test_schema.py
```

### Code Style

```bash
# Format code
uv run ruff format libs/vespa

# Lint code
uv run ruff check libs/vespa

# Type check
uv run mypy libs/vespa
```

---

## Configuration

Configuration is provided via `SystemConfig` from `cogniverse-foundation`:

```python
from cogniverse_foundation.config.unified_config import SystemConfig

config = SystemConfig(
    tenant_id="acme_corp",
    backend_url="http://localhost",
    backend_port=8080,
)
```

### Environment Variables

```bash
export VESPA_URL="http://localhost:8080"
export VESPA_CONFIG_URL="http://localhost:19071"
export TENANT_ID="acme_corp"
```

---

## Schema Format

JSON schema files follow this structure:

```json
{
  "name": "video_colpali_mv_frame",
  "document": {
    "fields": [
      {
        "name": "id",
        "type": "string",
        "indexing": "summary | attribute"
      },
      {
        "name": "embedding",
        "type": "tensor<float>(x[1024])",
        "indexing": "summary | attribute | index"
      },
      {
        "name": "title",
        "type": "string",
        "indexing": "summary | index"
      }
    ]
  },
  "rank-profiles": [
    {
      "name": "vector_similarity",
      "first-phase": "closeness(embedding)"
    }
  ]
}
```

---

## Multi-Tenant Naming Convention

The package automatically handles tenant isolation:

| Base Schema | Tenant ID | Actual Schema Name |
|-------------|-----------|-------------------|
| `video_colpali_mv_frame` | `acme_corp` | `video_colpali_mv_frame_acme_corp` |
| `video_videoprism_base` | `globex_inc` | `video_videoprism_base_globex_inc` |
| `agent_memories` | `default` | `agent_memories_default` |

**Key Points:**
- Schema names ALWAYS include tenant suffix
- No cross-tenant data access possible
- Each tenant has isolated document space
- Same base schema can be deployed for multiple tenants

---

## Documentation

- **Architecture**: [10-Package Architecture](../../docs/architecture/10-package-architecture.md)
- **Multi-Tenant**: [Multi-Tenant Architecture](../../docs/architecture/multi-tenant.md)
- **Diagrams**: [SDK Architecture Diagrams](../../docs/diagrams/sdk-architecture-diagrams.md)
- **Vespa Docs**: [PyVespa Documentation](https://pyvespa.readthedocs.io/)

---

## Troubleshooting

### Common Issues

**1. Connection Refused**
```bash
# Ensure Vespa is running
curl http://localhost:8080/
```

**2. Schema Deployment Fails**
```bash
# Check config server
curl http://localhost:19071/ApplicationStatus

# Verify schema JSON is valid
uv run python -c "import json; json.load(open('schema.json'))"
```

**3. Tenant Schema Not Found**
```python
# Check if tenant schema exists via schema_manager
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
    schema_registry=schema_registry,
)
exists = manager.tenant_schema_exists(
    tenant_id="acme_corp",
    base_schema_name="video_colpali_smol500_mv_frame"
)
print(f"Schema exists: {exists}")
```

**4. Document Feed Fails**
- Check document format matches schema fields
- Ensure tensor dimensions match schema definition (128 for ColPali patch, 768 for base, 1024 for large)

---

## Contributing

```bash
# Create feature branch
git checkout -b feature/vespa-improvement

# Make changes
# ...

# Run tests
uv run pytest tests/backends/unit/ -v -k vespa

# Submit PR
```

---

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Related Packages

- **cogniverse-core**: Multi-agent orchestration (depends on this)
- **cogniverse-agents**: Agent implementations (depends on this)
- **cogniverse-synthetic**: Synthetic data generation (depends on this)
- **cogniverse-runtime**: FastAPI server (depends on this)
