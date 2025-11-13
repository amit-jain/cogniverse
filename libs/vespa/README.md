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

Automatically appends tenant suffixes to schema names:

```python
from cogniverse_vespa.backends import VespaBackend
from cogniverse_core.config import UnifiedConfig

config = UnifiedConfig(tenant_id="acme_corp")
backend = VespaBackend(config)

# Automatically uses: video_colpali_mv_frame_acme_corp
await backend.feed_documents(
    docs,
    schema_name="video_colpali_mv_frame"
)
```

### 2. Dynamic Schema Deployment

```python
from cogniverse_vespa.backends import VespaSchemaManager
from cogniverse_vespa.json_parser import JsonSchemaParser

# Parse JSON schema definition
parser = JsonSchemaParser()
schema = parser.parse("schemas/video_colpali.json")

# Deploy with tenant isolation
manager = VespaSchemaManager(config)
await manager.deploy_schema(schema, tenant_id="acme_corp")
# Creates: video_colpali_acme_corp
```

### 3. Document Operations

```python
# Feed documents
response = await backend.feed_documents(
    documents=[
        {"id": "video1", "embedding": [...], "title": "ML Tutorial"},
        {"id": "video2", "embedding": [...], "title": "AI Basics"}
    ],
    schema_name="video_colpali_mv_frame"
)

# Search documents
results = await backend.search(
    query="machine learning",
    schema_name="video_colpali_mv_frame",
    limit=10
)

# Delete documents
await backend.delete_document(
    doc_id="video1",
    schema_name="video_colpali_mv_frame"
)
```

### 4. Tenant Schema Manager

```python
from cogniverse_vespa.backends import TenantSchemaManager

tenant_mgr = TenantSchemaManager(config)

# Deploy all schemas for a tenant
await tenant_mgr.deploy_tenant_schemas(
    tenant_id="acme_corp",
    schema_files=["video_colpali.json", "video_videoprism.json"]
)

# List tenant schemas
schemas = await tenant_mgr.list_tenant_schemas("acme_corp")
# Returns: ["video_colpali_acme_corp", "video_videoprism_acme_corp"]

# Delete tenant schemas
await tenant_mgr.delete_tenant_schemas("acme_corp")
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
from cogniverse_vespa.backends import VespaBackend
from cogniverse_core.config import UnifiedConfig

# Initialize config with tenant
config = UnifiedConfig(
    tenant_id="acme_corp",
    vespa_url="http://localhost:8080",
    vespa_config_url="http://localhost:19071"
)

# Initialize backend
backend = VespaBackend(config)
```

### Feed Documents

```python
documents = [
    {
        "id": "video_001",
        "title": "Machine Learning Basics",
        "embedding": [0.1, 0.2, ...],  # 1024-dim vector
        "timestamp": "2025-11-13T10:00:00Z"
    }
]

response = await backend.feed_documents(
    documents=documents,
    schema_name="video_colpali_mv_frame"
)

print(f"Fed {response.count} documents")
```

### Search Documents

```python
# Vector search
results = await backend.search(
    query="machine learning tutorial",
    schema_name="video_colpali_mv_frame",
    limit=10,
    rank_profile="vector_similarity"
)

for result in results:
    print(f"Document: {result['id']}, Score: {result['relevance']}")
```

### Deploy Custom Schema

```python
from cogniverse_vespa.backends import VespaSchemaManager
from cogniverse_vespa.json_parser import JsonSchemaParser

# Parse JSON schema
parser = JsonSchemaParser()
schema = parser.parse("custom_schema.json")

# Deploy
manager = VespaSchemaManager(config)
await manager.deploy_schema(
    schema=schema,
    tenant_id="acme_corp"
)
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

Configuration is provided via `UnifiedConfig` from `cogniverse-foundation`:

```python
config = UnifiedConfig(
    tenant_id="acme_corp",           # Required: Tenant identifier
    vespa_url="http://localhost:8080",  # Vespa query/feed endpoint
    vespa_config_url="http://localhost:19071",  # Vespa config server
    vespa_timeout=30,                # Request timeout (seconds)
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
# List all schemas
manager = VespaSchemaManager(config)
schemas = await manager.list_all_schemas()
print(schemas)
```

**4. Document Feed Fails**
- Verify schema exists: `await backend.get_schema(schema_name)`
- Check document format matches schema fields
- Ensure tensor dimensions match schema definition

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
