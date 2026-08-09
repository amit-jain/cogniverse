# Cogniverse Vespa Backend

`cogniverse-vespa` implements Cogniverse document ingestion, search, tenant
schema management, configuration storage, and adapter storage on Vespa.

## Installation

From the workspace root:

```bash
uv sync
```

To install only this package in editable mode:

```bash
uv pip install -e libs/vespa
```

The package declares its direct workspace dependencies
(`cogniverse-sdk`, `cogniverse-core`, and `cogniverse-foundation`) and its
direct external dependencies (`pyvespa`, `numpy`, `pydantic`, and `requests`).

## Package structure

| Module | Responsibility |
| --- | --- |
| `backend.py` | Registry-facing `VespaBackend` for ingestion, search, schema operations, raw document operations, and lifecycle management |
| `ingestion_client.py` | `VespaPyClient` document mapping plus synchronous and asynchronous pyvespa feeds |
| `search_backend.py` | Profile-aware queries, result conversion, connection pooling, metrics, health checks, and embedding export |
| `vespa_schema_manager.py` | Application-package deployment and tenant schema creation, discovery, and deletion |
| `json_schema_parser.py` and `metadata_schemas.py` | Conversion of repository JSON schemas to pyvespa objects and assembly of the four management schemas |
| `ranking_strategy_extractor.py` and `strategy_aware_processor.py` | Ranking-profile inspection and selection of the embedding fields required during ingestion and search |
| `embedding_processor.py` | Single-vector and multi-vector float or binary embedding conversion |
| `config/config_store.py` | Versioned tenant configuration persistence through `VespaConfigStore` |
| `registry/adapter_store.py` | Adapter metadata persistence and activation through `VespaAdapterStore` |
| `memory_config.py` | Strict `VespaConfig` model for vector-store connection settings |
| `config_utils.py` | Data-port and config-server-port conventions |
| `_vespa_factory.py` and `_yql.py` | Internal persistent-client, fail-fast response, and YQL-quoting helpers |

## Backend setup

Use `BackendRegistry` to construct an initialized backend. The registry wires
the schema registry and shares the backend instance with other Cogniverse
components.

`create_default_config_manager()` reads the Vespa data endpoint from the
bootstrap environment:

```bash
export BACKEND_URL="http://localhost"
export BACKEND_PORT="8080"
```

```python
from pathlib import Path

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.utils import create_default_config_manager

tenant_id = "acme:production"
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
config_manager = create_default_config_manager()

backend = BackendRegistry.get_instance().get_ingestion_backend(
    name="vespa",
    tenant_id=tenant_id,
    config={
        "backend": {
            "url": "http://localhost",
            "port": 8080,
            "config_port": 19071,
            "profiles": {
                "agent_memories": {
                    "type": "document",
                    "schema_name": "agent_memories",
                }
            },
        }
    },
    config_manager=config_manager,
    schema_loader=schema_loader,
)
```

The content endpoint and config server are separate boundaries. `port` is the
Vespa content port; `config_port` is the Vespa config-server port.

## Tenant schemas

Tenant IDs are canonicalized before they are used in schema names:

| Input tenant | Canonical tenant | Schema suffix |
| --- | --- | --- |
| `acme` | `acme:acme` | `acme_acme` |
| `acme:production` | `acme:production` | `acme_production` |

Deploy a base schema through the backend's schema registry, then ask the
backend for the physical tenant schema:

```python
backend.schema_registry.deploy_schema(
    tenant_id=tenant_id,
    base_schema_name="agent_memories",
)
tenant_schema = backend.get_tenant_schema_name(tenant_id, "agent_memories")
assert tenant_schema == "agent_memories_acme_production"
```

Deployment preserves registered and live schemas. If either registry
enumeration or Vespa document-type enumeration fails, deployment raises
instead of treating the unavailable source as empty.

Tenant deletion first removes the schema from Vespa and then records the
registry tombstone. A tombstone failure is reported to the caller and can be
retried even when Vespa has already removed the schema. A returned `[]` means
there was no live schema left to remove; it does not hide registry errors.

## Raw document operations

The public raw-field methods exercise Vespa's document API without going
through a model-dependent ingestion pipeline:

```python
fields = {
    "id": "memory-42",
    "text": "The deployment completed at 09:15 UTC.",
    "user_id": "operator-7",
    "agent_id": "release-agent",
    "metadata_": '{"tenant_id":"acme:production"}',
    "created_at": 1785038100000,
    "embedding": [0.0] * 768,
}

backend.put_document_fields(
    "memory-42",
    fields,
    schema_name=tenant_schema,
    namespace="memory_content",
)

stored = backend.get_document_fields(
    "memory-42",
    schema_name=tenant_schema,
    namespace="memory_content",
)
assert stored["text"] == fields["text"]

backend.delete_document_fields(
    "memory-42",
    schema_name=tenant_schema,
    namespace="memory_content",
)
```

Point reads should always pass both the physical schema and namespace. A
genuine 404 maps to `None` for `get_document_fields`. The mapped
`delete_document` operation treats a genuine 404 as idempotent; transport
failures and other non-success statuses raise.

## Search

Search uses a configured backend profile and always requires a tenant:

```python
import numpy as np

query_embedding = np.asarray([1.0] + [0.0] * 767, dtype=np.float32)
results = backend.search(
    {
        "query": "deployment completion time",
        "type": "document",
        "profile": "agent_memories",
        "strategy": "semantic_search",
        "tenant_id": tenant_id,
        "top_k": 5,
        "query_embeddings": query_embedding,
    }
)
```

The result is a list of `cogniverse_sdk.document.SearchResult` values in Vespa
ranking order. Missing or malformed hits, root-level Vespa errors, and
degraded coverage raise; they are not converted into an empty result set.

Tenant profile definitions override same-named global profiles. The search
request uses one immutable profile snapshot, so a concurrent profile update
cannot mix schema and ranking settings within a single request.

## Health and lifecycle

After the lazy search backend has been constructed, `backend.health_check()`
probes Vespa and rejects non-200 responses, root errors, and degraded coverage.
Before that search path exists, it reports whether schema management was
initialized; it does not issue a network probe.

`backend.close()` closes the search connection pool and cached document
sessions. Once a pool is closed, waiting and future acquisitions fail rather
than reusing a connection after shutdown.

## Development

Always run Python commands through `uv`:

```bash
uv run ruff check libs/vespa tests/backends
uv run ruff format --check libs/vespa tests/backends
uv run pytest tests/backends/unit -v --tb=long
uv run pytest tests/backends/integration -v --tb=long
```

Integration tests manage their own Vespa container and unique host ports.

Further details:

- [Backend module guide](../../docs/modules/backends.md)
- [Multi-tenant architecture](../../docs/architecture/multi-tenant.md)
- [Multi-tenant operations](../../docs/operations/multi-tenant-ops.md)
- [Vespa search strategies](../../docs/testing/vespa_search_strategies.md)
