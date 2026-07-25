# Cogniverse SDK

Foundation contracts and shared data records for Cogniverse backends and
storage providers. The package has no dependency on another Cogniverse package.

## Package contents

```text
cogniverse_sdk/
├── __init__.py
├── document.py
└── interfaces/
    ├── __init__.py
    ├── adapter_store.py
    ├── backend.py
    ├── config_store.py
    ├── schema_loader.py
    └── workflow_store.py
```

- `document.py` defines `Document`, `DocumentFieldMapping`, `SearchResult`,
  `ContentType`, and `ProcessingStatus`.
- `backend.py` defines the `IngestionBackend`, `SearchBackend`, and combined
  `Backend` abstract contracts.
- `config_store.py` defines scoped, versioned configuration records and the
  `ConfigStore` contract.
- `schema_loader.py` defines schema-loading and schema-validation contracts.
- `adapter_store.py` defines persistence operations for trained adapters.
- `workflow_store.py` defines workflow execution, performance, template, and
  learning-corpus records and storage operations.

## Documents

`Document` is a dataclass that carries core content, flexible metadata, and any
number of named embeddings:

```python
from pathlib import Path

from cogniverse_sdk.document import ContentType, Document

document = Document(
    id="video-001",
    content_type=ContentType.VIDEO,
    content_path=Path("videos/tutorial.mp4"),
    title="Python tutorial",
    text_content="Transcript text",
    metadata={
        "duration": 120.5,
        "fps": 30,
        "resolution": [1920, 1080],
    },
)
document.add_embedding(
    "videoprism_global",
    [0.1, 0.2, 0.3],
    metadata={"model": "videoprism"},
)

assert document.get_embedding("videoprism_global") == [0.1, 0.2, 0.3]
assert document.get_embedding_metadata("videoprism_global") == {
    "model": "videoprism"
}
```

Embedding entries use only the wrapper created by `add_embedding`. Each wrapper
contains exactly `data`, `metadata`, and an integer-second `created_at`;
direct raw-vector entries and incomplete or additional wrapper fields are
invalid.

### Schema field mapping

Schemas can declare a `document_mapping` block that translates generic
document fields into their own field names:

```python
from cogniverse_sdk.document import DocumentFieldMapping

mapping = DocumentFieldMapping.from_dict(
    {
        "id": "document_id",
        "title": "document_title",
        "text_content": "full_text",
        "created_at": "creation_timestamp",
        "created_at_format": "epoch",
        "metadata_fields": {"metadata_category": "category"},
        "embeddings": {"videoprism_global": "embedding"},
    }
)

fields = document.to_schema_fields(mapping)
```

When `metadata_fields` renames a key, the source key is consumed and only the
destination schema field is emitted. Set `include_metadata` to `false` in JSON
configuration to emit only explicitly mapped core and metadata fields.

Mapping construction, loading, and serialization reject unknown keys,
mistyped field names, mapping dictionaries, and booleans before a backend feed
begins.

## Backend contracts

`IngestionBackend` and `SearchBackend` can be implemented separately.
`Backend` combines both and adds document updates, deletes, schema deployment,
and embedding-requirement methods. All interface methods are synchronous.

```python
from cogniverse_sdk.interfaces.backend import Backend


def require_ready(backend: Backend, config: dict) -> None:
    backend.initialize(config)
    if not backend.health_check():
        raise RuntimeError(f"{backend.name} is not healthy")
```

Concrete implementations must implement every abstract method with the exact
SDK signature. Calls to `Backend.initialize()` on one instance are serialized:
the backend-specific hook runs once after success, and an exception leaves the
instance retryable. Separate backend instances use independent locks.

## Configuration records

`ConfigEntry` stores one scoped, versioned configuration value:

```python
from datetime import datetime, timezone

from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope

now = datetime.now(timezone.utc)
entry = ConfigEntry(
    tenant_id="acme:acme",
    scope=ConfigScope.SYSTEM,
    service="video-search",
    config_key="ranking_profile",
    config_value={"name": "hybrid"},
    version=1,
    created_at=now,
    updated_at=now,
)

payload = entry.to_dict()
restored = ConfigEntry.from_dict(payload)
assert restored == entry
```

`ConfigStore` provides save, load, delete, list, active-version, and history
operations for these records. Its concrete implementations own persistence.
Config and workflow record datetimes must be timezone-aware; they are normalized
to UTC and stored as timezone-bearing ISO-8601 strings. Configuration
identifiers are strings, values are dictionaries, and versions are positive
Python integers.

## Workflow and adapter storage

`WorkflowStore` records workflow executions, agent performance, reusable
templates, learned patterns, and complete learning corpora. Its serialized
datetimes use the same canonical UTC form. Complete-corpus replacement is
serialized per tenant and replaces empty pattern mappings as well as non-empty
ones. Compensation attempts every channel; a failed compensation surfaces
alongside the forward failure in an `ExceptionGroup`.

`AdapterStore` persists adapter metadata, artifacts, training examples,
metrics, and activation state. Model types default to `"llm"` where the
interface declares that default.

## Dependencies

The package has no internal Cogniverse dependency. Its project metadata
currently pins `numpy==2.4.4`; the SDK source itself uses standard-library
types, while callers may store NumPy arrays as embedding values.

## Development

Run project commands from the repository root:

```bash
uv run ruff check libs/sdk
uv run ruff format --check libs/sdk
uv run pytest tests/backends/unit/test_sdk_document_contracts.py -v --tb=long
```

See [`docs/modules/sdk.md`](../../docs/modules/sdk.md) for the complete API
guide.
