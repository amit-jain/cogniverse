# Cogniverse SDK

**Last Updated:** 2025-11-13
**Layer:** Foundation
**Dependencies:** None (zero internal Cogniverse dependencies)

Pure backend interfaces for Cogniverse implementations.

## Overview

The SDK package sits at the **Foundation Layer** of the Cogniverse architecture, providing pure interface definitions with zero dependencies on other Cogniverse packages. All backend implementations and storage providers must satisfy these interfaces.

This package defines the contract for:
- **Backend**: Document/vector storage interface (search + ingestion)
- **ConfigStore**: Configuration persistence interface
- **SchemaLoader**: Schema template loading interface
- **Document**: Universal multi-modal document model

## Package Structure

```
cogniverse_sdk/
├── __init__.py
├── document.py              # Universal document model with multi-modal support
└── interfaces/
    ├── backend.py           # Backend interface (search, ingestion, deletion)
    ├── config_store.py      # Configuration storage interface
    └── schema_loader.py     # Schema template loader interface
```

## Key Modules

### Document Model (`cogniverse_sdk.document`)

Universal document representation supporting multiple content types:
- **Text**: Plain text and structured text content
- **Image**: Image documents with visual embeddings
- **Video**: Video documents with frame-level embeddings
- **Audio**: Audio documents with temporal features
- **Multi-modal**: Mixed content types in a single document

**Key Classes:**
- `Document`: Core document model with metadata, embeddings, and content
- `ContentType`: Enum for content type classification
- `DocumentMetadata`: Structured metadata container

### Backend Interface (`cogniverse_sdk.interfaces.backend`)

Abstract interface for document storage and retrieval backends:
- **Search Operations**: Vector search, hybrid search, filtering
- **Ingestion**: Batch and streaming document insertion
- **Schema Management**: Index and schema creation
- **Multi-tenancy**: Tenant-aware operations

**Key Classes:**
- `Backend`: Abstract base class for all backend implementations
- `SearchRequest`: Search query parameters
- `SearchResponse`: Search results with rankings and metadata

### ConfigStore Interface (`cogniverse_sdk.interfaces.config_store`)

Configuration persistence abstraction:
- **CRUD Operations**: Create, read, update, delete configurations
- **Versioning**: Configuration version tracking
- **Validation**: Schema-based config validation

### SchemaLoader Interface (`cogniverse_sdk.interfaces.schema_loader`)

Schema template management:
- **Template Loading**: Load Jinja2 schema templates
- **Variable Substitution**: Dynamic schema generation
- **Validation**: Schema correctness checking

## Installation

```bash
uv add cogniverse-sdk
```

Or with pip:
```bash
pip install cogniverse-sdk
```

## Usage Examples

### Implementing a Custom Backend

```python
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_sdk.document import Document, ContentType
from typing import List, Optional

class MyVectorBackend(Backend):
    """Custom vector database backend implementation."""

    def _initialize_backend(self, config: dict) -> None:
        """Initialize connection to your vector database."""
        self.db = YourVectorDB(config)

    async def search(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
        filters: Optional[dict] = None,
        tenant_id: Optional[str] = None
    ) -> List[Document]:
        """Execute vector similarity search."""
        results = await self.db.search(
            embedding=embedding,
            limit=top_k,
            filter=filters
        )
        return [self._to_document(r) for r in results]

    async def ingest(
        self,
        documents: List[Document],
        tenant_id: Optional[str] = None
    ) -> dict:
        """Ingest documents into the backend."""
        await self.db.insert(documents)
        return {"inserted": len(documents)}
```

### Creating Multi-Modal Documents

```python
from cogniverse_sdk.document import Document, ContentType
import numpy as np

# Text document
text_doc = Document(
    id="doc_001",
    content="Introduction to machine learning",
    content_type=ContentType.TEXT,
    metadata={
        "source": "textbook",
        "chapter": 1
    },
    embedding=np.random.rand(768).tolist()
)

# Video document with multi-modal embeddings
video_doc = Document(
    id="video_001",
    content="path/to/video.mp4",
    content_type=ContentType.VIDEO,
    metadata={
        "duration": 120.5,
        "fps": 30,
        "resolution": "1920x1080"
    },
    embedding=np.random.rand(512).tolist(),  # Video-level embedding
    frame_embeddings=[  # Frame-level embeddings
        np.random.rand(512).tolist() for _ in range(10)
    ]
)

# Image document
image_doc = Document(
    id="img_001",
    content="path/to/image.jpg",
    content_type=ContentType.IMAGE,
    metadata={
        "width": 1920,
        "height": 1080,
        "format": "JPEG"
    },
    embedding=np.random.rand(1024).tolist()  # Visual embedding
)
```

### Using the ConfigStore Interface

```python
from cogniverse_sdk.interfaces.config_store import ConfigStore

class SQLiteConfigStore(ConfigStore):
    """SQLite-based configuration storage."""

    def save_config(self, key: str, config: dict, tenant_id: str) -> None:
        """Save configuration to SQLite."""
        self.db.execute(
            "INSERT OR REPLACE INTO configs VALUES (?, ?, ?)",
            (tenant_id, key, json.dumps(config))
        )

    def load_config(self, key: str, tenant_id: str) -> Optional[dict]:
        """Load configuration from SQLite."""
        row = self.db.execute(
            "SELECT config FROM configs WHERE tenant_id=? AND key=?",
            (tenant_id, key)
        ).fetchone()
        return json.loads(row[0]) if row else None
```

## Architecture Position

```
Foundation Layer (SDK):
  cogniverse-sdk ← YOU ARE HERE (zero dependencies)
    ↓
  cogniverse-foundation (config base, telemetry interfaces)
    ↓
Core Layer:
  cogniverse-core, cogniverse-evaluation
    ↓
Implementation Layer:
  cogniverse-agents, cogniverse-vespa, cogniverse-synthetic
    ↓
Application Layer:
  cogniverse-runtime, cogniverse-dashboard
```

## External Dependencies

Minimal external dependencies for maximum portability:
- `numpy>=1.24.0`: For embedding array operations

No dependencies on other Cogniverse packages.

## Design Principles

1. **Pure Interfaces**: No concrete implementations, only abstract base classes
2. **Zero Coupling**: No dependencies on other Cogniverse packages
3. **Multi-Modal First**: Built-in support for text, image, video, audio
4. **Provider Agnostic**: Works with any backend (Vespa, Weaviate, Qdrant, etc.)
5. **Type Safe**: Full type hints and Pydantic validation

## Multi-Modal Support

The SDK provides first-class support for multi-modal documents:
- **Unified Document Model**: Single model for all content types
- **Content-Type Aware**: Explicit content type classification
- **Flexible Embeddings**: Support for document-level and granular embeddings
- **Rich Metadata**: Extensible metadata for modality-specific attributes

## Development

```bash
# Install in editable mode
cd libs/sdk
uv pip install -e .

# Run tests
pytest tests/
```

## License

MIT
