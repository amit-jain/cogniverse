# Runtime Module

**Package:** `cogniverse_runtime`
**Location:** `libs/runtime/cogniverse_runtime/`
**Purpose:** FastAPI server with video ingestion pipeline and multi-modal search APIs
**Last Updated:** 2026-01-13

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [FastAPI Server](#fastapi-server)
   - [Application Lifecycle](#application-lifecycle)
   - [Router Architecture](#router-architecture)
4. [Ingestion Pipeline](#ingestion-pipeline)
   - [VideoIngestionPipeline](#videoingestionpipeline)
   - [Processing Strategies](#processing-strategies)
   - [Processor Architecture](#processor-architecture)
5. [Search Service](#search-service)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Testing](#testing)

---

## Overview

The Runtime module provides the **Application Layer** for Cogniverse:

- **FastAPI Server**: Production-ready HTTP server with async support
- **Video Ingestion Pipeline**: Configurable pipeline for video processing (keyframes, chunks, transcription, embeddings)
- **Search API**: Multi-modal search with tenant isolation and session tracking
- **Strategy Pattern**: Pluggable processing strategies for different video analysis approaches
- **Processor Architecture**: Auto-discovery of processors with configuration from YAML

The runtime sits at the top of the package hierarchy, depending on all other modules.

---

## Package Structure

```
cogniverse_runtime/
├── main.py                          # FastAPI app entry point
├── config_loader.py                 # Dynamic backend/agent loading
├── routers/                         # API routers
│   ├── health.py                    # Health check endpoints
│   ├── search.py                    # Search API endpoints
│   ├── ingestion.py                 # Video ingestion endpoints
│   ├── agents.py                    # Agent orchestration endpoints
│   └── admin.py                     # Admin/tenant management
├── ingestion/                       # Video processing pipeline
│   ├── pipeline.py                  # VideoIngestionPipeline
│   ├── processor_base.py            # BaseProcessor, BaseStrategy
│   ├── processor_manager.py         # ProcessorManager for auto-discovery
│   ├── strategies.py                # Processing strategy implementations
│   ├── strategy_factory.py          # StrategyFactory for profile-based config
│   ├── processing_strategy_set.py   # Strategy orchestration
│   ├── exceptions.py                # Pipeline-specific exceptions
│   └── processors/                  # Processor implementations
│       ├── keyframe_extractor.py    # Frame extraction (similarity-based)
│       ├── keyframe_extractor_fps.py # Frame extraction (FPS-based)
│       ├── keyframe_processor.py    # Keyframe processing wrapper
│       ├── chunk_processor.py       # Video chunk extraction
│       ├── video_chunk_extractor.py # Chunk extraction utilities
│       ├── audio_transcriber.py     # Whisper transcription
│       ├── audio_processor.py       # Audio processing utilities
│       ├── vlm_processor.py         # VLM description generation
│       ├── vlm_descriptor.py        # VLM descriptor implementation
│       ├── single_vector_processor.py # Single-vector embeddings
│       └── embedding_generator/     # Embedding generation
│           ├── embedding_generator.py # Main generator interface
│           ├── embedding_generator_impl.py # Implementation
│           ├── embedding_processors.py # Model-specific processors
│           ├── document_builders.py # Vespa document builders
│           └── backend_factory.py   # Backend initialization
├── search/                          # Search service
│   ├── service.py                   # SearchService implementation
│   └── base.py                      # Search base classes
├── admin/                           # Admin functionality
│   ├── tenant_manager.py            # Tenant management
│   ├── models.py                    # Admin models
│   └── profile_models.py            # Profile configuration models
├── inference/                       # Inference services
│   └── modal_inference_service.py   # Modal-based inference
└── instrumentation/                 # Monitoring
    └── phoenix.py                   # Phoenix instrumentation
```

---

## FastAPI Server

### Application Lifecycle

The server uses FastAPI's lifespan context manager for startup/shutdown:

```python
from cogniverse_runtime.main import app
import uvicorn

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Startup Sequence:**

1. Load configuration via `ConfigManager`
2. Initialize `SchemaLoader` for Vespa schemas
3. Set dependencies on routers
4. Initialize `BackendRegistry` and `AgentRegistry`
5. Wire registries to routers
6. Load backends and agents from config

```python
# From main.py
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app."""

    # 1. Load configuration
    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)

    # 2. Initialize SchemaLoader
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    # 3. Set dependencies on routers
    admin.set_config_manager(config_manager)
    ingestion.set_config_manager(config_manager)
    ingestion.set_schema_loader(schema_loader)

    # 4. Initialize registries
    backend_registry = BackendRegistry(config_manager=config_manager)
    agent_registry = AgentRegistry(config_manager=config_manager)

    # 5. Wire agent registry
    agents.set_agent_registry(agent_registry)

    # 6. Load from config
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents()

    yield

    # Shutdown logic here
```

### Router Architecture

The server uses modular routers for different functionality:

| Router | Prefix | Purpose |
|--------|--------|---------|
| `health` | `/health` | Health checks, readiness probes |
| `search` | `/search` | Multi-modal search API |
| `ingestion` | `/ingestion` | Video upload and processing |
| `agents` | `/agents` | Agent orchestration endpoints |
| `admin` | `/admin` | Tenant and profile management |

```python
# Router registration in main.py
app.include_router(health.router, tags=["health"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
```

---

## Ingestion Pipeline

### VideoIngestionPipeline

The central class for video processing:

```python
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline, PipelineConfig
from cogniverse_foundation.config.utils import create_default_config_manager

# Initialize pipeline
config_manager = create_default_config_manager()
pipeline = VideoIngestionPipeline(
    tenant_id="acme",
    config_manager=config_manager,
    schema_loader=schema_loader,
    schema_name="video_colpali_mv_frame",  # Processing profile
    debug_mode=True
)

# Process single video
result = await pipeline.process_video_async(Path("video.mp4"))

# Process directory with concurrency
results = pipeline.process_directory(
    video_dir=Path("videos/"),
    max_concurrent=3
)
```

**Key Features:**
- **Profile-based configuration**: Each `schema_name` maps to a processing profile
- **Concurrent processing**: Process multiple videos in parallel
- **Caching**: Optional caching of intermediate results (keyframes, transcripts)
- **Strategy-driven**: Processing steps determined by strategy configuration

**PipelineConfig:**

```python
@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""

    extract_keyframes: bool = True
    transcribe_audio: bool = True
    generate_descriptions: bool = True
    generate_embeddings: bool = True

    # Processing parameters
    keyframe_threshold: float = 0.999
    max_frames_per_video: int = 3000
    vlm_batch_size: int = 500

    # Backend selection
    search_backend: str = "vespa"
```

### Processing Strategies

Strategies define how videos are processed. Each strategy specifies required processors:

**FrameSegmentationStrategy** - Extract individual frames (for ColPali):
```python
from cogniverse_runtime.ingestion.strategies import FrameSegmentationStrategy

strategy = FrameSegmentationStrategy(
    fps=1.0,              # Extract 1 frame per second
    threshold=0.999,       # Similarity threshold for deduplication
    max_frames=3000        # Maximum frames per video
)

# Required processors
strategy.get_required_processors()
# -> {"keyframe": {"fps": 1.0, "threshold": 0.999, "max_frames": 3000}}
```

**ChunkSegmentationStrategy** - Extract video chunks (for ColQwen, VideoPrism):
```python
from cogniverse_runtime.ingestion.strategies import ChunkSegmentationStrategy

strategy = ChunkSegmentationStrategy(
    chunk_duration=30.0,   # 30-second chunks
    chunk_overlap=0.0,     # No overlap
    cache_chunks=True      # Cache extracted chunks
)
```

**SingleVectorSegmentationStrategy** - Single-vector embeddings (for VideoPrism LVT):
```python
from cogniverse_runtime.ingestion.strategies import SingleVectorSegmentationStrategy

strategy = SingleVectorSegmentationStrategy(
    strategy="sliding_window",
    segment_duration=6.0,
    segment_overlap=1.0,
    sampling_fps=2.0,
    max_frames_per_segment=12
)
```

**Embedding Strategies:**
```python
from cogniverse_runtime.ingestion.strategies import (
    MultiVectorEmbeddingStrategy,
    SingleVectorEmbeddingStrategy,
)

# Multi-vector (ColPali, ColQwen)
mv_strategy = MultiVectorEmbeddingStrategy(model_name="vidore/colsmol-500m")

# Single-vector (VideoPrism)
sv_strategy = SingleVectorEmbeddingStrategy(model_name="google/videoprism-base")
```

### Processor Architecture

Processors are pluggable components that perform specific tasks:

**BaseProcessor:**
```python
from cogniverse_runtime.ingestion.processor_base import BaseProcessor
from typing import Any
import logging

class CustomProcessor(BaseProcessor):
    """Custom processor implementation."""

    PROCESSOR_NAME = "custom"  # Required identifier

    def __init__(self, logger: logging.Logger, param1: str = "default"):
        super().__init__(logger, param1=param1)
        self.param1 = param1

    def process(self, *args, **kwargs) -> Any:
        """Process input data."""
        # Implementation here
        pass
```

**ProcessorManager:**

Manages processor lifecycle and auto-discovery:

```python
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager

# Initialize
manager = ProcessorManager(logger)

# Initialize from strategy set
manager.initialize_from_strategies(strategy_set)

# Get processor by name
keyframe_processor = manager.get_processor("keyframe")

# List available processors
manager.list_processors()
```

**Available Processors:**

| Processor | Name | Purpose |
|-----------|------|---------|
| `KeyframeExtractor` | `keyframe` | Extract frames using similarity |
| `KeyframeExtractorFPS` | `keyframe_fps` | Extract frames at fixed FPS |
| `VideoChunkExtractor` | `chunk` | Extract video chunks |
| `AudioTranscriber` | `audio` | Whisper transcription |
| `VLMDescriptor` | `vlm` | Generate frame descriptions |
| `SingleVectorProcessor` | `single_vector` | Process for single-vector embeddings |
| `EmbeddingGenerator` | `embedding` | Generate and store embeddings |

---

## Search Service

The search service provides multi-modal search with tenant isolation:

```python
from cogniverse_runtime.search.service import SearchService
from cogniverse_foundation.config.utils import get_config, create_default_config_manager

config_manager = create_default_config_manager()
config = get_config(tenant_id="acme", config_manager=config_manager)

# Create service
search_service = SearchService(
    config=config,
    profile="video_colpali_mv_frame"
)

# Execute search
results = search_service.search(
    query="find videos about machine learning",
    top_k=10,
    strategy="hybrid",
    filters={"modality": "video"},
    tenant_id="acme"
)
```

**Search Strategies:**

| Strategy | Description |
|----------|-------------|
| `semantic` | Pure vector similarity search |
| `bm25` | BM25 keyword-based search |
| `hybrid` | Combines semantic and BM25 |
| `learned` | ML-based reranking |
| `multi_modal` | Multi-modal reranking (text, video, audio) |

---

## API Reference

### Search Endpoints

**POST /search/** - Execute search query
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "profile": "video_colpali_mv_frame",
    "strategy": "hybrid",
    "top_k": 10,
    "tenant_id": "acme",
    "session_id": "user-session-123"
  }'
```

**GET /search/strategies** - List available strategies
```bash
curl http://localhost:8000/search/strategies
```

**GET /search/profiles** - List available profiles
```bash
curl http://localhost:8000/search/profiles
```

**POST /search/rerank** - Rerank existing results
```bash
curl -X POST http://localhost:8000/search/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "results": [...],
    "strategy": "learned"
  }'
```

### Ingestion Endpoints

**POST /ingestion/videos** - Process video
```bash
curl -X POST http://localhost:8000/ingestion/videos \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d '{
    "video_path": "/data/videos/tutorial.mp4",
    "profile": "video_colpali_mv_frame",
    "metadata": {"title": "ML Tutorial"}
  }'
```

**GET /ingestion/status/{job_id}** - Check processing status
```bash
curl http://localhost:8000/ingestion/status/job-123
```

### Admin Endpoints

**GET /admin/tenants** - List tenants
**POST /admin/tenants** - Create tenant
**GET /admin/profiles** - List processing profiles
**POST /admin/profiles** - Create/update profile

### Health Endpoints

**GET /health** - Health check
**GET /health/ready** - Readiness probe
**GET /health/live** - Liveness probe

---

## Configuration

### Profile Configuration (config.yml)

Processing profiles are defined in the backend configuration:

```yaml
backend:
  default_profile: video_colpali_mv_frame
  profiles:
    video_colpali_mv_frame:
      type: multi_vector
      embedding_model: vidore/colsmol-500m
      strategies:
        segmentation:
          class: FrameSegmentationStrategy
          params:
            fps: 1.0
            threshold: 0.999
            max_frames: 3000
        transcription:
          class: AudioTranscriptionStrategy
          params:
            model: whisper-large-v3
        description:
          class: NoDescriptionStrategy
          params: {}
        embedding:
          class: MultiVectorEmbeddingStrategy
          params:
            model_name: vidore/colsmol-500m

    video_videoprism_sv_chunk:
      type: single_vector
      embedding_model: google/videoprism-base
      strategies:
        segmentation:
          class: SingleVectorSegmentationStrategy
          params:
            segment_duration: 6.0
            segment_overlap: 1.0
            sampling_fps: 2.0
        transcription:
          class: AudioTranscriptionStrategy
          params:
            model: whisper-large-v3
        description:
          class: NoDescriptionStrategy
          params: {}
        embedding:
          class: SingleVectorEmbeddingStrategy
          params:
            model_name: google/videoprism-base
```

### Environment Variables

```bash
# Required
export TENANT_ID="acme"
export VESPA_URL="http://localhost:8080"

# Optional
export VESPA_CONFIG_URL="http://localhost:19071"
export PHOENIX_ENDPOINT="http://localhost:6006"
export REDIS_URL="redis://localhost:6379"
export LOG_LEVEL="INFO"
export DEBUG_PIPELINE="false"

# Server configuration
export RUNTIME_HOST="0.0.0.0"
export RUNTIME_PORT="8000"
```

---

## Deployment

### Development

```bash
# Start with auto-reload
uv run uvicorn cogniverse_runtime.main:app --reload --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Production

```bash
# Multiple workers
uv run uvicorn cogniverse_runtime.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop

# With Gunicorn
uv run gunicorn cogniverse_runtime.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync

CMD ["uv", "run", "uvicorn", "cogniverse_runtime.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  runtime:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TENANT_ID=acme
      - VESPA_URL=http://vespa:8080
      - PHOENIX_ENDPOINT=http://phoenix:6006
    depends_on:
      - vespa
      - phoenix

  vespa:
    image: vespaengine/vespa
    ports:
      - "8080:8080"
      - "19071:19071"

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"
```

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           cogniverse-runtime ◄─── YOU ARE HERE              ││
│  │  FastAPI server, ingestion pipeline, search API             ││
│  └─────────────────────────────────────────────────────────────┘│
│                     cogniverse-dashboard                         │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                 Implementation Layer                             │
│  cogniverse-agents │ cogniverse-vespa │ cogniverse-synthetic    │
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
- `cogniverse-core`: Registries, orchestration, memory
- `cogniverse-agents`: Agent implementations
- `cogniverse-vespa`: Vespa backend operations
- `cogniverse-foundation`: Configuration and telemetry

**Dependents:**
- `cogniverse-dashboard`: Uses runtime APIs

---

## Testing

```bash
# Run all runtime tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/runtime/ -v

# Run integration tests (requires services)
JAX_PLATFORM_NAME=cpu uv run pytest tests/runtime/integration/ -v

# Run specific tests
uv run pytest tests/runtime/unit/test_pipeline.py -v
uv run pytest tests/runtime/unit/test_search_service.py -v

# Test with coverage
uv run pytest tests/runtime/ --cov=cogniverse_runtime --cov-report=html
```

**Test Categories:**
- `tests/runtime/unit/` - Unit tests for pipeline, processors, strategies
- `tests/runtime/integration/` - Integration tests with Vespa, Phoenix

---

## Related Documentation

- [Core Module](./core.md) - Agent base classes and registries
- [Foundation Module](./foundation.md) - Configuration and telemetry
- [Agents Module](./agents.md) - Agent implementations
- [Vespa Backend](../backends/vespa.md) - Vespa integration details
- [Configuration System](../CONFIGURATION_SYSTEM.md) - Profile configuration guide

---

**Summary:** The Runtime module provides the FastAPI application layer for Cogniverse. `VideoIngestionPipeline` handles video processing with a strategy pattern for flexible configuration. The search service provides multi-modal search with session tracking. Processing profiles define which strategies and processors to use for different video analysis approaches (frame-based ColPali, chunk-based ColQwen, single-vector VideoPrism).
