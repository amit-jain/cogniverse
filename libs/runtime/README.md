# Cogniverse Runtime Server

**Package**: `cogniverse-runtime`
**Layer**: Application Layer (Light Blue/Purple)
**Version**: 0.1.0

FastAPI-based production server providing REST API endpoints for video ingestion, multi-modal search, and agent orchestration with multi-tenant isolation.

---

## Purpose

The `cogniverse-runtime` package provides:
- **FastAPI Server**: Production-ready HTTP server with async support
- **Ingestion Pipeline**: Video/document upload and processing endpoints
- **Search API**: Multi-modal search with routing and reranking
- **Agent Orchestration**: Route queries to appropriate agents
- **Multi-Tenant**: Tenant isolation at API level
- **Middleware**: Logging, error handling, CORS, rate limiting

---

## Architecture

### Position in 10-Package Structure

```
Foundation Layer (Blue)
├── cogniverse-sdk
└── cogniverse-foundation

Core Layer (Pink)
├── cogniverse-core
├── cogniverse-evaluation
└── cogniverse-telemetry-phoenix

Implementation Layer (Yellow/Green)
├── cogniverse-agents ← cogniverse-runtime depends on this
├── cogniverse-vespa ← cogniverse-runtime depends on this
└── cogniverse-synthetic ← cogniverse-runtime depends on this

Application Layer (Light Blue/Purple)
├── cogniverse-runtime ← YOU ARE HERE
└── cogniverse-dashboard
```

### Dependencies

**Workspace Dependencies:**
- `cogniverse-core` (required) - Multi-agent orchestration, memory, cache
- `cogniverse-agents` (required) - Routing, search, ingestion agents
- `cogniverse-vespa` (required) - Vespa backend operations
- `cogniverse-synthetic` (required) - Synthetic data generation
- `cogniverse-foundation` (transitive) - Base configuration and telemetry

**External Dependencies:**
- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `httpx>=0.25.0` - Async HTTP client

---

## Key Features

### 1. Video Ingestion Pipeline

Upload and process videos with automatic frame/chunk extraction:

```python
# POST /ingestion/videos
curl -X POST http://localhost:8000/ingestion/videos \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme_corp" \
  -d '{
    "video_path": "/data/videos/ml_tutorial.mp4",
    "profile": "frame_based",
    "metadata": {
      "title": "Machine Learning Tutorial",
      "tags": ["ml", "tutorial"]
    }
  }'
```

### 2. Multi-Modal Search

Search across video, image, and document modalities:

```python
# POST /search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme_corp" \
  -d '{
    "query": "machine learning tutorial",
    "modality": "video",
    "profile": "frame_based",
    "limit": 10,
    "rerank": true
  }'
```

### 3. Query Routing

Automatically route queries to appropriate agents:

```python
# POST /route
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme_corp" \
  -d '{
    "query": "show me videos about neural networks",
    "strategy": "llm"
  }'
```

### 4. Synthetic Data Generation

Generate training data for optimizers:

```python
# POST /synthetic/generate
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme_corp" \
  -d '{
    "optimizer": "modality",
    "count": 100,
    "modalities": ["video", "image", "pdf"]
  }'
```

---

## Installation

### Development (Editable Mode)

```bash
# From workspace root
uv sync

# Or install individually
uv pip install -e libs/runtime
```

### Production

```bash
pip install cogniverse-runtime

# Automatically installs all dependencies:
# - cogniverse-core, cogniverse-agents
# - cogniverse-vespa, cogniverse-synthetic
# - cogniverse-foundation
# - fastapi, uvicorn, pydantic
```

---

## Usage

### Starting the Server

```bash
# Development mode with auto-reload
uv run uvicorn cogniverse_runtime.main:app --reload --port 8000

# Production mode
uv run uvicorn cogniverse_runtime.main:app --host 0.0.0.0 --port 8000 --workers 4

# With environment variables
export TENANT_ID="acme_corp"
export VESPA_URL="http://localhost:8080"
export PHOENIX_ENDPOINT="http://localhost:6006"
uv run uvicorn cogniverse_runtime.main:app --port 8000
```

### Basic Integration

```python
from fastapi import FastAPI
from cogniverse_runtime import create_app

# Create app with custom config
app = create_app(
    title="Cogniverse API",
    version="0.1.0",
    enable_docs=True,
    enable_cors=True
)

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Custom Middleware

```python
from cogniverse_runtime import create_app
from cogniverse_runtime.middleware import (
    TenantMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware
)

app = create_app()

# Add custom middleware
app.add_middleware(TenantMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
```

---

## API Endpoints

### Ingestion

**POST /ingestion/videos**
- Upload and process videos
- Extracts frames/chunks, generates embeddings
- Stores in Vespa with tenant isolation

**POST /ingestion/documents**
- Upload PDF/text documents
- Extracts text, generates embeddings
- Stores in Vespa

**GET /ingestion/status/{job_id}**
- Check ingestion job status
- Returns progress and errors

### Search

**POST /search**
- Multi-modal search across videos, images, documents
- Supports vector search, hybrid search, reranking
- Tenant-isolated results

**POST /search/video**
- Video-specific search
- Frame-based or chunk-based search
- ColPali or VideoPrism profiles

**POST /search/image**
- Image-specific search
- CLIP embeddings
- Visual similarity

### Routing

**POST /route**
- Route queries to appropriate agents
- GLiNER or LLM-based routing
- Returns modality and strategy

**GET /route/strategies**
- List available routing strategies
- GLiNER, LLM, hybrid

### Synthetic Data

**POST /synthetic/generate**
- Generate training data
- Supports modality, crossmodal, routing, workflow optimizers
- Returns validated examples

**GET /synthetic/optimizers**
- List available optimizers

### Health & Metrics

**GET /health**
- Health check endpoint
- Returns status of all dependencies

**GET /metrics**
- Prometheus metrics endpoint
- Request counts, latencies, errors

---

## Configuration

Configuration via `SystemConfig` from `cogniverse-foundation`:

```python
from cogniverse_runtime import create_app
from cogniverse_foundation.config.unified_config import SystemConfig

config = SystemConfig(
    tenant_id="acme_corp",
    backend_url="http://localhost",
    backend_port=8080,
    telemetry_url="http://localhost:6006",
    environment="production",
)

app = create_app(config=config)
```

### Environment Variables

```bash
# Required
export TENANT_ID="acme_corp"
export VESPA_URL="http://localhost:8080"

# Optional
export VESPA_CONFIG_URL="http://localhost:19071"
export PHOENIX_ENDPOINT="http://localhost:6006"
export REDIS_URL="redis://localhost:6379"
export LOG_LEVEL="INFO"
export ENABLE_CORS="true"
export ALLOWED_ORIGINS="http://localhost:3000,https://app.example.com"
```

---

## Multi-Tenant Isolation

The runtime enforces tenant isolation at multiple levels:

### 1. Header-Based Tenant ID

```python
# All requests must include X-Tenant-ID header
curl -H "X-Tenant-ID: acme_corp" http://localhost:8000/search
```

### 2. Automatic Schema Suffixing

```python
# Runtime automatically uses tenant-suffixed schemas
# Base: video_colpali_mv_frame
# Actual: video_colpali_mv_frame_acme_corp
```

### 3. Tenant Middleware

```python
from cogniverse_runtime.middleware import TenantMiddleware

# Validates tenant_id on every request
# Rejects requests without valid tenant_id
# Attaches tenant context to request state
```

---

## Development

### Running Tests

```bash
# Run all runtime tests
uv run pytest tests/runtime/ -v

# Run integration tests (requires services)
uv run pytest tests/runtime/integration/ -v

# Run specific endpoint tests
uv run pytest tests/runtime/unit/test_search_endpoints.py -v
```

### Local Development Setup

```bash
# 1. Start required services
docker-compose up -d vespa phoenix redis

# 2. Run runtime server
uv run uvicorn cogniverse_runtime.main:app --reload

# 3. Access API docs
open http://localhost:8000/docs

# 4. Test endpoints
curl http://localhost:8000/health
```

### Code Style

```bash
# Format code
uv run ruff format libs/runtime

# Lint code
uv run ruff check libs/runtime

# Type check
uv run mypy libs/runtime
```

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv sync

# Run server
CMD ["uv", "run", "uvicorn", "cogniverse_runtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
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
      - TENANT_ID=acme_corp
      - VESPA_URL=http://vespa:8080
      - PHOENIX_ENDPOINT=http://phoenix:6006
      - REDIS_URL=redis://redis:6379
    depends_on:
      - vespa
      - phoenix
      - redis

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

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cogniverse-runtime
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cogniverse-runtime
  template:
    metadata:
      labels:
        app: cogniverse-runtime
    spec:
      containers:
      - name: runtime
        image: cogniverse/runtime:latest
        ports:
        - containerPort: 8000
        env:
        - name: TENANT_ID
          valueFrom:
            secretKeyRef:
              name: cogniverse-secrets
              key: tenant-id
        - name: VESPA_URL
          value: "http://vespa-service:8080"
        - name: PHOENIX_ENDPOINT
          value: "http://phoenix-service:6006"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
# Available at /metrics endpoint
from prometheus_client import Counter, Histogram

# Request metrics
http_requests_total = Counter('http_requests_total', 'Total HTTP requests')
http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Business metrics
video_ingestions_total = Counter('video_ingestions_total', 'Total video ingestions')
search_queries_total = Counter('search_queries_total', 'Total search queries')
routing_decisions_total = Counter('routing_decisions_total', 'Total routing decisions')
```

### OpenTelemetry Integration

```python
from cogniverse_foundation.telemetry import TelemetryManager

# Automatic tracing via foundation layer
telemetry = TelemetryManager(tenant_id="acme_corp")

# All requests automatically traced
# Spans sent to Phoenix for visualization
```

### Logging

```python
import logging

# Structured logging
logger = logging.getLogger("cogniverse.runtime")

logger.info("Video ingestion started", extra={
    "tenant_id": "acme_corp",
    "video_id": "video_001",
    "profile": "frame_based"
})
```

---

## Performance Tuning

### Concurrency

```bash
# Adjust worker count based on CPU cores
uv run uvicorn cogniverse_runtime.main:app --workers 4

# Enable async workers
uv run uvicorn cogniverse_runtime.main:app --workers 4 --loop uvloop
```

### Caching

```python
# Enable Redis caching for search results
config = SystemConfig(
    backend_url="http://localhost",
    backend_port=8080,
)
```

### Connection Pooling

```python
# HTTP connection pooling is configured at the HTTPX client level
config = SystemConfig(
    backend_url="http://localhost",
    backend_port=8080,
)
```

---

## Documentation

- **Architecture**: [10-Package Architecture](../../docs/architecture/10-package-architecture.md)
- **Multi-Tenant**: [Multi-Tenant Architecture](../../docs/architecture/multi-tenant.md)
- **API Docs**: [FastAPI Swagger UI](http://localhost:8000/docs)
- **Diagrams**: [SDK Architecture Diagrams](../../docs/diagrams/sdk-architecture-diagrams.md)

---

## Troubleshooting

### Common Issues

**1. Server Won't Start**
```bash
# Check port availability
lsof -i :8000

# Check dependencies
uv run python -c "import cogniverse_runtime; print('OK')"
```

**2. Tenant ID Not Found**
```bash
# Ensure X-Tenant-ID header is present
curl -H "X-Tenant-ID: acme_corp" http://localhost:8000/health
```

**3. Vespa Connection Failed**
```bash
# Test Vespa connectivity
curl http://localhost:8080/

# Check Vespa status
curl http://localhost:19071/ApplicationStatus
```

**4. Search Returns No Results**
- Verify documents exist: `curl http://localhost:8080/document/v1/...`
- Check tenant schema: should include `_tenant_id` suffix
- Ensure ingestion completed successfully

---

## Contributing

```bash
# Create feature branch
git checkout -b feature/runtime-improvement

# Make changes
# ...

# Run tests
uv run pytest tests/runtime/ -v

# Submit PR
```

---

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Related Packages

- **cogniverse-core**: Multi-agent orchestration (depends on this)
- **cogniverse-agents**: Agent implementations (depends on this)
- **cogniverse-vespa**: Vespa backend (depends on this)
- **cogniverse-synthetic**: Synthetic data (depends on this)
- **cogniverse-dashboard**: Streamlit UI (companion application)
