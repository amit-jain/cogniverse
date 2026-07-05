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
- **Agent Orchestration**: Dispatch queries to registered agents in-process
- **Multi-Tenant**: Tenant isolation at API level
- **Middleware**: CORS (the only middleware installed on the app)

---

## Architecture

### Position in the 13-Package Workspace

```
Foundation Layer
├── cogniverse-sdk
└── cogniverse-foundation

Core Layer
├── cogniverse-core
└── cogniverse-evaluation

Implementation Layer
├── cogniverse-telemetry-phoenix
├── cogniverse-agents ← cogniverse-runtime depends on this (optional extra)
├── cogniverse-vespa ← cogniverse-runtime depends on this (optional extra)
├── cogniverse-synthetic ← pulled in transitively via the `agents` extra
└── cogniverse-finetuning

Application Layer
├── cogniverse-runtime ← YOU ARE HERE
├── cogniverse-cli
├── cogniverse-messaging
└── cogniverse-dashboard
```

### Dependencies

**Workspace Dependencies (see `pyproject.toml`):**
- `cogniverse-sdk` (required) - Backend/config-store interfaces
- `cogniverse-core` (required) - Registries, memory, tenant utilities
- `cogniverse-foundation` (transitive, via `cogniverse-core`) - Base configuration and telemetry
- `cogniverse-vespa` (optional extra `[vespa]`) - Vespa backend operations
- `cogniverse-agents` (optional extra `[agents]`) - Routing, search, ingestion agents
- `cogniverse-synthetic` (transitive, via the `agents` extra) - Synthetic data generation; the `/synthetic/*` routes are always mounted (`main.py` imports the router unconditionally) so a deployment needs the `agents` extra installed for them to work

**External Dependencies:**
- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `httpx>=0.25.0` - Async HTTP client

---

## Key Features

### 1. Video Ingestion Pipeline

Start a directory-based ingestion job (`tenant_id` travels in the JSON body, not a header):

```bash
# POST /ingestion/start
curl -X POST http://localhost:8000/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "video_dir": "/data/videos/ml_tutorials",
    "profile": "frame_based_colpali",
    "backend": "vespa",
    "tenant_id": "acme_corp",
    "batch_size": 10
  }'
```

For single-file uploads, `POST /ingestion/upload` accepts a multipart form (`file`, `profile`, `backend`, `tenant_id` as form fields) and streams the upload to MinIO/Redis for queue-driven processing.

### 2. Multi-Modal Search

Search across configured backend profiles (`tenant_id` is a field on the request body):

```bash
# POST /search/
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "profile": "frame_based_colpali",
    "top_k": 10,
    "tenant_id": "acme_corp"
  }'
```

### 3. Agent Task Dispatch

Dispatch a query to a specific registered agent (there is no standalone `/route` endpoint — routing happens by calling an agent, e.g. `gateway_agent`, directly):

```bash
# POST /agents/{agent_name}/process
curl -X POST http://localhost:8000/agents/gateway_agent/process \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "gateway_agent",
    "query": "show me videos about neural networks",
    "top_k": 5
  }'
```

### 4. Synthetic Data Generation

Generate training data for optimizers (`optimizer` must be one of `profile`, `routing`, `workflow`, `unified`):

```bash
# POST /synthetic/generate
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "profile",
    "count": 100,
    "tenant_id": "acme_corp"
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
pip install cogniverse-runtime[vespa,agents]

# Installs:
# - cogniverse-sdk, cogniverse-core (required; cogniverse-foundation transitively)
# - cogniverse-vespa (via the `vespa` extra)
# - cogniverse-agents, and cogniverse-synthetic transitively (via the `agents` extra)
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
export BACKEND_URL="http://localhost"
export BACKEND_PORT="8080"
export TELEMETRY_OTLP_ENDPOINT="localhost:4317"
uv run uvicorn cogniverse_runtime.main:app --port 8000
```

### Basic Integration

```python
# The app object is defined at module level in cogniverse_runtime.main
from cogniverse_runtime.main import app

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## API Endpoints

### Ingestion

**POST /ingestion/start**
- Start a directory-based ingestion job in the background
- Body: `video_dir`, `profile`, `backend`, `tenant_id`, `org_id`, `max_videos`, `batch_size`
- Returns a `job_id` immediately; poll status separately

**GET /ingestion/status/{job_id}**
- Check ingestion job status
- Returns progress and errors

**POST /ingestion/upload**
- Multipart upload of a single file; streams to MinIO and enqueues via Redis
- Requires `REDIS_URL` and `MINIO_ENDPOINT` to be configured (503 otherwise)
- Query params `wait`, `wait_timeout`, `force` control synchronous behavior and idempotency

### Search

**POST /search/**
- Search a configured backend profile
- Body includes `query`, `profile`, `strategy`, `top_k`, `filters`, `tenant_id`, `stream`
- `stream: true` returns a Server-Sent Events response instead of JSON

**GET /search/strategies**
- List available ranking strategies (semantic, bm25, hybrid, learned, multi_modal)

**GET /search/profiles?tenant_id=...**
- List backend profiles visible to a tenant (name, model, type)

**POST /search/rerank**
- Rerank a previously fetched result set using a named strategy

### Agents

**GET /agents/**
- List all registered agents

**GET /agents/stats**
- Registry statistics including health status

**GET /agents/by-capability/{capability}**
- Find agents that advertise a given capability

**GET /agents/{agent_name}**
- Get metadata for a specific agent

**POST /agents/{agent_name}/process**
- Dispatch a task to an agent in-process; body is `AgentTask` (`agent_name`, `query`, `context`, `top_k`, plus optional multi-turn/enrichment fields)

**POST /agents/{agent_name}/message**
- Enqueue an inbound steering message for a running agent session

**GET /agents/{agent_name}/sessions/{session_id}?tenant_id=...**
- Check whether an agent session is currently active

**POST /agents/register** / **DELETE /agents/{agent_name}**
- Self-register or unregister an agent (Curated Registry / A2A pattern)

There is no standalone `/route` endpoint — query routing is performed by dispatching to an agent (e.g. `gateway_agent`) via `POST /agents/{agent_name}/process`.

### Synthetic Data

**POST /synthetic/generate**
- Generate training data for one optimizer per call
- `optimizer` must be one of `profile`, `routing`, `workflow`, `unified`; `count` and `tenant_id` are required

**GET /synthetic/optimizers**
- List available optimizers

**GET /synthetic/optimizers/{optimizer_name}**
- Get details for one optimizer

**GET /synthetic/health**
- Health check for the synthetic data service

**POST /synthetic/batch/generate**
- Generate data for multiple optimizers in one call

### Health

**GET /health**
- Health check endpoint; returns registered backend/agent counts, or 503 if the service can't assemble its status

**GET /health/live**
- Kubernetes liveness probe

**GET /health/ready**
- Kubernetes readiness probe; 503 until at least one backend is registered

There is no `/metrics` endpoint — the runtime does not expose Prometheus metrics.

---

## Configuration

Configuration via `SystemConfig` from `cogniverse-foundation`. The server reads config
at startup through `cogniverse_runtime.config_loader`:

```python
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.main import app  # module-level FastAPI instance

# Configuration is loaded from environment variables and configs/config.json
# at startup — see the Environment Variables section below.
```

### Environment Variables

Tenant identity is per-request (see [Multi-Tenant Isolation](#multi-tenant-isolation) below), not a server-wide env var — there is no `TENANT_ID` variable.

```bash
# Required
export BACKEND_URL="http://localhost"
export BACKEND_PORT="8080"

# Optional (deployment overrides read at startup — see main.py lifespan)
export TELEMETRY_OTLP_ENDPOINT="cogniverse-phoenix:4317"
export TELEMETRY_HTTP_ENDPOINT="http://localhost:6006"
export REDIS_URL="redis://localhost:6379"
export MINIO_ENDPOINT="http://localhost:9000"
export LLM_ENGINE="ollama"
export LLM_MODEL="gemma-4-e4b-it"
```

---

## Multi-Tenant Isolation

There is no tenant middleware and no `X-Tenant-ID` header. Tenant identity travels
in the request itself and is validated per-endpoint:

### 1. Body/Query-Based Tenant ID

Each request model carries its own `tenant_id` field (or query parameter where
the route has no body, e.g. `GET /search/profiles`). `cogniverse_core.common.
tenant_utils.require_tenant_id` rejects a missing value with HTTP 400, and
`assert_tenant_exists` rejects an unregistered tenant with HTTP 404:

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "profile": "frame_based_colpali", "tenant_id": "acme_corp"}'
```

### 2. Automatic Schema Suffixing

```python
# Runtime automatically uses tenant-suffixed schemas
# Base: video_colpali_mv_frame
# Actual: video_colpali_mv_frame_acme_corp
```

### 3. Tenant Self-Service

`cogniverse_runtime.routers.tenant` is mounted at `/admin/tenant` and provides
per-tenant self-service endpoints (instructions, memory browsing/deletion,
scheduled jobs) — it is a regular router, not a middleware, and does not run
on every request. Tenant registration/lifecycle (create, list, delete) lives
in `cogniverse_runtime.admin.tenant_manager`, mounted under `/admin`.

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
cogniverse up  # Starts Vespa, Phoenix, and other services via k3d

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
      - BACKEND_URL=http://vespa
      - BACKEND_PORT=8080
      - TELEMETRY_OTLP_ENDPOINT=phoenix:4317
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
        - name: BACKEND_URL
          value: "http://vespa-service"
        - name: BACKEND_PORT
          value: "8080"
        - name: TELEMETRY_OTLP_ENDPOINT
          value: "phoenix-service:4317"
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

The runtime does not expose a `/metrics` endpoint or Prometheus counters.

### OpenTelemetry Integration

```python
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

# Global singleton, loaded from ConfigManager on first call
telemetry = get_telemetry_manager()

with telemetry.span("search_service.search", tenant_id="acme_corp") as span:
    span.set_attribute("query", "test")
    # ... search logic ...

# Spans are sent to Phoenix (TELEMETRY_OTLP_ENDPOINT) for visualization
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

- **Architecture**: [System Architecture](../../docs/architecture/overview.md)
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
# Ensure tenant_id is set in the request body (or query param) and the
# tenant has been registered via POST /admin/tenants first
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "profile": "frame_based_colpali", "tenant_id": "acme_corp"}'
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

- **cogniverse-core**: Registries, memory, tenant utilities (required dependency)
- **cogniverse-agents**: Agent implementations (optional `agents` extra)
- **cogniverse-vespa**: Vespa backend (optional `vespa` extra)
- **cogniverse-synthetic**: Synthetic data generation (transitive, via the `agents` extra)
- **cogniverse-dashboard**: Streamlit UI (companion application)
