# Troubleshooting Guide

Common errors and solutions for Cogniverse.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Deployment and Cluster Issues](#deployment-and-cluster-issues)
3. [Import Errors](#import-errors)
4. [Vespa Issues](#vespa-issues)
5. [Phoenix Issues](#phoenix-issues)
6. [Embedding Issues](#embedding-issues)
7. [Agent Issues](#agent-issues)
8. [Configuration Issues](#configuration-issues)
9. [Testing Issues](#testing-issues)
10. [Runtime Issues](#runtime-issues)

---

## Installation Issues

### uv Not Found

**Error:**
```text
command not found: uv
```

**Solution:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify
uv --version
```

### Package Installation Fails

**Error:**
```text
error: Failed to resolve dependencies
```

**Solution:**
```bash
# Clear cache and reinstall
uv cache clean
uv sync --refresh

# If still failing, check Python version
python --version  # Should be 3.12+

# Install specific Python version
uv python install 3.12
uv sync
```

### Missing System Dependencies

**Error:**
```text
error: libffi.so not found
```

**Solution:**
```bash
# macOS
brew install libffi

# Ubuntu/Debian
sudo apt-get install libffi-dev

# Fedora/RHEL
sudo dnf install libffi-devel
```

---

## Deployment and Cluster Issues

Cogniverse deploys as a Helm chart onto a Kubernetes cluster. `cogniverse up`
(from the `cogniverse` CLI, `libs/cli/`) creates a local k3d cluster if none
exists and no other Kubernetes context is already reachable, then installs the
chart into the `cogniverse` namespace. Services are reached on the host through
the k3d loadbalancer's NodePort mappings, not raw `docker ps`/`docker logs`.

### Missing Prerequisites

**Error:**
```text
Missing prerequisites:
  k3d: curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
  kubectl: ...
  helm: ...
```

**Solution:**

`cogniverse up` checks for `docker`, `kubectl`, `helm`, and (when no existing
Kubernetes cluster is reachable) `k3d`, and offers to install what's missing.

```bash
# Let cogniverse up install missing tools interactively, or install manually
# with the printed commands, then re-run:
cogniverse up
```

### Port Already Allocated

**Error:**
```text
Error response from daemon: driver failed programming external connectivity:
Bind for 0.0.0.0:28000 failed: port is already allocated
```

**Solution:**

`cogniverse up` maps a fixed set of host ports through the k3d loadbalancer:
`8080`/`19071` (Vespa), `28000` (runtime), `28501` (dashboard), `26006`
(Phoenix), `4317` (OTLP), `11434` (Ollama), `2746` (Argo), plus `29001`-`29011`
for inference sidecars.

```bash
# Find what's holding a conflicting port
lsof -i :28000

# Exclude a port from the k3d loadbalancer mapping (e.g. a host LLM already
# using 11434) via the environment before `cogniverse up`
export COGNIVERSE_K3D_EXCLUDE_PORTS=11434
cogniverse up

# Or tear down and recreate the cluster
cogniverse down
cogniverse up
```

### Pods Stuck Pending / CrashLoopBackOff

**Solution:**
```bash
# Check overall status
cogniverse status

# List pods and their state directly
kubectl get pods -n cogniverse

# Tail logs for a specific service (runtime, dashboard, vespa, phoenix, llm, argo)
cogniverse logs runtime -f

# Describe a pod for scheduling/image-pull failures
kubectl describe pod <pod-name> -n cogniverse
```

---

## Import Errors

### ModuleNotFoundError

**Error:**
```text
ModuleNotFoundError: No module named 'cogniverse_core'
```

**Solution:**
```bash
# Ensure all packages are installed
uv sync

# Check package is installed
uv pip list | grep cogniverse

# Verify import works
uv run python -c "import cogniverse_core; print('OK')"

# If using IDE, ensure correct interpreter
which python
# Should point to .venv/bin/python
```

### Circular Import

**Error:**
```text
ImportError: cannot import name 'X' from partially initialized module
```

**Solution:**

- Check import order in `__init__.py`

- Use `TYPE_CHECKING` for type hints that cause cycles:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogniverse_core.agents import AgentBase

# Use string annotation
def process(agent: "AgentBase") -> None:
    ...
```

---

## Vespa Issues

### Connection Refused

**Error:**
```text
ConnectionRefusedError: [Errno 111] Connection refused
Backend: http://localhost:8080
```

**Solution:**
```bash
# Check Vespa pod status (cogniverse deploys to a k3d/Kubernetes cluster,
# not raw docker containers -- Vespa runs as the cogniverse-vespa StatefulSet)
cogniverse status

# Start Vespa (deploys the full stack, including Vespa, via k3d + Helm)
cogniverse up  # Starts all services including Vespa

# Check health
curl http://localhost:8080/ApplicationStatus

# Check logs
cogniverse logs vespa -f
```

### Schema Deployment Failed

**Error:**
```text
vespa.deployment.ApplicationError: Schema deployment failed
```

**Solution:**
```bash
# Check config server is ready
curl http://localhost:19071/ApplicationStatus

# Wait for config server (can take 30-60s after start)
sleep 60

# Schema deployment is handled by pyvespa ApplicationPackage
# See scripts/run_ingestion.py for schema deployment examples

# Check deployment status
curl http://localhost:8080/ApplicationStatus
```

### Embedding Dimension Mismatch

**Error:**
```text
Expected 768 values, got 1024
```

**Solution:**

- Check schema embedding dimension matches model:
  - ColPali (Tomoro ColQwen3), `video_colpali_smol500_mv_frame`: multi-vector, 320-dim patches (float) or 40-dim (binary): `tensor<bfloat16>(patch{}, v[320])`
  - VideoPrism base, single-vector profile `video_videoprism_lvt_base_sv_chunk_6s`: 768-dim: `tensor<float>(v[768])`
  - VideoPrism large, single-vector profile `video_videoprism_lvt_large_sv_chunk_6s`: 1024-dim: `tensor<float>(v[1024])`
  - VideoPrism base/large also ship multi-vector chunk profiles (`video_videoprism_base_mv_chunk_30s`, `video_videoprism_large_mv_chunk_30s`) using `tensor<bfloat16>(patch{}, v[768])` / `v[1024])`

```bash
# Check schema tensor types (schemas are JSON files in configs/schemas/)
cat configs/schemas/video_colpali_smol500_mv_frame_schema.json | grep -A2 '"embedding"'

# Embeddings use tensor type definitions with dimensions specified in brackets
# Example: tensor<bfloat16>(patch{}, v[320]) means multi-vector with 320-dim patches
```

### Search Returns No Results

**Possible Causes:**

1. Wrong tenant_id

2. Schema doesn't exist

3. No documents ingested

**Solution:**
```bash
# Check tenant_id matches (content cluster id is "cogniverse_content",
# defined in configs/services.xml)
curl http://localhost:8080/document/v1/?cluster=cogniverse_content

# List schemas
curl http://localhost:8080/ApplicationStatus | jq '.services'

# Check document count
curl http://localhost:8080/document/v1/<schema>/docid/?cluster=cogniverse_content | jq '.documentCount'

# Re-ingest if needed
uv run python scripts/run_ingestion.py --video_dir data/testset/evaluation/sample_videos --backend vespa
```

---

## Phoenix Issues

### Phoenix Won't Start

**Error:**
```text
Phoenix server failed to start
Port 6006 already in use
```

**Solution:**

`cogniverse up` deploys Phoenix as the `cogniverse-phoenix` StatefulSet inside the
k3d cluster; its container port (6006) is published on the host through the k3d
loadbalancer's NodePort **26006**, not 6006 directly.

```bash
# Check what's using the host port
lsof -i :26006

# Check pod status and logs
cogniverse status
cogniverse logs phoenix

# Restart services
cogniverse up
```

### No Traces Visible

**Possible Causes:**

1. Wrong project name

2. Telemetry not enabled

3. Spans not being exported

**Solution:**
```python
# Check project name format
# Should be: cogniverse-{tenant_id}-{project_name}

# Verify telemetry is enabled
from cogniverse_foundation.telemetry import get_telemetry_manager
telemetry = get_telemetry_manager()
print(telemetry.get_stats())

# Force flush spans
telemetry.force_flush(timeout_millis=30000)
```

```bash
# Check Phoenix UI (26006 is the k3d loadbalancer's host-published port
# for the cogniverse-phoenix service; its in-cluster port is 6006)
open http://localhost:26006

# Check OTLP endpoint is reachable
curl -s http://localhost:4317 || echo "Phoenix OTLP not reachable"
```

### OTLP Export Fails

**Error:**
```text
Failed to export spans: Connection refused to localhost:4317
```

**Solution:**
```bash
# Check OTLP endpoint is reachable
curl -s http://localhost:4317 || echo "Phoenix OTLP not reachable"

# Set the OTLP gRPC endpoint env var
export TELEMETRY_OTLP_ENDPOINT="localhost:4317"

# Or use the HTTP endpoint for Phoenix (26006 on the host under `cogniverse up`)
export TELEMETRY_HTTP_ENDPOINT="http://localhost:26006"
```

```python
# Or configure per-project overrides in code
from cogniverse_foundation.telemetry import get_telemetry_manager

telemetry = get_telemetry_manager()
telemetry.register_project(
    tenant_id="acme",
    project_name="test",
    otlp_endpoint="http://localhost:4317",
    http_endpoint="http://localhost:26006",
)
```

---

## Embedding Issues

### Model Loading Failed

**Error:**
```text
OSError: Unable to load weights from pytorch checkpoint
```

**Solution:**
```bash
# Clear huggingface cache
rm -rf ~/.cache/huggingface/hub

# Download model explicitly
uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('TomoroAI/tomoro-colqwen3-embed-4b')"

# Check disk space
df -h
```

### CUDA Out of Memory

**Error:**
```text
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
JAX_PLATFORM_NAME=cpu uv run python script.py

# Or limit frames per video
uv run python scripts/run_ingestion.py --tenant-id <tenant> --video_dir <dir> --max-frames 1

# Clear GPU memory
uv run python -c "import torch; torch.cuda.empty_cache()"
```

### JAX Platform Error

**Error:**
```text
RuntimeError: Unable to initialize JAX backend
```

**Solution:**
```bash
# Force CPU mode
export JAX_PLATFORM_NAME=cpu

# Or set in code (before any jax import)
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax  # Now uses CPU
```

---

## Agent Issues

### Agent Not Found

**Error:**
```text
KeyError: Agent 'search_agent' not found in registry
```

**Solution:**
```python
# Check agent is registered
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.common.agent_models import AgentEndpoint

config_manager = create_default_config_manager()
registry = AgentRegistry(tenant_id="your_org:production", config_manager=config_manager)
agents = registry.list_agents()
print(agents)

# Register agent endpoint
agent = AgentEndpoint(
    name="search_agent",
    url="http://localhost:8002",
    capabilities=["video_search"],
    health_endpoint="/health",
    process_endpoint="/process"
)
registry.register_agent(agent)
```

### Process Method Timeout

**Error:**
```text
asyncio.TimeoutError: Agent process exceeded timeout
```

**Solution:**

Agent timeout is controlled via `AgentEndpoint.timeout` (default: 30 seconds) or at the HTTP client level:

```python
# Configure timeout when registering agent
from cogniverse_core.common.agent_models import AgentEndpoint

agent = AgentEndpoint(
    name="search_agent",
    url="http://localhost:8002",
    capabilities=["video_search"],
    timeout=300  # 5 minutes
)
registry.register_agent(agent)

# HTTP client timeout is set in AgentRegistry (default: 10 seconds)
```

### Type Validation Error

**Error:**
```text
pydantic.ValidationError: 1 validation error for SearchInput
query
  field required
```

**Solution:**
```python
# Check required fields
from cogniverse_agents.search_agent import SearchInput

# All required fields must be provided
input = SearchInput(
    query="test",       # Required
    tenant_id="acme",   # Required
    modality="video",   # Has default value
    top_k=10            # Has default value
)
```

---

## Configuration Issues

### Config Not Found

**Error:**
```text
KeyError: Configuration for tenant 'acme' not found
```

**Solution:**

`ConfigManager.get_system_config()` never raises for a missing config -- if
nothing has been persisted yet it logs a warning and returns a `SystemConfig()`
built from defaults. To make that default durable, write it back explicitly:

```python
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
config = config_manager.get_system_config()  # returns defaults if unset

# Persist the (possibly default) config so future reads are explicit
config_manager.set_system_config(config)
```

### Invalid Profile

**Error:**
```text
ValueError: Profile 'video_colpali_frame' not found
```

**Solution:**
```bash
# List available profiles (port 8000 for a local `uv run uvicorn` runtime;
# 28000 if the runtime was deployed via `cogniverse up`)
curl http://localhost:8000/search/profiles

# Check config file directly
cat configs/config.json | jq '.backend.profiles'
```

```python
# Get available profiles for a tenant. `service` defaults to "backend" --
# the same config service the runtime admin API and dashboard read/write.
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
profiles = config_manager.list_backend_profiles(tenant_id="acme")
print(profiles.keys())
```

### Environment Variable Missing

**Error:**
```text
KeyError: 'BACKEND_URL' not set
```

**Solution:**
```bash
# Set required variables
export TENANT_ID="acme"
export BACKEND_URL="http://localhost"
export BACKEND_PORT="8080"

# Or use .env file
cat > .env << EOF
TENANT_ID=acme
BACKEND_URL=http://localhost
BACKEND_PORT=8080
EOF

# Load .env in Python
from dotenv import load_dotenv
load_dotenv()
```

---

## Testing Issues

### Tests Fail with JAX Error

**Error:**
```text
RuntimeError: jax.local_devices returned no devices
```

**Solution:**
```bash
# Always run tests with CPU mode
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# Add to conftest.py
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
```

### Async Test Not Running

**Error:**
```text
PytestUnhandledCoroutineWarning: async def test_...
```

**Solution:**

The repo's root `pytest.ini` sets `asyncio_mode = auto`, so `async def test_*`
functions run automatically -- no `@pytest.mark.asyncio` decorator is needed
when tests run from the repo root. This warning almost always means either
`pytest-asyncio` isn't installed in the active environment, or pytest was
invoked from a directory where `pytest.ini` isn't discovered:

```bash
# Confirm pytest-asyncio is installed (pinned in pyproject.toml)
uv run python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"

# Run from the repo root so pytest.ini (asyncio_mode = auto) is picked up
uv run pytest tests/path/to/test_module.py -v
```

### Fixture Not Found

**Error:**
```text
fixture 'config_manager' not found
```

**Solution:**
```python
# Check conftest.py is in test directory
# tests/conftest.py

import pytest

@pytest.fixture
def config_manager_memory():
    """ConfigManager with in-memory store for unit testing."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)
```

---

## Runtime Issues

### Server Won't Start

**Error:**
```text
uvicorn.error: Can't start server: Address already in use
```

**Solution:**
```bash
# Check what's using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uv run uvicorn cogniverse_runtime.main:app --port 8001
```

### API Returns 500 Error

**Solution:**
```bash
# Check server logs
uv run uvicorn cogniverse_runtime.main:app --log-level debug

# Check traceback in response (tenant_id is required)
curl -v http://localhost:8000/search/ -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "tenant_id": "acme"}'
```

### Streamlit Dashboard Crashes

**Error:**
```text
StreamlitAPIError: Unable to connect
```

**Solution:**
```bash
# Check dependencies
uv pip list | grep streamlit

# Clear cache
streamlit cache clear

# Run with debug
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --logger.level=debug

# Check port
lsof -i :8501
```

---

## Getting Help

If you can't resolve an issue:

1. **Check logs** for detailed error messages
2. **Search existing issues** on GitHub
3. **Create a minimal reproduction** of the problem
4. **Open an issue** with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)
   - Relevant config/code snippets
