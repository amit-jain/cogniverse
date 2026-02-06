# Troubleshooting Guide

Common errors and solutions for Cogniverse.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import Errors](#import-errors)
3. [Vespa Issues](#vespa-issues)
4. [Phoenix Issues](#phoenix-issues)
5. [Embedding Issues](#embedding-issues)
6. [Agent Issues](#agent-issues)
7. [Configuration Issues](#configuration-issues)
8. [Testing Issues](#testing-issues)
9. [Runtime Issues](#runtime-issues)

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
uv python install 3.11
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
python -c "import cogniverse_core; print('OK')"

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
# Check Vespa is running
docker ps | grep vespa

# Start Vespa
docker-compose up -d vespa

# Check health
curl http://localhost:8080/ApplicationStatus

# Check logs
docker logs vespa
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
  - ColPali: Multi-vector format with 128-dim patches (float) or 16-dim (binary): `tensor<bfloat16>(patch{}, v[128])`
  - VideoPrism base: 768-dim single vector: `tensor<float>(x[768])`
  - VideoPrism large: 1024-dim single vector: `tensor<float>(x[1024])`

```bash
# Check schema tensor types (schemas are JSON files in configs/schemas/)
cat configs/schemas/video_colpali_smol500_mv_frame_schema.json | grep -A2 '"embedding"'

# Embeddings use tensor type definitions with dimensions specified in brackets
# Example: tensor<bfloat16>(patch{}, v[128]) means multi-vector with 128-dim patches
```

### Search Returns No Results

**Possible Causes:**

1. Wrong tenant_id

2. Schema doesn't exist

3. No documents ingested

**Solution:**
```bash
# Check tenant_id matches
curl http://localhost:8080/document/v1/?cluster=content

# List schemas
curl http://localhost:8080/ApplicationStatus | jq '.services'

# Check document count
curl http://localhost:8080/document/v1/<schema>/docid/?cluster=content | jq '.documentCount'

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
```bash
# Check what's using port
lsof -i :6006

# Kill existing process
kill -9 <PID>

# Or use different port
PHOENIX_PORT=6007 docker-compose up -d phoenix

# Start fresh
docker-compose down phoenix
docker-compose up -d phoenix
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
from cogniverse_foundation.telemetry.manager import TelemetryManager
telemetry = TelemetryManager()
print(telemetry.get_stats())

# Force flush spans
telemetry.force_flush(timeout_millis=30000)
```

```bash
# Check Phoenix UI
open http://localhost:6006

# Check OTLP endpoint
curl http://localhost:4317/v1/traces
```

### OTLP Export Fails

**Error:**
```text
Failed to export spans: Connection refused to localhost:4317
```

**Solution:**
```bash
# Check OTLP endpoint is running
curl http://localhost:4317/v1/traces

# Use HTTP endpoint instead
export OTLP_ENDPOINT="http://localhost:6006/v1/traces"

# Or configure in code
telemetry.register_project(
    tenant_id="acme",
    project_name="test",
    http_endpoint="http://localhost:6006/v1/traces"
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
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vidore/colsmol-500m')"

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

# Or reduce batch size
uv run python scripts/run_ingestion.py --batch_size 1

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
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
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.common.agent_models import AgentEndpoint

registry = AgentRegistry(tenant_id="default", config_manager=config_manager)
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
    query="test",  # Required
    modality="video",  # Has default value
    top_k=10  # Has default value
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
```python
# Check config exists
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.config.unified_config import SystemConfig

config_manager = create_default_config_manager()
try:
    config = config_manager.get_system_config(tenant_id="acme")
except (KeyError, ValueError):
    # Create default config
    config_manager.set_system_config(
        SystemConfig(tenant_id="acme"),
        tenant_id="acme"
    )
```

### Invalid Profile

**Error:**
```text
ValueError: Profile 'video_colpali_mv_frame' not found
```

**Solution:**
```bash
# List available profiles
curl http://localhost:8000/search/profiles

# Check config file
cat configs/config.json | jq '.backend.profiles'
```

```python
# Get available profiles
profiles = config_manager.list_backend_profiles(tenant_id="acme")
print(profiles)
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
```python
# Add decorator
@pytest.mark.asyncio
async def test_async_operation():
    result = await agent.process(input)
    assert result is not None

# Or install plugin
# pip install pytest-asyncio
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

# Check traceback in response
curl -v http://localhost:8000/search/ -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
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
uv run streamlit run app.py --logger.level=debug

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
