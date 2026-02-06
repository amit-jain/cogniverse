# Troubleshooting Guide

---

## Table of Contents

1. [Test Failures](#test-failures)
2. [Import Errors](#import-errors)
3. [Model Loading Issues](#model-loading-issues)
4. [Memory and Performance](#memory-and-performance)
5. [Vespa Issues](#vespa-issues)
6. [Agent Communication](#agent-communication)

---

## Test Failures

### Segmentation Faults in Async Tests

**Symptoms:**
```text
Fatal Python error: Segmentation fault

Thread 0x000000033614f000 (most recent call first):
  File "/path/to/threading.py", line 359 in wait
  File "/path/to/tqdm/_monitor.py", line 60 in run
```

**Cause:**
Threading conflicts between pytest async event loops and background threads from:

- **tqdm** (transformers progress bars)

- **posthog** (mem0ai telemetry)

- **torch** (multi-threaded operations)

**Solution:**
The test suite is configured for single-threaded mode in `tests/conftest.py`:

```python
# Already configured - no action needed
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
```

If you still see segfaults:

1. **Check test markers**: Ensure async tests use `@pytest.mark.asyncio`
2. **Verify pytest.ini**: Must have `asyncio_mode = auto`
3. **Check manual threading**: Don't create threads manually in tests
4. **Update conftest.py**: Ensure using latest version with background thread cleanup

**Prevention:**

- Always run tests with: `JAX_PLATFORM_NAME=cpu uv run pytest`

- Don't override threading environment variables

- Use smaller models in tests (colsmol-500m vs colpali-v1.2)

---

### DSPy Training Data Errors

**Symptoms:**
```text
AttributeError: 'Example' object has no attribute 'primary_intent'
```

**Cause:**
Training examples missing required output fields that metrics try to access.

**Solution:**
Add all required output fields to your DSPy Examples:

```python
# ❌ Bad: Missing output fields
example = dspy.Example(
    query="test query",
).with_inputs("query")

# ✅ Good: All output fields present
example = dspy.Example(
    query="test query",
    primary_intent="search",
    complexity_level="simple",
    needs_video_search="true",
    needs_text_search="false",
    multimodal_query="false",
    temporal_pattern="none",
).with_inputs("query")
```

**Required Fields by Module:**

- **Query Analysis**: primary_intent, complexity_level, needs_video_search, needs_text_search, multimodal_query, temporal_pattern

- **Agent Routing**: recommended_workflow, primary_agent, routing_confidence

**Prevention:**

- Validate training data before optimization (see docs/modules/optimization.md)

- Use example templates from `libs/agents/cogniverse_agents/optimizer/dspy_agent_optimizer.py:327-385`

- Run unit tests for training data loading

---

### Model Loading Hangs or Crashes

**Symptoms:**
```text
Fetching 5 files: 100%|██████████| 5/5 [00:00<00:00, 80000.00it/s]
[Test hangs or segfaults]
```

**Cause:**
Large models (1B+ parameters) can cause threading issues or memory exhaustion in test environment.

**Solution:**
Use smaller, stable models for testing:

```python
# ❌ Bad: Large model
model_name = "vidore/colpali-v1.2"  # 1.2B params, unstable in tests

# ✅ Good: Smaller model
model_name = "vidore/colsmol-500m"  # 500M params, stable
```

**Default Models:**

- ColPali: `vidore/colsmol-500m` (recommended)

- VideoPrism: `google/videoprism-base`

- ColQwen: `vidore/colqwen-omni-v0.1`

**Prevention:**

- Check ingestion pipeline configuration for default models

- Update documentation when changing models

- Run ingestion tests before committing model changes

---

## Import Errors

### Module Import Timing Issues

**Symptoms:**
```text
KeyError: Module not found in sys.modules
ImportError: Cannot import module at function call time
```

**Cause:**
Function tries to access `sys.modules` or import modules after the module has already started loading, or imports modules in the wrong order.

**Solution:**
Import system modules (`sys`, `os`, `logging`) at module level, not inside functions:

```python
# ✅ Correct: Import at module level
import sys
import os

def function_using_modules():
    # Now can safely use sys, os, etc.
    sys.modules["some.module"] = ...
```

**Affected Files:**

- Configuration and memory management modules in core package

- Note: With layered architecture, ensure imports from correct layers (foundation, core, implementation, application)

**Prevention:**

- Import system modules (`sys`, `os`, `logging`) at module level

- Only use function-level imports for optional dependencies

- Run test collection before committing: `pytest --collect-only`

---

### Missing Dependencies

**Symptoms:**
```text
ImportError: No module named 'colpali_engine'
ModuleNotFoundError: No module named 'mem0'
```

**Solution:**
```bash
# Sync all dependencies
uv sync

# Dependencies are managed in pyproject.toml
# All required packages will be installed via uv sync
```

**Common Missing Dependencies:**

- `colpali-engine`: ColPali/ColQwen models

- `mem0ai`: Memory management (includes posthog)

- `gliner`: Relationship extraction

- `phoenix-otel`: Telemetry

**Prevention:**

- Always run `uv sync` after pulling changes

- Check `pyproject.toml` for required dependencies

- Use `uv run` instead of direct python execution

---

## Model Loading Issues

### First-Time Model Download

**Symptoms:**
```text
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]
[Very slow or timeout]
```

**Cause:**
First-time model downloads from HuggingFace can be large (several GB).

**Solution:**

1. **Be patient**: Initial download takes time

2. **Check disk space**: Models cache in `~/.cache/huggingface/`

3. **Use smaller models**: colsmol-500m vs colpali-v1.2

**Model Sizes:**

- `vidore/colsmol-500m`: ~2GB

- `vidore/colpali-v1.2`: ~5GB

- `google/videoprism-base`: ~3GB

**Prevention:**

- Pre-download models before running tests

- Use Docker with pre-cached models

- Set HF_HOME for custom cache location

---

### Model Cache Corruption

**Symptoms:**
```text
RuntimeError: Error loading model
OSError: Unable to load weights
```

**Solution:**
Clear the HuggingFace cache:

```bash
# Remove corrupted cache
rm -rf ~/.cache/huggingface/hub/models--vidore--colsmol-500m

# Re-run ingestion or tests to re-download
JAX_PLATFORM_NAME=cpu uv run pytest
```

**Prevention:**

- Don't interrupt model downloads

- Ensure sufficient disk space

- Use stable internet connection

---

## Memory and Performance

### Out of Memory (OOM)

**Symptoms:**
```text
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solution:**

1. **Reduce batch size**:
```python
# In config
batch_size = 8  # Instead of 32
```

2. **Use CPU instead of GPU**:
```bash
JAX_PLATFORM_NAME=cpu uv run python script.py
```

3. **Enable gradient checkpointing** (for training):
```python
model.gradient_checkpointing_enable()
```

**Prevention:**

- Monitor memory usage: `nvidia-smi` (GPU) or `htop` (CPU)

- Use smaller models for development

- Batch processing for large datasets

---

### Slow Test Execution

**Symptoms:**

- Test suite takes >30 minutes

- Individual tests timeout

**Solution:**

1. **Run tests in parallel**:
```bash
uv run pytest -n auto
```

2. **Skip slow tests**:
```bash
uv run pytest -m "not slow"
```

3. **Use test markers**:
```bash
# Only fast unit tests
uv run pytest -m unit -m "not slow"
```

**Prevention:**

- Mark slow tests with `@pytest.mark.slow`

- Use mocks for external dependencies

- Cache model loading in test fixtures

---

## Vespa Issues

### Connection Refused

**Symptoms:**
```text
ConnectionError: [Errno 111] Connection refused
requests.exceptions.ConnectionError: http://localhost:8080
```

**Cause:**
Vespa container not running.

**Solution:**
```bash
# Check if Vespa is running
docker ps | grep vespa

# Start Vespa
docker run -d -p 8080:8080 vespaengine/vespa

# Verify
curl http://localhost:8080/state/v1/health
```

**Prevention:**

- Add Vespa to docker-compose

- Use health checks in tests

- Document Vespa requirement in README

---

### Schema Deployment Failures

**Symptoms:**
```text
VespaError: Schema deployment failed
400 Bad Request: Unknown field 'embeddings'
```

**Cause:**
Mismatch between code expectations and deployed schema.

**Solution:**
```bash
# Re-deploy schema
uv run python scripts/deploy_all_schemas.py

# Verify deployment
curl http://localhost:8080/application/v2/tenant/default/application/default
```

**Prevention:**

- Version your schemas

- Test schema changes before deploying

- Use schema validation in CI/CD

---

## Agent Communication

### Agent Input Validation Errors

**Symptoms:**
```text
ValidationError: Invalid input format
KeyError: 'query' in agent request
pydantic.ValidationError: Field required
```

**Cause:**
Request doesn't match expected `AgentInput` schema or missing required fields.

**Solution:**
Ensure requests follow the agent's input model:

```python
# ✅ Correct format - inherit from AgentInput
from cogniverse_core.agents.base import AgentInput

class SearchInput(AgentInput):
    query: str
    top_k: int = 10

# Create valid input
input = SearchInput(query="user query here", top_k=5)

# ❌ Missing required fields
input = SearchInput(top_k=5)  # ValidationError: query is required
```

**Prevention:**

- Use `AgentInput` subclasses for type-safe inputs

- Check agent interface documentation for required fields

- Add Pydantic validation in tests

---

### Health Check Failures

**Symptoms:**
```text
GET /health/ready returns {"status": "not_ready", "reason": "No backends registered"}
Health check shows 0 agents or backends
```

**Cause:**
Runtime initialization incomplete or dependencies unavailable.

**Solution:**

1. **Check service logs**:
```bash
docker logs <container-name>
# Or for local development
tail -f outputs/logs/*.log
```

2. **Verify dependencies**:

- Backend (Vespa/Elasticsearch) running and accessible
- Environment variables set (BACKEND_URL, BACKEND_PORT)
- Configuration manager initialized

3. **Check health endpoints**:
```bash
# Basic health check
curl http://localhost:8000/health

# Kubernetes readiness probe
curl http://localhost:8000/health/ready

# Kubernetes liveness probe
curl http://localhost:8000/health/live
```

4. **Restart service**:
```bash
docker restart <container-name>
```

**Prevention:**

- Use /health/ready for readiness probes in Kubernetes

- Use /health/live for liveness probes

- Monitor backend and agent registry status

---

## Quick Reference

### Essential Commands

```bash
# Run all tests
JAX_PLATFORM_NAME=cpu uv run pytest

# Run with debugging
JAX_PLATFORM_NAME=cpu uv run pytest -xvs --pdb

# Check test collection
pytest --collect-only

# Clear caches
rm -rf ~/.cache/huggingface/
rm -rf .pytest_cache/

# Restart services
docker-compose down && docker-compose up -d
```

### Log Locations

```text
Test logs:       outputs/logs/*.log
Agent logs:      docker logs <container>
Vespa logs:      docker logs vespa
Phoenix traces:  http://localhost:6006
```

### Support

- **Documentation**: docs/
- **Issues**: GitHub Issues
- **Tests**: tests/README.md
- **API Reference**: docs/modules/

---
