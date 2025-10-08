# Troubleshooting Guide

Common issues and solutions for the Cogniverse multi-agent RAG system.

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
```
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
```python
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
- Use example templates from `src/app/agents/dspy_agent_optimizer.py:327-385`
- Run unit tests for training data loading

---

### Model Loading Hangs or Crashes

**Symptoms:**
```
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

**Default Models (as of 2025-10-08):**
- ColPali: `vidore/colsmol-500m` (recommended)
- VideoPrism: `google/videoprism-base`
- ColQwen: `vidore/colqwen-omni-v0.1`

**Prevention:**
- Check `src/app/ingestion/strategies.py` for default models
- Update documentation when changing models
- Run ingestion tests before committing model changes

---

## Import Errors

### Module Import Timing Issues

**Symptoms:**
```python
KeyError: 'src.common.vespa_memory_config'
```

**Cause:**
Function tries to access `sys.modules` before `sys` is imported.

**Example Problem:**
```python
def _register_vespa_provider():
    import sys  # ❌ Too late! Function called at module import time
    sys.modules["mem0.configs.vector_stores.vespa"] = ...
```

**Solution:**
Move `import sys` to module level:

```python
import sys  # ✅ Import at module level

def _register_vespa_provider():
    sys.modules["mem0.configs.vector_stores.vespa"] = ...
```

**Affected Files:**
- `src/common/mem0_memory_manager.py` (fixed in commit 28c8e45)

**Prevention:**
- Import system modules (`sys`, `os`, `logging`) at module level
- Only use function-level imports for optional dependencies
- Run test collection before committing: `pytest --collect-only`

---

### Missing Dependencies

**Symptoms:**
```python
ImportError: No module named 'colpali_engine'
ModuleNotFoundError: No module named 'mem0'
```

**Solution:**
```bash
# Sync all dependencies
uv sync

# For specific modules
pip install colpali-engine  # ColPali models
pip install mem0ai          # Memory management
pip install gliner          # Entity extraction
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
```
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
```
RuntimeError: Error loading model
OSError: Unable to load weights
```

**Solution:**
Clear the HuggingFace cache:

```bash
# Remove corrupted cache
rm -rf ~/.cache/huggingface/hub/models--vidore--colsmol-500m

# Re-run to re-download
JAX_PLATFORM_NAME=cpu uv run pytest tests/test_model_loading.py
```

**Prevention:**
- Don't interrupt model downloads
- Ensure sufficient disk space
- Use stable internet connection

---

## Memory and Performance

### Out of Memory (OOM)

**Symptoms:**
```
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
```
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
```
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

### A2A Protocol Errors

**Symptoms:**
```
A2AProtocolError: Invalid request format
KeyError: 'query' in agent request
```

**Cause:**
Request doesn't match expected A2A message format.

**Solution:**
Ensure requests follow A2A protocol:

```python
# ✅ Correct format
request = {
    "message_id": "unique-id",
    "query": "user query here",
    "context": {},
    "request_time": "2025-10-08T12:00:00Z"
}

# ❌ Missing required fields
request = {"query": "user query"}  # Missing message_id, request_time
```

**Prevention:**
- Use `A2ARequest` model for validation
- Check agent interface documentation
- Add schema validation in tests

---

### Health Check Failures

**Symptoms:**
```
HealthCheckError: Agent not ready
GET /health returned 503 Service Unavailable
```

**Cause:**
Agent initialization incomplete or dependencies unavailable.

**Solution:**

1. **Check agent logs**:
```bash
docker logs <agent-container>
```

2. **Verify dependencies**:
- Vespa running and accessible
- Required models loaded
- Environment variables set

3. **Restart agent**:
```bash
docker restart <agent-container>
```

**Prevention:**
- Implement proper health checks
- Add startup probes in Kubernetes
- Monitor agent metrics

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

```
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

**Last Updated:** 2025-10-08
