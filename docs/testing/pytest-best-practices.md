# Pytest Best Practices

Guide for writing and running tests in the Cogniverse project.

---

## Async Testing Configuration

### Threading Issues with Async Tests

Async tests in pytest can encounter threading conflicts when loading ML models. The codebase uses several libraries that spawn background threads:

- **tqdm** (from transformers): Progress bars during model downloads
- **posthog** (from mem0ai): Telemetry/analytics background threads
- **torch**: Multi-threaded tensor operations

These background threads can cause **segmentation faults** during pytest cleanup when combined with async event loops.

### Solution: Single-Threaded Mode

The test suite is configured for single-threaded execution to avoid threading conflicts:

**File: `tests/conftest.py`**
```python
import os
import torch

# Configure torch and tokenizers to avoid threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Set torch to single-threaded mode
torch.set_num_threads(1)
```

**File: `pytest.ini`**
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
```

### Background Thread Cleanup

The conftest also includes automatic cleanup for background threads:

```python
def cleanup_background_threads():
    """
    Clean up background threads from tqdm and posthog.

    Gives threads time to finish gracefully before pytest cleanup.
    """
    max_wait = 2.0  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        background_threads = [
            t for t in threading.enumerate()
            if t != threading.current_thread()
            and t.daemon
            and any(name in t.name.lower()
                    for name in ['tqdm', 'posthog', 'monitor'])
        ]

        if not background_threads:
            break

        time.sleep(0.1)

    gc.collect()
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
JAX_PLATFORM_NAME=cpu uv run pytest

# Run specific test file with 30-minute timeout
JAX_PLATFORM_NAME=cpu timeout 1800 uv run pytest tests/agents/test_routing.py

# Run with verbose output
JAX_PLATFORM_NAME=cpu uv run pytest -xvs
```

### Test Markers

```bash
# Run only unit tests
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"

# Run tests requiring specific models
uv run pytest -m requires_colpali
uv run pytest -m requires_ollama
```

### Async Test Timeout

Async tests use custom timeouts defined in test files:

```python
@pytest.mark.timeout(TEST_CONFIG["test_timeout"])
async def test_real_query_analysis_with_local_llm(self):
    # Test code here
    pass
```

Default timeouts from `CLAUDE.md`:
- Individual test files: 30 minutes (`timeout 1800`)
- Full test suite: 120 minutes (`timeout 7200`)

---

## Common Issues

### Segmentation Faults in Async Tests

**Symptoms:**
```
Fatal Python error: Segmentation fault
Thread 0x000000033614f000 (most recent call first):
  File "/path/to/threading.py", line 359 in wait
```

**Cause:** Threading conflict between pytest async event loop and background threads from model loading.

**Solution:** Already handled by `tests/conftest.py` configuration. If you still see segfaults:

1. Check that tests use `@pytest.mark.asyncio` correctly
2. Verify `pytest.ini` has `asyncio_mode = auto`
3. Ensure no manual thread creation in test code

### Model Loading Errors

**Symptoms:**
```
Fetching 5 files: 100%
[Segfault or hang]
```

**Cause:** Large models (e.g., `vidore/colpali-v1.2`) can cause threading issues.

**Solution:** Use smaller models in tests:
- ✅ Use: `vidore/colsmol-500m` (stable, 500M parameters)
- ❌ Avoid: `vidore/colpali-v1.2` (1.2B parameters, less stable in tests)

### Import Timing Issues

**Symptoms:**
```
KeyError: 'src.common.vespa_memory_config'
```

**Cause:** Module imports happening before required dependencies are loaded.

**Solution:** Ensure module-level imports for system modules (`sys`, `os`) come before function-level imports.

**Example Fix:**
```python
# ❌ Bad: sys imported inside function
def _register_vespa_provider():
    import sys  # Too late!
    sys.modules["mem0.configs.vector_stores.vespa"] = ...

# ✅ Good: sys imported at module level
import sys

def _register_vespa_provider():
    sys.modules["mem0.configs.vector_stores.vespa"] = ...
```

---

## Test Isolation

### State Cleanup Between Tests

Tests use auto-fixtures to clean up state:

```python
@pytest.fixture(autouse=True, scope="function")
def cleanup_dspy_state():
    """Clean up DSPy state between tests"""
    yield
    try:
        import dspy
        if hasattr(dspy.settings, '_instance'):
            dspy.settings._instance = None
    except (ImportError, AttributeError, RuntimeError):
        pass

    cleanup_background_threads()
```

### Best Practices

1. **Always use fixtures** for shared state
2. **Clean up resources** in fixture teardown
3. **Don't rely on test execution order** - tests should be independent
4. **Use unique IDs** for multi-tenant tests (tenant_id, user_id)

---

## Performance

### Test Execution Time

Optimize test performance:

```bash
# Parallel execution (be careful with async tests)
uv run pytest -n auto

# Run fastest tests first
uv run pytest --durations=10

# Profile slow tests
uv run pytest --profile
```

### Model Caching

Models are cached to speed up tests:

```python
# Models cached in _model_cache
_model_cache = {}

def get_or_load_model(model_name, config, logger):
    cache_key = model_name
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Load and cache
    model, processor = loader.load_model()
    _model_cache[cache_key] = (model, processor)
    return model, processor
```

---

## Debugging Tests

### Debug Output

```bash
# Show print statements
uv run pytest -s

# Show full error traceback
uv run pytest --tb=long

# Drop into debugger on failure
uv run pytest --pdb
```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in tests
@pytest.fixture
def debug_logging():
    logging.getLogger().setLevel(logging.DEBUG)
```

---

## CI/CD Considerations

### GitHub Actions Configuration

```yaml
- name: Run Tests
  env:
    JAX_PLATFORM_NAME: cpu
    TOKENIZERS_PARALLELISM: false
    OMP_NUM_THREADS: 1
  run: |
    timeout 7200 uv run pytest --tb=short
```

### Docker Testing

```dockerfile
# Ensure single-threaded mode
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN pytest
```

---

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Torch Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
