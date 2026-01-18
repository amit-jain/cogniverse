# Pytest Best Practices

**Last Updated:** 2025-01-18
**Architecture:** UV Workspace with 11-package layered architecture and multi-tenant support
**Purpose:** Guide for writing and running tests in the Cogniverse project

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

# Run only integration tests
uv run pytest -m integration

# Run fast CI tests (subset for quick feedback)
uv run pytest -m ci_fast

# Skip slow tests
uv run pytest -m "not slow"

# Run tests requiring specific models
uv run pytest -m requires_colpali
uv run pytest -m requires_ollama

# Skip tests requiring Ollama (for CI)
uv run pytest -m "not requires_ollama"
```

### Available Markers

Defined in `pytest.ini`:

```ini
[pytest]
markers =
    unit: Unit tests (no external dependencies)
    integration: Integration tests (require external services)
    ci_fast: Fast, essential tests for CI (subset of most important functionality)
    slow: Slow tests (skip in quick runs)
    requires_ollama: Tests requiring Ollama LLM
    requires_colpali: Tests requiring ColPali model
    requires_vespa: Tests requiring Vespa backend
    local_only: Tests that only run locally (not in CI)
```

### The `ci_fast` Marker

The `ci_fast` marker identifies tests suitable for the `fast-integration-tests` CI job:

```python
@pytest.mark.integration
@pytest.mark.ci_fast
async def test_tenant_schema_creation(vespa_docker):
    """Essential test for CI - verifies tenant schema lifecycle."""
    # This runs on every push in fast-integration-tests job
    ...
```

**Guidelines for `ci_fast` tests:**
- Essential functionality that must work
- Complete in under 2 minutes
- No external API calls (Ollama, OpenAI, etc.)
- Can use Docker containers (Vespa, Phoenix)

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
ModuleNotFoundError: No module named 'cogniverse_core'
```

**Cause:** UV workspace not synced or tests run outside virtual environment.

**Solution:** Always use `uv run pytest` which ensures workspace packages are available.

**Example Fix:**
```bash
# ❌ Bad: Direct pytest without uv
pytest tests/agents/

# ✅ Good: Use uv run to activate workspace
uv run pytest tests/agents/
```

**Package Import Patterns:**
```python
# ✅ Good: Absolute imports from workspace packages
from cogniverse_sdk.interfaces import AgentInterface
from cogniverse_foundation.telemetry import TelemetryManager
from cogniverse_core.config import SystemConfig
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_retrieval.vespa_backend import VespaBackend

# ❌ Bad: Old src-style imports (deprecated)
from src.agents.routing_agent import RoutingAgent  # ❌ Will fail
```

---

## Package Testing

### Testing Package Imports

**Verify Package Structure:**
```python
# tests/test_imports.py
import pytest

def test_sdk_package_imports():
    """Verify cogniverse_sdk package imports work"""
    from cogniverse_sdk.interfaces import AgentInterface, SearchBackend
    from cogniverse_sdk.types import QueryResult, SearchResponse

    assert AgentInterface is not None
    assert SearchBackend is not None

def test_foundation_package_imports():
    """Verify cogniverse_foundation package imports work"""
    from cogniverse_foundation.telemetry import TelemetryManager
    from cogniverse_foundation.logging import get_logger

    assert TelemetryManager is not None
    assert get_logger is not None

def test_core_package_imports():
    """Verify cogniverse_core package imports work"""
    from cogniverse_core.config import SystemConfig
    from cogniverse_core.orchestration import Orchestrator

    assert SystemConfig is not None
    assert Orchestrator is not None

def test_agents_package_imports():
    """Verify cogniverse_agents package imports work"""
    from cogniverse_agents.routing_agent import RoutingAgent
    from cogniverse_agents.video_search_agent import VideoSearchAgent

    assert RoutingAgent is not None
    assert VideoSearchAgent is not None

def test_retrieval_package_imports():
    """Verify cogniverse_retrieval package imports work"""
    from cogniverse_retrieval.vespa_backend import VespaBackend
    from cogniverse_retrieval.vespa_schema_manager import VespaSchemaManager

    assert VespaBackend is not None
    assert VespaSchemaManager is not None

def test_processing_package_imports():
    """Verify cogniverse_processing package imports work"""
    from cogniverse_processing.video_pipeline import VideoIngestionPipeline
    from cogniverse_processing.frame_extractor import FrameExtractor

    assert VideoIngestionPipeline is not None
    assert FrameExtractor is not None

def test_evaluation_package_imports():
    """Verify cogniverse_evaluation package imports work"""
    from cogniverse_evaluation.experiment_tracker import ExperimentTracker
    from cogniverse_evaluation.metrics import calculate_metrics

    assert ExperimentTracker is not None
    assert calculate_metrics is not None
```

### Package Dependency Testing

**Test Cross-Package Dependencies:**
```python
# tests/test_package_dependencies.py
import pytest
from cogniverse_sdk.interfaces import AgentInterface
from cogniverse_foundation.telemetry import TelemetryManager
from cogniverse_core.config import SystemConfig
from cogniverse_agents.routing_agent import RoutingAgent

def test_agents_depends_on_foundation_and_core():
    """Verify agents package can use foundation and core packages"""
    config = SystemConfig(tenant_id="test")
    telemetry = TelemetryManager(config)

    # RoutingAgent from agents package should accept SystemConfig from core
    agent = RoutingAgent(config, telemetry_manager=telemetry)

    assert agent.config is config
    assert agent.tenant_id == "test"

def test_services_depends_on_all():
    """Verify services package can use all dependencies"""
    from cogniverse_services.api import create_app
    from cogniverse_core.config import SystemConfig
    from cogniverse_agents.routing_agent import RoutingAgent
    from cogniverse_retrieval.vespa_backend import VespaBackend

    # Services should be able to import and use all packages
    config = SystemConfig(tenant_id="test")
    agent = RoutingAgent(config)
    backend = VespaBackend(config)

    assert agent is not None
    assert backend is not None

def test_layered_architecture_dependencies():
    """Verify proper layering - lower layers don't import higher layers"""
    # SDK should not import from other packages
    # Foundation can import SDK
    # Core can import SDK and Foundation
    # Agents can import SDK, Foundation, and Core
    # Implementation can import SDK, Foundation, and Core
    # Services can import all
    pass
```

---

## Multi-Tenant Testing

### Tenant Isolation Tests

**Test Tenant-Specific Configuration:**
```python
# tests/test_tenant_isolation.py
import pytest
from cogniverse_core.config import SystemConfig
from cogniverse_agents.agents.video_search_agent import VideoSearchAgent

def test_tenant_config_isolation():
    """Verify each tenant gets isolated configuration"""
    config_a = SystemConfig(tenant_id="acme_corp")
    config_b = SystemConfig(tenant_id="globex_inc")

    assert config_a.tenant_id == "acme_corp"
    assert config_b.tenant_id == "globex_inc"
    assert config_a.tenant_id != config_b.tenant_id

def test_tenant_schema_naming():
    """Verify tenant schemas use correct naming convention"""
    config = SystemConfig(tenant_id="acme_corp")
    agent = VideoSearchAgent(
        config,
        profile="video_colpali_smol500_mv_frame"
    )

    # Agent should target tenant-specific schema
    expected_schema = "video_colpali_smol500_mv_frame_acme_corp"
    assert agent.schema_name == expected_schema

def test_tenant_phoenix_project_isolation():
    """Verify Phoenix projects are tenant-specific"""
    from cogniverse_core.telemetry import TelemetryManager

    config_a = SystemConfig(
        tenant_id="acme_corp",
        phoenix_enabled=True
    )
    config_b = SystemConfig(
        tenant_id="globex_inc",
        phoenix_enabled=True
    )

    telemetry_a = TelemetryManager(config_a)
    telemetry_b = TelemetryManager(config_b)

    # Each tenant should have isolated Phoenix project
    assert telemetry_a.project_name == "acme_corp_project"
    assert telemetry_b.project_name == "globex_inc_project"
    assert telemetry_a.project_name != telemetry_b.project_name
```

### Multi-Tenant Data Isolation

**Test Cross-Tenant Data Boundaries:**
```python
# tests/integration/test_multi_tenant_isolation.py
import pytest
from cogniverse_core.config import SystemConfig
from cogniverse_processing.video_pipeline import VideoIngestionPipeline
from cogniverse_retrieval.vespa_backend import VespaBackend

@pytest.mark.integration
async def test_tenant_data_isolation(sample_video):
    """Verify tenants cannot access each other's data"""
    # Ingest video for tenant A
    config_a = SystemConfig(tenant_id="acme_corp")
    pipeline_a = VideoIngestionPipeline(
        profile="video_colpali_smol500_mv_frame",
        tenant_id="acme_corp",
        backend="vespa"
    )
    result_a = await pipeline_a.process_video(sample_video)
    assert result_a["status"] == "success"

    # Search as tenant B (should get no results from tenant A)
    config_b = SystemConfig(tenant_id="globex_inc")
    backend_b = VespaBackend(config_b)

    results = backend_b.search(
        query="test",
        schema_name="video_colpali_smol500_mv_frame_globex_inc"
    )

    # Tenant B should not see tenant A's documents
    assert len(results) == 0

@pytest.mark.integration
def test_tenant_memory_isolation():
    """Verify tenant memories are isolated"""
    from cogniverse_foundation.memory.mem0_memory_manager import Mem0MemoryManager

    config_a = SystemConfig(tenant_id="acme_corp")
    config_b = SystemConfig(tenant_id="globex_inc")

    memory_a = Mem0MemoryManager(config_a)
    memory_b = Mem0MemoryManager(config_b)

    # Add memory for tenant A
    memory_a.add(
        messages=[{"role": "user", "content": "Secret message"}],
        user_id="acme_corp_user1"
    )

    # Search as tenant B (should not find tenant A's memory)
    results_b = memory_b.search(
        query="Secret message",
        user_id="globex_inc_user1"
    )

    assert len(results_b["results"]) == 0
```

### Tenant-Aware Fixtures

**Create Reusable Tenant Fixtures:**
```python
# tests/conftest.py
import pytest
from cogniverse_core.config import SystemConfig
from cogniverse_agents.routing_agent import RoutingAgent

@pytest.fixture
def tenant_a_config():
    """Configuration for tenant A (acme_corp)"""
    return SystemConfig(
        tenant_id="acme_corp",
        vespa_url="http://localhost:8080",
        vespa_config_port=19071,
        phoenix_enabled=False  # Disable for unit tests
    )

@pytest.fixture
def tenant_b_config():
    """Configuration for tenant B (globex_inc)"""
    return SystemConfig(
        tenant_id="globex_inc",
        vespa_url="http://localhost:8080",
        vespa_config_port=19071,
        phoenix_enabled=False
    )

@pytest.fixture
def multi_tenant_configs():
    """Multiple tenant configurations for cross-tenant tests"""
    return {
        "acme_corp": SystemConfig(tenant_id="acme_corp"),
        "globex_inc": SystemConfig(tenant_id="globex_inc"),
        "default": SystemConfig(tenant_id="default")
    }

@pytest.fixture
def tenant_agent(tenant_a_config):
    """Create routing agent for tenant A"""
    return RoutingAgent(tenant_a_config)

# Use in tests:
def test_with_tenant_fixtures(tenant_a_config, tenant_agent):
    assert tenant_agent.tenant_id == "acme_corp"
    assert tenant_agent.config == tenant_a_config
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

## Docker Fixtures for Integration Tests

Integration tests use self-managed Docker containers via pytest fixtures.

### VespaDockerManager

Located in `tests/utils/vespa_docker.py`, manages Vespa containers:

```python
from tests.utils.vespa_docker import VespaDockerManager

@pytest.fixture(scope="module")
def vespa_docker():
    """
    Module-scoped Vespa container.

    Starts Vespa, waits for readiness, yields manager, then cleans up.
    Uses unique ports per module to avoid conflicts.
    """
    manager = VespaDockerManager(
        container_name=f"vespa_test_{int(time.time())}",
        vespa_port=8080,      # Query/feed port
        config_port=19071     # Config server port
    )
    manager.start()
    manager.wait_for_ready(timeout=120)
    yield manager
    manager.stop()

# Usage in tests
@pytest.mark.integration
async def test_vespa_operations(vespa_docker):
    """Test with real Vespa container."""
    client = vespa_docker.get_client()
    # ... test operations
```

**Key Features:**
- Generates unique container names to avoid conflicts
- Waits for both config server and application readiness
- Proper cleanup on test completion or failure
- Module-scoped to share container across tests in a file

### Phoenix Docker Fixtures

For telemetry and evaluation tests:

```python
import subprocess
import time

@pytest.fixture(scope="module")
def phoenix_container():
    """Start Phoenix container for telemetry tests."""
    container_name = f"phoenix_test_{int(time.time())}"

    # Start container
    subprocess.run([
        "docker", "run", "-d", "--name", container_name,
        "-p", "6006:6006",   # UI port
        "-p", "4317:4317",   # OTLP gRPC port
        "arizephoenix/phoenix:latest"
    ], check=True)

    # Wait for Phoenix to be ready
    for _ in range(30):
        try:
            response = requests.get("http://localhost:6006")
            if response.ok:
                break
        except requests.ConnectionError:
            time.sleep(2)

    yield {
        "container_name": container_name,
        "ui_url": "http://localhost:6006",
        "otlp_endpoint": "http://localhost:4317"
    }

    # Cleanup
    subprocess.run(["docker", "rm", "-f", container_name])

# Usage
@pytest.mark.integration
def test_telemetry_collection(phoenix_container):
    """Test with real Phoenix container."""
    provider = PhoenixProvider(endpoint=phoenix_container["otlp_endpoint"])
    # ... test operations
```

### Port Management

To avoid port conflicts when running tests in parallel:

```python
def get_unique_ports(base_port: int = 8080) -> tuple[int, int]:
    """Generate unique ports based on process ID and time."""
    import os
    offset = (os.getpid() % 100) * 10 + int(time.time()) % 10
    return base_port + offset, 19071 + offset
```

### CI Disk Space Requirements

Vespa requires disk usage below 75%. In GitHub Actions:

```yaml
- name: Free up disk space for Vespa
  run: |
    # Remove ~30GB of unused packages
    sudo rm -rf /usr/share/dotnet           # .NET SDK
    sudo rm -rf /usr/local/lib/android      # Android SDK
    sudo rm -rf /opt/ghc                    # Haskell
    sudo rm -rf /opt/hostedtoolcache/CodeQL # CodeQL
    sudo docker image prune -af
    df -h  # Verify disk usage

- name: Pre-pull Vespa Docker image
  run: docker pull vespaengine/vespa:latest
```

---

---

## Summary

This guide covers comprehensive testing for Cogniverse multi-agent system:

1. **Async Testing**: Thread-safe configuration for async tests with ML models
2. **SDK Package Testing**: Verify cross-package imports and dependencies
3. **Multi-Tenant Testing**: Ensure tenant isolation at all layers
4. **Test Execution**: Use `uv run pytest` for workspace packages
5. **Performance**: Model caching and parallel execution strategies
6. **CI/CD**: Configuration for automated testing

**Key Testing Principles:**
- Always use `uv run pytest` to activate workspace
- Test tenant isolation at schema, project, and memory levels
- Use fixtures for reusable tenant configurations
- Verify SDK package imports work correctly
- Maintain test independence (no shared state)
- Clean up background threads to avoid segfaults

**Test Organization by Package:**
```
tests/
├── sdk/                 # cogniverse_sdk tests
├── foundation/          # cogniverse_foundation tests
│   ├── telemetry/
│   └── memory/
├── core/                # cogniverse_core tests
│   └── config/
├── agents/              # cogniverse_agents tests
│   ├── unit/
│   └── integration/
├── retrieval/           # cogniverse_retrieval tests
│   └── vespa/
├── processing/          # cogniverse_processing tests
│   └── video/
├── synthetic/           # cogniverse_synthetic tests
├── vlm/                 # cogniverse_vlm tests
├── services/            # cogniverse_services tests
└── evaluation/          # cogniverse_evaluation tests
```

**Related Documentation:**
- [11-Package Architecture](../architecture/overview.md)
- [Multi-Tenant Architecture](../architecture/multi-tenant.md)
- [Package Development Guide](../development/package-development.md)

---

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Torch Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [UV Package Manager](https://github.com/astral-sh/uv)
