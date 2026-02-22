# Testing Guide
Comprehensive guide to testing practices in Cogniverse.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Fixtures and Mocking](#fixtures-and-mocking)
6. [Integration Tests](#integration-tests)
7. [Test Coverage](#test-coverage)
8. [CI/CD Testing](#cicd-testing)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Testing Philosophy

1. **Fix implementation to satisfy tests** - Never weaken tests
2. **100% pass rate required** before any commit
3. **No shortcuts** - No mocking failures, no disabling tests
4. **Test behavior, not implementation** - Tests should survive refactoring

### Test Stack

- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking support

---

## Test Organization

### Directory Structure

```text
tests/
├── agents/
│   ├── unit/
│   │   ├── test_routing_agent.py
│   │   ├── test_search_agent.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_workflow_checkpointing_integration.py
│   │   └── ...
│   └── e2e/
├── system/
│   ├── test_real_system_integration.py
│   ├── test_ensemble_search_e2e.py
│   └── conftest.py
├── ingestion/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── evaluation/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── routing/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── telemetry/
│   ├── unit/
│   └── integration/
├── finetuning/
│   ├── integration/
│   └── conftest.py
├── memory/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── backends/
│   ├── unit/
│   └── integration/
├── events/
│   ├── unit/
│   └── integration/
├── admin/
│   └── unit/
├── common/
│   ├── unit/
│   └── integration/
├── dashboard/
├── ui/
│   └── integration/
├── synthetic/
│   └── integration/
├── utils/
└── conftest.py                      # Shared fixtures
```

### Naming Conventions

- **Files**: `test_<module_name>.py`
- **Classes**: `Test<ClassName>`
- **Functions**: `test_<behavior_description>`

```python
# test_search_agent.py

class TestSearchAgent:
    """Tests for SearchAgent."""

    def test_search_by_text_returns_results(self):
        """Test that search_by_text returns results for valid query."""
        ...

    def test_search_with_invalid_query(self):
        """Test search with invalid query parameters."""
        ...

    @pytest.mark.asyncio
    async def test_handles_backend_timeout(self):
        """Test graceful handling of backend timeout."""
        ...
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# Run specific package tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ -v

# Run specific test file
uv run pytest tests/agents/unit/test_search_agent.py -v

# Run specific test
uv run pytest tests/agents/unit/test_search_agent.py::TestSearchAgent::test_search_by_text -v
```

### With Timeout

```bash
# 30-minute timeout for long tests (using system timeout command)
JAX_PLATFORM_NAME=cpu timeout 1800 uv run pytest tests/ -v

# For per-test timeout, install pytest-timeout first:
# uv pip install pytest-timeout
# Then use:
# uv run pytest tests/ -v --timeout=300
```

### Parallel Execution

Note: pytest-xdist is not currently installed. For parallel execution, install it first:

```bash
# Install pytest-xdist
uv pip install pytest-xdist

# Run with 4 workers
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v -n 4

# Auto-detect workers
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v -n auto
```

### With Output

```bash
# Show stdout/stderr
uv run pytest tests/ -v -s

# Log to file
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v 2>&1 | tee /tmp/test_output.log
```

### Finding Affected Tests

Before committing, find all tests affected by your changes:

```bash
# Find tests for a module
grep -r "def test_" --include="*.py" tests/ | grep -i "search_agent"

# Find all tests in a directory
find tests/agents -name "test_*.py" -exec grep -l "def test_" {} \;
```

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from pathlib import Path
from cogniverse_agents.search_agent import SearchAgent, SearchInput, SearchOutput, SearchAgentDeps
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

class TestSearchAgent:
    """Tests for SearchAgent."""

    @pytest.fixture
    def config_manager(self):
        """Create config manager for agent."""
        return create_default_config_manager()

    @pytest.fixture
    def schema_loader(self):
        """Create schema loader for agent."""
        # FilesystemSchemaLoader requires path to schema directory
        # Use relative path from project root (tests run from project root)
        from pathlib import Path
        schema_path = Path("configs/schemas")
        return FilesystemSchemaLoader(base_path=schema_path)

    @pytest.fixture
    def agent(self, config_manager, schema_loader):
        """Create test agent instance."""
        # Both config_manager and schema_loader are required for SearchAgent
        deps = SearchAgentDeps(
            tenant_id="test-tenant",
            backend_url="http://localhost",
            backend_port=8080
        )
        return SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8002  # A2A server port (optional, defaults to 8002)
        )

    @pytest.fixture
    def valid_input(self):
        """Create valid test input."""
        return SearchInput(
            query="machine learning tutorial",
            modality="video",
            top_k=10
        )

    def test_search_by_text_returns_results(self, agent, valid_input):
        """Test that search_by_text returns results."""
        results = agent.search_by_text(
            query=valid_input.query,
            modality=valid_input.modality,
            top_k=valid_input.top_k
        )

        assert isinstance(results, list)
        # Results is a list of dicts, each with id, score, video_id, frame_id, etc.
        if len(results) > 0:
            assert "video_id" in results[0] or "id" in results[0]
```

### Async Tests

Use `@pytest.mark.asyncio` for async tests. Note that VideoSearchAgent.search() is synchronous:

```python
import pytest
from concurrent.futures import ThreadPoolExecutor
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.config_store import ConfigStore
from unittest.mock import MagicMock

class TestVideoSearchAgent:

    @pytest.fixture
    def mock_config_store(self):
        """Create mock ConfigStore."""
        store = MagicMock(spec=ConfigStore)
        store.get_config.return_value = None
        return store

    @pytest.fixture
    def config_manager(self, mock_config_store):
        """Create ConfigManager with mock store."""
        return ConfigManager(store=mock_config_store)

    @pytest.fixture
    def schema_loader(self):
        """Create schema loader for agent."""
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from pathlib import Path
        return FilesystemSchemaLoader(Path("configs/schemas"))

    @pytest.fixture
    def agent(self, config_manager, schema_loader):
        """Create VideoSearchAgent with config manager and schema loader."""
        return VideoSearchAgent(
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

    def test_search(self, agent):
        """Test search operation (synchronous)."""
        result = agent.search("test query", profile="video_colpali_smol500_mv_frame", tenant_id="test", top_k=5)
        assert result is not None

    def test_concurrent_searches(self, agent):
        """Test concurrent search operations using threads."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(agent.search, "query1", "video_colpali_smol500_mv_frame", "test", 5),
                executor.submit(agent.search, "query2", "video_colpali_smol500_mv_frame", "test", 5),
                executor.submit(agent.search, "query3", "video_colpali_smol500_mv_frame", "test", 5),
            ]
            results = [f.result() for f in futures]

        assert len(results) == 3
```

### Parametrized Tests

```python
import pytest

class TestValidation:

    @pytest.mark.parametrize("query,expected_validation", [
        ("", "empty"),
        ("   ", "empty"),
        ("a" * 10001, "too long"),
    ])
    def test_invalid_queries(self, agent, query, expected_validation):
        """Test validation of invalid queries."""
        result = agent.search_by_text(query=query, modality="video", top_k=10)
        # SearchAgent returns a list (may be empty for invalid queries)
        assert isinstance(result, list)
```

### Testing Exceptions

```python
import pytest
from cogniverse_foundation.config.manager import ConfigManager

class TestConfigManager:

    def test_missing_tenant_raises_error(self, config_manager):
        """Test that missing tenant raises ValueError or KeyError."""
        with pytest.raises((ValueError, KeyError)) as exc_info:
            config_manager.get_system_config(tenant_id="nonexistent")

        # Error message should indicate the missing tenant
        assert "nonexistent" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()
```

---

## Fixtures and Mocking

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest

@pytest.fixture
def config_manager(backend_config_env):
    """Create ConfigManager with backend store for testing.
    Requires backend_config_env fixture to set environment variables.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager
    return create_default_config_manager()

@pytest.fixture
def config_manager_memory():
    """Create ConfigManager with in-memory store for unit testing.
    Does not require any backend infrastructure (Vespa, etc.).
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)

@pytest.fixture
def workflow_store(backend_config_env):
    """Create VespaWorkflowStore for testing."""
    from cogniverse_vespa.workflow.workflow_store import VespaWorkflowStore
    store = VespaWorkflowStore(
        vespa_url=os.environ.get("BACKEND_URL", "http://localhost"),
        vespa_port=int(os.environ.get("BACKEND_PORT", "8080")),
    )
    store.initialize()
    return store
```

### Mocking External Services

```python
from unittest.mock import AsyncMock, MagicMock, patch

class TestSearchAgentWithMocks:

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.search = AsyncMock(return_value=[
            {"id": "1", "score": 0.9, "title": "Result 1"},
            {"id": "2", "score": 0.8, "title": "Result 2"},
        ])
        return backend

    def test_search_calls_backend(self, mock_backend, config_manager, schema_loader):
        """Test that search calls backend with correct params."""
        deps = SearchAgentDeps(
            tenant_id="test-tenant",
            backend_url="http://localhost",
            backend_port=8080
        )
        agent = SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8002
        )

        result = agent.search_by_text(query="test", modality="video", top_k=5)

        # Verify backend was called (if using mock backend in deps)
        # mock_backend.search.assert_called()

    def test_handles_backend_error(self, mock_backend, config_manager, schema_loader):
        """Test graceful handling of backend error."""
        mock_backend.search.side_effect = ConnectionError("Backend down")

        deps = SearchAgentDeps(
            tenant_id="test-tenant",
            backend_url="http://localhost",
            backend_port=8080
        )
        agent = SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8002
        )

        result = agent.search_by_text(query="test", modality="video", top_k=10)

        # Result is a list (may be empty or contain error info depending on implementation)
        assert isinstance(result, list)
```

### Patching

```python
from unittest.mock import patch, MagicMock

class TestWithPatching:

    @patch("cogniverse_agents.search_agent.get_backend_registry")
    def test_with_patched_backend(self, mock_registry, config_manager, schema_loader):
        """Test with patched backend registry."""
        mock_backend = MagicMock()
        mock_registry.return_value.get_backend.return_value = mock_backend

        deps = SearchAgentDeps(
            tenant_id="test-tenant",
            backend_url="http://localhost",
            backend_port=8080
        )
        agent = SearchAgent(
            deps=deps,
            schema_loader=schema_loader,
            config_manager=config_manager
        )

        # Verify backend registry was accessed
        assert agent is not None
```

---

## Integration Tests

### With Real Services

```python
import pytest

@pytest.mark.integration
class TestVespaIntegration:
    """Integration tests requiring Vespa."""

    @pytest.fixture
    def vespa_client(self, config_manager):
        """Create real Vespa client."""
        import os
        from cogniverse_vespa.vespa_search_client import VespaVideoSearchClient
        vespa_url = os.getenv("BACKEND_URL", "http://localhost")
        vespa_port = int(os.getenv("BACKEND_PORT", "8080"))
        return VespaVideoSearchClient(
            vespa_url=vespa_url,
            vespa_port=vespa_port,
            tenant_id="test-tenant",
            config_manager=config_manager
        )

    def test_real_search(self, vespa_client):
        """Test search against real Vespa."""
        # VespaVideoSearchClient.search() accepts query_params (str or dict)
        results = vespa_client.search(
            query_params="test"  # Can also pass dict with query, ranking, top_k
        )
        assert isinstance(results, list)
```

### Skipping Without Services

```python
import pytest
import os

requires_vespa = pytest.mark.skipif(
    os.getenv("BACKEND_URL") is None,
    reason="BACKEND_URL not set"
)

@requires_vespa
class TestVespaRequired:
    """Tests that require Vespa."""

    def test_vespa_operation(self):
        ...
```

---

## Test Coverage

### Running with Coverage

```bash
# Basic coverage
uv run pytest tests/ --cov=cogniverse_core --cov-report=term

# HTML report
uv run pytest tests/ --cov=cogniverse_core --cov-report=html

# Multiple packages
uv run pytest tests/ \
    --cov=cogniverse_core \
    --cov=cogniverse_agents \
    --cov=cogniverse_foundation \
    --cov-report=html
```

### Coverage Configuration

**pytest.ini**:
```ini
[pytest]
# Coverage configuration is handled per-module in pytest.ini
# Not configured globally in pyproject.toml

# Run tests with coverage:
# uv run pytest tests/module/ --cov=libs/module/cogniverse_module
```

Note: Coverage settings are not in pyproject.toml - they're specified per command.

### Coverage Targets

| Package | Target |
|---------|--------|
| cogniverse-core | 80% |
| cogniverse-agents | 75% |
| cogniverse-foundation | 80% |
| cogniverse-runtime | 70% |

---

## CI/CD Testing

### GitHub Actions Workflows

The project has **12 GitHub workflow files** organized by module:

| Workflow | Module | Tests | Docker Services |
|----------|--------|-------|-----------------|
| `agents-tests.yml` | cogniverse-agents | unit + integration | None (mocked) |
| `core-tests.yml` | cogniverse-core | unit + integration | Vespa |
| `dashboard-tests.yml` | cogniverse-dashboard | unit + integration | None (TestClient) |
| `evaluation-tests.yml` | cogniverse-evaluation | unit + integration | Phoenix |
| `finetuning-tests.yml` | cogniverse-finetuning | unit + integration | Vespa |
| `ingestion-tests.yml` | cogniverse-runtime (ingestion) | unit + integration | Vespa |
| `routing-tests.yml` | cogniverse-agents (routing) | unit + integration | Vespa |
| `synthetic-tests.yml` | cogniverse-synthetic | unit + integration | Phoenix |
| `telemetry-tests.yml` | cogniverse-telemetry-phoenix | unit + integration | Phoenix |
| `vespa-tests.yml` | cogniverse-vespa | unit + integration | Vespa |
| `docs.yml` | Documentation | build check | None |
| `publish-packages.yml` | Package publishing | N/A | None |

### Workflow Structure

Each test workflow typically has these jobs:

1. **unit-tests** - Fast unit tests with mocked dependencies (often with ci_fast marker)
2. **integration-tests** - Integration tests (with ci_fast subset for quick feedback)
3. **lint** - Code linting with ruff/black
4. **test-cli** or **test-imports** - Verify package imports work
5. **coverage-report** - Combined coverage (if applicable)

Note: Workflows don't have separate "fast-integration-tests" jobs. Instead, integration-tests jobs use `-m ci_fast` to run essential tests quickly.

### CI Fast Integration Tests

Workflows run integration tests with the `ci_fast` marker on every push to provide quick feedback:

```yaml
integration-tests:
  runs-on: ubuntu-latest
  timeout-minutes: 60
  needs: unit-tests

  steps:
  - uses: actions/checkout@v4

  - name: Set up Python 3.12
    uses: actions/setup-python@v4
    with:
      python-version: '3.12'

  - name: Install uv
    uses: astral-sh/setup-uv@v4
    with:
      version: "latest"

  - name: Install dependencies
    run: |
      uv sync --all-packages --all-extras
      uv pip install pytest-cov

  - name: Free up disk space for Vespa (if needed)
    run: |
      # Vespa requires <75% disk usage
      sudo rm -rf /usr/share/dotnet
      sudo rm -rf /usr/local/lib/android
      sudo rm -rf /opt/ghc
      sudo rm -rf /opt/hostedtoolcache/CodeQL
      sudo docker image prune -af

  - name: Run integration tests (CI Fast subset)
    run: |
      JAX_PLATFORM_NAME=cpu uv run python -m pytest \
        tests/module/integration/ \
        -m ci_fast \
        -v --tb=short \
        --cov=libs/module/cogniverse_module
```

### Disk Cleanup for Vespa

Vespa requires disk usage below 75%. GitHub runners have ~14GB disk, so cleanup is required:

```bash
# Remove ~30GB of unused packages
sudo rm -rf /usr/share/dotnet           # .NET SDK (~6GB)
sudo rm -rf /usr/local/lib/android      # Android SDK (~10GB)
sudo rm -rf /opt/ghc                    # Haskell compiler (~2GB)
sudo rm -rf /opt/hostedtoolcache/CodeQL # CodeQL (~5GB)
sudo docker image prune -af             # Unused Docker images
```

### Docker Service Management

Tests manage their own Docker containers via fixtures:

**Vespa (VespaDockerManager):**
```python
# tests/utils/vespa_docker.py
from tests.utils.vespa_docker import VespaDockerManager

@pytest.fixture(scope="module")
def vespa_docker():
    """Start Vespa container for module tests."""
    manager = VespaDockerManager()
    container_info = manager.start_container(
        module_name="test_module",
        use_module_ports=True
    )
    yield container_info
    manager.stop_container(container_info)
```

**Phoenix:**
```python
@pytest.fixture(scope="module")
def phoenix_container():
    """Start Phoenix container for telemetry tests.
    Uses non-default ports to avoid conflicts with local Phoenix instances.
    """
    container_name = f"phoenix_test_{int(time.time() * 1000)}"
    subprocess.run([
        "docker", "run", "-d", "--name", container_name,
        "-p", "16006:6006", "-p", "14317:4317",
        "arizephoenix/phoenix:latest"
    ])
    # Wait for ready...
    yield container_name
    subprocess.run(["docker", "rm", "-f", container_name])
```

### Pre-commit Checklist

Before every commit:

```bash
# 1. Find affected tests
grep -r "def test_" --include="*.py" tests/ | grep -i "<module>"

# 2. Run tests
JAX_PLATFORM_NAME=cpu timeout 1800 uv run pytest tests/ -v 2>&1 | tee /tmp/test_output.log

# 3. Verify 100% pass
grep -E "passed|failed" /tmp/test_output.log

# 4. Lint
uv run make lint-all
```

---

## Troubleshooting

### JAX Platform Errors

```bash
# Always set JAX platform for CPU
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# Or in conftest.py
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
```

### Async Test Not Running

```python
# Ensure pytest-asyncio is installed and configured

# In pytest.ini (already configured):
# asyncio_mode = auto
# asyncio_default_fixture_loop_scope = function

# Use decorator for async tests
@pytest.mark.asyncio
async def test_async_function():
    ...
```

### Import Errors

```bash
# Ensure packages are installed
uv sync

# Check import path
uv run python -c "import cogniverse_core; print(cogniverse_core.__file__)"

# Run from project root
cd /path/to/cogniverse
uv run pytest tests/ -v
```

### Flaky Tests

For retries or timeout decorators, install the required packages first:

```bash
# Install pytest-timeout for timeout decorator
uv pip install pytest-timeout

# Install pytest-rerunfailures for flaky test retries
uv pip install pytest-rerunfailures
```

Then use:
```python
# Use retries for network-dependent tests (requires pytest-rerunfailures)
@pytest.mark.flaky(reruns=3)
def test_network_operation():
    ...

# Or add explicit timeout (requires pytest-timeout)
@pytest.mark.timeout(30)
def test_slow_operation():
    ...
```

### Test Isolation

```python
# Each test should be independent
# Use fixtures with appropriate scope

@pytest.fixture  # Default: function scope - fresh for each test
def config():
    return Config()

@pytest.fixture(scope="class")  # Shared within class
def expensive_resource():
    return create_expensive_resource()

@pytest.fixture(scope="session")  # Shared for entire session
def database():
    return setup_database()
```

---

## Summary

1. **Organize tests** by package (agents, system, ingestion, evaluation, etc.)
2. **Use fixtures** for common setup
3. **Run with JAX_PLATFORM_NAME=cpu** to avoid GPU issues
4. **100% pass rate required** before commit
5. **Find affected tests** with grep before committing
6. **Use mocks** for external services in unit tests
7. **Mark integration tests** that require real services
