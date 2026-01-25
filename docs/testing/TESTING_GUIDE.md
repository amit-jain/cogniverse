# Testing Guide

**Last Updated:** 2026-01-25

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
- **pytest-xdist**: Parallel execution

---

## Test Organization

### Directory Structure

```
tests/
├── agents/
│   ├── unit/
│   │   ├── test_routing_agent.py
│   │   ├── test_search_agent.py
│   │   └── test_workflow_checkpointing_integration.py
│   └── integration/
│       └── test_multi_agent_orchestrator.py
├── foundation/
│   ├── unit/
│   │   ├── test_config_manager.py
│   │   └── test_telemetry_manager.py
│   └── integration/
│       └── test_config_persistence.py
├── runtime/
│   ├── unit/
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_search_api.py
├── evaluation/
│   └── unit/
│       └── test_metrics.py
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

    def test_process_returns_results_for_valid_query(self):
        """Test that valid query returns search results."""
        ...

    def test_empty_query_returns_error_output(self):
        """Test that empty query returns error in output."""
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
uv run pytest tests/agents/unit/test_search_agent.py::TestSearchAgent::test_process -v
```

### With Timeout

```bash
# 30-minute timeout for long tests
JAX_PLATFORM_NAME=cpu timeout 1800 uv run pytest tests/ -v

# Per-test timeout
uv run pytest tests/ -v --timeout=300
```

### Parallel Execution

```bash
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
from cogniverse_agents.search import SearchAgent, SearchInput, SearchDeps

class TestSearchAgent:
    """Tests for SearchAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent instance."""
        deps = SearchDeps()
        return SearchAgent(deps=deps)

    @pytest.fixture
    def valid_input(self):
        """Create valid test input."""
        return SearchInput(
            query="machine learning tutorial",
            profile="default",
            top_k=10
        )

    def test_agent_name_is_correct(self, agent):
        """Test agent has correct name."""
        assert agent.AGENT_NAME == "search_agent"

    @pytest.mark.asyncio
    async def test_process_returns_output(self, agent, valid_input):
        """Test that process returns SearchOutput."""
        result = await agent.process(valid_input)

        assert isinstance(result, SearchOutput)
        assert result.error is None
```

### Async Tests

Use `@pytest.mark.asyncio` for async tests:

```python
import pytest

class TestAsyncOperations:

    @pytest.mark.asyncio
    async def test_async_search(self, agent):
        """Test async search operation."""
        result = await agent.search("test query")
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, agent):
        """Test concurrent operations."""
        import asyncio

        results = await asyncio.gather(
            agent.search("query1"),
            agent.search("query2"),
            agent.search("query3")
        )

        assert len(results) == 3
```

### Parametrized Tests

```python
import pytest

class TestValidation:

    @pytest.mark.parametrize("query,expected_error", [
        ("", "Query cannot be empty"),
        ("   ", "Query cannot be empty"),
        ("a" * 10001, "Query too long"),
    ])
    def test_invalid_queries(self, agent, query, expected_error):
        """Test validation of invalid queries."""
        result = await agent.process(SearchInput(query=query))
        assert result.error == expected_error
```

### Testing Exceptions

```python
import pytest
from cogniverse_foundation.config.manager import ConfigManager

class TestConfigManager:

    def test_missing_tenant_raises_error(self, config_manager):
        """Test that missing tenant raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager.get_config(tenant_id="nonexistent")

        assert "not found" in str(exc_info.value)
```

---

## Fixtures and Mocking

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.sqlite.config_store import InMemoryConfigStore

@pytest.fixture
def config_manager():
    """Create test config manager with in-memory store."""
    store = InMemoryConfigStore()
    return ConfigManager(store=store)

@pytest.fixture
def tenant_id():
    """Standard test tenant ID."""
    return "test-tenant"

@pytest.fixture
def sample_query():
    """Sample search query."""
    return "machine learning video tutorial"
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

    @pytest.mark.asyncio
    async def test_search_calls_backend(self, mock_backend):
        """Test that search calls backend with correct params."""
        deps = SearchDeps(backend=mock_backend)
        agent = SearchAgent(deps=deps)

        await agent.process(SearchInput(query="test", top_k=5))

        mock_backend.search.assert_called_once_with(
            query="test",
            top_k=5
        )

    @pytest.mark.asyncio
    async def test_handles_backend_error(self, mock_backend):
        """Test graceful handling of backend error."""
        mock_backend.search.side_effect = ConnectionError("Backend down")

        deps = SearchDeps(backend=mock_backend)
        agent = SearchAgent(deps=deps)

        result = await agent.process(SearchInput(query="test"))

        assert result.error is not None
        assert "Backend down" in result.error
```

### Patching

```python
from unittest.mock import patch

class TestWithPatching:

    @patch("cogniverse_agents.search.agent.VespaBackend")
    def test_with_patched_backend(self, mock_backend_class):
        """Test with patched backend class."""
        mock_instance = MagicMock()
        mock_backend_class.return_value = mock_instance

        agent = SearchAgent()

        mock_backend_class.assert_called_once()
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
    def vespa_client(self):
        """Create real Vespa client."""
        import os
        vespa_url = os.getenv("VESPA_URL", "http://localhost:8080")
        return VespaClient(url=vespa_url)

    @pytest.mark.asyncio
    async def test_real_search(self, vespa_client):
        """Test search against real Vespa."""
        results = await vespa_client.search(
            query="test",
            schema="test_schema"
        )
        assert isinstance(results, list)
```

### Skipping Without Services

```python
import pytest
import os

requires_vespa = pytest.mark.skipif(
    os.getenv("VESPA_URL") is None,
    reason="VESPA_URL not set"
)

@requires_vespa
class TestVespaRequired:
    """Tests that require Vespa."""

    def test_vespa_operation(self):
        ...
```

### Test Database Setup

```python
@pytest.fixture(scope="session")
def test_database():
    """Create test database for session."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)

@pytest.fixture
def clean_database(test_database):
    """Clean database before each test."""
    # Clear tables before test
    conn = sqlite3.connect(test_database)
    conn.execute("DELETE FROM configs")
    conn.commit()
    conn.close()
    return test_database
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

**pyproject.toml**:
```toml
[tool.coverage.run]
source = ["libs/"]
omit = ["**/tests/*", "**/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

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

Each test workflow has these jobs:

1. **unit-tests** - Fast unit tests with mocked dependencies
2. **fast-integration-tests** - Real integration tests (runs on every push)
3. **integration-tests** - Full integration suite (main/develop only)
4. **lint** - Code linting with ruff/black
5. **test-imports** - Verify package imports work
6. **coverage-report** - Combined coverage (if applicable)

### Fast Integration Tests

Every workflow runs `fast-integration-tests` on every push to provide quick feedback with real Docker containers:

```yaml
fast-integration-tests:
  runs-on: ubuntu-latest
  timeout-minutes: 15
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

  - name: Free up disk space for Vespa
    run: |
      # Vespa requires <75% disk usage
      sudo rm -rf /usr/share/dotnet
      sudo rm -rf /usr/local/lib/android
      sudo rm -rf /opt/ghc
      sudo rm -rf /opt/hostedtoolcache/CodeQL
      sudo docker image prune -af

  - name: Pre-pull Vespa Docker image
    run: docker pull vespaengine/vespa:latest

  - name: Run integration tests
    timeout-minutes: 10
    run: |
      JAX_PLATFORM_NAME=cpu uv run python -m pytest \
        tests/module/integration/ \
        -v --tb=short
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
@pytest.fixture(scope="module")
def vespa_docker():
    """Start Vespa container for module tests."""
    manager = VespaDockerManager()
    manager.start()
    yield manager
    manager.stop()
```

**Phoenix:**
```python
@pytest.fixture(scope="module")
def phoenix_container():
    """Start Phoenix container for telemetry tests."""
    container_name = f"phoenix_test_{int(time.time())}"
    subprocess.run([
        "docker", "run", "-d", "--name", container_name,
        "-p", "6006:6006", "-p", "4317:4317",
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

# In conftest.py
pytest_plugins = ('pytest_asyncio',)

# Or use decorator
@pytest.mark.asyncio
async def test_async_function():
    ...
```

### Import Errors

```bash
# Ensure packages are installed
uv sync

# Check import path
python -c "import cogniverse_core; print(cogniverse_core.__file__)"

# Run from project root
cd /path/to/cogniverse
uv run pytest tests/ -v
```

### Flaky Tests

```python
# Use retries for network-dependent tests
@pytest.mark.flaky(reruns=3)
def test_network_operation():
    ...

# Or add explicit timeout
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

1. **Organize tests** by package (agents, foundation, runtime)
2. **Use fixtures** for common setup
3. **Run with JAX_PLATFORM_NAME=cpu** to avoid GPU issues
4. **100% pass rate required** before commit
5. **Find affected tests** with grep before committing
6. **Use mocks** for external services in unit tests
7. **Mark integration tests** that require real services
