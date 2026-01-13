# Cogniverse Developer Guide

**Version:** 2.0.0 | **Last Updated:** 2025-11-13 | **Status:** Production Ready

Complete guide for developers contributing to Cogniverse - the multi-agent system for multi-modal content analysis and search.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [System Architecture](#system-architecture)
3. [Development Environment](#development-environment)
4. [Code Organization](#code-organization)
5. [Development Workflows](#development-workflows)
6. [Testing Strategy](#testing-strategy)
7. [Contributing Code](#contributing-code)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### For New Developers

Welcome to Cogniverse! This guide will help you:
- Understand the system architecture
- Set up your development environment
- Navigate the codebase
- Write and test code
- Submit contributions

### Prerequisites

Before you begin, ensure you have:
- **Python 3.12+** installed
- **Git** for version control
- **Docker** for running services
- **uv** package manager: `pip install uv`
- **IDE** (VS Code, PyCharm recommended)
- **16GB+ RAM** (32GB recommended)

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd cogniverse

# Install dependencies
uv sync

# Start services
docker compose up -d

# Run tests to verify setup
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ -v

# Success! You're ready to develop
```

---

## System Architecture

### 11-Package Layered Architecture

Cogniverse uses a **UV workspace** with 11 packages in layered architecture:

```
┌─────────────────────────────────────────────┐
│          APPLICATION LAYER                   │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │ runtime         │  │ dashboard        │  │
│  │ (FastAPI Server)│  │ (Streamlit UI)   │  │
│  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│        IMPLEMENTATION LAYER                  │
│  ┌─────────┐  ┌────────┐  ┌──────────────┐ │
│  │ agents  │  │ vespa  │  │ synthetic    │ │
│  └─────────┘  └────────┘  └──────────────┘ │
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│             CORE LAYER                       │
│  ┌───────┐ ┌─────────────┐ ┌──────────────┐│
│  │ core  │ │ evaluation  │ │ telemetry-   ││
│  │       │ │             │ │ phoenix      ││
│  └───────┘ └─────────────┘ └──────────────┘│
└─────────────────────────────────────────────┘
               ↓ depends on ↓
┌─────────────────────────────────────────────┐
│         FOUNDATION LAYER                     │
│  ┌────────────┐    ┌──────────────────────┐ │
│  │ sdk        │    │ foundation           │ │
│  │ (Interfaces)│    │ (Config & Telemetry)│ │
│  └────────────┘    └──────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Key Principles

1. **Layered Dependencies**: Dependencies flow downward only (no cycles)
2. **SDK Foundation**: Zero internal dependencies, pure interfaces
3. **Separation of Concerns**: Each package has clear responsibilities
4. **Plugin Architecture**: Telemetry providers via entry points
5. **Multi-Tenancy**: Complete tenant isolation at every layer

### Package Responsibilities

| Package | Layer | Purpose | Key Modules |
|---------|-------|---------|-------------|
| **sdk** | Foundation | Backend interfaces, Document model | interfaces/, document.py |
| **foundation** | Foundation | Config base, telemetry interfaces | config/, telemetry/ |
| **core** | Core | Base classes, registries, memory | agents/, common/, registries/ |
| **evaluation** | Core | Experiments, metrics, datasets | experiments/, metrics/ |
| **telemetry-phoenix** | Core | Phoenix telemetry provider (plugin) | provider.py, evaluation/ |
| **agents** | Implementation | Routing, search, orchestration | routing/, search/, tools/ |
| **vespa** | Implementation | Vespa backend, schema management | backends/, tenant/, schema/ |
| **synthetic** | Implementation | Synthetic data generation | generators/, service.py |
| **runtime** | Application | FastAPI server, ingestion | server/, ingestion/, middleware/ |
| **dashboard** | Application | Streamlit UI, analytics | phoenix/, streamlit/ |

---

## Development Environment

### Initial Setup

#### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

#### 2. Clone and Setup Repository

```bash
# Clone
git clone <repository-url>
cd cogniverse

# Install workspace (all 11 packages + dependencies)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

#### 3. Start Infrastructure Services

```bash
# Start Vespa, Phoenix, Ollama
docker compose up -d

# Verify services
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:6006/health             # Phoenix
curl http://localhost:11434/api/tags          # Ollama
```

#### 4. Verify Installation

```bash
# Run tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ -v

# Verify all packages installed
uv pip list | grep cogniverse

# Expected: 11 packages (sdk, foundation, core, evaluation, etc.)
```

### IDE Setup

#### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv": true
  }
}
```

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal"
    },
    {
      "name": "Python: FastAPI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "cogniverse_runtime.main:app",
        "--reload"
      ],
      "env": {
        "JAX_PLATFORM_NAME": "cpu"
      }
    }
  ]
}
```

#### PyCharm Configuration

1. **Set Python Interpreter**: File → Settings → Project → Python Interpreter → Select `.venv/bin/python`
2. **Enable Pytest**: Settings → Tools → Python Integrated Tools → Testing → pytest
3. **Configure Ruff**: Settings → Tools → External Tools → Add Ruff
4. **Mark Directories**: Right-click `libs/*/cogniverse_*` → Mark Directory as → Sources Root

---

## Code Organization

### Workspace Structure

```
cogniverse/
├── libs/                      # All SDK packages
│   ├── sdk/                   # Foundation: Pure interfaces
│   ├── foundation/            # Foundation: Config & telemetry base
│   ├── core/                  # Core: Base classes & registries
│   ├── evaluation/            # Core: Experiments & metrics
│   ├── telemetry-phoenix/     # Core: Phoenix provider (plugin)
│   ├── agents/                # Implementation: Agents
│   ├── vespa/                 # Implementation: Vespa backend
│   ├── synthetic/             # Implementation: Synthetic data
│   ├── runtime/               # Application: FastAPI server
│   └── dashboard/             # Application: Streamlit UI
├── tests/                     # Test suite
│   ├── common/                # Core package tests
│   ├── agents/                # Agents package tests
│   ├── routing/               # Routing integration tests
│   ├── memory/                # Memory tests
│   ├── ingestion/             # Ingestion tests
│   └── evaluation/            # Evaluation tests
├── scripts/                   # Operational scripts
│   ├── run_ingestion.py       # Video ingestion
│   ├── deploy_json_schema.py  # Schema deployment
│   └── run_experiments_with_visualization.py  # Experiments
├── configs/                   # Configuration files
│   ├── config.json            # Backend profiles
│   └── schemas/               # Vespa schema templates
├── docs/                      # Documentation
│   ├── architecture/          # Architecture guides
│   ├── modules/               # Module documentation
│   ├── operations/            # Operations guides
│   └── testing/               # Testing guides
├── pyproject.toml             # Workspace root config
├── uv.lock                    # Unified lockfile
└── config.yml                 # System configuration
```

### Package Structure Pattern

Each package follows this structure:

```
libs/my_package/
├── pyproject.toml             # Package configuration
├── README.md                  # Package documentation
├── cogniverse_my_package/     # Python package
│   ├── __init__.py            # Package exports
│   ├── module1/               # Feature module
│   │   ├── __init__.py
│   │   └── implementation.py
│   ├── module2/               # Feature module
│   │   ├── __init__.py
│   │   └── implementation.py
│   └── utils/                 # Utilities
│       └── helpers.py
└── tests/                     # Package-specific tests (optional)
```

### Naming Conventions

**Packages:**
- Installable name: `cogniverse-my-package` (hyphens)
- Import name: `cogniverse_my_package` (underscores)

**Modules:**
- Lowercase with underscores: `video_search_agent.py`
- Avoid abbreviations: `config.py` not `cfg.py`

**Classes:**
- PascalCase: `VideoSearchAgent`, `TenantSchemaManager`
- Descriptive names: `RoutingAgent` not `RA`

**Functions:**
- snake_case: `get_tenant_id()`, `deploy_schema()`
- Verb + noun: `create_agent()`, `load_config()`

**Constants:**
- UPPER_SNAKE_CASE: `MAX_BATCH_SIZE`, `DEFAULT_TIMEOUT`

---

## Development Workflows

### Working on a Single Package

```bash
# Navigate to package
cd libs/agents

# Install package in editable mode
uv pip install -e .

# Make changes
vim cogniverse_agents/routing/routing_agent.py

# Run package tests
cd ../..
uv run pytest tests/agents/ -v

# Run linting
uv run ruff check libs/agents/

# Format code
uv run ruff format libs/agents/
```

### Working Across Multiple Packages

```bash
# Make changes in core
vim libs/core/cogniverse_core/agents/base_agent.py

# Make changes in agents (uses core)
vim libs/agents/cogniverse_agents/routing/routing_agent.py

# Changes in core immediately visible to agents (workspace)
uv run pytest tests/agents/ -v
```

### Adding a New Feature

**Example: Add a new search strategy**

1. **Create feature branch**:
```bash
git checkout -b feature/add-bm25-rerank-strategy
```

2. **Implement in agents package**:
```python
# libs/agents/cogniverse_agents/search/strategies/bm25_rerank.py
from cogniverse_agents.search.strategies.base import SearchStrategy

class BM25RerankStrategy(SearchStrategy):
    """BM25 with dense reranking"""

    def search(self, query: str, top_k: int) -> List[Document]:
        # Get candidates with BM25
        candidates = self._bm25_search(query, top_k * 3)

        # Rerank with dense embeddings
        reranked = self._dense_rerank(query, candidates, top_k)

        return reranked
```

3. **Add tests**:
```python
# tests/agents/test_bm25_rerank_strategy.py
import pytest
from cogniverse_agents.search.strategies.bm25_rerank import BM25RerankStrategy

def test_bm25_rerank():
    strategy = BM25RerankStrategy()
    results = strategy.search("tutorial", top_k=10)
    assert len(results) == 10
    assert results[0].score > results[-1].score
```

4. **Run tests**:
```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/test_bm25_rerank_strategy.py -v
```

5. **Update documentation**:
```bash
vim docs/modules/agents.md  # Add new strategy
```

6. **Commit and push**:
```bash
git add .
git commit -m "Add BM25 rerank search strategy"
git push origin feature/add-bm25-rerank-strategy
```

### Adding a New Package Dependency

**To a specific package**:
```bash
cd libs/agents
uv add scikit-learn>=1.3.0  # Adds to agents/pyproject.toml
cd ../..
uv sync  # Update workspace
```

**To root workspace** (shared dependency):
```bash
uv add --project . numpy>=1.24.0  # Adds to root pyproject.toml
uv sync
```

### Running Scripts

```bash
# Ingestion
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
  --video_dir data/videos \
  --profile video_colpali_smol500_mv_frame \
  --tenant default

# Experiments
JAX_PLATFORM_NAME=cpu uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name golden_eval_v1 \
  --profiles video_colpali_smol500_mv_frame

# Deploy schema
uv run python scripts/deploy_json_schema.py \
  --schema-path configs/schemas/video_colpali_smol500_mv_frame.json \
  --tenant-id default
```

---

## Testing Strategy

### Test Organization

Tests are organized by package/module:

```
tests/
├── common/              # cogniverse_core tests
├── agents/              # cogniverse_agents tests
├── routing/             # Routing integration tests
├── memory/              # Memory tests
├── ingestion/           # Ingestion tests
├── evaluation/          # Evaluation tests
└── telemetry/           # Telemetry tests
```

### Running Tests

**Full test suite**:
```bash
JAX_PLATFORM_NAME=cpu timeout 7200 uv run pytest -v
```

**Package-specific tests**:
```bash
# Core package
uv run pytest tests/common/ -v

# Agents package
uv run pytest tests/agents/ -v

# Integration tests
uv run pytest tests/routing/integration/ -v
```

**Single test**:
```bash
uv run pytest tests/agents/test_routing_agent.py::TestRoutingAgent::test_route_query -v
```

**With coverage**:
```bash
uv run pytest tests/agents/ --cov=cogniverse_agents --cov-report=html
```

### Writing Tests

**Unit test example**:
```python
# tests/agents/test_routing_agent.py
import pytest
from cogniverse_agents.routing.routing_agent import RoutingAgent

@pytest.fixture
def routing_agent():
    return RoutingAgent(tenant_id="test")

class TestRoutingAgent:
    async def test_route_query(self, routing_agent):
        """Test basic query routing"""
        decision = await routing_agent.route_query("tutorial")

        assert decision.strategy is not None
        assert decision.confidence > 0
        assert len(decision.entities) >= 0

    async def test_route_query_with_entities(self, routing_agent):
        """Test routing with entity extraction"""
        decision = await routing_agent.route_query("Python tutorial by John Doe")

        assert "Python" in decision.entities
        assert decision.modality is not None
```

**Integration test example**:
```python
# tests/routing/integration/test_routing_with_vespa.py
import pytest
from cogniverse_agents.routing.routing_agent import RoutingAgent
from cogniverse_vespa.backends.vespa_search_client import VespaSearchClient

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_routing_pipeline():
    """Test routing agent with real Vespa backend"""
    # Initialize agent
    agent = RoutingAgent(tenant_id="test")

    # Initialize Vespa
    vespa = VespaSearchClient(host="localhost", port=8080, tenant_id="test")

    # Route query
    decision = await agent.route_query("machine learning tutorial")

    # Execute search with routed strategy
    results = await vespa.search(
        query="machine learning tutorial",
        strategy=decision.strategy,
        top_k=10
    )

    assert len(results) > 0
    assert results[0].score > 0
```

### Test Fixtures

**Workspace-level fixtures** (`tests/conftest.py`):
```python
import pytest
from cogniverse_core.config.unified_config import SystemConfig

@pytest.fixture(scope="session")
def system_config():
    """Shared config for all tests"""
    return SystemConfig(
        tenant_id="test",
        vespa_url="http://localhost:8080",
        phoenix_enabled=False  # Disable for tests
    )

@pytest.fixture
def tenant_id():
    """Test tenant ID"""
    return "test_tenant"
```

---

## Contributing Code

### Code Review Process

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Implement feature**: Write code + tests
3. **Run tests**: `uv run pytest -v`
4. **Run linting**: `uv run ruff check . && uv run ruff format .`
5. **Commit changes**: `git commit -m "Add feature X"`
6. **Push branch**: `git push origin feature/my-feature`
7. **Create PR**: Use GitHub PR template
8. **Address feedback**: Make requested changes
9. **Merge**: Once approved, squash and merge

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Documentation
- [ ] README updated
- [ ] Module docs updated
- [ ] API docs updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] No warnings from linters
```

### Commit Message Guidelines

**Format**: `<type>: <subject>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

**Examples**:
```
feat: Add BM25 rerank search strategy
fix: Resolve tensor encoding issue in ColPali embeddings
docs: Update architecture overview with 11-package structure
refactor: Extract tenant context logic to middleware
test: Add integration tests for routing agent
```

---

## Best Practices

### Code Quality

1. **Type Hints**: Always use type hints
```python
def search(query: str, top_k: int) -> List[Document]:
    pass
```

2. **Docstrings**: Document all public functions
```python
def search(query: str, top_k: int) -> List[Document]:
    """
    Search for documents matching query.

    Args:
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of matching documents
    """
    pass
```

3. **Error Handling**: Use specific exceptions
```python
try:
    result = backend.search(query, top_k)
except ConnectionError:
    logger.error("Backend connection failed")
    raise
except ValueError as e:
    logger.error(f"Invalid query: {e}")
    raise
```

4. **Logging**: Use structured logging
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing query", extra={"query": query, "tenant_id": tenant_id})
logger.error("Search failed", extra={"error": str(e)}, exc_info=True)
```

### Performance

1. **Async/Await**: Use async for I/O operations
```python
async def search(self, query: str) -> List[Document]:
    results = await self.backend.search_async(query)
    return results
```

2. **Batch Processing**: Process in batches
```python
for batch in batched(documents, batch_size=100):
    await backend.ingest_documents(batch)
```

3. **Caching**: Cache expensive operations
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)
```

### Security

1. **Tenant Isolation**: Always use tenant_id
```python
def search(self, query: str, tenant_id: str) -> List[Document]:
    schema = self.get_tenant_schema(tenant_id)
    return self.backend.search(query, schema=schema)
```

2. **Input Validation**: Validate all inputs
```python
def search(self, query: str, top_k: int) -> List[Document]:
    if not query or len(query) > 1000:
        raise ValueError("Query must be 1-1000 characters")
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be 1-100")
    return self._search(query, top_k)
```

---

## Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Reinstall workspace
uv sync
source .venv/bin/activate
```

**Issue**: Tests failing
```bash
# Check services are running
docker ps | grep -E "vespa|phoenix|ollama"

# Restart services
docker compose restart
```

**Issue**: Out of memory
```bash
# Reduce batch size
export EMBEDDING_BATCH_SIZE=8
export JAX_PLATFORM_NAME=cpu
```

### Debug Mode

Enable debug logging:
```bash
export COGNIVERSE_LOG_LEVEL=DEBUG
export PHOENIX_ENABLED=true

uv run pytest tests/agents/ -v -s  # -s shows print statements
```

---

## Next Steps

### For New Developers
- Read [Architecture Overview](architecture/overview.md)
- Read [SDK Architecture](architecture/sdk-architecture.md)
- Explore [Module Documentation](modules/)

### For Contributors
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Read [Testing Guide](testing/TESTING_GUIDE.md)
- Check [GitHub Issues](https://github.com/org/cogniverse/issues)

### For Advanced Features
- Read [Multi-Tenant Architecture](architecture/multi-tenant.md)
- Read [System Flows](architecture/system-flows.md)
- Read [Performance Monitoring](operations/performance-monitoring.md)

---

**Version**: 2.0.0
**Architecture**: UV Workspace (11 Packages - Layered Architecture)
**Last Updated**: 2025-11-13
**Status**: Production Ready
