# Cogniverse SDK Architecture

**Last Updated:** 2025-10-15
**Purpose:** Deep-dive into UV workspace structure, package design, and development workflows
**Audience:** Developers working on Cogniverse SDK packages

---

## Table of Contents

1. [Overview](#overview)
2. [UV Workspace Structure](#uv-workspace-structure)
3. [Package Architecture](#package-architecture)
4. [Dependency Management](#dependency-management)
5. [Development Workflows](#development-workflows)
6. [Testing Strategy](#testing-strategy)
7. [Building and Distribution](#building-and-distribution)
8. [Import Patterns](#import-patterns)
9. [Package Versioning](#package-versioning)

---

## Overview

Cogniverse is structured as a **UV workspace monorepo** containing 5 independent Python packages. This architecture provides:

- **Modular Design**: Clear separation of concerns across packages
- **Dependency Management**: Explicit package boundaries and dependencies
- **Independent Testing**: Test packages in isolation or together
- **Selective Deployment**: Deploy only what's needed for each use case
- **Faster Iteration**: Work on specific packages without full system overhead

### Key Statistics

- **Total Python Files**: 334 files across all packages
- **Packages**: 5 workspace members in `libs/` directory
- **Python Version**: >= 3.12 required
- **Build System**: Hatchling for all packages
- **Package Manager**: UV for workspace and dependency management

### Package Breakdown

```
libs/
├── core/          # 115 files - Core interfaces and utilities
├── agents/        # 99 files  - Agent implementations
├── vespa/         # 17 files  - Vespa backend integration
├── runtime/       # 49 files  - FastAPI server and ingestion
└── dashboard/     # 54 files  - Streamlit analytics UI
```

---

## UV Workspace Structure

### Root Configuration

The root `pyproject.toml` defines the workspace:

```toml
[tool.uv.workspace]
members = ["libs/*"]

[project]
name = "congniverse"  # Root project (legacy compatibility)
version = "0.1.0"
requires-python = ">=3.12"
```

**Key Points**:
- `members = ["libs/*"]` discovers all packages in `libs/` directory
- Root dependencies apply to all packages (heavy ML dependencies)
- Each package can override or extend root dependencies

### Workspace Benefits

1. **Shared Dependencies**: Common dependencies installed once
2. **Local Package Resolution**: Packages reference each other via workspace
3. **Unified Lock File**: Single `uv.lock` for reproducibility
4. **Cross-Package Development**: Edit multiple packages simultaneously
5. **Consistent Versions**: All packages use same version of shared dependencies

### Directory Structure

```
cogniverse/
├── pyproject.toml              # Root workspace config
├── uv.lock                     # Unified lock file
├── libs/                       # Workspace packages
│   ├── core/
│   │   ├── pyproject.toml      # Package config
│   │   ├── README.md           # Package docs
│   │   └── cogniverse_core/    # Python package
│   │       ├── __init__.py
│   │       ├── agents/         # Module
│   │       ├── common/         # Module
│   │       └── ...
│   ├── agents/
│   ├── vespa/
│   ├── runtime/
│   └── dashboard/
├── tests/                      # Workspace-level tests
├── scripts/                    # Development scripts
└── docs/                       # Documentation
```

---

## Package Architecture

### Package 1: cogniverse-core

**Purpose**: Foundation package providing core interfaces, registries, and base classes.

**Package Name**: `cogniverse-core` (installable)
**Import Name**: `cogniverse_core` (Python import)
**Files**: 115 Python files

#### Module Structure

```
cogniverse_core/
├── __init__.py
├── agents/                     # Base agent classes
│   ├── __init__.py
│   ├── base_agent.py          # BaseAgent interface
│   ├── memory_aware_mixin.py  # Memory integration mixin
│   └── health_check_mixin.py  # Health monitoring mixin
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── tenant_utils.py        # Tenant context management
│   ├── mem0_memory_manager.py # Mem0 memory wrapper
│   └── config_store.py        # Configuration storage
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── unified_config.py      # SystemConfig class
│   ├── schema.py              # Config schemas
│   └── api_mixin.py           # API configuration
├── telemetry/                  # Observability
│   ├── __init__.py
│   ├── manager.py             # TelemetryManager
│   └── modality_metrics.py    # Metrics tracking
├── evaluation/                 # Evaluation framework
│   ├── __init__.py
│   ├── data/
│   │   ├── storage.py         # PhoenixStorage
│   │   └── experiment.py      # Experiment tracking
│   └── metrics/
│       └── calculator.py      # Metric calculations
└── types/                      # Shared type definitions
    ├── __init__.py
    └── common_types.py        # Common types
```

#### Dependencies

```toml
dependencies = [
    "dspy-ai>=3.0.2",           # LLM programming
    "litellm>=1.73.0",          # Multi-provider LLM
    "opentelemetry-api>=1.20.0", # Telemetry
    "arize-phoenix>=4.0.0",     # Observability
    "pydantic>=2.0.0",          # Data validation
    "mem0ai>=0.1.118",          # Memory management
]
```

#### Key Responsibilities

- **Base Classes**: Abstract agent interfaces and mixins
- **Configuration**: Unified config system with schema validation
- **Telemetry**: Phoenix integration and metrics tracking
- **Memory**: Mem0 wrapper with multi-tenant support
- **Evaluation**: Phoenix storage and experiment tracking

---

### Package 2: cogniverse-vespa

**Purpose**: Vespa backend implementation with multi-tenant schema management.

**Package Name**: `cogniverse-vespa` (installable)
**Import Name**: `cogniverse_vespa` (Python import)
**Files**: 17 Python files

#### Module Structure

```
cogniverse_vespa/
├── __init__.py
├── tenant/                     # Multi-tenant management
│   ├── __init__.py
│   ├── tenant_schema_manager.py  # Schema lifecycle
│   └── tenant_backend.py         # Tenant-aware backend
├── backends/                   # Vespa clients
│   ├── __init__.py
│   ├── vespa_search_client.py    # Search operations
│   └── vespa_admin_client.py     # Admin operations
├── schemas/                    # Schema definitions
│   ├── __init__.py
│   ├── video_frames.sd           # Video frame schema
│   └── agent_memories.sd         # Memory schema
└── ingestion/                  # Data ingestion
    ├── __init__.py
    └── vespa_ingestor.py          # Batch ingestion
```

#### Dependencies

```toml
dependencies = [
    "cogniverse-core",          # Core interfaces (workspace)
    "pyvespa>=0.59.0",          # Vespa Python client
    "numpy>=1.24.0",            # Array operations
]

[tool.uv.sources]
cogniverse-core = { workspace = true }
```

#### Key Responsibilities

- **Schema Management**: Deploy and manage tenant-specific Vespa schemas
- **Search Backend**: Query execution and result processing
- **Data Ingestion**: Batch ingestion with schema validation
- **Tenant Isolation**: Schema-per-tenant pattern implementation

---

### Package 3: cogniverse-agents

**Purpose**: Agent implementations including routing, video search, and orchestration.

**Package Name**: `cogniverse-agents` (installable)
**Import Name**: `cogniverse_agents` (Python import)
**Files**: 99 Python files

#### Module Structure

```
cogniverse_agents/
├── __init__.py
├── routing/                    # Routing agents
│   ├── __init__.py
│   ├── routing_agent.py       # Main routing agent
│   ├── modality_cache.py      # Query modality caching
│   └── parallel_executor.py   # Parallel agent execution
├── search/                     # Search agents
│   ├── __init__.py
│   ├── video_search_agent.py  # ColPali/VideoPrism search
│   └── multi_modal_reranker.py # Reranking logic
├── orchestration/              # Composing agents
│   ├── __init__.py
│   └── composing_agent.py     # Multi-agent orchestrator
└── optimization/               # Query optimization
    ├── __init__.py
    └── query_optimizer.py     # Query refinement
```

#### Dependencies

```toml
dependencies = [
    "cogniverse-core",          # Base classes (workspace)
    "torch>=2.5.0",             # Deep learning
    "transformers>=4.50.0",     # Hugging Face models
    "colpali-engine>=0.3.12",   # ColPali embeddings
    "sentence-transformers>=5.1.0", # Embeddings
    "xgboost>=3.0.5",           # ML optimization
    "spacy>=3.7.0",             # NLP
    "gliner>=0.2.21",           # NER
]

[tool.uv.sources]
cogniverse-core = { workspace = true }
```

#### Key Responsibilities

- **Agent Implementations**: Concrete agent classes (routing, search, composing)
- **Query Processing**: Modality detection and query optimization
- **Search Enhancement**: Multi-modal reranking and relevance scoring
- **Parallel Execution**: Concurrent agent execution with timeout handling

---

### Package 4: cogniverse-runtime

**Purpose**: FastAPI server with ingestion pipeline and tenant middleware.

**Package Name**: `cogniverse-runtime` (installable)
**Import Name**: `cogniverse_runtime` (Python import)
**Files**: 49 Python files

#### Module Structure

```
cogniverse_runtime/
├── __init__.py
├── server/                     # FastAPI server
│   ├── __init__.py
│   ├── app.py                 # Main application
│   └── routes.py              # API routes
├── middleware/                 # Request middleware
│   ├── __init__.py
│   └── tenant_context.py      # Tenant extraction
├── ingestion/                  # Data ingestion
│   ├── __init__.py
│   ├── pipeline.py            # Ingestion pipeline
│   ├── video_processor.py     # Video processing
│   ├── audio_processor.py     # Audio transcription
│   └── document_processor.py  # Document processing
└── processing/                 # Processing utilities
    ├── __init__.py
    ├── keyframe_extractor.py  # Keyframe extraction
    └── embeddings.py          # Embedding generation
```

#### Dependencies

```toml
dependencies = [
    "cogniverse-core",          # Core utilities (workspace)
    "fastapi>=0.100.0",         # Web framework
    "uvicorn>=0.20.0",          # ASGI server
    "opencv-python>=4.12.0",    # Video processing
    "openai-whisper>=20231117", # Audio transcription
    "pdf2image>=1.17.0",        # Document processing
    "jax>=0.7.0",               # VideoPrism embeddings
]

[project.optional-dependencies]
vespa = ["cogniverse-vespa"]    # Optional Vespa backend
agents = ["cogniverse-agents"]  # Optional agent support
all = ["cogniverse-vespa", "cogniverse-agents"]

[tool.uv.sources]
cogniverse-core = { workspace = true }
cogniverse-vespa = { workspace = true }
cogniverse-agents = { workspace = true }
```

#### Key Responsibilities

- **API Server**: FastAPI endpoints for search, ingestion, health checks
- **Tenant Middleware**: Extract `tenant_id` from JWT or headers
- **Ingestion Pipeline**: Process video, audio, documents with configurable profiles
- **Dynamic Backend Loading**: Load backends (Vespa) based on configuration

---

### Package 5: cogniverse-dashboard

**Purpose**: Streamlit analytics UI with Phoenix integration for experiment visualization.

**Package Name**: `cogniverse-dashboard` (installable)
**Import Name**: `cogniverse_dashboard` (Python import)
**Files**: 54 Python files

#### Module Structure

```
cogniverse_dashboard/
├── __init__.py
├── app.py                      # Main Streamlit app
├── components/                 # UI components
│   ├── __init__.py
│   ├── experiment_viewer.py   # Experiment results
│   ├── metrics_dashboard.py   # System metrics
│   └── trace_viewer.py        # Phoenix trace viewer
└── analytics/                  # Analytics logic
    ├── __init__.py
    ├── experiment_analyzer.py # Experiment analysis
    └── visualization.py       # Chart generation
```

#### Dependencies

```toml
dependencies = [
    "cogniverse-core",          # Core utilities (workspace)
    "streamlit>=1.29.0",        # Dashboard UI
    "plotly>=6.0.0",            # Interactive charts
    "arize-phoenix>=4.0.0",     # Observability UI
    "arize-phoenix-evals>=0.4.0", # Evaluation UI
    "inspect-ai>=0.3.0",        # LLM inspection
    "embedding-atlas>=0.2.0",   # Embedding visualization
    "umap-learn>=0.5.0",        # Dimensionality reduction
]

[tool.uv.sources]
cogniverse-core = { workspace = true }
```

#### Key Responsibilities

- **Analytics Dashboard**: Visualize experiment results and system metrics
- **Phoenix Integration**: Embedded Phoenix UI for trace analysis
- **Experiment Management**: Browse and compare evaluation experiments
- **Embedding Visualization**: UMAP plots of video embeddings

---

## Dependency Management

### Dependency Graph

```
cogniverse-core (foundation)
    ↓
    ├─→ cogniverse-vespa (Vespa integration)
    ├─→ cogniverse-agents (agent implementations)
    ├─→ cogniverse-runtime (server + ingestion)
    └─→ cogniverse-dashboard (analytics UI)
```

**Key Principles**:
- **Core is Foundation**: All packages depend on `cogniverse-core`
- **No Circular Dependencies**: Clean dependency hierarchy
- **Optional Dependencies**: Runtime can work without agents/vespa
- **Workspace References**: Packages reference each other via `{ workspace = true }`

### Workspace Dependency Declaration

Each package declares workspace dependencies in `pyproject.toml`:

```toml
[project]
dependencies = [
    "cogniverse-core",  # Workspace dependency
    # ... external dependencies
]

[tool.uv.sources]
cogniverse-core = { workspace = true }
```

**Benefits**:
- UV resolves to local package (no PyPI lookup)
- Editable installs for development
- Changes in core immediately available to dependent packages

### External Dependencies

Heavy ML dependencies are declared in **root** `pyproject.toml` for shared installation:

```toml
[project]
dependencies = [
    "torch>=2.5.0",
    "jax>=0.7.0",
    "transformers>=4.50.0",
    # ... shared across all packages
]
```

**Strategy**:
- Root declares heavy ML libraries (torch, jax, transformers)
- Packages declare only package-specific dependencies
- Reduces duplication and ensures version consistency

---

## Development Workflows

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/cogniverse.git
cd cogniverse

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install workspace (all packages + dependencies)
uv sync

# Verify installation
uv run python -c "import cogniverse_core; print('OK')"
```

### Working on Single Package

```bash
# Navigate to package directory
cd libs/core

# Run tests for this package only
uv run pytest ../../tests/common/  # Tests for core

# Run linting
uv run ruff check cogniverse_core/

# Build package
uv build
```

### Working Across Multiple Packages

Workspace mode allows editing multiple packages simultaneously:

```bash
# From root directory
cd cogniverse/

# Make changes in core
vim libs/core/cogniverse_core/config/unified_config.py

# Make changes in agents (uses core)
vim libs/agents/cogniverse_agents/routing/routing_agent.py

# Test both together
uv run pytest tests/routing/  # Uses both packages
```

**Key Point**: Changes in `cogniverse-core` are immediately visible to `cogniverse-agents` without reinstalling.

### Running Scripts

Scripts in `scripts/` directory use workspace packages:

```bash
# Run ingestion script (uses runtime + vespa)
uv run python scripts/run_ingestion.py \
    --video_dir data/videos \
    --backend vespa

# Run experiments (uses agents + core + dashboard)
uv run python scripts/run_experiments_with_visualization.py \
    --dataset-path data/queries.csv \
    --profiles frame_based_colpali
```

### Adding Dependencies

**To a specific package**:

```bash
cd libs/agents
uv add scikit-learn>=1.3.0  # Adds to agents/pyproject.toml
```

**To root workspace** (shared dependency):

```bash
cd cogniverse/
uv add --project . numpy>=1.24.0  # Adds to root pyproject.toml
```

---

## Testing Strategy

### Test Organization

Tests are organized by package/module in `tests/` directory:

```
tests/
├── common/              # Tests for cogniverse_core
│   ├── test_config.py
│   └── test_tenant_utils.py
├── agents/              # Tests for cogniverse_agents
│   ├── test_routing_agent.py
│   └── test_video_search_agent.py
├── routing/             # Integration tests
│   ├── unit/
│   └── integration/
├── memory/              # Memory tests (core)
├── ingestion/           # Ingestion tests (runtime)
├── evaluation/          # Evaluation tests (core)
└── telemetry/           # Telemetry tests (core)
```

### Running Tests

**Full test suite**:

```bash
# From root
JAX_PLATFORM_NAME=cpu timeout 7200 uv run pytest -v
```

**Package-specific tests**:

```bash
# Test core package
uv run pytest tests/common/ tests/telemetry/ tests/evaluation/ -v

# Test agents package
uv run pytest tests/agents/ tests/routing/ -v

# Test vespa package
uv run pytest tests/backends/ -v

# Test runtime package
uv run pytest tests/ingestion/ -v
```

**Test isolation**:

```bash
# Run single test file
uv run pytest tests/routing/unit/test_routing_agent.py -v

# Run single test class
uv run pytest tests/memory/unit/test_mem0_memory_manager.py::TestMem0MemoryManager -v

# Run single test method
uv run pytest tests/routing/unit/test_routing_agent.py::TestRoutingAgent::test_route_query -v
```

### Test Fixtures

Workspace-level fixtures in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
from cogniverse_core.config.unified_config import SystemConfig
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

@pytest.fixture(scope="session")
def system_config():
    """Shared config for all tests"""
    return SystemConfig.from_yaml("config/test_config.yaml")

@pytest.fixture
def tenant_id():
    """Test tenant ID"""
    return "test_tenant"
```

### Integration Tests

Integration tests validate cross-package interactions:

```python
# tests/routing/integration/test_routing_with_vespa.py
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_vespa.vespa_search_client import VespaSearchClient
from cogniverse_core.common.tenant_utils import with_tenant_context

@pytest.mark.integration
async def test_routing_agent_with_vespa_backend(tenant_id):
    """Test routing agent with real Vespa backend"""

    # Initialize Vespa backend
    vespa_client = VespaSearchClient(
        host="localhost",
        port=8080,
        tenant_id=tenant_id
    )

    # Initialize routing agent
    agent = RoutingAgent(tenant_id=tenant_id)

    # Execute query through full stack
    result = await agent.route_query(
        query="cooking videos",
        backend=vespa_client
    )

    assert result["selected_agents"]
    assert result["query_modality"]
```

---

## Building and Distribution

### Building Individual Packages

Each package can be built independently:

```bash
cd libs/core
uv build

# Creates:
# dist/cogniverse_core-0.1.0-py3-none-any.whl
# dist/cogniverse_core-0.1.0.tar.gz
```

### Building All Packages

```bash
# From root
for pkg in libs/*; do
    echo "Building $pkg..."
    (cd $pkg && uv build)
done
```

### Distribution Strategy

**Development Installation** (editable):

```bash
# Install package in editable mode
uv pip install -e libs/core
uv pip install -e libs/agents
```

**Production Installation** (from built wheels):

```bash
# Install from wheel
uv pip install dist/cogniverse_core-0.1.0-py3-none-any.whl

# Install with dependencies
uv pip install cogniverse_core[dev]
```

### Package Publication

For internal or public PyPI:

```bash
# Build package
cd libs/core
uv build

# Publish to PyPI (requires credentials)
uv publish --token $PYPI_TOKEN

# Publish to private index
uv publish --index-url https://pypi.your-org.com
```

---

## Import Patterns

### Package-Level Imports

**Core utilities**:

```python
# Configuration
from cogniverse_core.config.unified_config import SystemConfig
from cogniverse_core.config.schema import ConfigSchema

# Base classes
from cogniverse_core.agents.base_agent import BaseAgent
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

# Telemetry
from cogniverse_core.telemetry.manager import TelemetryManager
from cogniverse_core.telemetry.modality_metrics import ModalityMetricsTracker

# Memory
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

# Tenant utilities
from cogniverse_core.common.tenant_utils import (
    get_tenant_id,
    with_tenant_context
)
```

**Vespa integration**:

```python
# Tenant management
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager
from cogniverse_vespa.tenant_aware_search_client import TenantAwareSearchClient

# Search clients
from cogniverse_vespa.vespa_search_client import VespaSearchClient
from cogniverse_vespa.vespa_search_client import VespaSearchClient  # Admin operations via search client

# Ingestion
from cogniverse_vespa.ingestion_client import VespaPyClient
```

**Agent implementations**:

```python
# Agents
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_agents.video_search_agent import VideoSearchAgent
from cogniverse_agents.composing_agent import ComposingAgent

# Utilities
from cogniverse_agents.routing.modality_cache import ModalityCacheManager
from cogniverse_agents.routing.parallel_executor import ParallelAgentExecutor
from cogniverse_agents.search.multi_modal_reranker import QueryModality
```

**Runtime components**:

```python
# Server
from cogniverse_runtime.server.app import create_app

# Middleware
from cogniverse_runtime.middleware.tenant_context import inject_tenant_context

# Ingestion
from cogniverse_runtime.ingestion.pipeline import IngestionPipeline
from cogniverse_runtime.ingestion.video_processor import VideoProcessor
```

**Dashboard**:

```python
# Dashboard app
from cogniverse_dashboard.app import run_dashboard

# Components
from cogniverse_dashboard.components.experiment_viewer import ExperimentViewer
from cogniverse_dashboard.analytics.experiment_analyzer import ExperimentAnalyzer
```

### Cross-Package Imports

**Agents using Core and Vespa**:

```python
# In cogniverse_agents/routing/routing_agent.py

from cogniverse_core.agents.base_agent import BaseAgent
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.telemetry.manager import TelemetryManager

class RoutingAgent(BaseAgent, MemoryAwareMixin):
    """
    Routing agent using core base classes and telemetry.

    Vespa backend injected at runtime (not imported directly).
    """

    def __init__(self, tenant_id: str):
        super().__init__(tenant_id=tenant_id)
        self.telemetry = TelemetryManager.get_instance(tenant_id)
```

**Runtime using Agents and Vespa**:

```python
# In cogniverse_runtime/server/app.py

from cogniverse_core.config.unified_config import SystemConfig
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

# Optional imports (only if extras installed)
try:
    from cogniverse_agents.routing_agent import RoutingAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

def create_app(config: SystemConfig):
    """Create FastAPI app with dynamic feature loading"""
    if AGENTS_AVAILABLE:
        routing_agent = RoutingAgent(tenant_id="default")
```

---

## Package Versioning

### Current Versioning

All packages use **synchronized versioning** at `0.1.0`:

```toml
# All libs/*/pyproject.toml
[project]
version = "0.1.0"
```

### Versioning Strategy

**Pre-1.0 (Current)**:
- All packages at same version (`0.1.0`)
- Breaking changes allowed without major version bump
- Fast iteration without version compatibility constraints

**Post-1.0 (Future)**:
- Packages can version independently
- Core follows semantic versioning strictly
- Dependent packages specify version ranges:

```toml
# Future: cogniverse-agents/pyproject.toml
dependencies = [
    "cogniverse-core>=1.0.0,<2.0.0",  # Major version compatibility
]
```

### Version Bumping

**Manual version bumps**:

```bash
# Update all packages to 0.2.0
for pkg in libs/*/pyproject.toml; do
    sed -i 's/version = "0.1.0"/version = "0.2.0"/' $pkg
done
```

**Automated version bumps** (future):

```bash
# Using bump2version or similar tool
bump2version minor  # 0.1.0 -> 0.2.0
```

### Compatibility Matrix

| Core Version | Agents | Vespa | Runtime | Dashboard |
|-------------|--------|-------|---------|-----------|
| 0.1.0       | 0.1.0  | 0.1.0 | 0.1.0   | 0.1.0     |

**Future independent versioning**:

| Core Version | Agents | Vespa | Runtime | Dashboard |
|-------------|--------|-------|---------|-----------|
| 1.2.0       | 1.0.0  | 1.1.0 | 1.0.0   | 1.3.0     |
| 1.3.0       | 1.1.0  | 1.1.0 | 1.1.0   | 1.4.0     |

---

## Best Practices

### 1. Package Boundaries

**DO**:
- Keep core lightweight (interfaces, utilities, base classes)
- Put concrete implementations in specialized packages (agents, vespa)
- Use dependency injection for cross-package coupling

**DON'T**:
- Import from vespa/agents in core (violates dependency direction)
- Create circular dependencies between packages
- Put heavy ML code in core (belongs in agents/runtime)

### 2. Dependency Management

**DO**:
- Declare workspace dependencies with `{ workspace = true }`
- Pin minimum versions for external dependencies (`>=1.0.0`)
- Use optional dependencies for optional features

**DON'T**:
- Hard-pin exact versions (`==1.0.0`) unless necessary
- Declare workspace packages in `dependencies` and forget `[tool.uv.sources]`
- Duplicate heavy dependencies across multiple packages

### 3. Testing

**DO**:
- Test packages in isolation when possible
- Use integration tests for cross-package workflows
- Mock external dependencies (Vespa, Phoenix) in unit tests

**DON'T**:
- Assume all packages are installed in tests
- Skip integration tests (they catch real issues)
- Test implementation details instead of interfaces

### 4. Imports

**DO**:
- Use absolute imports from package roots (`from cogniverse_core.config import ...`)
- Keep imports at top of file (except for optional imports)
- Use `try/except ImportError` for optional dependencies

**DON'T**:
- Use relative imports across packages (`from ...agents import ...`)
- Import entire modules (`from cogniverse_core import *`)
- Assume dependent packages are always available

---

## Common Development Scenarios

### Scenario 1: Adding a New Agent

**Steps**:

1. **Create agent class** in `libs/agents/cogniverse_agents/my_agent/`:

```python
# libs/agents/cogniverse_agents/my_agent/my_new_agent.py
from cogniverse_core.agents.base_agent import BaseAgent
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

class MyNewAgent(BaseAgent, MemoryAwareMixin):
    """New agent implementation"""

    def __init__(self, tenant_id: str):
        super().__init__(tenant_id=tenant_id)

    async def execute(self, query: str) -> dict:
        """Execute agent logic"""
        pass
```

2. **Add tests** in `tests/agents/test_my_new_agent.py`:

```python
import pytest
from cogniverse_agents.my_agent.my_new_agent import MyNewAgent

class TestMyNewAgent:
    async def test_execute(self):
        agent = MyNewAgent(tenant_id="test")
        result = await agent.execute("test query")
        assert result["status"] == "success"
```

3. **Run tests**:

```bash
uv run pytest tests/agents/test_my_new_agent.py -v
```

4. **Update exports** in `libs/agents/cogniverse_agents/__init__.py`:

```python
from .my_agent.my_new_agent import MyNewAgent

__all__ = ["MyNewAgent", ...]
```

### Scenario 2: Adding Multi-Tenant Feature to Core

**Steps**:

1. **Add utility** in `libs/core/cogniverse_core/common/tenant_utils.py`
2. **Add tests** in `tests/common/test_tenant_utils.py`
3. **Verify dependent packages** (agents, vespa, runtime) still work:

```bash
uv run pytest tests/agents/ tests/routing/ tests/memory/ -v
```

4. **Update documentation** in `docs/modules/common.md`

### Scenario 3: Adding New Vespa Schema

**Steps**:

1. **Create schema file** in `libs/vespa/cogniverse_vespa/schemas/my_schema.sd`
2. **Add schema manager support** in `tenant_schema_manager.py`:

```python
def get_schema_path(self, schema_name: str) -> Path:
    """Get path to schema file"""
    schema_files = {
        "video_frames": "video_frames.sd",
        "agent_memories": "agent_memories.sd",
        "my_schema": "my_schema.sd",  # New schema
    }
    return self.schema_dir / schema_files[schema_name]
```

3. **Test schema deployment**:

```bash
uv run pytest tests/backends/test_schema_deployment.py -v
```

4. **Update ingestion** to support new schema in `runtime/ingestion/pipeline.py`

---

## Troubleshooting

### Issue: Package not found

**Error**: `ModuleNotFoundError: No module named 'cogniverse_core'`

**Solution**:

```bash
# Reinstall workspace
cd cogniverse/
uv sync

# Verify installation
uv run python -c "import cogniverse_core"
```

### Issue: Workspace dependency not resolving

**Error**: `Package cogniverse-core not found`

**Solution**: Check `[tool.uv.sources]` declaration:

```toml
[tool.uv.sources]
cogniverse-core = { workspace = true }  # Must match package name
```

### Issue: Circular import

**Error**: `ImportError: cannot import name 'X' from partially initialized module 'Y'`

**Solution**: Refactor to break circular dependency:
- Move shared code to core
- Use dependency injection
- Use late imports (`import` inside function)

### Issue: Test failures after package change

**Error**: Tests pass in package but fail in workspace

**Solution**: Run full test suite to catch integration issues:

```bash
uv run pytest -v  # Full suite
```

---

## Next Steps

- **Multi-Tenant Guide**: See [multi-tenant.md](./multi-tenant.md) for tenant architecture
- **Module Guides**: See [docs/modules/](../modules/) for package-specific details
- **Development Guide**: See [docs/development/package-dev.md](../development/package-dev.md) for workflows
- **Testing Guide**: See [docs/testing/](../testing/) for comprehensive testing strategies

---

## Summary

Cogniverse SDK uses a **UV workspace** with **5 independent packages**:

1. **cogniverse-core**: Foundation (interfaces, config, telemetry, memory)
2. **cogniverse-vespa**: Vespa backend (schemas, search, multi-tenant)
3. **cogniverse-agents**: Agent implementations (routing, search, orchestration)
4. **cogniverse-runtime**: FastAPI server (API, ingestion, middleware)
5. **cogniverse-dashboard**: Streamlit UI (analytics, experiments, Phoenix)

**Key Characteristics**:
- 334 Python files across packages
- Clean dependency hierarchy (core → vespa/agents → runtime/dashboard)
- UV workspace for unified dependency management
- Python >= 3.12, Hatchling build system
- Modular design for flexible deployment
