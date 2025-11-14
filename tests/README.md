# Cogniverse Test Suite

**Last Updated:** 2025-11-13

Comprehensive test coverage for the Cogniverse 10-package layered architecture supporting multi-modal content intelligence (video, audio, images, documents, text, dataframes) with multi-agent orchestration, GEPA optimization, and multi-tenant isolation.

## Quick Start

```bash
# Run all tests with UV workspace
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# Run layer-by-layer
JAX_PLATFORM_NAME=cpu uv run pytest tests/foundation/ -v  # Foundation
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ tests/evaluation/ -v  # Core
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ tests/routing/ -v  # Implementation
uv run pytest tests/ingestion/ tests/system/ -v  # Application

# Run with coverage across workspace
JAX_PLATFORM_NAME=cpu uv run pytest --cov=libs/ --cov-report=html --cov-report=term-missing

# CI-safe tests only
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -m ci_safe -v

# Local tests with real services
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -m "not ci_safe" -v
```

## 10-Package Architecture Under Test

### Foundation Layer
- **cogniverse-sdk** (`tests/sdk/`): Backend interfaces, universal document model for all modalities
- **cogniverse-foundation** (`tests/foundation/`): Configuration base, telemetry interfaces

### Core Layer
- **cogniverse-core** (`tests/common/`): Base agents, registries, Mem0 memory, tenant utilities
- **cogniverse-evaluation** (`tests/evaluation/`): Provider-agnostic experiments, metrics, datasets
- **cogniverse-telemetry-phoenix** (`tests/telemetry/`): Phoenix provider plugin (entry points)

### Implementation Layer
- **cogniverse-agents** (`tests/agents/`, `tests/routing/`): DSPy 3.0 routing, GEPA optimization, search agents
- **cogniverse-vespa** (`tests/backends/`): Multi-tenant schemas, 9 ranking strategies, search client
- **cogniverse-synthetic** (`tests/synthetic/`): GEPA/MIPRO/Bootstrap training data generation

### Application Layer
- **cogniverse-runtime** (`tests/ingestion/`): Multi-modal pipeline, FastAPI server, tenant middleware
- **cogniverse-dashboard** (`tests/dashboard/`): Streamlit analytics, Phoenix experiments, UMAP viz

## Test Organization

```
tests/
├── foundation/                 # Foundation layer (config, telemetry interfaces)
├── sdk/                        # SDK layer (backend interface, document model)
├── common/                     # Core layer (base agents, memory, registries)
├── evaluation/                 # Evaluation framework (experiments, metrics)
├── telemetry/                  # Phoenix telemetry provider
├── agents/                     # Implementation layer - agents
│   ├── unit/                  # Unit tests (mocked, fast)
│   ├── integration/           # Integration tests (selective mocking)
│   └── e2e/                   # E2E tests (real Ollama, Vespa, Phoenix)
├── routing/                    # Routing and optimization
│   ├── unit/                  # Strategy unit tests
│   └── integration/           # Tiered routing integration
├── backends/                   # Vespa backend tests
├── ingestion/                  # Application layer - ingestion pipeline
│   ├── unit/                  # Processor unit tests (80%+ coverage)
│   └── integration/           # Pipeline integration tests
├── memory/                     # Mem0 memory system tests
└── system/                     # System integration tests
    └── resources/             # Test data and schemas
```

## Multi-Modal Testing

The test suite validates all content modalities:

| Modality | Processors | Embedding Models | Schema Types |
|----------|-----------|------------------|--------------|
| **Video** | Frame extraction, chunk segmentation | ColPali, VideoPrism, ColQwen | Multi-vector, single-vector |
| **Audio** | Whisper transcription, chunking | Text embeddings | BM25 + dense hybrid |
| **Images** | Frame-level processing | ColPali visual embeddings | Frame-based |
| **Documents** | PDF extraction, text chunking | ColPali, text embeddings | Document schemas |
| **Text** | Chunk processing | Dense, binary, BM25 | Hybrid ranking |
| **Dataframes** | Column processing | Column embeddings | Structured search |

## Test Categories

### Pytest Markers

#### Test Types
- `@pytest.mark.unit` - Fast, isolated unit tests (< 1 second each)
- `@pytest.mark.integration` - Integration tests with services (1-10 seconds)
- `@pytest.mark.e2e` - End-to-end tests with real services (10-60 seconds)
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.slow` - Tests > 5 seconds

#### Environment Markers
- `@pytest.mark.ci_safe` - Runs in CI (mocked dependencies)
- `@pytest.mark.ci_fast` - Fast subset for CI (< 2 minutes total)
- `@pytest.mark.local_only` - Requires local resources (heavy models, GPU)

#### Backend Requirements
- `@pytest.mark.requires_vespa` - Needs Vespa running (localhost:8080)
- `@pytest.mark.requires_docker` - Needs Docker daemon
- `@pytest.mark.requires_modal` - Needs Modal account/token
- `@pytest.mark.requires_ollama` - Needs Ollama service (localhost:11434)

#### Model Requirements
- `@pytest.mark.requires_colpali` - ColPali model (colpali-engine)
- `@pytest.mark.requires_videoprism` - VideoPrism model (../videoprism/)
- `@pytest.mark.requires_gliner` - GLiNER models (gliner package)
- `@pytest.mark.requires_llm` - LLM models via Ollama
- `@pytest.mark.requires_whisper` - Whisper model (openai-whisper)

#### Resource Requirements
- `@pytest.mark.requires_gpu` - GPU required (CUDA)
- `@pytest.mark.high_memory` - > 8GB RAM needed

## Running Tests by Layer

### Foundation Layer Tests
```bash
# SDK interface contracts
uv run pytest tests/sdk/test_backend_interface.py -v
uv run pytest tests/sdk/test_document_model.py -v

# Foundation configuration and telemetry
uv run pytest tests/foundation/ -v
```

### Core Layer Tests
```bash
# Core functionality (requires JAX_PLATFORM_NAME=cpu for DSPy)
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ -v --cov=libs/core/cogniverse_core

# Evaluation framework
uv run pytest tests/evaluation/ -v --cov=libs/evaluation/cogniverse_evaluation

# Phoenix telemetry provider
uv run pytest tests/telemetry/ -v --cov=libs/telemetry-phoenix/cogniverse_telemetry_phoenix

# Memory system (Mem0 wrapper)
JAX_PLATFORM_NAME=cpu uv run pytest tests/memory/ -v
```

### Implementation Layer Tests
```bash
# Agents with DSPy 3.0 optimization
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit/ -m "unit and ci_fast" -v
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/integration/ -v
JAX_PLATFORM_NAME=cpu timeout 600 uv run pytest tests/agents/e2e/ -v

# Routing system (GLiNER, LLM, GEPA)
JAX_PLATFORM_NAME=cpu uv run pytest tests/routing/unit/ -v
JAX_PLATFORM_NAME=cpu uv run pytest tests/routing/integration/ -v

# Vespa backend (multi-tenant, 9 ranking strategies)
uv run pytest tests/backends/ -v --cov=libs/vespa/cogniverse_vespa

# Synthetic data generation
JAX_PLATFORM_NAME=cpu uv run pytest tests/synthetic/ -v
```

### Application Layer Tests
```bash
# Multi-modal ingestion pipeline
uv run pytest tests/ingestion/unit/ -v --cov=libs/runtime/cogniverse_runtime/ingestion
uv run python scripts/test_ingestion.py --integration --ci-safe

# System integration tests
uv run pytest tests/system/ -v

# Dashboard tests
uv run pytest tests/dashboard/ -v
```

## Running Tests by Modality

```bash
# Video processing tests
uv run pytest tests/ -k video -v

# Audio processing tests
uv run pytest tests/ -k audio -v

# Image processing tests
uv run pytest tests/ -k image -v

# Document processing tests
uv run pytest tests/ -k document -v

# Multi-modal search tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ tests/routing/ -k search -v
```

## Coverage Reports

```bash
# Full workspace coverage
JAX_PLATFORM_NAME=cpu uv run pytest --cov=libs/ --cov-report=html --cov-report=term-missing

# Package-specific coverage
uv run pytest tests/common/ --cov=libs/core/cogniverse_core --cov-report=html
uv run pytest tests/agents/ --cov=libs/agents/cogniverse_agents --cov-report=html
uv run pytest tests/backends/ --cov=libs/vespa/cogniverse_vespa --cov-report=html

# Coverage thresholds
JAX_PLATFORM_NAME=cpu uv run pytest --cov=libs/ --cov-fail-under=80
```

## CI/CD Integration

### Layer-Based CI Strategy

Tests run in separate CI jobs for each architecture layer:

**Foundation CI:**
```yaml
- name: Test Foundation Layer
  run: uv run pytest tests/sdk/ tests/foundation/ -v --cov=libs/sdk --cov=libs/foundation
```

**Core CI:**
```yaml
- name: Test Core Layer
  run: |
    JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ tests/evaluation/ tests/telemetry/ \
      -v --cov=libs/core --cov=libs/evaluation --cov=libs/telemetry-phoenix
```

**Implementation CI:**
```yaml
- name: Test Implementation Layer
  run: |
    JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit/ tests/routing/unit/ \
      -m "unit and ci_fast" -v --cov=libs/agents
```

**Application CI:**
```yaml
- name: Test Application Layer
  run: |
    uv run pytest tests/ingestion/unit/ -m "unit and ci_safe" -v \
      --cov=libs/runtime/cogniverse_runtime/ingestion
```

### Environment Variables

```bash
# CI environment detection
export CI=true                           # Auto-detected in GitHub Actions
export JAX_PLATFORM_NAME=cpu            # Required for DSPy tests
export PYTEST_TIMEOUT=1800              # 30-minute timeout
export RUN_HEAVY_TESTS=false            # Skip heavy model tests in CI

# Local development
export ENABLE_INTEGRATION_TESTS=true
export VESPA_HOST=localhost
export OLLAMA_HOST=localhost:11434
export PHOENIX_HOST=localhost:6006
```

## Test Data

### Test Data Location
- `data/testset/evaluation/sample_videos/` - Sample video files
- `data/testset/evaluation/video_search_queries.csv` - Test queries
- `tests/system/resources/videos/` - Small test videos for integration tests
- `tests/system/resources/configs/` - Test configurations
- `tests/system/resources/schemas/` - Vespa schema definitions

### Using Test Utilities
```python
from tests.utils.markers import has_ffmpeg, has_gpu, vespa_running

# Check for available resources
if has_ffmpeg():
    # Run video processing tests
    pass

if vespa_running():
    # Run Vespa integration tests
    pass
```

## Writing Tests

### Test Structure by Layer

**Foundation Layer Test:**
```python
import pytest
from cogniverse_sdk.interfaces.backend import BackendInterface

@pytest.mark.unit
@pytest.mark.ci_safe
class TestBackendInterface:
    """Test SDK backend interface contract."""

    def test_search_method_signature(self):
        """Test search method has correct signature."""
        # Test implementation
        pass
```

**Core Layer Test:**
```python
import pytest
from cogniverse_core.agents.base_agent import BaseAgent

@pytest.mark.unit
@pytest.mark.ci_safe
class TestBaseAgent:
    """Test core base agent functionality."""

    def test_agent_initialization(self):
        """Test agent initialization with tenant ID."""
        # Test implementation
        pass
```

**Implementation Layer Test:**
```python
import pytest
from cogniverse_agents.routing.routing_agent import RoutingAgent

@pytest.mark.unit
@pytest.mark.ci_fast
class TestRoutingAgent:
    """Test routing agent with DSPy optimization."""

    def test_route_query_with_gepa(self):
        """Test GEPA-optimized routing."""
        # Test implementation with mocked LLM
        pass
```

**Application Layer Test:**
```python
import pytest
from cogniverse_runtime.ingestion.video_processor import VideoProcessor

@pytest.mark.unit
@pytest.mark.ci_safe
class TestVideoProcessor:
    """Test multi-modal video processing."""

    def test_process_video_frames(self):
        """Test frame extraction and embedding."""
        # Test implementation with mocked dependencies
        pass
```

### Best Practices

1. **Use appropriate markers** - Help CI/local execution filtering
2. **Mock external dependencies** - Keep tests fast and reliable
3. **Test across all modalities** - Video, audio, images, documents, text, dataframes
4. **Maintain coverage** - 80%+ for core, 85%+ for new code
5. **Use descriptive names** - Clear test purpose and expected behavior
6. **Use fixtures** - Shared test data via conftest.py
7. **Test async properly** - Use pytest-asyncio for async code
8. **Document complex tests** - Add docstrings explaining test scenarios
9. **Test multi-tenancy** - Validate tenant isolation and context management
10. **Test layer boundaries** - Ensure packages respect dependency hierarchy

## Debugging Tests

### Verbose Output
```bash
# Show print statements and full output
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v -s

# Show warnings
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v --tb=short

# Full traceback
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v --tb=long
```

### Debug Specific Test
```bash
# Run single test with debugger
JAX_PLATFORM_NAME=cpu uv run pytest tests/routing/unit/test_routing_agent.py::TestRoutingAgent::test_route_query -vv --pdb

# Run with full output
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit/test_routing_agent.py -vv -s
```

### Check Test Discovery
```bash
# List all tests
uv run pytest tests/ --collect-only

# List tests with specific marker
uv run pytest tests/ --collect-only -m integration

# List tests matching keyword
uv run pytest tests/ --collect-only -k routing
```

## Performance Testing

### Benchmark Tests
```bash
# Run benchmarks only
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -m benchmark --benchmark-only

# Compare benchmark results
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -m benchmark --benchmark-compare
```

### Load Testing
```python
@pytest.mark.benchmark
def test_routing_performance(benchmark):
    router = create_router()
    result = benchmark(router.route, "test query")
    assert result.latency < 0.1  # 100ms threshold
```

## Troubleshooting

### Common Issues

**ImportError in tests**
```bash
# Ensure workspace is synced
uv sync

# Verify packages are installed
uv run python -c "import cogniverse_core; import cogniverse_agents"
```

**Vespa connection errors**
```bash
# Check Vespa is running
docker ps | grep vespa
curl http://localhost:8080/ApplicationStatus
```

**DSPy/JAX errors**
```bash
# Always use JAX_PLATFORM_NAME=cpu for DSPy tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ -v
```

**Model download failures**
```bash
# Pre-download Ollama models
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

**CI test failures**
```bash
# Run CI-safe tests locally
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -m ci_safe -v
```

## Coverage Requirements

- **Foundation Layer**: 90% minimum (pure interfaces)
- **Core Layer**: 85% minimum (critical functionality)
- **Implementation Layer**: 80% minimum (complex integrations)
- **Application Layer**: 80% minimum (pipeline orchestration)
- **New Code**: 90% minimum (maintain quality)

Monitor coverage trends:
```bash
# Generate coverage badge
coverage-badge -o coverage.svg

# Upload to codecov (in CI)
codecov --file coverage.xml
```

## Summary

The Cogniverse test suite provides comprehensive validation of:

- **10-Package Architecture**: Layer-by-layer testing (Foundation → Core → Implementation → Application)
- **Multi-Modal Processing**: Video, audio, images, documents, text, dataframes
- **Multi-Agent Orchestration**: DSPy 3.0 routing, GEPA optimization, A2A protocol
- **Multi-Tenant Isolation**: Schema-per-tenant, context management, memory isolation
- **UV Workspace**: Cross-package testing with unified dependency management
- **CI/CD Integration**: Layer-based CI strategy with environment-aware execution

**Key Testing Principles:**
1. Test from Foundation upward through layers
2. Mock appropriately for CI (ci_safe) vs local (local_only)
3. Validate all modalities (video, audio, images, documents, text, dataframes)
4. Use `JAX_PLATFORM_NAME=cpu` for DSPy tests
5. Use `uv run pytest` for workspace testing
6. Maintain 80%+ coverage across all layers
