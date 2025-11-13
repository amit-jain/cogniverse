# Agent Tests CI Strategy

**Last Updated:** 2025-11-13

## Overview

CI testing strategy for **cogniverse-agents** package (Implementation Layer) in the 10-package layered architecture. This strategy balances comprehensive coverage with fast CI execution using tiered testing approach.

## 10-Package Architecture Context

**cogniverse-agents** is in the **Implementation Layer** and depends on:
- **Foundation Layer**: cogniverse-sdk (interfaces), cogniverse-foundation (config)
- **Core Layer**: cogniverse-core (base agents, registries, memory), cogniverse-evaluation (metrics)

**Tests validate**:
- DSPy 3.0 routing agents with GEPA optimization
- Multi-modal search agents (video, audio, images, documents, text, dataframes)
- A2A protocol communication
- Query modality detection (GLiNER, LLM strategies)
- Relationship extraction and relevance boosting

## Two-Tier Testing Approach

### CI Fast Tests (`@pytest.mark.ci_fast`)
**Purpose**: Essential coverage for every commit
**Runtime**: ~1:48 minutes (78 tests)
**Coverage**: 37% of cogniverse_agents package with critical paths
**Execution**: `JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m "unit and ci_fast"`

**What runs**:
1. **Core Agent Tests** (39 tests)
   - Routing agent initialization and basic routing
   - Video search agent core functionality
   - Agent registry operations
   - Memory-aware mixins

2. **DSPy Integration Tests** (9 tests)
   - GEPA optimizer initialization
   - Query enhancement modules
   - Relationship extraction
   - Signature validation

3. **Multi-Agent Orchestration** (22 tests)
   - Workflow template generation
   - Result aggregation logic
   - Agent utilization tracking
   - Parallel execution patterns

4. **System Integration** (8 tests)
   - Agent endpoint creation
   - Registry initialization
   - Configuration loading

**Coverage highlights**:
- `routing_agent.py`: 82% (comprehensive core functionality)
- `video_search_agent.py`: 76% (search and reranking)
- DSPy modules: Core signature and module structure
- Multi-agent workflows: Template and aggregation logic

### Full Test Suite
**Purpose**: Comprehensive validation
**Runtime**: ~6+ minutes (381 tests for unit tests)
**Coverage**: Complete package validation
**Execution**: `JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m unit`

**What runs**:
- All 78 ci_fast tests
- Extended agent functionality tests
- Edge case handling
- Performance benchmarks
- Integration scenarios
- E2E workflows (optional, with real services)

## Multi-Modal Testing in CI

### CI-Safe Multi-Modal Tests

**Video Processing**:
```python
@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.requires_cv2  # Mocked in CI
class TestVideoSearchAgent:
    def test_video_frame_analysis(self):
        # Mocked frame extraction
        # Mocked ColPali embeddings
        pass
```

**Audio Processing**:
```python
@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.requires_whisper  # Mocked in CI
class TestAudioAgent:
    def test_audio_transcription(self):
        # Mocked Whisper
        pass
```

**Multi-Modal Search**:
```python
@pytest.mark.unit
@pytest.mark.ci_fast
class TestMultiModalSearch:
    def test_cross_modal_reranking(self):
        # Test relationship boosting
        # Test modality fusion
        pass
```

## Usage

### CI (Automatic)
```bash
# Runs automatically in GitHub Actions
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m "unit and ci_fast" -v --cov=libs/agents/cogniverse_agents

# Implementation Layer CI job
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit tests/routing/unit -m "unit and ci_fast" -v
```

### Local Development
```bash
# Run all unit tests (comprehensive)
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m unit -v

# Run only fast tests (quick validation before commit)
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m "unit and ci_fast" -v

# Run integration tests (requires services)
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/integration -v

# Run E2E tests (requires Ollama, Vespa, Phoenix)
JAX_PLATFORM_NAME=cpu timeout 600 uv run pytest tests/agents/e2e -v
```

## Marking Tests as CI Fast

Mark essential tests with both markers:

```python
import pytest

@pytest.mark.unit
@pytest.mark.ci_fast
class TestRoutingAgent:
    """Core routing agent functionality."""

    def test_route_video_query(self):
        """Test video query routing decision."""
        # Core functionality test
        pass

    def test_route_with_gepa_optimization(self):
        """Test GEPA-optimized routing."""
        # Essential optimization test
        pass

@pytest.mark.unit
# No ci_fast marker - runs only in full suite
class TestRoutingAgentEdgeCases:
    """Extended edge case tests."""

    def test_malformed_query_handling(self):
        """Test handling of malformed queries."""
        # Extended test, not in ci_fast
        pass
```

## DSPy Testing with JAX_PLATFORM_NAME=cpu

**Critical**: All DSPy-based tests require `JAX_PLATFORM_NAME=cpu` to avoid GPU initialization overhead:

```bash
# ✅ Correct
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m "unit and ci_fast"

# ❌ Incorrect - will be slow or fail
uv run pytest tests/agents/unit -m "unit and ci_fast"
```

**Why**: DSPy uses JAX for optimization, which attempts GPU initialization by default. CI environments typically don't have GPUs, and initialization adds significant overhead.

## Current CI Fast Tests Summary

### Core Agent Tests (39 tests)

**Routing Agent** (12 tests):
- Initialization and configuration
- Basic routing decisions (video/text/both)
- Modality detection with GLiNER
- LLM-based routing with DSPy
- Workflow determination
- Error handling

**Video Search Agent** (10 tests):
- Initialization with profiles (ColPali, VideoPrism, ColQwen)
- Video frame analysis
- Multi-modal search execution
- Reranking with relationship boosting
- Result formatting

**Enhanced Search Agent** (8 tests):
- Text search functionality
- Multi-modal query processing
- Cross-modal relevance scoring
- Hybrid ranking strategies

**Agent Registry** (5 tests):
- Agent registration
- Agent discovery
- Registry operations
- Multi-tenant isolation

**Memory Integration** (4 tests):
- Mem0 memory wrapper
- Context management
- Tenant-specific memory

### DSPy Integration Tests (9 tests)

**GEPA Optimization**:
- Optimizer initialization
- Experience buffer management
- Policy adaptation
- Performance tracking

**Query Enhancement**:
- Entity extraction with GLiNER
- Relationship detection
- Query expansion
- Modality classification

**Module Structure**:
- Signature validation
- Module composition
- Teleprompter integration

### Multi-Agent Orchestration (22 tests)

**Workflow Intelligence** (12 tests):
- Template generation for multi-agent workflows
- Workflow optimization strategies
- Execution tracking and metrics
- Agent coordination patterns

**Result Aggregator** (7 tests):
- Result enhancement and fusion
- Cross-modal aggregation logic
- Edge case handling (empty results, errors)
- Tenant isolation in aggregation

**Multi-Agent Orchestrator** (4 tests):
- Workflow planning and scheduling
- Agent utilization tracking
- Parallel execution management
- A2A protocol communication

### System Integration (8 tests)

**Endpoint Creation**:
- FastAPI agent endpoints
- A2A protocol handlers
- Health check endpoints

**Registry Initialization**:
- Agent registry setup
- Backend registry integration
- DSPy module registry

**Configuration Loading**:
- Multi-tenant configuration
- Profile-based initialization
- Environment detection

## Layer Dependency Testing

**Foundation Layer Dependencies**:
```python
# Tests validate cogniverse-sdk usage
from cogniverse_sdk.interfaces.backend import BackendInterface
from cogniverse_sdk.document import Document

# Tests validate cogniverse-foundation usage
from cogniverse_foundation.config.base_config import BaseConfig
```

**Core Layer Dependencies**:
```python
# Tests validate cogniverse-core usage
from cogniverse_core.agents.base_agent import BaseAgent
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.common.tenant_utils import with_tenant_context
from cogniverse_core.registries.agent_registry import AgentRegistry

# Tests validate cogniverse-evaluation usage
from cogniverse_evaluation.metrics.accuracy import AccuracyMetric
```

## Multi-Tenant Testing in CI

**Tenant Isolation**:
```python
@pytest.mark.unit
@pytest.mark.ci_fast
class TestMultiTenantRouting:
    """Test tenant-specific routing."""

    def test_tenant_specific_agent_registry(self):
        """Test agent registry per tenant."""
        agent1 = RoutingAgent(tenant_id="tenant_a")
        agent2 = RoutingAgent(tenant_id="tenant_b")
        # Validate isolation
        pass

    def test_tenant_specific_memory(self):
        """Test memory isolation per tenant."""
        # Validate Mem0 memory isolation
        pass
```

## CI Workflow Configuration

**GitHub Actions Implementation Layer CI**:
```yaml
name: Test Implementation Layer

on: [push, pull_request]

jobs:
  agents-ci-fast:
    runs-on: ubuntu-latest
    timeout-minutes: 5  # Fast tests should complete in < 2 minutes
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv sync
      - name: Run CI Fast Tests
        run: |
          JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit tests/routing/unit \
            -m "unit and ci_fast" -v \
            --cov=libs/agents/cogniverse_agents \
            --cov-report=xml \
            --cov-fail-under=37

  agents-full:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv sync
      - name: Run Full Unit Tests
        run: |
          JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit tests/routing/unit \
            -m unit -v \
            --cov=libs/agents/cogniverse_agents \
            --cov-report=xml \
            --cov-fail-under=80
```

## Coverage Requirements

**CI Fast Tests**: 37% minimum (focused on critical paths)
**Full Unit Tests**: 80% minimum (comprehensive coverage)
**Integration Tests**: Functional validation (coverage not enforced)
**E2E Tests**: End-to-end workflows (optional in CI)

## Best Practices

### For Developers

1. **Add ci_fast marker to critical tests**:
   ```python
   @pytest.mark.unit
   @pytest.mark.ci_fast  # Essential test
   def test_core_functionality(self):
       pass
   ```

2. **Keep ci_fast tests fast** (< 1 second each):
   - Mock heavy dependencies (LLMs, embeddings)
   - Use small test data
   - Avoid network calls

3. **Use JAX_PLATFORM_NAME=cpu for DSPy tests**:
   ```bash
   JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/
   ```

4. **Test multi-modality in ci_fast**:
   - Include tests for video, audio, images, documents, text
   - Mock modality-specific processors in CI

5. **Validate layer dependencies**:
   - Ensure agents use cogniverse-core base classes
   - Verify proper use of cogniverse-evaluation metrics
   - Validate cogniverse-sdk interface compliance

### For CI Configuration

1. **Set timeouts appropriately**:
   - CI fast: 5 minutes
   - Full unit: 10 minutes
   - Integration: 15 minutes
   - E2E: 30 minutes

2. **Use coverage thresholds**:
   - CI fast: 37% minimum
   - Full unit: 80% minimum

3. **Layer-specific testing**:
   - Run foundation tests first (fastest)
   - Run core tests second
   - Run implementation tests third (agents, routing)
   - Run application tests last (ingestion)

## Troubleshooting

### Tests Slow in CI

**Issue**: CI fast tests taking > 2 minutes

**Solutions**:
1. Verify `JAX_PLATFORM_NAME=cpu` is set
2. Check for unmocked heavy dependencies (LLMs, embeddings)
3. Reduce test data size
4. Remove network calls

### Coverage Drops Below Threshold

**Issue**: CI fast coverage < 37%

**Solutions**:
1. Add more ci_fast markers to critical tests
2. Review coverage report: `JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit -m "unit and ci_fast" --cov=libs/agents/cogniverse_agents --cov-report=html`
3. Identify untested critical paths
4. Add targeted tests

### DSPy/JAX Errors

**Issue**: JAX initialization errors in CI

**Solutions**:
1. Always use `JAX_PLATFORM_NAME=cpu`
2. Mock DSPy teleprompters in unit tests
3. Use real DSPy only in integration/E2E tests

## Summary

The agent tests CI strategy provides:

1. **Fast Feedback**: ci_fast tests complete in ~1:48 minutes
2. **Critical Coverage**: 37% coverage of essential agent functionality
3. **Comprehensive Validation**: Full suite with 80%+ coverage
4. **Layer-Aware Testing**: Validates dependencies on Foundation and Core layers
5. **Multi-Modal Support**: Tests video, audio, images, documents, text, dataframes
6. **Multi-Tenant Validation**: Tests tenant isolation and context management
7. **DSPy Optimization**: Tests GEPA, query enhancement, relationship extraction
8. **UV Workspace Integration**: Uses workspace packages via `uv run pytest`

**Key Principles**:
- Mark essential tests with `@pytest.mark.ci_fast`
- Always use `JAX_PLATFORM_NAME=cpu` for DSPy tests
- Mock heavy dependencies in CI (LLMs, embeddings, external services)
- Validate layer boundaries and package dependencies
- Test all modalities with appropriate mocking strategy
- Maintain coverage thresholds (37% ci_fast, 80% full)
