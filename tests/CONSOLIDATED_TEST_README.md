# Cogniverse Test Suite - Comprehensive Guide

**Last Updated:** 2025-11-13

## Overview

This is a comprehensive test suite for the Cogniverse multi-modal content intelligence platform built on a **10-package layered architecture**. The suite validates multi-agent orchestration, multi-modal processing (video, audio, images, documents, text, dataframes), routing optimization, and backend integration with environment-aware execution and CI/CD integration.

### Architecture Under Test

**10-Package Layered Architecture:**
- **Foundation Layer**: cogniverse-sdk (backend interfaces), cogniverse-foundation (config, telemetry base)
- **Core Layer**: cogniverse-core (agents, registries, memory), cogniverse-evaluation (metrics), cogniverse-telemetry-phoenix (Phoenix provider)
- **Implementation Layer**: cogniverse-agents (routing, search), cogniverse-vespa (backend), cogniverse-synthetic (data generation)
- **Application Layer**: cogniverse-runtime (FastAPI, ingestion), cogniverse-dashboard (Streamlit UI)

## Quick Start

### Core Test Commands
```bash
# Run all tests with UV workspace (recommended)
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v --tb=long 2>&1 > /tmp/cogniverse-tests.log

# Run layer-by-layer tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/foundation/ -v --tb=long 2>&1 > /tmp/foundation-tests.log  # Foundation layer
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ tests/evaluation/ -v --tb=long 2>&1 > /tmp/core-tests.log  # Core layer
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ tests/routing/ -v --tb=long 2>&1 > /tmp/implementation-tests.log  # Implementation layer
JAX_PLATFORM_NAME=cpu uv run pytest tests/ingestion/ -v --tb=long 2>&1 > /tmp/application-tests.log  # Application layer

# Run comprehensive routing tests
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py 2>&1 > /tmp/routing-tests.log

# Run ingestion tests with environment detection
uv run python scripts/test_ingestion.py --integration 2>&1 > /tmp/ingestion-integration.log
```

**Note:** Use `JAX_PLATFORM_NAME=cpu` for DSPy tests to avoid GPU initialization overhead.

## Test Organization

### Layer-by-Layer Testing Approach

Tests are organized following the 10-package architecture layers, testing from Foundation → Core → Implementation → Application:

```
tests/
├── foundation/                 # Foundation layer tests
│   ├── test_config_base.py    # Configuration base classes
│   └── test_telemetry_interfaces.py  # Telemetry interfaces
├── sdk/                        # SDK layer tests
│   ├── test_backend_interface.py     # Backend interface contracts
│   └── test_document_model.py        # Universal document model
├── common/                     # Core layer tests
│   ├── test_base_agent.py     # Agent base classes
│   ├── test_mem0_memory_manager.py   # Memory management
│   └── test_tenant_utils.py   # Multi-tenant utilities
├── evaluation/                 # Evaluation framework tests
│   ├── test_experiments.py    # Experiment tracking
│   ├── test_metrics.py        # Provider-agnostic metrics
│   └── test_datasets.py       # Dataset handling
├── telemetry/                  # Telemetry provider tests
│   ├── test_phoenix_provider.py      # Phoenix telemetry
│   └── test_evaluation_provider.py   # Phoenix evaluation
├── agents/                     # Implementation layer - agents
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests with real services
├── routing/                    # Routing and optimization tests
│   ├── unit/                  # Strategy unit tests
│   ├── integration/           # Tiered routing integration
│   └── test_comprehensive_routing.py  # Multi-model benchmarks
├── backends/                   # Vespa backend tests
│   └── test_tenant_schema_manager.py # Multi-tenant schemas
├── runtime/                    # Runtime integration tests
│   └── integration/
│       └── test_ranking_strategies_real.py  # VespaSearchBackend ranking strategies (real Vespa + ColPali)
├── ingestion/                  # Application layer - runtime
│   ├── unit/                  # Processor unit tests
│   ├── integration/           # Pipeline integration tests
│   └── fixtures/              # Test fixtures and mocks
├── memory/                     # Memory system tests
│   ├── unit/                  # Mem0 wrapper tests
│   └── integration/           # Vespa memory backend tests
└── system/                     # System integration tests
    ├── test_e2e_workflows.py  # Complete workflows
    └── resources/             # Test data and schemas
```

## Test Categories

### 1. Multi-Modal Testing Strategy

The test suite validates processing across **all content modalities**:

| Modality | Processors Tested | Embedding Models | Backend Tests |
|----------|------------------|------------------|---------------|
| **Video** | Frame extraction, chunk segmentation | ColPali, VideoPrism, ColQwen | Multi-vector, single-vector |
| **Audio** | Whisper transcription, chunking | Text embeddings on transcripts | BM25 + dense hybrid |
| **Images** | Frame-level processing | ColPali (visual embeddings) | Frame-based schemas |
| **Documents** | PDF, text extraction | ColPali, text embeddings | Document schemas |
| **Text** | Chunk processing | Dense, binary, BM25 | Hybrid ranking |
| **Dataframes** | Structured data processing | Column embeddings | Structured search |

### 2. Layer-by-Layer Test Coverage

**Foundation Layer Tests:**
- Backend interface contracts (`cogniverse-sdk`)
- Universal document model for all modalities
- Configuration base classes (`cogniverse-foundation`)
- Telemetry provider interfaces

**Core Layer Tests:**
- Base agent classes and mixins (`cogniverse-core`)
- Mem0 memory manager with Vespa backend
- Multi-tenant utilities and context management
- Provider-agnostic evaluation framework (`cogniverse-evaluation`)
- Phoenix telemetry provider (`cogniverse-telemetry-phoenix`)

**Implementation Layer Tests:**
- DSPy 3.0 routing agent with GEPA optimization (`cogniverse-agents`)
- Multi-modal search and reranking
- Vespa backend with 9 ranking strategies (`cogniverse-vespa`)
- Synthetic data generation for optimizers (`cogniverse-synthetic`)

**Application Layer Tests:**
- Multi-modal ingestion pipeline (`cogniverse-runtime`)
- FastAPI endpoints and tenant middleware
- Streamlit dashboard components (`cogniverse-dashboard`)

### 3. Routing System Tests

#### Unit Tests (`tests/routing/unit/`)
- **GLiNER Strategy**: Named entity recognition for modality detection
- **LLM Strategy**: DSPy-based routing with local LLMs
- **GEPA Optimization**: Experience-guided policy adaptation
- **Modality Caching**: Query modality cache performance

#### Integration Tests (`tests/routing/integration/`)
- **Tiered Routing**: Multi-tier escalation logic with fallback
- **Multi-Modal Classification**: Video/audio/image/document/text/dataframe detection
- **Relationship Extraction**: Entity relationship detection for relevance boosting
- **Search Modality Detection**: Content type detection across modalities
- **Caching Behavior**: Route caching with tenant isolation
- **Error Handling**: Circuit breaker patterns and timeout handling

#### Comprehensive Testing
```bash
# Test all routing models and strategies (DSPy requires CPU-only JAX)
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py 2>&1 > /tmp/routing-tests.log

# Test specific components
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py llm-only 2>&1 > /tmp/routing-llm-tests.log      # LLM models only
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py gliner-only 2>&1 > /tmp/routing-gliner-tests.log  # GLiNER models only
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py hybrid-only 2>&1 > /tmp/routing-hybrid-tests.log  # Combined approaches
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py quick 2>&1 > /tmp/routing-quick-tests.log        # Fast subset
```

**Models Tested:**
- **DeepSeek R1**: 1.5b, 7b, 8b variants (via Ollama)
- **Gemma 3**: 1b, 4b, 12b variants (via Ollama)
- **Qwen 3**: 0.6b, 1.7b, 4b, 8b variants (via Ollama)
- **GLiNER**: All configured models in config.json (local inference)

### 4. Multi-Modal Ingestion Pipeline Tests

#### Unit Tests (80%+ Coverage Achieved)
- **AudioProcessor**: 99% coverage (67/67 statements)
- **ChunkProcessor**: 100% coverage (67/67 statements)
- **KeyframeProcessor**: 98% coverage (97/97 statements)

#### Integration Tests
- **Backend Integration**: Mock, Vespa, and real model tests
- **Pipeline Orchestration**: Multi-processor coordination
- **End-to-end Processing**: Complete video processing workflows

#### Environment-Aware Execution
```bash
# Check what's available in your environment
uv run python scripts/test_ingestion.py --env-info

# Run CI-safe tests (mocked dependencies)
uv run python scripts/test_ingestion.py --unit --ci-safe 2>&1 > /tmp/ingestion-unit-ci-safe.log

# Run local-only tests (requires heavy models)
uv run python scripts/test_ingestion.py --integration --local-only 2>&1 > /tmp/ingestion-local-models.log

# Run specific backend tests
uv run python scripts/test_ingestion.py --requires-vespa 2>&1 > /tmp/ingestion-vespa.log

# Run a test that declares its exact inference service requirement
uv run pytest tests/ingestion/integration/test_backend_ingestion.py::TestVespaBackendIngestion::test_colpali_vespa_ingestion -v --tb=long
```

## Pytest Markers System

### Test Types
- `unit`: Unit tests for individual components
- `integration`: Integration tests with multiple components
- `benchmark`: Performance benchmark tests
- `slow`: Tests that take significant time

### Environment Markers
- `ci_safe`: Tests safe to run in CI environment (lightweight, mocked)
- `local_only`: Tests that should only run locally (heavy models, real backends)

### Backend Requirements
- `requires_vespa`: Tests requiring Vespa backend
- `requires_docker`: Tests requiring Docker
- `requires_cv2`: Tests requiring OpenCV
- `requires_ffmpeg`: Tests requiring FFmpeg

### Inference Service Requirements
- `requires_inference("vllm_colpali")`: ColPali and ColQwen embedding tests
- `requires_inference("videoprism_jax")`: VideoPrism embedding tests
- `requires_inference("vllm_asr")`: Remote ASR tests
- `requires_whisper`: Local Whisper model tests

### Resource Requirements
- `requires_gpu`: Tests requiring GPU availability
- `local_only`: Heavy model tests excluded from CI

## Running Tests

### UV Workspace Testing

All tests run via `uv run pytest` to use the 10-package workspace:

```bash
# Full test suite (30 min timeout for integration tests)
JAX_PLATFORM_NAME=cpu timeout 1800 uv run pytest -v --tb=long 2>&1 > /tmp/cogniverse-tests.log

# Foundation layer tests
uv run pytest tests/sdk/ tests/foundation/ -v --tb=long 2>&1 > /tmp/foundation-tests.log

# Core layer tests (requires JAX_PLATFORM_NAME=cpu for DSPy)
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ tests/evaluation/ tests/telemetry/ -v --tb=long 2>&1 > /tmp/core-tests.log

# Implementation layer tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ tests/routing/ tests/backends/ -v --tb=long 2>&1 > /tmp/implementation-tests.log

# Application layer tests
uv run pytest tests/ingestion/ tests/system/ -v --tb=long 2>&1 > /tmp/application-tests.log

# With coverage reporting across workspace
JAX_PLATFORM_NAME=cpu uv run pytest -v --tb=long --cov=libs/ --cov-report=html --cov-report=term-missing 2>&1 > /tmp/cogniverse-coverage.log
```

### Layer-Specific Testing

**Foundation Layer:**
```bash
# SDK interface tests
uv run pytest tests/sdk/test_backend_interface.py -v --tb=long
uv run pytest tests/sdk/test_document_model.py -v --tb=long

# Foundation tests
uv run pytest tests/foundation/test_config_base.py -v --tb=long
```

**Core Layer:**
```bash
# Core functionality (DSPy integration requires CPU)
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/ -v --tb=long --cov=libs/core/cogniverse_core 2>&1 > /tmp/common-tests.log

# Evaluation framework
uv run pytest tests/evaluation/ -v --tb=long --cov=libs/evaluation/cogniverse_evaluation 2>&1 > /tmp/evaluation-tests.log

# Phoenix telemetry provider
uv run pytest tests/telemetry/test_phoenix_provider.py -v --tb=long
```

**Implementation Layer:**
```bash
# Agents with DSPy optimization
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit/ -m "unit and ci_fast" -v --tb=long 2>&1 > /tmp/agent-unit-tests.log
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/integration/ -v --tb=long 2>&1 > /tmp/agent-integration-tests.log

# Routing system
JAX_PLATFORM_NAME=cpu uv run pytest tests/routing/unit/ tests/routing/integration/ -v --tb=long 2>&1 > /tmp/routing-pytest.log

# Vespa backend
uv run pytest tests/backends/ -v --tb=long --cov=libs/vespa/cogniverse_vespa 2>&1 > /tmp/backend-tests.log
```

**Application Layer:**
```bash
# Multi-modal ingestion pipeline
uv run pytest tests/ingestion/unit/ -v --tb=long --cov=libs/runtime/cogniverse_runtime/ingestion 2>&1 > /tmp/ingestion-unit-tests.log

# Integration tests with environment detection
uv run python scripts/test_ingestion.py --integration --ci-safe 2>&1 > /tmp/ingestion-integration-ci-safe.log

# System integration tests
uv run pytest tests/system/ -v --tb=long 2>&1 > /tmp/system-tests.log
```

### Integration Tests with Services
```bash
# Mock backend tests (always available)
uv run python scripts/test_ingestion.py --integration --ci-safe 2>&1 > /tmp/ingestion-integration-ci-safe.log

# Vespa backend tests (requires Vespa running)
./scripts/start_vespa.sh
uv run python scripts/test_ingestion.py --requires-vespa 2>&1 > /tmp/ingestion-vespa.log

# Heavy model tests (local development only)
uv run python scripts/test_ingestion.py --integration --local-only 2>&1 > /tmp/ingestion-local-models.log

# E2E tests with real Ollama
JAX_PLATFORM_NAME=cpu timeout 600 uv run pytest tests/agents/e2e/ -v --tb=long 2>&1 > /tmp/agent-e2e-tests.log
```

### Specific Test Scenarios
```bash
# Test specific marker combinations
uv run pytest tests/ingestion/ -m "integration and requires_vespa" -v --tb=long 2>&1 > /tmp/ingestion-vespa-tests.log
uv run pytest tests/ingestion/ -m "unit and ci_safe" -v --tb=long 2>&1 > /tmp/ingestion-ci-safe-tests.log
uv run pytest tests/ingestion/ -m "integration and local_only" -v --tb=long 2>&1 > /tmp/ingestion-local-tests.log

# Test with coverage threshold
uv run python scripts/test_ingestion.py --unit --coverage-fail-under=80 2>&1 > /tmp/ingestion-coverage.log

# Exclude heavy models (CI default)
uv run python scripts/test_ingestion.py --exclude-heavy 2>&1 > /tmp/ingestion-light-tests.log
```

## Test Fixtures and Utilities

### Mock Dependencies
- **OpenCV**: Video processing mocks
- **Whisper**: Audio transcription mocks
- **PyTorch**: Model loading mocks
- **Vespa Client**: Backend ingestion mocks

### Test Data Generation
- **Synthetic Video Files**: Generated test videos
- **Expected Outputs**: Baseline processing results
- **Performance Benchmarks**: Timing expectations

### Smart Environment Detection
The test system automatically detects:
- Available dependencies (Docker, Vespa, FFmpeg, OpenCV)
- Available models (ColPali, VideoPrism, ColQwen, Whisper)
- Environment type (CI vs Local development)
- Resource availability (GPU, memory)

## CI/CD Integration

### GitHub Actions Workflows

**Layer-Based CI Strategy:**

1. **Foundation Layer CI** (`test-foundation.yml`)
   - SDK interface contract tests
   - Universal document model tests
   - Configuration base tests
   - Zero external dependencies

2. **Core Layer CI** (`test-core.yml`)
   - Core functionality tests with `JAX_PLATFORM_NAME=cpu`
   - Evaluation framework tests
   - Phoenix telemetry provider tests
   - 80%+ coverage requirement

3. **Implementation Layer CI** (`test-implementation.yml`)
   - Agent tests: `JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/unit/ -m "unit and ci_fast" -v --tb=long 2>&1 > /tmp/agent-unit-tests.log`
   - Routing tests with mocked LLMs
   - Vespa backend tests with lightweight Docker container
   - Synthetic data generation tests

4. **Application Layer CI** (`test-application.yml`)
   - Ingestion pipeline tests: `uv run pytest tests/ingestion/unit/ -m "unit and ci_safe" -v --tb=long 2>&1 > /tmp/ingestion-unit-ci-safe.log`
   - Integration tests with mocked dependencies
   - System integration tests

**Multi-Modal Testing in CI:**
- Video: Mocked frame extraction, ColPali embeddings
- Audio: Mocked Whisper transcription
- Images: Mocked image processing
- Documents: Mocked PDF extraction
- Text: Real text processing
- Dataframes: Real structured data processing

### Environment Variables
- `CI=true`: Automatically detected, excludes `local_only` tests
- `JAX_PLATFORM_NAME=cpu`: Required for DSPy tests in CI (avoids GPU initialization)
- `RUN_HEAVY_TESTS=1`: Override to include heavy models in CI
- `PYTEST_TIMEOUT=1800`: 30-minute timeout for integration tests

## Output and Reporting

### Generated Reports
- `comprehensive_test_results.csv`: Detailed per-query routing results
- `comprehensive_summary_report.json`: Aggregated metrics and rankings
- `htmlcov/index.html`: Coverage reports

### Performance Metrics
- **Overall Accuracy**: Combined routing + temporal correctness
- **Routing Accuracy**: Video/text/both classification accuracy
- **Temporal Accuracy**: Date/time extraction accuracy
- **Success Rate**: Percentage of successful inferences
- **Response Time**: Performance benchmarks

### Example Output
```
🏆 COMPREHENSIVE ROUTING TEST RESULTS
================================================================================

🎯 OVERALL RANKINGS (Total Queries: 25)
--------------------------------------------------------------------------------
 1. LLM: qwen3:8b                    92.5% (R:95.0% T:90.0% 1.23s)
 2. HYBRID: gliner_multi-v2.1        90.0% (R:92.0% T:88.0% 0.87s)
 3. LLM: gemma3:12b                  88.5% (R:90.0% T:87.0% 2.14s)

🥇 BEST PERFORMERS
--------------------------------------------------
Overall:  llm:qwen3:8b (92.5%)
Routing:  llm:qwen3:8b (95.0%)
Temporal: hybrid:gliner_multi-v2.1 (90.0%)
Fastest:  gliner:gliner_multi-v2.1 (0.65s)
```

## Development Guidelines

### Adding New Tests

#### Unit Tests
```python
@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_cv2
class TestKeyframeProcessor:
    def test_extract_keyframes(self):
        # Test implementation
        pass
```

#### Integration Tests
```python
@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_vespa
@pytest.mark.requires_inference("vllm_colpali")
class TestColPaliVespaIngestion:
    def test_full_ingestion_pipeline(self):
        # Heavy integration test
        pass
```

### Test Development Best Practices
- **Unit Tests**: Always mark with `@pytest.mark.unit` and `@pytest.mark.ci_safe`
- **Integration Tests**: Mark with appropriate backend and model requirements
- **Heavy Models**: Add `@pytest.mark.local_only` for tests using large models
- **Dependencies**: Add appropriate `@pytest.mark.requires_*` markers

### Skip Conditions
```python
from tests.utils.markers import skip_if_no_vespa, skip_if_ci

@skip_if_no_vespa
def test_vespa_specific_feature():
    # Only runs if Vespa is available
    pass

@skip_if_ci  
def test_local_development_only():
    # Only runs in local development
    pass
```

## Troubleshooting

### Common Issues

#### Tests Getting Skipped
```bash
# Check what's missing in your environment
uv run python scripts/test_ingestion.py --env-info

# Common fixes
./scripts/start_vespa.sh  # Start Vespa
pip install colpali-engine  # Install ColPali
# Ensure ../videoprism/ directory exists for VideoPrism
```

#### CI Failures
- Use `--ci-safe` marker for CI-compatible tests
- Heavy model tests should be marked `local_only`
- Check that mocks are properly configured for external dependencies

#### Coverage Issues
- Only unit tests count toward coverage (80%+ target)
- Integration tests provide functional validation
- Use `--coverage-fail-under=80` to enforce thresholds

#### Model Not Found
- **Routing**: Ensure Ollama models are installed and available
- **Ingestion**: Check that required model packages are installed
- **GLiNER**: Configure models in `config.json` under `available_gliner_models`

### Debug Commands
```bash
# Dry run to see what would execute
uv run python scripts/test_ingestion.py --integration --dry-run

# Simulate CI environment locally
CI=1 uv run python scripts/test_ingestion.py --unit 2>&1 > /tmp/ingestion-ci-unit.log

# Force run all tests (will show skips/failures)
uv run pytest tests/ingestion/integration/ -v --tb=long 2>&1 > /tmp/ingestion-integration-all.log

# Test specific file regardless of dependencies
uv run pytest tests/ingestion/integration/test_backend_ingestion.py::TestMockBackendIngestion -v --tb=long
```

## Configuration

### Test Queries
Edit `tests/routing/test_queries.txt`:
```
query_text, expected_routing, expected_temporal
Show me videos from yesterday, video, yesterday
Find documents about AI, text, none
Search content from 2024-01-15, both, 2024-01-15
```

### Model Configuration
Configure available models in `config.json`:
```json
{
  "local_llm_model": "qwen3:4b",
  "query_inference_engine": {
    "available_gliner_models": [
      "urchade/gliner_multi-v2.1",
      "urchade/gliner_large-v2.1"
    ]
  }
}
```

---

**🎉 Happy Testing!** This comprehensive test suite provides everything needed to evaluate and optimize the Cogniverse system across routing, ingestion, and video processing pipelines.
