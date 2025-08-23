# Cogniverse Test Suite - Comprehensive Guide

## Overview

This is a comprehensive test suite for the Cogniverse multi-agent RAG system, covering routing, ingestion, and video processing pipelines. The test suite is designed with environment-aware execution, smart dependency detection, and CI/CD integration.

## Quick Start

### Core Test Commands
```bash
# Install test dependencies
uv pip install pytest pytest-asyncio pytest-cov

# Run all tests
uv run python run_tests.py

# Run comprehensive routing tests (recommended)
python tests/test_comprehensive_routing.py

# Run ingestion tests with environment detection
python scripts/test_ingestion.py --integration
```

## Test Organization

### Directory Structure
```
tests/
├── routing/                    # Query routing system tests
│   ├── unit/                  # Routing strategy unit tests
│   ├── integration/           # Tiered routing integration tests
│   └── demo_routing_tiers.py  # Interactive demonstration
├── ingestion/                 # Video ingestion pipeline tests
│   ├── unit/                  # Individual component tests
│   ├── integration/           # Backend and pipeline tests
│   ├── utils/markers.py       # Smart environment detection
│   └── fixtures/              # Test fixtures and mocks
└── comprehensive/             # Cross-system integration tests
```

## Test Categories

### 1. Routing System Tests

#### Unit Tests (`tests/routing/unit/`)
- **GLiNER Strategy**: Named entity recognition routing
- **LLM Strategy**: Language model-based routing
- **Keyword Strategy**: Simple keyword-based routing
- **LangExtract Strategy**: Language extraction routing

#### Integration Tests (`tests/routing/integration/`)
- **Tiered Routing**: Multi-tier escalation logic
- **Generation Type Classification**: Video vs text vs both
- **Search Modality Detection**: Content type detection
- **Caching Behavior**: Route caching mechanisms
- **Error Handling**: Failure recovery patterns

#### Comprehensive Testing
```bash
# Test all routing models and strategies
python tests/test_comprehensive_routing.py

# Test specific components
python tests/test_comprehensive_routing.py llm-only      # LLM models only
python tests/test_comprehensive_routing.py gliner-only  # GLiNER models only
python tests/test_comprehensive_routing.py hybrid-only  # Combined approaches
python tests/test_comprehensive_routing.py quick        # Fast subset
```

**Models Tested:**
- **DeepSeek R1**: 1.5b, 7b, 8b variants
- **Gemma 3**: 1b, 4b, 12b variants
- **Qwen 3**: 0.6b, 1.7b, 4b, 8b variants
- **GLiNER**: All configured models in config.json

### 2. Ingestion Pipeline Tests

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
python scripts/test_ingestion.py --env-info

# Run CI-safe tests (mocked dependencies)
python scripts/test_ingestion.py --unit --ci-safe

# Run local-only tests (requires heavy models)
python scripts/test_ingestion.py --integration --local-only

# Run specific backend tests
python scripts/test_ingestion.py --requires-vespa
python scripts/test_ingestion.py --requires-colpali
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

### Model Requirements
- `requires_colpali`: Tests requiring ColPali models
- `requires_videoprism`: Tests requiring VideoPrism models
- `requires_colqwen`: Tests requiring ColQwen models
- `requires_whisper`: Tests requiring Whisper models

### Resource Requirements
- `requires_gpu`: Tests requiring GPU availability
- `local_only`: Heavy model tests excluded from CI

## Running Tests

### Unit Tests
```bash
# CI-safe unit tests with coverage
uv run pytest tests/ingestion/unit/test_*_real.py -v --cov=src/app/ingestion/processors

# Routing unit tests
uv run pytest tests/routing/unit/ -m "unit" -v

# With coverage reporting
uv run pytest --cov=src/routing --cov-report=html
```

### Integration Tests
```bash
# Mock backend tests (always available)
python scripts/test_ingestion.py --integration --ci-safe

# Vespa backend tests (requires Vespa running)
./scripts/start_vespa.sh
python scripts/test_ingestion.py --requires-vespa

# Heavy model tests (local development only)
python scripts/test_ingestion.py --integration --local-only

# Routing integration tests
uv run pytest tests/routing/integration/ -v
```

### Specific Test Scenarios
```bash
# Test specific marker combinations
pytest tests/ingestion/ -m "integration and requires_vespa" -v
pytest tests/ingestion/ -m "unit and ci_safe" -v
pytest tests/ingestion/ -m "integration and local_only" -v

# Test with coverage threshold
python scripts/test_ingestion.py --unit --coverage-fail-under=80

# Exclude heavy models (CI default)
python scripts/test_ingestion.py --exclude-heavy
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

#### Ingestion Tests (`test-ingestion.yml`)
1. **Unit Tests Job**: Run `unit and ci_safe` tests with 80% coverage requirement
2. **Integration Tests Job**: Run `integration and ci_safe` with lightweight Vespa container
3. **Security/Lint Jobs**: Static analysis and code quality checks

#### Routing Tests
- Mocked dependencies for unit tests
- Real model testing for integration (when available)
- Async test support with proper fixtures

### Environment Variables
- `CI=true`: Automatically detected, excludes `local_only` tests
- `RUN_HEAVY_TESTS=1`: Override to include heavy models in CI

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
@pytest.mark.requires_colpali
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
python scripts/test_ingestion.py --env-info

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
python scripts/test_ingestion.py --integration --dry-run

# Simulate CI environment locally
CI=1 python scripts/test_ingestion.py --unit

# Force run all tests (will show skips/failures)
pytest tests/ingestion/integration/ -v --tb=short

# Test specific file regardless of dependencies
pytest tests/ingestion/integration/test_backend_ingestion.py::TestMockBackendIngestion -v
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