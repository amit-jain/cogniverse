# Test Suite

Comprehensive test coverage for the Cogniverse system including routing, ingestion, evaluation, and integration tests.

## Quick Start

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# CI-safe tests only
uv run pytest tests/ -m ci_safe

# Local tests with GPU
uv run pytest tests/ -m "not ci_safe"
```

## Test Organization

```
tests/
├── routing/           # Query routing tests
│   ├── unit/         # Strategy unit tests
│   └── integration/  # Router integration tests
├── ingestion/        # Pipeline tests
│   ├── unit/         # Processor unit tests
│   └── integration/  # Backend integration tests
├── evaluation/       # Evaluation framework tests
└── utils/            # Test utilities and markers
```

## Test Categories

### Pytest Markers

#### Test Types
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests requiring external services
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.slow` - Tests > 5 seconds

#### Environment Markers
- `@pytest.mark.ci_safe` - Can run in CI environment
- `@pytest.mark.local_only` - Requires local resources

#### Backend Requirements
- `@pytest.mark.requires_vespa` - Needs Vespa running
- `@pytest.mark.requires_docker` - Needs Docker
- `@pytest.mark.requires_modal` - Needs Modal account
- `@pytest.mark.requires_ollama` - Needs Ollama service

#### Model Requirements
- `@pytest.mark.requires_colpali` - ColPali model
- `@pytest.mark.requires_videoprism` - VideoPrism model
- `@pytest.mark.requires_gliner` - GLiNER models
- `@pytest.mark.requires_llm` - LLM models

#### Resource Requirements
- `@pytest.mark.requires_gpu` - GPU required
- `@pytest.mark.high_memory` - > 8GB RAM needed

## Running Tests

### Unit Tests
```bash
# Fast unit tests only
uv run pytest tests/ -m unit -v

# Specific module
uv run pytest tests/routing/unit/ -v
uv run pytest tests/ingestion/unit/ -v
```

### Integration Tests
```bash
# All integration tests (requires services)
uv run pytest tests/ -m integration

# Skip slow tests
uv run pytest tests/ -m "integration and not slow"

# Vespa tests only
uv run pytest tests/ -m requires_vespa
```

### Coverage Reports
```bash
# HTML coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Check coverage thresholds
uv run pytest tests/ --cov=src --cov-fail-under=80
```

### Specific Test Scenarios

#### Routing Tests
```bash
# GLiNER strategies
uv run pytest tests/routing/ -k gliner

# LLM routing with specific model
ROUTING_LLM_MODEL=gemma2:2b uv run pytest tests/routing/ -k llm

# Tiered routing integration
uv run pytest tests/routing/integration/test_tiered_routing.py
```

#### Ingestion Tests
```bash
# Frame extraction
uv run pytest tests/ingestion/ -k frame_extraction

# Video processing (requires FFmpeg)
uv run pytest tests/ingestion/ -k video_processor

# Embedding generation
uv run pytest tests/ingestion/ -k embedding
```

#### Comprehensive Tests
```bash
# Test router comprehensive functionality
uv run python tests/routing/integration/test_router_comprehensive.py

# Test combined routing strategies
uv run python tests/routing/test_combined_routing.py
```

## Test Fixtures

### Common Fixtures

```python
# conftest.py provides:
- routing_config()      # Routing configuration
- mock_gliner_model()   # Mocked GLiNER model
- mock_ollama_response()# Mocked Ollama response
- test fixtures for various components
```

### Mock Dependencies

The test suite automatically mocks heavy dependencies in CI:
- OpenCV (cv2)
- Whisper
- PyTorch models
- Vespa connections
- Modal endpoints

### Environment Detection

Tests automatically detect available resources:
```python
# Auto-skips if dependencies missing
@pytest.mark.skipif(not has_ffmpeg(), reason="FFmpeg not installed")
@pytest.mark.skipif(not has_gpu(), reason="GPU not available")
@pytest.mark.skipif(not vespa_running(), reason="Vespa not running")
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv pip install -r requirements.txt
      - run: uv run pytest tests/ -m ci_safe --cov=src
```

### Environment Variables

```bash
# CI environment
export CI=true
export PYTEST_MARKERS="ci_safe"
export MOCK_HEAVY_DEPS=true

# Local development
export ENABLE_INTEGRATION_TESTS=true
export VESPA_HOST=localhost
export OLLAMA_HOST=localhost:11434
```

## Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
@pytest.mark.ci_safe
class TestMyComponent:
    """Test suite for MyComponent."""
    
    def test_basic_functionality(self, mock_config):
        """Test basic component behavior."""
        component = MyComponent(mock_config)
        result = component.process("input")
        assert result is not None
    
    @pytest.mark.integration
    @pytest.mark.requires_vespa
    async def test_vespa_integration(self, vespa_client):
        """Test Vespa backend integration."""
        # Test implementation
```

### Best Practices

1. **Use appropriate markers** - Help others run relevant tests
2. **Mock external dependencies** - Keep tests fast and reliable
3. **Test edge cases** - Empty inputs, errors, timeouts
4. **Maintain coverage** - Aim for 80%+ coverage
5. **Write descriptive names** - Clear test purpose
6. **Use fixtures** - Avoid duplication
7. **Test async properly** - Use pytest-asyncio
8. **Document complex tests** - Add docstrings

## Debugging Tests

### Verbose Output
```bash
# Show print statements
uv run pytest tests/ -v -s

# Show warnings
uv run pytest tests/ -v --tb=short
```

### Debug Specific Test
```bash
# Run single test
uv run pytest tests/routing/unit/test_gliner.py::test_entity_extraction -vv

# Debug with pdb
uv run pytest tests/ --pdb
```

### Check Test Discovery
```bash
# List all tests
uv run pytest tests/ --collect-only

# List tests with markers
uv run pytest tests/ --collect-only -m integration
```

## Performance Testing

### Benchmark Tests
```bash
# Run benchmarks
uv run pytest tests/ -m benchmark --benchmark-only

# Compare results
uv run pytest tests/ -m benchmark --benchmark-compare
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
# Ensure src is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Vespa connection errors**
```bash
# Check Vespa is running
docker ps | grep vespa
curl http://localhost:8080/ApplicationStatus
```

**Model download failures**
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"
```

**CI test failures**
```bash
# Run CI-safe tests locally
uv run pytest tests/ -m ci_safe --mock-heavy-deps
```

## Test Data

### Test Data Location
Test data is located in `data/testset/` with the following structure:
- `data/testset/evaluation/sample_videos/` - Sample video files for testing
- `data/testset/evaluation/video_search_queries.csv` - Test queries and expected results

### Using Test Utilities
```python
from tests.utils.markers import has_ffmpeg, has_gpu, vespa_running

# Check for available resources
if has_ffmpeg():
    # Run video processing tests
    pass
```

## Coverage Requirements

- **Overall**: 80% minimum
- **Core modules**: 85% minimum
- **New code**: 90% minimum
- **Critical paths**: 95% minimum

Monitor coverage trends:
```bash
# Generate coverage badge
coverage-badge -o coverage.svg

# Upload to codecov (in CI)
codecov --file coverage.xml
```