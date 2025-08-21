# Ingestion Module Test Suite

This directory contains comprehensive unit and integration tests for the video ingestion pipeline.

## Structure

```
tests/ingestion/
├── unit/                           # Unit tests for individual components
│   ├── test_processor_base.py     # Base processor and strategy tests ✅
│   ├── test_processor_manager.py  # Processor manager tests 🔧
│   ├── test_keyframe_processor.py # Keyframe extraction tests 🔧  
│   ├── test_audio_processor.py    # Audio transcription tests 🔧
│   └── test_chunk_processor.py    # Video chunking tests 🔧
├── integration/                    # Integration tests
│   ├── test_pipeline_orchestration.py  # Pipeline coordination tests ✅
│   └── test_end_to_end_processing.py   # Complete pipeline tests ✅
├── fixtures/                       # Test fixtures and mocks
│   └── mock_dependencies.py       # Mock implementations ✅
└── conftest.py                     # Shared test fixtures ✅
```

## Test Coverage

### ✅ Completed
- **Base Infrastructure**: Processor base classes, strategy interfaces
- **Test Framework**: Fixtures, mocks, and test utilities
- **Integration Architecture**: Pipeline orchestration and end-to-end tests
- **CI/CD Integration**: GitHub Actions workflow for automated testing

### 🔧 In Progress  
- **Individual Processor Tests**: Need alignment with actual implementation APIs
- **Mock Refinement**: Some mocks need adjustment to match real interfaces

## Key Features

### Unit Tests
- **Processor Base Classes**: Tests for `BaseProcessor` and `BaseStrategy` abstract classes
- **Factory Methods**: Configuration-based processor instantiation
- **Error Handling**: Proper exception handling and validation
- **Caching**: Processor caching and lifecycle management

### Integration Tests  
- **Pipeline Orchestration**: Multi-processor coordination
- **Strategy-driven Processing**: Dynamic processor configuration from strategies
- **End-to-end Workflows**: Complete video processing pipelines
- **Error Propagation**: Failure handling across pipeline stages

### Test Infrastructure
- **Mock Dependencies**: OpenCV, Whisper, PyTorch, Vespa client mocks
- **Test Data Generation**: Synthetic video files and expected outputs
- **Async Testing**: Support for async processor operations
- **Performance Testing**: Benchmark markers for performance-critical code

## Running Tests

### Unit Tests Only
```bash
uv run python -m pytest tests/ingestion/unit -v --cov=src/app/ingestion
```

### Integration Tests Only  
```bash
uv run python -m pytest tests/ingestion/integration -v -m integration
```

### All Ingestion Tests
```bash
uv run python -m pytest tests/ingestion -v --cov=src/app/ingestion --cov-report=html
```

### Specific Test Categories
```bash
# Fast tests only
uv run python -m pytest tests/ingestion -v -m "not slow"

# Performance benchmarks
uv run python -m pytest tests/ingestion -v -m benchmark --benchmark-only
```

## Test Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests with multiple components  
- `slow`: Tests that take significant time to run
- `benchmark`: Performance benchmark tests
- `ingestion`: All ingestion-related tests

## GitHub Actions Integration

The test suite is integrated with GitHub Actions via `.github/workflows/ingestion-tests.yml`:

- **Unit Tests**: Run on every push/PR with 80% coverage requirement
- **Integration Tests**: Multi-stage pipeline testing
- **Linting**: Code quality checks with ruff, black, isort
- **Performance**: Benchmark tracking for performance regression detection
- **Coverage Reports**: Detailed HTML coverage reports as artifacts

## Development Workflow

1. **Write Tests First**: Follow TDD practices for new processors
2. **Run Locally**: Use pytest markers to run specific test categories
3. **Check Coverage**: Maintain >80% coverage for new code  
4. **CI Validation**: All tests must pass in GitHub Actions before merge

## Mock Strategy

The test suite uses comprehensive mocking to avoid external dependencies:

- **OpenCV**: Mock video capture and image processing
- **Whisper**: Mock audio transcription with deterministic outputs
- **PyTorch/Models**: Mock embedding generation and model loading
- **File I/O**: Mock file system operations with temporary directories
- **Caching**: Mock cache managers with in-memory storage

This ensures tests are:
- **Fast**: No heavy model loading or video processing
- **Deterministic**: Consistent results across environments
- **Isolated**: No external service dependencies
- **Comprehensive**: Full coverage of code paths without side effects

## Next Steps

1. **Align Mock Interfaces**: Update processor tests to match actual implementation APIs
2. **Add Performance Tests**: Benchmark critical video processing operations
3. **Expand Coverage**: Add tests for error conditions and edge cases
4. **Documentation**: Add docstrings and examples for complex test scenarios