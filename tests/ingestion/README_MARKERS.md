# Ingestion Test Markers and Conditional Execution

This document explains the marker system for running ingestion tests conditionally based on available dependencies and environment.

## 🏷️ Available Markers

### Test Types
- `unit`: Unit tests for individual components  
- `integration`: Integration tests with multiple components
- `benchmark`: Performance benchmark tests
- `slow`: Tests that take significant time

### Environment Markers
- `ci_safe`: Tests safe to run in CI environment (lightweight, mocked)
- `local_only`: Tests that should only run locally (heavy models, real backends)

### Backend Requirements
- `requires_vespa`: Tests requiring Vespa backend to be running
- `requires_docker`: Tests requiring Docker
- `requires_cv2`: Tests requiring OpenCV
- `requires_ffmpeg`: Tests requiring FFmpeg

### Model Requirements  
- `requires_colpali`: Tests requiring ColPali models
- `requires_videoprism`: Tests requiring VideoPrism models (from adjacent directory)
- `requires_colqwen`: Tests requiring ColQwen models
- `requires_whisper`: Tests requiring Whisper models

### Resource Requirements
- `requires_gpu`: Tests requiring GPU availability
- `local_only`: Heavy model tests excluded from CI

## 🚀 Usage Examples

### Run CI-Safe Unit Tests Only
```bash
# Using pytest directly
pytest tests/ingestion/unit/ -m "unit and ci_safe" -v

# Using our test script
python scripts/test_ingestion.py --unit --ci-safe
```

### Run Local Integration Tests with Heavy Models
```bash
# Local-only tests with all models
python scripts/test_ingestion.py --integration --local-only

# Specific model tests  
python scripts/test_ingestion.py --requires-colpali
python scripts/test_ingestion.py --requires-videoprism
```

### Run Vespa Backend Tests
```bash
# Only if Vespa is running
pytest tests/ingestion/ -m "requires_vespa" -v

# Or with our script
python scripts/test_ingestion.py --requires-vespa
```

### Exclude Heavy Models (CI Default)
```bash
# Automatically excludes local_only tests in CI
python scripts/test_ingestion.py --exclude-heavy

# Force include heavy tests
python scripts/test_ingestion.py --include-heavy
```

## 🔍 Environment Detection

The test system automatically detects:

### Available Dependencies
- ✅/❌ Docker availability
- ✅/❌ Vespa backend running (localhost:8080)
- ✅/❌ FFmpeg installation
- ✅/❌ OpenCV (cv2) availability
- ✅/❌ Whisper availability

### Available Models
- ✅/❌ ColPali (colpali_engine module)
- ✅/❌ VideoPrism (videoprism module in adjacent directory + 8GB memory)
- ✅/❌ ColQwen (transformers + 8GB memory)  
- ✅/❌ Whisper (whisper module)

### Environment Type
- CI vs Local development
- GPU availability
- Memory sufficiency (8GB+ for heavy models)

## 📊 Test Coverage

### Current Status
- **AudioProcessor**: 99% coverage (67/67 statements, 1 miss)
- **ChunkProcessor**: 100% coverage (67/67 statements)  
- **KeyframeProcessor**: 98% coverage (97/97 statements, 2 miss)

### Running Coverage Tests
```bash
# Unit tests with coverage threshold
python scripts/test_ingestion.py --unit --coverage-fail-under=80

# Integration tests don't count toward coverage
python scripts/test_ingestion.py --integration --ci-safe
```

## 🎯 GitHub Actions Integration

The `.github/workflows/test-ingestion.yml` workflow uses markers to:

1. **Unit Tests Job**: Run `unit and ci_safe` tests with 80% coverage requirement
2. **Integration Tests Job**: Run `integration and ci_safe` with lightweight Vespa container
3. **Security/Lint Jobs**: Static analysis and code quality checks

### CI Environment Variables
- `CI=true`: Automatically detected, excludes `local_only` tests
- `RUN_HEAVY_TESTS=1`: Override to include heavy models in CI

## 🛠️ Test Development Guidelines

### Adding New Tests

1. **Unit Tests**: Always mark with `@pytest.mark.unit` and `@pytest.mark.ci_safe`
2. **Integration Tests**: Mark with `@pytest.mark.integration` and appropriate backend markers
3. **Heavy Models**: Add `@pytest.mark.local_only` for tests using large models
4. **Dependencies**: Add appropriate `@pytest.mark.requires_*` markers

### Example Test Class
```python
@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_cv2
class TestKeyframeProcessor:
    def test_extract_keyframes(self):
        # Test implementation
        pass

@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_vespa
@pytest.mark.requires_colpali
class TestColPaliVespaIngestion:
    def test_full_ingestion_pipeline(self):
        # Heavy integration test
        pass
```

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

## 🔧 Local Development

### Check Your Environment
```bash
# See what's available in your environment
python scripts/test_ingestion.py --env-info
```

### Test Different Scenarios
```bash
# Simulate CI environment
CI=1 python scripts/test_ingestion.py --unit

# Test with specific models
python scripts/test_ingestion.py --requires-colpali --verbose

# Dry run to see what would execute  
python scripts/test_ingestion.py --integration --dry-run
```

## 📋 Common Issues

### Tests Getting Skipped
- Check environment with `--env-info`
- Ensure required dependencies are installed
- For Vespa tests, make sure Vespa is running on localhost:8080

### CI Failures
- Use `--ci-safe` marker for CI-compatible tests
- Heavy model tests should be marked `local_only`
- Check that mocks are properly configured for external dependencies

### Coverage Issues
- Only unit tests count toward coverage
- Integration tests provide functional validation
- Target 80%+ coverage on processor modules