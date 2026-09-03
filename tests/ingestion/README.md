# Ingestion Module Test Suite

**Last Updated:** 2025-11-13

This directory contains comprehensive unit and integration tests for **cogniverse-runtime** (Application Layer) multi-modal ingestion pipeline with smart environment-aware execution. Tests validate processing of all content modalities: video, audio, images, documents, text, and dataframes with ColPali, X-CLIP, ColQwen embeddings and Vespa backend integration.

## Structure

```
tests/ingestion/
├── unit/                           # Unit tests for individual components
│   ├── test_*_real.py             # Working tests with real APIs ✅ 80%+ coverage
│   ├── test_processor_base.py     # Base processor and strategy tests ✅
│   ├── test_processor_manager.py  # Processor manager tests 🔧
│   ├── test_keyframe_processor.py # Legacy keyframe tests 🔧  
│   ├── test_audio_processor.py    # Legacy audio tests 🔧
│   └── test_chunk_processor.py    # Legacy chunking tests 🔧
├── integration/                    # Integration tests with real backends
│   ├── test_backend_ingestion.py     # Mock/Vespa backend tests ✅
│   ├── test_pipeline_orchestration.py  # Pipeline coordination tests ✅
│   └── test_end_to_end_processing.py   # Complete pipeline tests ✅
├── utils/                          # Test utilities and markers
│   ├── __init__.py                # Utils package init
│   └── markers.py                 # Smart environment detection ✅
├── fixtures/                       # Test fixtures and mocks
│   └── mock_dependencies.py       # Mock implementations ✅
├── README_MARKERS.md              # Marker system documentation ✅
├── MARKER_EXAMPLES.md             # Detailed usage examples ✅
└── conftest.py                     # Shared test fixtures ✅
```

## Test Coverage

### ✅ Completed (80%+ Coverage Achieved!)
- **Core Processor Tests**: AudioProcessor (99%), ChunkProcessor (100%), KeyframeProcessor (98%)
- **Smart Environment Detection**: Automatic dependency and model detection
- **Conditional Test Execution**: CI-safe vs local-only test separation
- **Integration Backend Tests**: Mock, Vespa, and real model ingestion tests
- **GitHub Actions Workflow**: Multi-stage CI with proper dependency handling

### 🔧 Legacy Tests (Being Phased Out)
- **Old Processor Tests**: test_*_processor.py files (non-functional, API mismatches)
- **Mock Refinement**: Some integration mocks need real pipeline integration

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

## 🚀 Running Tests Locally

We provide a smart test runner that automatically detects your environment and runs appropriate tests.

### **Quick Start**
```bash
# Check what's available in your environment
uv run python scripts/test_ingestion.py --env-info

# Run all available tests
uv run python scripts/test_ingestion.py --integration 2>&1 > /tmp/ingestion-integration.log
```

### **Unit Tests (80%+ Coverage)**

**Use UV workspace for all tests:**

```bash
# CI-safe unit tests (always work)
uv run python scripts/test_ingestion.py --unit --ci-safe 2>&1 > /tmp/ingestion-unit-ci-safe.log

# Direct pytest for unit tests with UV workspace
uv run pytest tests/ingestion/unit/test_*_real.py -v --tb=long --cov=libs/runtime/cogniverse_runtime/ingestion 2>&1 > /tmp/ingestion-unit-real.log
```

### **Integration Tests (Environment-Aware)**

#### **1. Mock Backend Tests (Always Available)**
```bash
# Lightweight integration tests with mocked dependencies
uv run python scripts/test_ingestion.py --integration --ci-safe 2>&1 > /tmp/ingestion-integration-ci-safe.log
```

#### **2. Vespa Backend Tests (Requires Vespa Running)**
```bash
# Start Vespa first
./scripts/start_vespa.sh

# Run Vespa integration tests
uv run python scripts/test_ingestion.py --requires-vespa 2>&1 > /tmp/ingestion-vespa.log
```

#### **3. Exact Inference-Service Tests**

Inference-backed tests name the exact service they require. ColPali and ColQwen
share the `vllm_colpali` embedding service, while X-CLIP and ASR use their
own services:

```python
@pytest.mark.requires_inference("vllm_colpali")
def test_colpali_or_colqwen_ingestion(): ...


@pytest.mark.requires_inference("video_embed")
def test_xclip_ingestion(): ...


@pytest.mark.requires_inference("vllm_asr")
def test_remote_asr(): ...
```

Run an annotated test by node instead of selecting a removed model-specific
runner flag:

```bash
uv run pytest tests/ingestion/integration/test_backend_ingestion.py::TestVespaBackendIngestion::test_colpali_vespa_ingestion -v --tb=long
```

#### **4. Heavy Model Tests (Local Development Only)**
```bash
# All available models with real document ingestion
uv run python scripts/test_ingestion.py --integration --local-only 2>&1 > /tmp/ingestion-local-models.log

# Combined: Vespa + all available models
uv run python scripts/test_ingestion.py --local-only --requires-vespa 2>&1 > /tmp/ingestion-local-vespa.log
```

### **Direct pytest Commands**
```bash
# All integration tests (may skip based on dependencies)
uv run pytest tests/ingestion/integration/ -v --tb=long 2>&1 > /tmp/ingestion-integration-pytest.log

# Specific test classes
uv run pytest tests/ingestion/integration/test_backend_ingestion.py::TestVespaBackendIngestion -v --tb=long

# With markers
uv run pytest tests/ingestion/integration/ -m "integration and requires_vespa" -v --tb=long 2>&1 > /tmp/ingestion-integration-vespa.log

# Local-only tests
uv run pytest tests/ingestion/integration/ -m "integration and local_only" -v --tb=long 2>&1 > /tmp/ingestion-integration-local.log
```

### **Environment Detection Examples**

Your environment will be automatically detected:

**Local Development:**
```
🔍 Test Environment Information:
==================================================
ci_environment: ❌  # Running locally
docker_available: ✅
vespa_running: ✅  # Started with ./scripts/start_vespa.sh
ffmpeg_available: ✅
available_models:
  colpali: ✅  # pip install colpali-engine
  xclip: ✅  # ../xclip/ directory exists
  colqwen: ✅  # transformers package available
  whisper: ✅  # whisper package available
  cv2: ✅  # opencv-python available
==================================================
```

**CI Environment:**
```
🔍 Test Environment Information:
==================================================
ci_environment: ✅  # GitHub Actions
vespa_running: ✅  # Lightweight container in CI
available_models:
  colpali: ❌  # Heavy models not installed in CI
  xclip: ❌  # Adjacent directory not in CI
  colqwen: ❌  # Heavy models not installed in CI
  whisper: ✅  # Lightweight whisper in CI
  cv2: ✅  # OpenCV available in CI
==================================================
```

### **Troubleshooting Integration Tests**

**Tests being skipped?**
```bash
# Check what's missing
uv run python scripts/test_ingestion.py --env-info

# Common fixes:
./scripts/start_vespa.sh  # Start Vespa
pip install colpali-engine  # Install ColPali
# Ensure ../xclip/ directory exists for X-CLIP
```

**Want to run specific tests regardless of dependencies?**
```bash
# Run specific test file (may fail if deps missing)
uv run pytest tests/ingestion/integration/test_backend_ingestion.py::TestMockBackendIngestion -v --tb=long

# Force run all tests (will show skips/failures)
uv run pytest tests/ingestion/integration/ -v --tb=long 2>&1 > /tmp/ingestion-integration-all.log
```

## 🏷️ Smart Test Markers

Our test system uses intelligent markers for conditional execution:

### **Test Types**
- `unit`: Unit tests for individual components
- `integration`: Integration tests with multiple components  
- `slow`: Tests that take significant time to run
- `benchmark`: Performance benchmark tests

### **Environment Markers**
- `ci_safe`: Tests safe to run in CI (lightweight, mocked)
- `local_only`: Tests that should only run locally (heavy models, real backends)

### **Backend Requirements**
- `requires_vespa`: Tests requiring Vespa backend running
- `requires_docker`: Tests requiring Docker
- `requires_cv2`: Tests requiring OpenCV
- `requires_ffmpeg`: Tests requiring FFmpeg

### **Inference Service Requirements**
- `requires_inference("vllm_colpali")`: ColPali and ColQwen embedding tests
- `requires_inference("video_embed")`: X-CLIP embedding tests
- `requires_inference("vllm_asr")`: Remote ASR tests
- `requires_whisper`: Local Whisper model tests

### **Resource Requirements**
- `requires_gpu`: Tests requiring GPU availability
- `local_only`: Heavy model tests excluded from CI

**📖 For detailed marker documentation, see [README_MARKERS.md](README_MARKERS.md) and [MARKER_EXAMPLES.md](MARKER_EXAMPLES.md)**

## 🤖 GitHub Actions Integration

The test suite is integrated with GitHub Actions via `.github/workflows/test-ingestion.yml`:

### **CI Test Strategy**
- **Unit Tests**: Run `unit and ci_safe` tests with 80% coverage requirement
- **Integration Tests**: Run `integration and ci_safe` with lightweight Vespa container
- **Security**: Bandit security scan and safety dependency checks
- **Code Quality**: ruff, black, mypy validation
- **Multi-Python**: Test against Python 3.11 and 3.12

### **What Runs in CI vs Local**

**✅ CI Runs (Fast, Lightweight)**
```python
@pytest.mark.unit
@pytest.mark.ci_safe  # ← Key marker
class TestAudioProcessor:  # Mocked Whisper, no model loading

@pytest.mark.integration  
@pytest.mark.ci_safe  # ← Key marker
class TestMockBackendIngestion:  # Mocked backends, no heavy models
```

**❌ CI Skips (Heavy, Local-Only)**  
```python
@pytest.mark.local_only  # ← Automatically skipped in CI
@pytest.mark.requires_inference("vllm_colpali")
class TestColPaliVespaIngestion:  # Real model + backend integration

@pytest.mark.local_only  # ← Automatically skipped in CI
@pytest.mark.requires_inference("video_embed")
class TestX-CLIPIngestion:  # Exact X-CLIP service required
```

### **CI Workflow Stages**
1. **Unit Tests**: Fast mocked tests with coverage reporting
2. **Integration Tests**: Lightweight backend integration with Docker Vespa
3. **Security Scan**: Static analysis and dependency vulnerability checks
4. **Code Quality**: Linting and type checking
5. **Test Summary**: Aggregate results and failure reporting

## 🔄 Development Workflow

### **For New Processors**
1. **Write Tests First**: Create both unit and integration tests
2. **Use Real APIs**: Test against actual implementation, not assumptions
3. **Add Proper Markers**: Use `@pytest.mark.unit`, `@pytest.mark.ci_safe`, etc.
4. **Check Coverage**: Aim for 80%+ coverage with meaningful tests
5. **Test Locally**: Use `uv run python scripts/test_ingestion.py --env-info` to check environment

### **For Integration Tests**  
1. **Mock for CI**: Use `@pytest.mark.ci_safe` for lightweight CI tests
2. **Real for Local**: Use `@pytest.mark.local_only` for heavy model tests  
3. **Dependency Markers**: Add `@pytest.mark.requires_*` for specific requirements
4. **Test Backends**: Ensure tests work with both mock and real backends

### **Before Committing**
```bash
# Run the working unit tests
uv run python scripts/test_ingestion.py --unit --ci-safe 2>&1 > /tmp/ingestion-unit-ci-safe.log

# Test integration if Vespa available  
uv run python scripts/test_ingestion.py --integration --ci-safe 2>&1 > /tmp/ingestion-integration-ci-safe.log

# Check what CI will run
CI=1 uv run python scripts/test_ingestion.py --unit --dry-run
```

## 🎭 Mock Strategy

The test suite uses **smart mocking** that adapts to the environment:

### **Unit Tests (Always Mocked)**
- **OpenCV**: Mock `cv2.VideoCapture`, `cv2.imwrite`, histogram functions
- **Whisper**: Mock `whisper.load_model`, transcription with deterministic outputs
- **FFmpeg**: Mock `subprocess.run` for ffprobe/ffmpeg calls  
- **File I/O**: Mock file operations with temporary directories
- **Output Manager**: Mock directory creation and path management

### **Integration Tests (Conditional)**
- **CI Environment**: Everything mocked, lightweight Vespa container
- **Local Environment**: Real models, real Vespa, actual document ingestion

### **Benefits**
- ✅ **Fast CI**: Unit tests complete in seconds
- ✅ **Comprehensive Local**: Full model integration testing  
- ✅ **Deterministic**: Consistent results across environments
- ✅ **Isolated**: No external service dependencies in CI
- ✅ **Real Coverage**: Tests actual implementation code paths

## 📋 Current Status & Next Steps

### **✅ Completed**
- [x] 80%+ unit test coverage for core processors
- [x] Smart environment detection and conditional execution
- [x] Integration tests with mock and real backends
- [x] GitHub Actions CI workflow with proper separation
- [x] X-CLIP detection from adjacent directory structure
- [x] Comprehensive documentation and examples

### **🔄 Future Enhancements**
- [ ] **Performance Benchmarks**: Add benchmark tests for video processing operations
- [ ] **End-to-End Tests**: Complete pipeline tests with real video files
- [ ] **Stress Testing**: High-volume ingestion testing
- [ ] **Error Recovery**: Test pipeline resilience and error handling
- [ ] **Memory Profiling**: Monitor memory usage during heavy model tests

### **📚 Documentation**
- `README_MARKERS.md`: Complete marker system documentation
- `MARKER_EXAMPLES.md`: Detailed usage examples and scenarios
- `.github/workflows/test-ingestion.yml`: CI workflow configuration  
- `scripts/test_ingestion.py`: Smart test runner with environment detection
