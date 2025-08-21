# Test Marker Examples and Usage

This document provides concrete examples of how the marker system works in different scenarios.

## 🔍 **Environment Detection Examples**

### **Local Development Environment**
```bash
$ uv run python scripts/test_ingestion.py --env-info

🔍 Test Environment Information:
==================================================
ci_environment: ❌  # No CI env vars (CI, GITHUB_ACTIONS, etc.)
docker_available: ✅  # `docker --version` succeeds
vespa_running: ✅  # HTTP 200 from localhost:8080/ApplicationStatus
ffmpeg_available: ✅  # `ffmpeg -version` succeeds
gpu_available: ❌  # torch.cuda.is_available() = False (local Mac)
sufficient_memory: ✅  # psutil shows 16GB+ RAM
available_models:
  colpali: ✅  # import colpali_engine works
  videoprism: ✅  # ../videoprism/videoprism/__init__.py exists
  colqwen: ✅  # import transformers works + memory check
  whisper: ✅  # import whisper works
  cv2: ✅  # import cv2 works
==================================================
```

### **GitHub Actions CI Environment**
```bash
# Environment automatically detected from CI=true
🔍 Test Environment Information:
==================================================
ci_environment: ✅  # CI=true environment variable set
docker_available: ✅  # Pre-installed in GitHub runners
vespa_running: ✅  # Lightweight container started in workflow
ffmpeg_available: ✅  # Pre-installed in Ubuntu runner
gpu_available: ❌  # Standard runners have no GPU
sufficient_memory: ✅  # 7GB available in standard runner
available_models:
  colpali: ❌  # Heavy model packages not installed in CI
  videoprism: ❌  # Adjacent directory not available in CI
  colqwen: ❌  # Heavy transformer models not in CI
  whisper: ✅  # Lightweight whisper package installed
  cv2: ✅  # OpenCV available in CI
==================================================
```

## 🎯 **Test Execution Examples**

### **Example 1: CI-Safe Unit Tests**

**Command:**
```bash
uv run python scripts/test_ingestion.py --unit --ci-safe
```

**Generated pytest command:**
```bash
pytest tests/ingestion/unit/ -m "unit and ci_safe" -v
```

**Tests that run:**
```python
@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_cv2
class TestKeyframeProcessor:  # ✅ Runs with mocked OpenCV

@pytest.mark.unit
@pytest.mark.ci_safe  
@pytest.mark.requires_whisper
class TestAudioProcessor:  # ✅ Runs with mocked Whisper

@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_ffmpeg
class TestChunkProcessor:  # ✅ Runs with mocked FFmpeg
```

**Tests that are skipped:**
```python
@pytest.mark.local_only
class TestHeavyModel:  # ❌ Skipped - local_only in CI environment

@pytest.mark.requires_colpali
class TestColPaliModel:  # ❌ Skipped - ColPali not available in CI
```

### **Example 2: Local VideoPrism Tests**

**Command:**
```bash
uv run python scripts/test_ingestion.py --requires-videoprism
```

**Environment Check:**
```python
# Tests/utils/markers.py logic:
def is_videoprism_available():
    # Check: ../videoprism/videoprism/__init__.py exists
    project_root = Path(__file__).parent.parent.parent  # cogniverse/
    videoprism_path = project_root.parent / "videoprism"  # ../videoprism/
    
    if videoprism_path.exists() and (videoprism_path / "videoprism" / "__init__.py").exists():
        # Add to Python path and test import
        sys.path.insert(0, str(videoprism_path))
        try:
            import videoprism  # ✅ Success
            return True
        finally:
            sys.path.remove(str(videoprism_path))
```

**Tests that run:**
```python
@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_videoprism
class TestVideoPrismVespaIngestion:  # ✅ Runs - VideoPrism available locally
    def test_videoprism_vespa_ingestion(self):
        # Real VideoPrism model + Vespa backend test
        pass
```

### **Example 3: Vespa Integration Tests**

**Command:**
```bash
uv run python scripts/test_ingestion.py --requires-vespa
```

**Environment Check:**
```python
def is_vespa_running():
    try:
        response = requests.get('http://localhost:8080/ApplicationStatus', timeout=5)
        return response.status_code == 200  # ✅ Vespa running
    except requests.RequestException:
        return False  # ❌ Vespa not running
```

**Tests that run:**
```python
@pytest.mark.integration
@pytest.mark.requires_vespa
class TestVespaBackendIngestion:
    def test_vespa_connection(self):  # ✅ Basic connectivity test
        
    def test_lightweight_vespa_ingestion(self):  # ✅ No models, just indexing
```

### **Example 4: Comprehensive Local Testing**

**Command:**
```bash
uv run python scripts/test_ingestion.py --integration --local-only
```

**Tests that run (all dependencies available):**
```python
@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_vespa
@pytest.mark.requires_colpali
class TestColPaliVespaIngestion:  # ✅ Runs - all deps available

@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_vespa
@pytest.mark.requires_videoprism  
class TestVideoPrismVespaIngestion:  # ✅ Runs - VideosPrism from ../videoprism/

@pytest.mark.integration
@pytest.mark.local_only
@pytest.mark.requires_colqwen
class TestColQwenIngestion:  # ✅ Runs - transformers available
```

## 🚫 **Skip Logic Examples**

### **Automatic Skipping in CI**

```python
# This test automatically skips in CI environments:
@pytest.mark.local_only  # ← Key marker
@pytest.mark.requires_colpali
def test_colpali_heavy_ingestion():
    # This never runs in GitHub Actions, even if ColPali was installed
    pass

# Reason: CI environment detected → local_only tests skipped
```

### **Dependency-Based Skipping**

```python
# Local environment where ColPali is NOT installed:
@pytest.mark.requires_colpali
def test_colpali_functionality():
    # ❌ SKIPPED: colpali_engine module not found
    pass

# Local environment where VideoPrism directory doesn't exist:
@pytest.mark.requires_videoprism
def test_videoprism_functionality():
    # ❌ SKIPPED: ../videoprism/videoprism/__init__.py not found
    pass
```

### **Resource-Based Skipping**

```python
# System with only 4GB RAM:
@pytest.mark.requires_videoprism  # Requires 8GB+ memory
def test_videoprism_ingestion():
    # ❌ SKIPPED: Insufficient memory (requires 8GB+)
    pass
```

## 🔧 **Override Examples**

### **Force Heavy Tests in CI**
```yaml
# In GitHub Actions workflow:
env:
  RUN_HEAVY_TESTS: "1"  # Override local_only skipping
```

```python
# Test logic:
@pytest.mark.local_only
def test_heavy_model():
    if os.getenv('RUN_HEAVY_TESTS'):
        # ✅ Runs in CI with override
        run_heavy_test()
    else:
        pytest.skip("Heavy test skipped in CI")
```

### **Simulate CI Locally**
```bash
# Force CI-like behavior on local machine:
CI=1 uv run python scripts/test_ingestion.py --unit

# Result: Skips local_only tests even though dependencies available
```

## 📊 **Marker Combination Examples**

### **Multiple Required Dependencies**
```python
@pytest.mark.integration
@pytest.mark.requires_vespa
@pytest.mark.requires_colpali
def test_colpali_vespa_search():
    # Runs ONLY if: Vespa running AND ColPali available
    pass
```

### **Exclude Specific Tests**
```bash
# Run all tests except heavy models:
pytest -m "not local_only"

# Run only unit tests that are CI-safe:
pytest -m "unit and ci_safe"

# Run integration tests but exclude GPU-dependent ones:
pytest -m "integration and not requires_gpu"
```

## 📁 **Directory Structure Impact**

```
/Users/amjain/source/hobby/
├── cogniverse/           # Main project
│   ├── tests/
│   │   ├── utils/
│   │   │   └── markers.py  # Detection logic here
│   └── scripts/
│       └── test_ingestion.py
└── videoprism/           # Adjacent VideoPrism module
    ├── setup.py
    └── videoprism/
        ├── __init__.py   # ← This file detected
        ├── models.py
        └── encoders.py
```

**Detection Logic:**
```python
# From cogniverse/tests/utils/markers.py:
project_root = Path(__file__).parent.parent.parent  # → cogniverse/
videoprism_path = project_root.parent / "videoprism"  # → ../videoprism/

# Checks: ../videoprism/videoprism/__init__.py exists
if (videoprism_path / "videoprism" / "__init__.py").exists():
    # ✅ VideoPrism available
```

This structure allows the test system to automatically detect and use the VideoPrism module when developing locally, while gracefully skipping those tests in CI where the module isn't available.