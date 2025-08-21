# Multi-Module Testing Architecture: Evaluation, Ingestion & Routing

This document explains how unit and integration tests interplay across the different modules (evaluation, ingestion, routing) and how coverage is calculated in this multi-module system.

## 🏗️ **Overall Testing Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cogniverse Test Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ EVALUATION  │  │ INGESTION   │  │ ROUTING     │            │
│  │   Module    │  │   Module    │  │   Module    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│       │                │                │                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Unit Tests   │  │Unit Tests   │  │Unit Tests   │            │
│  │@unit        │  │@unit        │  │@unit        │            │
│  │@ci_safe     │  │@ci_safe     │  │@ci_safe     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│       │                │                │                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Integration  │  │Integration  │  │Integration  │            │
│  │Tests        │  │Tests        │  │Tests        │            │
│  │@integration │  │@integration │  │@integration │            │
│  │@local_only  │  │@local_only  │  │@local_only  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│       │                │                │                     │
│       └────────────────┼────────────────┘                     │
│                        │                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Cross-Module Integration Tests                  │  │
│  │    @integration @requires_vespa @requires_phoenix       │  │
│  │  • End-to-end video search with evaluation              │  │
│  │  • Routing + Ingestion + Evaluation pipeline            │  │
│  │  • Performance benchmarking across modules              │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📂 **Module-Specific Testing Structure**

### **1. Evaluation Module** (`src/evaluation/`)
```
src/evaluation/
├── tests/
│   ├── unit/                    # @unit @ci_safe
│   │   ├── test_scorers.py     # Scoring algorithm tests
│   │   ├── test_evaluators.py  # Evaluator component tests
│   │   ├── test_metrics.py     # Metric calculation tests
│   │   └── test_*.py           # Individual component tests
│   ├── integration/             # @integration
│   │   ├── test_end_to_end.py  # Complete evaluation pipeline
│   │   ├── test_phoenix_production.py  # @requires_phoenix
│   │   └── test_schema_driven_pipeline.py
│   └── conftest.py             # Shared fixtures
```

**Coverage Target:** 75% (Line 57 in evaluation-tests.yml)
**Focus:** Evaluation algorithms, scorers, metrics, Phoenix integration

### **2. Ingestion Module** (`src/app/ingestion/`)
```
tests/ingestion/
├── unit/                       # @unit @ci_safe
│   ├── test_*_real.py         # Working tests (80%+ coverage)
│   ├── test_audio_processor.py # Legacy tests (being phased out)
│   └── test_*.py              # Component-specific tests
├── integration/                # @integration
│   ├── test_backend_ingestion.py   # @requires_vespa @local_only
│   ├── test_pipeline_orchestration.py
│   └── test_end_to_end_processing.py
└── utils/
    └── markers.py              # Smart environment detection
```

**Coverage Target:** 80% (Line 39 in pytest.ini)
**Focus:** Video processing, embedding generation, Vespa integration

### **3. Routing Module** (`src/app/routing/`)
```
tests/routing/
├── unit/
│   └── test_routing_strategies.py  # Strategy algorithm tests
├── integration/
│   └── test_tiered_routing.py      # Multi-tier routing tests
└── optimization/                   # DSPy optimization tests
    ├── dspy_optimizer.py
    └── gliner_optimizer.py
```

**Coverage Target:** Currently no explicit target, inherits from pytest.ini
**Focus:** Query routing, optimization, strategy selection

## 🔄 **Test Interplay and Dependencies**

### **1. Hierarchical Dependencies**
```
┌─────────────────┐
│ Cross-Module    │ ← End-to-end system tests
│ Integration     │   (All modules working together)
└─────────────────┘
         │
┌─────────────────┐
│ Module          │ ← Individual module integration
│ Integration     │   (Within each module)  
└─────────────────┘
         │
┌─────────────────┐
│ Unit Tests      │ ← Component-level testing
│ (Individual)    │   (Isolated components)
└─────────────────┘
```

### **2. Data Flow Testing**
```
Video Input → Ingestion → Embeddings → Search → Evaluation
     ↓            ↓           ↓          ↓         ↓
Unit Tests:   Processor   Embedding   Backend   Scorer
              Tests       Tests       Tests     Tests
     ↓            ↓           ↓          ↓         ↓
Integration:  Pipeline    Backend     Search    End-to-End
              Tests       Tests       Tests     Evaluation
```

### **3. Cross-Module Integration Points**

**Ingestion → Evaluation:**
```python
# Integration test that spans both modules
@pytest.mark.integration
@pytest.mark.requires_vespa
@pytest.mark.requires_phoenix
def test_ingestion_to_evaluation_pipeline():
    # 1. Ingest video with processors (ingestion)
    pipeline = VideoIngestionPipeline(config)
    result = pipeline.process_video(video_path)
    
    # 2. Perform search (backend)
    search_results = vespa_client.search(query)
    
    # 3. Evaluate results (evaluation)  
    scores = evaluator.evaluate(search_results, ground_truth)
    
    # Test cross-module data flow
    assert scores.precision > 0.8
```

**Routing → Evaluation:**
```python
# Router performance evaluation
@pytest.mark.integration
def test_routing_evaluation():
    # 1. Route queries (routing)
    router = QueryRouter(strategies)
    routed_queries = router.route_batch(queries)
    
    # 2. Evaluate routing decisions (evaluation)
    routing_scores = evaluate_routing_decisions(routed_queries)
    
    assert routing_scores.accuracy > 0.9
```

## 📊 **Coverage Calculation Strategy**

### **1. Coverage Configuration (pytest.ini)**
```ini
[coverage:run]
source = src/evaluation,src/app/ingestion  # Which modules to track
omit = 
    */tests/integration/*                   # Exclude integration tests

[coverage:report]  
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

### **2. Module-Specific Coverage**

**Evaluation Module:**
```yaml
# .github/workflows/evaluation-tests.yml
pytest src/evaluation/tests/unit -v -m unit 
  --cov=src/evaluation                    # Only evaluation module
  --cov-fail-under=75                     # 75% threshold
```

**Ingestion Module:**
```yaml
# .github/workflows/test-ingestion.yml  
pytest tests/ingestion/unit/ -m "unit and ci_safe"
  --cov=src/app/ingestion/processors     # Only processor submodule
  --cov-fail-under=80                    # 80% threshold
```

**Routing Module:**
```bash
# No dedicated workflow yet, uses global pytest.ini settings
pytest tests/routing/ --cov=src/app/routing
```

### **3. Coverage Aggregation**

**Global Coverage (pytest.ini):**
```ini
--cov=src/evaluation         # Evaluation module coverage
--cov=src/app/ingestion      # Ingestion module coverage
# Missing: --cov=src/app/routing  # TODO: Add routing coverage
```

**Why Integration Tests Are Excluded:**
```ini
omit = */tests/integration/*
```
- Integration tests are **functional validation**, not code coverage
- They test **data flow** and **system behavior**
- Coverage should focus on **unit test quality**

## 🚀 **Running Tests Across Modules**

### **1. Module-Specific Testing**
```bash
# Evaluation only
pytest src/evaluation/tests/unit -m unit --cov=src/evaluation

# Ingestion only  
python scripts/test_ingestion.py --unit --ci-safe

# Routing only
pytest tests/routing/ -v
```

### **2. Cross-Module Integration**
```bash
# End-to-end system tests
pytest tests/ -m "integration and not local_only" -v

# Full system with heavy models (local only)
pytest tests/ -m "integration and local_only" -v

# Cross-module performance tests
pytest tests/ -m benchmark -v
```

### **3. CI vs Local Execution**

**CI Environment (GitHub Actions):**
```bash
# Fast, lightweight tests only
pytest -m "unit and ci_safe" --cov-fail-under=75
pytest -m "integration and ci_safe" --maxfail=5
```

**Local Development:**
```bash
# Full system testing with real dependencies
pytest -m integration --cov=src/evaluation,src/app/ingestion
python scripts/test_ingestion.py --integration --local-only
```

## 📈 **Coverage Metrics by Module**

### **Current Coverage Status**

| Module | Unit Coverage | Target | Integration Coverage | 
|--------|---------------|--------|---------------------|
| **Evaluation** | ~75% | 75% | Phoenix + E2E tests |
| **Ingestion** | **80%+** | 80% | Vespa + Model tests |
| **Routing** | Unknown | TBD | Strategy tests |
| **Common** | Partial | TBD | Shared utilities |

### **Coverage Calculation Examples**

**Ingestion Module Coverage:**
```
AudioProcessor:    99% (66/67 statements) ✅
ChunkProcessor:   100% (67/67 statements) ✅  
KeyframeProcessor: 98% (95/97 statements) ✅
Overall Module:    80%+ (meets threshold) ✅
```

**Evaluation Module Coverage:**
```
Scorers:           85% (scoring algorithms)
Evaluators:        78% (evaluation workflow)
Metrics:           80% (metric calculations)
Overall Module:    75%+ (meets threshold) ✅
```

## 🔧 **Testing Best Practices**

### **1. Module Isolation**
- **Unit tests** should test individual components in isolation
- **Integration tests** should test module interactions
- **Cross-module tests** should test end-to-end workflows

### **2. Coverage Strategy**  
- **Focus on unit test coverage** (excludes integration from coverage)
- **Integration tests validate behavior**, not coverage
- **Each module has appropriate coverage thresholds**

### **3. Environment Awareness**
- **CI**: Fast, mocked, lightweight tests
- **Local**: Real dependencies, heavy models, full integration
- **Conditional execution** based on available dependencies

### **4. Future Improvements**
- [ ] Add routing module to coverage tracking
- [ ] Create cross-module integration test suite  
- [ ] Implement performance regression testing
- [ ] Add memory profiling for heavy model tests
- [ ] Create unified test runner for all modules

This architecture ensures **comprehensive testing** while maintaining **fast CI builds** and **thorough local validation** across the entire Cogniverse system.