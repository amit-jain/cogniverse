# Evaluation Framework Redesign Proposal

## Executive Summary

Redesign the evaluation framework using **Inspect AI as the orchestration layer**, **RAGAS for proven RAG metrics**, and **Phoenix for data management**. Build alongside existing `experiments.py` to maintain dashboard compatibility. Design for future extraction as a standalone Python package.

## Current State Analysis

### What Actually Works
1. **experiments.py** - Runs new experiments successfully, integrated with dashboard
2. **Phoenix dashboard** - Expects specific data format from experiments
3. **run_experiments_with_visualization.py** - Good CLI and visualization

### What's Completely Broken
1. **Batch evaluation** - Calls non-existent Phoenix APIs (`get_traces_by_ids`)
2. **Orchestrator** - Complex mess with broken methods
3. **No reference-free evaluation** - PRD vision never implemented

### Dashboard Dependencies
The Phoenix dashboard (`phoenix_dashboard_standalone.py`) requires:
- Phoenix datasets queryable via GraphQL
- Experiment results with profile/strategy structure
- Specific metrics: MRR, Recall@1, Recall@5
- Traces with proper metadata

## Proposed Architecture

### Core Technology Stack

1. **Inspect AI** - Evaluation orchestration framework
   - Flexible solver chains for different evaluation modes
   - Tool integration for our search backends
   - Experiment tracking and logging

2. **RAGAS** - RAG-specific metrics
   - Reference-free metrics (context relevancy)
   - Reference-based metrics (precision, recall)
   - Generation metrics for future use

3. **Phoenix** - Data and experiment management
   - Dataset storage and versioning
   - Trace repository
   - Experiment tracking
   - Evaluation results storage

### Module Structure

```
src/evaluation/
├── experiments.py               # KEEP AS IS - Dashboard depends on it
│                               # Don't modify until dashboard is updated
│
├── core/                       # NEW Inspect AI evaluation system
│   ├── __init__.py
│   ├── task.py                 # Main evaluation task orchestrator
│   ├── solvers.py              # Data acquisition strategies
│   │                           # - retrieval_solver (new searches)
│   │                           # - trace_loader_solver (existing traces)
│   │                           # - live_trace_solver (real-time)
│   ├── scorers.py              # Evaluation metrics
│   │                           # - RAGAS metric wrappers
│   │                           # - Custom scorers
│   └── tools.py                # External service integration
│
├── data/
│   ├── __init__.py
│   ├── datasets.py             # Phoenix dataset management
│   ├── traces.py               # Trace fetching and management
│   └── storage.py              # Phoenix client wrapper
│
├── metrics/
│   ├── __init__.py
│   ├── ragas.py                # RAGAS metric implementations
│   ├── custom.py               # Domain-specific metrics
│   └── reference_free.py       # Metrics without ground truth
│
├── cli.py                      # Unified CLI entry point
│
└── tests/                      # Comprehensive test suite
    ├── unit/
    │   ├── test_scorers.py
    │   ├── test_solvers.py
    │   └── test_metrics.py
    ├── integration/
    │   ├── test_phoenix_integration.py
    │   ├── test_inspect_tasks.py
    │   └── test_end_to_end.py
    └── fixtures/
        └── sample_data.py
```

## Implementation Details

### 1. Core Evaluation Task (core/task.py)

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
import phoenix as px
from typing import List, Optional, Dict, Any

@task
def evaluation_task(
    mode: str,
    dataset_name: str,
    profiles: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    trace_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Unified evaluation task for all modes.
    
    Args:
        mode: One of "experiment", "batch", or "live"
        dataset_name: Phoenix dataset name
        profiles: Video processing profiles (for experiment mode)
        strategies: Ranking strategies (for experiment mode)
        trace_ids: Specific traces to evaluate (for batch mode)
        config: Additional configuration
    """
    # Validate inputs based on mode
    if mode == "experiment" and not (profiles and strategies):
        raise ValueError("profiles and strategies required for experiment mode")
    
    # Load dataset from Phoenix
    phoenix_client = px.Client()
    dataset = phoenix_client.get_dataset(dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Choose solver based on mode
    from .solvers import retrieval_solver, trace_loader_solver, live_trace_solver
    
    if mode == "experiment":
        solver = retrieval_solver(profiles, strategies)
    elif mode == "batch":
        solver = trace_loader_solver(trace_ids)
    elif mode == "live":
        solver = live_trace_solver()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Get configured scorers
    from .scorers import get_configured_scorers
    scorers = get_configured_scorers(config)
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        metadata={
            "mode": mode,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat()
        }
    )
```

### 2. Phoenix Integration with Dashboard Compatibility

```python
# data/storage.py
import phoenix as px
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime

class PhoenixStorage:
    """
    Phoenix storage interface maintaining dashboard compatibility.
    """
    
    def __init__(self):
        self.client = px.Client()
    
    def log_experiment_results(
        self,
        experiment_name: str,
        profile: str,
        strategy: str,
        results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> str:
        """
        Log results in format expected by dashboard.
        
        Dashboard expects:
        - Experiments grouped by profile/strategy
        - Metrics: mrr, recall@1, recall@5
        - Query-level results with expected/actual videos
        """
        # Format for dashboard compatibility
        formatted_results = {
            "profile": profile,
            "strategy": strategy,
            "aggregate_metrics": {
                "mrr": {"mean": metrics.get("mrr", 0.0)},
                "recall@1": {"mean": metrics.get("recall@1", 0.0)},
                "recall@5": {"mean": metrics.get("recall@5", 0.0)}
            },
            "queries": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to Phoenix
        experiment_id = self.client.log_experiment(
            name=experiment_name,
            metadata={
                "profile": profile,
                "strategy": strategy,
                "framework": "inspect_ai"  # Tag to identify new system
            },
            results=formatted_results
        )
        
        return experiment_id
    
    def get_traces_for_evaluation(
        self,
        trace_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        filter_condition: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get traces using working Phoenix methods.
        """
        if trace_ids:
            filter_condition = f"trace_id in {trace_ids}"
        
        # Use actual working Phoenix method
        df = self.client.get_spans_dataframe(
            filter_condition=filter_condition,
            start_time=start_time,
            root_spans_only=True,
            limit=1000
        )
        
        return df
```

### 3. Testing Strategy

#### Unit Tests

```python
# tests/unit/test_scorers.py
import pytest
from unittest.mock import Mock
from src.evaluation.core.scorers import (
    ragas_context_relevancy_scorer,
    custom_diversity_scorer
)

class TestRagasScorers:
    def test_context_relevancy_scorer(self):
        """Test RAGAS context relevancy scorer"""
        scorer = ragas_context_relevancy_scorer()
        
        # Mock state
        state = Mock()
        state.input.query = "test query"
        state.outputs = {
            "retrieved": [
                {"content": "relevant content"},
                {"content": "another relevant"}
            ]
        }
        
        score = scorer(state)
        assert score.value >= 0.0 and score.value <= 1.0
        assert "relevancy" in score.explanation.lower()
    
    def test_diversity_scorer(self):
        """Test custom diversity scorer"""
        scorer = custom_diversity_scorer()
        
        state = Mock()
        state.outputs = {
            "retrieved": [
                {"video_id": "video1"},
                {"video_id": "video2"},
                {"video_id": "video1"}  # Duplicate
            ]
        }
        
        score = scorer(state)
        assert score.value == 2/3  # 2 unique out of 3
        assert "2/3 unique videos" in score.explanation

class TestReferenceFreeScorors:
    def test_temporal_coherence_scorer(self):
        """Test temporal coherence without ground truth"""
        from src.evaluation.metrics.reference_free import temporal_coherence_scorer
        
        scorer = temporal_coherence_scorer()
        
        # Temporal query
        state = Mock()
        state.input.query = "What happened after the meeting?"
        state.outputs = {
            "retrieved": [
                {"timestamp": 100},
                {"timestamp": 200},
                {"timestamp": 150}  # Out of order
            ]
        }
        
        score = scorer(state)
        assert score.value == 0.0  # Not properly ordered
```

#### Integration Tests

```python
# tests/integration/test_phoenix_integration.py
import pytest
from src.evaluation.data.storage import PhoenixStorage
from datetime import datetime

@pytest.fixture
def phoenix_storage():
    """Phoenix storage fixture"""
    return PhoenixStorage()

class TestPhoenixIntegration:
    def test_dataset_creation(self, phoenix_storage):
        """Test creating dataset in Phoenix"""
        queries = [
            {"query": "test query 1", "expected_videos": ["v1", "v2"]},
            {"query": "test query 2", "expected_videos": ["v3"]}
        ]
        
        dataset_id = phoenix_storage.create_dataset(
            name="test_dataset",
            queries=queries
        )
        
        assert dataset_id is not None
        
        # Verify dataset exists
        dataset = phoenix_storage.client.get_dataset(dataset_id)
        assert dataset is not None
        assert len(dataset.examples) == 2
    
    def test_trace_fetching(self, phoenix_storage):
        """Test fetching traces from Phoenix"""
        # Fetch recent traces
        df = phoenix_storage.get_traces_for_evaluation(
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        assert df is not None
        if not df.empty:
            assert 'trace_id' in df.columns
            assert 'attributes.input.value' in df.columns
    
    def test_dashboard_compatible_logging(self, phoenix_storage):
        """Test logging in dashboard-compatible format"""
        results = [
            {
                "query": "test query",
                "results": ["video1", "video2"],
                "expected": ["video1", "video3"]
            }
        ]
        
        metrics = {
            "mrr": 1.0,
            "recall@1": 1.0,
            "recall@5": 0.5
        }
        
        exp_id = phoenix_storage.log_experiment_results(
            experiment_name="test_exp",
            profile="colpali",
            strategy="binary",
            results=results,
            metrics=metrics
        )
        
        assert exp_id is not None
        
        # Verify dashboard can read this
        # (Would check GraphQL query here)
```

#### End-to-End Tests

```python
# tests/integration/test_end_to_end.py
import pytest
from inspect_ai import eval
from src.evaluation.core.task import evaluation_task

class TestEndToEnd:
    @pytest.mark.integration
    def test_experiment_mode(self):
        """Test full experiment evaluation"""
        task = evaluation_task(
            mode="experiment",
            dataset_name="test_dataset",
            profiles=["frame_based_colpali"],
            strategies=["binary_binary"]
        )
        
        results = eval(task)
        
        assert results is not None
        assert len(results.samples) > 0
        assert all(s.scores for s in results.samples)
    
    @pytest.mark.integration
    def test_batch_mode(self):
        """Test batch evaluation of existing traces"""
        # First create some traces
        # ... 
        
        task = evaluation_task(
            mode="batch",
            dataset_name="test_dataset",
            trace_ids=["trace1", "trace2"]
        )
        
        results = eval(task)
        
        assert results is not None
        assert len(results.samples) == 2
```

### 4. Configuration for Testing

```yaml
# evaluation_config.test.yaml
evaluation:
  phoenix_url: "http://localhost:6006"
  dataset_name: "test_dataset"
  
  test_mode: true  # Use mock services where appropriate
  
  modes:
    experiment:
      profiles: ["test_profile"]
      strategies: ["test_strategy"]
    
    batch:
      trace_source: "phoenix"
      max_traces: 10
  
  scorers:
    ragas:
      - context_relevancy
    custom:
      - diversity
    
  testing:
    mock_search_service: true
    mock_llm_evaluator: true
    fixtures_path: "tests/fixtures"
```

### 5. Package Structure for Standalone Distribution

```
cogniverse-eval/                 # Future standalone package
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── cogniverse_eval/
│       ├── __init__.py
│       ├── core/
│       ├── data/
│       ├── metrics/
│       └── cli.py
├── tests/
└── docs/
```

**pyproject.toml:**
```toml
[project]
name = "cogniverse-eval"
version = "0.1.0"
description = "Flexible evaluation framework for RAG systems"
dependencies = [
    "inspect-ai>=0.3.0",
    "ragas>=0.1.0",
    "arize-phoenix>=2.0.0",
    "pandas>=2.0.0",
    "click>=8.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0"
]

[project.scripts]
cogniverse-eval = "cogniverse_eval.cli:main"
```

## Migration Path

### Phase 1: Foundation (Week 1)
1. Set up Inspect AI and RAGAS
2. Create core module structure
3. Implement Phoenix storage with dashboard compatibility
4. Write initial unit tests

### Phase 2: Core Implementation (Week 2)
1. Implement evaluation task orchestrator
2. Create retrieval and trace loader solvers
3. Wrap RAGAS metrics as scorers
4. Add custom and reference-free metrics
5. Write integration tests

### Phase 3: Testing & Validation (Week 3)
1. Comprehensive unit test coverage (>80%)
2. Integration tests with Phoenix
3. End-to-end test scenarios
4. Performance benchmarking
5. Dashboard compatibility verification

### Phase 4: Package Preparation (Week 4)
1. Extract as standalone package
2. Documentation (Sphinx/MkDocs)
3. CI/CD pipeline setup
4. PyPI release preparation
5. Migration guide for existing users

## Testing Requirements

### Coverage Goals
- Unit test coverage: >80%
- Integration test coverage: >60%
- Critical paths: 100% coverage

### Test Categories
1. **Unit Tests**: All scorers, solvers, metrics
2. **Integration Tests**: Phoenix operations, Inspect AI tasks
3. **End-to-End Tests**: Complete evaluation workflows
4. **Performance Tests**: Large dataset handling
5. **Compatibility Tests**: Dashboard integration

### CI/CD Integration
```yaml
# .github/workflows/evaluation-tests.yml
name: Evaluation Framework Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run unit tests
        run: |
          pytest tests/unit --cov=src/evaluation --cov-report=xml
      - name: Run integration tests
        run: |
          docker-compose up -d phoenix
          pytest tests/integration -m integration
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Success Criteria

1. **Dashboard Compatibility** - Existing dashboard continues working
2. **Test Coverage** - >80% unit, >60% integration coverage
3. **Reference-Free Evaluation** - Works without ground truth
4. **Batch Evaluation** - Can evaluate existing traces
5. **Package Ready** - Can be extracted as standalone package
6. **Clear Failures** - No silent defaults or hidden errors
7. **Performance** - <100ms overhead per evaluation

## Why This Architecture

1. **Inspect AI provides flexibility** - Proven framework, production-grade
2. **RAGAS provides metrics** - Both reference-free and reference-based
3. **Phoenix provides persistence** - Datasets, traces, experiments
4. **Dashboard compatibility maintained** - No breaking changes
5. **Testable design** - Clear interfaces, mockable dependencies
6. **Package-ready** - Clean separation for future extraction
7. **Gradual migration** - Build alongside existing system