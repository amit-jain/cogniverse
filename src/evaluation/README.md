# Cogniverse Evaluation Module - Developer Documentation

> **For user documentation and usage examples, see [`docs/EVALUATION_FRAMEWORK.md`](../../docs/EVALUATION_FRAMEWORK.md)**

This README contains technical documentation for developers working on the evaluation module itself.

## Features

- **Inspect AI Integration**
  - Task-based evaluation with custom solvers and scorers
  - Plugin architecture for extensibility
  - Async execution with proper error handling

- **Phoenix Integration**
  - Centralized storage for traces, datasets, and experiments
  - Real-time observability and monitoring
  - Experiment tracking and comparison

- **Flexible Metrics**
  - **Quality Metrics**: Relevance, diversity, distribution, temporal coverage
  - **LLM Evaluators**: Visual judge with configurable providers (Ollama, Modal, OpenAI)
  - **Ground Truth**: Automated extraction and evaluation
  - **Reranking**: Temporal and semantic strategies

- **Experiment Tracking**
  - Use `src/evaluation/core/experiment_tracker.py` for running experiments
  - Full compatibility with legacy `run_experiments_with_visualization.py`
  - Automatic HTML report generation

## Installation

### As Part of Cogniverse

```bash
uv pip install -e .
```

### As Standalone Module

```bash
cd src/evaluation
pip install -e .
```

### With Visual Evaluation Support

```bash
pip install -e ".[visual]"
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using Experiment Tracker

```bash
# Run experiments with new evaluation framework
python src/evaluation/core/experiment_tracker.py \
  --dataset-name golden_eval_v1 \
  --profiles frame_based_colpali \
  --strategies binary_binary \
  --quality-evaluators
```

### Python API

```python
from src.evaluation.core.task import evaluation_task
from inspect_ai import eval

# Create evaluation task
task = evaluation_task(
    mode="experiment",
    dataset_name="my_dataset",
    profiles=["frame_based_colpali"],
    strategies=["binary_binary"]
)

# Run evaluation
results = await eval(task)
```

For detailed usage and examples, see `docs/EVALUATION_FRAMEWORK.md`

## Configuration

### Evaluation Config

Create a JSON or YAML config file:

```json
{
  "use_ragas": true,
  "ragas_metrics": ["context_relevancy", "answer_relevancy"],
  "use_custom": true,
  "custom_metrics": ["diversity", "temporal_coherence"],
  "use_visual": false,
  "top_k": 10
}
```

Use with CLI:

```bash
cogniverse-eval evaluate --mode experiment \
  --dataset my_dataset \
  --config config.json \
  -p profile1 -s strategy1
```

### Dataset Format

#### CSV Format

```csv
query,expected_videos,category
"person wearing red shirt","video1,video2",visual
"what happened after the meeting",video3,temporal
```

#### JSON Format

```json
[
  {
    "query": "person wearing red shirt",
    "expected_videos": ["video1", "video2"],
    "category": "visual"
  }
]
```

## Testing

### Run All Tests

```bash
pytest src/evaluation/tests -v
```

### Run Unit Tests Only

```bash
pytest src/evaluation/tests/unit -v -m unit
```

### Run Integration Tests

```bash
pytest src/evaluation/tests/integration -v -m integration
```

### Check Coverage

```bash
pytest src/evaluation/tests --cov=src/evaluation --cov-report=term-missing
```

The framework requires 80% test coverage minimum.

## Architecture

### Core Components

- **Task Orchestrator** (`core/task.py`): Main evaluation coordinator using Inspect AI
- **Experiment Tracker** (`core/experiment_tracker.py`): High-level experiment runner
- **Solvers** (`core/inspect_solvers.py`): Inspect AI solvers for data acquisition
  - `CogniverseRetrievalSolver`: Runs new searches
  - Protocol implementations for analysis
- **Scorers** (`core/scorers.py`): Metric calculation
  - Quality scorers for relevance, diversity
  - Custom scorers for domain-specific metrics
- **Evaluators** (`evaluators/`): LLM and quality evaluators
  - `configurable_visual_judge.py`: Multi-provider visual evaluation
  - `quality_evaluator.py`: Reference-free quality metrics
- **Plugins** (`plugins/`): Extensible functionality
  - `visual_evaluator.py`: Visual evaluation plugin
  - `phoenix_experiment.py`: Experiment tracking plugin
- **Tools** (`core/tools.py`): Phoenix integration utilities
- **Reranking** (`core/reranking.py`): Result reranking strategies
- **Ground Truth** (`core/ground_truth.py`): Ground truth extraction

### Design Principles

1. **No Hidden Defaults**: All configuration must be explicit
2. **Loud Failures**: Errors fail fast with clear messages
3. **Dashboard Compatibility**: Preserves existing dashboard functionality
4. **Extensibility**: Easy to add new metrics and solvers
5. **Testability**: Comprehensive test coverage with mocking

## Dashboard Integration

The framework maintains compatibility with the existing Phoenix dashboard by:

1. Preserving legacy metric names (MRR, Recall@1, Recall@5)
2. Maintaining the same data structure for experiment results
3. Supporting the existing GraphQL queries
4. Keeping profile/strategy comparison functionality

## Extending the Framework

### Adding a New Metric

```python
from inspect_ai import Score, scorer

@scorer
def my_custom_scorer():
    def score(state):
        # Calculate metric
        value = calculate_my_metric(state.outputs)
        
        return Score(
            value=value,
            explanation=f"My metric: {value:.3f}"
        )
    
    return score

# Register in get_configured_scorers()
```

### Adding a New Solver

```python
from inspect_ai import solver

@solver
def my_custom_solver(config):
    def solve(state):
        # Acquire data
        data = fetch_my_data(state.dataset)
        
        # Return updated state
        return state.update(outputs=data)
    
    return solve
```

## Troubleshooting

### Phoenix Connection Issues

```bash
# Check Phoenix is running
curl http://localhost:6006/health

# Set custom Phoenix URL
export PHOENIX_URL=http://my-phoenix:6006
```

### Missing Dependencies

```bash
# Install all dependencies
uv pip install -e ".[dev,visual]"
```

### Test Failures

```bash
# Run with verbose output
pytest -vv --tb=short

# Run specific test
pytest src/evaluation/tests/unit/test_task.py::TestEvaluationTask::test_experiment_mode
```

## Contributing

1. Write tests for any new functionality
2. Ensure 80% code coverage minimum
3. Follow existing patterns for consistency
4. Update documentation as needed
5. Run the full test suite before submitting

## License

See the main Cogniverse LICENSE file.