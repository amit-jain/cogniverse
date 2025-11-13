# Cogniverse Evaluation

Provider-agnostic evaluation framework for experiments and metrics.

## Overview

This package provides a unified evaluation framework that works across different telemetry and experiment tracking providers. It defines provider-independent interfaces for experiments, metrics, datasets, and storage.

## Key Components

### Experiments (`cogniverse_evaluation.experiments`)

Experiment management and tracking:
- `Experiment`: Experiment data model
- `ExperimentManager`: Experiment lifecycle management
- Run tracking and comparison
- Hyperparameter logging

### Metrics (`cogniverse_evaluation.metrics`)

Provider-agnostic evaluation metrics:
- `AccuracyMetric`: Accuracy calculations
- `RelevanceMetric`: Relevance scoring
- `MetricCalculator`: Unified metric computation
- Custom metric definitions

### Datasets (`cogniverse_evaluation.datasets`)

Dataset handling and validation:
- `DatasetLoader`: Load evaluation datasets
- `DatasetValidator`: Validate dataset schemas
- Format conversion utilities
- Test dataset management

### Storage (`cogniverse_evaluation.storage`)

Storage interface abstraction:
- `StorageInterface`: Provider-independent storage
- Result persistence
- Query capabilities
- Export utilities

## Installation

```bash
pip install cogniverse-evaluation
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Pure backend interfaces
- `cogniverse-foundation`: Configuration and base classes

**External:**
- `inspect-ai>=0.3.0`: LLM inspection and evaluation
- `numpy>=1.24.0`: Numerical computations
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: ML metrics
- `pillow>=10.0.0`: Image processing

## Usage

```python
from cogniverse_evaluation.experiments import Experiment, ExperimentManager
from cogniverse_evaluation.metrics import AccuracyMetric, RelevanceMetric
from cogniverse_evaluation.datasets import DatasetLoader

# Create experiment
experiment = Experiment(
    name="routing_optimization_v1",
    description="Testing GEPA optimizer",
    hyperparameters={"learning_rate": 0.001}
)

# Track metrics
accuracy_metric = AccuracyMetric()
relevance_metric = RelevanceMetric()

results = accuracy_metric.calculate(predictions, ground_truth)
experiment.log_metric("accuracy", results["accuracy"])

# Load evaluation dataset
loader = DatasetLoader()
dataset = loader.load("golden_eval_v1.csv")
```

## Provider Plugins

The evaluation framework supports multiple providers through plugins:
- **Phoenix**: `cogniverse-telemetry-phoenix` package implements Phoenix-specific evaluation provider
- **Custom Providers**: Implement `EvaluationProvider` interface

## Architecture Position

Evaluation sits in the **Core Layer** of the Cogniverse architecture:

```
Foundation Layer:
  cogniverse-sdk → cogniverse-foundation
    ↓
Core Layer:
  cogniverse-evaluation ← YOU ARE HERE
  cogniverse-core
  cogniverse-telemetry-phoenix (plugin)
```

## Development

```bash
# Install in editable mode
cd libs/evaluation
pip install -e .

# Run tests
pytest tests/evaluation/
```

## Testing

The evaluation framework includes:
- Unit tests for metrics calculations
- Integration tests with mock providers
- Dataset validation tests
- Storage interface tests

## License

MIT
