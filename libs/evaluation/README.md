# Cogniverse Evaluation

**Last Updated:** 2025-11-13
**Layer:** Core
**Dependencies:** cogniverse-sdk, cogniverse-foundation

Provider-agnostic evaluation framework for experiments, metrics, and multi-modal assessments.

## Overview

The Evaluation package sits in the **Core Layer**, providing a unified evaluation framework that works across different telemetry and experiment tracking providers. It defines provider-independent interfaces for experiments, metrics, datasets, and evaluators with first-class support for multi-modal content.

This package enables rigorous evaluation of agents, search systems, and multi-modal RAG pipelines using standardized metrics and golden datasets.

## Package Structure

```
cogniverse_evaluation/
├── __init__.py
├── cli.py                   # Command-line interface for evaluations
├── span_evaluator.py        # Span-level evaluation
├── online_evaluator.py      # Online (live) evaluation
├── quality_monitor.py       # Quality monitoring
├── analysis/                # Evaluation analysis tools
├── core/                    # Core evaluation primitives
│   ├── task.py              # Inspect AI Task construction
│   ├── solvers.py           # Retrieval/batch/live solvers
│   ├── inspect_scorers.py   # Configured Inspect scorers
│   ├── experiment_tracker.py
│   ├── ground_truth.py
│   ├── reranking.py
│   ├── schema_analyzer.py
│   └── solver_output.py
├── evaluators/              # Built-in evaluators
│   ├── base.py              # BaseEvaluator abstract class
│   ├── golden_dataset.py    # GoldenDatasetEvaluator
│   ├── llm_judge.py         # LLMJudgeCore, SyncLLMReferenceFreeEvaluator, etc.
│   ├── configurable_visual_judge.py # ConfigurableVisualJudge
│   ├── _media_helpers.py    # Shared source_url resolution + frame extraction
│   ├── reference_free.py    # QueryResultRelevanceEvaluator, etc. (async)
│   ├── sync_reference_free.py # SyncQueryResultRelevanceEvaluator, etc.
│   ├── metadata_fetcher.py  # Metadata fetching helpers
│   └── routing_evaluator.py # RoutingEvaluator
├── metrics/                 # Evaluation metrics
│   └── custom.py            # calculate_mrr, calculate_ndcg, etc. (no CustomMetric class)
├── plugins/                 # Plugin system for providers
└── providers/               # Evaluation provider implementations
```

## Key Modules

### Evaluators (`cogniverse_evaluation.evaluators`)

Built-in evaluators for different evaluation scenarios:

**Base Evaluators** (in `cogniverse_evaluation.evaluators.base`):
- `Evaluator`: Abstract base class for all evaluators
- `EvaluationResult`: Dataclass returned by all evaluators

**Dataset Evaluators:**
- `GoldenDatasetEvaluator`: Evaluate against golden datasets
- `RoutingEvaluator`: Specialized routing agent evaluation

**LLM-Based Evaluators** (in `cogniverse_evaluation.evaluators.llm_judge`):
- `LLMJudgeCore`: Base class for LLM judge evaluators (OAI-compatible endpoint)
- `SyncLLMReferenceFreeEvaluator`: Synchronous LLM reference-free evaluation
- `SyncLLMReferenceBasedEvaluator`: Synchronous LLM reference-based evaluation
- `SyncLLMHybridEvaluator`: Combines reference-free and reference-based
- `ConfigurableVisualJudge`: Visual evaluation; provider, model, and endpoint
  come from the evaluator config. Resolves frames from each result's
  ``source_url`` via :class:`MediaLocator`.

**Reference-Free Evaluators** (in `cogniverse_evaluation.evaluators.reference_free`):
- `QueryResultRelevanceEvaluator`: Heuristic query-result relevance (async)
- `ResultDiversityEvaluator`: Evaluates result set diversity
- `TemporalCoverageEvaluator`: Evaluates temporal coverage
- `CompositeEvaluator`: Combines multiple evaluators

**Synchronous Reference-Free** (in `cogniverse_evaluation.evaluators.sync_reference_free`):
- `SyncQueryResultRelevanceEvaluator`: Synchronous query-result relevance
- `SyncResultDiversityEvaluator`: Synchronous diversity evaluation

### Metrics (`cogniverse_evaluation.metrics`)

Standardized metrics for evaluation:

**Built-in Metrics:**
- **Accuracy**: Classification accuracy
- **Relevance**: Retrieval relevance scoring
- **Precision/Recall**: Information retrieval metrics
- **F1 Score**: Harmonic mean of precision and recall
- **NDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

**Custom Metrics:**
- Define custom metrics with `CustomMetric` base class
- Support for multi-modal metrics
- Aggregation and statistical analysis

**Reference-Free Metrics:**
- Quality assessment without ground truth
- Coherence and consistency scoring
- Fluency and readability metrics

### Inspect AI Integration (`cogniverse_evaluation.core`)

Integration with the Inspect AI framework lives in `core/task.py`
(`evaluation_task` builds the Inspect `Task`), `core/solvers.py`, and
`core/inspect_scorers.py`:
- Inspect `Task` construction per evaluation mode (experiment/batch/live)
- Retrieval/batch/live solvers
- Configured Inspect scorers
- Result analysis and reporting

### Span Evaluation (`cogniverse_evaluation.span_evaluator`)

Trace span evaluation for observability:
- Evaluate individual spans in distributed traces
- Performance metrics per span
- Multi-modal span assessment
- Integration with telemetry providers

### CLI Interface (`cogniverse_evaluation.cli`)

Command-line interface for running evaluations:
```bash
# Run golden dataset evaluation
cogniverse-eval golden --dataset path/to/dataset.csv --agent routing

# Run reference-free evaluation
cogniverse-eval reference-free --queries queries.txt --agent video_search

# Run visual judge evaluation
cogniverse-eval visual-judge --images image_dir/ --model qwen2-vl
```

## Installation

```bash
uv add cogniverse-evaluation
```

Or with pip:
```bash
pip install cogniverse-evaluation
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Pure backend interfaces
- `cogniverse-foundation`: Configuration and telemetry base

**External:**
- `inspect-ai>=0.3.0`: LLM inspection and evaluation framework
- `numpy>=1.24.0`: Numerical computations
- `pandas>=2.0.0`: Data manipulation and analysis
- `scikit-learn>=1.3.0`: ML metrics and evaluation
- `pillow>=10.0.0`: Image processing for multi-modal evaluation

## Usage Examples

### Golden Dataset Evaluation

```python
from cogniverse_evaluation.evaluators.golden_dataset import GoldenDatasetEvaluator

# Initialize evaluator with a dict mapping query -> expected results
golden_dataset = {
    "machine learning tutorial": {
        "expected_videos": ["v_abc123", "v_def456"],
        "relevance_scores": {"v_abc123": 1.0, "v_def456": 0.8},
    }
}
evaluator = GoldenDatasetEvaluator(golden_dataset=golden_dataset)

# Evaluate a retrieval result
result = await evaluator.evaluate(
    input="machine learning tutorial",
    output=[{"source_id": "v_abc123"}, {"source_id": "v_xyz789"}],
    metadata={"is_test_query": True},
)

print(f"Score (MRR): {result.score:.3f}")
print(f"Label: {result.label}")
```

### LLM-as-Judge Evaluation

```python
from cogniverse_evaluation.evaluators.llm_judge import (
    LLMJudgeCore,
    SyncLLMReferenceFreeEvaluator,
)

# Synchronous LLM-based reference-free evaluator
# (used directly in Phoenix experiment runners)
judge = SyncLLMReferenceFreeEvaluator(
    model_name="qwen2-vl",
    base_url="http://localhost:11434",
)

# Evaluate a query-result pair
result = judge.evaluate(
    input={"query": "machine learning tutorials"},
    output=[{"source_id": "v_abc123", "score": 0.92}],
)

print(f"Score: {result.score:.2f}")
print(f"Label: {result.label}")
```

### Multi-Modal Visual Evaluation

```python
from cogniverse_core.common.media import MediaConfig, MediaLocator
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_evaluation.evaluators.configurable_visual_judge import (
    ConfigurableVisualJudge,
)

# The provider, model, and endpoint come from the tenant's evaluator config
# (configured under evaluators.<evaluator_name>); the constructor only takes
# the locator and the config key.
locator = MediaLocator(tenant_id=SYSTEM_TENANT_ID, config=MediaConfig())
visual_judge = ConfigurableVisualJudge(
    locator=locator, evaluator_name="visual_judge"
)

# Each search result must carry source_url; the judge resolves it through the
# locator, extracts frames, and asks the configured LLM whether they match.
result = visual_judge.evaluate(
    input={"query": "red sports car"},
    output={"results": search_results},
)
print(f"Score: {result.score:.2f} ({result.label})")
```

### Reference-Free Evaluation

```python
from cogniverse_evaluation.evaluators.reference_free import QueryResultRelevanceEvaluator

# Heuristic reference-free evaluator (no ground truth required)
evaluator = QueryResultRelevanceEvaluator(min_score_threshold=0.5)

# Evaluate retrieved results for a query
result = await evaluator.evaluate(
    input="machine learning video tutorial",
    output=[{"source_id": "v_abc123", "score": 0.87}],
)

print(f"Score: {result.score:.3f}")
print(f"Label: {result.label}")
```

### Built-in Metrics

The `cogniverse_evaluation.metrics` module exports function-based metrics
(there is no `CustomMetric` base class):

```python
from cogniverse_evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

retrieved = ["v_abc123", "v_def456", "v_xyz789"]
expected  = ["v_abc123", "v_ghi000"]

print(f"MRR:        {calculate_mrr(retrieved, expected):.3f}")
print(f"NDCG@10:    {calculate_ndcg(retrieved, expected):.3f}")
print(f"Precision@3:{calculate_precision_at_k(retrieved, expected, k=3):.3f}")
print(f"Recall@3:   {calculate_recall_at_k(retrieved, expected, k=3):.3f}")
```

### Orchestrator Agent Evaluation

```python
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator

# Initialize routing evaluator
evaluator = RoutingEvaluator(
    golden_dataset="routing_golden_v2.csv",
    metrics=["routing_accuracy", "confidence", "latency"]
)

# Evaluate routing decisions via OrchestratorAgent
results = await evaluator.evaluate(
    agent=orchestrator_agent,
    test_queries=test_queries
)

print(f"Routing Accuracy: {results['routing_accuracy']:.2%}")
print(f"Avg Confidence: {results['avg_confidence']:.3f}")
print(f"P95 Latency: {results['p95_latency']:.0f}ms")
```

## Multi-Modal Evaluation

The evaluation framework provides first-class support for multi-modal content:

### Visual Evaluation
- **Image Quality Assessment**: Evaluate image search results
- **Video Frame Analysis**: Frame-by-frame video evaluation
- **Cross-Modal Relevance**: Text-to-image/video relevance scoring

### Audio Evaluation
- **Speech Quality**: Audio clarity and transcription accuracy
- **Audio-Text Alignment**: Evaluate audio transcription quality

### Multi-Modal Consistency
- **Cross-Modal Coherence**: Consistency across modalities
- **Alignment Metrics**: Text-image-video alignment scoring

## Provider Plugins

The evaluation framework supports multiple providers through a plugin system:

**Built-in Providers:**
- **Phoenix**: Integration via `cogniverse-telemetry-phoenix` package
- **MLflow**: Experiment tracking integration
- **Weights & Biases**: W&B integration
- **TensorBoard**: TensorBoard logging

**Custom Providers:**
Implement the provider interface to add support for new backends.

## Architecture Position

```
Foundation Layer:
  cogniverse-sdk → cogniverse-foundation
    ↓
Core Layer:
  cogniverse-evaluation ← YOU ARE HERE
  cogniverse-core
  cogniverse-telemetry-phoenix (plugin)
    ↓
Implementation Layer:
  cogniverse-agents (uses evaluation for optimization)
  cogniverse-synthetic (uses evaluation for quality)
    ↓
Application Layer:
  cogniverse-runtime (evaluation endpoints)
  cogniverse-dashboard (evaluation visualization)
```

## Development

```bash
# Install in editable mode
cd libs/evaluation
uv pip install -e .

# Run tests
pytest tests/evaluation/

# Run specific evaluator tests
pytest tests/evaluation/unit/test_visual_plugin.py
pytest tests/evaluation/unit/test_media_helpers.py
```

## Testing

The evaluation package includes:
- Unit tests for all evaluators
- Metrics calculation tests
- Multi-modal evaluation tests
- Provider plugin tests
- Integration tests with golden datasets
- Reference-free evaluation tests

## Performance

- **Golden Dataset Evaluation**: Process 1000+ samples/minute
- **LLM Judge**: 10-20 judgments/second (model-dependent)
- **Visual Judge**: 5-10 images/second (GPU-dependent)
- **Batch Processing**: Efficient batch evaluation support

## License

MIT
