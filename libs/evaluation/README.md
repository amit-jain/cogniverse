# Cogniverse Evaluation

**Last Updated:** 2026-07-25
**Layer:** Core
**Dependencies:** See `pyproject.toml`

Evaluation framework for experiments, retrieval metrics, trace analysis, and
configured visual assessments.

## Overview

The Evaluation package sits in the **Core Layer**. It defines provider
interfaces for experiments, metrics, datasets, and evaluators. The current
dataset, trace, and experiment workflows are exercised with the Phoenix
provider supplied by `cogniverse-telemetry-phoenix`.

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
│   ├── llm_judge.py         # LLMJudgeCore (live-traffic relevance scoring)
│   ├── configurable_visual_judge.py # ConfigurableVisualJudge
│   ├── _media_helpers.py    # Shared source_url resolution + frame extraction
│   ├── reference_free.py    # QueryResultRelevanceEvaluator, etc. (async)
│   ├── sync_reference_free.py # SyncQueryResultRelevanceEvaluator, etc.
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
- `LLMJudgeCore`: LLM judge for live-traffic relevance scoring (OAI-compatible endpoint), used by the quality monitor
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

Function-based retrieval metrics:

**Built-in Metrics:**
- **MRR**: Reciprocal rank of the first relevant result
- **NDCG**: Discounted ranking quality
- **Precision/Recall/F1 at k**: Cutoff-based retrieval quality
- **MAP**: Average precision across the ranked result list
- **Metric suite**: Combined retrieval metric calculation

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
# Run an experiment over profile/strategy combinations
uv run cogniverse-eval evaluate \
  --mode experiment \
  --dataset golden_eval_v1 \
  --profiles frame_based_colpali \
  --strategies binary_binary

# Evaluate selected traces
uv run cogniverse-eval evaluate \
  --mode batch \
  --dataset golden_eval_v1 \
  --tenant-id acme:acme \
  --trace-ids TRACE_ID

# Inspect recent traces for a tenant
uv run cogniverse-eval list-traces \
  --tenant-id acme:acme \
  --hours 2 \
  --limit 50
```

## Installation

```bash
uv sync --extra dev --extra cpu
```

## Dependencies

The installable package dependencies and exact versions are defined in
`libs/evaluation/pyproject.toml`. Workspace development must use the root
`uv.lock`.

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

Repeated identifiers consume a rank but receive relevance credit only once.
Metric cutoffs must be non-negative.

## Multi-Modal Evaluation

The evaluation framework supports video and image-backed search results:

### Visual Evaluation

- **Video frame analysis**: Resolve a result's `source_url` and extract frames.
- **Configured visual judgment**: Send those frames to the configured
  OpenAI-compatible evaluator endpoint.
- **Retrieval relevance**: Score text-to-video results with golden or
  reference-free evaluators.

## Provider Plugins

The provider registry discovers implementations from the
`cogniverse.evaluation.providers` entry-point group.

**Built-in Providers:**

- **Phoenix**: Supplied by the `cogniverse-telemetry-phoenix` package.

**Custom Providers:**

Implement `EvaluationProvider` and register the implementation through the
entry-point group.

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
# Install the workspace
uv sync --extra dev --extra cpu

# Run the complete evaluation suite with full tracebacks
JAX_PLATFORM_NAME=cpu uv run pytest tests/evaluation -v --tb=long

# Run specific evaluator tests
uv run pytest tests/evaluation/unit/test_visual_plugin.py -v --tb=long
uv run pytest tests/evaluation/unit/test_media_helpers.py -v --tb=long
```

## Testing

The evaluation package includes:
- Unit tests for all evaluators
- Metrics calculation tests
- Multi-modal evaluation tests
- Provider plugin tests
- Real Phoenix and Vespa integration tests
- Reference-free evaluation tests

## License

MIT
