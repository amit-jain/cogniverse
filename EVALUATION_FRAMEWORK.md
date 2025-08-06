# Evaluation Framework for Cogniverse

## Overview

The evaluation framework provides comprehensive assessment of the video retrieval system through three complementary approaches:

1. **Reference-free evaluation** of existing production traces
2. **Golden dataset evaluation** for marked test queries
3. **Controlled experiments** using Phoenix

## Architecture

### 1. Span Evaluator (`src/evaluation/span_evaluator.py`)

Evaluates **existing traces** from actual system usage:

- **Reference-free evaluators**:
  - `QueryResultRelevanceEvaluator`: Assesses relevance based on retrieval scores
  - `ResultDiversityEvaluator`: Measures diversity of retrieved videos
  - `TemporalCoverageEvaluator`: Evaluates temporal distribution of results
  - `LLMRelevanceEvaluator`: (Placeholder) Uses LLM to judge relevance

- **Golden dataset evaluator**:
  - Evaluates spans marked with `is_test_query=True` or `dataset_id`
  - Compares against expected results
  - Calculates MRR, NDCG, Precision@k, Recall@k

### 2. Phoenix Experiments (`src/evaluation/phoenix_experiments.py`)

Runs **controlled experiments** with new queries:

- Creates test datasets including challenging queries
- Runs queries through different configurations
- Uses Phoenix's native experiment tracking
- Compares performance metrics across configurations

### 3. Inspect AI Integration (`src/evaluation/inspect_integration.py`)

Provides the **evaluation framework**:

- **Structured Tasks**: Combine datasets, solvers, and scorers
- **Composable Solvers**: Chain evaluation steps
- **Flexible Scorers**: Implement custom metrics
- **Async Support**: Efficient concurrent evaluation

## Key Concepts

### Marking Test Queries

When running test queries, mark them for evaluation:

```python
# In your search code
span.set_attributes({
    "is_test_query": True,
    "dataset_id": "golden_test_v1",
    "evaluation_enabled": True
})
```

### Reference-free vs Golden Evaluation

- **Reference-free**: Can evaluate ANY span without ground truth
- **Golden dataset**: Only evaluates spans marked with dataset identifiers

### Phoenix Integration

- **SpanEvaluations**: Evaluation results are uploaded to Phoenix
- **Experiments**: Controlled A/B testing with Phoenix experiment tracking
- **Tracing**: All evaluations are traced for observability

## Usage

### Evaluate Existing Spans

```bash
# Evaluate spans from last 24 hours
uv run python scripts/run_proper_evaluation.py --hours 24
```

### Run Experiments

```bash
# Run experiments comparing configurations
uv run python scripts/run_proper_evaluation.py --skip-spans
```

### Full Demo

```bash
# Run complete demonstration
uv run python scripts/run_proper_evaluation.py --full-demo
```

## How Inspect AI Helps

1. **Structure**: Provides clear abstractions for evaluation tasks
2. **Modularity**: Evaluators can be mixed and matched
3. **Scalability**: Async execution for efficient evaluation
4. **Integration**: Works seamlessly with Phoenix and other tools
5. **Reproducibility**: Evaluation pipelines are code, not scripts

## Example Flow

1. **Production Usage**: Users query the system, creating spans
2. **Span Evaluation**: Reference-free evaluators assess all spans
3. **Test Queries**: Marked queries are evaluated against golden dataset
4. **Experiments**: New configurations tested with controlled experiments
5. **Analysis**: Results viewed in Phoenix UI with full tracing

## Benefits

- **No Special Traces**: Evaluates real production usage
- **Flexible Evaluation**: Reference-free for general assessment
- **Precise Testing**: Golden datasets for specific scenarios
- **Continuous Improvement**: Experiments drive optimization
- **Observable**: Everything traced and visible in Phoenix