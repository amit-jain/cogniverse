# Evaluation Guide

## Overview

Cogniverse provides comprehensive evaluation of video search quality through Phoenix-integrated experiments, reference-free scorers, and multi-strategy comparison. All evaluations are tracked in Phoenix for reproducibility and analysis.

## Quick Start

### 1. Start Phoenix Server

```bash
# Local Phoenix server
docker run -d --name phoenix \
  -p 6006:6006 \
  -p 4317:4317 \
  -v phoenix-data:/data \
  arizephoenix/phoenix:phoenix:latest

# Verify Phoenix is running
curl http://localhost:6006/health
```

### 2. Create Evaluation Dataset

```bash
# Upload golden queries to Phoenix
uv run python scripts/create_phoenix_dataset.py \
  --csv data/testset/evaluation/video_search_queries.csv \
  --dataset-name golden_eval_v1

# Dataset format (CSV):
# query,expected_video,relevance
# "machine learning tutorial","ml_basics.mp4",1.0
# "deep learning intro","dl_intro.mp4",1.0
```

### 3. Run Evaluation

```bash
# Quality metrics evaluation (reference-free)
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name golden_eval_v1 \
  --profiles video_colpali_smol500_mv_frame \
  --test-multiple-strategies \
  --quality-evaluators

# With visual LLM evaluation
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name test_visual \
  --profiles video_videoprism_base_mv_chunk_30s \
  --llm-evaluators \
  --llm-model ollama/llava:7b \
  --test-multiple-strategies
```

### 4. View Results in Phoenix

```bash
# Access Phoenix dashboard
open http://localhost:6006

# Navigate to:
# - Experiments tab: View all evaluation runs
# - Datasets tab: Manage evaluation datasets
# - Traces tab: Analyze individual query traces
```

## Evaluation Types

### 1. Phoenix Experiments

Phoenix experiments provide complete evaluation tracking with experiment history, metric comparison, and reproducibility.

**Features:**
- Automatic experiment creation with unique IDs
- Complete metric tracking per query
- Side-by-side strategy comparison
- Historical experiment comparisons
- Dataset versioning

**Usage:**
```python
from src.evaluation.plugins.phoenix_experiment import PhoenixExperimentPlugin
import phoenix as px

# Get Phoenix client
client = px.Client()

# Get or create dataset
dataset = client.get_dataset(name="golden_eval_v1")

# Run experiment
result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
    dataset_name="golden_eval_v1",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["hybrid_float_bm25", "float_float"],
    evaluators=[quality_evaluator, visual_judge],
    config={"top_k": 10}
)

# Results automatically stored in Phoenix
print(f"Experiment ID: {result.experiment_id}")
print(f"Mean quality: {result.metrics['quality'].mean()}")
```

### 2. Reference-Free Quality Metrics

Evaluate search quality without ground truth labels using semantic analysis.

**QualityScorer:**
```python
from src.evaluation.evaluators.sync_reference_free import QualityScorer

scorer = QualityScorer()

# Evaluate search results
score = scorer.score(
    query="machine learning tutorial",
    results=[
        {"video_id": "ml_basics.mp4", "score": 0.9, "content": "Intro to ML"},
        {"video_id": "dl_intro.mp4", "score": 0.8, "content": "Deep learning"}
    ]
)

# Returns:
# {
#   "relevance_score": 0.85,  # Semantic similarity to query
#   "diversity_score": 0.72,  # Result variety
#   "distribution_score": 0.88  # Score distribution quality
# }
```

**Metrics Explained:**

**Relevance Score** (0-1):
- Semantic similarity between query and results
- Computed using embedding similarity
- Thresholds: High (≥0.8), Relevant (≥0.5), Low (<0.5)

**Diversity Score** (0-1):
- Measures variety in search results
- Calculated as: 1 - avg(pairwise_similarity)
- Higher = more diverse results

**Distribution Score** (0-1):
- Quality of score distribution across results
- Penalizes flat or bunched distributions
- Higher = better score separation

### 3. Visual LLM Judges

Use vision-language models to assess visual relevance and quality.

**ConfigurableVisualJudge:**
```python
from src.evaluation.evaluators.configurable_visual_judge import ConfigurableVisualJudge

# Initialize with Ollama LLaVA
visual_judge = ConfigurableVisualJudge(
    provider="ollama",
    model="llava:7b",
    base_url="http://localhost:11434/v1"
)

# Evaluate visual relevance
score = visual_judge.evaluate(
    query="machine learning tutorial",
    result={
        "video_id": "ml_basics.mp4",
        "frames": [frame1_path, frame2_path, frame3_path],
        "content": "Introduction to machine learning concepts"
    }
)

# Returns:
# {
#   "visual_relevance": 0.9,
#   "content_quality": 0.85,
#   "explanation": "Video clearly shows ML concepts with diagrams"
# }
```

**Supported Providers:**
- `ollama`: Local LLaVA/Bakllava models
- `openai`: GPT-4V
- `anthropic`: Claude 3 Vision

### 4. Retrieval Metrics (With Ground Truth)

Classical retrieval metrics when ground truth labels are available.

**Available Metrics:**
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG (Normalized DCG)**: Ranking quality with graded relevance
- **Precision@k**: Fraction of relevant results in top-k
- **Recall@k**: Fraction of relevant results retrieved
- **F1@k**: Harmonic mean of precision and recall

```python
from src.evaluation.core.retrieval_metrics import RetrievalMetrics

metrics = RetrievalMetrics()

# Compute metrics
results = metrics.compute_all(
    query_results=[
        {"video_id": "ml_basics.mp4", "rank": 1, "score": 0.9},
        {"video_id": "dl_intro.mp4", "rank": 2, "score": 0.8},
        {"video_id": "irrelevant.mp4", "rank": 3, "score": 0.7}
    ],
    relevant_docs=["ml_basics.mp4", "dl_intro.mp4"],
    k=10
)

# Returns:
# {
#   "mrr": 1.0,  # First result is relevant
#   "ndcg@10": 0.95,
#   "precision@10": 0.67,  # 2/3 top results relevant
#   "recall@10": 1.0,  # Found all relevant docs
#   "f1@10": 0.80
# }
```

## Multi-Strategy Comparison

### Comparing Ranking Strategies

Evaluate all 9 Vespa ranking strategies in a single experiment:

```bash
# Run all strategies
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name golden_eval_v1 \
  --profiles video_colpali_smol500_mv_frame \
  --test-multiple-strategies

# Strategies tested:
# - bm25_only
# - float_float
# - binary_binary
# - float_binary
# - phased
# - hybrid_float_bm25 (recommended)
# - binary_bm25
# - bm25_binary_rerank
# - bm25_float_rerank
```

### Results Analysis

```python
import phoenix as px

client = px.Client()

# Get experiment
experiment = client.get_experiment(name="inspect_eval_golden_eval_v1_20251004_120000")

# Compare strategies
for strategy in ["hybrid_float_bm25", "float_float", "bm25_only"]:
    strategy_results = [
        r for r in experiment.results
        if r.metadata.get("strategy") == strategy
    ]

    avg_quality = sum(r.metrics["quality"] for r in strategy_results) / len(strategy_results)
    print(f"{strategy}: {avg_quality:.3f}")

# Output:
# hybrid_float_bm25: 0.875
# float_float: 0.812
# bm25_only: 0.645
```

### Multi-Profile Comparison

Compare different embedding models (ColPali vs VideoPrism):

```bash
# Run multi-profile evaluation
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name golden_eval_v1 \
  --profiles \
    video_colpali_smol500_mv_frame \
    video_videoprism_base_mv_chunk_30s \
  --test-multiple-strategies
```

## Phoenix Dashboard Features

### Experiments Tab

**View All Experiments:**
- Experiment list with creation time
- Quick metrics summary (avg quality, count)
- Filter by dataset, profile, strategy

**Experiment Details:**
- Per-query metrics breakdown
- Score distributions
- Failure analysis
- Metadata (config, profiles, strategies)

### Compare Experiments

```python
# Compare two experiments
client = px.Client()

exp1 = client.get_experiment(name="baseline_experiment")
exp2 = client.get_experiment(name="optimized_experiment")

# Metrics comparison
baseline_quality = exp1.metrics["quality"].mean()
optimized_quality = exp2.metrics["quality"].mean()

improvement = (optimized_quality - baseline_quality) / baseline_quality * 100
print(f"Quality improvement: +{improvement:.1f}%")

# Per-query comparison
for i, (r1, r2) in enumerate(zip(exp1.results, exp2.results)):
    query = r1.input["query"]
    delta = r2.metrics["quality"] - r1.metrics["quality"]
    print(f"Query {i}: {query} → Δ{delta:+.3f}")
```

### Datasets Tab

**Manage Evaluation Datasets:**
- Upload CSV datasets
- View dataset examples
- Version tracking
- Dataset statistics (size, query distribution)

**Dataset Operations:**
```python
# Create dataset from CSV
import pandas as pd
import phoenix as px

client = px.Client()

# Load queries
df = pd.read_csv("data/testset/evaluation/video_search_queries.csv")

# Upload to Phoenix
dataset = client.upload_dataset(
    dataset_name="golden_eval_v1",
    inputs=[{"query": row["query"]} for _, row in df.iterrows()],
    outputs=[{"expected_video": row["expected_video"]} for _, row in df.iterrows()]
)

print(f"Created dataset: {dataset.name} ({len(df)} examples)")
```

## Evaluation Workflows

### Workflow 1: Quick Quality Check

For rapid iteration during development:

```bash
# Single profile, single strategy, quality metrics only
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name dev_queries \
  --profiles video_colpali_smol500_mv_frame \
  --quality-evaluators

# View results immediately
open http://localhost:6006
```

### Workflow 2: Comprehensive Evaluation

For thorough analysis before production deployment:

```bash
# All profiles, all strategies, all evaluators
uv run python scripts/run_experiments_with_visualization.py \
  --dataset-name golden_eval_v1 \
  --profiles \
    video_colpali_smol500_mv_frame \
    video_videoprism_base_mv_chunk_30s \
  --test-multiple-strategies \
  --quality-evaluators \
  --llm-evaluators \
  --llm-model ollama/llava:7b

# Results in Phoenix with full metric breakdown
```

### Workflow 3: A/B Testing

Compare baseline vs optimized system:

```python
# Run baseline experiment
baseline_result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
    dataset_name="prod_queries",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["hybrid_float_bm25"],
    evaluators=[quality_scorer],
    config={"experiment_tag": "baseline"}
)

# Run optimized experiment (after GEPA optimization)
optimized_result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
    dataset_name="prod_queries",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["hybrid_float_bm25"],
    evaluators=[quality_scorer],
    config={"experiment_tag": "optimized_routing"}
)

# Compare in Phoenix dashboard
```

## Custom Evaluators

### Creating Custom Scorers

```python
from src.evaluation.core.base import BaseEvaluator

class CustomRelevanceScorer(BaseEvaluator):
    """Custom relevance scorer with domain-specific logic"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, query: str, results: list) -> dict:
        """Evaluate search results"""

        # Custom relevance logic
        relevance_scores = []
        for result in results:
            # Domain-specific scoring
            score = self._compute_domain_relevance(query, result)
            relevance_scores.append(score)

        # Aggregate metrics
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        high_quality_count = sum(1 for s in relevance_scores if s >= self.threshold)

        return {
            "avg_relevance": avg_relevance,
            "high_quality_ratio": high_quality_count / len(results),
            "scores": relevance_scores
        }

    def _compute_domain_relevance(self, query: str, result: dict) -> float:
        """Domain-specific relevance computation"""
        # Implement custom logic
        pass

# Use in experiments
custom_scorer = CustomRelevanceScorer(threshold=0.8)

result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
    dataset_name="domain_queries",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["hybrid_float_bm25"],
    evaluators=[custom_scorer]
)
```

## Troubleshooting

### Phoenix Not Recording Experiments

**Symptom**: Experiments don't appear in Phoenix dashboard

**Solutions:**
```bash
# 1. Check Phoenix is running
curl http://localhost:6006/health

# 2. Verify Phoenix client connection
python -c "import phoenix as px; client = px.Client(); print(client.get_datasets())"

# 3. Check telemetry export
export PHOENIX_COLLECTOR_ENDPOINT=localhost:4317
export PHOENIX_ENABLED=true

# 4. Force flush telemetry
python scripts/flush_telemetry.py
```

### Low Evaluation Scores

**Symptom**: All queries score poorly (<0.5)

**Causes:**
1. Wrong embedding model for query type
2. Schema not deployed for tenant
3. No documents ingested

**Solutions:**
```bash
# Check documents exist
curl "http://localhost:8080/document/v1/video_colpali_smol500_mv_frame_default/video/" | jq '.documents | length'

# Verify schema deployed
curl http://localhost:8080/ApplicationStatus | jq '.schemas'

# Re-ingest test videos
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --profile video_colpali_smol500_mv_frame
```

### Visual Judge Failures

**Symptom**: LLM evaluator returns errors

**Solutions:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required model
docker exec ollama ollama pull llava:7b

# Test LLM directly
curl http://localhost:11434/v1/chat/completions \
  -d '{"model": "llava:7b", "messages": [{"role": "user", "content": "test"}]}'
```

## Best Practices

### Dataset Design

1. **Diverse Queries**: Include simple, moderate, and complex queries
2. **Representative**: Match production query distribution
3. **Sufficient Size**: Minimum 50 queries for reliable metrics
4. **Ground Truth**: When available, include relevance labels

### Evaluation Frequency

- **Development**: After each significant change
- **Staging**: Daily with nightly test suite
- **Production**: Weekly with production query sample

### Metric Selection

- **Fast Iteration**: Quality metrics only (reference-free)
- **Comprehensive**: Quality + Visual + Retrieval metrics
- **Production**: Quality + Retrieval (with feedback labels)

### Experiment Naming

Use descriptive experiment names:
```python
experiment_name = f"{dataset_name}_{profile}_{strategy}_{date}"
# Example: "golden_eval_v1_colpali_hybrid_20251004"
```

## Related Documentation

- [Architecture Overview](architecture.md) - System design
- [Phoenix Integration](phoenix-integration.md) - Telemetry details
- [System Flows](system-flows.md) - Request traces
- [Optimization System](optimization-system.md) - GEPA evaluation

**Last Updated**: 2025-10-04
