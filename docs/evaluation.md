# Evaluation Guide

## Overview

The Cogniverse evaluation framework provides comprehensive assessment of video search quality using multiple evaluation strategies and metrics.

## Quick Start

### 1. Start Phoenix Server

```bash
# Local server
python scripts/start_phoenix.py

# Or use Docker
docker run -p 6006:6006 arizephoenix/phoenix:latest
```

### 2. Run Basic Evaluation

```bash
# Quality metrics evaluation
uv run python src/evaluation/core/experiment_tracker.py \
  --dataset-name golden_eval_v1 \
  --profiles video_colpali_smol500_mv_frame \
  --quality-evaluators

# With visual LLM evaluation
uv run python src/evaluation/core/experiment_tracker.py \
  --dataset-name test_visual \
  --profiles video_videoprism_base_mv_chunk_30s \
  --llm-evaluators \
  --llm-model ollama/llava:7b
```

## Evaluation Types

### 1. Quality Metrics (Reference-Free)

**Relevance Score**
- Semantic similarity between query and results
- Range: 0-1 (higher is better)
- Thresholds: High (≥0.8), Relevant (≥0.5), Low (<0.5)

**Diversity Score**
- Measures variety in search results
- Calculated as: 1 - avg(pairwise_similarity)
- Higher scores indicate more diverse results

**Distribution Analysis**
- Score spread and standard deviation
- Quality score: mean × (1 - std/2)
- Identifies retrieval consistency

**Temporal Coverage**
- For video queries with time ranges
- Measures coverage of video timeline
- Critical for moment retrieval

### 2. LLM-as-Judge Evaluation

**Visual Judge**
- Analyzes actual video frames
- 30 frames/video × 2 videos (60 total)
- Supports Ollama, Modal, OpenAI providers

**Setup LLM Models:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull evaluation models
ollama pull llava:7b          # Visual evaluation
ollama pull deepseek-r1:7b    # Text evaluation

# Start service
ollama serve
```

### 3. Retrieval Metrics

- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **Precision@k**: Relevant items in top-k
- **Recall@k**: Coverage of all relevant items
- **MAP**: Mean Average Precision
- **Temporal IoU**: For moment queries

## Creating Evaluation Datasets

### From CSV File

```csv
query,expected_videos,category
"person in red shirt",video1|video2,visual
"meeting discussion",video3,content
"last week's presentation",video4,temporal
```

```bash
uv run python scripts/create_dataset_from_csv.py \
  --csv-path data/queries.csv \
  --dataset-name my_eval_dataset
```

### From Production Traces

```bash
# Bootstrap from successful queries
uv run python scripts/bootstrap_dataset_from_traces.py \
  --hours 48 \
  --min-score 0.8 \
  --dataset-name production_dataset
```

### Synthetic Generation

```bash
# Generate from existing videos
uv run python scripts/generate_dataset_from_videos.py \
  --video-dir data/videos \
  --num-queries 100 \
  --use-llm \
  --dataset-name synthetic_dataset
```

## Running Experiments

### Command-Line Options

```bash
# Dataset options
--dataset-name NAME       # Existing dataset
--csv-path PATH          # Create from CSV
--force-new             # Recreate dataset

# Profile options
--profiles PROFILE [...]  # Ingestion profiles
--strategies STRAT [...]  # Search strategies
--all-strategies         # Test all strategies

# Evaluator options
--quality-evaluators     # Enable quality metrics
--llm-evaluators        # Enable LLM evaluation
--evaluator CONFIG      # Evaluator config name
--llm-model MODEL       # LLM model to use
```

### Example Experiments

**Compare Multiple Profiles:**
```bash
uv run python src/evaluation/core/experiment_tracker.py \
  --dataset-name benchmark_v1 \
  --profiles video_colpali_smol500_mv_frame video_videoprism_base_mv_chunk_30s \
  --all-strategies \
  --quality-evaluators
```

**Visual Evaluation:**
```bash
uv run python src/evaluation/core/experiment_tracker.py \
  --dataset-name visual_test \
  --profiles video_colpali_smol500_mv_frame \
  --strategies float_float binary_binary \
  --llm-evaluators \
  --evaluator visual_judge \
  --llm-model ollama/llava:7b
```

## Phoenix Dashboard

### Accessing Results

1. Open Phoenix UI: http://localhost:6006
2. Navigate to Experiments tab
3. View experiment comparisons
4. Analyze individual traces

### Dashboard Features

- **Traces View**: Detailed query traces
- **Datasets View**: Dataset management
- **Experiments View**: Side-by-side comparisons
- **Evaluations View**: Metric scores

### Analytics Dashboard

```bash
# Start Streamlit dashboard
streamlit run scripts/phoenix_dashboard_standalone.py

# Access at http://localhost:8501
```

Features:
- Real-time metrics
- Performance trends
- Error analysis
- Profile comparisons

## Configuration

### Evaluation Config

```json
{
  "evaluation": {
    "use_custom": true,
    "custom_metrics": ["diversity", "temporal_coherence"],
    "top_k": 10,
    "batch_size": 5,
    "evaluators": {
      "visual_judge": {
        "provider": "ollama",
        "model": "llava:7b",
        "base_url": "http://localhost:11434",
        "frames_per_video": 30,
        "max_videos": 2,
        "temperature": 0.1
      }
    }
  }
}
```

### Environment Variables

```bash
# Phoenix configuration
export PHOENIX_PORT=6006
export PHOENIX_WORKING_DIR=./data/phoenix

# Evaluation settings
export EVAL_BATCH_SIZE=10
export EVAL_PARALLEL_RUNS=false
export EVAL_TIMEOUT=300
```

## Interpreting Results

### Quality Scores

| Score Range | Interpretation | Action |
|------------|----------------|--------|
| 0.8-1.0 | Excellent | Ship to production |
| 0.6-0.8 | Good | Minor improvements needed |
| 0.4-0.6 | Fair | Significant improvements needed |
| 0.0-0.4 | Poor | Major issues to address |

### Common Patterns

**Low Relevance + High Diversity:**
- Retrieval is too broad
- Tighten search parameters

**High Relevance + Low Diversity:**
- Results too similar
- Consider result diversification

**Poor Temporal Coverage:**
- Missing video segments
- Check chunking strategy

## Best Practices

1. **Consistent Datasets**: Version and track datasets
2. **Regular Baselines**: Run weekly evaluations
3. **Multiple Metrics**: Don't rely on single metric
4. **Visual Inspection**: Manually review samples
5. **A/B Testing**: Compare configurations
6. **Production Monitoring**: Track live metrics
7. **Error Analysis**: Investigate failures
8. **Documentation**: Record configuration changes

## Troubleshooting

### Phoenix Connection Issues
```bash
# Check Phoenix status
curl http://localhost:6006/health

# Restart Phoenix
docker restart phoenix
```

### LLM Evaluation Failures
```bash
# Check Ollama status
ollama list
ollama serve

# Test model
ollama run llava:7b "test"
```

### Out of Memory
- Reduce batch size
- Limit frames per video
- Use smaller models

### Slow Evaluation
- Enable parallel processing
- Use GPU acceleration
- Reduce dataset size for testing

## Advanced Usage

### Custom Evaluators

```python
from src.evaluation.evaluators import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, query, results):
        # Custom evaluation logic
        score = calculate_custom_metric(query, results)
        return {"custom_score": score}
```

### Programmatic Evaluation

```python
from src.evaluation.pipeline import EvaluationPipeline

# Initialize pipeline
pipeline = EvaluationPipeline(config_path="config.json")

# Run evaluation
results = await pipeline.run_evaluation(
    dataset_name="my_dataset",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["float_float"],
    evaluators=["quality", "visual_judge"]
)

# Analyze results
print(f"Mean relevance: {results['relevance']['mean']:.3f}")
print(f"Mean diversity: {results['diversity']['mean']:.3f}")
```