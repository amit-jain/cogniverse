# Advanced Evaluation Framework Documentation

## Table of Contents
1. [Overview](#overview)
2. [Evaluator Types](#evaluator-types)
3. [Quality Evaluators](#quality-evaluators)
4. [LLM-as-Judge Evaluators](#llm-as-judge-evaluators)
5. [Running Experiments](#running-experiments)
6. [Configuration](#configuration)
7. [Results Analysis](#results-analysis)
8. [Best Practices](#best-practices)

## Overview

The advanced evaluation framework provides comprehensive assessment of video retrieval quality through multiple evaluator types:

- **Statistical Evaluators**: Fast, deterministic metrics (relevance, diversity, distribution)
- **LLM Evaluators**: Deep semantic understanding using language models
- **Golden Dataset Evaluators**: Comparison against ground truth
- **Hybrid Approaches**: Combining multiple evaluation strategies

## Evaluator Types

### 1. Reference-Free Evaluators

Evaluate retrieval quality without ground truth data:

```python
# Quality evaluators (statistical)
- SyncQueryResultRelevanceEvaluator  # Score distribution analysis
- SyncResultDiversityEvaluator       # Unique video ratio
- SyncResultDistributionEvaluator    # Statistical properties
- SyncTemporalCoverageEvaluator      # Video segment coverage

# LLM evaluators
- SyncLLMReferenceFreeEvaluator      # Semantic relevance judgment
```

### 2. Reference-Based Evaluators

Compare against expected results:

```python
- SyncGoldenDatasetEvaluator         # Exact match against golden set
- SyncLLMReferenceBasedEvaluator     # LLM comparison with ground truth
```

### 3. Hybrid Evaluators

```python
- SyncLLMHybridEvaluator             # Combines reference-free and reference-based
```

## Quality Evaluators

Quality evaluators assess search results without requiring ground truth labels. Located in `src/evaluation/evaluators/sync_reference_free.py`.

### Relevance Evaluator (`SyncRelevanceScoreEvaluator`)
Measures semantic similarity between query and results:
- **Method**: Cosine similarity using sentence transformers
- **Score Range**: 0-1 (higher is better)
- **Thresholds**:
  - High relevance: ≥ 0.8
  - Relevant: ≥ 0.5
  - Low relevance: < 0.5
- **Usage**: Automatically included with `--quality-evaluators` flag

### Diversity Evaluator (`SyncDiversityEvaluator`)
Measures variety in search results:
- **Method**: Average pairwise distance between result embeddings
- **Score Range**: 0-1 (higher indicates more diverse results)
- **Benefits**: Identifies redundant or overly similar results
- **Calculation**: `1 - avg(cosine_similarity(result_i, result_j))`

### Distribution Evaluator (`SyncResultDistributionEvaluator`)
Analyzes statistical properties of result scores:
- **Metrics**:
  - Score spread (max - min)
  - Standard deviation
  - Rank correlation
- **Quality Score**: `mean × (1 - std/2)`
- **Purpose**: Identifies retrieval consistency issues

### Temporal Coverage Evaluator (`SyncTemporalCoverageEvaluator`)
Evaluates temporal distribution for video segments:
- **Method**: Analyzes timestamp coverage across video duration
- **Score**: Ratio of covered time to total video duration
- **Use Case**: Video moment retrieval evaluation
- **Merging**: Overlapping segments are merged before calculation

## LLM-as-Judge Evaluators

### Prerequisites

1. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull models:
```bash
# For text evaluation
ollama pull deepseek-r1:7b

# For visual evaluation
ollama pull llava:7b        # Multimodal vision model
# or
ollama pull qwen2-vl:7b     # Alternative vision model
```

3. Start Ollama service:
```bash
ollama serve  # Default: http://localhost:11434
```

### Visual Judge Evaluator

The most advanced evaluator that analyzes actual video frames using multimodal LLMs.

**Configuration** (`configs/config.json`):
```json
"evaluators": {
  "visual_judge": {
    "provider": "ollama",
    "model": "llava:7b",
    "base_url": "http://localhost:11434",
    "frames_per_video": 30,
    "max_videos": 2,
    "max_total_frames": 60
  }
}
```

**Features**:
- Extracts up to 60 frames from top videos
- Supports multiple providers (Ollama, Modal, OpenAI)
- Configurable frame extraction strategy
- Actual visual understanding of content

**Usage**:
```bash
uv run python scripts/run_experiments_with_visualization.py \
  --llm-evaluators \
  --evaluator visual_judge
```

**Location**: `src/evaluation/evaluators/configurable_visual_judge.py`

### Reference-Free LLM Evaluation

Evaluates query-result relevance using LLM judgment:

```python
evaluator = SyncLLMReferenceFreeEvaluator(
    model_name="deepseek-r1:7b",
    base_url="http://localhost:11434"
)
```

**What it evaluates:**
- Semantic relevance of results to query
- Ranking quality
- Result diversity
- Overall search quality

**Example prompt sent to LLM:**
```
Query: "person playing sports outdoors"

Search Results:
1. Video: sports_001 (Score: 0.920)
2. Video: outdoor_activity (Score: 0.850)
3. Video: sports_002 (Score: 0.780)

Please evaluate these search results...
```

### Reference-Based LLM Evaluation

Compares results against ground truth:

```python
evaluator = SyncLLMReferenceBasedEvaluator(
    model_name="deepseek-r1:7b",
    fetch_metadata=True  # Fetch video metadata from Vespa
)
```

**What it evaluates:**
- Precision: Retrieved videos that are relevant
- Recall: Relevant videos that were retrieved
- Ranking: Are expected videos ranked highly?
- F1 Score: Harmonic mean of precision and recall

**Metadata Integration:**
- Fetches video titles, descriptions from Vespa
- Enriches LLM context for better judgment
- Caches metadata for performance

### Hybrid LLM Evaluation

Combines both approaches:

```python
evaluator = SyncLLMHybridEvaluator(
    model_name="deepseek-r1:7b",
    reference_weight=0.5  # 50% reference, 50% relevance
)
```

**Benefits:**
- Comprehensive evaluation
- Balances semantic relevance with correctness
- Configurable weighting

## Running Experiments

### Basic Usage

```bash
# With quality evaluators only (default)
uv run python scripts/run_experiments_with_visualization.py

# With LLM evaluators
uv run python scripts/run_experiments_with_visualization.py --llm-evaluators

# Specific configuration
uv run python scripts/run_experiments_with_visualization.py \
    --llm-evaluators \
    --llm-model "llama2:7b" \
    --llm-base-url "http://localhost:11434" \
    --profiles frame_based_colpali direct_video_global \
    --strategies binary_binary float_float
```

### Command-Line Options

```bash
# Evaluator Options
--quality-evaluators          # Enable quality evaluators (default: True)
--no-quality-evaluators       # Disable quality evaluators
--llm-evaluators             # Enable LLM evaluators (default: False)
--llm-model MODEL            # LLM model to use (default: deepseek-r1:7b)
--llm-base-url URL           # LLM API endpoint (default: http://localhost:11434)

# Experiment Options
--profiles PROFILE [...]      # Specific profiles to test
--strategies STRATEGY [...]   # Specific strategies to test
--all-strategies             # Test all available strategies
--dataset-name NAME          # Use specific dataset
--csv-path PATH              # Load queries from CSV
--force-new                  # Force create new dataset
```

### Example Workflows

#### 1. Quick Quality Assessment
```bash
# Fast statistical evaluation
uv run python scripts/run_experiments_with_visualization.py \
    --no-llm-evaluators \
    --profiles frame_based_colpali
```

#### 2. Deep Semantic Evaluation
```bash
# Comprehensive LLM-based evaluation
uv run python scripts/run_experiments_with_visualization.py \
    --llm-evaluators \
    --llm-model "deepseek-r1:7b"
```

#### 3. A/B Testing Configurations
```bash
# Compare multiple strategies
uv run python scripts/run_experiments_with_visualization.py \
    --llm-evaluators \
    --profiles direct_video_global \
    --strategies binary_binary float_float phased
```

## Configuration

### Phoenix Experiment Runner

```python
from src.evaluation.phoenix_experiments_final import PhoenixExperimentRunner

runner = PhoenixExperimentRunner(
    experiment_project_name="experiments",
    enable_quality_evaluators=True,
    enable_llm_evaluators=True,
    llm_model="deepseek-r1:7b",
    llm_base_url="http://localhost:11434"
)
```

### Custom Evaluator Configuration

```python
from src.evaluation.evaluators.llm_judge import create_llm_evaluators

# Create custom evaluators
evaluators = create_llm_evaluators(
    model_name="mistral:7b",
    base_url="http://remote-ollama:11434",
    include_hybrid=True
)

# Use in experiments
runner.run_experiment(
    profile="frame_based_colpali",
    strategy="binary_binary",
    dataset=dataset,
    evaluators=evaluators
)
```

## Results Analysis

### Viewing Results

1. **Console Output**: Immediate feedback during experiments
2. **Phoenix UI**: http://localhost:6006/projects/experiments
3. **CSV Export**: `outputs/experiment_results/`
4. **JSON Details**: `outputs/experiment_results/experiment_details_*.json`
5. **HTML Report**: Generated integrated report with visualizations

### Understanding Scores

#### Quality Evaluator Scores (0-1 scale):
- **0.8-1.0**: Excellent
- **0.6-0.8**: Good
- **0.4-0.6**: Fair
- **0.0-0.4**: Poor

#### LLM Evaluator Labels:
- **Reference-Free**: `highly_relevant`, `relevant`, `partially_relevant`, `not_relevant`
- **Reference-Based**: `excellent_match`, `good_match`, `partial_match`, `poor_match`
- **Hybrid**: `excellent`, `good`, `fair`, `poor`

### Metrics Interpretation

| Metric | Description | Good Range |
|--------|-------------|------------|
| Relevance | Query-result match | > 0.7 |
| Diversity | Unique results ratio | > 0.6 |
| Distribution | Score consistency | > 0.6 |
| Temporal Coverage | Time segment coverage | > 0.5 |
| Precision | Correct retrievals | > 0.7 |
| Recall | Found relevant items | > 0.6 |
| F1 Score | Balance of P & R | > 0.65 |

## Best Practices

### 1. Evaluator Selection

**Use Quality Evaluators when:**
- Need fast, deterministic results
- Running many experiments
- Initial configuration testing

**Use LLM Evaluators when:**
- Need semantic understanding
- Final quality assessment
- Comparing subtle differences
- Have ground truth available

### 2. Performance Optimization

```bash
# For faster experiments
--no-llm-evaluators  # Skip LLM calls
--profiles frame_based_colpali  # Test single profile
--strategies binary_binary  # Test single strategy

# For comprehensive analysis
--llm-evaluators
--all-strategies
```

### 3. Dataset Management

```bash
# List available datasets
uv run python scripts/run_experiments_with_visualization.py --list-datasets

# Create custom dataset
uv run python scripts/run_experiments_with_visualization.py \
    --csv-path data/custom_queries.csv \
    --dataset-name custom_eval_v1
```

### 4. Troubleshooting

**LLM Evaluator Issues:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test with mock responses
# (Evaluators fall back to mock when Ollama unavailable)

# Use different model
--llm-model "llama2:7b"

# Use remote Ollama
--llm-base-url "http://remote-server:11434"
```

**Memory Issues:**
```bash
# Reduce concurrent evaluations
# Edit src/evaluation/phoenix_experiments_final.py
# Set concurrency=1 in run_experiment()
```

### 5. Production Recommendations

1. **Development**: Use quality evaluators for rapid iteration
2. **Staging**: Add LLM evaluators for semantic validation
3. **Production**: Monitor with quality evaluators, periodic LLM audits
4. **CI/CD**: Quality evaluators in PR checks, LLM evaluators in nightly tests

## Advanced Usage

### Custom Evaluator Implementation

```python
from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult

class CustomEvaluator(Evaluator):
    def evaluate(self, *, input=None, output=None, **kwargs):
        # Your evaluation logic
        score = calculate_custom_metric(output)
        
        return EvaluationResult(
            score=score,
            label="custom_label",
            explanation="Custom evaluation logic",
            metadata={"custom_field": "value"}
        )
```

### Batch Evaluation

```python
# Evaluate multiple datasets
datasets = ["golden_eval_v1", "challenging_queries_v2"]

for dataset_name in datasets:
    runner.run_experiment(
        profile="frame_based_colpali",
        strategy="binary_binary",
        dataset_name=dataset_name
    )
```

### Export for Analysis

```python
import pandas as pd
import json

# Load experiment results
with open("outputs/experiment_results/latest.json") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results["experiments"])

# Analyze
print(df.groupby("profile")["score"].mean())
print(df.pivot_table(
    values="score",
    index="profile",
    columns="strategy"
))
```

## Automatic Golden Dataset Creation

### Overview

The framework can automatically identify challenging queries from historical traces and create golden datasets for continuous improvement.

### How It Works

1. **Trace Analysis**: Examines Phoenix traces over a specified time period
2. **Performance Scoring**: Identifies queries with consistently low evaluation scores
3. **Pattern Recognition**: Groups similar failing queries
4. **Dataset Generation**: Creates golden datasets with expected results

### Usage

#### Generate Golden Dataset from Traces

```bash
# Basic usage - analyze last 48 hours
uv run python scripts/create_golden_dataset_from_traces.py

# Custom parameters
uv run python scripts/create_golden_dataset_from_traces.py \
    --hours 72 \                    # Look back 72 hours
    --min-occurrences 3 \            # Query must appear 3+ times
    --score-threshold 0.4 \          # Max avg score for inclusion
    --top-n 30 \                     # Include top 30 queries
    --csv                            # Also export as CSV
```

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hours` | 48 | Hours back to analyze |
| `--min-occurrences` | 2 | Minimum query occurrences |
| `--score-threshold` | 0.5 | Maximum avg score for challenging queries |
| `--top-n` | 20 | Number of queries to include |
| `--output` | auto | Output file path |
| `--csv` | False | Also save as CSV |
| `--dry-run` | False | Preview without saving |

### Dataset Management

#### List Available Datasets

```bash
uv run python scripts/manage_golden_datasets.py list
```

Output:
```
Available golden datasets:
  auto_golden_dataset_20240106_143022.json     12.3KB  2024-01-06 14:30
  manual_golden_dataset.json                    8.5KB  2024-01-05 10:15
  merged_dataset_20240105.json                 20.1KB  2024-01-05 16:22
```

#### Merge Multiple Datasets

```bash
# Union strategy (combine all queries)
uv run python scripts/manage_golden_datasets.py merge \
    dataset1.json dataset2.json \
    --strategy union \
    --output merged.json

# Intersection (only common queries)
uv run python scripts/manage_golden_datasets.py merge \
    dataset1.json dataset2.json \
    --strategy intersection
```

#### Filter Datasets

```bash
# Filter by score and difficulty
uv run python scripts/manage_golden_datasets.py filter \
    dataset.json \
    --max-score 0.3 \
    --difficulty challenging \
    --min-videos 2
```

#### View Statistics

```bash
uv run python scripts/manage_golden_datasets.py stats dataset.json
```

Output:
```
Dataset Statistics for auto_golden_dataset.json
==================================================
Total queries: 20
Total expected videos: 85
Avg videos per query: 4.2
Queries with scores: 15

Score Statistics:
  Average: 0.342
  Min: 0.125
  Max: 0.498

Difficulty Distribution:
  challenging: 15
  medium: 5
```

#### Update Existing Dataset

```bash
# Add new queries to existing dataset
uv run python scripts/manage_golden_datasets.py update \
    base_dataset.json \
    new_queries.json \
    --output updated.json

# Overwrite existing queries
uv run python scripts/manage_golden_datasets.py update \
    base_dataset.json \
    new_queries.json \
    --overwrite
```

#### Export as Python Code

```bash
# Generate Python code for integration
uv run python scripts/manage_golden_datasets.py export \
    dataset.json \
    --output golden_dataset.py
```

### Integration with Evaluators

The golden dataset evaluator automatically loads generated datasets:

```python
from src.evaluation.evaluators.golden_dataset import load_golden_dataset_from_file

# Automatically loads latest generated dataset
dataset = load_golden_dataset_from_file()

# Or specify a specific file
dataset = load_golden_dataset_from_file("data/golden_datasets/custom.json")
```

### Automated Workflow

#### Continuous Improvement Pipeline

1. **Run experiments** with current configuration
2. **Analyze traces** to find low-scoring queries
3. **Generate golden dataset** from failures
4. **Update evaluators** with new challenging queries
5. **Re-run experiments** to validate improvements

#### Example Automation Script

```bash
#!/bin/bash
# Weekly golden dataset update

# 1. Generate dataset from last week's traces
uv run python scripts/create_golden_dataset_from_traces.py \
    --hours 168 \
    --score-threshold 0.4 \
    --output data/golden_datasets/weekly_$(date +%Y%m%d).json

# 2. Merge with existing master dataset
uv run python scripts/manage_golden_datasets.py merge \
    data/golden_datasets/master.json \
    data/golden_datasets/weekly_*.json \
    --strategy union \
    --output data/golden_datasets/master_new.json

# 3. Run experiments with new dataset
uv run python scripts/run_experiments_with_visualization.py \
    --dataset-path data/golden_datasets/master_new.json
```

### Best Practices

#### 1. Regular Updates
- Generate golden datasets weekly or after major changes
- Review and curate automatically generated queries
- Track dataset evolution over time

#### 2. Score Thresholds
- **< 0.3**: Very challenging queries (system limitations)
- **0.3-0.5**: Moderately challenging (improvement targets)
- **0.5-0.7**: Edge cases (refinement opportunities)

#### 3. Dataset Curation
- Review auto-generated queries for relevance
- Remove queries that are genuinely ambiguous
- Add manual annotations for expected results
- Balance dataset across different query types

#### 4. Version Control
- Track golden datasets in git for history
- Document changes and rationale
- Tag datasets with experiment results

### Troubleshooting

#### No Traces Found
```bash
# Check Phoenix is running and has data
curl http://localhost:6006/api/traces

# Increase time window
--hours 168  # Look back 1 week

# Lower occurrence threshold
--min-occurrences 1
```

#### Empty Golden Dataset
```bash
# Increase score threshold (include higher scores)
--score-threshold 0.7

# Reduce minimum occurrences
--min-occurrences 1

# Check evaluation scores exist
uv run python -c "import phoenix as px; print(px.Client().get_evaluations_dataframe())"
```

#### Dataset Too Large
```bash
# Limit number of queries
--top-n 10

# Filter by score
--score-threshold 0.3  # Only very low scores

# Filter after generation
uv run python scripts/manage_golden_datasets.py filter \
    large_dataset.json \
    --max-score 0.4 \
    --output filtered.json
```

## Dataset Management

### Golden Dataset Creation

Automatically generate evaluation datasets from Phoenix traces:

```bash
# Create from low-scoring traces
uv run python scripts/create_golden_dataset_from_traces.py \
  --hours 24 \
  --min-score 0.7 \
  --output data/golden_datasets/high_quality.csv

# Create from specific queries
uv run python scripts/create_golden_dataset_from_traces.py \
  --query-pattern "outdoor*" \
  --min-occurrences 3 \
  --output data/golden_datasets/outdoor_queries.csv
```

**Location**: `scripts/create_golden_dataset_from_traces.py`

### Dataset Registry

Central management for all evaluation datasets:

```bash
# List all datasets
uv run python scripts/manage_golden_datasets.py list

# Merge datasets
uv run python scripts/manage_golden_datasets.py merge \
  dataset1.csv dataset2.csv \
  --output merged.csv

# Filter datasets
uv run python scripts/manage_golden_datasets.py filter \
  dataset.csv \
  --max-score 0.5 \
  --output filtered.csv

# Get statistics
uv run python scripts/manage_golden_datasets.py stats dataset.csv

# Export for sharing
uv run python scripts/manage_golden_datasets.py export \
  dataset.csv \
  --format json \
  --output dataset.json
```

**Registry Location**: `configs/dataset_registry.json`

## Appendix

### Supported Models

#### Text Evaluation Models (Ollama)
- `deepseek-r1:7b` (Recommended for text)
- `llama2:7b`, `llama2:13b`
- `mistral:7b`
- `mixtral:8x7b`

#### Visual Evaluation Models (Ollama)
- `llava:7b` (Recommended for vision)
- `qwen2-vl:7b`
- `bakllava:7b`

### Environment Variables

```bash
# Phoenix configuration
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
export PHOENIX_PROJECT_NAME="experiments"

# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="deepseek-r1:7b"

# Visual evaluation
export FRAMES_PER_VIDEO=30
export MAX_VIDEOS=2
```

### Performance Characteristics

| Evaluator Type | Speed | Accuracy | Resource Usage |
|---------------|-------|----------|----------------|
| Quality Evaluators | ~100ms | Medium | Low (CPU only) |
| Text LLM Judge | ~2s | High | Medium |
| Visual Judge (60 frames) | ~10s | Highest | High (GPU recommended) |
| Phoenix Upload | ~50ms | N/A | Low |

### Related Documentation

- [EVALUATION_FRAMEWORK.md](./EVALUATION_FRAMEWORK.md) - Basic framework overview
- [VISUAL_EVALUATOR_CONFIG.md](./VISUAL_EVALUATOR_CONFIG.md) - Visual evaluation details
- [PHOENIX_DASHBOARD.md](./PHOENIX_DASHBOARD.md) - Phoenix UI guide
- [Phoenix Experiments](https://docs.arize.com/phoenix/experiments) - Official Phoenix docs