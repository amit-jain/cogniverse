# Evaluation Framework Capabilities

## Overview
The evaluation framework provides comprehensive assessment of video search and retrieval systems through multiple evaluation approaches integrated with Phoenix Arize for observability.

## 1. Quality Evaluators (Reference-Free)

These evaluators assess search results without ground truth labels:

### 1.1 Relevance Score Evaluator
- **Purpose**: Measures query-result relevance using semantic similarity
- **Method**: Cosine similarity between query and result embeddings
- **Score Range**: 0-1 (higher is better)
- **Location**: `src/evaluation/evaluators/sync_reference_free.py::SyncRelevanceScoreEvaluator`

### 1.2 Diversity Evaluator
- **Purpose**: Measures diversity among search results
- **Method**: Average pairwise distance between result embeddings
- **Score Range**: 0-1 (higher is more diverse)
- **Location**: `src/evaluation/evaluators/sync_reference_free.py::SyncDiversityEvaluator`

### 1.3 Result Distribution Evaluator
- **Purpose**: Analyzes distribution patterns in search results
- **Method**: Statistical analysis of result scores and ranks
- **Metrics**: Score spread, rank correlation, distribution uniformity
- **Location**: `src/evaluation/evaluators/sync_reference_free.py::SyncResultDistributionEvaluator`

### 1.4 Temporal Coverage Evaluator
- **Purpose**: Measures temporal coverage across video segments
- **Method**: Analyzes timestamp distribution in results
- **Score Range**: 0-1 (1 = perfect temporal coverage)
- **Location**: `src/evaluation/evaluators/sync_reference_free.py::SyncTemporalCoverageEvaluator`

## 2. LLM-as-Judge Evaluators

### 2.1 Visual Judge (Multimodal)
- **Purpose**: Evaluates visual relevance using multimodal LLMs
- **Capabilities**:
  - Extracts up to 60 frames from top videos
  - Supports multiple providers (Ollama, Modal, OpenAI)
  - Configurable frame extraction (30 frames/video × 2 videos default)
- **Models Supported**:
  - LLaVA (via Ollama)
  - Qwen2-VL (via Modal/Ollama)
  - GPT-4V (via OpenAI)
- **Configuration**: `configs/config.json::evaluators.visual_judge`
- **Location**: `src/evaluation/evaluators/configurable_visual_judge.py`

### 2.2 Text-Based LLM Judge
- **Purpose**: Evaluates text relevance and quality
- **Types**:
  - **Reference-Free**: Judges query-result relevance
  - **Reference-Based**: Compares against ground truth
  - **Hybrid**: Combines both approaches
- **Models**: Any Ollama model (default: deepseek-r1:7b)
- **Location**: `src/evaluation/evaluators/llm_judge.py`

## 3. Phoenix Integration

### 3.1 Experiment Management
- **Dataset Upload**: Automatic dataset creation in Phoenix
- **Experiment Tracking**: Each run creates traceable experiment
- **URL**: `http://localhost:6006/datasets/{dataset_id}/experiments`
- **Location**: `src/evaluation/phoenix_experiments_final.py`

### 3.2 Span Tracing
- **Components Traced**:
  - Video ingestion pipeline
  - Search operations (Vespa queries)
  - Query encoding (ColPali, VideoPrism)
  - Evaluation runs
- **Metrics**: Latency, token usage, error rates
- **Location**: `src/instrumentation/phoenix_instrumentation.py`

### 3.3 Evaluation Storage
- **Automatic Upload**: All evaluation results uploaded to Phoenix
- **Comparison**: Side-by-side experiment comparison
- **Analysis**: Built-in visualization and metrics

## 4. Dataset Management

### 4.1 Golden Dataset Creation
- **Automatic Generation**: Creates datasets from low-scoring traces
- **Configurable Thresholds**: Score, relevance, latency filters
- **Format**: CSV with query, expected results, metadata
- **Location**: `scripts/create_golden_dataset_from_traces.py`

### 4.2 Dataset Registry
- **Central Registry**: `configs/dataset_registry.json`
- **Operations**: List, merge, filter, export datasets
- **Management Tool**: `scripts/manage_golden_datasets.py`

## 5. Experiment Execution

### 5.1 Main Runner Script
```bash
uv run python scripts/run_experiments_with_visualization.py \
  --profiles frame_based_colpali \
  --strategies binary_binary \
  --quality-evaluators \
  --llm-evaluators \
  --evaluator visual_judge \
  --dataset-name golden_eval_v1
```

### 5.2 Supported Profiles
- `frame_based_colpali`: ColPali model on video frames
- `frame_based_colqwen`: ColQwen model on video frames
- `direct_video_global`: VideoPrism global embeddings
- `direct_video_global_large`: VideoPrism large model

### 5.3 Search Strategies
- `binary_binary`: Binary quantized embeddings
- `float_float`: Full precision embeddings
- `float_binary`: Mixed precision
- `maxsim`: Maximum similarity scoring

## 6. Visualization & Analysis

### 6.1 Phoenix Dashboard
```bash
uv run streamlit run scripts/phoenix_dashboard_standalone.py
```
- **Features**: Real-time metrics, evaluation results, span analysis
- **URL**: `http://localhost:8501`

### 6.2 Experiment Results
- **Output Directory**: `outputs/experiment_results/`
- **Formats**: CSV summaries, detailed JSON logs
- **Metrics**: Average scores, per-query breakdowns

## 7. Configuration

### 7.1 Main Config (`configs/config.json`)
```json
{
  "evaluators": {
    "visual_judge": {
      "provider": "ollama",
      "model": "llava:7b",
      "frames_per_video": 30,
      "max_videos": 2,
      "max_total_frames": 60
    }
  },
  "evaluation_datasets": {
    "default": "golden_eval_v1",
    "registry_path": "configs/dataset_registry.json"
  }
}
```

### 7.2 Environment Variables
- `PHOENIX_API_KEY`: Phoenix authentication
- `PHOENIX_ENDPOINT`: Phoenix server URL (default: http://localhost:6006)

## 8. Advanced Features

### 8.1 Batch Evaluation
- Parallel evaluation across multiple queries
- Progress bars with ETA
- Automatic retry on failures

### 8.2 Caching
- Frame extraction caching
- Embedding caching
- Results caching in Phoenix

### 8.3 Error Handling
- Graceful degradation on evaluator failures
- Detailed error logging
- Partial result recovery

## Usage Examples

### Basic Evaluation Run
```bash
# Quality evaluators only
uv run python scripts/run_experiments_with_visualization.py \
  --profiles frame_based_colpali \
  --quality-evaluators

# Visual evaluation with LLaVA
uv run python scripts/run_experiments_with_visualization.py \
  --profiles frame_based_colpali \
  --llm-evaluators \
  --evaluator visual_judge

# Full evaluation suite
uv run python scripts/run_experiments_with_visualization.py \
  --profiles frame_based_colpali direct_video_global \
  --strategies binary_binary maxsim \
  --quality-evaluators \
  --llm-evaluators \
  --dataset-name golden_eval_v1
```

### Create Golden Dataset
```bash
# From Phoenix traces
uv run python scripts/create_golden_dataset_from_traces.py \
  --hours 24 \
  --min-score 0.7 \
  --output data/golden_datasets/high_quality.csv

# Manage datasets
uv run python scripts/manage_golden_datasets.py list
uv run python scripts/manage_golden_datasets.py merge dataset1 dataset2 --output merged
```

## Performance Characteristics

| Evaluator Type | Speed | Accuracy | Resource Usage |
|---------------|-------|----------|----------------|
| Quality Evaluators | Fast (~100ms) | Medium | Low (CPU only) |
| Text LLM Judge | Medium (~2s) | High | Medium (model dependent) |
| Visual Judge | Slow (~10s) | Highest | High (60 frames processed) |
| Phoenix Upload | Fast (~50ms) | N/A | Low (async) |

## Limitations

1. **Visual Evaluation**: 
   - Requires actual video files
   - Limited by model context window
   - Frame extraction adds latency

2. **LLM Judges**:
   - Dependent on model quality
   - May have biases
   - Costs for API-based models

3. **Phoenix Integration**:
   - Requires Phoenix server running
   - Storage grows with experiments
   - Network dependency for remote Phoenix