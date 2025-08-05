# Cogniverse Evaluation Framework Documentation

## Overview

The Cogniverse Evaluation Framework provides comprehensive evaluation capabilities for the video RAG system using:
- **Inspect AI**: Structured task-based evaluation with custom solvers and scorers
- **Arize Phoenix**: Real-time tracing, observability, dataset management, and experiment tracking

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Phoenix Server Setup](#phoenix-server-setup)
3. [Running Evaluations](#running-evaluations)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## Installation & Setup

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized Phoenix)
- 8GB+ RAM recommended
- 10GB+ disk space for Phoenix data

### 1. Install Dependencies

```bash
# Install evaluation framework dependencies
uv pip install inspect-ai arize-phoenix phoenix-evals opentelemetry-api opentelemetry-sdk

# Or install from pyproject.toml
uv sync
```

### 2. Phoenix Server Setup

Phoenix can be run in several ways:

#### Option A: Local Phoenix Server (Recommended for Development)

```bash
# Start Phoenix with persistent storage
python scripts/start_phoenix.py

# Phoenix will be available at http://localhost:6006
```

#### Option B: Docker Deployment (Recommended for Production)

```bash
# Using docker-compose for persistent storage
docker-compose -f docker/phoenix-compose.yml up -d

# Phoenix will be available at http://localhost:6006
```

#### Option C: Manual Setup

```bash
# Create data directory
mkdir -p data/phoenix

# Set environment variable for data persistence
export PHOENIX_WORKING_DIR="$(pwd)/data/phoenix"

# Start Phoenix
python -m phoenix.server.main serve

# Or with custom port
python -m phoenix.server.main serve --port 6007
```

### 3. Verify Installation

```bash
# Check Phoenix is running
curl http://localhost:6006/health

# Run quick test evaluation
python scripts/run_evaluation.py test
```

## Phoenix Server Setup

### Data Persistence

Phoenix stores traces, datasets, and experiments in a working directory. To ensure data persistence:

1. **Set the working directory**:
```bash
export PHOENIX_WORKING_DIR=/path/to/persistent/storage
```

2. **Use the provided startup script**:
```bash
python scripts/start_phoenix.py --data-dir ./data/phoenix
```

### Docker Deployment with Volumes

The provided Docker setup includes persistent volumes:

```yaml
# docker/phoenix-compose.yml
version: '3.8'

services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
    volumes:
      - phoenix-data:/data
      - ./data/phoenix:/phoenix-data
    environment:
      - PHOENIX_WORKING_DIR=/phoenix-data
      - PHOENIX_PORT=6006
    restart: unless-stopped

volumes:
  phoenix-data:
    driver: local
```

### Configuration

Phoenix can be configured via environment variables:

```bash
# Phoenix configuration
export PHOENIX_PORT=6006                    # Server port
export PHOENIX_WORKING_DIR=./data/phoenix   # Data directory
export PHOENIX_ENABLE_PROMETHEUS=true       # Enable metrics
export PHOENIX_MAX_TRACES=100000           # Max traces to store
export PHOENIX_ENABLE_CORS=true            # Enable CORS for API access
```

## Running Evaluations

### Full Evaluation

Run a comprehensive evaluation across multiple profiles and strategies:

```bash
python scripts/run_evaluation.py full \
  --name "video_rag_eval_v1" \
  --profiles frame_based_colpali direct_video_global \
  --strategies binary_binary float_float hybrid_binary_bm25 \
  --tasks video_retrieval_accuracy temporal_understanding \
  --config configs/evaluation/eval_config.yaml
```

### Batch Evaluation on Existing Traces

Evaluate traces that have already been collected:

```bash
# Create dataset from queries
python scripts/create_phoenix_dataset.py \
  --name "eval_dataset_v1" \
  --queries data/eval/video_queries.json

# Run batch evaluation
python scripts/run_evaluation.py batch \
  --dataset eval_dataset_v1 \
  --trace-ids trace_id_1 trace_id_2 trace_id_3
```

### Quick Test

Run a minimal evaluation for testing:

```bash
python scripts/run_evaluation.py test --config configs/evaluation/eval_config.yaml
```

### Using Configuration Files

Create a custom evaluation configuration:

```yaml
# configs/evaluation/custom_eval.yaml
evaluation:
  name: "Custom Video RAG Evaluation"
  
  profiles:
    - frame_based_colpali
    - direct_video_global_large
  
  strategies:
    - binary_binary
    - phased
  
  metrics:
    retrieval:
      - mrr
      - ndcg_at_5
      - precision_at_5
  
  phoenix:
    instrumentation:
      enabled: true
    monitoring:
      enabled: true
```

Run with custom config:

```bash
python scripts/run_evaluation.py full \
  --name "custom_eval" \
  --config configs/evaluation/custom_eval.yaml
```

## Understanding Results

### Phoenix Dashboard

Access the Phoenix dashboard at http://localhost:6006 to view:

1. **Traces View**: All evaluation traces with timing and metadata
2. **Datasets View**: Managed evaluation datasets with versions
3. **Experiments View**: Experiment results and comparisons
4. **Evaluations View**: Scores attached to traces

### Evaluation Reports

Reports are generated in multiple formats:

```
outputs/
├── evaluations/        # JSON evaluation results
│   └── video_rag_eval_v1_20240115_143022.json
├── reports/           # HTML and Markdown reports
│   ├── video_rag_eval_v1_20240115_143022.html
│   └── video_rag_eval_v1_20240115_143022.md
└── inspect_logs/      # Inspect AI detailed logs
    └── video_rag_eval_v1/
```

### Metrics Explained

#### Retrieval Metrics
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality
- **Precision@k**: Fraction of relevant results in top-k
- **Recall@k**: Fraction of all relevant results found in top-k

#### Performance Metrics
- **Latency P50/P95/P99**: Response time percentiles
- **Throughput**: Queries per second
- **Error Rate**: Percentage of failed queries

### Viewing Evaluation Scores in Phoenix

1. **Navigate to Traces**: http://localhost:6006/traces
2. **Filter by evaluation**: Use the filter `metadata.evaluation_name = "your_eval_name"`
3. **View scores**: Click on any trace to see attached evaluation scores
4. **Export results**: Use the export button to download traces with scores

## Advanced Features

### Custom Evaluators

Create custom evaluators for specific metrics:

```python
# src/evaluation/custom_evaluators.py
from phoenix.evals import Evaluator

class VideoRelevanceEvaluator(Evaluator):
    def evaluate(self, query, results, expected):
        # Custom evaluation logic
        score = calculate_custom_metric(results, expected)
        return {
            "score": score,
            "explanation": "Custom relevance metric"
        }

# Register evaluator
from src.evaluation.phoenix.experiments import ExperimentOrchestrator
orchestrator = ExperimentOrchestrator()
orchestrator.register_evaluator("video_relevance", VideoRelevanceEvaluator())
```

### Continuous Evaluation

Set up continuous evaluation in CI/CD:

```yaml
# .github/workflows/evaluation.yml
name: Evaluation Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Start Phoenix
        run: |
          docker-compose -f docker/phoenix-compose.yml up -d
          sleep 10
      
      - name: Run Evaluation
        run: |
          python scripts/run_evaluation.py full \
            --name "ci_eval_${{ github.sha }}" \
            --profiles frame_based_colpali \
            --strategies binary_binary
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: outputs/reports/
```

### Regression Detection

Monitor for performance regression:

```python
# scripts/check_regression.py
import json
from pathlib import Path

def check_regression(current_eval, baseline_eval, threshold=0.05):
    """Check for metric regression"""
    current = json.load(open(current_eval))
    baseline = json.load(open(baseline_eval))
    
    regressions = []
    for metric in ["mrr", "ndcg", "precision_at_5"]:
        current_val = current["metrics"][metric]
        baseline_val = baseline["metrics"][metric]
        
        if current_val < baseline_val * (1 - threshold):
            regressions.append({
                "metric": metric,
                "current": current_val,
                "baseline": baseline_val,
                "drop": (baseline_val - current_val) / baseline_val
            })
    
    return regressions
```

### Experiment Comparison

Compare multiple experiments:

```python
from src.evaluation.phoenix.experiments import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator()

# Compare experiments
comparison_df = orchestrator.compare_experiments(
    experiment_ids=["exp_1", "exp_2", "exp_3"],
    metrics=["mrr", "latency_p95"]
)

# Save comparison
comparison_df.to_csv("experiment_comparison.csv")
```

## Troubleshooting

### Common Issues

#### Phoenix Connection Error
```
Error: Cannot connect to Phoenix at localhost:6006
```

**Solution**:
1. Check Phoenix is running: `ps aux | grep phoenix`
2. Check port availability: `lsof -i :6006`
3. Restart Phoenix: `python scripts/start_phoenix.py`

#### Out of Memory During Evaluation
```
Error: Process killed due to memory limit
```

**Solution**:
1. Reduce batch size in config: `batch_size: 5`
2. Limit parallel evaluations: `parallel_runs: false`
3. Increase system memory or use swap

#### Missing Traces in Phoenix
```
Warning: No traces found for evaluation
```

**Solution**:
1. Check instrumentation is enabled
2. Verify PHOENIX_WORKING_DIR is set
3. Check trace filters in Phoenix UI

#### Dataset Not Found
```
Error: Dataset 'eval_dataset_v1' not found in Phoenix
```

**Solution**:
1. List available datasets: `python scripts/list_phoenix_datasets.py`
2. Create dataset: `python scripts/create_phoenix_dataset.py`
3. Check Phoenix data directory permissions

### Debug Mode

Enable debug logging for detailed information:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run with debug output
python scripts/run_evaluation.py test --debug

# Check Phoenix logs
tail -f data/phoenix/phoenix.log
```

### Performance Optimization

For large-scale evaluations:

1. **Use batch evaluation**: Process multiple queries together
2. **Enable caching**: Set `PHOENIX_ENABLE_CACHE=true`
3. **Optimize trace storage**: Set `PHOENIX_MAX_TRACES` appropriately
4. **Use sampling**: Evaluate subset of queries for quick feedback

### Getting Help

1. **Check logs**: `outputs/logs/evaluation_*.log`
2. **Phoenix UI**: http://localhost:6006/help
3. **Inspect AI docs**: https://inspect.ai-safety-institute.org.uk/
4. **Phoenix docs**: https://docs.arize.com/phoenix/

## Best Practices

1. **Version datasets**: Always version evaluation datasets for reproducibility
2. **Monitor continuously**: Set up alerts for metric degradation
3. **Document changes**: Track configuration changes in git
4. **Regular baselines**: Update baseline metrics weekly/monthly
5. **Clean old data**: Periodically clean old traces to save space

## API Reference

### Evaluation Pipeline API

```python
from src.evaluation.pipeline.orchestrator import EvaluationPipeline

# Initialize pipeline
pipeline = EvaluationPipeline(config_path="configs/evaluation/eval_config.yaml")

# Run evaluation
results = await pipeline.run_comprehensive_evaluation(
    evaluation_name="api_eval",
    profiles=["frame_based_colpali"],
    strategies=["binary_binary"],
    tasks=["video_retrieval_accuracy"]
)

# Export report
report_path = pipeline.export_evaluation_report(format="html")
```

### Phoenix Dataset API

```python
from src.evaluation.phoenix.datasets import PhoenixDatasetManager

# Initialize manager
manager = PhoenixDatasetManager()

# Create dataset
dataset_id = manager.create_dataset_from_queries(
    name="custom_dataset",
    queries=[
        {
            "query": "person skiing",
            "expected_videos": ["video_1", "video_2"],
            "category": "action"
        }
    ]
)

# Run batch evaluation
results = manager.run_batch_evaluation(
    trace_ids=["trace_1", "trace_2"],
    dataset_name="custom_dataset"
)
```

### Monitoring API

```python
from src.evaluation.phoenix.monitoring import RetrievalMonitor

# Initialize monitor
monitor = RetrievalMonitor()
monitor.start()

# Log evaluation event
monitor.log_retrieval_event({
    "query": "test query",
    "profile": "frame_based_colpali",
    "strategy": "binary_binary",
    "latency_ms": 150,
    "mrr": 0.8
})

# Get metrics summary
summary = monitor.get_metrics_summary()
```

## Analytics and Visualization

The framework provides comprehensive analytics capabilities for analyzing collected traces:

### Command-Line Analytics

```bash
# Analyze traces from the last 24 hours
python scripts/analyze_traces.py analyze --hours 24

# Real-time monitoring dashboard
python scripts/analyze_traces.py monitor --refresh 30 --window 60

# Generate comprehensive report
python scripts/analyze_traces.py report --format all
```

### Available Analytics

- **Request Statistics**: Total requests, requests per time period
- **Response Time Analysis**: Mean, median, percentiles (P50, P75, P90, P95, P99)
- **Outlier Detection**: Identify and visualize response time outliers
- **Temporal Patterns**: Request distribution by hour, day
- **Profile Comparison**: Compare performance across different profiles
- **Error Analysis**: Error rates and types

### Visualization Types

1. **Time Series Plots**: Response times over time with percentile bands
2. **Distribution Plots**: Histograms, box plots, violin plots, ECDF
3. **Heatmaps**: Request patterns by hour and day
4. **Outlier Plots**: Highlight and analyze outliers
5. **Comparison Charts**: Compare metrics across profiles and strategies

### Programmatic Analytics

```python
from src.evaluation.phoenix.analytics import PhoenixAnalytics
from datetime import datetime, timedelta

# Initialize analytics
analytics = PhoenixAnalytics()

# Get traces from last hour
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)
traces = analytics.get_traces(start_time, end_time)

# Calculate statistics
stats = analytics.calculate_statistics(traces)
print(f"Mean response time: {stats['response_time']['mean']:.2f}ms")
print(f"P95 response time: {stats['response_time']['p95']:.2f}ms")
print(f"Outlier percentage: {stats['outliers']['percentage']:.2f}%")

# Create visualizations
time_series_fig = analytics.create_time_series_plot(traces)
outlier_fig = analytics.create_outlier_plot(traces)

# Generate comprehensive report
report = analytics.generate_report(start_time, end_time, "report.json")
```

### Real-Time Monitoring

Monitor traces in real-time with automatic refresh:

```bash
# Start real-time monitoring with 30-second refresh
python scripts/analyze_traces.py monitor --refresh 30 --window 60
```

This displays:
- Current request rate
- Response time statistics
- Recent traces
- Active alerts
- Auto-refreshes every 30 seconds

### Interactive Dashboard

For a comprehensive web-based dashboard with real-time visualizations:

```bash
# Start the Streamlit dashboard
streamlit run scripts/phoenix_dashboard.py

# Access at http://localhost:8501
```

The dashboard provides:
- **Real-time analytics** with auto-refresh
- **Interactive visualizations** (time series, distributions, heatmaps)
- **Root Cause Analysis** for automatic failure diagnosis
- **Export capabilities** (JSON, CSV, HTML reports)

See [Phoenix Dashboard Documentation](PHOENIX_DASHBOARD.md) for detailed information.

### Root Cause Analysis

The framework includes automated root cause analysis for failures:

```python
from src.evaluation.phoenix.root_cause_analysis import RootCauseAnalyzer

# Initialize analyzer
rca = RootCauseAnalyzer()

# Analyze failures
analysis = rca.analyze_failures(
    traces,
    include_performance=True,
    performance_threshold_percentile=95
)

# Get top root causes
for hypothesis in analysis['root_causes'][:3]:
    print(f"{hypothesis.hypothesis} (Confidence: {hypothesis.confidence:.0%})")
    print(f"Suggested Action: {hypothesis.suggested_action}")
```

RCA features:
- Automated error classification
- Pattern detection and correlation analysis
- Temporal pattern analysis (burst detection)
- Performance degradation analysis
- Prioritized recommendations