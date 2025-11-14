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
├── analysis/                # Evaluation analysis tools
├── core/                    # Core evaluation primitives
├── evaluators/              # Built-in evaluators
│   ├── base_evaluator.py
│   ├── golden_dataset.py    # Golden dataset evaluation
│   ├── llm_judge.py         # LLM-as-judge evaluation
│   ├── visual_judge.py      # Multi-modal visual evaluation
│   ├── qwen_visual_judge.py # Qwen-based visual judge
│   ├── reference_free.py    # Reference-free evaluation
│   └── routing_evaluator.py # Routing agent evaluation
├── inspect_tasks/           # Inspect AI task definitions
├── metrics/                 # Evaluation metrics
│   ├── custom.py            # Custom metric definitions
│   └── reference_free.py    # Reference-free metrics
├── plugins/                 # Plugin system for providers
└── providers/               # Evaluation provider implementations
```

## Key Modules

### Evaluators (`cogniverse_evaluation.evaluators`)

Built-in evaluators for different evaluation scenarios:

**Base Evaluators:**
- `BaseEvaluator`: Abstract base class for all evaluators
- `BaseEvaluatorNoTrace`: Evaluator without trace dependencies

**Dataset Evaluators:**
- `GoldenDatasetEvaluator`: Evaluate against golden datasets
- `SyncGoldenDatasetEvaluator`: Synchronous golden dataset evaluation
- `RoutingEvaluator`: Specialized routing agent evaluation

**LLM-Based Evaluators:**
- `LLMJudge`: Use LLMs as judges for quality assessment
- `VisualJudge`: Multi-modal evaluation with vision-language models
- `QwenVisualJudge`: Qwen2-VL based visual evaluation
- `ConfigurableVisualJudge`: Configurable visual assessment

**Reference-Free Evaluators:**
- `ReferenceFreeEvaluator`: Evaluate without ground truth
- `SyncReferenceFreeEvaluator`: Synchronous reference-free evaluation

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

### Inspect AI Integration (`cogniverse_evaluation.inspect_tasks`)

Integration with Inspect AI framework:
- Task definitions for LLM evaluation
- Automated test suite generation
- Benchmark integration
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
from cogniverse_evaluation.evaluators import GoldenDatasetEvaluator
from cogniverse_agents.routing import RoutingAgent

# Initialize evaluator with golden dataset
evaluator = GoldenDatasetEvaluator(
    dataset_path="golden_dataset_v1.csv",
    metrics=["accuracy", "precision", "recall", "f1"]
)

# Evaluate routing agent
routing_agent = RoutingAgent(config=config)
results = await evaluator.evaluate(
    agent=routing_agent,
    num_samples=100
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
```

### LLM-as-Judge Evaluation

```python
from cogniverse_evaluation.evaluators import LLMJudge

# Initialize LLM judge
judge = LLMJudge(
    model="claude-sonnet-4.5",
    criteria=[
        "relevance",
        "accuracy",
        "completeness",
        "coherence"
    ]
)

# Evaluate search results
judgment = await judge.evaluate(
    query="machine learning tutorials",
    response=search_results,
    context={"expected_modality": "video"}
)

print(f"Relevance: {judgment['relevance']}/10")
print(f"Overall Score: {judgment['overall']:.2f}")
print(f"Reasoning: {judgment['reasoning']}")
```

### Multi-Modal Visual Evaluation

```python
from cogniverse_evaluation.evaluators import QwenVisualJudge
from PIL import Image

# Initialize visual judge
visual_judge = QwenVisualJudge(
    model="qwen2-vl-7b",
    criteria=["visual_quality", "relevance", "clarity"]
)

# Evaluate image search results
images = [Image.open(f"result_{i}.jpg") for i in range(5)]
judgment = await visual_judge.evaluate(
    query="red sports car",
    images=images,
    query_type="image_search"
)

print(f"Visual Quality: {judgment['visual_quality']:.2f}")
print(f"Relevance: {judgment['relevance']:.2f}")
```

### Reference-Free Evaluation

```python
from cogniverse_evaluation.evaluators import ReferenceFreeEvaluator

# Initialize reference-free evaluator
evaluator = ReferenceFreeEvaluator(
    metrics=["coherence", "fluency", "consistency"]
)

# Evaluate generated summaries without ground truth
results = await evaluator.evaluate(
    generated_texts=summaries,
    context={"source_documents": docs}
)

print(f"Coherence: {results['coherence']:.3f}")
print(f"Fluency: {results['fluency']:.3f}")
```

### Custom Metrics

```python
from cogniverse_evaluation.metrics import CustomMetric
import numpy as np

class SemanticSimilarityMetric(CustomMetric):
    """Custom semantic similarity metric."""

    def __init__(self, model_name: str):
        super().__init__(name="semantic_similarity")
        self.model = load_embedding_model(model_name)

    def compute(self, predictions, references):
        pred_embeds = self.model.encode(predictions)
        ref_embeds = self.model.encode(references)

        # Cosine similarity
        similarities = np.sum(pred_embeds * ref_embeds, axis=1) / (
            np.linalg.norm(pred_embeds, axis=1) *
            np.linalg.norm(ref_embeds, axis=1)
        )

        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "scores": similarities.tolist()
        }

# Use custom metric
metric = SemanticSimilarityMetric(model_name="all-MiniLM-L6-v2")
results = metric.compute(predicted_summaries, reference_summaries)
```

### Routing Agent Evaluation

```python
from cogniverse_evaluation.evaluators import RoutingEvaluator

# Initialize routing evaluator
evaluator = RoutingEvaluator(
    golden_dataset="routing_golden_v2.csv",
    metrics=["routing_accuracy", "confidence", "latency"]
)

# Evaluate routing decisions
results = await evaluator.evaluate(
    routing_agent=routing_agent,
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
pytest tests/evaluation/test_golden_dataset.py
pytest tests/evaluation/test_visual_judge.py
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
