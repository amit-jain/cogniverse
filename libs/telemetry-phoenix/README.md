# Cogniverse Telemetry Phoenix

**Package**: `cogniverse-telemetry-phoenix`
**Layer**: Core Layer - Plugin (Pink)
**Version**: 0.1.0

Phoenix telemetry provider plugin for Cogniverse, providing Phoenix-specific implementations of telemetry interfaces for querying spans, managing annotations, datasets, and experiments.

---

## Purpose

The `cogniverse-telemetry-phoenix` package provides:
- **Phoenix Provider**: Plugin implementation for Phoenix telemetry backend
- **Span Querying**: Retrieve and filter OpenTelemetry spans from Phoenix
- **Annotations**: Add human feedback and labels to spans
- **Dataset Management**: Create and manage evaluation datasets
- **Experiment Tracking**: Track DSPy optimization experiments
- **Entry Point Discovery**: Auto-discovered via Python entry points

---

## Architecture

### Position in 10-Package Structure

```
Foundation Layer (Blue)
├── cogniverse-sdk
└── cogniverse-foundation ← cogniverse-telemetry-phoenix depends on this

Core Layer (Pink)
├── cogniverse-core
├── cogniverse-evaluation ← cogniverse-telemetry-phoenix depends on this
└── cogniverse-telemetry-phoenix ← YOU ARE HERE (Plugin)

Implementation Layer (Yellow/Green)
├── cogniverse-agents
├── cogniverse-vespa
└── cogniverse-synthetic

Application Layer (Light Blue/Purple)
├── cogniverse-runtime
└── cogniverse-dashboard ← Uses telemetry-phoenix
```

### Plugin Architecture

This package implements the **Provider Plugin Pattern**:

```python
# Auto-discovered via entry points
[project.entry-points."cogniverse.telemetry.providers"]
phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"
```

The foundation layer automatically discovers and loads the Phoenix provider without explicit imports.

### Dependencies

**Workspace Dependencies:**
- `cogniverse-foundation` (required) - Base telemetry interfaces
- `cogniverse-evaluation` (required) - Experiment tracking interfaces

**External Dependencies:**
- `arize-phoenix-otel>=1.0.0` - Phoenix OpenTelemetry SDK
- `httpx>=0.25.0` - Async HTTP client for Phoenix API
- `pandas>=2.1.0` - DataFrame operations for spans
- `polars>=0.19.0` - Fast dataframe processing

---

## Key Features

### 1. Auto-Discovery via Entry Points

The provider is automatically discovered and loaded:

```python
from cogniverse_foundation.telemetry import TelemetryManager

# Phoenix provider automatically loaded
telemetry = TelemetryManager()
provider = telemetry.get_provider(tenant_id="acme_corp")

# Provider is PhoenixProvider instance
assert provider.name == "phoenix"
```

### 2. Span Querying

Query spans from Phoenix with filtering:

```python
from cogniverse_foundation.telemetry import TelemetryManager

telemetry = TelemetryManager()
provider = telemetry.get_provider(tenant_id="acme_corp")

# Query spans
spans_df = await provider.traces.get_spans(
    project="cogniverse-acme_corp-search",
    limit=1000,
    start_time="2025-11-01T00:00:00Z",
    end_time="2025-11-13T23:59:59Z",
    filter_condition="span.status_code = 'OK'"
)

print(f"Retrieved {len(spans_df)} spans")
```

### 3. Annotations

Add human feedback to spans:

```python
# Add annotation
await provider.annotations.add_annotation(
    span_id="abc123",
    name="human_approval",
    label="approved",
    score=1.0,
    metadata={"reviewer": "alice", "comments": "Looks good"},
    project="cogniverse-acme_corp-search"
)

# Get annotations for span
annotations = await provider.annotations.get_annotations(
    span_id="abc123",
    project="cogniverse-acme_corp-search"
)
```

### 4. Dataset Management

Create and manage evaluation datasets:

```python
# Create dataset
dataset_id = await provider.datasets.create_dataset(
    name="video_search_queries",
    description="Evaluation queries for video search",
    project="cogniverse-acme_corp-search",
    data=[
        {"query": "machine learning tutorial", "expected_modality": "video"},
        {"query": "python programming", "expected_modality": "video"}
    ]
)

# Get dataset
dataset = await provider.datasets.get_dataset(
    dataset_id=dataset_id,
    project="cogniverse-acme_corp-search"
)

# List datasets
datasets = await provider.datasets.list_datasets(
    project="cogniverse-acme_corp-search"
)
```

### 5. Experiment Tracking

Track DSPy optimization experiments:

```python
# Create experiment
experiment_id = await provider.experiments.create_experiment(
    name="modality_routing_v1",
    description="GLiNER-based modality routing",
    project="cogniverse-acme_corp-search",
    metadata={
        "optimizer": "BootstrapFewShotWithRandomSearch",
        "dataset": "video_search_queries",
        "model": "gpt-4"
    }
)

# Log experiment run
await provider.experiments.log_run(
    experiment_id=experiment_id,
    metrics={
        "accuracy": 0.85,
        "f1_score": 0.82,
        "latency_ms": 150
    },
    parameters={
        "temperature": 0.7,
        "num_examples": 5
    }
)

# Get experiment results
results = await provider.experiments.get_experiment(
    experiment_id=experiment_id,
    project="cogniverse-acme_corp-search"
)
```

---

## Installation

### Development (Editable Mode)

```bash
# From workspace root
uv sync

# Or install individually
uv pip install -e libs/telemetry-phoenix
```

### Production

```bash
pip install cogniverse-telemetry-phoenix

# Automatically installs:
# - cogniverse-foundation
# - cogniverse-evaluation
# - arize-phoenix-otel
# - httpx, pandas, polars
```

---

## Configuration

Configuration via `TelemetryConfig` from `cogniverse-foundation`:

```python
from cogniverse_foundation.telemetry import TelemetryManager, TelemetryConfig

config = TelemetryConfig(
    provider="phoenix",  # Optional - auto-detects if omitted
    provider_config={
        "http_endpoint": "http://localhost:6006",  # Phoenix HTTP API
        "grpc_endpoint": "http://localhost:4317",  # Phoenix gRPC OTLP (optional)
    }
)

telemetry = TelemetryManager(config=config)
provider = telemetry.get_provider(tenant_id="acme_corp")
```

### Environment Variables

```bash
# Required
export PHOENIX_HTTP_ENDPOINT="http://localhost:6006"

# Optional
export PHOENIX_GRPC_ENDPOINT="http://localhost:4317"
export PHOENIX_API_KEY="your-api-key"  # For Phoenix Cloud
```

---

## Usage

### Basic Setup

```python
from cogniverse_foundation.telemetry import TelemetryManager

# Initialize telemetry manager
telemetry = TelemetryManager()

# Get Phoenix provider (auto-discovered)
provider = telemetry.get_provider(tenant_id="acme_corp")

# Provider is automatically configured for tenant
assert provider.project == "cogniverse-acme_corp-project"
```

### Query Spans

```python
# Get recent spans
spans_df = await provider.traces.get_spans(
    project="cogniverse-acme_corp-search",
    limit=100
)

# Filter spans
video_search_spans = spans_df[
    spans_df["name"] == "video_search"
]

# Get span details
span = await provider.traces.get_span(
    span_id="abc123",
    project="cogniverse-acme_corp-search"
)
```

### Add Annotations

```python
# Add thumbs up annotation
await provider.annotations.add_annotation(
    span_id="abc123",
    name="thumbs_up",
    label="positive",
    score=1.0,
    project="cogniverse-acme_corp-search"
)

# Add detailed feedback
await provider.annotations.add_annotation(
    span_id="abc123",
    name="detailed_feedback",
    label="needs_improvement",
    score=0.5,
    metadata={
        "issue": "results_not_relevant",
        "suggestion": "improve_query_understanding"
    },
    project="cogniverse-acme_corp-search"
)
```

### Create Dataset

```python
# Create dataset from spans
dataset_id = await provider.datasets.create_from_spans(
    name="search_queries_nov_2025",
    span_ids=["span1", "span2", "span3"],
    project="cogniverse-acme_corp-search"
)

# Create dataset from file
dataset_id = await provider.datasets.create_from_file(
    name="evaluation_queries",
    file_path="/data/queries.csv",
    project="cogniverse-acme_corp-search"
)
```

### Track Experiment

```python
# Start experiment
experiment = await provider.experiments.start_experiment(
    name="routing_optimization",
    dataset_id=dataset_id,
    project="cogniverse-acme_corp-search"
)

# Log metrics during training
for epoch in range(10):
    metrics = train_epoch()
    await provider.experiments.log_metrics(
        experiment_id=experiment.id,
        metrics=metrics,
        step=epoch
    )

# Finish experiment
await provider.experiments.finish_experiment(
    experiment_id=experiment.id,
    status="completed"
)
```

---

## Multi-Tenant Project Mapping

The provider automatically maps tenants to Phoenix projects:

| Tenant ID | Phoenix Project | Purpose |
|-----------|----------------|---------|
| `acme_corp` | `cogniverse-acme_corp-project` | Default project |
| `acme_corp` | `cogniverse-acme_corp-search` | Search traces |
| `acme_corp` | `cogniverse-acme_corp-ingestion` | Ingestion traces |
| `acme_corp` | `cogniverse-acme_corp-synthetic_data` | Synthetic data gen |
| `globex_inc` | `cogniverse-globex_inc-project` | Default project |
| `default` | `cogniverse-default-project` | Default project |

**Project Naming Convention:**
```
cogniverse-{tenant_id}-{service}
```

---

## Development

### Running Tests

```bash
# Run all telemetry-phoenix tests
uv run pytest tests/telemetry-phoenix/ -v

# Run integration tests (requires Phoenix)
uv run pytest tests/telemetry-phoenix/integration/ -v

# Run specific provider tests
uv run pytest tests/telemetry-phoenix/unit/test_phoenix_provider.py -v
```

### Local Phoenix Instance

```bash
# Start Phoenix using Docker
docker run --detach --name phoenix \
  --publish 6006:6006 --publish 4317:4317 \
  arizephoenix/phoenix:latest

# Wait for Phoenix to be ready
curl -s http://localhost:6006/health

# Access Phoenix UI
open http://localhost:6006
```

### Code Style

```bash
# Format code
uv run ruff format libs/telemetry-phoenix

# Lint code
uv run ruff check libs/telemetry-phoenix

# Type check
uv run mypy libs/telemetry-phoenix
```

---

## Plugin Implementation

### Provider Interface

The package implements the `TelemetryProvider` interface:

```python
from cogniverse_foundation.telemetry.interfaces import TelemetryProvider

class PhoenixProvider(TelemetryProvider):
    """Phoenix telemetry provider implementation."""

    @property
    def name(self) -> str:
        return "phoenix"

    @property
    def traces(self) -> TraceInterface:
        return self._traces

    @property
    def annotations(self) -> AnnotationInterface:
        return self._annotations

    @property
    def datasets(self) -> DatasetInterface:
        return self._datasets

    @property
    def experiments(self) -> ExperimentInterface:
        return self._experiments
```

### Entry Point Registration

In `pyproject.toml`:

```toml
[project.entry-points."cogniverse.telemetry.providers"]
phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"
```

This enables automatic discovery by the foundation layer.

---

## API Reference

### TraceInterface

```python
# Get spans
spans_df = await provider.traces.get_spans(
    project: str,
    limit: int = 1000,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filter_condition: Optional[str] = None
) -> pd.DataFrame

# Get single span
span = await provider.traces.get_span(
    span_id: str,
    project: str
) -> Span

# Export spans
await provider.traces.export_spans(
    project: str,
    output_path: str,
    format: str = "parquet"
)
```

### AnnotationInterface

```python
# Add annotation
await provider.annotations.add_annotation(
    span_id: str,
    name: str,
    label: str,
    score: float,
    metadata: Optional[Dict] = None,
    project: str
)

# Get annotations
annotations = await provider.annotations.get_annotations(
    span_id: str,
    project: str
) -> List[Annotation]

# Delete annotation
await provider.annotations.delete_annotation(
    annotation_id: str,
    project: str
)
```

### DatasetInterface

```python
# Create dataset
dataset_id = await provider.datasets.create_dataset(
    name: str,
    description: str,
    data: List[Dict],
    project: str
) -> str

# Get dataset
dataset = await provider.datasets.get_dataset(
    dataset_id: str,
    project: str
) -> Dataset

# List datasets
datasets = await provider.datasets.list_datasets(
    project: str
) -> List[Dataset]

# Delete dataset
await provider.datasets.delete_dataset(
    dataset_id: str,
    project: str
)
```

### ExperimentInterface

```python
# Create experiment
experiment_id = await provider.experiments.create_experiment(
    name: str,
    description: str,
    metadata: Dict,
    project: str
) -> str

# Log run
await provider.experiments.log_run(
    experiment_id: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any]
)

# Get experiment
experiment = await provider.experiments.get_experiment(
    experiment_id: str,
    project: str
) -> Experiment

# List experiments
experiments = await provider.experiments.list_experiments(
    project: str
) -> List[Experiment]
```

---

## Documentation

- **Architecture**: [10-Package Architecture](../../docs/architecture/10-package-architecture.md)
- **Telemetry Guide**: [Telemetry Architecture](../../docs/architecture/telemetry.md)
- **Phoenix Docs**: [Arize Phoenix Documentation](https://docs.arize.com/phoenix/)
- **Diagrams**: [SDK Architecture Diagrams](../../docs/diagrams/sdk-architecture-diagrams.md)

---

## Troubleshooting

### Common Issues

**1. Provider Not Found**
```python
# Ensure package is installed
pip list | grep cogniverse-telemetry-phoenix

# Check entry points
python -c "from importlib.metadata import entry_points; print([ep for ep in entry_points()['cogniverse.telemetry.providers']])"
```

**2. Phoenix Connection Failed**
```bash
# Test Phoenix connectivity
curl http://localhost:6006/health

# Check Phoenix logs
docker logs phoenix
```

**3. Project Not Found**
```python
# List available projects
import httpx
response = httpx.get("http://localhost:6006/v1/projects")
print(response.json())
```

**4. Spans Not Appearing**
- Verify spans are being sent: Check OTLP endpoint (4317)
- Verify project name matches convention
- Check Phoenix UI for spans

---

## Performance

### Batch Operations

```python
# Batch span queries
spans = await provider.traces.get_spans(
    project=project,
    limit=10000  # Fetch large batches
)

# Batch annotations
annotations = [
    {"span_id": span_id, "label": "approved"}
    for span_id in span_ids
]
await provider.annotations.add_annotations_batch(
    annotations=annotations,
    project=project
)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_spans(project: str, limit: int) -> pd.DataFrame:
    return await provider.traces.get_spans(
        project=project,
        limit=limit
    )
```

---

## Contributing

```bash
# Create feature branch
git checkout -b feature/phoenix-improvement

# Make changes
# ...

# Run tests
uv run pytest tests/telemetry-phoenix/ -v

# Submit PR
```

---

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Related Packages

- **cogniverse-foundation**: Base telemetry interfaces (depends on this)
- **cogniverse-evaluation**: Experiment tracking (depends on this)
- **cogniverse-dashboard**: UI for Phoenix data (depends on this)
- **cogniverse-core**: Multi-agent orchestration (uses this)
