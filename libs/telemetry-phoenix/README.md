# cogniverse-telemetry-phoenix

Phoenix telemetry provider for Cogniverse.

Provides Phoenix-specific implementations of telemetry interfaces for querying spans, managing annotations, datasets, and experiments.

## Installation

```bash
pip install cogniverse-telemetry-phoenix
```

## Configuration

The provider is auto-discovered via entry points. Configure via environment variables or TelemetryConfig:

```python
from cogniverse_core.telemetry.config import TelemetryConfig

config = TelemetryConfig(
    provider="phoenix",  # Optional - auto-detects if omitted
    provider_config={
        "http_endpoint": "http://localhost:6006",  # Phoenix HTTP API
        "grpc_endpoint": "http://localhost:4317",  # Phoenix gRPC OTLP (optional)
    }
)
```

## Usage

```python
from cogniverse_core.telemetry.manager import TelemetryManager

telemetry = TelemetryManager()
provider = telemetry.get_provider(tenant_id="customer-123")

# Query spans
spans_df = await provider.traces.get_spans(
    project="cogniverse-customer-123-search",
    limit=1000
)

# Add annotations
await provider.annotations.add_annotation(
    span_id="abc123",
    name="human_approval",
    label="approved",
    score=1.0,
    metadata={"reviewer": "alice"},
    project="cogniverse-customer-123-synthetic_data"
)
```
