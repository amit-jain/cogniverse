# Phoenix Integration & Telemetry

## Overview

Cogniverse uses **Phoenix (Arize AI)** for comprehensive observability with multi-tenant project isolation. Phoenix provides distributed tracing, experiment tracking, root cause analysis, and performance monitoring through OpenTelemetry instrumentation.

## Phoenix Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌────────────────────────────────────────────────┐         │
│  │ Multi-Tenant Telemetry Manager (Singleton)     │         │
│  │ - Lazy tracer provider initialization          │         │
│  │ - LRU caching (max 100 tenants)                │         │
│  │ - Thread-safe provider/tracer maps             │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 Tenant-Specific Tracers                      │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │ Tenant: acme       │  │ Tenant: globex     │            │
│  │ Service: routing   │  │ Service: routing   │            │
│  │ Project:           │  │ Project:           │            │
│  │  cogniverse-acme-  │  │  cogniverse-globex-│            │
│  │  routing           │  │  routing           │            │
│  └────────────────────┘  └────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              OpenTelemetry Exporters                         │
│  ┌────────────────────────────────────────────────┐         │
│  │ Production: BatchSpanProcessor (async)         │         │
│  │ - Queue size: 2048                             │         │
│  │ - Batch size: 512                              │         │
│  │ - Schedule delay: 500ms                        │         │
│  │ - Drop on full: true                           │         │
│  └────────────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────┐         │
│  │ Test Mode: SimpleSpanProcessor (sync)          │         │
│  │ - Immediate export on span end                 │         │
│  │ - 5 second timeout                             │         │
│  │ - TELEMETRY_SYNC_EXPORT=true                   │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                Phoenix Server (OTLP gRPC)                    │
│  ┌────────────────────────────────────────────────┐         │
│  │ Endpoint: localhost:4317                        │         │
│  │ Protocol: OTLP/gRPC                            │         │
│  │ Projects: Multi-tenant isolation               │         │
│  │ Storage: Spans, experiments, datasets          │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Tenant Telemetry

### Tenant-Specific Tracer Providers

Each tenant gets isolated tracer provider with dedicated Phoenix project:

```python
from src.app.telemetry.manager import TelemetryManager

telemetry = TelemetryManager()

# Tenant "acme" gets project: cogniverse-acme-routing
with telemetry.span("search", tenant_id="acme") as span:
    span.set_attribute("query", "machine learning")
    # Span exported to Phoenix project "cogniverse-acme-routing"

# Tenant "globex" gets project: cogniverse-globex-routing
with telemetry.span("search", tenant_id="globex") as span:
    span.set_attribute("query", "deep learning")
    # Span exported to Phoenix project "cogniverse-globex-routing"
```

### Project Naming Convention

```
Pattern: cogniverse-{tenant_id}-{service}

Examples:
- cogniverse-acme-video-search
- cogniverse-globex-routing
- cogniverse-default-orchestration
```

Configured via `TelemetryConfig.tenant_project_template`

### LRU Caching

```python
# TelemetryManager configuration
config = TelemetryConfig(
    max_cached_tenants=100,  # LRU cache size
    tenant_cache_ttl_seconds=3600  # 1 hour TTL
)

# Cache metrics
stats = telemetry.get_stats()
# {
#   "cache_hits": 1234,
#   "cache_misses": 56,
#   "cached_tenants": 45,
#   "cached_tracers": 90
# }
```

Eviction policy: Remove oldest tenant providers when cache exceeds 100 entries

## Span Hierarchy

### Standard Span Structure

```
cogniverse.request (root)
├── cogniverse.routing
│   ├── routing.query_analysis
│   └── routing.decision
├── cogniverse.orchestration
│   ├── orchestration.routing_agent
│   ├── orchestration.video_search_agent
│   │   ├── video_search.embedding
│   │   ├── video_search.vespa_query
│   │   └── video_search.result_formatting
│   └── orchestration.summarizer_agent
└── cogniverse.response
```

### Span Attributes

All spans include:
- `tenant.id`: Tenant identifier
- `service.name`: Service name (e.g., "video-search", "routing")
- `environment`: Environment (development/staging/production)

**Routing Spans:**
```python
with telemetry.span("cogniverse.routing", tenant_id=tenant_id) as span:
    span.set_attribute("query.text", query)
    span.set_attribute("routing.chosen_agent", "video_search_agent")
    span.set_attribute("routing.confidence", 0.92)
    span.set_attribute("routing.reasoning", "Video search needed")
```

**Video Search Spans:**
```python
with telemetry.span("video_search.query", tenant_id=tenant_id) as span:
    span.set_attribute("query.text", query)
    span.set_attribute("search.profile", "video_colpali_smol500_mv_frame")
    span.set_attribute("search.ranking_strategy", "hybrid_float_bm25")
    span.set_attribute("search.top_k", 10)
    span.set_attribute("search.results_count", 8)
    span.set_attribute("search.latency_ms", 120)
```

**Orchestration Spans:**
```python
with telemetry.span("cogniverse.orchestration", tenant_id=tenant_id,
                     service_name="cogniverse.orchestration") as span:
    span.set_attribute("orchestration.workflow_id", workflow_id)
    span.set_attribute("orchestration.agent_sequence", ["routing", "video_search", "summarizer"])
    span.set_attribute("orchestration.total_agents", 3)
```

## Configuration

### Environment Variables

```bash
# Core telemetry settings
TELEMETRY_ENABLED=true
TELEMETRY_LEVEL=detailed  # disabled, basic, detailed, verbose
ENVIRONMENT=production

# Phoenix settings
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=localhost:4317
PHOENIX_USE_TLS=false

# Test mode (synchronous export)
TELEMETRY_SYNC_EXPORT=false  # true for tests

# Service identification
SERVICE_VERSION=1.0.0
```

### Telemetry Levels

| Level | Components Instrumented |
|-------|------------------------|
| `disabled` | None |
| `basic` | Search service only |
| `detailed` | Search + backend + encoder |
| `verbose` | Everything (search + backend + encoder + pipeline + agents) |

```python
from src.app.telemetry.config import TelemetryConfig, TelemetryLevel

config = TelemetryConfig(
    enabled=True,
    level=TelemetryLevel.DETAILED,
    environment="production"
)

# Check if component should be instrumented
if config.should_instrument_level("search_service"):
    # Add search tracing
    pass
```

### Batch Export Configuration

```python
from src.app.telemetry.config import BatchExportConfig

batch_config = BatchExportConfig(
    max_queue_size=2048,          # Span queue capacity
    max_export_batch_size=512,    # Spans per batch
    export_timeout_millis=30_000, # 30 second export timeout
    schedule_delay_millis=500,    # Batch every 500ms
    drop_on_queue_full=True,      # Drop spans when queue full
    log_dropped_spans=True,       # Log dropped span warnings
    max_drop_log_rate_per_minute=10  # Rate limit drop logs
)

config = TelemetryConfig(batch_config=batch_config)
```

## Usage Examples

### Basic Span Creation

```python
from src.app.telemetry.manager import get_telemetry_manager

telemetry = get_telemetry_manager()

# Simple span
with telemetry.span("search", tenant_id="acme") as span:
    span.set_attribute("query", "machine learning")
    results = search_function(query)
    span.set_attribute("result_count", len(results))
```

### Nested Spans

```python
# Root span
with telemetry.span("cogniverse.request", tenant_id="acme") as request_span:
    request_span.set_attribute("request.id", request_id)

    # Child span: routing
    with telemetry.span("cogniverse.routing", tenant_id="acme") as routing_span:
        routing_span.set_attribute("routing.chosen_agent", "video_search")

    # Child span: search
    with telemetry.span("video_search.query", tenant_id="acme") as search_span:
        search_span.set_attribute("search.strategy", "hybrid_float_bm25")
        results = execute_search()
```

### Error Handling

```python
with telemetry.span("risky_operation", tenant_id="acme") as span:
    try:
        result = risky_function()
    except Exception as e:
        # Exception automatically recorded in span
        # Span status set to ERROR
        # Exception re-raised
        raise
```

Auto-instrumentation includes:
- Exception recording via `span.record_exception(e)`
- Status set to `StatusCode.ERROR`
- Error message in span status

### Manual Error Recording

```python
from opentelemetry.trace import Status, StatusCode

with telemetry.span("validation", tenant_id="acme") as span:
    if not is_valid(data):
        span.set_status(Status(StatusCode.ERROR, "Validation failed"))
        span.set_attribute("error.type", "ValidationError")
        span.set_attribute("error.message", "Invalid query format")
```

### Multi-Service Tracing

```python
# Routing service
with telemetry.span("routing", tenant_id="acme",
                     service_name="routing") as span:
    decision = make_routing_decision()

# Orchestration service
with telemetry.span("orchestration", tenant_id="acme",
                     service_name="cogniverse.orchestration") as span:
    result = execute_workflow()
```

Phoenix projects:
- `cogniverse-acme-routing`
- `cogniverse-acme-cogniverse.orchestration`

## Experiment Tracking

### Phoenix Experiments Integration

Phoenix experiments track evaluation runs with complete reproducibility:

```python
from src.evaluation.plugins.phoenix_experiment import PhoenixExperimentPlugin

# Run Inspect AI evaluation tracked by Phoenix
result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
    dataset_name="golden_eval_v1",
    profiles=["video_colpali_smol500_mv_frame"],
    strategies=["hybrid_float_bm25", "float_float"],
    evaluators=[visual_judge, quality_scorer],
    config={
        "top_k": 10,
        "enable_llm_evaluators": True
    }
)

# Result stored in Phoenix with:
# - Experiment name: inspect_eval_golden_eval_v1_20251004_120000
# - Metadata: profiles, strategies, config
# - All evaluation results and metrics
```

### Creating Datasets

```python
import phoenix as px

client = px.Client()

# Create dataset from golden queries
examples = [
    {"query": "machine learning tutorial", "expected_video": "ml_basics.mp4"},
    {"query": "deep learning intro", "expected_video": "dl_intro.mp4"}
]

dataset = client.upload_dataset(
    dataset_name="golden_eval_v1",
    inputs=[{"query": ex["query"]} for ex in examples],
    outputs=[{"expected_video": ex["expected_video"]} for ex in examples]
)

print(f"Created dataset: {dataset.name} with {len(examples)} examples")
```

### Running Experiments

```python
from phoenix.experiments import run_experiment

def search_task(example):
    """Task function for Phoenix experiment."""
    query = example.input["query"]

    # Execute search
    from src.app.search.service import SearchService
    service = SearchService(config, profile="video_colpali_smol500_mv_frame")
    results = service.search(query, ranking_strategy="hybrid_float_bm25")

    return {
        "query": query,
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }

# Run experiment
experiment_result = run_experiment(
    dataset=dataset,
    task=search_task,
    evaluators=[quality_evaluator],
    experiment_name="search_eval_v1"
)

# Access results
print(f"Experiment ID: {experiment_result.experiment_id}")
print(f"Mean quality score: {experiment_result.metrics['quality'].mean()}")
```

### Experiment Comparison

```python
# Compare two experiments
client = px.Client()

experiment_1 = client.get_experiment(name="search_eval_baseline")
experiment_2 = client.get_experiment(name="search_eval_optimized")

# Compare metrics
baseline_accuracy = experiment_1.metrics["accuracy"].mean()
optimized_accuracy = experiment_2.metrics["accuracy"].mean()

improvement = (optimized_accuracy - baseline_accuracy) / baseline_accuracy * 100
print(f"Accuracy improvement: +{improvement:.2f}%")
```

## Phoenix Dashboard

### Accessing the Dashboard

```bash
# Start standalone Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard_standalone.py --server.port 8501

# Navigate to: http://localhost:8501
```

### Dashboard Features

**1. Traces View:**
- Real-time span visualization
- Tenant filtering
- Service filtering
- Latency analysis
- Error tracking

**2. Experiments Tab:**
- Experiment history
- Metric comparison
- Dataset versioning
- Evaluator results

**3. Projects Tab:**
- Multi-tenant project list
- Per-project span counts
- Resource usage

**4. Analytics:**
- Query performance trends
- Agent selection distribution
- Routing accuracy over time
- Search latency percentiles

### Dashboard Configuration

```python
# In scripts/phoenix_dashboard_standalone.py
import streamlit as st
import phoenix as px

# Connect to Phoenix
client = px.Client(endpoint="http://localhost:6006")

# Get all projects (multi-tenant)
projects = client.get_projects()

# Filter by tenant
selected_tenant = st.selectbox("Tenant", ["acme", "globex", "default"])
project_name = f"cogniverse-{selected_tenant}-routing"

# Load spans
spans = client.get_spans(project_name=project_name)
st.dataframe(spans)
```

## Root Cause Analysis

### Analyzing Failed Requests

```python
import phoenix as px

client = px.Client()

# Get spans for failed requests
spans = client.query_spans(
    project_name="cogniverse-acme-routing",
    filter_condition="status.code = 'ERROR'",
    start_time="2025-10-04T00:00:00Z",
    end_time="2025-10-04T23:59:59Z"
)

# Analyze error patterns
for span in spans:
    print(f"Error: {span.attributes.get('error.message')}")
    print(f"Trace ID: {span.trace_id}")
    print(f"Duration: {span.duration_ms}ms")

    # Get full trace
    trace = client.get_trace(span.trace_id)
    # Analyze trace to find root cause
```

### Performance Debugging

```python
# Find slow requests
slow_spans = client.query_spans(
    project_name="cogniverse-acme-routing",
    filter_condition="duration > 1000",  # > 1 second
    order_by="duration DESC",
    limit=10
)

# Analyze slowest operation in each trace
for span in slow_spans:
    trace = client.get_trace(span.trace_id)

    # Find slowest child span
    slowest = max(trace.spans, key=lambda s: s.duration_ms)
    print(f"Bottleneck: {slowest.name} ({slowest.duration_ms}ms)")
    print(f"Attributes: {slowest.attributes}")
```

## Monitoring & Alerts

### Telemetry Manager Metrics

```python
from src.app.telemetry.manager import get_telemetry_manager

telemetry = get_telemetry_manager()

stats = telemetry.get_stats()
# {
#   "cache_hits": 1234,
#   "cache_misses": 56,
#   "failed_initializations": 2,
#   "cached_tenants": 45,
#   "cached_tracers": 90,
#   "config": {
#     "enabled": true,
#     "level": "detailed",
#     "environment": "production"
#   }
# }

# Monitor cache hit rate
hit_rate = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
print(f"Cache hit rate: {hit_rate:.2%}")

# Monitor failures
if stats["failed_initializations"] > 0:
    print(f"Warning: {stats['failed_initializations']} failed tenant initializations")
```

### Graceful Degradation

When Phoenix unavailable, telemetry automatically degrades:

```python
# TelemetryManager handles failures gracefully
with telemetry.span("search", tenant_id="acme") as span:
    # If Phoenix down, span is NoOpSpan (no-op operations)
    span.set_attribute("query", "test")  # No error, silently ignored
    results = search()  # Search continues normally
```

NoOpSpan methods:
- `set_attribute()` - no-op
- `add_event()` - no-op
- `set_status()` - no-op
- `record_exception()` - no-op

### Force Flush

```python
# Flush all pending spans before shutdown
success = telemetry.force_flush(timeout_millis=10000)

if not success:
    print("Warning: Some spans may not have been exported")

# Graceful shutdown
telemetry.shutdown()
```

## Troubleshooting

### Phoenix Connection Failed

**Symptom**: "Failed to create tracer for tenant X"

**Cause**: Phoenix server not running or wrong endpoint

**Solution**:
```bash
# Check Phoenix is running
curl http://localhost:4317

# Start Phoenix if needed
phoenix serve

# Verify endpoint in config
export PHOENIX_COLLECTOR_ENDPOINT=localhost:4317
```

### Spans Not Appearing

**Symptom**: No spans visible in Phoenix dashboard

**Causes**:
1. Telemetry disabled
2. Wrong project name
3. Batch export delay

**Solutions**:
```python
# 1. Check telemetry enabled
config = TelemetryConfig.from_env()
print(f"Enabled: {config.enabled}, Phoenix: {config.phoenix_enabled}")

# 2. Verify project name
project_name = config.get_project_name("acme", "routing")
print(f"Project: {project_name}")

# 3. Force flush to see pending spans
telemetry.force_flush()
```

### High Cache Miss Rate

**Symptom**: `cache_misses` >> `cache_hits`

**Cause**: Too many unique tenant:service combinations

**Solution**: Increase cache size
```python
config = TelemetryConfig(
    max_cached_tenants=200  # Increase from default 100
)
```

### Dropped Spans Warning

**Symptom**: "Dropped X spans due to full queue"

**Cause**: Span generation rate exceeds export rate

**Solution**: Increase queue size or batch size
```python
batch_config = BatchExportConfig(
    max_queue_size=4096,  # Increase from 2048
    max_export_batch_size=1024  # Increase from 512
)
```

## Best Practices

### Span Naming

1. **Use hierarchical names**: `cogniverse.routing`, `cogniverse.orchestration`
2. **Be specific**: `video_search.vespa_query` not `search`
3. **Use constants**: Define in `telemetry.config` to avoid typos

```python
from src.app.telemetry.config import SPAN_NAME_ROUTING

with telemetry.span(SPAN_NAME_ROUTING, tenant_id=tenant_id) as span:
    # Consistent naming across codebase
    pass
```

### Attribute Conventions

1. **Use dots for namespaces**: `routing.chosen_agent`, `search.strategy`
2. **Include units**: `latency_ms`, `size_bytes`
3. **Be consistent**: Same attribute names across spans

```python
# Good
span.set_attribute("search.latency_ms", 120)
span.set_attribute("search.results_count", 10)

# Bad
span.set_attribute("latency", 120)  # Missing unit
span.set_attribute("num_results", 10)  # Inconsistent naming
```

### Tenant Context

Always include tenant_id in spans:

```python
# Extract tenant from request
tenant_id = request.state.tenant_id

# Pass to all telemetry
with telemetry.span("search", tenant_id=tenant_id) as span:
    # Span exported to correct tenant project
    pass
```

### Test Mode

Use synchronous export in tests for immediate verification:

```bash
# Enable sync export for tests
export TELEMETRY_SYNC_EXPORT=true

# Run tests
pytest tests/
```

```python
# In test
def test_search_telemetry():
    telemetry = get_telemetry_manager()

    with telemetry.span("test", tenant_id="test") as span:
        span.set_attribute("test.name", "search")

    # Force flush to ensure export
    telemetry.force_flush()

    # Verify span in Phoenix
    client = px.Client()
    spans = client.get_spans(project_name="cogniverse-test-video-search")
    assert len(spans) > 0
```

## Related Documentation

- [Architecture Overview](architecture.md) - System architecture
- [Multi-Tenant System](multi-tenant-system.md) - Tenant isolation
- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [Optimization System](optimization-system.md) - GEPA/MIPRO/SIMBA

**Last Updated**: 2025-10-04
