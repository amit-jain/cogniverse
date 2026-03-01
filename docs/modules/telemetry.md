# Telemetry Module Documentation

**Packages:**

- **Foundation Interfaces**: `cogniverse-foundation` (libs/foundation/cogniverse_foundation/telemetry/)

- **Core Re-exports**: `cogniverse-core` (re-exports from foundation for convenience)

- **Phoenix Plugin**: `cogniverse-telemetry-phoenix` (libs/telemetry-phoenix/)
**Layer:** Foundation Layer (interfaces and infrastructure) + Core Layer (re-exports) + Plugin (Phoenix provider)

**Architecture Note:** Cogniverse uses a **plugin-based telemetry architecture** with three layers:
1. **Foundation Layer** (`cogniverse-foundation`): Telemetry provider interfaces, infrastructure, manager, and configuration
2. **Core Layer** (`cogniverse-core`): Re-exports from foundation for backward compatibility
3. **Plugin Layer** (`cogniverse-telemetry-phoenix`): Phoenix-specific implementation auto-discovered via Python entry points

This design enables clean separation between telemetry infrastructure and provider-specific implementations, allowing easy swapping of telemetry backends (Phoenix, Datadog, New Relic, etc.).

---

## Package Structure

### 1. Foundation Layer (cogniverse-foundation)
```text
libs/foundation/cogniverse_foundation/telemetry/
├── __init__.py              # Package initialization
├── manager.py               # TelemetryManager singleton
├── config.py                # TelemetryConfig and BatchExportConfig
├── context.py               # Span context helpers
├── exporter.py              # OTLP exporter utilities
├── registry.py              # Provider registry for auto-discovery
└── providers/               # Provider interfaces
    ├── __init__.py
    └── base.py              # TelemetryProvider abstract base class
```

**Purpose:** Telemetry provider interfaces and infrastructure with multi-tenant management, configuration, and context helpers. Zero dependencies on specific telemetry backends (Phoenix, LangSmith, etc.).

**Key Classes:**

- `TelemetryProvider`: Abstract base class for all telemetry providers
- `TelemetryConfig`: Configuration for telemetry systems with BatchExportConfig
- `TelemetryManager`: Singleton manager for multi-tenant tracer providers
- Context helpers for common operations (search, encode, backend)

### 2. Core Layer Re-exports (cogniverse-core)

The `cogniverse-core` package provides backward compatibility by re-exporting telemetry from foundation:

```python
# libs/core/cogniverse_core/telemetry/__init__.py
from cogniverse_foundation.telemetry import (
    TelemetryConfig,
    TelemetryManager,
    config,
    context,
    exporter,
    get_telemetry_manager,
    manager,
    providers,
    registry,
)
```

**What's Re-exported:**
- `TelemetryManager`, `TelemetryConfig`, `get_telemetry_manager`
- Submodules: `config`, `context`, `exporter`, `manager`, `providers`, `registry`

### 3. Agents Layer: Modality Metrics (cogniverse-agents)
```text
libs/agents/cogniverse_agents/routing/
└── modality_metrics.py      # ModalityMetricsTracker (for video, audio, images, etc.)
```

**Purpose:** Per-modality performance tracking for routing decisions.

**Key Classes:**

- `ModalityMetricsTracker`: Performance tracking per content type (VIDEO, IMAGE, AUDIO, DOCUMENT, TEXT, MIXED)

### 4. Plugin Layer: Phoenix Telemetry Provider (cogniverse-telemetry-phoenix)
```text
libs/telemetry-phoenix/cogniverse_telemetry_phoenix/
├── __init__.py              # Package initialization & exports
├── provider.py              # PhoenixProvider implementation
└── evaluation/              # Phoenix evaluation subsystem
    ├── __init__.py
    ├── analytics.py         # Analytics utilities
    ├── analytics_provider.py    # PhoenixAnalyticsProvider
    ├── evaluation_provider.py   # PhoenixEvaluationProvider
    ├── experiments.py       # Experiment management
    ├── framework.py         # Evaluation framework
    ├── monitoring.py        # Monitoring utilities
    ├── monitoring_provider.py   # PhoenixMonitoringProvider
    └── provider.py          # Evaluation provider base
```

**Purpose:** Phoenix-specific implementation of telemetry interfaces. Auto-discovered via Python entry points.

**Plugin Registration:** The Phoenix provider is auto-discovered via entry points defined in `pyproject.toml`:
```toml
[project.entry-points."cogniverse.telemetry.providers"]
phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"

[project.entry-points."cogniverse.evaluation.providers"]
phoenix = "cogniverse_telemetry_phoenix.evaluation.evaluation_provider:PhoenixEvaluationProvider"
```

**Benefits of Plugin Architecture:**

- **Swappable Providers**: Easy to add Datadog, New Relic, Jaeger, or custom providers

- **Zero Core Dependencies**: Core package doesn't depend on Phoenix SDK

- **Auto-discovery**: Providers automatically registered via entry points

- **Clean Separation**: Provider-specific code isolated in plugins

---

## Table of Contents
1. [Module Overview](#module-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Provider Abstraction](#provider-abstraction)
4. [Phoenix Implementation](#phoenix-implementation)
5. [Core Components](#core-components)
6. [Usage Examples](#usage-examples)
7. [Production Considerations](#production-considerations)
8. [Testing](#testing)

---

## Module Overview

### Purpose and Responsibilities

The Telemetry Module provides **multi-tenant observability** infrastructure for the Cogniverse system with:

- **Multi-Tenant Isolation**: Separate Phoenix projects per tenant for data isolation
- **Lazy Initialization**: Tracer providers created on-demand with LRU caching
- **Batch Export**: Configurable batch processing for high-throughput span export
- **Graceful Degradation**: System continues functioning even when telemetry fails
- **Performance Metrics**: Per-modality latency tracking, success rates, and error analysis
- **OpenTelemetry Integration**: Standards-based distributed tracing with Phoenix backend

### Key Features

1. **Singleton Manager Pattern**
   - Thread-safe singleton TelemetryManager
   - Global instance accessible throughout application
   - Configuration from environment variables

2. **Tenant-Aware Tracing**
   - Project-based separation: `cogniverse-{tenant_id}-{service}`
   - LRU cache for tenant providers (configurable max tenants)
   - Automatic tenant context propagation

3. **Flexible Export Modes**
   - **Production**: Batch export with queue management (async)
   - **Testing**: Synchronous export for immediate flush
   - Configurable queue size, batch size, timeout

4. **Modality Performance Tracking**
   - Rolling window metrics (configurable window size)
   - P50/P95/P99 latency percentiles
   - Success rates and error breakdowns
   - Throughput calculation (QPS)

5. **Context Helpers**
   - Pre-built span creators for search, encoding, backend operations
   - Exact format matching with old instrumentation system
   - OpenInference semantic conventions

### Dependencies

**Foundation Package** (`cogniverse-foundation`):

- `opentelemetry-api`: OpenTelemetry API
- `opentelemetry-sdk`: OpenTelemetry Python SDK
- `pydantic`: Data validation
- `sqlalchemy`: Database ORM
- `pandas`: Data structures

**Phoenix Plugin** (`cogniverse-telemetry-phoenix`):

- `arize-phoenix`: Phoenix observability platform
- `pandas`: Data structures

**Agents Package** (for ModalityMetricsTracker):

- `numpy`: Statistical calculations for metrics

---

## Architecture Diagrams

### 1. Multi-Tenant Telemetry Architecture

```mermaid
flowchart TB
    subgraph AppLayer["<span style='color:#000'>Application Layer</span>"]
        RoutingA["<span style='color:#000'>RoutingAgent<br/>Tenant A</span>"]
        VideoB["<span style='color:#000'>VideoAgent<br/>Tenant B</span>"]
        SummarizerA["<span style='color:#000'>SummarizerAgent<br/>Tenant A</span>"]
    end

    RoutingA --> Manager
    VideoB --> Manager
    SummarizerA --> Manager

    subgraph Manager["<span style='color:#000'>Telemetry Manager Singleton</span>"]
        Cache["<span style='color:#000'>Thread-Safe LRU Cache<br/>Tenant A Cache: Tracer Providers<br/>Tenant B Cache: Tracer Providers<br/>Configuration:<br/>• max_cached_tenants: 100 LRU eviction<br/>• tenant_project_template: cogniverse-tenant_id-service<br/>• batch_export_config: queue, batch size, timeout</span>"]
    end

    Manager --> ProviderA["<span style='color:#000'>TracerProvider<br/>Project:<br/>cogniverse-tenant-a-routing</span>"]
    Manager --> ProviderB["<span style='color:#000'>TracerProvider<br/>Project:<br/>cogniverse-tenant-b-video</span>"]
    Manager --> ProviderC["<span style='color:#000'>TracerProvider<br/>Project:<br/>cogniverse-tenant-a-summary</span>"]

    ProviderA --> Phoenix["<span style='color:#000'>Phoenix Backend<br/>OpenTelemetry<br/>• Project-based separation<br/>• Span storage<br/>• Analytics UI</span>"]
    ProviderB --> Phoenix
    ProviderC --> Phoenix

    style AppLayer fill:#90caf9,stroke:#1565c0,color:#000
    style Manager fill:#ffcc80,stroke:#ef6c00,color:#000
    style Cache fill:#b0bec5,stroke:#546e7a,color:#000
    style RoutingA fill:#90caf9,stroke:#1565c0,color:#000
    style VideoB fill:#90caf9,stroke:#1565c0,color:#000
    style SummarizerA fill:#90caf9,stroke:#1565c0,color:#000
    style ProviderA fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ProviderB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ProviderC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
```

**Key Points:**

- Single TelemetryManager instance per application

- Lazy creation of TracerProviders per tenant

- LRU cache evicts old providers when max_cached_tenants exceeded

- Each tenant gets isolated Phoenix project

---

### 2. Span Lifecycle and Export Flow

```mermaid
flowchart TB
    AppCode["<span style='color:#000'>Application Code<br/>with telemetry.span search, tenant_id=acme as span:<br/>span.set_attribute query, test<br/>business logic<br/>span.set_status Status.OK</span>"]

    AppCode --> TelManager["<span style='color:#000'>Telemetry Manager<br/>1. Get or create tracer for tenant:<br/>• Check LRU cache<br/>• Create TracerProvider if needed<br/>• Add tenant context attributes<br/><br/>2. Start span with tracer:<br/>• Add default attributes tenant.id, service.name, env<br/>• Add user attributes openinference.project.name, etc<br/>• Start as current context span</span>"]

    TelManager --> ProdProcessor{"<span style='color:#000'>Mode?</span>"}

    ProdProcessor -->|Production| BatchProc["<span style='color:#000'>Span Processor Production<br/>BatchSpanProcessor<br/>Span Queue max: 2048<br/>Contains: Span, Span, Span, ...<br/>Schedule every 500ms or when<br/>batch size 512 reached<br/>OTLP Exporter:<br/>• Endpoint: localhost:4317<br/>• Protocol: gRPC<br/>• Timeout: 30s</span>"]

    ProdProcessor -->|Test| SimpleProc["<span style='color:#000'>Span Processor Test Mode<br/>SimpleSpanProcessor SYNC<br/>• Immediate export no queue<br/>• 5s timeout to prevent hanging<br/>• Enabled via TELEMETRY_SYNC_EXPORT</span>"]

    BatchProc --> Phoenix["<span style='color:#000'>Phoenix Backend</span>"]
    SimpleProc --> Phoenix

    style AppCode fill:#90caf9,stroke:#1565c0,color:#000
    style TelManager fill:#ffcc80,stroke:#ef6c00,color:#000
    style ProdProcessor fill:#b0bec5,stroke:#546e7a,color:#000
    style BatchProc fill:#ce93d8,stroke:#7b1fa2,color:#000
    style SimpleProc fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
```

**Export Modes:**

1. **Production (Batch)**: Async export with queue, scheduled at intervals

2. **Testing (Sync)**: Immediate export for integration tests

**Configuration:**
```python
batch_config = BatchExportConfig(
    max_queue_size=2048,           # Max spans in queue
    max_export_batch_size=512,     # Spans per export
    export_timeout_millis=30_000,  # Export timeout
    schedule_delay_millis=500,     # Export interval
    drop_on_queue_full=True,       # Drop vs block when full
)
```

---

### 3. Phoenix Project Isolation

```mermaid
flowchart TB
    subgraph Phoenix["<span style='color:#000'>Phoenix Backend</span>"]
        subgraph Project1["<span style='color:#000'>Project: cogniverse-acme-corp-routing</span>"]
            Acme1Spans["<span style='color:#000'>Spans from Tenant: acme-corp<br/>Service: routing<br/>Span 1: routing | Span 2: search | Span 3: enhance | ...</span>"]
        end

        subgraph Project2["<span style='color:#000'>Project: cogniverse-acme-corp-video-search</span>"]
            Acme2Spans["<span style='color:#000'>Spans from Tenant: acme-corp<br/>Service: video-search<br/>Span 1: encode | Span 2: search | Span 3: results | ...</span>"]
        end

        subgraph Project3["<span style='color:#000'>Project: cogniverse-techstart-routing</span>"]
            TechSpans["<span style='color:#000'>Spans from Tenant: techstart<br/>Service: routing<br/>Span 1: routing | Span 2: search | Span 3: enhance | ...</span>"]
        end

        Info["<span style='color:#000'>Project Naming:<br/>Template: cogniverse-tenant_id-service<br/><br/>Benefits:<br/>✓ Complete data isolation per tenant<br/>✓ Independent analytics per tenant<br/>✓ Separate retention policies per tenant<br/>✓ Easy tenant data deletion</span>"]
    end

    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
    style Project1 fill:#90caf9,stroke:#1565c0,color:#000
    style Project2 fill:#ffcc80,stroke:#ef6c00,color:#000
    style Project3 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Acme1Spans fill:#b0bec5,stroke:#546e7a,color:#000
    style Acme2Spans fill:#b0bec5,stroke:#546e7a,color:#000
    style TechSpans fill:#b0bec5,stroke:#546e7a,color:#000
    style Info fill:#b0bec5,stroke:#546e7a,color:#000
```

**Project Name Resolution:**
```python
def get_project_name(tenant_id: str, service: str) -> str:
    return f"cogniverse-{tenant_id}-{service}"

# Examples:
# tenant="acme-corp", service="routing" → "cogniverse-acme-corp-routing"
# tenant="default", service="video-search" → "cogniverse-default-video-search"
```

---

## Provider Abstraction

**File:** `libs/foundation/cogniverse_foundation/telemetry/providers/base.py`

The telemetry system uses a **provider abstraction** that defines interfaces for telemetry operations. This allows swapping backends (Phoenix, LangSmith, Datadog) without changing application code.

### Store Interfaces

```mermaid
flowchart TB
    subgraph Abstraction["<span style='color:#000'>Foundation Layer - Provider Interfaces</span>"]
        TelemetryProvider["<span style='color:#000'>TelemetryProvider<br/><br/>• initialize(config)<br/>• configure_span_export()<br/>• session_context()</span>"]
        TraceStore["<span style='color:#000'>TraceStore<br/><br/>• get_spans()<br/>• get_span_by_id()</span>"]
        AnnotationStore["<span style='color:#000'>AnnotationStore<br/><br/>• add_annotation()<br/>• get_annotations()<br/>• log_evaluations()</span>"]
        DatasetStore["<span style='color:#000'>DatasetStore<br/><br/>• create_dataset()<br/>• get_dataset()<br/>• append_to_dataset()</span>"]
        ExperimentStore["<span style='color:#000'>ExperimentStore<br/><br/>• create_experiment()<br/>• log_run()</span>"]
        AnalyticsStore["<span style='color:#000'>AnalyticsStore<br/><br/>• get_metrics()</span>"]
    end

    TelemetryProvider --> TraceStore
    TelemetryProvider --> AnnotationStore
    TelemetryProvider --> DatasetStore
    TelemetryProvider --> ExperimentStore
    TelemetryProvider --> AnalyticsStore

    subgraph Implementation["<span style='color:#000'>Plugin Layer - Phoenix Implementation</span>"]
        PhoenixProvider["<span style='color:#000'>PhoenixProvider</span>"]
        PhoenixTraceStore["<span style='color:#000'>PhoenixTraceStore</span>"]
        PhoenixAnnotationStore["<span style='color:#000'>PhoenixAnnotationStore</span>"]
        PhoenixDatasetStore["<span style='color:#000'>PhoenixDatasetStore</span>"]
        PhoenixExperimentStore["<span style='color:#000'>PhoenixExperimentStore</span>"]
        PhoenixAnalyticsStore["<span style='color:#000'>PhoenixAnalyticsStore</span>"]
    end

    PhoenixProvider -.->|implements| TelemetryProvider
    PhoenixTraceStore -.->|implements| TraceStore
    PhoenixAnnotationStore -.->|implements| AnnotationStore
    PhoenixDatasetStore -.->|implements| DatasetStore
    PhoenixExperimentStore -.->|implements| ExperimentStore
    PhoenixAnalyticsStore -.->|implements| AnalyticsStore

    style Abstraction fill:#a5d6a7,stroke:#388e3c,color:#000
    style Implementation fill:#ce93d8,stroke:#7b1fa2,color:#000
    style TelemetryProvider fill:#a5d6a7,stroke:#388e3c,color:#000
    style TraceStore fill:#a5d6a7,stroke:#388e3c,color:#000
    style AnnotationStore fill:#a5d6a7,stroke:#388e3c,color:#000
    style DatasetStore fill:#a5d6a7,stroke:#388e3c,color:#000
    style ExperimentStore fill:#a5d6a7,stroke:#388e3c,color:#000
    style AnalyticsStore fill:#a5d6a7,stroke:#388e3c,color:#000
    style PhoenixProvider fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixTraceStore fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixAnnotationStore fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixDatasetStore fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixExperimentStore fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixAnalyticsStore fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### TraceStore Interface

Query traces and spans from telemetry backend.

```python
class TraceStore(ABC):
    @abstractmethod
    async def get_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with standardized columns:
        - context.span_id: Unique span identifier
        - name: Span operation name
        - attributes.*: Span attributes (flattened)
        - start_time, end_time: Timestamps
        """
        pass

    @abstractmethod
    async def get_span_by_id(
        self, span_id: str, project: str
    ) -> Optional[Dict[str, Any]]:
        """Get single span by ID."""
        pass
```

### AnnotationStore Interface

Manage human/LLM annotations on spans for approval workflows and evaluations.

```python
class AnnotationStore(ABC):
    @abstractmethod
    async def add_annotation(
        self,
        span_id: str,
        name: str,           # e.g., "human_approval", "relevance_score"
        label: str,          # e.g., "approved", "rejected"
        score: float,        # 0.0-1.0
        metadata: Dict[str, Any],
        project: str,
    ) -> str:
        """Add annotation to a span."""
        pass

    @abstractmethod
    async def get_annotations(
        self,
        spans_df: pd.DataFrame,
        project: str,
        annotation_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get annotations for spans."""
        pass

    @abstractmethod
    async def log_evaluations(
        self,
        eval_name: str,
        evaluations_df: pd.DataFrame,  # Must contain: span_id, score, label
        project: str,
    ) -> None:
        """Bulk upload evaluation results as span annotations."""
        pass
```

### DatasetStore Interface

Manage training datasets for model fine-tuning.

```python
class DatasetStore(ABC):
    @abstractmethod
    async def create_dataset(
        self, name: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new dataset."""
        pass

    @abstractmethod
    async def get_dataset(self, name: str) -> pd.DataFrame:
        """Load dataset by name."""
        pass

    @abstractmethod
    async def append_to_dataset(self, name: str, data: pd.DataFrame) -> None:
        """Append records to existing dataset."""
        pass
```

### ExperimentStore Interface

Manage experiments and evaluation runs.

```python
class ExperimentStore(ABC):
    @abstractmethod
    async def create_experiment(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new experiment."""
        pass

    @abstractmethod
    async def log_run(
        self,
        experiment_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log experiment run with inputs/outputs."""
        pass
```

### AnalyticsStore Interface

Query aggregated metrics and analytics.

```python
class AnalyticsStore(ABC):
    @abstractmethod
    async def get_metrics(
        self,
        project: str,
        metric_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        - timestamp: Metric timestamp
        - metric_name: Metric name
        - value: Metric value
        """
        pass
```

### TelemetryProvider Base Class

Providers implement all store interfaces and handle backend-specific initialization.

```python
class TelemetryProvider(ABC):
    def __init__(self, name: str):
        self.name = name
        self._trace_store: Optional[TraceStore] = None
        self._annotation_store: Optional[AnnotationStore] = None
        self._dataset_store: Optional[DatasetStore] = None
        self._experiment_store: Optional[ExperimentStore] = None
        self._analytics_store: Optional[AnalyticsStore] = None

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize provider with configuration.
        Provider extracts its own keys from generic config dict.

        Example config for Phoenix:
            {"tenant_id": "acme", "http_endpoint": "http://localhost:6006"}

        Example config for LangSmith:
            {"tenant_id": "acme", "api_key": "xxx", "project": "my-project"}
        """
        pass

    @abstractmethod
    def configure_span_export(
        self,
        endpoint: str,
        project_name: str,
        use_batch_export: bool = True,
    ) -> Any:
        """Configure OTLP span export, returns TracerProvider."""
        pass

    @property
    def traces(self) -> TraceStore:
        """Get trace store (query spans)"""
        return self._trace_store

    @property
    def annotations(self) -> AnnotationStore:
        """Get annotation store (manage annotations)"""
        return self._annotation_store

    @property
    def datasets(self) -> DatasetStore:
        """Get dataset store (manage training datasets)"""
        return self._dataset_store

    @property
    def experiments(self) -> ExperimentStore:
        """Get experiment store (manage experiments/runs)"""
        return self._experiment_store

    @property
    def analytics(self) -> AnalyticsStore:
        """Get analytics store (query metrics/aggregations)"""
        return self._analytics_store

    @abstractmethod
    @contextmanager
    def session_context(self, session_id: str) -> Generator[None, None, None]:
        """Context manager for session tracking."""
        yield
```

---

## Phoenix Implementation

**Package:** `cogniverse-telemetry-phoenix`
**Location:** `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/`

Phoenix is the default telemetry provider, implementing all store interfaces using Phoenix AsyncClient.

### PhoenixProvider

**File:** `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/provider.py`

```python
from cogniverse_telemetry_phoenix import PhoenixProvider

# Initialize Phoenix provider (provider.initialize() is called by TelemetryManager)
# Required keys: tenant_id, http_endpoint, grpc_endpoint
provider = PhoenixProvider()
provider.initialize({
    "tenant_id": "acme",
    "http_endpoint": "http://localhost:6006",     # HTTP API endpoint (required)
    "grpc_endpoint": "http://localhost:4317",     # gRPC OTLP endpoint (required)
})

# Query spans
spans_df = await provider.traces.get_spans(
    project="cogniverse-acme-routing",
    start_time=datetime.now() - timedelta(hours=1),
    limit=100
)

# Add annotation
await provider.annotations.add_annotation(
    span_id="abc123",
    name="human_approval",
    label="approved",
    score=1.0,
    metadata={"reviewer": "user@example.com"},
    project="cogniverse-acme-routing"
)

# Create dataset
dataset_id = await provider.datasets.create_dataset(
    name="golden_queries_v1",
    data=approved_queries_df,
    metadata={"version": "1.0"}
)

# Log experiment run
run_id = await provider.experiments.log_run(
    experiment_id="routing_eval_001",
    inputs={"query": "find videos"},
    outputs={"agent": "video_search", "confidence": 0.95}
)
```

### PhoenixAnalyticsProvider

**File:** `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/evaluation/analytics_provider.py`

Provides analytics operations using Phoenix client.

```python
from cogniverse_telemetry_phoenix.evaluation.analytics_provider import PhoenixAnalyticsProvider

analytics = PhoenixAnalyticsProvider(phoenix_client)

# Get trace statistics
stats = await analytics.get_trace_statistics(
    project="cogniverse-acme-routing",
    start_time=datetime.now() - timedelta(hours=24)
)
# Returns: {count, mean_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, error_rate}
```

### PhoenixMonitoringProvider

**File:** `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/evaluation/monitoring_provider.py`

Provides monitoring operations including alerts and metrics windows.

```python
from cogniverse_telemetry_phoenix.evaluation.monitoring_provider import PhoenixMonitoringProvider

monitoring = PhoenixMonitoringProvider(phoenix_client)

# Create alert
alert_id = await monitoring.create_alert(
    name="high_latency",
    condition="latency > threshold",
    threshold=1000.0,  # 1s
    project="cogniverse-acme-routing"
)

# Get recent metrics window
metrics = await monitoring.get_metrics_window(
    project="cogniverse-acme-routing",
    window_minutes=5
)
```

### Plugin Registration

Phoenix provider is auto-discovered via Python entry points in `pyproject.toml`:

```toml
[project.entry-points."cogniverse.telemetry.providers"]
phoenix = "cogniverse_telemetry_phoenix:PhoenixProvider"

[project.entry-points."cogniverse.evaluation.providers"]
phoenix = "cogniverse_telemetry_phoenix.evaluation.evaluation_provider:PhoenixEvaluationProvider"
```

### Implementing a Custom Provider

To implement a new telemetry provider (e.g., LangSmith, Datadog):

```python
from cogniverse_foundation.telemetry.providers.base import (
    TelemetryProvider, TraceStore, AnnotationStore
)

class LangSmithTraceStore(TraceStore):
    async def get_spans(self, project, start_time=None, end_time=None,
                        filters=None, limit=1000) -> pd.DataFrame:
        # LangSmith-specific implementation
        runs = await self.client.list_runs(project_name=project, ...)
        return self._convert_to_dataframe(runs)

    async def get_span_by_id(self, span_id, project) -> Optional[Dict]:
        return await self.client.read_run(run_id=span_id)

class LangSmithProvider(TelemetryProvider):
    def __init__(self):
        super().__init__("langsmith")

    def initialize(self, config: Dict[str, Any]) -> None:
        self.client = LangSmithClient(api_key=config["api_key"])
        self._trace_store = LangSmithTraceStore(self.client)
        # Initialize other stores...

    def configure_span_export(self, endpoint, project_name, use_batch_export=True):
        # LangSmith uses different export mechanism
        return LangSmithTracerProvider(project=project_name)
```

Register via entry points:

```toml
[project.entry-points."cogniverse.telemetry.providers"]
langsmith = "my_package:LangSmithProvider"
```

---

## Core Components

### 1. TelemetryManager

**File:** `libs/foundation/cogniverse_foundation/telemetry/manager.py`

**Purpose:** Singleton manager for multi-tenant tracer providers with lazy initialization and LRU caching.

**Singleton Pattern:** TelemetryManager implements a thread-safe singleton pattern using `__new__` and a class-level lock. Only one instance exists per Python process, and subsequent instantiations return the same instance.

**Key Attributes:**
```python
_instance: Optional[TelemetryManager]  # Singleton instance (class variable)
_lock: threading.Lock                   # Thread-safe singleton creation (class variable)
_initialized: bool                      # Tracks if __init__ has run
config: TelemetryConfig                # Telemetry configuration
_tenant_providers: Dict[str, TracerProvider]  # Cached providers per tenant
_tenant_tracers: Dict[str, Tracer]     # Cached tracers per (tenant, service)
_project_configs: Dict[str, Dict]      # Per-project configuration overrides
_cache_hits: int                        # Cache performance metrics
_cache_misses: int
_failed_initializations: int
```

**Initialization Pattern:**

The singleton uses double-checked locking for thread safety:

```python
class TelemetryManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[TelemetryConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[TelemetryConfig] = None):
        if self._initialized:
            return  # Skip re-initialization
        # ... initialization code ...
        self._initialized = True
```

**Recommended Usage:**

```python
from cogniverse_foundation.telemetry.manager import TelemetryManager, get_telemetry_manager

# Option 1: Direct instantiation (singleton pattern handles multiple calls)
telemetry = TelemetryManager()  # Creates instance on first call
telemetry2 = TelemetryManager()  # Returns same instance
assert telemetry is telemetry2  # True

# Option 2: Use the convenience function (preferred)
telemetry = get_telemetry_manager()  # Gets or creates singleton

# Configuration is only applied on FIRST instantiation
# To reconfigure, use reset() first (TESTS ONLY):
TelemetryManager.reset()  # Clears singleton, shuts down providers
telemetry = TelemetryManager(new_config)  # Fresh instance with new config
```

**Main Methods:**

#### `__new__(cls, config: Optional[TelemetryConfig] = None) -> TelemetryManager`
Thread-safe singleton constructor.

**Parameters:**

- `config`: Required telemetry configuration (only applied on first instantiation; raises `ValueError` if None). Use `get_telemetry_manager()` to auto-load from ConfigManager.

**Returns:** Singleton TelemetryManager instance

**Example:**
```python
# First call creates instance with config
config = TelemetryConfig(enabled=True, level=TelemetryLevel.DETAILED)
manager1 = TelemetryManager(config)

# Subsequent calls return same instance (config parameter ignored)
manager2 = TelemetryManager()
assert manager1 is manager2  # True
assert manager2.config is config  # True - uses original config
```

#### `reset() -> None` (class method)
Reset singleton instance - **FOR TESTS ONLY**.

Shuts down all tracer providers, clears caches, and resets the singleton. This allows tests to start with a fresh TelemetryManager.

**Example:**
```python
# In test setup/teardown
TelemetryManager.reset()  # Clear singleton state

# Now can create fresh instance with test config
test_config = TelemetryConfig(
    otlp_endpoint="localhost:24317",
    batch_config=BatchExportConfig(use_sync_export=True)
)
manager = TelemetryManager(test_config)
```

#### `get_telemetry_manager(config_manager=None, tenant_id="default") -> TelemetryManager` (module function)
Get the global telemetry manager instance. On first call, loads config from ConfigManager automatically. This is the preferred way to access the singleton.

**Parameters:**
- `config_manager`: Optional ConfigManager instance. If None on first call, creates one via `create_default_config_manager()`.
- `tenant_id`: Tenant ID for loading telemetry config (default: "default").

**Example:**
```python
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

# Throughout application code (auto-loads config from ConfigManager)
telemetry = get_telemetry_manager()
with telemetry.span("operation", tenant_id="acme") as span:
    span.set_attribute("key", "value")
```

---

#### `get_tracer(tenant_id: str, project_name: Optional[str] = None) -> Optional[Tracer]`
Get or create tracer for a specific tenant. **Note: This is LEGACY - use `span()` instead.**

**Parameters:**

- `tenant_id`: Tenant identifier for project isolation

- `project_name`: Optional project name for management operations (e.g., "experiments", "synthetic_data", "system"). If None, uses tenant-only project.

**Returns:**

- `Tracer` instance if telemetry enabled

- `None` if telemetry disabled or initialization failed

**Caching Logic:**

1. Check LRU cache for `{tenant_id}:{project_name}` key

2. If cached, return tracer (cache hit)

3. If not cached:
   - Create TracerProvider for tenant if needed
   - Get tracer from provider
   - Cache tracer with LRU eviction
4. Return tracer or None on error

**Example:**
```python
manager = TelemetryManager()

# Get tracer for tenant
tracer = manager.get_tracer("acme-corp", "routing")

# Use tracer
with tracer.start_as_current_span("process_query") as span:
    span.set_attribute("query", "test")
```

---

#### `span(name: str, tenant_id: str, project_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> ContextManager`
Context manager for creating tenant-specific spans.

**Parameters:**

- `name`: Span name (e.g., "search", "cogniverse.routing")

- `tenant_id`: Tenant identifier

- `project_name`: Optional project name for span isolation

- `attributes`: Optional span attributes

**Automatic Attributes:**

- `tenant.id`: Tenant identifier

- `service.name`: Service name

- `environment`: Environment (development, production)

**Graceful Degradation:**

- If telemetry disabled or fails, yields no-op span

- Application continues normally even when telemetry fails

**Example:**
```python
manager = TelemetryManager()

# Basic span
with manager.span("search", tenant_id="acme-corp") as span:
    span.set_attribute("query", "test")
    results = search(query)
    span.set_attribute("num_results", len(results))

# Span with custom attributes
with manager.span(
    "cogniverse.routing",
    tenant_id="acme-corp",
    attributes={
        "openinference.project.name": "cogniverse-acme-corp-routing",
        "routing.chosen_agent": "video_search"
    }
) as span:
    decision = route_query(query)
```

---

#### `session_span(name: str, tenant_id: str, session_id: str, project_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> ContextManager`
Context manager for creating spans within a session context.

**Purpose:** Links multiple requests in a multi-turn conversation session. All spans created within a session share the same `session.id` attribute and are grouped together in Phoenix Sessions view.

**Parameters:**

- `name`: Span name

- `tenant_id`: Tenant identifier

- `session_id`: Session identifier (client-generated UUID)

- `project_name`: Optional Phoenix project name

- `attributes`: Optional span attributes

**Automatic Attributes:**

- `session.id`: Session identifier (for grouping in Phoenix)

- All attributes from `span()` method

**Example:**
```python
manager = TelemetryManager()

# Multi-turn conversation with session tracking
session_id = "user-session-uuid-123"

# Turn 1: First search
with manager.session_span("search", tenant_id="acme", session_id=session_id) as span:
    span.set_attribute("query", "find basketball videos")
    results = search("find basketball videos")

# Turn 2: Follow-up search (same session)
with manager.session_span("search", tenant_id="acme", session_id=session_id) as span:
    span.set_attribute("query", "show me dunks")
    results = search("show me dunks")

# Both spans will be grouped in Phoenix Sessions view under session_id
```

**Phoenix Sessions View:**

- Traces with the same `session.id` are grouped together

- Enables viewing complete conversation trajectories

- Supports session-level evaluation and fine-tuning data extraction

---

#### `register_project(tenant_id: str, project_name: str, **kwargs) -> None`
Register a project with optional config overrides.

**Purpose:** Allows per-project configuration (e.g., different telemetry endpoints for tests). Single source of truth for project settings.

**Parameters:**

- `tenant_id`: Tenant identifier

- `project_name`: Project name (e.g., "search", "synthetic_data", "routing")

- `**kwargs`: Optional overrides:
  - `otlp_endpoint`: Override OTLP gRPC endpoint for span export
  - `http_endpoint`: Override HTTP endpoint for span queries
  - `use_sync_export`: Override batch/sync export mode

**Project Naming:** `cogniverse-{tenant_id}-{project_name}`

**Example:**
```python
manager = TelemetryManager()

# Use defaults from config
manager.register_project(tenant_id="customer-123", project_name="search")

# Override endpoints for tests
manager.register_project(
    tenant_id="test-tenant1",
    project_name="synthetic_data",
    otlp_endpoint="http://localhost:24317",
    http_endpoint="http://localhost:26006",
    use_sync_export=True
)
```

---

#### `session(tenant_id: str, session_id: str, project_name: Optional[str] = None) -> ContextManager`
Context manager for session tracking without creating a span.

**Purpose:** Wraps multiple operations that should share a session_id. All spans created within this context will be associated with the session.

**Parameters:**

- `tenant_id`: Tenant identifier

- `session_id`: Session identifier for cross-request correlation

- `project_name`: Optional project name suffix

**Example:**
```python
manager = TelemetryManager()

# Wrap multiple spans in a session context
with manager.session(tenant_id="acme", session_id="user-session-abc"):
    # Multiple spans created here share the session_id
    with manager.span("operation1", tenant_id="acme") as span:
        pass
    with manager.span("operation2", tenant_id="acme") as span:
        pass
```

---

#### `get_provider(tenant_id: str, project_name: Optional[str] = None) -> TelemetryProvider`
Get telemetry provider for querying spans/annotations/datasets.

**Purpose:** This is separate from span export (which uses OpenTelemetry OTLP). Providers are used for reading data from the telemetry backend.

**Parameters:**

- `tenant_id`: Tenant identifier

- `project_name`: Optional project name to get project-specific config

**Returns:** `TelemetryProvider` instance

**Raises:** `ValueError` if no providers available or provider initialization fails

**Example:**
```python
manager = TelemetryManager()

# Get provider for querying
provider = manager.get_provider(tenant_id="customer-123")

# Query spans
spans_df = await provider.traces.get_spans(
    project="cogniverse-customer-123-search",
    start_time=datetime(2025, 1, 1),
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

---

#### `force_flush(timeout_millis: int = 10000) -> bool`
Force flush all spans for all tenants.

**Parameters:**

- `timeout_millis`: Timeout in milliseconds for flush operation

**Returns:** `True` if all flushes succeeded, `False` otherwise

**Usage:** Call before application shutdown or after critical operations in tests

**Example:**
```python
manager = TelemetryManager()

# ... application code ...

# Ensure all spans exported before shutdown
success = manager.force_flush(timeout_millis=5000)
if not success:
    logger.warning("Some spans may not have been exported")
```

---

#### `get_stats() -> Dict[str, Any]`
Get telemetry manager statistics.

**Returns:**
```python
{
    "cache_hits": 150,          # Number of cache hits
    "cache_misses": 10,         # Number of cache misses
    "failed_initializations": 0,  # Failed tracer creations
    "cached_tenants": 5,        # Number of cached providers
    "cached_tracers": 8,        # Number of cached tracers
    "config": {
        "enabled": True,
        "level": "detailed",
        "environment": "production"
    }
}
```

**Example:**
```python
manager = TelemetryManager()
stats = manager.get_stats()
print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")
```

---

### 2. TelemetryConfig

**File:** `libs/foundation/cogniverse_foundation/telemetry/config.py`

**Purpose:** Configuration for telemetry system with environment variable support.

**Key Attributes:**
```python
# Core settings
enabled: bool                       # Enable/disable telemetry (default: true)
level: TelemetryLevel              # DISABLED, BASIC, DETAILED, VERBOSE
environment: str                    # development, production

# OpenTelemetry span export settings (generic OTLP - backend-agnostic)
otlp_enabled: bool                  # Enable OTLP span export (default: true)
otlp_endpoint: str                  # OTLP collector endpoint (default: localhost:4317)
otlp_use_tls: bool                 # Use TLS for OTLP connection (default: false)

# Provider selection (for querying spans/annotations/datasets)
provider: Optional[str]             # Provider name ("phoenix", "langsmith", None=auto-detect)
provider_config: Dict[str, Any]     # Provider-specific config (dict interpreted by provider)

# Multi-tenant settings
tenant_project_template: str       # "cogniverse-{tenant_id}"
tenant_service_template: str       # "cogniverse-{tenant_id}-{service}"
default_tenant_id: str             # "default"
max_cached_tenants: int            # LRU cache size (default: 100)
tenant_cache_ttl_seconds: int      # Cache TTL (default: 3600)

# Batch export settings
batch_config: BatchExportConfig    # Batch export configuration

# Service identification
service_name: str                  # Service name (default: "video-search")
service_version: str               # Service version (default: "1.0.0")

# Resource attributes
extra_resource_attributes: Dict[str, str]  # Additional resource attributes
```

**TelemetryLevel Enum:**
```python
DISABLED = "disabled"   # No telemetry
BASIC = "basic"         # Only search operations
DETAILED = "detailed"   # Search + encoders + backend
VERBOSE = "verbose"     # Everything including internal operations
```

**BatchExportConfig:**
```python
max_queue_size: int = 2048           # Max spans in queue
max_export_batch_size: int = 512     # Spans per export batch
export_timeout_millis: int = 30_000  # Export timeout
schedule_delay_millis: int = 500     # Export interval

# Queue behavior when full
drop_on_queue_full: bool = True      # Drop spans vs block
log_dropped_spans: bool = True
max_drop_log_rate_per_minute: int = 10

# Test mode
use_sync_export: bool = False        # Set via TELEMETRY_SYNC_EXPORT env var
```

**Main Methods:**

#### `from_dict(data: Dict[str, Any]) -> TelemetryConfig`
Deserialize from dictionary (used by ConfigManager for persistence).

#### `to_dict() -> Dict[str, Any]`
Serialize to dictionary for persistence (used by ConfigManager).

**Example:**
```python
# Get default config
config = TelemetryConfig()

# Custom: override defaults
config = TelemetryConfig(
    enabled=True,
    level=TelemetryLevel.DETAILED,
    otlp_endpoint="phoenix.internal:4317",
    max_cached_tenants=200
)
```

---

#### `get_project_name(tenant_id: str, service: Optional[str] = None) -> str`
Generate project name for a tenant.

**Parameters:**

- `tenant_id`: Tenant identifier

- `service`: Service name (defaults to self.service_name)

**Returns:** Project name formatted with template

**Example:**
```python
config = TelemetryConfig()
project = config.get_project_name("acme-corp", "routing")
# Returns: "cogniverse-acme-corp-routing"
```

---

#### `should_instrument_level(component: str) -> bool`
Check if a component should be instrumented based on level.

**Parameters:**

- `component`: Component name ("search_service", "backend", "encoder", "pipeline", "agents")

**Returns:** `True` if component should be instrumented

**Level-Based Components:**

- `DISABLED`: No components

- `BASIC`: search_service only

- `DETAILED`: search_service, backend, encoder

- `VERBOSE`: All components

**Example:**
```python
config = TelemetryConfig(level=TelemetryLevel.DETAILED)

config.should_instrument_level("search_service")  # True
config.should_instrument_level("backend")         # True
config.should_instrument_level("pipeline")        # False (VERBOSE only)
```

---

### 3. ModalityMetricsTracker

**File:** `libs/agents/cogniverse_agents/routing/modality_metrics.py`

**Purpose:** Track performance metrics per modality (text, video, image, audio) with rolling window statistics.

**Key Attributes:**
```python
window_size: int                              # Rolling window size (default: 1000)
modality_latencies: Dict[QueryModality, List[float]]  # Latency values per modality
modality_errors: Dict[QueryModality, Dict[str, int]]  # Error counts by type
modality_success: Dict[QueryModality, int]    # Success counts
modality_requests: Dict[QueryModality, int]   # Total requests
first_request_time: Dict[QueryModality, datetime]  # First request timestamp
last_request_time: Dict[QueryModality, datetime]   # Last request timestamp
```

**Main Methods:**

#### `record_modality_execution(modality: QueryModality, latency_ms: float, success: bool, error: Optional[str] = None)`
Record execution metrics for a modality.

**Parameters:**

- `modality`: Query modality (TEXT, VIDEO, IMAGE, AUDIO)

- `latency_ms`: Execution latency in milliseconds

- `success`: Whether execution succeeded

- `error`: Error message if failed (optional)

**Rolling Window:** Maintains last `window_size` latencies per modality

**Example:**
```python
tracker = ModalityMetricsTracker(window_size=1000)

# Record successful execution
tracker.record_modality_execution(
    modality=QueryModality.VIDEO,
    latency_ms=234.5,
    success=True
)

# Record failed execution
tracker.record_modality_execution(
    modality=QueryModality.TEXT,
    latency_ms=150.0,
    success=False,
    error="Connection timeout"
)
```

---

#### `get_modality_stats(modality: QueryModality) -> Dict[str, Any]`
Get aggregated statistics for a modality.

**Parameters:**

- `modality`: Query modality

**Returns:**
```python
{
    "modality": "video",
    "total_requests": 1000,
    "success_count": 980,
    "error_count": 20,
    "success_rate": 0.98,
    "p50_latency": 120.5,      # Median latency
    "p95_latency": 450.2,      # 95th percentile
    "p99_latency": 890.7,      # 99th percentile
    "avg_latency": 200.3,
    "min_latency": 45.1,
    "max_latency": 1200.5,
    "error_breakdown": {
        "connection_timeout": 15,
        "invalid_format": 5
    },
    "throughput_qps": 12.5     # Queries per second
}
```

**Example:**
```python
tracker = ModalityMetricsTracker()

# ... record executions ...

stats = tracker.get_modality_stats(QueryModality.VIDEO)
print(f"Video search P95 latency: {stats['p95_latency']:.0f}ms")
print(f"Success rate: {stats['success_rate']:.2%}")
```

---

#### `get_all_stats() -> Dict[str, Dict[str, Any]]`
Get stats for all modalities with activity.

**Returns:** Dictionary mapping modality names to stats (only includes modalities with requests)

**Example:**
```python
tracker = ModalityMetricsTracker()

# ... record executions ...

all_stats = tracker.get_all_stats()
for modality, stats in all_stats.items():
    print(f"{modality}: {stats['total_requests']} requests, {stats['success_rate']:.2%} success")
```

---

#### `get_summary_stats() -> Dict[str, Any]`
Get summary statistics across all modalities.

**Returns:**
```python
{
    "total_requests": 5000,
    "overall_success_rate": 0.96,
    "active_modalities": 3,
    "avg_latency_p95": 350.5,
    "modality_breakdown": {
        "video": 3000,
        "text": 1500,
        "image": 500
    }
}
```

**Example:**
```python
tracker = ModalityMetricsTracker()
summary = tracker.get_summary_stats()
print(f"Overall success rate: {summary['overall_success_rate']:.2%}")
print(f"Active modalities: {summary['active_modalities']}")
```

---

#### `get_slowest_modalities(top_k: int = 3) -> List[Dict[str, Any]]`
Get slowest modalities by P95 latency.

**Parameters:**

- `top_k`: Number of slowest modalities to return

**Returns:** List of `{modality, p95_latency, total_requests}` sorted by P95 latency (descending)

**Example:**
```python
tracker = ModalityMetricsTracker()
slowest = tracker.get_slowest_modalities(top_k=3)
for entry in slowest:
    print(f"{entry['modality']}: P95={entry['p95_latency']:.0f}ms ({entry['total_requests']} requests)")
```

---

#### `get_error_prone_modalities(min_error_rate: float = 0.1) -> List[Dict[str, Any]]`
Get modalities with high error rates.

**Parameters:**

- `min_error_rate`: Minimum error rate to include (0-1, default: 0.1 = 10%)

**Returns:** List of `{modality, error_rate, error_count, error_breakdown}` sorted by error rate (descending)

**Example:**
```python
tracker = ModalityMetricsTracker()
error_prone = tracker.get_error_prone_modalities(min_error_rate=0.05)
for entry in error_prone:
    print(f"{entry['modality']}: {entry['error_rate']:.2%} error rate")
    print(f"  Error breakdown: {entry['error_breakdown']}")
```

---

### 4. Telemetry Context Helpers

**File:** `libs/foundation/cogniverse_foundation/telemetry/context.py`

**Purpose:** Pre-built span creators matching old instrumentation format with OpenInference semantic conventions.

**Main Functions:**

#### `search_span(tenant_id: str, query: str, top_k: int = 10, ranking_strategy: str = "default", profile: str = "unknown", backend: str = "vespa") -> ContextManager`
Create search service span.

**Attributes Set:**

- `openinference.span.kind`: "CHAIN"

- `operation.name`: "search"

- `backend`, `query`, `strategy`, `top_k`, `profile`

- `input.value`: JSON-encoded query parameters

- `latency_ms`: Search latency (automatic)

**Example:**
```python
from cogniverse_foundation.telemetry.context import search_span

with search_span(
    tenant_id="acme-corp",
    query="Marie Curie radioactivity",
    top_k=10,
    ranking_strategy="HYBRID_FLOAT_BM25"
) as span:
    results = vespa_client.search(query)
    span.set_attribute("num_results", len(results))
```

---

#### `encode_span(tenant_id: str, encoder_type: str, query_length: int = 0, query: str = "") -> ContextManager`
Create encoder span.

**Attributes Set:**

- `openinference.span.kind`: "EMBEDDING"

- `operation.name`: `f"encode.{encoder_type.lower()}"`

- `encoder_type`, `query_length`

- `input.value`: Query text

- `encoding_time_ms`: Encoding time (automatic)

**Example:**
```python
from cogniverse_foundation.telemetry.context import encode_span

with encode_span(
    tenant_id="acme-corp",
    encoder_type="ColPali",
    query="test query"
) as span:
    embeddings = encoder.encode(query)
    span.set_attribute("embedding_dim", embeddings.shape[-1])
```

---

#### `backend_search_span(tenant_id: str, backend_type: str = "vespa", schema_name: str = "unknown", ranking_strategy: str = "default", top_k: int = 10, has_embeddings: bool = False, query_text: str = "") -> ContextManager`
Create backend search span.

**Attributes Set:**

- `openinference.span.kind`: "RETRIEVER"

- `operation.name`: "search.execute"

- `backend`, `query`, `strategy`, `top_k`, `schema`, `has_embeddings`

- `input.value`: JSON-encoded search parameters

- `latency_ms`: Search latency (automatic)

**Example:**
```python
from cogniverse_foundation.telemetry.context import backend_search_span

with backend_search_span(
    tenant_id="acme-corp",
    schema_name="video_colpali_mv",
    ranking_strategy="BINARY_BINARY",
    top_k=10,
    has_embeddings=True
) as span:
    results = vespa_client.execute_query(yql, embeddings)
    span.set_attribute("num_results", len(results))
```

---

#### `add_search_results_to_span(span, results)`
Add search results details to span.

**Attributes Set:**

- `num_results`: Number of results

- `top_score`: Score of top result

**Events Added:**

- `search_results`: Top 3 results with rank, document_id, video_id, score, content_type

**Example:**
```python
from cogniverse_foundation.telemetry.context import search_span, add_search_results_to_span

with search_span(tenant_id="acme", query="test") as span:
    results = search(query)
    add_search_results_to_span(span, results)
```

---

#### `@with_telemetry(span_name: str, tenant_id_param: str = "tenant_id", extract_attributes: Optional[Dict[str, str]] = None)`
Decorator for automatic telemetry instrumentation.

**Parameters:**

- `span_name`: Name for the span

- `tenant_id_param`: Function parameter name containing tenant_id (default: "tenant_id")

- `extract_attributes`: Map of span attribute names to function parameter names

**Example:**
```python
from cogniverse_foundation.telemetry.context import with_telemetry

@with_telemetry(
    span_name="video_search.search",
    extract_attributes={
        "query": "query_text",
        "top_k": "max_results"
    }
)
def search_videos(tenant_id: str, query_text: str, max_results: int = 10):
    # Span automatically created with tenant_id, query, top_k attributes
    return perform_search(query_text, max_results)

# Usage
results = search_videos(tenant_id="acme", query_text="test", max_results=5)
```

---

## Usage Examples

### Example 1: Basic Multi-Tenant Telemetry Setup

```python
"""
Initialize telemetry manager and use it across application.
"""
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.config import TelemetryConfig, TelemetryLevel

# Initialize once at application startup
config = TelemetryConfig(
    enabled=True,
    level=TelemetryLevel.DETAILED,
    otlp_endpoint="localhost:4317",
    max_cached_tenants=100
)
telemetry = TelemetryManager(config)

# Use in request handlers
def handle_search_request(tenant_id: str, query: str):
    """Process search request with telemetry."""
    with telemetry.span("search_service.search", tenant_id=tenant_id) as span:
        span.set_attribute("query", query)
        span.set_attribute("user_agent", "mobile-app")

        # Business logic
        results = perform_search(query)

        # Add result metrics
        span.set_attribute("num_results", len(results))
        span.set_attribute("top_score", results[0].score if results else 0)

        return results

# Different tenants automatically isolated
results_acme = handle_search_request("acme-corp", "test query")  # → cogniverse-acme-corp-video-search
results_tech = handle_search_request("techstart", "test query")   # → cogniverse-techstart-video-search
```

---

### Example 2: Nested Spans with Context Propagation

```python
"""
Create nested span hierarchy for complex operations.
"""
from cogniverse_foundation.telemetry.manager import TelemetryManager

telemetry = TelemetryManager()

def process_video_search(tenant_id: str, query: str):
    """Multi-step video search with nested spans."""

    # Parent span for entire search operation
    with telemetry.span(
        "video_search.process",
        tenant_id=tenant_id,
        attributes={"query": query}
    ) as parent_span:

        # Child span: Query encoding
        with telemetry.span(
            "video_search.encode",
            tenant_id=tenant_id,
            attributes={"encoder": "ColPali"}
        ) as encode_span:
            embeddings = encode_query(query)
            encode_span.set_attribute("embedding_dim", embeddings.shape[-1])

        # Child span: Vespa search
        with telemetry.span(
            "video_search.vespa_search",
            tenant_id=tenant_id,
            attributes={
                "schema": "video_colpali_mv",
                "ranking": "BINARY_BINARY"
            }
        ) as search_span:
            results = vespa_client.search(embeddings)
            search_span.set_attribute("num_results", len(results))

        # Child span: Post-processing
        with telemetry.span(
            "video_search.postprocess",
            tenant_id=tenant_id
        ) as postprocess_span:
            filtered_results = filter_results(results)
            postprocess_span.set_attribute("filtered_count", len(filtered_results))

        parent_span.set_attribute("total_results", len(filtered_results))
        return filtered_results
```

**Resulting Phoenix trace hierarchy:**
```text
video_search.process (parent)
├── video_search.encode
├── video_search.vespa_search
└── video_search.postprocess
```

---

### Example 3: Using Context Helpers for Standard Operations

```python
"""
Use pre-built context helpers for common operations.
"""
from cogniverse_foundation.telemetry.context import search_span, encode_span, backend_search_span

def search_videos_with_telemetry(tenant_id: str, query: str):
    """Search with standardized telemetry spans."""

    # High-level search span
    with search_span(
        tenant_id=tenant_id,
        query=query,
        top_k=10,
        ranking_strategy="HYBRID_FLOAT_BM25",
        profile="video_colpali_mv"
    ) as span:

        # Encoding span
        with encode_span(
            tenant_id=tenant_id,
            encoder_type="ColPali",
            query=query
        ) as enc_span:
            embeddings = colpali_encoder.encode(query)

        # Backend search span
        with backend_search_span(
            tenant_id=tenant_id,
            schema_name="video_colpali_mv_frame",
            ranking_strategy="BINARY_BINARY",
            top_k=10,
            has_embeddings=True,
            query_text=query
        ) as backend_span:
            results = vespa_client.search(query, embeddings)

        # Add results to parent span
        from cogniverse_foundation.telemetry.context import add_search_results_to_span
        add_search_results_to_span(span, results)

        return results
```

**Generated Spans:**

- `search_service.search` (openinference.span.kind=CHAIN)

- `encoder.colpali.encode` (openinference.span.kind=EMBEDDING)

- `search.execute` (openinference.span.kind=RETRIEVER)

---

### Example 4: Per-Modality Performance Tracking

```python
"""
Track performance metrics per query modality.
"""
from cogniverse_agents.routing.modality_metrics import ModalityMetricsTracker
from cogniverse_agents.search.multi_modal_reranker import QueryModality
import time

# Initialize tracker
tracker = ModalityMetricsTracker(window_size=1000)

def search_with_metrics(query: str, modality: QueryModality):
    """Search with modality-specific performance tracking."""
    start_time = time.time()
    success = False
    error = None

    try:
        # Perform search
        results = perform_search(query, modality)
        success = True
        return results

    except Exception as e:
        error = str(e)
        raise

    finally:
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        tracker.record_modality_execution(
            modality=modality,
            latency_ms=latency_ms,
            success=success,
            error=error
        )

# Execute searches
search_with_metrics("Marie Curie", QueryModality.VIDEO)
search_with_metrics("Show me documents", QueryModality.TEXT)
search_with_metrics("Find similar images", QueryModality.IMAGE)

# Get statistics
video_stats = tracker.get_modality_stats(QueryModality.VIDEO)
print(f"Video Search Performance:")
print(f"  P95 Latency: {video_stats['p95_latency']:.0f}ms")
print(f"  Success Rate: {video_stats['success_rate']:.2%}")
print(f"  Throughput: {video_stats['throughput_qps']:.1f} QPS")

# Find slowest modalities
slowest = tracker.get_slowest_modalities(top_k=3)
for entry in slowest:
    print(f"{entry['modality']}: {entry['p95_latency']:.0f}ms")

# Find error-prone modalities
error_prone = tracker.get_error_prone_modalities(min_error_rate=0.05)
for entry in error_prone:
    print(f"{entry['modality']}: {entry['error_rate']:.2%} errors")
    print(f"  Breakdown: {entry['error_breakdown']}")
```

---

### Example 5: Production Configuration with Batch Export

```python
"""
Production telemetry setup with optimized batch export.
"""
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.config import TelemetryConfig, TelemetryLevel, BatchExportConfig

# Production batch export configuration
batch_config = BatchExportConfig(
    max_queue_size=4096,           # Large queue for high throughput
    max_export_batch_size=1024,    # Large batches for efficiency
    export_timeout_millis=60_000,  # 60s timeout
    schedule_delay_millis=1000,    # Export every 1s
    drop_on_queue_full=True,       # Drop spans instead of blocking
    log_dropped_spans=True,
    max_drop_log_rate_per_minute=10
)

# Production telemetry configuration
production_config = TelemetryConfig(
    enabled=True,
    level=TelemetryLevel.DETAILED,
    environment="production",
    otlp_enabled=True,
    otlp_endpoint="phoenix.internal:4317",
    otlp_use_tls=True,
    max_cached_tenants=500,        # Support many tenants
    batch_config=batch_config,
    service_name="video-search",
    service_version="2.1.0"
)

# Initialize manager
telemetry = TelemetryManager(production_config)

# Use in high-throughput application
def handle_requests():
    """Process many requests with efficient batching."""
    for request in incoming_requests:
        tenant_id = extract_tenant_id(request)

        with telemetry.span("process_request", tenant_id=tenant_id) as span:
            span.set_attribute("request_id", request.id)
            process_request(request)

    # Flush before shutdown
    telemetry.force_flush(timeout_millis=10000)

# Monitor cache performance
stats = telemetry.get_stats()
cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
print(f"Tracer cache hit rate: {cache_hit_rate:.2%}")
print(f"Cached tenants: {stats['cached_tenants']}")
```

**Production Checklist:**

- ✅ Large queue size (4096+) for high throughput

- ✅ Batch exports (1024 spans/batch) for efficiency

- ✅ Drop on queue full (don't block application)

- ✅ TLS enabled for production Phoenix endpoint

- ✅ Large tenant cache (500+) for multi-tenant apps

- ✅ Force flush before shutdown

---

### Example 6: Test Mode with Synchronous Export

```python
"""
Testing configuration with immediate span export.
"""
import os
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.config import TelemetryConfig, TelemetryLevel, BatchExportConfig

# Set environment variable for sync export
os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

# Test configuration
test_config = TelemetryConfig(
    enabled=True,
    level=TelemetryLevel.VERBOSE,
    environment="test",
    otlp_enabled=True,
    otlp_endpoint="localhost:4317",
    otlp_use_tls=False,
    batch_config=BatchExportConfig(
        use_sync_export=True  # Synchronous export
    )
)

telemetry = TelemetryManager(test_config)

def test_search_telemetry():
    """Test that spans are immediately exported."""
    tenant_id = "test-tenant"

    # Create span - will be exported immediately
    with telemetry.span("test.search", tenant_id=tenant_id) as span:
        span.set_attribute("query", "test")

    # Force flush to ensure export (returns immediately in sync mode)
    success = telemetry.force_flush(timeout_millis=5000)
    assert success, "Flush should succeed in sync mode"

    # Verify span in Phoenix (query Phoenix API)
    # ... verification logic ...

# Sync export ensures spans available immediately for assertions
test_search_telemetry()
```

**Test Mode Benefits:**

- Immediate export (no batching delay)

- Predictable span timing for assertions

- Simpler debugging (spans appear immediately in Phoenix)

---

## Production Considerations

### 1. Multi-Tenant Isolation

**Complete Data Separation:**
```python
# Each tenant gets isolated Phoenix project
tenant_a_project = "cogniverse-acme-corp-routing"
tenant_b_project = "cogniverse-techstart-routing"

# Benefits:
# ✓ Independent analytics per tenant
# ✓ Separate retention policies
# ✓ Easy tenant data deletion (GDPR compliance)
# ✓ No cross-tenant data leakage
```

**Tenant Context Propagation:**

- Tenant ID automatically added to all spans

- Service name identifies span origin

- Environment tag for development/staging/production separation

**Best Practices:**

- Use consistent tenant ID format across application

- Include tenant ID in all telemetry spans

- Monitor per-tenant span volume to detect anomalies

---

### 2. Performance and Scalability

**LRU Cache Configuration:**
```python
# Adjust cache size based on tenant count
config = TelemetryConfig(
    max_cached_tenants=500  # Increase for many tenants
)

# Monitor cache performance
stats = telemetry.get_stats()
hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])

# Target: >95% cache hit rate
if hit_rate < 0.95:
    logger.warning(f"Low cache hit rate: {hit_rate:.2%}")
```

**Batch Export Tuning:**
```python
# High-throughput configuration
batch_config = BatchExportConfig(
    max_queue_size=8192,          # Larger queue for burst traffic
    max_export_batch_size=2048,   # Larger batches
    schedule_delay_millis=2000,   # Less frequent exports
    drop_on_queue_full=True       # Don't block on full queue
)

# Low-latency configuration
batch_config = BatchExportConfig(
    max_queue_size=1024,
    max_export_batch_size=256,
    schedule_delay_millis=100,    # Frequent exports
    drop_on_queue_full=False      # Wait for queue space
)
```

**Memory Management:**

- Modality metrics use rolling window (default: 1000 samples)

- Adjust window size based on available memory:
```python
tracker = ModalityMetricsTracker(window_size=500)  # Lower memory
tracker = ModalityMetricsTracker(window_size=5000)  # More history
```

**Throughput Benchmarks:**

- Single tenant: 10,000+ spans/second

- Multi-tenant (100 tenants): 5,000+ spans/second per tenant

- LRU cache hit rate: >95% in production

---

### 3. Graceful Degradation

**No-Op Spans:**
```python
# If telemetry fails, application continues with no-op spans
with telemetry.span("search", tenant_id="acme") as span:
    # Span might be NoOpSpan if telemetry disabled/failed
    span.set_attribute("query", "test")  # Safe to call
    results = search(query)  # Business logic always executes
```

**Error Handling:**
```python
# Manager catches and logs telemetry errors
try:
    tracer = manager.get_tracer("tenant-123")
except Exception as e:
    logger.warning(f"Failed to create tracer: {e}")
    tracer = None  # Returns None instead of crashing

# Application handles None gracefully
if tracer:
    with tracer.start_as_current_span("operation"):
        do_work()
else:
    # No telemetry, but work continues
    do_work()
```

**Configuration Validation:**
```python
config = TelemetryConfig()
try:
    config.validate()
except ValueError as e:
    logger.error(f"Invalid telemetry config: {e}")
    # Disable telemetry gracefully
    config.enabled = False
```

---

### 4. Monitoring and Alerting

**Key Metrics to Monitor:**

1. **Cache Performance:**
```python
stats = telemetry.get_stats()
cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])

# Alert if hit rate < 95%
if cache_hit_rate < 0.95:
    alert("Low telemetry cache hit rate", severity="warning")
```

2. **Failed Initializations:**
```python
if stats['failed_initializations'] > 0:
    alert("Telemetry provider initialization failures", severity="error")
```

3. **Modality Performance:**
```python
tracker = ModalityMetricsTracker()

# Alert on high latency
slowest = tracker.get_slowest_modalities(top_k=1)
if slowest and slowest[0]['p95_latency'] > 1000:  # 1s threshold
    alert(f"High {slowest[0]['modality']} latency: {slowest[0]['p95_latency']}ms")

# Alert on high error rate
error_prone = tracker.get_error_prone_modalities(min_error_rate=0.10)
if error_prone:
    for entry in error_prone:
        alert(f"High {entry['modality']} error rate: {entry['error_rate']:.2%}")
```

4. **Export Queue Health:**
```python
# Monitor queue size via Phoenix metrics
# Alert if queue consistently full (indicates export bottleneck)
```

**Recommended Alerts:**

- Cache hit rate < 95%

- Failed initializations > 0

- Per-modality P95 latency > 1000ms

- Per-modality error rate > 10%

- Queue drops > 100/minute

---

### 5. Security and Compliance

**Tenant Data Isolation:**

- Each tenant's spans in separate Phoenix project

- No cross-tenant span visibility

- Per-tenant retention policies

**PII Handling:**
```python
# Avoid logging sensitive data in spans
with telemetry.span("search", tenant_id=tenant_id) as span:
    # ✅ Safe: Metadata only
    span.set_attribute("query_length", len(query))
    span.set_attribute("has_filters", bool(filters))

    # ❌ Unsafe: PII in span
    # span.set_attribute("user_email", user.email)
    # span.set_attribute("full_query", query)  # May contain PII
```

**Data Retention:**
```python
# Configure Phoenix retention per project
# Example: 30 days for production, 7 days for development
config = TelemetryConfig(
    extra_resource_attributes={
        "retention_days": "30" if environment == "production" else "7"
    }
)
# Note: tenant_project_template and tenant_service_template use defaults
```

**GDPR Compliance:**

- Easy tenant data deletion (delete Phoenix project)

- Audit log of span exports (Phoenix provides this)

- Minimal PII in spans (query length, not content)

---

### 6. Testing Strategies

**Unit Testing with Mocked Telemetry:**
```python
from unittest.mock import Mock, patch

def test_search_with_telemetry():
    """Test search with mocked telemetry."""
    mock_manager = Mock(spec=TelemetryManager)
    mock_span = Mock()
    mock_manager.span.return_value.__enter__.return_value = mock_span

    with patch('cogniverse_foundation.telemetry.get_telemetry_manager', return_value=mock_manager):
        results = search_with_telemetry("test query")

    # Verify span created
    mock_manager.span.assert_called_once_with(
        "search",
        tenant_id="test-tenant",
        attributes=ANY
    )

    # Verify attributes set
    mock_span.set_attribute.assert_any_call("query", "test query")
```

**Integration Testing with Real Phoenix:**
```python
import os

def test_telemetry_integration():
    """Test telemetry with real Phoenix instance."""
    # Enable sync export for tests
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

    telemetry = TelemetryManager()
    tenant_id = "test-tenant"

    # Create span
    with telemetry.span("test.search", tenant_id=tenant_id) as span:
        span.set_attribute("query", "test")

    # Force flush
    success = telemetry.force_flush(timeout_millis=5000)
    assert success

    # Query Phoenix to verify span
    # ... Phoenix API query ...
```

**Performance Testing:**
```python
import time

def test_telemetry_throughput():
    """Test telemetry can handle high throughput."""
    telemetry = TelemetryManager()

    start = time.time()
    span_count = 10000

    for i in range(span_count):
        with telemetry.span(f"test.span.{i}", tenant_id="perf-test") as span:
            span.set_attribute("iteration", i)

    duration = time.time() - start
    throughput = span_count / duration

    print(f"Throughput: {throughput:.0f} spans/second")
    assert throughput > 5000, "Should handle 5k+ spans/second"
```

---

## Testing

### Key Test Files

**Unit Tests:**

- None (telemetry module focuses on integration)

**Integration Tests:**

- `tests/telemetry/integration/test_multi_tenant_telemetry.py`
  - Multi-tenant tracer provider isolation
  - LRU cache behavior
  - Batch vs sync export modes
  - Force flush functionality
  - Graceful degradation

**Test Scenarios:**

1. **Tenant Isolation:**
```python
def test_tenant_isolation():
    """Verify tenants get separate Phoenix projects."""
    manager = TelemetryManager()

    # Create tracers for different tenants
    tracer_a = manager.get_tracer("tenant-a")
    tracer_b = manager.get_tracer("tenant-b")

    # Verify different providers
    assert manager._tenant_providers["tenant-a"] != manager._tenant_providers["tenant-b"]
```

2. **LRU Cache Eviction:**
```python
def test_lru_eviction():
    """Verify old tracers evicted when cache full."""
    config = TelemetryConfig(max_cached_tenants=2)
    manager = TelemetryManager(config)

    # Create 3 tracers (exceeds cache limit)
    manager.get_tracer("tenant-1")
    manager.get_tracer("tenant-2")
    manager.get_tracer("tenant-3")

    # Oldest (tenant-1) should be evicted
    assert len(manager._tenant_tracers) == 2
```

3. **Sync Export:**
```python
def test_sync_export():
    """Verify synchronous export in test mode."""
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"
    manager = TelemetryManager()

    with manager.span("test", tenant_id="test") as span:
        span.set_attribute("key", "value")

    # Span should be exported immediately
    success = manager.force_flush(timeout_millis=1000)
    assert success
```

4. **Graceful Degradation:**
```python
def test_graceful_degradation():
    """Verify app continues when telemetry fails."""
    # Disable OTLP span export
    config = TelemetryConfig(otlp_enabled=False)
    manager = TelemetryManager(config)

    # Should return no-op span
    with manager.span("test", tenant_id="test") as span:
        # Should not crash
        span.set_attribute("key", "value")
```

---

**Test Coverage:**

- Multi-tenant isolation: ✅

- LRU cache behavior: ✅

- Batch export configuration: ✅

- Sync export mode: ✅

- Graceful degradation: ✅

- Performance metrics tracking: ✅

---

## Summary

The Telemetry Module provides **production-ready, multi-tenant observability** with:

**Core Features:**

- ✅ Multi-tenant isolation via Phoenix projects

- ✅ Lazy initialization with LRU caching

- ✅ Configurable batch export (async) or sync export (tests)

- ✅ Per-modality performance tracking

- ✅ Graceful degradation

- ✅ OpenTelemetry standards compliance

**Production Strengths:**

- Handles 10,000+ spans/second per tenant

- >95% cache hit rate in production

- Complete tenant data isolation

- Minimal performance overhead

- Robust error handling

**Integration Points:**

- All agents use `telemetry.span()` context manager

- Search operations use pre-built context helpers

- Modality metrics track per-query-type performance

- Phoenix provides analytics and visualization

---

**For detailed examples and production configurations, see:**

- Architecture Overview: `docs/architecture/overview.md`

- Agents Module: `docs/modules/agents.md`

- Common Module (Config): `docs/modules/common.md`

**Source Files:**

- Manager: `libs/foundation/cogniverse_foundation/telemetry/manager.py`

- Config: `libs/foundation/cogniverse_foundation/telemetry/config.py`

- Metrics: `libs/agents/cogniverse_agents/routing/modality_metrics.py`

- Context: `libs/foundation/cogniverse_foundation/telemetry/context.py`
