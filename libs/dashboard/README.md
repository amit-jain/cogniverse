# Cogniverse Dashboard

**Package**: `cogniverse-dashboard`
**Layer**: Application Layer (Light Blue/Purple)
**Version**: 0.1.0

Streamlit-based analytics UI providing Phoenix integration, experiment management, dataset visualization, and system monitoring for the Cogniverse SDK.

---

## Purpose

The `cogniverse-dashboard` package provides:
- **Phoenix Analytics**: Visualize traces, spans, and experiments from Phoenix
- **Experiment Management**: Track, compare, and analyze DSPy experiments
- **Dataset Management**: View, create, and manage evaluation datasets
- **System Monitoring**: Monitor Vespa, ingestion, and search performance
- **Multi-Tenant UI**: Switch between tenants and view tenant-specific data

---

## Architecture

### Position in 10-Package Structure

```
Foundation Layer (Blue)
├── cogniverse-sdk
└── cogniverse-foundation

Core Layer (Pink)
├── cogniverse-core ← cogniverse-dashboard depends on this
├── cogniverse-evaluation ← cogniverse-dashboard depends on this
└── cogniverse-telemetry-phoenix ← cogniverse-dashboard depends on this

Implementation Layer (Yellow/Green)
├── cogniverse-agents
├── cogniverse-vespa
└── cogniverse-synthetic

Application Layer (Light Blue/Purple)
├── cogniverse-runtime
└── cogniverse-dashboard ← YOU ARE HERE
```

### Dependencies

**Workspace Dependencies:**
- `cogniverse-core` (required) - Multi-agent orchestration
- `cogniverse-evaluation` (required) - Experiment tracking, datasets
- `cogniverse-telemetry-phoenix` (required) - Phoenix provider for spans/traces
- `cogniverse-foundation` (transitive) - Base configuration and telemetry

**External Dependencies:**
- `streamlit>=1.28.0` - Web UI framework
- `plotly>=5.17.0` - Interactive charts
- `pandas>=2.1.0` - Data manipulation
- `polars>=0.19.0` - Fast dataframe operations

---

## Key Features

### 1. Phoenix Trace Visualization

View and analyze OpenTelemetry traces from Phoenix:

```python
import streamlit as st
from cogniverse_dashboard.pages import phoenix_traces

# Display traces for tenant
phoenix_traces.show_traces(
    tenant_id="acme_corp",
    project="acme_corp_project",
    time_range="last_24h"
)
```

### 2. Experiment Comparison

Compare DSPy experiments side-by-side:

```python
from cogniverse_dashboard.pages import experiments

# Compare modality routing experiments
experiments.compare_experiments(
    experiment_ids=["exp_001", "exp_002", "exp_003"],
    metrics=["accuracy", "f1_score", "latency"]
)
```

### 3. Dataset Management

Create and manage evaluation datasets:

```python
from cogniverse_dashboard.pages import datasets

# Upload new dataset
datasets.upload_dataset(
    name="video_search_queries",
    format="csv",
    file_path="/data/queries.csv"
)

# View dataset statistics
datasets.show_statistics(dataset_id="dataset_001")
```

### 4. System Monitoring

Monitor system health and performance:

```python
from cogniverse_dashboard.pages import monitoring

# Show Vespa metrics
monitoring.show_vespa_metrics(tenant_id="acme_corp")

# Show ingestion pipeline status
monitoring.show_ingestion_status(tenant_id="acme_corp")

# Show search performance
monitoring.show_search_metrics(tenant_id="acme_corp")
```

---

## Installation

### Development (Editable Mode)

```bash
# From workspace root
uv sync

# Or install individually
uv pip install -e libs/dashboard
```

### Production

```bash
pip install cogniverse-dashboard

# Automatically installs:
# - cogniverse-core
# - cogniverse-evaluation
# - cogniverse-telemetry-phoenix
# - cogniverse-foundation
# - streamlit, plotly, pandas, polars
```

---

## Usage

### Starting the Dashboard

```bash
# Development mode
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py

# Production mode
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0

# With environment variables
export TENANT_ID="acme_corp"
export PHOENIX_ENDPOINT="http://localhost:6006"
export VESPA_URL="http://localhost:8080"
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py
```

### Custom Dashboard

```python
import streamlit as st
from cogniverse_dashboard import (
    PhoenixTraceViewer,
    ExperimentComparison,
    DatasetManager,
    SystemMonitor
)

# Set page config
st.set_page_config(
    page_title="Cogniverse Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Sidebar tenant selector
tenant_id = st.sidebar.selectbox(
    "Select Tenant",
    ["acme_corp", "globex_inc", "default"]
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Phoenix Traces",
    "Experiments",
    "Datasets",
    "Monitoring"
])

with tab1:
    viewer = PhoenixTraceViewer(tenant_id=tenant_id)
    viewer.render()

with tab2:
    comparison = ExperimentComparison(tenant_id=tenant_id)
    comparison.render()

with tab3:
    datasets = DatasetManager(tenant_id=tenant_id)
    datasets.render()

with tab4:
    monitor = SystemMonitor(tenant_id=tenant_id)
    monitor.render()
```

---

## Dashboard Pages

### Phoenix Traces

**URL**: `/Phoenix_Traces`

Features:
- View all traces for tenant
- Filter by time range, status, operation
- Drill down into spans
- View span attributes, events, links
- Export traces to CSV/JSON

### Experiments

**URL**: `/Experiments`

Features:
- List all experiments
- Compare experiment results
- View optimizer settings
- Visualize metric trends
- Export experiment data

### Datasets

**URL**: `/Datasets`

Features:
- List all datasets
- View dataset statistics
- Upload new datasets
- Preview dataset samples
- Validate dataset format
- Export datasets

### Monitoring

**URL**: `/Monitoring`

Features:
- Vespa health and metrics
- Ingestion pipeline status
- Search performance metrics
- Agent performance
- Resource utilization

### Admin

**URL**: `/Admin`

Features:
- Tenant management
- User management
- System configuration
- Schema deployment
- Backup/restore

---

## Configuration

Configuration via `SystemConfig` from `cogniverse-foundation`:

```python
from cogniverse_dashboard import create_dashboard
from cogniverse_foundation.config.unified_config import SystemConfig

config = SystemConfig(
    tenant_id="acme_corp",
    telemetry_url="http://localhost:6006",
    backend_url="http://localhost",
    backend_port=8080,
)

dashboard = create_dashboard(config=config)
```

### Environment Variables

```bash
# Required
export TENANT_ID="acme_corp"
export PHOENIX_ENDPOINT="http://localhost:6006"

# Optional
export VESPA_URL="http://localhost:8080"
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_THEME="dark"
```

### Streamlit Config

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B9D"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

---

## Development

### Running in Development

```bash
# Install dev dependencies
uv sync --dev

# Run dashboard
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py

# Watch for changes (auto-reload)
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --server.runOnSave true
```

### Running Tests

```bash
# Run dashboard tests
uv run pytest tests/dashboard/ -v

# Run specific component tests
uv run pytest tests/dashboard/unit/test_phoenix_viewer.py -v
```

### Code Style

```bash
# Format code
uv run ruff format libs/dashboard

# Lint code
uv run ruff check libs/dashboard

# Type check
uv run mypy libs/dashboard
```

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv sync

# Expose Streamlit port
EXPOSE 8501

# Run dashboard
CMD ["uv", "run", "streamlit", "run", "libs/dashboard/cogniverse_dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TENANT_ID=acme_corp
      - PHOENIX_ENDPOINT=http://phoenix:6006
      - VESPA_URL=http://vespa:8080
    depends_on:
      - phoenix
      - vespa

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"

  vespa:
    image: vespaengine/vespa
    ports:
      - "8080:8080"
      - "19071:19071"
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cogniverse-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cogniverse-dashboard
  template:
    metadata:
      labels:
        app: cogniverse-dashboard
    spec:
      containers:
      - name: dashboard
        image: cogniverse/dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: TENANT_ID
          value: "acme_corp"
        - name: PHOENIX_ENDPOINT
          value: "http://phoenix-service:6006"
        - name: VESPA_URL
          value: "http://vespa-service:8080"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  selector:
    app: cogniverse-dashboard
  ports:
  - port: 8501
    targetPort: 8501
  type: LoadBalancer
```

---

## Components

### Phoenix Trace Viewer

```python
from cogniverse_dashboard.components import PhoenixTraceViewer

viewer = PhoenixTraceViewer(
    tenant_id="acme_corp",
    project="acme_corp_project"
)

# Render trace viewer
viewer.render()

# Get selected trace
if viewer.selected_trace:
    st.write(f"Selected trace: {viewer.selected_trace.trace_id}")
```

### Experiment Comparison

```python
from cogniverse_dashboard.components import ExperimentComparison

comparison = ExperimentComparison(
    tenant_id="acme_corp"
)

# Render comparison view
comparison.render()

# Get comparison data
if comparison.selected_experiments:
    data = comparison.get_comparison_data()
    st.write(data)
```

### Dataset Manager

```python
from cogniverse_dashboard.components import DatasetManager

manager = DatasetManager(
    tenant_id="acme_corp"
)

# Render dataset manager
manager.render()

# Upload dataset
if st.button("Upload"):
    manager.upload_dataset(
        file=uploaded_file,
        name="new_dataset"
    )
```

---

## Visualizations

### Trace Timeline

```python
from cogniverse_dashboard.viz import TraceTimeline
import plotly.graph_objects as go

timeline = TraceTimeline(traces=traces)
fig = timeline.create_figure()

st.plotly_chart(fig, use_container_width=True)
```

### Experiment Metrics

```python
from cogniverse_dashboard.viz import ExperimentMetrics

metrics = ExperimentMetrics(experiments=experiments)
fig = metrics.create_comparison_chart(
    metrics=["accuracy", "f1_score"],
    chart_type="bar"
)

st.plotly_chart(fig, use_container_width=True)
```

### Dataset Distribution

```python
from cogniverse_dashboard.viz import DatasetDistribution

dist = DatasetDistribution(dataset=dataset)
fig = dist.create_histogram(column="query_length")

st.plotly_chart(fig, use_container_width=True)
```

---

## Multi-Tenant Support

The dashboard supports multi-tenant isolation:

### Tenant Selector

```python
# Sidebar tenant selector
tenant_id = st.sidebar.selectbox(
    "Select Tenant",
    options=["acme_corp", "globex_inc", "default"],
    key="tenant_selector"
)

# Use tenant_id throughout app
st.session_state.tenant_id = tenant_id
```

### Tenant-Specific Data

```python
# Phoenix traces for tenant
traces = phoenix_provider.get_traces(
    project=f"{tenant_id}_project",
    limit=100
)

# Experiments for tenant
experiments = eval_tracker.list_experiments(
    tenant_id=tenant_id
)

# Datasets for tenant
datasets = dataset_manager.list_datasets(
    tenant_id=tenant_id
)
```

---

## Documentation

- **Architecture**: [10-Package Architecture](../../docs/architecture/10-package-architecture.md)
- **Multi-Tenant**: [Multi-Tenant Architecture](../../docs/architecture/multi-tenant.md)
- **Streamlit Docs**: [Streamlit Documentation](https://docs.streamlit.io/)
- **Diagrams**: [SDK Architecture Diagrams](../../docs/diagrams/sdk-architecture-diagrams.md)

---

## Troubleshooting

### Common Issues

**1. Dashboard Won't Start**
```bash
# Check Streamlit version
uv run streamlit --version

# Check port availability
lsof -i :8501

# Run with debug logging
uv run streamlit run app.py --logger.level=debug
```

**2. Phoenix Connection Failed**
```bash
# Test Phoenix connectivity
curl http://localhost:6006/

# Check Phoenix project exists
curl http://localhost:6006/v1/projects
```

**3. No Traces Visible**
- Verify Phoenix project name: `{tenant_id}_project`
- Check traces exist in Phoenix UI: http://localhost:6006
- Ensure tenant_id matches

**4. Experiment Data Missing**
- Verify experiments exist: Use evaluation package
- Check database connection
- Ensure experiment tracking is enabled

---

## Performance Tips

### Caching

```python
import streamlit as st

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_traces(tenant_id: str):
    return phoenix_provider.get_traces(
        project=f"{tenant_id}_project"
    )

# Use cached function
traces = load_traces(tenant_id)
```

### Pagination

```python
# Paginate large datasets
page_size = 100
page_number = st.number_input("Page", min_value=1)

offset = (page_number - 1) * page_size
traces = phoenix_provider.get_traces(
    limit=page_size,
    offset=offset
)
```

### Lazy Loading

```python
# Load data only when tab is selected
tab1, tab2, tab3 = st.tabs(["Traces", "Experiments", "Datasets"])

with tab1:
    if st.session_state.get("load_traces", False):
        traces = load_traces(tenant_id)
        render_traces(traces)
```

---

## Contributing

```bash
# Create feature branch
git checkout -b feature/dashboard-improvement

# Make changes
# ...

# Run tests
uv run pytest tests/dashboard/ -v

# Submit PR
```

---

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## Related Packages

- **cogniverse-core**: Multi-agent orchestration (depends on this)
- **cogniverse-evaluation**: Experiment tracking (depends on this)
- **cogniverse-telemetry-phoenix**: Phoenix provider (depends on this)
- **cogniverse-runtime**: FastAPI server (companion application)
