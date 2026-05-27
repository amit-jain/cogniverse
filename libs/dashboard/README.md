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

View and analyze OpenTelemetry traces from Phoenix. The evaluation tab
renders Phoenix dataset and experiment data directly via GraphQL:

```python
# The dashboard is a Streamlit app; tabs are rendered via render_*() functions.
# Example: launching the dashboard and loading the evaluation tab
from cogniverse_dashboard.tabs import evaluation

# render_evaluation_tab() is called automatically by app.py inside st.tabs
evaluation.get_phoenix_datasets()   # returns list of datasets from Phoenix
```

### 2. Experiment and Evaluation

DSPy experiments and routing evaluation are rendered by dedicated tabs:

```python
from cogniverse_dashboard.tabs import routing_evaluation, evaluation

# Each tab module exposes a top-level render function called by app.py
# routing_evaluation.render_routing_evaluation_tab()
# evaluation tab queries Phoenix and renders results inline
```

### 3. Profile and Optimization

Backend profile metrics and optimization live in their own tabs:

```python
from cogniverse_dashboard.tabs import profile_metrics, optimization

# profile_metrics.render_profile_metrics_tab()
# optimization.render_optimization_tab()
```

### 4. System and Tenant Management

Tenant management and approval queue:

```python
from cogniverse_dashboard.tabs import tenant_management, approval_queue

# tenant_management.render_tenant_management_tab()
# approval_queue.render_approval_queue_tab()
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

### Running the Dashboard

The dashboard is a single Streamlit application (`app.py`) that renders
all functionality through tab modules. There are no importable viewer or
manager classes — the entry point is `streamlit run`:

```bash
# Run with default settings
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py

# Run with custom environment
export PHOENIX_ENDPOINT="http://localhost:6006"
export RUNTIME_API_URL="http://localhost:8000"
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py \
  --server.port 8501
```

---

## Dashboard Tabs

The dashboard is a single-page Streamlit app (`app.py`) with the following
top-level tabs rendered via `st.tabs`:

| Tab | What it shows |
|---|---|
| Analytics | Phoenix trace overview, time series, distributions, heatmaps, outliers, trace explorer |
| Evaluation | Phoenix datasets and experiment results via GraphQL |
| Embedding Atlas | Embedding space visualization |
| Routing Evaluation | Routing agent decision metrics |
| Orchestration Annotation | Annotation UI for orchestration traces |
| Profile Routing Metrics | Backend profile performance statistics |
| Optimization | DSPy optimizer run management |
| Synthetic Data & Optimization | Synthetic data generation and optimizer triggers |
| Approval Queue | Human-in-the-loop review of synthetic examples |
| Ingestion Testing | Trigger and monitor ingestion jobs |
| Interactive Search | Run search queries against the runtime |
| Chat | Multi-turn agent chat interface |
| Configuration | System configuration management |
| Tenant Management | Organization and tenant CRUD |
| Memory | Agent memory state viewer |
| RLM A/B Compare | RLM A/B comparison view |

---

## Configuration

The dashboard reads its configuration from environment variables at
startup. There is no `create_dashboard` factory function — the app is
launched directly via `streamlit run`:

```bash
export PHOENIX_ENDPOINT="http://localhost:6006"
export RUNTIME_API_URL="http://localhost:8000"
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py
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

## Tab Modules

The dashboard has no importable component classes or visualization
objects. All UI is rendered by tab modules in `cogniverse_dashboard.tabs`.
Each module exposes a single `render_*_tab()` function that `app.py`
calls inside an `st.tabs` context.

| Module | Render function | What it shows |
|---|---|---|
| `tabs.evaluation` | `render_evaluation_tab()` | Phoenix datasets and experiments |
| `tabs.routing_evaluation` | `render_routing_evaluation_tab()` | Routing agent metrics |
| `tabs.profile_metrics` | `render_profile_metrics_tab()` | Backend profile performance |
| `tabs.optimization` | `render_enhanced_optimization_tab()` | DSPy optimizer runs |
| `tabs.approval_queue` | `render_approval_queue_tab()` | Synthetic data review queue |
| `tabs.tenant_management` | `render_tenant_management_tab()` | Tenant and org CRUD |
| `tabs.memory_management` | `render_memory_management_tab()` | Agent memory state |
| `tabs.config_management` | `render_config_management_tab()` | System configuration |
| `tabs.backend_profile` | `render_backend_profile_tab()` | Vespa profile selection |
| `tabs.embedding_atlas` | `render_embedding_atlas_tab()` | Embedding space visualization |
| `tabs.rlm_ab_compare` | `render_rlm_ab_compare_tab()` | RLM A/B comparison |
| `tabs.orchestration_annotation` | `render_orchestration_annotation_tab()` | Orchestration annotation |

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

All dashboard tabs scope their data to the active tenant set in the
sidebar. The evaluation tab fetches Phoenix datasets and experiments by
querying the configured Phoenix endpoint; analytics fetches traces from
the same Phoenix instance. The active tenant is stored in
`st.session_state["current_tenant"]` and propagated to every tab.

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

The dashboard uses `@st.cache_data` to avoid redundant API calls. The
30-second TTL on agent connectivity checks and the 60-second TTL on
heavier analytics queries keep the UI responsive without hammering Phoenix:

```python
import streamlit as st
from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

analytics = PhoenixAnalytics(telemetry_url="http://localhost:6006")

@st.cache_data(ttl=60)
def load_traces(start_time, end_time):
    return analytics.get_traces(start_time=start_time, end_time=end_time, limit=1000)
```

### Pagination

Streamlit's `st.number_input` and manual offset slicing are used for trace
pagination within the Trace Explorer tab:

```python
page_size = 20
page = st.selectbox("Page", range(1, num_pages + 1))
start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, len(traces_df))
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
