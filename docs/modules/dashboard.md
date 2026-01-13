# Dashboard Module

**Package:** `cogniverse_dashboard`
**Location:** `libs/dashboard/cogniverse_dashboard/`
**Purpose:** Streamlit-based analytics UI with Phoenix integration and system monitoring
**Last Updated:** 2026-01-13

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Main Application](#main-application)
4. [Dashboard Tabs](#dashboard-tabs)
   - [Analytics Tab](#analytics-tab)
   - [Evaluation Tab](#evaluation-tab)
   - [Embedding Atlas Tab](#embedding-atlas-tab)
   - [Routing Evaluation Tab](#routing-evaluation-tab)
   - [Interactive Search Tab](#interactive-search-tab)
   - [Orchestration Annotation Tab](#orchestration-annotation-tab)
   - [Approval Queue Tab](#approval-queue-tab)
   - [Configuration Management Tab](#configuration-management-tab)
   - [Memory Management Tab](#memory-management-tab)
5. [Phoenix Data Manager](#phoenix-data-manager)
6. [Configuration](#configuration)
7. [Deployment](#deployment)
8. [Testing](#testing)

---

## Overview

The Dashboard module provides a **Streamlit-based UI** for:

- **Phoenix Analytics**: Visualize OpenTelemetry traces, spans, and experiments
- **Evaluation Management**: Track experiments, compare results, view metrics
- **Embedding Visualization**: Atlas visualization of embeddings and clusters
- **Interactive Search**: Live search testing with per-result annotation
- **HITL Workflows**: Human approval queues and orchestration annotation
- **System Monitoring**: Configuration, memory, and ingestion management

The dashboard integrates with agents via the **A2A protocol** for real-time interaction.

---

## Package Structure

```
cogniverse_dashboard/
├── app.py                           # Main Streamlit application
├── __init__.py                      # Package exports
└── utils/
    ├── phoenix_launcher.py          # Phoenix server launcher
    └── phoenix_data_manager.py      # Data backup/restore utilities

scripts/                             # Tab implementations (external)
├── phoenix_dashboard_evaluation_tab_tabbed.py  # Evaluation tab
├── embedding_atlas_tab.py           # Embedding visualization
├── routing_evaluation_tab.py        # Routing metrics
├── interactive_search_tab.py        # Search testing
├── orchestration_annotation_tab.py  # Workflow annotation
├── approval_queue_tab.py            # Human approval queue
├── config_management_tab.py         # Config editor
├── memory_management_tab.py         # Memory inspection
├── backend_profile_tab.py           # Profile management
├── ingestion_testing_tab.py         # Video ingestion testing
├── multi_modal_chat_tab.py          # Multi-modal chat UI
└── enhanced_optimization_tab.py     # DSPy optimization UI
```

---

## Main Application

The main application (`app.py`) orchestrates all dashboard tabs:

```python
# Start the dashboard
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py

# With custom port
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --server.port 8501
```

**Key Components:**

1. **Analytics Session State**: Maintains Phoenix analytics instance across interactions
2. **Tab Loading**: Dynamically imports tab modules with graceful fallback
3. **A2A Client**: Communicates with agents for real-time operations
4. **Async Wrapper**: Handles async operations in Streamlit's sync context

```python
# Async helper for Streamlit
def run_async_in_streamlit(coro):
    """Run async operations in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)
```

**Session State Management:**

```python
import streamlit as st

# Initialize analytics session
if 'analytics' not in st.session_state:
    st.session_state.analytics = PhoenixAnalytics()

# Track refresh timing
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Auto-refresh toggle
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
```

---

## Dashboard Tabs

### Analytics Tab

**Purpose:** Phoenix trace visualization and performance analysis

**Features:**
- View all traces for tenant/project
- Filter by time range, status, operation
- Drill down into individual spans
- View span attributes and events
- Export traces to CSV/JSON

```python
from cogniverse_dashboard.pages import analytics

# Render analytics view
analytics.render_traces_view(
    tenant_id="acme",
    project="acme_project",
    time_range="last_24h"
)
```

### Evaluation Tab

**File:** `scripts/phoenix_dashboard_evaluation_tab_tabbed.py`

**Purpose:** Experiment tracking and comparison

**Features:**
- List all experiments with filters
- Compare experiment results side-by-side
- View optimizer settings and hyperparameters
- Visualize metric trends over time
- Export experiment data

```python
from phoenix_dashboard_evaluation_tab_tabbed import render_evaluation_tab

# Render in dashboard
render_evaluation_tab(st.session_state.analytics)
```

### Embedding Atlas Tab

**File:** `scripts/embedding_atlas_tab.py`

**Purpose:** Embedding space visualization

**Features:**
- UMAP/t-SNE dimensionality reduction
- Interactive embedding plots
- Cluster visualization
- Similarity search from embeddings
- Per-profile embedding analysis

```python
from embedding_atlas_tab import render_embedding_atlas_tab

# Render embedding visualization
render_embedding_atlas_tab()
```

### Routing Evaluation Tab

**File:** `scripts/routing_evaluation_tab.py`

**Purpose:** Modality routing performance analysis

**Features:**
- Routing accuracy by modality
- Confidence distribution analysis
- Misrouting pattern identification
- Per-tier performance breakdown
- Routing latency metrics

```python
from routing_evaluation_tab import render_routing_evaluation_tab

# Render routing metrics
render_routing_evaluation_tab()
```

### Interactive Search Tab

**File:** `scripts/interactive_search_tab.py`

**Purpose:** Live search testing with relevance annotation

**Features:**
- Execute search queries in real-time
- View results with thumbnails
- Annotate relevance per result
- Session-aware search tracking
- Export search sessions for evaluation

```python
from interactive_search_tab import render_interactive_search_tab

# Enable session tracking for multi-turn evaluation
render_interactive_search_tab(
    enable_session_tracking=True,
    session_id="evaluation-session-001"
)
```

### Orchestration Annotation Tab

**File:** `scripts/orchestration_annotation_tab.py`

**Purpose:** Workflow quality scoring and annotation

**Features:**
- View multi-agent orchestration traces
- Score workflow quality
- Annotate agent decisions
- Identify workflow patterns
- Feed annotations to optimizer

```python
from orchestration_annotation_tab import render_orchestration_annotation_tab

render_orchestration_annotation_tab()
```

### Approval Queue Tab

**File:** `scripts/approval_queue_tab.py`

**Purpose:** Human-in-the-loop approval workflows

**Features:**
- View pending approval requests
- Approve/reject with comments
- Bulk approval operations
- Confidence-based filtering
- Approval history

```python
from approval_queue_tab import render_approval_queue_tab

# Render approval queue for tenant
render_approval_queue_tab(tenant_id="acme")
```

### Configuration Management Tab

**File:** `scripts/config_management_tab.py`

**Purpose:** System configuration editor

**Features:**
- View all configuration types
- Edit configuration values
- Profile management
- Configuration history
- Export/import configs

```python
from config_management_tab import render_config_management_tab

render_config_management_tab()
```

### Memory Management Tab

**File:** `scripts/memory_management_tab.py`

**Purpose:** Memory system inspection

**Features:**
- View semantic memories
- Memory search
- Memory statistics
- Clear/reset memory
- Memory export

```python
from memory_management_tab import render_memory_management_tab

render_memory_management_tab()
```

---

## Phoenix Data Manager

The `PhoenixDataManager` class provides utilities for managing Phoenix persistent data:

```python
from cogniverse_dashboard.utils.phoenix_data_manager import PhoenixDataManager

manager = PhoenixDataManager(data_dir="./data/phoenix")

# Create backup
backup_path = manager.backup(name="before_experiment")

# Restore from backup
manager.restore("before_experiment", force=True)

# List available backups
backups = manager.list_backups()
for backup in backups:
    print(f"{backup['name']}: {backup['size_mb']:.2f} MB")

# Clean old data
manager.clean(older_than_days=30, dry_run=True)

# Analyze data directory
analysis = manager.analyze()
print(f"Total size: {analysis['total_size_mb']:.2f} MB")
print(f"Traces: {analysis['traces']['count']} files")

# Export/import datasets
manager.export_datasets("./exports/datasets")
manager.import_datasets("./imports/datasets")
```

**CLI Usage:**

```bash
# Backup
uv run python -m cogniverse_dashboard.utils.phoenix_data_manager backup --name my_backup

# Restore
uv run python -m cogniverse_dashboard.utils.phoenix_data_manager restore my_backup --force

# List backups
uv run python -m cogniverse_dashboard.utils.phoenix_data_manager list

# Clean old data
uv run python -m cogniverse_dashboard.utils.phoenix_data_manager clean --older-than 30

# Analyze
uv run python -m cogniverse_dashboard.utils.phoenix_data_manager analyze
```

---

## Configuration

### Environment Variables

```bash
# Required
export TENANT_ID="acme"
export PHOENIX_ENDPOINT="http://localhost:6006"

# Optional
export VESPA_URL="http://localhost:8080"
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
```

### Streamlit Config

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true
runOnSave = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B9D"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### Caching for Performance

```python
import streamlit as st

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_traces(tenant_id: str, project: str):
    """Load traces with caching."""
    return phoenix_provider.get_traces(
        project=project,
        limit=1000
    )

# Use cached function
traces = load_traces(tenant_id, f"{tenant_id}_project")
```

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │          cogniverse-dashboard ◄─── YOU ARE HERE             ││
│  │  Streamlit UI, Phoenix analytics, HITL workflows            ││
│  └─────────────────────────────────────────────────────────────┘│
│                     cogniverse-runtime                           │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                       Core Layer                                 │
│  cogniverse-core │ cogniverse-evaluation │ cogniverse-telemetry │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation Layer                              │
│           cogniverse-foundation │ cogniverse-sdk                │
└─────────────────────────────────────────────────────────────────┘
```

**Dependencies:**
- `cogniverse-core`: Memory, orchestration
- `cogniverse-evaluation`: Experiment tracking, metrics
- `cogniverse-telemetry-phoenix`: Phoenix provider for spans/traces
- `cogniverse-foundation`: Configuration and telemetry

**External Dependencies:**
- `streamlit>=1.28.0`: Web UI framework
- `plotly>=5.17.0`: Interactive charts
- `pandas>=2.1.0`: Data manipulation

---

## Deployment

### Development

```bash
# Install dependencies
uv sync

# Run with auto-reload
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --server.runOnSave true

# Access dashboard
open http://localhost:8501
```

### Production

```bash
# Run in production mode
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
```

### Docker

```dockerfile
FROM python:3.11-slim

RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", \
     "libs/dashboard/cogniverse_dashboard/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
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
      - TENANT_ID=acme
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
```

---

## Testing

```bash
# Run dashboard tests
uv run pytest tests/dashboard/ -v

# Run specific tab tests
uv run pytest tests/dashboard/unit/test_evaluation_tab.py -v

# Test with coverage
uv run pytest tests/dashboard/ --cov=cogniverse_dashboard --cov-report=html
```

**Test Categories:**
- `tests/dashboard/unit/` - Unit tests for individual tabs
- `tests/dashboard/integration/` - Integration tests with Phoenix

---

## Related Documentation

- [Core Module](./core.md) - Memory and orchestration
- [Foundation Module](./foundation.md) - Configuration and telemetry
- [Evaluation Module](./evaluation.md) - Experiment tracking
- [Telemetry Module](./telemetry.md) - Phoenix provider
- [Runtime Module](./runtime.md) - FastAPI companion application

---

**Summary:** The Dashboard module provides a comprehensive Streamlit UI for Cogniverse. It includes tabs for Phoenix analytics, evaluation management, embedding visualization, interactive search, and HITL workflows. The `PhoenixDataManager` provides utilities for data backup/restore. All tabs integrate with the A2A protocol for real-time agent communication.
