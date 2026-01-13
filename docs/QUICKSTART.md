# Cogniverse Quickstart

Get started with Cogniverse in 5 minutes.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Vespa and Phoenix)

---

## 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/cogniverse.git
cd cogniverse

# Install dependencies with uv
uv sync
```

---

## 2. Start Services

```bash
# Start Vespa (vector database) and Phoenix (telemetry)
docker-compose up -d vespa phoenix

# Wait for services to be ready (~30 seconds)
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:6006/                   # Phoenix
```

---

## 3. Run Your First Search

```python
# quick_search.py
from cogniverse_vespa import VespaBackend
from cogniverse_foundation.config.utils import create_default_config_manager, get_config

# Initialize configuration
config_manager = create_default_config_manager()
config = get_config(tenant_id="quickstart", config_manager=config_manager)

# Create backend and search
backend = VespaBackend(config)

# Execute a search
results = backend.search(
    query="machine learning tutorial",
    profile="video_colpali_mv_frame",
    top_k=10
)

for result in results:
    print(f"- {result.title}: {result.score:.2f}")
```

```bash
uv run python quick_search.py
```

---

## 4. Process a Video

```python
# ingest_video.py
from pathlib import Path
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()

pipeline = VideoIngestionPipeline(
    tenant_id="quickstart",
    config_manager=config_manager,
    schema_name="video_colpali_mv_frame"
)

# Process a single video
result = await pipeline.process_video_async(Path("my_video.mp4"))
print(f"Processed: {result['video_id']}, frames: {result['results']['keyframes']['count']}")
```

```bash
uv run python ingest_video.py
```

---

## 5. Start the Runtime Server

```bash
# Start FastAPI server
uv run uvicorn cogniverse_runtime.main:app --reload --port 8000

# Test the health endpoint
curl http://localhost:8000/health

# Execute a search via API
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "profile": "video_colpali_mv_frame",
    "top_k": 10,
    "tenant_id": "quickstart"
  }'
```

---

## 6. View in Dashboard

```bash
# Start Streamlit dashboard
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py

# Open in browser
open http://localhost:8501
```

---

## What's Next?

| Goal | Documentation |
|------|---------------|
| Understand the architecture | [Architecture Overview](./architecture/overview.md) |
| Create a custom agent | [Creating Agents Tutorial](./tutorials/creating-agents.md) |
| Learn key concepts | [Glossary](./GLOSSARY.md) |
| Deep dive into modules | [Core Module](./modules/core.md), [Foundation](./modules/foundation.md) |
| Configure profiles | [Configuration System](./CONFIGURATION_SYSTEM.md) |
| Run tests | [Testing Guide](./testing/TESTING_GUIDE.md) |

---

## Project Structure

```
cogniverse/
├── libs/                    # 11 packages in layered architecture
│   ├── sdk/                 # Pure interfaces
│   ├── foundation/          # Config + telemetry
│   ├── core/                # Agent base, orchestration
│   ├── agents/              # Agent implementations
│   ├── vespa/               # Vespa backend
│   ├── evaluation/          # Metrics, experiments
│   ├── telemetry-phoenix/   # Phoenix provider
│   ├── synthetic/           # Training data generation
│   ├── runtime/             # FastAPI server
│   └── dashboard/           # Streamlit UI
├── configs/                 # Configuration files
│   ├── config.yml           # Main config
│   └── schemas/             # Vespa schemas
├── scripts/                 # Utility scripts
├── tests/                   # Test suites
└── docs/                    # Documentation
```

---

## Common Commands

```bash
# Run tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# Lint code
uv run make lint-all

# Format code
uv run ruff format .

# Run ingestion
uv run python scripts/run_ingestion.py --video_dir data/videos --backend vespa

# View Phoenix traces
open http://localhost:6006
```

---

## Troubleshooting

**Vespa won't start:**
```bash
docker logs vespa
# Check port 8080 is free
lsof -i :8080
```

**Import errors:**
```bash
# Ensure all packages are installed
uv sync
# Check PYTHONPATH
python -c "import cogniverse_core; print('OK')"
```

**Search returns no results:**
- Check tenant_id matches ingested data
- Verify schema exists: `curl http://localhost:8080/document/v1/`
- Ensure embeddings were generated during ingestion

---

## Getting Help

- [Full Documentation](./README.md)
- [Study Guide](./plan/STUDY_GUIDE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [GitHub Issues](https://github.com/your-org/cogniverse/issues)
