# Cogniverse Quickstart

Get started with Cogniverse in 5 minutes.

---

## Prerequisites

- Python 3.12+
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
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# NOTE: Backend must be initialized with schemas deployed before searching
# Run: uv run python scripts/deploy_all_schemas.py first

# Initialize configuration
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# Get backend from registry (handles instantiation and caching)
backend = BackendRegistry.get_search_backend(
    name="vespa",
    tenant_id="quickstart",
    config_manager=config_manager,
    schema_loader=schema_loader
)

# Execute a search using query dict format
results = backend.search({
    "query": "machine learning tutorial",
    "type": "video",
    "top_k": 10,
    "profile": "video_colpali_smol500_mv_frame"
})

# Results are SearchResult objects with document and score
for result in results:
    doc_id = result.document.id
    score = result.score
    print(f"- {doc_id}: {score:.2f}")
```

```bash
uv run python quick_search.py
```

---

## 4. Process a Video

```python
# ingest_video.py
import asyncio
from pathlib import Path
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

# Initialize dependencies
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# Create pipeline with required parameters
pipeline = VideoIngestionPipeline(
    tenant_id="quickstart",
    config_manager=config_manager,
    schema_loader=schema_loader,
    schema_name="video_colpali_smol500_mv_frame"
)

async def main():
    # Process a single video
    result = await pipeline.process_video_async(Path("my_video.mp4"))

    # Result is a dict with: video_id, video_path, duration, status, results
    print(f"Processed: {result.get('video_id', 'unknown')}")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Duration: {result.get('duration', 0):.1f}s")

    # Access strategy results (populated by processing strategies)
    if result.get('results'):
        for strategy_name, strategy_result in result['results'].items():
            print(f"  {strategy_name}: {strategy_result}")

asyncio.run(main())
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
    "profile": "video_colpali_smol500_mv_frame",
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

```text
cogniverse/
├── libs/                    # 11-package workspace
│   ├── sdk/                 # Pure interfaces
│   ├── foundation/          # Config + telemetry
│   ├── core/                # Agent base, orchestration, caching
│   ├── agents/              # Agent implementations
│   ├── vespa/               # Vespa backend
│   ├── evaluation/          # Metrics, experiments
│   ├── finetuning/          # LLM fine-tuning (SFT, DPO)
│   ├── telemetry-phoenix/   # Phoenix telemetry provider
│   ├── synthetic/           # Training data generation
│   ├── runtime/             # FastAPI server
│   └── dashboard/           # Streamlit UI
├── configs/                 # Configuration files
│   ├── config.json          # Main config
│   └── schemas/             # Schema definitions
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
uv run python scripts/run_ingestion.py --video_dir data/videos --backend vespa --profile video_colpali_smol500_mv_frame

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

- [Full Documentation](./index.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [GitHub Issues](https://github.com/your-org/cogniverse/issues)
