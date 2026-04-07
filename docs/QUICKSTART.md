# Cogniverse Quickstart

Get started with Cogniverse in 5 minutes.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Vespa and Phoenix)
- [Deno](https://deno.land/) 2.0+ (required for RLM sandboxed code execution: `curl -fsSL https://deno.land/install.sh | sh`)

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
# Start all services (Vespa, Phoenix, Ollama) via k3d
cogniverse up

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

# Get shared search backend from registry (handles instantiation and caching)
backend = BackendRegistry.get_search_backend(
    name="vespa",
    config_manager=config_manager,
    schema_loader=schema_loader
)

# Execute a search — tenant_id is required in query_dict
results = backend.search({
    "query": "machine learning tutorial",
    "type": "video",
    "top_k": 10,
    "profile": "video_colpali_smol500_mv_frame",
    "tenant_id": "quickstart",
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

## Optional: Enable Messaging Gateway

Allow users to interact with Cogniverse via Telegram:

```bash
# Start with Telegram bot enabled
cogniverse up --messaging

# Or set in Helm values (production):
# messaging.enabled: true
# messaging.mode: webhook  # polling for dev
```

```bash
# Generate an invite token for a user
curl -X POST http://localhost:8000/admin/messaging/invite \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default", "expires_in_hours": 24}'

# Returns: {"token": "abc123...", "tenant_id": "default"}
# User sends: /start abc123... to the bot to register
```

---

## Optional: Wiki Knowledge Base

The wiki knowledge base automatically saves substantial agent interactions as searchable pages in Vespa. It requires no configuration — the `wiki_pages` schema is deployed on startup.

```bash
# Save a session manually
curl -X POST http://localhost:8000/wiki/save \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does ColPali work?",
    "response": {"answer": "ColPali uses patch-level embeddings..."},
    "entities": ["ColPali", "patch_embeddings"],
    "agent_name": "routing_agent",
    "tenant_id": "default"
  }'

# Search the wiki
curl -X POST http://localhost:8000/wiki/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ColPali embeddings", "tenant_id": "default", "top_k": 5}'
```

Via Telegram (after connecting the messaging gateway):
```text
/wiki search ColPali
/wiki topic ColPali
/wiki index
```

Auto-filing triggers automatically when an interaction has 3+ extracted entities, comes from `detailed_report_agent` or `deep_research_agent`, or spans 4+ conversation turns.

---

## Optional: Enable Quality Monitor

The quality monitor runs as a sidecar, continuously evaluating all agents and triggering optimization when quality degrades:

```bash
# Run directly
python -m cogniverse_runtime.quality_monitor_cli \
  --tenant-id default \
  --runtime-url http://localhost:8000 \
  --phoenix-url http://localhost:6006

# Enabled by default in Helm (runtime.qualityMonitor.enabled: true)
```

---

## What's Next?

| Goal | Documentation |
|------|---------------|
| Understand the architecture | [Architecture Overview](./architecture/overview.md) |
| Use the interactive coding agent | [Coding Agent CLI](./user/coding-agent-cli.md) |
| Extract a knowledge graph from code / docs | [Knowledge Graph](./user/knowledge-graph.md) |
| Create a custom agent | [Creating Agents Tutorial](./tutorials/creating-agents.md) |
| Learn key concepts | [Glossary](./GLOSSARY.md) |
| Deep dive into modules | [Core Module](./modules/core.md), [Foundation](./modules/foundation.md) |
| Configure profiles | [Configuration System](./CONFIGURATION_SYSTEM.md) |
| Run tests | [Testing Guide](./testing/TESTING_GUIDE.md) |

---

## Project Structure

```text
cogniverse/
├── libs/                    # 12-package workspace
│   ├── sdk/                 # Pure interfaces
│   ├── foundation/          # Config + telemetry
│   ├── core/                # Agent base, orchestration, caching
│   ├── agents/              # Agent implementations + strategy learner
│   ├── vespa/               # Vespa backend
│   ├── evaluation/          # Metrics, experiments, quality monitor
│   ├── finetuning/          # LLM fine-tuning (SFT, DPO)
│   ├── telemetry-phoenix/   # Phoenix telemetry provider
│   ├── synthetic/           # Training data generation
│   ├── runtime/             # FastAPI server + quality monitor CLI
│   ├── messaging/           # Telegram messaging gateway
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
