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

## 3. Create a Tenant & Deploy Its Schemas

Schema deployment is always per-tenant in cogniverse — there is no
"default" tenant. Create a tenant first; the runtime provisions the
tenant's metadata schemas. Then deploy the profile's content schema
for that tenant.

```bash
RUNTIME_URL=http://localhost:8000

# Create the tenant (auto-provisions org + deploys tenant metadata)
curl -sfX POST "$RUNTIME_URL/admin/tenants" \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "quickstart",
    "created_by": "admin"
  }'

# Deploy the profile's content schema for that tenant
curl -sfX POST "$RUNTIME_URL/admin/profiles/video_colpali_smol500_mv_frame/deploy" \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id": "quickstart", "force": false}'
```

> **Note:** The tenant management API is available on the main Runtime (port 8000) at `/admin/tenants`, or as a standalone service on port 9000 if run separately.

---

## 4. Run Your First Search

```python
# quick_search.py
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Requires: schemas deployed and tenant created (see step 3)

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# Create agent with dependencies
deps = SearchAgentDeps(profile="video_colpali_smol500_mv_frame")
agent = SearchAgent(deps=deps, config_manager=config_manager, schema_loader=schema_loader)

# Search by text — tenant_id is required per-request
results = agent.search_by_text(
    query="machine learning tutorial",
    tenant_id="quickstart",
    top_k=10,
)

for result in results:
    print(f"- {result.get('documentid', 'unknown')}: {result.get('relevance', 0):.2f}")
```

```bash
uv run python quick_search.py
```

---

## 5. Process a Video

The easiest way to ingest videos is via the CLI script:

```bash
# Ingest a single directory of videos
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame

# Ingest with multiple profiles for richer retrieval
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
           video_videoprism_base_mv_chunk_30s

# Test mode — limited frames for faster iteration
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
  --test-mode --max-frames 1
```

---

## 6. Start the Runtime Server

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

## 7. View in Dashboard

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
