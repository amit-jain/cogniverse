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
# Deploy the full stack (Vespa, Phoenix, Ollama, Runtime, Dashboard) via k3d
cogniverse up

# Wait for services to be ready (~30 seconds). k3d's loadbalancer publishes
# each service's NodePort on localhost — these differ from the in-cluster
# ports (e.g. Runtime listens on 8000 inside the pod, but is reachable at
# 28000 on the host).
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:26006/health            # Phoenix (NodePort 26006)

# Backend connection env vars for any host-side script or CLI that talks to
# Vespa directly (create_default_config_manager() requires these)
export BACKEND_URL=http://localhost
export BACKEND_PORT=8080
```

---

## 3. Create a Tenant & Deploy Its Schemas

Schema deployment is always per-tenant in cogniverse — there is no
"default" tenant. Create a tenant first; the runtime provisions the
tenant's metadata schemas. Then deploy the profile's content schema
for that tenant.

```bash
RUNTIME_URL=http://localhost:28000

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

> **Note:** The tenant management API is available on the main Runtime (NodePort 28000, in-cluster port 8000) at `/admin/tenants`, or as a standalone service on port 9000 if run separately (`python -m cogniverse_runtime.admin.tenant_manager`, not deployed by `cogniverse up`).

---

## 4. Run Your First Search

```python
# quick_search.py
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Requires: schemas deployed and tenant created (see step 3), and
# BACKEND_URL / BACKEND_PORT exported (see step 2)

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
    print(f"- {result.get('id', 'unknown')}: {result.get('score', 0):.2f}")
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
  --tenant-id quickstart \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame

# Ingest with multiple profiles for richer retrieval
uv run python scripts/run_ingestion.py \
  --tenant-id quickstart \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
           video_videoprism_base_mv_chunk_30s

# Test mode — limited frames for faster iteration
uv run python scripts/run_ingestion.py \
  --tenant-id quickstart \
  --video_dir data/testset/evaluation/sample_videos \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
  --test-mode --max-frames 1
```

---

## 6. Query the Runtime API

`cogniverse up` (step 2) already deployed the Runtime FastAPI server — it's
reachable on the host at NodePort 28000 (in-cluster port 8000).

```bash
# Test the health endpoint
curl http://localhost:28000/health

# Execute a search via API
curl -X POST http://localhost:28000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "profile": "video_colpali_smol500_mv_frame",
    "top_k": 10,
    "tenant_id": "quickstart"
  }'
```

For local development without k3d (hot-reload on code changes), run the
server directly against the same Vespa instance instead:

```bash
uv run uvicorn cogniverse_runtime.main:app --reload --port 8000
curl http://localhost:8000/health
```

---

## 7. View in Dashboard

`cogniverse up` (step 2) also deployed the Streamlit dashboard — it's
reachable on the host at NodePort 28501.

```bash
open http://localhost:28501
```

For local development without k3d, run the dashboard directly instead:

```bash
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py
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
curl -X POST http://localhost:28000/admin/messaging/invite \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "quickstart", "expires_in_hours": 24}'

# Returns: {"token": "abc123...", "tenant_id": "quickstart"}
# User sends: /start abc123... to the bot to register
```

---

## Optional: Wiki Knowledge Base

The wiki knowledge base automatically saves substantial agent interactions as searchable pages in Vespa. It requires no configuration — the `wiki_pages` schema is deployed automatically, per-tenant, the first time a tenant saves or searches the wiki (no manual schema step needed).

```bash
# Save a session manually
curl -X POST http://localhost:28000/wiki/save \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does ColPali work?",
    "response": {"answer": "ColPali uses patch-level embeddings..."},
    "entities": ["ColPali", "patch_embeddings"],
    "agent_name": "summarizer_agent",
    "tenant_id": "quickstart"
  }'

# Search the wiki
curl -X POST http://localhost:28000/wiki/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ColPali embeddings", "tenant_id": "quickstart", "top_k": 5}'
```

Via Telegram (after connecting the messaging gateway):
```text
/wiki save            — Save the current session to the wiki
/wiki search ColPali  — Search the wiki knowledge base
/wiki topic ColPali   — Look up a topic page by name
/wiki index           — Show the wiki index
/wiki lint            — Check wiki for orphan, stale, or empty pages
```

Auto-filing triggers automatically when an interaction has 3+ extracted entities, comes from `detailed_report_agent` or `deep_research_agent`, or spans 4+ conversation turns.

---

## Optional: Enable Quality Monitor

The quality monitor runs as a sidecar, continuously evaluating all agents and triggering optimization when quality degrades:

```bash
# Run directly
python -m cogniverse_runtime.quality_monitor_cli \
  --tenant-id quickstart \
  --runtime-url http://localhost:28000 \
  --phoenix-url http://localhost:26006 \
  --llm-model qwen3:4b

# Enabled by default in Helm (runtime.qualityMonitor.enabled: true)
```

---

## What's Next?

| Goal | Documentation |
|------|---------------|
| Understand the architecture | [Architecture Overview](./architecture/overview.md) |
| Use the interactive coding agent | [Coding Agent CLI](./user/coding-agent-cli.md) |
| Use the Knowledge Management Layer | [User Guide — Knowledge Management](./USER_GUIDE.md#knowledge-management) |
| Extract a knowledge graph from code / docs | [Knowledge Graph](./user/knowledge-graph.md) |
| Create a custom agent | [Creating Agents Tutorial](./tutorials/creating-agents.md) |
| Learn key concepts | [Glossary](./GLOSSARY.md) |
| Deep dive into modules | [Core Module](./modules/core.md), [Foundation](./modules/foundation.md), [Agents](./modules/agents.md), [Runtime](./modules/runtime.md), [Vespa Backend](./modules/backends.md) |
| Configure profiles | [Configuration System](./CONFIGURATION_SYSTEM.md) |
| Run tests | [Testing Guide](./testing/TESTING_GUIDE.md) |

---

## Project Structure

```text
cogniverse/
├── libs/                    # 13-package workspace
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
│   ├── cli/                 # cogniverse CLI (deploy, manage)
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
make lint-all

# Format code
uv run ruff format .

# Run ingestion
uv run python scripts/run_ingestion.py --tenant-id quickstart --video_dir data/videos --backend vespa --profile video_colpali_smol500_mv_frame

# View Phoenix traces
open http://localhost:26006
```

---

## Troubleshooting

**Vespa won't start:**
```bash
# cogniverse deploys to a k3d/Kubernetes cluster, not raw docker containers
# -- Vespa runs as the cogniverse-vespa StatefulSet
cogniverse status
cogniverse logs vespa -f
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
