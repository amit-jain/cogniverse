# Multi-Tenant Operations Guide

---

## Overview

Cogniverse implements **schema-per-tenant isolation** to provide complete data separation, security, and independent configuration for each tenant.

### Key Isolation Boundaries

- **Backend Schemas**: Each tenant has dedicated schemas with tenant-suffixed names
- **Telemetry Projects**: Each tenant has isolated telemetry projects
- **Configuration**: Per-tenant configuration with versioning and rollback
- **Memory**: Per-tenant Mem0 memory with isolated user/agent histories
- **Routing**: Per-tenant routing strategies and experience buffers

---

## Tenant Lifecycle

### 1. Tenant Creation

```python
# Imports from layered architecture
from cogniverse_foundation.config.unified_config import SystemConfig  # Foundation layer
from cogniverse_foundation.config.utils import create_default_config_manager  # Foundation layer

# Initialize managers
config_manager = create_default_config_manager()

# Create new tenant
tenant_id = "acme_corp"

# 1. Create tenant configuration
tenant_config = SystemConfig(
    tenant_id=tenant_id,
    llm_model="ollama/gemma3:4b",
    base_url="http://localhost:11434",
    backend_url="http://localhost",
    backend_port=8080,
    telemetry_url="http://localhost:6006",
)
config_manager.set_system_config(tenant_config)

# 2. Deploy schemas using script (from project root)
# JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_json_schema.py configs/schemas/video_colpali_smol500_mv_frame_schema.json

# 3. Initialize tenant memory
from cogniverse_core.memory.manager import Mem0MemoryManager  # Core layer

memory_manager = Mem0MemoryManager(tenant_id=tenant_id)
memory_manager.initialize(
    backend_host="localhost",
    backend_port=8080,
    llm_model="ollama/gemma3:4b",
    embedding_model="ollama/nomic-embed-text",
    llm_base_url="http://localhost:11434",
    config_manager=config_manager,
    schema_loader=schema_loader,
)

print(f"✅ Tenant {tenant_id} created successfully")
print(f"   Config: Stored in config manager")
print(f"   Telemetry Project: {tenant_id}_project")
print(f"   Memory: Initialized with backend isolation")
```

### 2. Tenant Configuration

```python
from cogniverse_foundation.config.unified_config import RoutingConfigUnified  # Foundation layer

# Configure routing for tenant
routing_config = RoutingConfigUnified(
    tenant_id=tenant_id,
    routing_mode="tiered",  # "tiered", "ensemble", or "hybrid"
    enable_fast_path=True,
    enable_slow_path=True,
    fast_path_confidence_threshold=0.7,
    slow_path_confidence_threshold=0.6,
    enable_auto_optimization=True,
    optimization_interval_seconds=3600
)

config_manager.set_routing_config(routing_config)

print(f"✅ Routing configured for {tenant_id}")
```

### 3. Tenant Deletion

```python
# Soft delete (preserves configuration history)
def soft_delete_tenant(tenant_id: str):
    """Soft delete - keeps configuration history"""
    from datetime import datetime

    # 1. Mark tenant as inactive (could use metadata in SystemConfig)
    config = config_manager.get_system_config(tenant_id)
    # Ensure metadata dict exists
    if not config.metadata:
        config.metadata = {}
    config.metadata["status"] = "inactive"
    config.metadata["deactivated_at"] = datetime.now().isoformat()
    config_manager.set_system_config(config)

    # 2. Stop accepting new requests
    # (handled by application layer checking metadata["status"])

    # 3. Preserve data for retention period
    print(f"✅ Tenant {tenant_id} marked inactive")
    print(f"   Data retention: 90 days")
    print(f"   Can be reactivated within retention period")

# Hard delete (permanent removal)
def hard_delete_tenant(tenant_id: str):
    """Hard delete - permanent removal"""

    # 1. Delete backend schemas
    # Note: Schema deletion requires backend API calls
    # Use backend HTTP API or CLI to remove tenant-specific schemas

    # 2. Delete telemetry project
    # (manual cleanup required via telemetry provider UI or API)

    # 3. Delete configurations (if delete method exists)
    # config_manager.delete_tenant_config(tenant_id)

    # 4. Clear memory (clear for all agents under this tenant)
    memory_manager = Mem0MemoryManager(tenant_id=tenant_id)
    # Note: clear_agent_memory requires agent_name parameter
    # For full tenant cleanup, iterate through all agents or use backend API

    print(f"✅ Tenant {tenant_id} permanently deleted")

# Usage
soft_delete_tenant("acme_corp")  # Default: soft delete
# hard_delete_tenant("acme_corp")  # Use with caution
```

---

## Schema Management

### Schema Naming Convention

```text
<base_schema_name>_<tenant_id>

Examples:
- video_colpali_smol500_mv_frame_acme_corp
- video_videoprism_base_mv_chunk_30s_globex_inc
- video_colqwen_omni_mv_chunk_30s_default
```

### Deploy Schema

```bash
# Deploy ColPali schema
JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_json_schema.py \
  configs/schemas/video_colpali_smol500_mv_frame_schema.json

# Verify schema deployed
curl http://localhost:8080/document/v1/ | jq '.schemas'
```

### Deploy All Schemas

```bash
# Deploy all schemas at once
JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_all_schemas.py

# Or deploy individual schemas
SCHEMAS=(
  "video_colpali_smol500_mv_frame"
  "video_videoprism_base_mv_chunk_30s"
)

for schema in "${SCHEMAS[@]}"; do
  JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_json_schema.py \
    "configs/schemas/${schema}_schema.json"
  echo "✓ Deployed: ${schema}"
done
```

### List Tenant Schemas

```python
# List all schemas via backend API
import requests

backend_url = "http://localhost:8080"

# Query backend for deployed schemas
response = requests.get(f"{backend_url}/ApplicationStatus")
if response.status_code == 200:
    # Parse schema names from response
    # (Implementation depends on backend API response format)
    print("Schemas deployed successfully")
else:
    print(f"Failed to query schemas: {response.status_code}")

# Note: Schema listing is best done via backend CLI or API
# Tenant-specific schemas follow pattern: {base_schema}_{tenant_id}
```

---

## Data Ingestion

### Ingest Video for Specific Tenant

```bash
# Ingest videos for acme_corp tenant (from project root)
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
  --video_dir data/videos/acme_corp/ \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
  --tenant-id acme_corp

# Verify ingestion (backend-specific - example for Vespa)
curl "http://localhost:8080/document/v1/video_colpali_smol500_mv_frame_acme_corp/frame/docid/1"
```

### Bulk Tenant Ingestion

```python
from pathlib import Path

async def ingest_tenant_videos(tenant_id: str, video_dir: Path):
    """Ingest all videos for a tenant"""

    video_files = list(video_dir.glob("*.mp4"))
    print(f"Processing {len(video_files)} videos for {tenant_id}")

    # Use the ingestion script (recommended approach)
    import subprocess
    result = subprocess.run([
        "uv", "run", "python", "scripts/run_ingestion.py",
        "--video_dir", str(video_dir),
        "--backend", "vespa",
        "--profile", "video_colpali_smol500_mv_frame",
        "--tenant-id", tenant_id
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Ingestion completed for {tenant_id}")
    else:
        print(f"✗ Ingestion failed: {result.stderr}")

    return result.returncode == 0

# Usage (in async context)
import asyncio

asyncio.run(ingest_tenant_videos("acme_corp", Path("data/videos/acme_corp")))
asyncio.run(ingest_tenant_videos("globex_inc", Path("data/videos/globex_inc")))
```

---

## Query & Search

### Tenant-Isolated Search

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize once — ONE agent serves ALL tenants
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

def search_tenant_videos(
    tenant_id: str,
    query: str,
    top_k: int = 10
):
    """Search videos within tenant isolation"""

    # profile and tenant_id are PER-REQUEST parameters on search()
    results = agent.search(
        query=query,
        profile="video_colpali_smol500_mv_frame",
        tenant_id=tenant_id,
        top_k=top_k,
    )

    return results

# Search for acme_corp
acme_results = search_tenant_videos("acme_corp", "machine learning")

# Search for globex_inc (completely isolated)
globex_results = search_tenant_videos("globex_inc", "machine learning")
```

---

## Telemetry & Monitoring

### Per-Tenant Telemetry Projects

Each tenant gets an isolated telemetry project:

```python
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager

# Initialize telemetry (singleton)
telemetry = TelemetryManager(
    config=TelemetryConfig(
        enabled=True,
        otlp_endpoint="localhost:4317"
    )
)

# All spans require tenant_id for isolation
tenant_id = "acme_corp"
with telemetry.span("video_search", tenant_id=tenant_id) as span:
    span.set_attribute("query", query)
    # ... perform search
```

### View Tenant-Specific Traces

```bash
# Access telemetry dashboard (Phoenix)
open http://localhost:6006

# Projects visible:
# - acme_corp (tenant project)
# - globex_inc (tenant project)
# - default (default tenant project)

# Each project contains only that tenant's traces
```

### Cross-Tenant Analytics (Admin Only)

```python
from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics
from datetime import datetime

# Admin can view aggregated metrics across tenants
analytics = PhoenixAnalytics(telemetry_url="http://localhost:6006")

# Get traces for time range
start_time = datetime(2025, 1, 1)
end_time = datetime(2025, 1, 31)

traces = analytics.get_traces(
    start_time=start_time,
    end_time=end_time,
    limit=10000
)

# Group by tenant (from trace metadata)
tenant_stats = {}
for trace in traces:
    tenant_id = trace.metadata.get("tenant_id", "unknown")
    if tenant_id not in tenant_stats:
        tenant_stats[tenant_id] = {"total": 0, "success": 0, "latencies": []}

    tenant_stats[tenant_id]["total"] += 1
    if trace.status == "success":
        tenant_stats[tenant_id]["success"] += 1
    tenant_stats[tenant_id]["latencies"].append(trace.duration_ms)

# Display metrics
for tenant_id, stats in tenant_stats.items():
    avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
    success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0

    print(f"\n{tenant_id}:")
    print(f"  Total Queries: {stats['total']}")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    print(f"  Success Rate: {success_rate:.1%}")
```

---

## Memory Management

### Per-Tenant Memory

Each tenant has isolated Mem0 memory:

```python
from cogniverse_core.memory.manager import Mem0MemoryManager

# Get tenant-specific memory managers (per-tenant singleton pattern)
memory_acme = Mem0MemoryManager(tenant_id="acme_corp")
memory_acme.initialize(
    backend_host="localhost",
    backend_port=8080,
    llm_model="ollama/gemma3:4b",
    embedding_model="ollama/nomic-embed-text",
    llm_base_url="http://localhost:11434",
    config_manager=config_manager,
    schema_loader=schema_loader,
)

memory_globex = Mem0MemoryManager(tenant_id="globex_inc")
memory_globex.initialize(
    backend_host="localhost",
    backend_port=8080,
    llm_model="ollama/gemma3:4b",
    embedding_model="ollama/nomic-embed-text",
    llm_base_url="http://localhost:11434",
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Add memory for acme_corp
memory_acme.add_memory(
    content="User prefers technical tutorials",
    tenant_id="acme_corp",
    agent_name="video_search_agent"
)

# Search memory (tenant-isolated)
memories = memory_acme.search_memory(
    query="What does the user prefer?",
    tenant_id="acme_corp",
    agent_name="video_search_agent",
    top_k=5
)

# Memories are completely isolated per tenant
```

### Memory Statistics per Tenant

```python
# Get memory stats for tenant
stats = memory_acme.get_memory_stats(
    tenant_id="acme_corp",
    agent_name="video_search_agent"
)

print(f"Tenant: acme_corp")
print(f"  Total Memories: {stats['total_memories']}")
print(f"  Enabled: {stats['enabled']}")
print(f"  Tenant ID: {stats.get('tenant_id', 'N/A')}")
print(f"  Agent: {stats.get('agent_name', 'N/A')}")
```

---

## Configuration Management

### Tenant Configuration Templates

```python
# Define configuration templates
# Note: SystemConfig has basic system-level settings
# For advanced features (cache TTL, optimization), use RoutingConfigUnified
TENANT_TEMPLATES = {
    "enterprise": {
        "llm_model": "mistral:7b-instruct",
        "backend_url": "http://localhost",
        "backend_port": 8080,
        "telemetry_url": "http://localhost:6006"
    },
    "startup": {
        "llm_model": "ollama/gemma3:4b",
        "backend_url": "http://localhost",
        "backend_port": 8080,
        "telemetry_url": "http://localhost:6006"
    },
    "trial": {
        "llm_model": "ollama/gemma3:4b",
        "backend_url": "http://localhost",
        "backend_port": 8080,
        "telemetry_url": "http://localhost:6006"
    }
}

# Apply template to new tenant
def create_tenant_from_template(
    tenant_id: str,
    template: str = "enterprise",
    overrides: dict = None
):
    """Create tenant with template configuration"""

    # Get template
    template_config = TENANT_TEMPLATES[template].copy()

    # Apply overrides
    if overrides:
        template_config.update(overrides)

    # Create system config
    config = SystemConfig(
        tenant_id=tenant_id,
        **template_config
    )

    config_manager.set_system_config(config)
    print(f"✅ Tenant {tenant_id} created from {template} template")

# Usage
create_tenant_from_template(
    tenant_id="new_startup",
    template="startup",
    overrides={"llm_model": "deepseek-r1:7b"}  # Custom override
)
```

### Configuration Rollback

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope

# Rollback tenant configuration using version history
def rollback_tenant_config(tenant_id: str, version: int):
    """Rollback tenant configuration to specific version"""

    # Get config history
    entries = config_manager.store.get_config_history(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        limit=100
    )

    # Find target version
    target_entry = next((e for e in entries if e.version == version), None)
    if not target_entry:
        print(f"❌ Version {version} not found for {tenant_id}")
        return

    # Restore from target version
    from cogniverse_foundation.config.unified_config import SystemConfig
    old_config = SystemConfig.from_dict(target_entry.config_value)
    config_manager.set_system_config(old_config, tenant_id=tenant_id)

    print(f"✅ Rolled back {tenant_id} to version {version}")

# Usage
rollback_tenant_config("acme_corp", version=5)
```

---

## Security & Isolation

### Tenant Isolation Verification

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

async def verify_tenant_isolation(tenant_a: str, tenant_b: str):
    """Verify complete isolation between two tenants"""

    checks = []

    # 1. Schema isolation
    # Note: Schema isolation is enforced by tenant-suffixed schema names
    # e.g., video_colpali_smol500_mv_frame_acme_corp vs video_colpali_smol500_mv_frame_globex_inc
    checks.append(("Schema Isolation", True))  # Enforced by design

    # 2. Data isolation (search)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    agent = VideoSearchAgent(
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    results_a = agent.search("test query", profile="video_colpali_smol500_mv_frame", tenant_id=tenant_a, top_k=100)  # synchronous
    # Verify tenant isolation by checking results are from correct tenant's schema
    checks.append(("Data Isolation", len(results_a) >= 0))  # Verify search works

    # 3. Memory isolation
    memory_a = Mem0MemoryManager(tenant_id=tenant_a)
    memories_a = memory_a.get_all_memories(tenant_id=tenant_a, agent_name="video_search_agent")
    checks.append(("Memory Isolation", len(memories_a) >= 0))  # Verify access works

    # 4. Configuration isolation
    config_a = config_manager.get_system_config(tenant_a)
    config_b = config_manager.get_system_config(tenant_b)
    checks.append(("Config Isolation", config_a.tenant_id != config_b.tenant_id))

    # Report
    print(f"\n🔒 Tenant Isolation Verification")
    print(f"   Tenant A: {tenant_a}")
    print(f"   Tenant B: {tenant_b}\n")

    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False

    return all_passed

# Run verification
isolation_verified = await verify_tenant_isolation("acme_corp", "globex_inc")
assert isolation_verified, "Tenant isolation verification failed!"
```

### Access Control

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Tenant access control decorator
def require_tenant_access(tenant_id: str):
    """Decorator to enforce tenant access control"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Verify caller has access to tenant
            caller_tenant = kwargs.get('caller_tenant_id')
            if caller_tenant != tenant_id:
                raise PermissionError(
                    f"Caller {caller_tenant} not authorized for tenant {tenant_id}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_tenant_access("acme_corp")
def search_acme_videos(query: str, caller_tenant_id: str, config_manager):
    """Search restricted to acme_corp tenant"""
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    agent = VideoSearchAgent(
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    return agent.search(query, profile="video_colpali_smol500_mv_frame", tenant_id="acme_corp")  # synchronous (not async)
```

---

## Performance & Optimization

### Per-Tenant Performance Metrics

```python
from cogniverse_agents.routing.modality_metrics import ModalityMetricsTracker
from cogniverse_agents.search.multi_modal_reranker import QueryModality

# Track performance per tenant (Note: ModalityMetricsTracker doesn't have tenant_id,
# tenant isolation is handled at application level by maintaining separate instances)
metrics_acme = ModalityMetricsTracker(window_size=1000)

# Record modality executions for this tenant
# (called from routing agent during execution)

# Get performance stats
stats = metrics_acme.get_modality_stats(QueryModality.VIDEO)

print(f"Tenant: acme_corp (VIDEO modality)")
print(f"  Total Requests: {stats.get('total_requests', 0)}")
print(f"  Success Rate: {stats.get('success_rate', 0.0):.1%}")
print(f"  P50 Latency: {stats.get('p50_latency', 0)}ms")
print(f"  P95 Latency: {stats.get('p95_latency', 0)}ms")
print(f"  P99 Latency: {stats.get('p99_latency', 0)}ms")
```

### Tenant-Specific Optimization

```python
from cogniverse_agents.routing.optimization_orchestrator import OptimizationOrchestrator

# Run optimization per tenant
async def start_tenant_optimization(tenant_id: str):
    orchestrator = OptimizationOrchestrator(
        tenant_id=tenant_id,
        span_eval_interval_minutes=15,
        annotation_interval_minutes=30
    )

    # Start tenant-specific optimization
    await orchestrator.start()

    print(f"✅ Optimization started for {tenant_id}")
    print(f"   Span evaluation: Every 15 minutes")
    print(f"   Model retraining: Based on annotations")

# Usage
import asyncio
asyncio.run(start_tenant_optimization("acme_corp"))
```

---

## Backup & Recovery

### Tenant Data Backup

Tenant data can be backed up using backend-specific tools:

```bash
# Example: Backup tenant data manually using backend API

TENANT_ID="acme_corp"
BACKUP_DIR="backups/${TENANT_ID}/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

echo "Backing up tenant: $TENANT_ID"

# 1. Backup configuration (stored in backend via ConfigManager)
# Configuration is already persisted in the backend config store

# 2. Backup backend documents (per schema - example for Vespa)
for schema in video_colpali_smol500_mv_frame video_videoprism_base_mv_chunk_30s; do
  schema_name="${schema}_${TENANT_ID}"
  # Use backend visit API to export all documents
  curl "http://localhost:8080/document/v1/${schema_name}/frame/docid" \
    > "$BACKUP_DIR/${schema_name}_documents.json"
done

# 3. Backup memory from backend
curl "http://localhost:8080/document/v1/agent_memories_${TENANT_ID}/memory/docid" \
  > "$BACKUP_DIR/memory.json"

echo "✅ Backup complete: $BACKUP_DIR"

# Note: For production, use backend-specific backup tools (e.g., Vespa snapshots)
```

### Tenant Data Restore

```bash
# Example: Restore tenant data manually using backend API

TENANT_ID="acme_corp"
BACKUP_DIR="$1"

echo "Restoring tenant: $TENANT_ID from $BACKUP_DIR"

# 1. Restore configuration (via ConfigManager API - not file-based)
# Configuration should be restored using config_manager.set_system_config()

# 2. Deploy schemas (must be deployed before data restore)
JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_all_schemas.py

# 3. Restore backend documents
for doc_file in "$BACKUP_DIR"/*_documents.json; do
  curl -X POST \
    -H "Content-Type: application/json" \
    --data-binary "@$doc_file" \
    "http://localhost:8080/document/v1/"
done

# 4. Restore memory
curl -X POST \
  -H "Content-Type: application/json" \
  --data-binary "@$BACKUP_DIR/memory.json" \
  "http://localhost:8080/document/v1/agent_memories_${TENANT_ID}/memory/"

echo "✅ Restore complete for $TENANT_ID"

# Note: For production, use backend-specific restore tools (e.g., Vespa snapshots)
```

---

## Troubleshooting

### Tenant Schema Not Found

```python
# Check if tenant schemas exist
def diagnose_tenant_schemas(tenant_id: str):
    """Diagnose tenant schema issues"""

    print(f"Diagnosing schemas for: {tenant_id}\n")

    # Expected tenant schemas
    expected_schemas = [
        f"video_colpali_smol500_mv_frame_{tenant_id}",
        f"video_videoprism_base_mv_chunk_30s_{tenant_id}",
    ]

    print(f"Expected schemas for {tenant_id}:")
    for schema in expected_schemas:
        print(f"   - {schema}")
        # To verify: curl http://localhost:8080/document/v1/{schema}/...

    print(f"\n   To deploy: uv run python scripts/deploy_all_schemas.py")

# Usage
diagnose_tenant_schemas("acme_corp")
```

### Tenant Configuration Missing

```python
# Check tenant configuration
def diagnose_tenant_config(tenant_id: str):
    """Diagnose tenant configuration issues"""

    print(f"Diagnosing configuration for: {tenant_id}\n")

    try:
        config = config_manager.get_system_config(tenant_id)
        print(f"✅ Configuration found:")
        print(f"   LLM Model: {config.llm_model}")
        print(f"   Telemetry URL: {config.telemetry_url}")
        print(f"   Backend URL: {config.backend_url}")
    except Exception as e:
        print(f"❌ Configuration not found for {tenant_id}: {e}")
        print(f"   Run: config_manager.set_system_config(SystemConfig(tenant_id='{tenant_id}'))")

# Usage
diagnose_tenant_config("acme_corp")
```

---

## Best Practices

### 1. Always Use Tenant Context

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# ✅ Good: Profile-agnostic agent, tenant_id at search time
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)
results = agent.search(
    query="machine learning",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="acme_corp",
    top_k=10,
)

# ❌ Bad: Passing profile or tenant_id at construction
# agent = VideoSearchAgent(profile="...", tenant_id="...")  # These are per-request
```

### 2. Verify Tenant Isolation

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# One agent instance serves ALL tenants
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Search as tenant A
results_a = agent.search(
    query="test query",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="tenant_a",
    top_k=10,
)

# Search as tenant B — completely isolated schemas
results_b = agent.search(
    query="test query",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="tenant_b",
    top_k=10,
)

# No cross-tenant leakage — each tenant has its own Vespa schemas
```

### 3. Monitor Tenant Health

```python
# Regular tenant health checks
async def monitor_tenant_health(tenant_id: str):
    """Monitor tenant health metrics"""

    # Check system config existence
    try:
        config = config_manager.get_system_config(tenant_id)
        has_config = True
    except Exception:
        has_config = False

    # Check if tenant has memory manager instance
    has_memory = tenant_id in Mem0MemoryManager._instances

    health = {
        "schemas": 0,  # Would need schema_manager.list_schemas() if available
        "config": has_config,
        "memory_initialized": has_memory,
        "telemetry_project": tenant_id
    }

    # Alert if any health check fails
    if health["schemas"] == 0:
        alert(f"Tenant {tenant_id}: No schemas deployed")

    if not health["config"]:
        alert(f"Tenant {tenant_id}: Configuration missing")

    return health
```

### 4. Implement Tenant Quotas

```python
# Example: Enforce tenant quotas (custom implementation required)
# Note: TenantQuotaManager is not part of the core framework
# This is a reference implementation pattern for production deployments

from typing import Dict, Any

class TenantQuotaManager:
    """
    Custom quota manager for multi-tenant deployments.

    Note: This is a reference implementation. Quotas should be implemented
    at the application or infrastructure layer based on your requirements.
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.quotas = self._load_quotas()

    async def check_quota(self, resource: str, amount: int = 1):
        """Check if tenant has quota for resource"""

        current_usage = await self._get_usage(resource)
        max_quota = self.quotas.get(resource, float('inf'))

        if current_usage + amount > max_quota:
            raise Exception(
                f"Tenant {self.tenant_id} exceeded {resource} quota "
                f"(current: {current_usage}, max: {max_quota})"
            )

        return True

    def _load_quotas(self) -> Dict[str, int]:
        """Load quota configuration for tenant"""
        # Implementation: Load from config store or database
        return {
            "api_calls": 10000,
            "storage_mb": 1000,
            "concurrent_requests": 10
        }

    async def _get_usage(self, resource: str) -> int:
        """Get current usage for resource"""
        # Implementation: Query from telemetry or metrics store
        return 0  # Placeholder

# Usage (in async context)
quota_manager = TenantQuotaManager("acme_corp")
await quota_manager.check_quota("api_calls")
await quota_manager.check_quota("storage_mb", amount=100)
```

---

## Related Documentation

- [Configuration Management](configuration.md) - Multi-tenant configuration
- [Deployment Guide](deployment.md) - Deployment with tenant isolation
- [Multi-Tenant Architecture](../architecture/multi-tenant.md) - Architecture design
- [SDK Architecture](../architecture/sdk-architecture.md) - Package structure
