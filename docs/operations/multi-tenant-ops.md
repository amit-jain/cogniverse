# Multi-Tenant Operations Guide

**Last Updated:** 2025-10-15
**Architecture:** Schema-per-Tenant Isolation
**Purpose:** Comprehensive guide for multi-tenant deployment, operations, and management

---

## Overview

Cogniverse implements **schema-per-tenant isolation** to provide complete data separation, security, and independent configuration for each tenant.

### Key Isolation Boundaries

- **Vespa Schemas**: Each tenant has dedicated schemas with tenant-suffixed names
- **Phoenix Projects**: Each tenant has isolated telemetry projects
- **Configuration**: Per-tenant configuration with versioning and rollback
- **Memory**: Per-tenant Mem0 memory with isolated user/agent histories
- **Routing**: Per-tenant routing strategies and experience buffers

---

## Tenant Lifecycle

### 1. Tenant Creation

```python
from cogniverse_core.config.manager import get_config_manager
from cogniverse_core.config.unified_config import SystemConfig
from cogniverse_vespa.backends.vespa_schema_manager import VespaSchemaManager

# Initialize managers
config_manager = get_config_manager()
schema_manager = VespaSchemaManager(
    vespa_url="http://localhost:8080",
    vespa_config_port=19071
)

# Create new tenant
tenant_id = "acme_corp"

# 1. Create tenant configuration
tenant_config = SystemConfig(
    tenant_id=tenant_id,
    llm_model="gpt-4",
    llm_base_url="https://api.openai.com/v1",
    vespa_url="http://localhost:8080",
    vespa_config_port=19071,
    phoenix_project_name=f"{tenant_id}_project",
    phoenix_enabled=True
)
config_manager.set_system_config(tenant_config)

# 2. Deploy tenant-specific schemas
schemas = [
    "video_colpali_smol500_mv_frame",
    "video_videoprism_base_mv_chunk_30s"
]

for schema_name in schemas:
    schema_manager.deploy_schema(
        schema_path=f"configs/schemas/{schema_name}.json",
        tenant_id=tenant_id,
        schema_suffix=f"_{tenant_id}"
    )

# 3. Initialize tenant memory
from cogniverse_core.common.memory.mem0_memory_manager import Mem0MemoryManager

memory_manager = Mem0MemoryManager.get_instance(tenant_id)
await memory_manager.initialize(
    config={
        "vector_store": {
            "provider": "vespa",
            "config": {
                "url": "http://localhost:8080",
                "port": 8080,
                "tenant_id": tenant_id
            }
        }
    }
)

print(f"✅ Tenant {tenant_id} created successfully")
print(f"   Schemas: {[f'{s}_{tenant_id}' for s in schemas]}")
print(f"   Phoenix Project: {tenant_id}_project")
print(f"   Memory: Initialized with Vespa isolation")
```

### 2. Tenant Configuration

```python
from cogniverse_core.config.unified_config import RoutingConfigUnified

# Configure routing for tenant
routing_config = RoutingConfigUnified(
    tenant_id=tenant_id,
    routing_mode="ensemble",
    strategies=["gliner", "llm", "langextract"],
    confidence_thresholds={
        "gliner": 0.7,
        "llm": 0.6,
        "langextract": 0.5
    },
    experience_buffer_size=10000,
    optimization_enabled=True
)

config_manager.set_routing_config(routing_config)

print(f"✅ Routing configured for {tenant_id}")
```

### 3. Tenant Deletion

```python
# Soft delete (preserves configuration history)
def soft_delete_tenant(tenant_id: str):
    """Soft delete - keeps configuration history"""

    # 1. Mark tenant as inactive
    config_manager.deactivate_tenant(tenant_id)

    # 2. Stop accepting new requests
    # (handled by runtime router)

    # 3. Preserve data for retention period
    print(f"✅ Tenant {tenant_id} marked inactive")
    print(f"   Data retention: 90 days")
    print(f"   Can be reactivated within retention period")

# Hard delete (permanent removal)
def hard_delete_tenant(tenant_id: str):
    """Hard delete - permanent removal"""

    # 1. Delete Vespa schemas
    for schema in schema_manager.list_schemas(tenant_id):
        schema_manager.delete_schema(schema)

    # 2. Delete Phoenix project
    # (manual cleanup required)

    # 3. Delete configurations
    config_manager.delete_tenant_config(tenant_id)

    # 4. Clear memory
    memory_manager = Mem0MemoryManager.get_instance(tenant_id)
    await memory_manager.clear_all_memory()

    print(f"✅ Tenant {tenant_id} permanently deleted")

# Usage
soft_delete_tenant("acme_corp")  # Default: soft delete
# hard_delete_tenant("acme_corp")  # Use with caution
```

---

## Schema Management

### Schema Naming Convention

```
<base_schema_name>_<tenant_id>

Examples:
- video_colpali_smol500_mv_frame_acme_corp
- video_videoprism_base_mv_chunk_30s_globex_inc
- video_colqwen_omni_mv_chunk_30s_default
```

### Deploy Schema for Single Tenant

```bash
# Deploy ColPali schema for tenant
JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_json_schema.py \
  --schema-path configs/schemas/video_colpali_smol500_mv_frame.json \
  --tenant-id acme_corp

# Verify schema deployed
curl http://localhost:8080/document/v1/ | jq '.schemas' | grep acme_corp
```

### Deploy Schemas for All Tenants

```bash
#!/bin/bash
# scripts/deploy_tenant_schemas.sh

TENANTS=("acme_corp" "globex_inc" "default")
SCHEMAS=(
  "video_colpali_smol500_mv_frame"
  "video_videoprism_base_mv_chunk_30s"
)

for tenant in "${TENANTS[@]}"; do
  echo "Deploying schemas for tenant: $tenant"

  for schema in "${SCHEMAS[@]}"; do
    JAX_PLATFORM_NAME=cpu uv run python scripts/deploy_json_schema.py \
      --schema-path "configs/schemas/${schema}.json" \
      --tenant-id "$tenant"

    echo "  ✓ Deployed: ${schema}_${tenant}"
  done
done

echo "✅ All schemas deployed for all tenants"
```

### List Tenant Schemas

```python
from cogniverse_vespa.backends.vespa_schema_manager import VespaSchemaManager

schema_manager = VespaSchemaManager()

# List all schemas
all_schemas = schema_manager.list_all_schemas()
print(f"Total schemas: {len(all_schemas)}")

# Group by tenant
tenant_schemas = {}
for schema in all_schemas:
    # Extract tenant from schema name (suffix after last underscore)
    parts = schema.split('_')
    if len(parts) > 1:
        tenant_id = parts[-1]
        if tenant_id not in tenant_schemas:
            tenant_schemas[tenant_id] = []
        tenant_schemas[tenant_id].append(schema)

# Display tenant schemas
for tenant_id, schemas in tenant_schemas.items():
    print(f"\nTenant: {tenant_id}")
    for schema in schemas:
        print(f"  - {schema}")
```

---

## Data Ingestion

### Ingest Video for Specific Tenant

```bash
# Ingest videos for acme_corp tenant
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
  --video_dir data/videos/acme_corp/ \
  --backend vespa \
  --profile video_colpali_smol500_mv_frame \
  --tenant-id acme_corp

# Verify ingestion
curl "http://localhost:8080/document/v1/video_colpali_smol500_mv_frame_acme_corp/frame/docid/1"
```

### Bulk Tenant Ingestion

```python
from cogniverse_agents.ingestion.pipeline import VideoIngestionPipeline
from pathlib import Path

async def ingest_tenant_videos(tenant_id: str, video_dir: Path):
    """Ingest all videos for a tenant"""

    pipeline = VideoIngestionPipeline(
        profile="video_colpali_smol500_mv_frame",
        tenant_id=tenant_id,
        backend="vespa"
    )

    video_files = list(video_dir.glob("*.mp4"))
    print(f"Processing {len(video_files)} videos for {tenant_id}")

    results = []
    for video_file in video_files:
        result = await pipeline.process_video(str(video_file))
        results.append(result)
        print(f"  ✓ Processed: {video_file.name} ({result.document_count} docs)")

    return results

# Usage
await ingest_tenant_videos("acme_corp", Path("data/videos/acme_corp"))
await ingest_tenant_videos("globex_inc", Path("data/videos/globex_inc"))
```

---

## Query & Search

### Tenant-Isolated Search

```python
from cogniverse_agents.agents.video_search_agent import VideoSearchAgent

async def search_tenant_videos(
    tenant_id: str,
    query: str,
    top_k: int = 10
):
    """Search videos within tenant isolation"""

    # Agent automatically uses tenant-specific schema
    agent = VideoSearchAgent(
        profile="video_colpali_smol500_mv_frame",
        tenant_id=tenant_id
    )

    results = await agent.search(
        query=query,
        ranking_strategy="hybrid_float_bm25",
        top_k=top_k
    )

    return results

# Search for acme_corp
acme_results = await search_tenant_videos("acme_corp", "machine learning")

# Search for globex_inc (completely isolated)
globex_results = await search_tenant_videos("globex_inc", "machine learning")

# Results are completely isolated
assert len(set([r.tenant_id for r in acme_results])) == 1
assert acme_results[0].tenant_id == "acme_corp"
```

---

## Telemetry & Monitoring

### Per-Tenant Phoenix Projects

Each tenant gets an isolated Phoenix project:

```python
from cogniverse_core.telemetry import TelemetryManager, TelemetryConfig

# Initialize telemetry for tenant
telemetry = TelemetryManager(
    config=TelemetryConfig(
        enabled=True,
        project_name=f"{tenant_id}_project",
        endpoint="localhost:4317",
        export_mode="async"
    ),
    tenant_id=tenant_id
)

# All spans are automatically tagged with tenant_id
with telemetry.start_span("video_search") as span:
    span.set_attribute("tenant_id", tenant_id)
    span.set_attribute("query", query)
    # ... perform search
```

### View Tenant-Specific Traces

```bash
# Access Phoenix dashboard
open http://localhost:6006

# Projects visible:
# - acme_corp_project
# - globex_inc_project
# - default_project

# Each project contains only that tenant's traces
```

### Cross-Tenant Analytics (Admin Only)

```python
from cogniverse_core.telemetry.phoenix_analytics import PhoenixAnalytics

# Admin can view aggregated metrics across tenants
analytics = PhoenixAnalytics(admin_mode=True)

# Get metrics by tenant
tenant_metrics = analytics.get_tenant_metrics(
    start_date="2025-01-01",
    end_date="2025-01-31"
)

for tenant_id, metrics in tenant_metrics.items():
    print(f"\n{tenant_id}:")
    print(f"  Total Queries: {metrics['total_queries']}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']}ms")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
```

---

## Memory Management

### Per-Tenant Memory

Each tenant has isolated Mem0 memory:

```python
from cogniverse_core.common.memory.mem0_memory_manager import Mem0MemoryManager

# Get tenant-specific memory manager
memory_acme = Mem0MemoryManager.get_instance("acme_corp")
memory_globex = Mem0MemoryManager.get_instance("globex_inc")

# Add memory for acme_corp
await memory_acme.add_memory(
    message="User prefers technical tutorials",
    user_id="user123",
    agent_id="video_search_agent"
)

# Search memory (tenant-isolated)
memories = await memory_acme.search_memory(
    query="What does the user prefer?",
    user_id="user123"
)

# Memories are completely isolated per tenant
assert all(m.tenant_id == "acme_corp" for m in memories)
```

### Memory Statistics per Tenant

```python
# Get memory stats for tenant
stats = await memory_acme.get_memory_stats()

print(f"Tenant: acme_corp")
print(f"  Total Memories: {stats['total_memories']}")
print(f"  Unique Users: {stats['unique_users']}")
print(f"  Unique Agents: {stats['unique_agents']}")
print(f"  Storage Size: {stats['storage_mb']} MB")
```

---

## Configuration Management

### Tenant Configuration Templates

```python
# Define configuration templates
TENANT_TEMPLATES = {
    "enterprise": {
        "llm_model": "gpt-4",
        "max_qps": 100,
        "cache_ttl": 3600,
        "memory_enabled": True,
        "optimization_enabled": True
    },
    "startup": {
        "llm_model": "gpt-3.5-turbo",
        "max_qps": 10,
        "cache_ttl": 1800,
        "memory_enabled": True,
        "optimization_enabled": False
    },
    "trial": {
        "llm_model": "gpt-3.5-turbo",
        "max_qps": 5,
        "cache_ttl": 600,
        "memory_enabled": False,
        "optimization_enabled": False
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
    overrides={"max_qps": 20}  # Custom override
)
```

### Configuration Rollback

```python
# Rollback tenant configuration
def rollback_tenant_config(tenant_id: str, version: int):
    """Rollback tenant configuration to specific version"""

    config_manager.rollback_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        target_version=version
    )

    print(f"✅ Rolled back {tenant_id} to version {version}")

# Usage
rollback_tenant_config("acme_corp", version=5)
```

---

## Security & Isolation

### Tenant Isolation Verification

```python
async def verify_tenant_isolation(tenant_a: str, tenant_b: str):
    """Verify complete isolation between two tenants"""

    checks = []

    # 1. Schema isolation
    schemas_a = schema_manager.list_schemas(tenant_a)
    schemas_b = schema_manager.list_schemas(tenant_b)
    checks.append(("Schema Isolation", len(set(schemas_a) & set(schemas_b)) == 0))

    # 2. Data isolation (search)
    agent_a = VideoSearchAgent(tenant_id=tenant_a)
    results_a = await agent_a.search("test query", top_k=100)
    checks.append(("Data Isolation", all(r.tenant_id == tenant_a for r in results_a)))

    # 3. Memory isolation
    memory_a = Mem0MemoryManager.get_instance(tenant_a)
    memories_a = await memory_a.get_all_memories()
    checks.append(("Memory Isolation", all(m.tenant_id == tenant_a for m in memories_a)))

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
# Tenant access control decorator
def require_tenant_access(tenant_id: str):
    """Decorator to enforce tenant access control"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Verify caller has access to tenant
            caller_tenant = kwargs.get('caller_tenant_id')
            if caller_tenant != tenant_id:
                raise PermissionError(
                    f"Caller {caller_tenant} not authorized for tenant {tenant_id}"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_tenant_access("acme_corp")
async def search_acme_videos(query: str, caller_tenant_id: str):
    """Search restricted to acme_corp tenant"""
    agent = VideoSearchAgent(tenant_id="acme_corp")
    return await agent.search(query)
```

---

## Performance & Optimization

### Per-Tenant Performance Metrics

```python
from cogniverse_core.telemetry.modality_metrics import ModalityMetricsTracker

# Track performance per tenant
metrics = ModalityMetricsTracker(tenant_id="acme_corp")

# Get tenant-specific performance
stats = metrics.get_summary_stats()

print(f"Tenant: acme_corp")
print(f"  Total Requests: {stats['total_requests']}")
print(f"  Success Rate: {stats['overall_success_rate']:.1%}")
print(f"  P50 Latency: {stats['p50_latency_ms']}ms")
print(f"  P95 Latency: {stats['p95_latency_ms']}ms")
print(f"  P99 Latency: {stats['p99_latency_ms']}ms")
```

### Tenant-Specific Optimization

```python
from cogniverse_agents.routing.optimization_orchestrator import OptimizationOrchestrator

# Run optimization per tenant
orchestrator = OptimizationOrchestrator(
    tenant_id="acme_corp",
    span_eval_interval_minutes=15,
    annotation_interval_minutes=30
)

# Start tenant-specific optimization
await orchestrator.start()

# Optimization runs in background, improving tenant's routing
print(f"✅ Optimization started for acme_corp")
print(f"   Span evaluation: Every 15 minutes")
print(f"   Model retraining: Based on annotations")
```

---

## Backup & Recovery

### Tenant Data Backup

```bash
#!/bin/bash
# scripts/backup_tenant_data.sh

TENANT_ID="acme_corp"
BACKUP_DIR="backups/${TENANT_ID}/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

echo "Backing up tenant: $TENANT_ID"

# 1. Export configuration
uv run python scripts/export_tenant_config.py \
  --tenant-id "$TENANT_ID" \
  --output "$BACKUP_DIR/config.json"

# 2. Backup Vespa documents (per schema)
for schema in video_colpali_smol500_mv_frame video_videoprism_base_mv_chunk_30s; do
  schema_name="${schema}_${TENANT_ID}"

  curl "http://localhost:8080/document/v1/${schema_name}/frame/docid" \
    > "$BACKUP_DIR/${schema_name}_documents.json"
done

# 3. Export Phoenix traces (optional)
# Manual export from Phoenix dashboard

# 4. Backup memory
uv run python scripts/export_tenant_memory.py \
  --tenant-id "$TENANT_ID" \
  --output "$BACKUP_DIR/memory.json"

echo "✅ Backup complete: $BACKUP_DIR"
```

### Tenant Data Restore

```bash
#!/bin/bash
# scripts/restore_tenant_data.sh

TENANT_ID="acme_corp"
BACKUP_DIR="$1"

echo "Restoring tenant: $TENANT_ID from $BACKUP_DIR"

# 1. Restore configuration
uv run python scripts/import_tenant_config.py \
  --tenant-id "$TENANT_ID" \
  --input "$BACKUP_DIR/config.json"

# 2. Deploy schemas
# (Schemas must be deployed before data restore)

# 3. Restore Vespa documents
for doc_file in "$BACKUP_DIR"/*_documents.json; do
  curl -X POST \
    -H "Content-Type: application/json" \
    --data-binary "@$doc_file" \
    "http://localhost:8080/document/v1/"
done

# 4. Restore memory
uv run python scripts/import_tenant_memory.py \
  --tenant-id "$TENANT_ID" \
  --input "$BACKUP_DIR/memory.json"

echo "✅ Restore complete for $TENANT_ID"
```

---

## Troubleshooting

### Tenant Schema Not Found

```python
# Check if tenant schemas exist
def diagnose_tenant_schemas(tenant_id: str):
    """Diagnose tenant schema issues"""

    print(f"Diagnosing schemas for: {tenant_id}\n")

    # List all schemas
    all_schemas = schema_manager.list_all_schemas()
    tenant_schemas = [s for s in all_schemas if s.endswith(f"_{tenant_id}")]

    if not tenant_schemas:
        print(f"❌ No schemas found for {tenant_id}")
        print(f"   Run: uv run python scripts/deploy_json_schema.py --tenant-id {tenant_id}")
    else:
        print(f"✅ Found {len(tenant_schemas)} schemas:")
        for schema in tenant_schemas:
            print(f"   - {schema}")

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
        print(f"   Phoenix Project: {config.phoenix_project_name}")
        print(f"   Vespa URL: {config.vespa_url}")
    except ConfigNotFoundError:
        print(f"❌ Configuration not found for {tenant_id}")
        print(f"   Run: config_manager.set_system_config(SystemConfig(tenant_id='{tenant_id}'))")

# Usage
diagnose_tenant_config("acme_corp")
```

---

## Best Practices

### 1. Always Use Tenant Context

```python
# ✅ Good: Explicit tenant ID
agent = VideoSearchAgent(tenant_id="acme_corp")
results = await agent.search(query)

# ❌ Bad: Implicit default tenant
agent = VideoSearchAgent()  # Uses "default" tenant
```

### 2. Verify Tenant Isolation

```python
# Always verify isolation in tests
async def test_tenant_isolation():
    # Create test data for tenant A
    agent_a = VideoSearchAgent(tenant_id="tenant_a")
    # ... ingest test data

    # Search from tenant B
    agent_b = VideoSearchAgent(tenant_id="tenant_b")
    results = await agent_b.search("test query")

    # Verify no cross-tenant leakage
    assert len(results) == 0, "Tenant isolation violated!"
```

### 3. Monitor Tenant Health

```python
# Regular tenant health checks
async def monitor_tenant_health(tenant_id: str):
    """Monitor tenant health metrics"""

    health = {
        "schemas": len(schema_manager.list_schemas(tenant_id)),
        "config": config_manager.has_system_config(tenant_id),
        "memory_initialized": Mem0MemoryManager.has_instance(tenant_id),
        "phoenix_project": f"{tenant_id}_project"
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
# Enforce tenant quotas
class TenantQuotaManager:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.quotas = self._load_quotas()

    async def check_quota(self, resource: str, amount: int = 1):
        """Check if tenant has quota for resource"""

        current_usage = await self._get_usage(resource)
        max_quota = self.quotas[resource]

        if current_usage + amount > max_quota:
            raise QuotaExceededError(
                f"Tenant {self.tenant_id} exceeded {resource} quota"
            )

        return True

    async def _get_usage(self, resource: str):
        """Get current usage for resource"""
        # Query from telemetry or database
        pass

# Usage
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
