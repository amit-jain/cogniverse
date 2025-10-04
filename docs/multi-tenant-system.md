# Multi-Tenant Architecture

## Overview

Cogniverse implements **schema-per-tenant isolation** in Vespa, ensuring complete data separation between tenants. Each tenant gets their own dedicated schema with independent indexing, querying, and telemetry.

## Tenant Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Tenant Registration                                   │
│    - Tenant ID assigned                                  │
│    - Configuration initialized                           │
│    - Phoenix project created                             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. First Request (Lazy Schema Creation)                 │
│    - Tenant context extracted from JWT/header           │
│    - Schema deployed to Vespa if not exists             │
│    - Telemetry tracer provider initialized              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Subsequent Requests (Reuse)                          │
│    - Tenant context extraction                           │
│    - Use existing schema                                 │
│    - Use cached tracer provider                          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Data Isolation                                        │
│    - All queries scoped to tenant schema                 │
│    - Complete document isolation                         │
│    - Separate telemetry projects                         │
└─────────────────────────────────────────────────────────┘
```

## Tenant Context Extraction

### JWT-Based Authentication

```python
from src.middleware.tenant import extract_tenant_from_jwt

# Middleware automatically extracts tenant_id
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    # Extract from Authorization header
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    tenant_id = extract_tenant_from_jwt(token)

    # Store in request state
    request.state.tenant_id = tenant_id

    response = await call_next(request)
    return response
```

### Header-Based (Development)

```python
# Alternative: Extract from X-Tenant-ID header
tenant_id = request.headers.get("X-Tenant-ID", "default")
request.state.tenant_id = tenant_id
```

### Query Parameter (Testing)

```python
# For testing: tenant_id in query params
tenant_id = request.query_params.get("tenant_id", "default")
```

## Schema-Per-Tenant Architecture

### Vespa Schema Naming

Each tenant gets a unique schema name:

```
Pattern: {base_schema}_{tenant_id}

Examples:
- video_colpali_smol500_mv_frame_acme
- video_videoprism_base_mv_chunk_30s_acme
- video_colqwen_omni_mv_chunk_30s_globex
```

### Lazy Schema Deployment

```python
from src.backends.vespa.schema_manager import VespaSchemaManager

schema_manager = VespaSchemaManager(
    vespa_endpoint="http://localhost:8080",
    vespa_port=19071
)

# Automatic schema creation on first request
tenant_schema = f"{base_schema}_{tenant_id}"

if not schema_manager.schema_exists(tenant_schema):
    # Clone base schema with tenant suffix
    schema_manager.deploy_tenant_schema(
        base_schema=base_schema,
        tenant_id=tenant_id
    )
```

### Schema Deployment Process

```
1. Check if tenant schema exists
2. If not:
   a. Load base schema definition
   b. Add tenant suffix to schema name
   c. Update document type name
   d. Deploy to Vespa config server
   e. Wait for schema activation
3. Cache schema metadata
```

## Tenant Isolation Guarantees

### Data Isolation

- **Complete document separation**: Documents stored in tenant-specific schemas
- **Query isolation**: All queries scoped to tenant schema only
- **Index isolation**: Separate indexes per tenant
- **No cross-tenant queries**: Impossible to query another tenant's data

### Telemetry Isolation

```python
from src.app.telemetry.manager import TelemetryManager

telemetry = TelemetryManager()

# Tenant-specific tracer provider
with telemetry.span("search", tenant_id="acme") as span:
    span.set_attribute("query", "machine learning")
    # Exported to Phoenix project: "acme-routing"
```

**Phoenix Project Mapping:**
- Tenant ID: `acme` → Phoenix Project: `acme-routing`
- Tenant ID: `globex` → Phoenix Project: `globex-routing`
- Default tenant → Phoenix Project: `default-routing`

### Resource Isolation

- **LRU cache per tenant**: Tracer providers cached separately
- **Connection pooling**: Shared but isolated by schema
- **Rate limiting**: Per-tenant request limits (future)

## Configuration Management

### Tenant-Specific Config

```python
from src.common.config_manager import get_config_manager

config_manager = get_config_manager()

# Get tenant-specific configuration
system_config = config_manager.get_system_config(tenant_id="acme")
routing_config = config_manager.get_routing_config(tenant_id="acme")
telemetry_config = config_manager.get_telemetry_config(tenant_id="acme")
```

### Config Hierarchy

```
1. Tenant-specific config (if exists)
   ↓ (fallback)
2. Default tenant config
   ↓ (fallback)
3. System defaults
```

## API Usage Examples

### Ingestion Request (Tenant-Aware)

```bash
# With JWT authentication
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <jwt_with_tenant_claim>" \
  -F "video=@sample.mp4" \
  -F "profile=video_colpali_smol500_mv_frame"

# With header
curl -X POST http://localhost:8000/ingest \
  -H "X-Tenant-ID: acme" \
  -F "video=@sample.mp4" \
  -F "profile=video_colpali_smol500_mv_frame"
```

Result:
- Video processed with ColPali
- Documents fed to schema: `video_colpali_smol500_mv_frame_acme`
- Telemetry sent to Phoenix project: `acme-routing`

### Search Request (Tenant-Aware)

```bash
# Search is automatically scoped to tenant
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer <jwt_token>" \
  -d '{
    "query": "machine learning tutorial",
    "ranking_strategy": "hybrid_float_bm25",
    "top_k": 10
  }'
```

Execution:
1. Extract `tenant_id` from JWT → `acme`
2. Query schema: `video_colpali_smol500_mv_frame_acme`
3. Return only ACME's documents
4. Log to Phoenix project: `acme-routing`

## Vespa Query Scoping

### Automatic Schema Selection

```python
from src.backends.vespa.client import VespaSearchClient

client = VespaSearchClient(
    vespa_url="http://localhost",
    vespa_port=8080
)

# Search automatically scoped to tenant schema
results = client.search(
    query="machine learning",
    tenant_id="acme",  # Auto-selects schema: video_colpali_smol500_mv_frame_acme
    schema="video_colpali_smol500_mv_frame",
    ranking_strategy="hybrid_float_bm25"
)
```

### Query Construction

```python
# Internal query building
tenant_schema = f"{schema}_{tenant_id}"

yql = f"""
    SELECT * FROM {tenant_schema}
    WHERE userQuery()
    LIMIT {top_k}
"""
```

## Performance Considerations

### Schema Caching

- Schema metadata cached after first deployment
- No repeated checks for existing schemas
- Cache invalidation on schema updates

### Connection Pooling

```python
# Shared connection pool across tenants
# But queries isolated by schema name
vespa_client = VespaSearchClient(
    connection_pool_size=100,
    max_connections_per_host=20
)
```

### Tracer Provider Caching

```python
# LRU cache with max 100 tenant providers
from functools import lru_cache

@lru_cache(maxsize=100)
def get_tracer_provider(tenant_id: str):
    return create_tenant_tracer_provider(tenant_id)
```

## Monitoring & Debugging

### Tenant Request Tracking

All requests include tenant context in logs:

```json
{
  "timestamp": "2025-10-04T12:00:00Z",
  "level": "INFO",
  "message": "Search request received",
  "tenant_id": "acme",
  "schema": "video_colpali_smol500_mv_frame_acme",
  "query": "machine learning"
}
```

### Phoenix Dashboard

View tenant-specific traces:
- Navigate to Phoenix project: `acme-routing`
- Filter by tenant.id attribute
- Analyze per-tenant performance

### Vespa Metrics

Query Vespa for tenant-specific metrics:

```bash
# Document count per tenant schema
curl "http://localhost:8080/document/v1/tenant/video_colpali_smol500_mv_frame_acme/docid/?selection=true&cluster=video"
```

## Security Considerations

### Tenant Verification

```python
def verify_tenant_access(token: str, requested_tenant: str) -> bool:
    """Verify user has access to requested tenant"""
    claims = decode_jwt(token)
    user_tenants = claims.get("tenants", [])

    return requested_tenant in user_tenants
```

### Schema Access Control

- Only authenticated users can access tenant schemas
- JWT claims specify allowed tenant IDs
- Middleware validates tenant access before request processing

### Data Privacy

- No cross-tenant data leakage possible
- Complete schema isolation in Vespa
- Separate Phoenix projects for telemetry
- Tenant context required for all operations

## Migration & Onboarding

### New Tenant Onboarding

```bash
# 1. Create tenant configuration
python scripts/create_tenant.py --tenant-id acme --plan enterprise

# 2. First request automatically creates schema
curl -X POST http://localhost:8000/ingest \
  -H "X-Tenant-ID: acme" \
  -F "video=@sample.mp4" \
  -F "profile=video_colpali_smol500_mv_frame"

# 3. Verify schema deployment
curl "http://localhost:19071/application/v2/tenant/default/application/default"
```

### Tenant Migration

```python
# Migrate documents from tenant A to tenant B
from src.scripts.migrate_tenant import migrate_tenant_data

migrate_tenant_data(
    source_tenant="acme_old",
    target_tenant="acme_new",
    schema="video_colpali_smol500_mv_frame"
)
```

## Troubleshooting

### Schema Not Found

**Symptom**: Query fails with "Schema not found"

**Solution**:
```bash
# Force schema recreation
python scripts/deploy_schema.py \
  --tenant-id acme \
  --schema video_colpali_smol500_mv_frame \
  --force
```

### Cross-Tenant Query Errors

**Symptom**: "Access denied" or "No results" for tenant

**Cause**: Tenant context missing or incorrect

**Solution**: Ensure tenant_id is correctly extracted:
```python
# Check request.state.tenant_id
print(f"Tenant: {request.state.tenant_id}")
```

### Phoenix Project Not Found

**Symptom**: Telemetry not appearing in Phoenix

**Solution**: Ensure Phoenix project exists:
```bash
# Create Phoenix project for tenant
python scripts/create_phoenix_project.py --tenant-id acme
```

## Related Documentation

- [Architecture Overview](architecture.md) - System architecture
- [Phoenix Integration](phoenix-integration.md) - Telemetry setup
- [Configuration System](CONFIGURATION_SYSTEM.md) - Config management

**Last Updated**: 2025-10-04
