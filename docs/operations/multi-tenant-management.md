# Multi-Tenant Management Guide

---

## Overview

Cogniverse implements a **schema-per-tenant** multi-tenant architecture providing complete physical data isolation. Each tenant operates in a dedicated Vespa schema with isolated storage, preventing any cross-tenant data access.

### Key Features

- **Physical Isolation**: Schema-per-tenant architecture in Vespa
- **Organizational Hierarchy**: Organizations contain multiple tenants
- **Lazy Schema Creation**: Schemas created on first tenant access
- **REST API Management**: Complete tenant lifecycle via HTTP API
- **Automatic Routing**: Tenant-aware query routing in all agents
- **Storage Isolation**: Dedicated directories per tenant

### Architecture at a Glance

```mermaid
flowchart TB
    Client["<span style='color:#000'>Client Application</span>"]

    Client --> TenantAPI["<span style='color:#000'>Tenant Management API<br/>Port 9000</span>"]
    Client --> Agents["<span style='color:#000'>Multi-Tenant Agents</span>"]

    TenantAPI --> TenantMgr["<span style='color:#000'>VespaSchemaManager</span>"]
    Agents --> TenantSearch["<span style='color:#000'>TenantAwareVespaSearchClient</span>"]

    TenantMgr --> OrgMeta[("<span style='color:#000'>organization_metadata</span>")]
    TenantMgr --> TenantMeta[("<span style='color:#000'>tenant_metadata</span>")]
    TenantMgr --> Schemas["<span style='color:#000'>Tenant Schemas</span>"]

    TenantSearch --> Schemas

    Schemas --> AcmeProd[("<span style='color:#000'>Schema: acme_production</span>")]
    Schemas --> AcmeDev[("<span style='color:#000'>Schema: acme_dev</span>")]
    Schemas --> InitechProd[("<span style='color:#000'>Schema: initech_production</span>")]

    style Client fill:#90caf9,stroke:#1565c0,color:#000
    style TenantAPI fill:#ffcc80,stroke:#ef6c00,color:#000
    style Agents fill:#ce93d8,stroke:#7b1fa2,color:#000
    style TenantMgr fill:#ce93d8,stroke:#7b1fa2,color:#000
    style TenantSearch fill:#a5d6a7,stroke:#388e3c,color:#000
    style OrgMeta fill:#b0bec5,stroke:#546e7a,color:#000
    style TenantMeta fill:#b0bec5,stroke:#546e7a,color:#000
    style Schemas fill:#b0bec5,stroke:#546e7a,color:#000
    style AcmeProd fill:#a5d6a7,stroke:#388e3c,color:#000
    style AcmeDev fill:#a5d6a7,stroke:#388e3c,color:#000
    style InitechProd fill:#a5d6a7,stroke:#388e3c,color:#000
```

---

## Tenant Hierarchy

### Organization and Tenant Concepts

**Organization**: Top-level entity representing a company or business unit
- Examples: `acme`, `initech`, `hooli`
- Validation: Alphanumeric and underscore only (`^[a-zA-Z0-9_]+$`)
- Purpose: Group related tenants under common ownership

**Tenant**: Environment or deployment within an organization
- Examples: `production`, `staging`, `dev`, `customer-123`
- Validation: Alphanumeric, underscore, and hyphen (`^[a-zA-Z0-9_-]+$`)
- Purpose: Isolated data environment for specific use case

### Tenant ID Format

All tenant operations use the **org:tenant** format:

```json
{organization}:{tenant_name}

Examples:
- acme:production
- acme:staging
- acme:dev
- initech:production
- hooli:customer_123
```

**Parsing Rules**:

- Colon `:` separates organization from tenant name
- Exactly one colon required
- Both parts must pass validation
- Case-sensitive (lowercase recommended)

### Validation Rules

| Component | Pattern | Valid Examples | Invalid Examples |
|-----------|---------|----------------|------------------|
| **Organization** | `^[a-zA-Z0-9_]+$` | `acme`, `org_123`, `ACME` | `acme-corp`, `acme.com`, `acme corp` |
| **Tenant Name** | `^[a-zA-Z0-9_-]+$` | `production`, `dev-2024`, `cust_1` | `prod.env`, `staging:v2`, `test env` |

**Common Validation Errors**:

- ❌ `acme-corp:production` - Hyphen not allowed in organization
- ❌ `acme:prod.env` - Dot not allowed in tenant name
- ❌ `acme:prod env` - Space not allowed
- ❌ `acme::production` - Empty component
- ❌ `production` - Missing organization prefix
- ✅ `acme:production` - Valid format

### Storage Structure

Tenants are organized in a hierarchical directory structure:

```text
base_dir/
├── acme/                      # Organization
│   ├── production/            # Tenant
│   │   ├── videos/
│   │   ├── embeddings/
│   │   └── metadata/
│   ├── staging/               # Tenant
│   └── dev/                   # Tenant
├── initech/                   # Organization
│   └── production/            # Tenant
└── hooli/                     # Organization
    ├── production/            # Tenant
    └── customer_123/          # Tenant
```

---

## Tenant Management REST API

The tenant management service (`libs/runtime/cogniverse_runtime/admin/tenant_manager.py`) provides a complete REST API for organization and tenant lifecycle management.

### API Endpoints

| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| POST | `/admin/organizations` | Create organization | None (open) |
| POST | `/admin/tenants` | Create tenant | None (open) |
| GET | `/admin/organizations` | List all organizations | None (open) |
| GET | `/admin/organizations/{org_id}` | Get organization details | None (open) |
| GET | `/admin/organizations/{org_id}/tenants` | List tenants for organization | None (open) |
| GET | `/admin/tenants/{tenant_full_id}` | Get tenant details | None (open) |
| DELETE | `/admin/tenants/{tenant_full_id}` | Delete tenant and all data | None (open) |
| DELETE | `/admin/organizations/{org_id}` | Delete organization and all tenants | None (open) |

### Create Organization

Creates a new organization and initializes metadata storage.

**Request**:
```bash
curl -X POST http://localhost:9000/admin/organizations \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "acme",
    "org_name": "ACME Corporation",
    "created_by": "admin"
  }'
```

**Response** (201 Created):
```json
{
  "org_id": "acme",
  "org_name": "ACME Corporation",
  "created_at": 1728470000000,
  "created_by": "admin",
  "status": "active",
  "tenant_count": 0,
  "config": {}
}
```

**Validation**:

- `org_id` must match `^[a-zA-Z0-9_]+$`
- `org_id` cannot be empty
- Organization must not already exist

### Create Tenant

Creates a new tenant within an organization, initializes storage, and creates Vespa schemas.

**Request**:
```bash
curl -X POST http://localhost:9000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "acme",
    "tenant_id": "production",
    "created_by": "admin"
  }'
```

**Alternate Format** (full tenant_id):
```bash
curl -X POST http://localhost:9000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme:production",
    "created_by": "admin"
  }'
```

**Response** (201 Created):
```json
{
  "tenant_full_id": "acme:production",
  "org_id": "acme",
  "tenant_name": "production",
  "created_at": 1728470100000,
  "created_by": "admin",
  "status": "active",
  "schemas_deployed": ["video_colpali_smol500_mv_frame"]
}
```

**Validation**:

- Organization must exist before creating tenant
- `tenant_id` must match `^[a-zA-Z0-9_-]+$` (if using org_id + tenant_id format)
- Full `tenant_id` must be valid org:tenant format
- Tenant must not already exist in organization

### List Organizations

Retrieves all organizations in the system.

**Request**:
```bash
curl http://localhost:9000/admin/organizations
```

**Response** (200 OK):
```json
{
  "organizations": [
    {
      "org_id": "acme",
      "org_name": "ACME Corporation",
      "created_at": 1728470000000,
      "created_by": "admin",
      "status": "active",
      "tenant_count": 3
    },
    {
      "org_id": "initech",
      "org_name": "Initech Inc",
      "created_at": 1728471800000,
      "created_by": "admin",
      "status": "active",
      "tenant_count": 1
    }
  ],
  "total_count": 2
}
```

### List Tenants

Retrieves all tenants, optionally filtered by organization.

**Request** (list tenants for org):
```bash
curl http://localhost:9000/admin/organizations/acme/tenants
```

**Response** (200 OK):
```json
{
  "tenants": [
    {
      "tenant_full_id": "acme:production",
      "org_id": "acme",
      "tenant_name": "production",
      "created_at": 1728470100000,
      "created_by": "admin",
      "status": "active",
      "schemas_deployed": ["video_colpali_smol500_mv_frame"]
    },
    {
      "tenant_full_id": "acme:staging",
      "org_id": "acme",
      "tenant_name": "staging",
      "created_at": 1728470160000,
      "created_by": "admin",
      "status": "active",
      "schemas_deployed": ["video_colpali_smol500_mv_frame"]
    }
  ],
  "total_count": 2,
  "org_id": "acme"
}
```

### Delete Tenant

Deletes a tenant and all associated data (schemas, documents, storage).

**Request**:
```bash
curl -X DELETE http://localhost:9000/admin/tenants/acme:staging
```

**Response** (200 OK):
```json
{
  "status": "deleted",
  "tenant_full_id": "acme:staging",
  "schemas_deleted": 1,
  "deleted_schemas": ["video_colpali_smol500_mv_frame_acme_staging"]
}
```

**Warning**: This operation is **irreversible**. All tenant data is permanently deleted.

### Error Responses

All endpoints return FastAPI HTTPException responses:

**400 Bad Request** (validation failure):
```json
{
  "detail": "Invalid org_id 'acme-corp': only alphanumeric and underscore allowed"
}
```

**404 Not Found** (resource doesn't exist):
```json
{
  "detail": "Organization acme not found"
}
```

**409 Conflict** (resource already exists):
```json
{
  "detail": "Organization acme already exists"
}
```

**500 Internal Server Error** (backend failure):
```json
{
  "detail": "Failed to create organization acme in backend"
}
```

---

## Tenant Lifecycle Flows

### Create Organization Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Tenant API
    participant Validator
    participant VespaClient as Vespa Client
    participant Storage

    Client->>API: POST /organizations<br/>{org_id: "acme"}

    API->>Validator: validate_org_id("acme")
    alt Invalid org_id
        Validator-->>API: ValueError
        API-->>Client: 400 Bad Request
    end

    Validator-->>API: Valid

    API->>VespaClient: Check if org exists
    alt Organization exists
        VespaClient-->>API: Found
        API-->>Client: 409 Conflict
    end

    VespaClient-->>API: Not found

    API->>VespaClient: Feed organization_metadata
    VespaClient-->>API: Success

    API->>Storage: Create directory: base_dir/acme/
    Storage-->>API: Created

    API-->>Client: 201 Created<br/>{org_id: "acme"}
```

### Create Tenant Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Tenant API
    participant Validator
    participant SchemaManager as VespaSchemaManager
    participant VespaClient as Vespa Client
    participant Storage

    Client->>API: POST /tenants<br/>{org_id: "acme", tenant_id: "production"}

    API->>Validator: parse_tenant_id("acme:production")
    Validator-->>API: ("acme", "production")

    API->>Validator: validate_org_id("acme")
    API->>Validator: validate_tenant_name("production")

    alt Validation fails
        Validator-->>API: ValueError
        API-->>Client: 400 Bad Request
    end

    Validator-->>API: Valid

    API->>VespaClient: Check if org exists
    alt Organization not found
        VespaClient-->>API: Not found
        API-->>Client: 404 Not Found
    end

    VespaClient-->>API: Found

    API->>SchemaManager: Create tenant schemas via BackendRegistry

    SchemaManager->>SchemaManager: Initialize tenant-specific schemas<br/>(lazy creation via SchemaRegistry)

    SchemaManager->>VespaClient: Feed tenant_metadata
    VespaClient-->>SchemaManager: Success

    SchemaManager->>Storage: Create directory:<br/>base_dir/acme/production/
    Storage-->>SchemaManager: Created

    SchemaManager-->>API: Tenant schemas initialized

    API-->>Client: 201 Created<br/>{tenant_id: "acme:production"}
```

### Query with Tenant Routing Flow

```mermaid
sequenceDiagram
    participant Client
    participant Agent as VideoSearchAgent
    participant SearchClient as TenantAwareVespaSearchClient
    participant Parser as parse_tenant_id()
    participant SchemaManager as VespaSchemaManager
    participant Vespa

    Client->>Agent: search(query="tutorial",<br/>tenant_id="acme:production")

    Agent->>SearchClient: search(query, tenant_id)

    SearchClient->>Parser: parse_tenant_id("acme:production")
    Parser-->>SearchClient: ("acme", "production")

    alt Invalid tenant_id
        Parser-->>SearchClient: ValueError
        SearchClient-->>Client: 400 Bad Request
    end

    SearchClient->>SchemaManager: get_tenant_schema_name(tenant_id, base_schema_name)
    SchemaManager-->>SearchClient: "video_colpali_smol500_mv_frame_acme_production"

    SearchClient->>Vespa: Query schema:<br/>video_colpali_acme_production
    Vespa-->>SearchClient: Results (isolated to tenant)

    SearchClient-->>Agent: Search results
    Agent-->>Client: Results
```

### Delete Tenant Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Tenant API
    participant Validator
    participant SchemaManager as VespaSchemaManager
    participant VespaClient as Vespa Client
    participant Storage

    Client->>API: DELETE /tenants/acme:staging

    API->>Validator: parse_tenant_id("acme:staging")
    Validator-->>API: ("acme", "staging")

    API->>SchemaManager: Check if tenant exists
    alt Tenant not found
        SchemaManager-->>API: Not found
        API-->>Client: 404 Not Found
    end

    SchemaManager-->>API: Exists

    API->>SchemaManager: Get tenant schemas
    SchemaManager-->>API: List of schema names

    loop For each schema
        API->>VespaClient: Delete schema
        VespaClient-->>API: Deleted
    end

    API->>VespaClient: Delete tenant_metadata
    VespaClient-->>API: Deleted

    API->>Storage: Delete directory:<br/>base_dir/acme/staging/
    Storage-->>API: Deleted

    API-->>Client: 200 OK<br/>{schemas_deleted: 4}
```

---

## Core Components

### VespaSchemaManager

**Location**: `libs/vespa/cogniverse_vespa/vespa_schema_manager.py` (implementation layer)

**Purpose**: Manages the lifecycle of tenant-specific Vespa schemas with lazy creation and automatic tenant isolation.

**Key Responsibilities**:

- Lazy schema creation on first tenant access
- Tenant registration and validation
- Schema naming with tenant isolation
- Metadata schema management (organization_metadata, tenant_metadata)

**Schema Naming Convention**:
```json
{profile}_{org}_{tenant}

Examples:
- video_colpali_smol500_mv_frame_acme_production
- video_videoprism_base_mv_chunk_30s_initech_staging
```

**Usage**:
```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager  # Implementation layer
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from pathlib import Path

# Initialize dependencies
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))  # For loading schema templates
config_manager = create_default_config_manager()
schema_registry = SchemaRegistry(config_manager=config_manager, backend=None, schema_loader=schema_loader)

# Initialize schema manager (backend_endpoint and backend_port are REQUIRED)
schema_manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080,
    schema_loader=schema_loader,
    schema_registry=schema_registry
)

# Get tenant-specific schema name
schema_name = schema_manager.get_tenant_schema_name(
    tenant_id="acme:production",
    base_schema_name="video_colpali_smol500_mv_frame"
)
# Returns: "video_colpali_smol500_mv_frame_acme_production"
```

### TenantAwareVespaSearchClient

**Location**: `libs/vespa/cogniverse_vespa/tenant_aware_search_client.py` (implementation layer)

**Purpose**: Automatic tenant-aware query routing ensuring all search operations are isolated to the correct tenant schema.

**Key Responsibilities**:

- Parse and validate tenant_id from requests
- Route queries to tenant-specific schemas
- Prevent cross-tenant data access
- Handle tenant-not-found errors

**Usage**:
```python
from cogniverse_vespa.tenant_aware_search_client import TenantAwareVespaSearchClient  # Implementation layer
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize dependencies (all REQUIRED)
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))  # REQUIRED for schema operations

search_client = TenantAwareVespaSearchClient(
    tenant_id="acme:production",
    base_schema_name="video_colpali_smol500_mv_frame",
    config_manager=config_manager,
    schema_loader=schema_loader,
    backend_url="http://localhost",
    backend_port=8080
)

# Search with automatic tenant isolation
results = search_client.search(
    query_text="machine learning tutorial",
    strategy="hybrid_float_bm25",
    top_k=10
)
# Automatically routes to: video_colpali_smol500_mv_frame_acme_production
```

### parse_tenant_id() Utility

**Location**: `libs/core/cogniverse_core/common/tenant_utils.py` (core layer)

**Purpose**: Parse and validate org:tenant format.

**Signature**:
```python
def parse_tenant_id(tenant_id: str) -> tuple[str, str]:
    """
    Parse tenant_id into org_id and tenant_name.

    Supports two formats:
    - Simple: "acme" -> ("acme", "acme")
    - Org:tenant: "acme:production" -> ("acme", "production")

    Args:
        tenant_id: Tenant identifier (simple or org:tenant format)

    Returns:
        Tuple of (org_id, tenant_name)

    Raises:
        ValueError: If tenant_id is empty or has invalid format
    """
```

**Usage**:
```python
from cogniverse_core.common.tenant_utils import parse_tenant_id  # Core layer

# Parse tenant ID
org_id, tenant_name = parse_tenant_id("acme:production")
# Returns: ("acme", "production")

# Simple format (no colon) returns same value for both org and tenant
parse_tenant_id("acme")     # Returns: ("acme", "acme")

# Validation errors
parse_tenant_id("")          # ValueError: tenant_id cannot be empty
parse_tenant_id("a:b:c")    # ValueError: Expected 'org:tenant' with single colon
parse_tenant_id("acme:")     # ValueError: both org and tenant parts must be non-empty
```

### Agent Factory Pattern

**Purpose**: Per-tenant agent instances with caching

All agents implement a factory function for tenant-aware instantiation:

```python
# Video Search Agent (implementation layer)
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# ONE agent serves ALL tenants — profile and tenant_id are per-request
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Routing Agent (implementation layer)
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_foundation.telemetry.config import TelemetryConfig

deps = RoutingDeps(
    tenant_id="acme:production",
    telemetry_config=TelemetryConfig(),
    model_name="smollm3:3b",
    base_url="http://localhost:11434/v1"
)
router = RoutingAgent(deps=deps)
```

**Caching**: Agents are cached per tenant_id to avoid re-initialization overhead.

**Validation**: All agents raise `ValueError` if tenant_id is empty or None.

---

## Usage Examples

### Complete Tenant Setup

```bash
# 1. Create organization
curl -X POST http://localhost:9000/admin/organizations \
  -H "Content-Type: application/json" \
  -d '{"org_id": "acme", "org_name": "ACME Corporation", "created_by": "admin"}'

# 2. Create production tenant
curl -X POST http://localhost:9000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme:production", "created_by": "admin"}'

# 3. Create staging tenant
curl -X POST http://localhost:9000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme:staging", "created_by": "admin"}'
```

### Video Ingestion with Tenant Isolation

```python
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline  # Application layer
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize pipeline for tenant
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))  # Optional but recommended for schema operations

pipeline = VideoIngestionPipeline(
    tenant_id="acme:production",
    config_manager=config_manager,
    schema_loader=schema_loader,
    schema_name="video_colpali_smol500_mv_frame"
)

# Process video (automatically isolated to acme:production schema)
result = await pipeline.process_video_async(
    video_path=Path("/path/to/video.mp4")
)
```

### Video Search with Tenant Isolation

```python
from cogniverse_agents.video_agent_refactored import VideoSearchAgent  # Implementation layer
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# ONE agent serves ALL tenants — no profile or tenant_id at construction
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Search (automatically isolated to acme:production) - synchronous
# profile and tenant_id are PER-REQUEST parameters on search()
results = agent.search(
    query="machine learning tutorial",
    profile="video_colpali_smol500_mv_frame",
    tenant_id="acme:production",
    top_k=10,
)

# Results only from acme:production schema - no cross-tenant access
```

### List All Tenants for Organization

```bash
# List all tenants for "acme" organization
curl http://localhost:9000/admin/organizations/acme/tenants

# Response shows: acme:production, acme:staging, acme:dev
```

### Delete Tenant

```bash
# Delete staging tenant and all data
curl -X DELETE http://localhost:9000/admin/tenants/acme:staging

# Warning: This permanently deletes:
# - All Vespa schemas for this tenant
# - All documents in tenant schemas
# - All storage directories
# - Tenant metadata
```

---

## Security and Isolation Guarantees

### Physical Data Isolation

- **Schema-per-tenant**: Each tenant has dedicated Vespa schemas
- **No shared schemas**: Zero possibility of cross-tenant data leakage
- **Automatic routing**: TenantAwareVespaSearchClient enforces tenant boundaries
- **Validation at entry**: All API requests validate tenant_id before processing

### Storage Isolation

- **Separate directories**: Each tenant has isolated storage path
- **No shared files**: Video files, embeddings, metadata stored separately
- **Access control**: Directory structure prevents cross-tenant access

### Query Isolation

- **Tenant-specific schemas**: Queries only search tenant's schemas
- **No cross-tenant queries**: Impossible to query data from different tenant
- **Validated routing**: parse_tenant_id() validates before schema selection

### Metadata Isolation

- **organization_metadata**: Separate Vespa schema for organization data
- **tenant_metadata**: Separate Vespa schema for tenant data
- **No shared metadata**: Each tenant's metadata completely isolated

---

## Related Documentation

- [Deployment Guide](deployment.md) - Multi-tenant deployment procedures
- [Configuration Guide](configuration.md) - Multi-tenant configuration spanning foundation and core layers
- [Multi-Tenant Operations](multi-tenant-ops.md) - Day-to-day tenant operations
- [Architecture Documentation](../architecture/sdk-architecture.md) - layered architecture
