# Multi-Tenant Architecture Diagrams

**Last Updated:** 2025-11-13
**Purpose:** Comprehensive visual documentation of multi-tenant architecture patterns with 11-package layered structure

---

## Table of Contents
1. [Tenant Isolation Overview](#tenant-isolation-overview)
2. [Schema-Per-Tenant Pattern](#schema-per-tenant-pattern)
3. [Tenant Data Flow](#tenant-data-flow)
4. [Phoenix Project Isolation](#phoenix-project-isolation)
5. [Memory Isolation](#memory-isolation)
6. [Deployment Patterns](#deployment-patterns)

---

## Tenant Isolation Overview

### Multi-Tenant System Architecture

```mermaid
graph TB
    subgraph "Tenant A - acme_corp"
        TenantA[Tenant: acme_corp]
        ConfigA[SystemConfig<br/>tenant_id=acme_corp]
        SchemaA[Vespa Schemas<br/>video_*_acme_corp]
        PhoenixA[Phoenix Project<br/>acme_corp_project]
        MemoryA[Mem0 Memory<br/>user_id=acme_corp_*]
    end

    subgraph "Tenant B - globex_inc"
        TenantB[Tenant: globex_inc]
        ConfigB[SystemConfig<br/>tenant_id=globex_inc]
        SchemaB[Vespa Schemas<br/>video_*_globex_inc]
        PhoenixB[Phoenix Project<br/>globex_inc_project]
        MemoryB[Mem0 Memory<br/>user_id=globex_inc_*]
    end

    subgraph "Tenant C - default"
        TenantC[Tenant: default]
        ConfigC[SystemConfig<br/>tenant_id=default]
        SchemaC[Vespa Schemas<br/>video_*_default]
        PhoenixC[Phoenix Project<br/>default_project]
        MemoryC[Mem0 Memory<br/>user_id=default_*]
    end

    subgraph "Shared Infrastructure"
        Vespa[Vespa Instance<br/>Port 8080, 19071]
        Phoenix[Phoenix Instance<br/>Port 6006, 4317]
        Mem0[Mem0 Backend<br/>Vespa Schema]
    end

    TenantA --> ConfigA
    ConfigA --> SchemaA
    ConfigA --> PhoenixA
    ConfigA --> MemoryA

    TenantB --> ConfigB
    ConfigB --> SchemaB
    ConfigB --> PhoenixB
    ConfigB --> MemoryB

    TenantC --> ConfigC
    ConfigC --> SchemaC
    ConfigC --> PhoenixC
    ConfigC --> MemoryC

    SchemaA --> Vespa
    SchemaB --> Vespa
    SchemaC --> Vespa

    PhoenixA --> Phoenix
    PhoenixB --> Phoenix
    PhoenixC --> Phoenix

    MemoryA --> Mem0
    MemoryB --> Mem0
    MemoryC --> Mem0

    style TenantA fill:#e1f5ff
    style TenantB fill:#fff4e1
    style TenantC fill:#ffe1f5
    style Vespa fill:#e1ffe1
    style Phoenix fill:#f5e1ff
    style Mem0 fill:#ffe0e0
```

### Tenant Isolation Layers (11-Package Architecture)

```mermaid
graph TB
    subgraph "Application Layer"
        Request[HTTP Request<br/>tenant_id in header/body]
        Runtime[cogniverse_runtime]
    end

    subgraph "Foundation Layer"
        ConfigMgr[ConfigManager<br/>cogniverse_foundation]
        TenantConfig[UnifiedConfig<br/>per tenant]
        TelemetryBase[TelemetryManager<br/>cogniverse_foundation]
    end

    subgraph "Core Layer"
        Agent[Agent Context<br/>cogniverse_core]
        Memory[MemoryManager<br/>cogniverse_core]
    end

    subgraph "Implementation Layer"
        VespaBackend[VespaBackend<br/>cogniverse_vespa]
        AgentImpl[Agent Implementations<br/>cogniverse_agents]
    end

    subgraph "Storage Layer"
        VespaSchema[Vespa Schema<br/>schema_name_tenant_id]
        PhoenixProject[Phoenix Project<br/>tenant_project]
        MemoryStore[Memory Store<br/>user_id prefix]
    end

    Request --> Runtime
    Runtime --> ConfigMgr
    ConfigMgr --> TenantConfig
    TenantConfig --> Agent
    TenantConfig --> TelemetryBase
    TenantConfig --> Memory
    Agent --> AgentImpl
    AgentImpl --> VespaBackend
    VespaBackend --> VespaSchema
    TelemetryBase --> PhoenixProject
    Memory --> MemoryStore

    style Request fill:#B4A7D6,color:#fff
    style Runtime fill:#B4A7D6,color:#fff
    style ConfigMgr fill:#5BA3F5,color:#fff
    style TenantConfig fill:#5BA3F5,color:#fff
    style TelemetryBase fill:#5BA3F5,color:#fff
    style Agent fill:#FF6B9D,color:#fff
    style Memory fill:#FF6B9D,color:#fff
    style VespaBackend fill:#93C47D
    style AgentImpl fill:#FFD966
    style VespaSchema fill:#e1ffe1
    style PhoenixProject fill:#e1ffe1
    style MemoryStore fill:#e1ffe1
```

---

## Schema-Per-Tenant Pattern

### Vespa Schema Naming Convention

```mermaid
graph LR
    subgraph "Base Schemas"
        ColPali[video_colpali_smol500_mv_frame]
        VideoPrism[video_videoprism_base_mv_chunk_30s]
        ColQwen[video_colqwen_omni_mv_chunk_30s]
    end

    subgraph "Tenant A - acme_corp"
        ColPaliA[video_colpali_smol500_mv_frame_acme_corp]
        VideoPrismA[video_videoprism_base_mv_chunk_30s_acme_corp]
        ColQwenA[video_colqwen_omni_mv_chunk_30s_acme_corp]
    end

    subgraph "Tenant B - globex_inc"
        ColPaliB[video_colpali_smol500_mv_frame_globex_inc]
        VideoPrismB[video_videoprism_base_mv_chunk_30s_globex_inc]
        ColQwenB[video_colqwen_omni_mv_chunk_30s_globex_inc]
    end

    ColPali -->|+ _acme_corp| ColPaliA
    VideoPrism -->|+ _acme_corp| VideoPrismA
    ColQwen -->|+ _acme_corp| ColQwenA

    ColPali -->|+ _globex_inc| ColPaliB
    VideoPrism -->|+ _globex_inc| VideoPrismB
    ColQwen -->|+ _globex_inc| ColQwenB

    style ColPali fill:#e1f5ff
    style VideoPrism fill:#e1f5ff
    style ColQwen fill:#e1f5ff
    style ColPaliA fill:#fff4e1
    style VideoPrismA fill:#fff4e1
    style ColQwenA fill:#fff4e1
    style ColPaliB fill:#ffe1f5
    style VideoPrismB fill:#ffe1f5
    style ColQwenB fill:#ffe1f5
```

### Schema Deployment Flow (Multi-Tenant)

```mermaid
sequenceDiagram
    participant Script as deploy_all_schemas.py
    participant Parser as JsonSchemaParser
    participant Manager as VespaSchemaManager
    participant Vespa as Vespa Config Server

    Note over Script: Define tenants: [acme_corp, globex_inc, default]

    loop For each tenant
        Script->>Script: tenant_id = "acme_corp"

        loop For each schema file
            Script->>Parser: load_schema("video_colpali.json")
            Parser-->>Script: Schema object (base_name="video_colpali")

            Script->>Manager: deploy_schema(schema, tenant_id, suffix="_acme_corp")
            Manager->>Manager: tenant_schema = "video_colpali_acme_corp"
            Manager->>Manager: Create ApplicationPackage
            Manager->>Manager: Add tenant-suffixed schema

            Manager->>Vespa: POST /prepareandactivate
            Note over Manager,Vespa: Deploy video_colpali_acme_corp
            Vespa-->>Manager: 200 OK

            Manager-->>Script: Deployment success

            Script->>Script: Log: ✓ Deployed video_colpali_acme_corp
        end

        Script->>Script: Log: All schemas deployed for acme_corp
    end
```

### Schema Isolation in Vespa

```mermaid
graph TB
    subgraph "Vespa Instance"
        subgraph "Tenant: acme_corp"
            SchemaA1[video_colpali_acme_corp<br/>Documents: 1000]
            SchemaA2[video_videoprism_acme_corp<br/>Documents: 500]
        end

        subgraph "Tenant: globex_inc"
            SchemaB1[video_colpali_globex_inc<br/>Documents: 2000]
            SchemaB2[video_videoprism_globex_inc<br/>Documents: 800]
        end

        subgraph "Tenant: default"
            SchemaC1[video_colpali_default<br/>Documents: 300]
            SchemaC2[video_videoprism_default<br/>Documents: 150]
        end
    end

    QueryA[Query from acme_corp] -->|Targets| SchemaA1
    QueryA -->|Targets| SchemaA2
    QueryA -.->|❌ Cannot access| SchemaB1
    QueryA -.->|❌ Cannot access| SchemaC1

    QueryB[Query from globex_inc] -->|Targets| SchemaB1
    QueryB -->|Targets| SchemaB2
    QueryB -.->|❌ Cannot access| SchemaA1
    QueryB -.->|❌ Cannot access| SchemaC1

    style SchemaA1 fill:#e1f5ff
    style SchemaA2 fill:#e1f5ff
    style SchemaB1 fill:#fff4e1
    style SchemaB2 fill:#fff4e1
    style SchemaC1 fill:#ffe1f5
    style SchemaC2 fill:#ffe1f5
    style QueryA fill:#cce5ff
    style QueryB fill:#ffebcc
```

---

## Tenant Data Flow

### Video Ingestion Flow (Tenant-Specific - 11-Package Architecture)

```mermaid
sequenceDiagram
    participant User as User (acme_corp)
    participant Script as run_ingestion.py
    participant Foundation as cogniverse_foundation
    participant Core as cogniverse_core
    participant Agents as cogniverse_agents
    participant Vespa as cogniverse_vespa
    participant Schema as Vespa Schema

    User->>Script: Ingest video for tenant: acme_corp

    Script->>Foundation: UnifiedConfig(tenant_id="acme_corp")
    Foundation-->>Script: Config with tenant isolation

    Script->>Agents: VideoIngestionPipeline(config, profile="frame_based")
    Agents->>Core: Initialize agent context
    Core->>Foundation: Get TelemetryManager
    Foundation-->>Core: Telemetry manager
    Core-->>Agents: Context ready

    Agents->>Agents: Extract frames
    Agents->>Agents: Generate embeddings

    Agents->>Vespa: VespaBackend(config)
    Vespa->>Foundation: Get tenant_id from config
    Vespa->>Vespa: schema_name = "video_colpali_smol500_mv_frame_acme_corp"

    Agents->>Vespa: feed_documents(docs, schema_name)
    Vespa->>Schema: Insert into video_colpali_smol500_mv_frame_acme_corp
    Schema-->>Vespa: Documents inserted

    Vespa-->>Agents: Upload success
    Agents->>Foundation: Record telemetry span
    Agents-->>Script: Processing complete
    Script-->>User: Video ingested for acme_corp
```

### Search Flow (Tenant-Isolated - 11-Package Architecture)

```mermaid
sequenceDiagram
    participant User as User (acme_corp)
    participant Runtime as cogniverse_runtime
    participant Foundation as cogniverse_foundation
    participant Core as cogniverse_core
    participant Agents as cogniverse_agents
    participant Vespa as cogniverse_vespa
    participant Phoenix as Phoenix Telemetry

    User->>Runtime: POST /search {"query": "ML tutorial", "tenant_id": "acme_corp"}

    Runtime->>Foundation: UnifiedConfig(tenant_id="acme_corp")
    Foundation-->>Runtime: Tenant config

    Runtime->>Agents: VideoSearchAgent(config, profile="frame_based")
    Agents->>Core: Initialize agent context
    Core->>Foundation: Get TelemetryManager
    Foundation-->>Core: Telemetry manager
    Core-->>Agents: Context ready

    Agents->>Agents: Generate query embedding

    Agents->>Vespa: search(query, schema="video_colpali_smol500_mv_frame_acme_corp")
    Note over Vespa: Query only searches acme_corp schema
    Vespa-->>Agents: Search results (acme_corp documents only)

    Agents->>Agents: Multi-modal reranking

    Agents->>Foundation: Record span with tenant_id="acme_corp"
    Foundation->>Phoenix: Send to project: acme_corp_project
    Phoenix-->>Foundation: Span recorded

    Agents-->>Runtime: Reranked results
    Runtime-->>User: Search results (acme_corp data only)
```

### Cross-Tenant Isolation Verification

```mermaid
graph TB
    subgraph "Tenant A Request"
        RequestA[Query: tenant_id=acme_corp]
        ProcessA[Process with acme_corp config]
        SearchA[Search: video_*_acme_corp]
        ResultsA[Results: acme_corp data only]
    end

    subgraph "Tenant B Request"
        RequestB[Query: tenant_id=globex_inc]
        ProcessB[Process with globex_inc config]
        SearchB[Search: video_*_globex_inc]
        ResultsB[Results: globex_inc data only]
    end

    subgraph "Isolation Boundary"
        Firewall[Schema-level Isolation]
    end

    RequestA --> ProcessA
    ProcessA --> SearchA
    SearchA --> Firewall
    Firewall --> ResultsA

    RequestB --> ProcessB
    ProcessB --> SearchB
    SearchB --> Firewall
    Firewall --> ResultsB

    SearchA -.->|❌ BLOCKED| ResultsB
    SearchB -.->|❌ BLOCKED| ResultsA

    style RequestA fill:#e1f5ff
    style ProcessA fill:#e1f5ff
    style SearchA fill:#e1f5ff
    style ResultsA fill:#e1f5ff
    style RequestB fill:#fff4e1
    style ProcessB fill:#fff4e1
    style SearchB fill:#fff4e1
    style ResultsB fill:#fff4e1
    style Firewall fill:#ffcccc
```

---

## Phoenix Project Isolation

### Per-Tenant Phoenix Projects

```mermaid
graph TB
    subgraph "Phoenix Instance (Port 6006)"
        subgraph "Project: acme_corp_project"
            SpansA[Spans<br/>All acme_corp traces]
            ExperimentsA[Experiments<br/>acme_corp evaluations]
            DatasetsA[Datasets<br/>acme_corp queries]
        end

        subgraph "Project: globex_inc_project"
            SpansB[Spans<br/>All globex_inc traces]
            ExperimentsB[Experiments<br/>globex_inc evaluations]
            DatasetsB[Datasets<br/>globex_inc queries]
        end

        subgraph "Project: default_project"
            SpansC[Spans<br/>All default traces]
            ExperimentsC[Experiments<br/>default evaluations]
            DatasetsC[Datasets<br/>default queries]
        end
    end

    TenantA[Tenant: acme_corp] --> SpansA
    TenantA --> ExperimentsA
    TenantA --> DatasetsA

    TenantB[Tenant: globex_inc] --> SpansB
    TenantB --> ExperimentsB
    TenantB --> DatasetsB

    TenantC[Tenant: default] --> SpansC
    TenantC --> ExperimentsC
    TenantC --> DatasetsC

    style SpansA fill:#e1f5ff
    style ExperimentsA fill:#e1f5ff
    style DatasetsA fill:#e1f5ff
    style SpansB fill:#fff4e1
    style ExperimentsB fill:#fff4e1
    style DatasetsB fill:#fff4e1
    style SpansC fill:#ffe1f5
    style ExperimentsC fill:#ffe1f5
    style DatasetsC fill:#ffe1f5
```

### Telemetry Flow (Per-Tenant Phoenix Projects)

```mermaid
sequenceDiagram
    participant AgentA as Agent (acme_corp)
    participant TelemetryA as TelemetryManager<br/>(acme_corp)
    participant Phoenix as Phoenix Collector<br/>(Port 4317)
    participant ProjectA as acme_corp_project
    participant UI as Phoenix UI<br/>(Port 6006)

    AgentA->>TelemetryA: start_span("search")
    TelemetryA->>TelemetryA: Attach attributes:<br/>tenant_id=acme_corp
    TelemetryA->>TelemetryA: Set project: acme_corp_project

    AgentA->>AgentA: Execute search operation

    AgentA->>TelemetryA: end_span(status="success")
    TelemetryA->>Phoenix: Export span via OTLP<br/>project=acme_corp_project
    Phoenix->>ProjectA: Store span in acme_corp_project

    Note over ProjectA: Span visible ONLY in acme_corp_project

    ProjectA-->>UI: Spans for acme_corp_project
    UI-->>User: View acme_corp traces<br/>(no cross-tenant visibility)
```

### Phoenix UI Access Pattern

```mermaid
graph LR
    subgraph "Phoenix UI"
        Dashboard[Phoenix Dashboard<br/>localhost:6006]
    end

    subgraph "Project Selection"
        DropDown[Project Dropdown]
        ProjectA[acme_corp_project]
        ProjectB[globex_inc_project]
        ProjectC[default_project]
    end

    subgraph "Tenant A View"
        ViewA[Spans: acme_corp only<br/>Experiments: acme_corp only<br/>Datasets: acme_corp only]
    end

    subgraph "Tenant B View"
        ViewB[Spans: globex_inc only<br/>Experiments: globex_inc only<br/>Datasets: globex_inc only]
    end

    Dashboard --> DropDown
    DropDown --> ProjectA
    DropDown --> ProjectB
    DropDown --> ProjectC

    ProjectA --> ViewA
    ProjectB --> ViewB

    ProjectA -.->|❌ Cannot see| ViewB
    ProjectB -.->|❌ Cannot see| ViewA

    style Dashboard fill:#e1f5ff
    style ViewA fill:#fff4e1
    style ViewB fill:#ffe1f5
```

---

## Memory Isolation

### Mem0 Memory Isolation with User ID Prefix

```mermaid
graph TB
    subgraph "Mem0 Memory Store (Vespa Backend)"
        subgraph "Tenant A Memories"
            MemA1[user_id: acme_corp_user1<br/>Conversation history]
            MemA2[user_id: acme_corp_user2<br/>Preferences]
        end

        subgraph "Tenant B Memories"
            MemB1[user_id: globex_inc_user1<br/>Conversation history]
            MemB2[user_id: globex_inc_admin<br/>Preferences]
        end

        subgraph "Default Memories"
            MemC1[user_id: default_user1<br/>Conversation history]
        end
    end

    AgentA[Agent: acme_corp] -->|Search memories| MemA1
    AgentA -->|Search memories| MemA2
    AgentA -.->|❌ Cannot access| MemB1

    AgentB[Agent: globex_inc] -->|Search memories| MemB1
    AgentB -->|Search memories| MemB2
    AgentB -.->|❌ Cannot access| MemA1

    style MemA1 fill:#e1f5ff
    style MemA2 fill:#e1f5ff
    style MemB1 fill:#fff4e1
    style MemB2 fill:#fff4e1
    style MemC1 fill:#ffe1f5
```

### Memory Manager Flow (Tenant-Aware)

```mermaid
sequenceDiagram
    participant Agent as Agent (acme_corp)
    participant Memory as Mem0MemoryManager
    participant Vespa as Vespa Backend
    participant Schema as agent_memories_acme_corp

    Agent->>Memory: search_memories(user_id="acme_corp_user1", query="previous conversations")

    Memory->>Memory: Construct user_id: acme_corp_user1
    Memory->>Memory: Schema: agent_memories_acme_corp

    Memory->>Vespa: search(schema="agent_memories_acme_corp", filter=user_id)
    Vespa->>Schema: Query with user_id filter
    Schema-->>Vespa: Memories for acme_corp_user1 only

    Vespa-->>Memory: Search results
    Memory-->>Agent: Memories (tenant-isolated)

    Note over Agent,Schema: User cannot access memories from other tenants
```

### Memory Schema Naming (Per-Tenant)

```mermaid
graph LR
    subgraph "Base Memory Schema"
        BaseSchema[agent_memories]
    end

    subgraph "Tenant-Specific Memory Schemas"
        SchemaA[agent_memories_acme_corp]
        SchemaB[agent_memories_globex_inc]
        SchemaC[agent_memories_default]
    end

    BaseSchema -->|+ _acme_corp| SchemaA
    BaseSchema -->|+ _globex_inc| SchemaB
    BaseSchema -->|+ _default| SchemaC

    SchemaA --> DocA[Documents:<br/>user_id prefix: acme_corp_*]
    SchemaB --> DocB[Documents:<br/>user_id prefix: globex_inc_*]
    SchemaC --> DocC[Documents:<br/>user_id prefix: default_*]

    style BaseSchema fill:#e1f5ff
    style SchemaA fill:#fff4e1
    style SchemaB fill:#ffe1f5
    style SchemaC fill:#e1ffe1
```

---

## Deployment Patterns

### Single Vespa Instance Multi-Tenant Deployment

```mermaid
graph TB
    subgraph "Infrastructure"
        Vespa[Vespa Instance<br/>Single deployment]
        Phoenix[Phoenix Instance<br/>Single deployment]
        Mem0Backend[Mem0 Vespa Backend<br/>Shared]
    end

    subgraph "Tenant A - acme_corp"
        AppA[Application: acme_corp]
        ConfigA[Config: acme_corp]
        SchemasA[Schemas: *_acme_corp]
        ProjectA[Phoenix: acme_corp_project]
        MemoryA[Memory: user_id=acme_corp_*]
    end

    subgraph "Tenant B - globex_inc"
        AppB[Application: globex_inc]
        ConfigB[Config: globex_inc]
        SchemasB[Schemas: *_globex_inc]
        ProjectB[Phoenix: globex_inc_project]
        MemoryB[Memory: user_id=globex_inc_*]
    end

    AppA --> ConfigA
    ConfigA --> SchemasA
    ConfigA --> ProjectA
    ConfigA --> MemoryA

    AppB --> ConfigB
    ConfigB --> SchemasB
    ConfigB --> ProjectB
    ConfigB --> MemoryB

    SchemasA --> Vespa
    SchemasB --> Vespa
    ProjectA --> Phoenix
    ProjectB --> Phoenix
    MemoryA --> Mem0Backend
    MemoryB --> Mem0Backend

    style Vespa fill:#e1ffe1
    style Phoenix fill:#f5e1ff
    style Mem0Backend fill:#ffe0e0
    style AppA fill:#e1f5ff
    style AppB fill:#fff4e1
```

### Tenant Lifecycle Management

```mermaid
sequenceDiagram
    participant Admin as Admin
    participant TenantMgr as TenantManager
    participant ConfigStore as Config Store
    participant SchemaManager as VespaSchemaManager
    participant Phoenix as Phoenix API
    participant Mem0 as Mem0

    Note over Admin: Create new tenant: "new_corp"

    Admin->>TenantMgr: create_tenant("new_corp")

    TenantMgr->>ConfigStore: Store SystemConfig(tenant_id="new_corp")
    ConfigStore-->>TenantMgr: Config saved

    TenantMgr->>SchemaManager: deploy_schemas(tenant_id="new_corp")
    loop For each schema
        SchemaManager->>SchemaManager: Create schema: video_*_new_corp
        SchemaManager->>SchemaManager: Deploy to Vespa
    end
    SchemaManager-->>TenantMgr: Schemas deployed

    TenantMgr->>Phoenix: create_project("new_corp_project")
    Phoenix-->>TenantMgr: Project created

    TenantMgr->>Mem0: Deploy memory schema: agent_memories_new_corp
    Mem0-->>TenantMgr: Memory schema ready

    TenantMgr-->>Admin: ✅ Tenant "new_corp" created successfully

    Note over Admin,Mem0: Tenant can now ingest videos, search, and use memory
```

### Tenant Deletion/Cleanup Flow

```mermaid
sequenceDiagram
    participant Admin as Admin
    participant TenantMgr as TenantManager
    participant SchemaManager as VespaSchemaManager
    participant Phoenix as Phoenix API
    participant Backup as Backup Service

    Note over Admin: Delete tenant: "old_corp"

    Admin->>TenantMgr: delete_tenant("old_corp")

    TenantMgr->>Backup: backup_tenant_data("old_corp")
    Backup->>Backup: Export all schemas data
    Backup->>Backup: Export Phoenix traces
    Backup->>Backup: Export memories
    Backup-->>TenantMgr: Backup complete (old_corp_backup_20251015.tar.gz)

    TenantMgr->>SchemaManager: delete_schemas(tenant_id="old_corp")
    loop For each schema
        SchemaManager->>SchemaManager: Delete schema: video_*_old_corp
        SchemaManager->>SchemaManager: Delete all documents
    end
    SchemaManager-->>TenantMgr: Schemas deleted

    TenantMgr->>Phoenix: delete_project("old_corp_project")
    Phoenix-->>TenantMgr: Project deleted

    TenantMgr->>TenantMgr: Remove tenant config

    TenantMgr-->>Admin: ✅ Tenant "old_corp" deleted<br/>Backup: old_corp_backup_20251015.tar.gz
```

### Multi-Region Deployment (Future)

```mermaid
graph TB
    subgraph "US Region"
        VespaUS[Vespa US]
        PhoenixUS[Phoenix US]

        subgraph "US Tenants"
            TenantUS1[acme_corp_us]
            TenantUS2[globex_us]
        end

        TenantUS1 --> VespaUS
        TenantUS1 --> PhoenixUS
        TenantUS2 --> VespaUS
        TenantUS2 --> PhoenixUS
    end

    subgraph "EU Region"
        VespaEU[Vespa EU]
        PhoenixEU[Phoenix EU]

        subgraph "EU Tenants"
            TenantEU1[acme_corp_eu]
            TenantEU2[globex_eu]
        end

        TenantEU1 --> VespaEU
        TenantEU1 --> PhoenixEU
        TenantEU2 --> VespaEU
        TenantEU2 --> PhoenixEU
    end

    LoadBalancer[Global Load Balancer<br/>Route by tenant region]
    LoadBalancer --> VespaUS
    LoadBalancer --> VespaEU

    style VespaUS fill:#e1f5ff
    style VespaEU fill:#e1f5ff
    style PhoenixUS fill:#fff4e1
    style PhoenixEU fill:#fff4e1
    style LoadBalancer fill:#ffe1f5
```

---

## Summary

This diagram collection provides comprehensive visual documentation of multi-tenant architecture across the **11-package layered structure**:

1. **Tenant Isolation**: Complete separation at schema, project, and memory levels across all layers
2. **Schema-Per-Tenant**: Naming convention with `_tenant_id` suffix managed by cogniverse-vespa
3. **Data Flow**: Tenant-specific routing from ingestion to search through layered architecture
4. **Phoenix Projects**: Per-tenant observability via cogniverse-telemetry-phoenix plugin
5. **Memory Isolation**: User ID prefixes and tenant-specific schemas via cogniverse-core
6. **Lifecycle Management**: Tenant creation, deletion, and backup workflows

**Key Principles:**
- **Schema Isolation**: Each tenant has dedicated Vespa schemas (Implementation Layer)
- **Project Isolation**: Each tenant has dedicated Phoenix project (Core Layer Plugin)
- **Memory Isolation**: User IDs prefixed with tenant_id (Core Layer)
- **No Cross-Tenant Access**: Firewall at every layer of the 11-package architecture
- **Shared Infrastructure**: Single Vespa/Phoenix instances serve all tenants
- **Configuration-Driven**: Tenant isolation configured via cogniverse-foundation

**Tenant Naming Conventions:**
- Vespa schemas: `{base_schema}_{tenant_id}` (cogniverse-vespa)
- Phoenix projects: `{tenant_id}_project` (cogniverse-telemetry-phoenix)
- Memory user IDs: `{tenant_id}_{user_id}` (cogniverse-core)

**11-Package Architecture Integration:**
- **Foundation Layer**: Provides UnifiedConfig with tenant_id, TelemetryManager base
- **Core Layer**: Manages agent context, memory, and cache with tenant isolation
- **Implementation Layer**: Vespa backend applies tenant suffixes, agents enforce isolation
- **Application Layer**: Runtime and dashboard respect tenant boundaries

**Related Documentation:**
- [11-Package Architecture Guide](../architecture/overview.md)
- [Multi-Tenant Architecture](../architecture/multi-tenant.md)
- [Multi-Tenant Operations](../operations/multi-tenant-ops.md)
- [Configuration Guide](../operations/configuration.md)
