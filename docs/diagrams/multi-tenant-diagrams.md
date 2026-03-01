# Multi-Tenant Architecture Diagrams
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
flowchart TB
    subgraph "Tenant A - acme_corp"
        TenantA["<span style='color:#000'>Tenant: acme_corp</span>"]
        ConfigA["<span style='color:#000'>SystemConfig<br/>tenant_id=acme_corp</span>"]
        SchemaA["<span style='color:#000'>Vespa Schemas<br/>video_*_acme_corp</span>"]
        PhoenixA["<span style='color:#000'>Phoenix Project<br/>cogniverse-acme_corp-video-search</span>"]
        MemoryA["<span style='color:#000'>Mem0 Memory<br/>user_id=acme_corp_*</span>"]
    end

    subgraph "Tenant B - globex_inc"
        TenantB["<span style='color:#000'>Tenant: globex_inc</span>"]
        ConfigB["<span style='color:#000'>SystemConfig<br/>tenant_id=globex_inc</span>"]
        SchemaB["<span style='color:#000'>Vespa Schemas<br/>video_*_globex_inc</span>"]
        PhoenixB["<span style='color:#000'>Phoenix Project<br/>cogniverse-globex_inc-video-search</span>"]
        MemoryB["<span style='color:#000'>Mem0 Memory<br/>user_id=globex_inc_*</span>"]
    end

    subgraph "Tenant C - default"
        TenantC["<span style='color:#000'>Tenant: default</span>"]
        ConfigC["<span style='color:#000'>SystemConfig<br/>tenant_id=default</span>"]
        SchemaC["<span style='color:#000'>Vespa Schemas<br/>video_*_default</span>"]
        PhoenixC["<span style='color:#000'>Phoenix Project<br/>cogniverse-default-video-search</span>"]
        MemoryC["<span style='color:#000'>Mem0 Memory<br/>user_id=default_*</span>"]
    end

    subgraph "Shared Infrastructure"
        Vespa["<span style='color:#000'>Vespa Instance<br/>Port 8080, 19071</span>"]
        Phoenix["<span style='color:#000'>Phoenix Instance<br/>Port 6006, 4317</span>"]
        Mem0["<span style='color:#000'>Mem0 Backend<br/>Vespa Schema</span>"]
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

    style TenantA fill:#90caf9,stroke:#1565c0,color:#000
    style ConfigA fill:#b0bec5,stroke:#546e7a,color:#000
    style SchemaA fill:#90caf9,stroke:#1565c0,color:#000
    style PhoenixA fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryA fill:#90caf9,stroke:#1565c0,color:#000
    style TenantB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ConfigB fill:#b0bec5,stroke:#546e7a,color:#000
    style SchemaB fill:#ffcc80,stroke:#ef6c00,color:#000
    style PhoenixB fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryB fill:#ffcc80,stroke:#ef6c00,color:#000
    style TenantC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ConfigC fill:#b0bec5,stroke:#546e7a,color:#000
    style SchemaC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PhoenixC fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Vespa fill:#a5d6a7,stroke:#388e3c,color:#000
    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
    style Mem0 fill:#90caf9,stroke:#1565c0,color:#000
```

### Tenant Isolation Layers (Layered Architecture)

```mermaid
flowchart TB
    subgraph "Application Layer"
        Request["<span style='color:#000'>HTTP Request<br/>tenant_id in header/body</span>"]
        Runtime["<span style='color:#000'>cogniverse_runtime</span>"]
    end

    subgraph "Foundation Layer"
        ConfigMgr["<span style='color:#000'>ConfigManager<br/>cogniverse_foundation</span>"]
        TenantConfig["<span style='color:#000'>Tenant Configuration<br/>(SystemConfig + other configs)</span>"]
        TelemetryBase["<span style='color:#000'>TelemetryManager<br/>cogniverse_foundation</span>"]
    end

    subgraph "Core Layer"
        Agent["<span style='color:#000'>Agent Context<br/>cogniverse_core</span>"]
        Memory["<span style='color:#000'>Mem0MemoryManager<br/>cogniverse_core</span>"]
    end

    subgraph "Implementation Layer"
        VespaBackend["<span style='color:#000'>VespaBackend<br/>cogniverse_vespa</span>"]
        AgentImpl["<span style='color:#000'>Agent Implementations<br/>cogniverse_agents</span>"]
    end

    subgraph "Storage Layer"
        VespaSchema["<span style='color:#000'>Vespa Schema<br/>schema_name_tenant_id</span>"]
        PhoenixProject["<span style='color:#000'>Phoenix Project<br/>cogniverse-{tenant_id}-{service}</span>"]
        MemoryStore["<span style='color:#000'>Memory Store<br/>user_id prefix</span>"]
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

    style Request fill:#90caf9,stroke:#1565c0,color:#000
    style Runtime fill:#90caf9,stroke:#1565c0,color:#000
    style ConfigMgr fill:#a5d6a7,stroke:#388e3c,color:#000
    style TenantConfig fill:#b0bec5,stroke:#546e7a,color:#000
    style TelemetryBase fill:#a5d6a7,stroke:#388e3c,color:#000
    style Agent fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Memory fill:#90caf9,stroke:#1565c0,color:#000
    style VespaBackend fill:#90caf9,stroke:#1565c0,color:#000
    style AgentImpl fill:#ce93d8,stroke:#7b1fa2,color:#000
    style VespaSchema fill:#90caf9,stroke:#1565c0,color:#000
    style PhoenixProject fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryStore fill:#90caf9,stroke:#1565c0,color:#000
```

---

## Schema-Per-Tenant Pattern

### Vespa Schema Naming Convention

```mermaid
flowchart LR
    subgraph "Base Schemas"
        ColPali["<span style='color:#000'>video_colpali_smol500_mv_frame</span>"]
        VideoPrism["<span style='color:#000'>video_videoprism_base_mv_chunk_30s</span>"]
        ColQwen["<span style='color:#000'>video_colqwen_omni_mv_chunk_30s</span>"]
    end

    subgraph "Tenant A - acme_corp"
        ColPaliA["<span style='color:#000'>video_colpali_smol500_mv_frame_acme_corp</span>"]
        VideoPrismA["<span style='color:#000'>video_videoprism_base_mv_chunk_30s_acme_corp</span>"]
        ColQwenA["<span style='color:#000'>video_colqwen_omni_mv_chunk_30s_acme_corp</span>"]
    end

    subgraph "Tenant B - globex_inc"
        ColPaliB["<span style='color:#000'>video_colpali_smol500_mv_frame_globex_inc</span>"]
        VideoPrismB["<span style='color:#000'>video_videoprism_base_mv_chunk_30s_globex_inc</span>"]
        ColQwenB["<span style='color:#000'>video_colqwen_omni_mv_chunk_30s_globex_inc</span>"]
    end

    ColPali -->|+ _acme_corp| ColPaliA
    VideoPrism -->|+ _acme_corp| VideoPrismA
    ColQwen -->|+ _acme_corp| ColQwenA

    ColPali -->|+ _globex_inc| ColPaliB
    VideoPrism -->|+ _globex_inc| VideoPrismB
    ColQwen -->|+ _globex_inc| ColQwenB

    style ColPali fill:#90caf9,stroke:#1565c0,color:#000
    style VideoPrism fill:#90caf9,stroke:#1565c0,color:#000
    style ColQwen fill:#90caf9,stroke:#1565c0,color:#000
    style ColPaliA fill:#ffcc80,stroke:#ef6c00,color:#000
    style VideoPrismA fill:#ffcc80,stroke:#ef6c00,color:#000
    style ColQwenA fill:#ffcc80,stroke:#ef6c00,color:#000
    style ColPaliB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style VideoPrismB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ColQwenB fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### Schema Deployment Flow (Multi-Tenant)

**Primary Path: Tenant Provisioning**

When an admin creates a tenant via `POST /admin/tenants`, schemas are automatically deployed:

```mermaid
sequenceDiagram
    participant Admin as Admin
    participant API as POST /admin/tenants
    participant TenantMgr as tenant_manager.py
    participant Registry as schema_registry
    participant Vespa as Vespa Config Server

    Admin->>API: POST /admin/tenants<br/>{"tenant_id": "acme:production", "base_schemas": [...]}

    API->>TenantMgr: create_tenant(request)
    TenantMgr->>TenantMgr: Parse tenant_id → org_id="acme", tenant_name="production"
    TenantMgr->>TenantMgr: Auto-create org if not exists

    loop For each base_schema in request.base_schemas
        TenantMgr->>Registry: deploy_schema(tenant_id="acme:production", base_schema_name)
        Registry->>Registry: tenant_schema = "video_colpali_smol500_mv_frame_acme_production"
        Registry->>Vespa: Deploy tenant-specific schema
        Vespa-->>Registry: Schema deployed
        Registry-->>TenantMgr: Success
    end

    TenantMgr->>TenantMgr: Store tenant metadata with schemas_deployed list
    TenantMgr-->>API: Tenant created with deployed schemas
    API-->>Admin: {"tenant_full_id": "acme:production", "schemas_deployed": [...]}
```

**Helper Script: deploy_all_schemas.py**

The script supports both base schema deployment and tenant-specific deployment:

```mermaid
sequenceDiagram
    participant Admin as Admin
    participant Script as deploy_all_schemas.py
    participant Registry as SchemaRegistry
    participant Vespa as Vespa Config Server

    alt No --tenant-id (Base Schema Mode)
        Admin->>Script: python deploy_all_schemas.py
        Script->>Script: Load all configs/schemas/*.json
        Script->>Vespa: Deploy base schema templates
        Vespa-->>Script: Base schemas deployed
    else With --tenant-id (Tenant Mode)
        Admin->>Script: python deploy_all_schemas.py --tenant-id acme:prod
        loop For each base schema
            Script->>Registry: deploy_schema(tenant_id, base_schema_name)
            Registry->>Vespa: Deploy tenant-specific schema
            Vespa-->>Registry: Schema deployed
        end
        Script-->>Admin: Tenant schemas deployed
    end
```

### Schema Isolation in Vespa

```mermaid
flowchart TB
    subgraph "Vespa Instance"
        subgraph "Tenant: acme_corp"
            SchemaA1["<span style='color:#000'>video_colpali_smol500_mv_frame_acme_corp<br/>Documents: 1000</span>"]
            SchemaA2["<span style='color:#000'>video_videoprism_base_mv_chunk_30s_acme_corp<br/>Documents: 500</span>"]
        end

        subgraph "Tenant: globex_inc"
            SchemaB1["<span style='color:#000'>video_colpali_smol500_mv_frame_globex_inc<br/>Documents: 2000</span>"]
            SchemaB2["<span style='color:#000'>video_videoprism_base_mv_chunk_30s_globex_inc<br/>Documents: 800</span>"]
        end

        subgraph "Tenant: default"
            SchemaC1["<span style='color:#000'>video_colpali_smol500_mv_frame_default<br/>Documents: 300</span>"]
            SchemaC2["<span style='color:#000'>video_videoprism_base_mv_chunk_30s_default<br/>Documents: 150</span>"]
        end
    end

    QueryA["<span style='color:#000'>Query from acme_corp</span>"] -->|Targets| SchemaA1
    QueryA -->|Targets| SchemaA2
    QueryA -.->|❌ Cannot access| SchemaB1
    QueryA -.->|❌ Cannot access| SchemaC1

    QueryB["<span style='color:#000'>Query from globex_inc</span>"] -->|Targets| SchemaB1
    QueryB -->|Targets| SchemaB2
    QueryB -.->|❌ Cannot access| SchemaA1
    QueryB -.->|❌ Cannot access| SchemaC1

    style SchemaA1 fill:#90caf9,stroke:#1565c0,color:#000
    style SchemaA2 fill:#90caf9,stroke:#1565c0,color:#000
    style SchemaB1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style SchemaB2 fill:#ffcc80,stroke:#ef6c00,color:#000
    style SchemaC1 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style SchemaC2 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style QueryA fill:#90caf9,stroke:#1565c0,color:#000
    style QueryB fill:#ffcc80,stroke:#ef6c00,color:#000
```

---

## Tenant Data Flow

### Video Ingestion Flow (Tenant-Specific - Layered Architecture)

```mermaid
sequenceDiagram
    participant User as User (acme_corp)
    participant Script as run_ingestion.py
    participant Foundation as cogniverse_foundation
    participant Builder as VideoIngestionPipelineBuilder
    participant Pipeline as VideoIngestionPipeline
    participant Registry as BackendRegistry
    participant Vespa as VespaBackend

    User->>Script: Run with --tenant-id acme_corp --profile video_colpali_smol500_mv_frame

    Script->>Foundation: create_default_config_manager()
    Foundation-->>Script: config_manager

    Script->>Foundation: get_config(tenant_id="default", config_manager)
    Foundation-->>Script: app_config

    Script->>Builder: build_simple_pipeline(tenant_id, video_dir, schema, backend)
    Builder->>Builder: Validate tenant_id and config_manager
    Builder->>Builder: Create PipelineConfig
    Builder->>Pipeline: Initialize VideoIngestionPipeline
    Pipeline->>Registry: get_ingestion_backend("vespa", tenant_id, config)
    Registry->>Vespa: Create VespaBackend instance
    Vespa->>Vespa: Apply tenant suffix to schema_name
    Registry-->>Pipeline: Backend instance
    Builder-->>Script: Configured pipeline

    Script->>Pipeline: process_videos_concurrent(video_files, max_concurrent)
    Pipeline->>Pipeline: Extract frames
    Pipeline->>Pipeline: Generate embeddings
    Pipeline->>Vespa: ingest_documents(docs, tenant_schema)
    Vespa->>Vespa: Insert into video_colpali_smol500_mv_frame_acme_corp
    Vespa-->>Pipeline: Ingestion success

    Pipeline-->>Script: Processing results
    Script-->>User: Video ingested for acme_corp
```

### Search Flow (Tenant-Isolated - Layered Architecture)

```mermaid
sequenceDiagram
    participant User as User (acme_corp)
    participant Runtime as cogniverse_runtime
    participant Foundation as cogniverse_foundation
    participant Agent as VideoSearchAgent
    participant Backend as VespaBackend
    participant Telemetry as TelemetryManager

    User->>Runtime: POST /search {"query": "ML tutorial", "tenant_id": "acme_corp"}

    Runtime->>Foundation: get_config(tenant_id="acme_corp")
    Foundation-->>Runtime: SystemConfig with tenant_id

    Runtime->>Agent: create_video_search_agent(config, tenant_id)
    Agent->>Backend: get_search_backend(schema_name)
    Backend-->>Agent: Shared search backend
    Agent->>Backend: search(query_dict with tenant_id)
    Backend->>Backend: Apply tenant suffix: video_colpali_smol500_mv_frame_acme_corp

    Agent->>Agent: Generate query embedding

    Agent->>Backend: search(query, tenant_schema)
    Note over Backend: Query targets acme_corp schema only
    Backend-->>Agent: Search results (acme_corp documents only)

    Agent->>Agent: Rerank results

    Note over Agent,Telemetry: Telemetry spans recorded automatically via context manager

    Agent-->>Runtime: Reranked results
    Runtime-->>User: Search results (acme_corp data only)
```

### Cross-Tenant Isolation Verification

```mermaid
flowchart TB
    subgraph "Tenant A Request"
        RequestA["<span style='color:#000'>Query: tenant_id=acme_corp</span>"]
        ProcessA["<span style='color:#000'>Process with acme_corp config</span>"]
        SearchA["<span style='color:#000'>Search: video_*_acme_corp</span>"]
        ResultsA["<span style='color:#000'>Results: acme_corp data only</span>"]
    end

    subgraph "Tenant B Request"
        RequestB["<span style='color:#000'>Query: tenant_id=globex_inc</span>"]
        ProcessB["<span style='color:#000'>Process with globex_inc config</span>"]
        SearchB["<span style='color:#000'>Search: video_*_globex_inc</span>"]
        ResultsB["<span style='color:#000'>Results: globex_inc data only</span>"]
    end

    subgraph "Isolation Boundary"
        Firewall["<span style='color:#000'>Schema-level Isolation</span>"]
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

    style RequestA fill:#90caf9,stroke:#1565c0,color:#000
    style ProcessA fill:#90caf9,stroke:#1565c0,color:#000
    style SearchA fill:#90caf9,stroke:#1565c0,color:#000
    style ResultsA fill:#90caf9,stroke:#1565c0,color:#000
    style RequestB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ProcessB fill:#ffcc80,stroke:#ef6c00,color:#000
    style SearchB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ResultsB fill:#ffcc80,stroke:#ef6c00,color:#000
    style Firewall fill:#ce93d8,stroke:#7b1fa2,color:#000
```

---

## Phoenix Project Isolation

### Per-Tenant Phoenix Projects

```mermaid
flowchart TB
    subgraph "Phoenix Instance (Port 6006)"
        subgraph "Project: cogniverse-acme_corp-video-search"
            SpansA["<span style='color:#000'>Spans<br/>All acme_corp traces</span>"]
            ExperimentsA["<span style='color:#000'>Experiments<br/>acme_corp evaluations</span>"]
            DatasetsA["<span style='color:#000'>Datasets<br/>acme_corp queries</span>"]
        end

        subgraph "Project: cogniverse-globex_inc-video-search"
            SpansB["<span style='color:#000'>Spans<br/>All globex_inc traces</span>"]
            ExperimentsB["<span style='color:#000'>Experiments<br/>globex_inc evaluations</span>"]
            DatasetsB["<span style='color:#000'>Datasets<br/>globex_inc queries</span>"]
        end

        subgraph "Project: cogniverse-default-video-search"
            SpansC["<span style='color:#000'>Spans<br/>All default traces</span>"]
            ExperimentsC["<span style='color:#000'>Experiments<br/>default evaluations</span>"]
            DatasetsC["<span style='color:#000'>Datasets<br/>default queries</span>"]
        end
    end

    TenantA["<span style='color:#000'>Tenant: acme_corp</span>"] --> SpansA
    TenantA --> ExperimentsA
    TenantA --> DatasetsA

    TenantB["<span style='color:#000'>Tenant: globex_inc</span>"] --> SpansB
    TenantB --> ExperimentsB
    TenantB --> DatasetsB

    TenantC["<span style='color:#000'>Tenant: default</span>"] --> SpansC
    TenantC --> ExperimentsC
    TenantC --> DatasetsC

    style TenantA fill:#90caf9,stroke:#1565c0,color:#000
    style TenantB fill:#ffcc80,stroke:#ef6c00,color:#000
    style TenantC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style SpansA fill:#90caf9,stroke:#1565c0,color:#000
    style ExperimentsA fill:#90caf9,stroke:#1565c0,color:#000
    style DatasetsA fill:#90caf9,stroke:#1565c0,color:#000
    style SpansB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ExperimentsB fill:#ffcc80,stroke:#ef6c00,color:#000
    style DatasetsB fill:#ffcc80,stroke:#ef6c00,color:#000
    style SpansC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ExperimentsC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style DatasetsC fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### Telemetry Flow (Per-Tenant Phoenix Projects)

```mermaid
sequenceDiagram
    participant AgentA as Agent (acme_corp)
    participant TelemetryA as TelemetryManager<br/>(acme_corp)
    participant Phoenix as Phoenix Collector<br/>(Port 4317)
    participant ProjectA as cogniverse-acme_corp-video-search
    participant UI as Phoenix UI<br/>(Port 6006)

    AgentA->>TelemetryA: with span("search", tenant_id="acme_corp")
    TelemetryA->>TelemetryA: Attach attributes:<br/>tenant_id=acme_corp
    TelemetryA->>TelemetryA: Set project: cogniverse-acme_corp-video-search

    AgentA->>AgentA: Execute search operation

    TelemetryA->>TelemetryA: Span context ends
    TelemetryA->>Phoenix: Export span via OTLP<br/>project=cogniverse-acme_corp-video-search
    Phoenix->>ProjectA: Store span in cogniverse-acme_corp-video-search

    Note over ProjectA: Span visible ONLY in cogniverse-acme_corp-video-search

    ProjectA-->>UI: Spans for cogniverse-acme_corp-video-search
    UI-->>User: View acme_corp traces<br/>(no cross-tenant visibility)
```

### Phoenix UI Access Pattern

```mermaid
flowchart LR
    subgraph "Phoenix UI"
        Dashboard["<span style='color:#000'>Phoenix Dashboard<br/>localhost:6006</span>"]
    end

    subgraph "Project Selection"
        DropDown["<span style='color:#000'>Project Dropdown</span>"]
        ProjectA["<span style='color:#000'>cogniverse-acme_corp-video-search</span>"]
        ProjectB["<span style='color:#000'>cogniverse-globex_inc-video-search</span>"]
        ProjectC["<span style='color:#000'>cogniverse-default-video-search</span>"]
    end

    subgraph "Tenant A View"
        ViewA["<span style='color:#000'>Spans: acme_corp only<br/>Experiments: acme_corp only<br/>Datasets: acme_corp only</span>"]
    end

    subgraph "Tenant B View"
        ViewB["<span style='color:#000'>Spans: globex_inc only<br/>Experiments: globex_inc only<br/>Datasets: globex_inc only</span>"]
    end

    Dashboard --> DropDown
    DropDown --> ProjectA
    DropDown --> ProjectB
    DropDown --> ProjectC

    ProjectA --> ViewA
    ProjectB --> ViewB

    ProjectA -.->|❌ Cannot see| ViewB
    ProjectB -.->|❌ Cannot see| ViewA

    style Dashboard fill:#90caf9,stroke:#1565c0,color:#000
    style DropDown fill:#b0bec5,stroke:#546e7a,color:#000
    style ProjectA fill:#90caf9,stroke:#1565c0,color:#000
    style ProjectB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ProjectC fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ViewA fill:#90caf9,stroke:#1565c0,color:#000
    style ViewB fill:#ffcc80,stroke:#ef6c00,color:#000
```

---

## Memory Isolation

### Mem0 Memory Isolation with User ID Prefix

```mermaid
flowchart TB
    subgraph "Mem0 Memory Store (Vespa Backend)"
        subgraph "Tenant A Memories"
            MemA1["<span style='color:#000'>user_id: acme_corp_user1<br/>Conversation history</span>"]
            MemA2["<span style='color:#000'>user_id: acme_corp_user2<br/>Preferences</span>"]
        end

        subgraph "Tenant B Memories"
            MemB1["<span style='color:#000'>user_id: globex_inc_user1<br/>Conversation history</span>"]
            MemB2["<span style='color:#000'>user_id: globex_inc_admin<br/>Preferences</span>"]
        end

        subgraph "Default Memories"
            MemC1["<span style='color:#000'>user_id: default_user1<br/>Conversation history</span>"]
        end
    end

    AgentA["<span style='color:#000'>Agent: acme_corp</span>"] -->|Search memories| MemA1
    AgentA -->|Search memories| MemA2
    AgentA -.->|❌ Cannot access| MemB1

    AgentB["<span style='color:#000'>Agent: globex_inc</span>"] -->|Search memories| MemB1
    AgentB -->|Search memories| MemB2
    AgentB -.->|❌ Cannot access| MemA1

    style AgentA fill:#ce93d8,stroke:#7b1fa2,color:#000
    style AgentB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style MemA1 fill:#90caf9,stroke:#1565c0,color:#000
    style MemA2 fill:#90caf9,stroke:#1565c0,color:#000
    style MemB1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style MemB2 fill:#ffcc80,stroke:#ef6c00,color:#000
    style MemC1 fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### Memory Manager Flow (Tenant-Aware)

```mermaid
sequenceDiagram
    participant Agent as Agent (acme_corp)
    participant Memory as Mem0MemoryManager
    participant Mem0 as Mem0 Library
    participant Backend as BackendVectorStore
    participant Schema as agent_memories_acme_corp

    Agent->>Memory: search_memory(query="previous conversations", tenant_id="acme_corp", agent_name="video_agent")

    Memory->>Memory: Validate inputs
    Memory->>Mem0: memory.search(query, user_id=tenant_id, agent_id=agent_name)

    Mem0->>Backend: search(query, filters)
    Backend->>Schema: Query agent_memories_acme_corp with user_id filter
    Schema-->>Backend: Memories for acme_corp tenant only

    Backend-->>Mem0: Search results
    Mem0-->>Memory: Formatted results
    Memory-->>Agent: Memories (tenant-isolated)

    Note over Agent,Schema: Tenant isolation via schema suffix and user_id prefix
```

### Memory Schema Naming (Per-Tenant)

```mermaid
flowchart LR
    subgraph "Base Memory Schema"
        BaseSchema["<span style='color:#000'>agent_memories</span>"]
    end

    subgraph "Tenant-Specific Memory Schemas"
        SchemaA["<span style='color:#000'>agent_memories_acme_corp</span>"]
        SchemaB["<span style='color:#000'>agent_memories_globex_inc</span>"]
        SchemaC["<span style='color:#000'>agent_memories_default</span>"]
    end

    BaseSchema -->|+ _acme_corp| SchemaA
    BaseSchema -->|+ _globex_inc| SchemaB
    BaseSchema -->|+ _default| SchemaC

    SchemaA --> DocA["<span style='color:#000'>Documents:<br/>user_id prefix: acme_corp_*</span>"]
    SchemaB --> DocB["<span style='color:#000'>Documents:<br/>user_id prefix: globex_inc_*</span>"]
    SchemaC --> DocC["<span style='color:#000'>Documents:<br/>user_id prefix: default_*</span>"]

    style BaseSchema fill:#90caf9,stroke:#1565c0,color:#000
    style SchemaA fill:#ffcc80,stroke:#ef6c00,color:#000
    style SchemaB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style SchemaC fill:#a5d6a7,stroke:#388e3c,color:#000
    style DocA fill:#ffcc80,stroke:#ef6c00,color:#000
    style DocB fill:#ce93d8,stroke:#7b1fa2,color:#000
    style DocC fill:#a5d6a7,stroke:#388e3c,color:#000
```

---

## Deployment Patterns

### Single Vespa Instance Multi-Tenant Deployment

```mermaid
flowchart TB
    subgraph "Infrastructure"
        Vespa["<span style='color:#000'>Vespa Instance<br/>Single deployment</span>"]
        Phoenix["<span style='color:#000'>Phoenix Instance<br/>Single deployment</span>"]
        Mem0Backend["<span style='color:#000'>Mem0 Vespa Backend<br/>Shared</span>"]
    end

    subgraph "Tenant A - acme_corp"
        AppA["<span style='color:#000'>Application: acme_corp</span>"]
        ConfigA["<span style='color:#000'>Config: acme_corp</span>"]
        SchemasA["<span style='color:#000'>Schemas: *_acme_corp</span>"]
        ProjectA["<span style='color:#000'>Phoenix: cogniverse-acme_corp-video-search</span>"]
        MemoryA["<span style='color:#000'>Memory: user_id=acme_corp_*</span>"]
    end

    subgraph "Tenant B - globex_inc"
        AppB["<span style='color:#000'>Application: globex_inc</span>"]
        ConfigB["<span style='color:#000'>Config: globex_inc</span>"]
        SchemasB["<span style='color:#000'>Schemas: *_globex_inc</span>"]
        ProjectB["<span style='color:#000'>Phoenix: cogniverse-globex_inc-video-search</span>"]
        MemoryB["<span style='color:#000'>Memory: user_id=globex_inc_*</span>"]
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

    style Vespa fill:#a5d6a7,stroke:#388e3c,color:#000
    style Phoenix fill:#a5d6a7,stroke:#388e3c,color:#000
    style Mem0Backend fill:#90caf9,stroke:#1565c0,color:#000
    style AppA fill:#90caf9,stroke:#1565c0,color:#000
    style ConfigA fill:#b0bec5,stroke:#546e7a,color:#000
    style SchemasA fill:#90caf9,stroke:#1565c0,color:#000
    style ProjectA fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryA fill:#90caf9,stroke:#1565c0,color:#000
    style AppB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ConfigB fill:#b0bec5,stroke:#546e7a,color:#000
    style SchemasB fill:#ffcc80,stroke:#ef6c00,color:#000
    style ProjectB fill:#a5d6a7,stroke:#388e3c,color:#000
    style MemoryB fill:#ffcc80,stroke:#ef6c00,color:#000
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

    TenantMgr->>Phoenix: create_project("cogniverse-new_corp-video-search")
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

    TenantMgr->>Phoenix: delete_project("cogniverse-old_corp-video-search")
    Phoenix-->>TenantMgr: Project deleted

    TenantMgr->>TenantMgr: Remove tenant config

    TenantMgr-->>Admin: ✅ Tenant "old_corp" deleted<br/>Backup: old_corp_backup_20251015.tar.gz
```

### Multi-Region Deployment (Future)

```mermaid
flowchart TB
    subgraph "US Region"
        VespaUS["<span style='color:#000'>Vespa US</span>"]
        PhoenixUS["<span style='color:#000'>Phoenix US</span>"]

        subgraph "US Tenants"
            TenantUS1["<span style='color:#000'>acme_corp_us</span>"]
            TenantUS2["<span style='color:#000'>globex_us</span>"]
        end

        TenantUS1 --> VespaUS
        TenantUS1 --> PhoenixUS
        TenantUS2 --> VespaUS
        TenantUS2 --> PhoenixUS
    end

    subgraph "EU Region"
        VespaEU["<span style='color:#000'>Vespa EU</span>"]
        PhoenixEU["<span style='color:#000'>Phoenix EU</span>"]

        subgraph "EU Tenants"
            TenantEU1["<span style='color:#000'>acme_corp_eu</span>"]
            TenantEU2["<span style='color:#000'>globex_eu</span>"]
        end

        TenantEU1 --> VespaEU
        TenantEU1 --> PhoenixEU
        TenantEU2 --> VespaEU
        TenantEU2 --> PhoenixEU
    end

    LoadBalancer["<span style='color:#000'>Global Load Balancer<br/>Route by tenant region</span>"]
    LoadBalancer --> VespaUS
    LoadBalancer --> VespaEU

    style VespaUS fill:#90caf9,stroke:#1565c0,color:#000
    style VespaEU fill:#90caf9,stroke:#1565c0,color:#000
    style PhoenixUS fill:#a5d6a7,stroke:#388e3c,color:#000
    style PhoenixEU fill:#a5d6a7,stroke:#388e3c,color:#000
    style TenantUS1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style TenantUS2 fill:#ffcc80,stroke:#ef6c00,color:#000
    style TenantEU1 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style TenantEU2 fill:#ce93d8,stroke:#7b1fa2,color:#000
    style LoadBalancer fill:#b0bec5,stroke:#546e7a,color:#000
```

---

## Summary

This diagram collection provides comprehensive visual documentation of multi-tenant architecture across the **layered structure**:

1. **Tenant Isolation**: Complete separation at schema, project, and memory levels across all layers
2. **Schema-Per-Tenant**: Naming convention with `_tenant_id` suffix managed by cogniverse_vespa
3. **Data Flow**: Tenant-specific routing from ingestion to search through layered architecture
4. **Phoenix Projects**: Per-tenant observability via cogniverse_telemetry_phoenix plugin
5. **Memory Isolation**: User ID prefixes and tenant-specific schemas via cogniverse_core
6. **Lifecycle Management**: Tenant creation, deletion, and backup workflows

**Key Principles:**

- **Schema Isolation**: Each tenant has dedicated Vespa schemas (Implementation Layer)

- **Project Isolation**: Each tenant has dedicated Phoenix project (Implementation Layer Plugin)

- **Memory Isolation**: User IDs prefixed with tenant_id (Core Layer)

- **No Cross-Tenant Access**: Firewall at every layer of the layered architecture

- **Shared Infrastructure**: Single Vespa/Phoenix instances serve all tenants

- **Configuration-Driven**: Tenant isolation configured via cogniverse_foundation

**Tenant Naming Conventions:**

- Vespa schemas: `{base_schema}_{tenant_id}` (cogniverse_vespa)

- Phoenix projects: `cogniverse-{tenant_id}-{service}` (cogniverse_telemetry_phoenix)

- Memory user IDs: `{tenant_id}_{user_id}` (cogniverse_core)

**Layered Architecture Integration:**

- **Foundation Layer**: Provides SystemConfig with tenant_id, TelemetryManager base

- **Core Layer**: Manages agent context, memory, and cache with tenant isolation

- **Implementation Layer**: Vespa backend applies tenant suffixes, agents enforce isolation

- **Application Layer**: Runtime and dashboard respect tenant boundaries

**Related Documentation:**

- [Layered Architecture Guide](../architecture/overview.md)

- [Multi-Tenant Architecture](../architecture/multi-tenant.md)

- [Multi-Tenant Operations](../operations/multi-tenant-ops.md)

- [Configuration Guide](../operations/configuration.md)
