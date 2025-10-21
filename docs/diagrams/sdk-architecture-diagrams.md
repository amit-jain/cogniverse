# SDK Architecture Diagrams

**Last Updated:** 2025-10-15
**Purpose:** Comprehensive visual documentation of UV workspace SDK architecture

---

## Table of Contents
1. [Package Dependency Graph](#package-dependency-graph)
2. [Package Internal Structure](#package-internal-structure)
3. [Cross-Package Data Flow](#cross-package-data-flow)
4. [Import Patterns](#import-patterns)
5. [Build and Deployment](#build-and-deployment)

---

## Package Dependency Graph

### High-Level Package Dependencies

```mermaid
graph TB
    subgraph "UV Workspace"
        Core[cogniverse_core<br/>v0.1.0<br/><br/>• Config<br/>• Telemetry<br/>• Evaluation<br/>• Common Utils]
        Agents[cogniverse_agents<br/>v0.1.0<br/><br/>• Agents<br/>• Routing<br/>• Ingestion<br/>• Search]
        Vespa[cogniverse_vespa<br/>v0.1.0<br/><br/>• Backends<br/>• Schema Mgmt<br/>• JSON Parser]
        Runtime[cogniverse_runtime<br/>v0.1.0<br/><br/>• FastAPI Server<br/>• API Endpoints<br/>• Middleware]
        Dashboard[cogniverse_dashboard<br/>v0.1.0<br/><br/>• Streamlit UI<br/>• Analytics<br/>• Management]
    end

    Agents -->|depends on| Core
    Vespa -->|depends on| Core
    Runtime -->|depends on| Core
    Runtime -->|depends on| Agents
    Runtime -->|depends on| Vespa
    Dashboard -->|depends on| Core
    Dashboard -->|depends on| Agents

    style Core fill:#e1f5ff,stroke:#0077cc,stroke-width:3px
    style Agents fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style Vespa fill:#ffe1f5,stroke:#cc0077,stroke-width:2px
    style Runtime fill:#f5e1ff,stroke:#7700cc,stroke-width:2px
    style Dashboard fill:#e1ffe1,stroke:#00cc77,stroke-width:2px
```

### Detailed Dependency Chain

```mermaid
graph LR
    subgraph "Foundation Layer"
        Core[cogniverse_core]
    end

    subgraph "Integration Layer"
        Agents[cogniverse_agents]
        Vespa[cogniverse_vespa]
    end

    subgraph "Application Layer"
        Runtime[cogniverse_runtime]
        Dashboard[cogniverse_dashboard]
    end

    Core --> Agents
    Core --> Vespa
    Core --> Runtime
    Core --> Dashboard
    Agents --> Runtime
    Agents --> Dashboard
    Vespa --> Runtime

    style Core fill:#e1f5ff
    style Agents fill:#fff4e1
    style Vespa fill:#ffe1f5
    style Runtime fill:#f5e1ff
    style Dashboard fill:#e1ffe1
```

---

## Package Internal Structure

### cogniverse_core Package Structure

```mermaid
graph TB
    CorePkg[cogniverse_core]

    subgraph "Configuration"
        ConfigMgr[config_manager]
        UnifiedConfig[unified_config]
        TenantConfig[tenant_config]
    end

    subgraph "Telemetry"
        TelemetryMgr[TelemetryManager]
        PhoenixIntegration[Phoenix Integration]
        SpanExporter[Span Exporter]
    end

    subgraph "Evaluation"
        ExperimentTracker[ExperimentTracker]
        Evaluators[Evaluators]
        DatasetMgr[DatasetManager]
    end

    subgraph "Common"
        Cache[Cache System]
        Memory[Memory Mgmt]
        Utils[Utilities]
    end

    CorePkg --> ConfigMgr
    CorePkg --> UnifiedConfig
    CorePkg --> TenantConfig
    CorePkg --> TelemetryMgr
    CorePkg --> PhoenixIntegration
    CorePkg --> SpanExporter
    CorePkg --> ExperimentTracker
    CorePkg --> Evaluators
    CorePkg --> DatasetMgr
    CorePkg --> Cache
    CorePkg --> Memory
    CorePkg --> Utils

    style CorePkg fill:#e1f5ff,stroke:#0077cc,stroke-width:3px
    style ConfigMgr fill:#cce5ff
    style UnifiedConfig fill:#cce5ff
    style TenantConfig fill:#cce5ff
    style TelemetryMgr fill:#b3d9ff
    style PhoenixIntegration fill:#b3d9ff
    style SpanExporter fill:#b3d9ff
    style ExperimentTracker fill:#99ccff
    style Evaluators fill:#99ccff
    style DatasetMgr fill:#99ccff
    style Cache fill:#80bfff
    style Memory fill:#80bfff
    style Utils fill:#80bfff
```

### cogniverse_agents Package Structure

```mermaid
graph TB
    AgentsPkg[cogniverse_agents]

    subgraph "Agents"
        BaseAgent[BaseAgent]
        RoutingAgent[RoutingAgent]
        VideoSearchAgent[VideoSearchAgent]
        ComposingAgent[ComposingAgent]
    end

    subgraph "Routing"
        RoutingConfig[RoutingConfig]
        GLiNERStrategy[GLiNER Strategy]
        LLMStrategy[LLM Strategy]
        Optimizer[Optimizer]
    end

    subgraph "Ingestion"
        Pipeline[Pipeline]
        Processors[Processors]
        Extractors[Extractors]
    end

    subgraph "Search"
        Reranker[Multi-Modal Reranker]
        SearchEngine[Search Engine]
    end

    subgraph "Tools"
        A2ATools[A2A Tools]
        VideoPlayer[Video Player]
    end

    AgentsPkg --> BaseAgent
    AgentsPkg --> RoutingAgent
    AgentsPkg --> VideoSearchAgent
    AgentsPkg --> ComposingAgent
    AgentsPkg --> RoutingConfig
    AgentsPkg --> GLiNERStrategy
    AgentsPkg --> LLMStrategy
    AgentsPkg --> Optimizer
    AgentsPkg --> Pipeline
    AgentsPkg --> Processors
    AgentsPkg --> Extractors
    AgentsPkg --> Reranker
    AgentsPkg --> SearchEngine
    AgentsPkg --> A2ATools
    AgentsPkg --> VideoPlayer

    style AgentsPkg fill:#fff4e1,stroke:#ff9900,stroke-width:3px
    style BaseAgent fill:#ffebcc
    style RoutingAgent fill:#ffebcc
    style VideoSearchAgent fill:#ffebcc
    style ComposingAgent fill:#ffebcc
    style RoutingConfig fill:#ffe0b3
    style GLiNERStrategy fill:#ffe0b3
    style LLMStrategy fill:#ffe0b3
    style Optimizer fill:#ffe0b3
    style Pipeline fill:#ffd699
    style Processors fill:#ffd699
    style Extractors fill:#ffd699
    style Reranker fill:#ffcc80
    style SearchEngine fill:#ffcc80
    style A2ATools fill:#ffc266
    style VideoPlayer fill:#ffc266
```

### cogniverse_vespa Package Structure

```mermaid
graph TB
    VespaPkg[cogniverse_vespa]

    subgraph "Backends"
        VespaBackend[VespaBackend]
        SchemaManager[SchemaManager]
        TenantSchemaManager[TenantSchemaManager]
        JSONParser[JSON Schema Parser]
    end

    subgraph "Schema Components"
        Schema[Schema]
        Document[Document]
        Fieldset[Fieldset]
        RankProfile[RankProfile]
    end

    VespaPkg --> VespaBackend
    VespaPkg --> SchemaManager
    VespaPkg --> TenantSchemaManager
    VespaPkg --> JSONParser
    VespaPkg --> Schema
    VespaPkg --> Document
    VespaPkg --> Fieldset
    VespaPkg --> RankProfile

    style VespaPkg fill:#ffe1f5,stroke:#cc0077,stroke-width:3px
    style VespaBackend fill:#ffcce5
    style SchemaManager fill:#ffcce5
    style TenantSchemaManager fill:#ffcce5
    style JSONParser fill:#ffcce5
    style Schema fill:#ffb3d9
    style Document fill:#ffb3d9
    style Fieldset fill:#ffb3d9
    style RankProfile fill:#ffb3d9
```

---

## Cross-Package Data Flow

### Video Ingestion Flow Across Packages

```mermaid
sequenceDiagram
    participant Script as scripts/run_ingestion.py
    participant Core as cogniverse_core
    participant Agents as cogniverse_agents
    participant Vespa as cogniverse_vespa

    Script->>Core: Import SystemConfig
    Core-->>Script: SystemConfig class

    Script->>Core: config = SystemConfig(tenant_id="acme_corp")
    Core-->>Script: config instance

    Script->>Agents: Import VideoIngestionPipeline
    Agents-->>Script: Pipeline class

    Script->>Agents: pipeline = VideoIngestionPipeline(config)
    Agents->>Core: Use config.tenant_id
    Agents->>Core: Initialize TelemetryManager(tenant_id)
    Core-->>Agents: Telemetry manager

    Script->>Agents: pipeline.process_video(video_path)
    Agents->>Agents: Extract frames/chunks
    Agents->>Agents: Generate embeddings
    Agents->>Agents: Build documents

    Agents->>Vespa: Import VespaBackend
    Vespa-->>Agents: Backend class

    Agents->>Vespa: backend = VespaBackend(config)
    Vespa->>Core: Use config.vespa_url, tenant_id

    Agents->>Vespa: backend.feed_documents(docs, schema_name)
    Vespa->>Vespa: Append tenant suffix to schema
    Vespa->>Vespa: Upload to video_colpali_mv_frame_acme_corp

    Vespa-->>Agents: Success response
    Agents->>Core: Record telemetry span
    Core->>Core: Send to Phoenix project: acme_corp_project

    Agents-->>Script: Process result
```

### Query Routing Flow Across Packages

```mermaid
sequenceDiagram
    participant User as User Query
    participant Runtime as cogniverse_runtime
    participant Core as cogniverse_core
    participant Agents as cogniverse_agents

    User->>Runtime: POST /route {"query": "ML videos", "tenant_id": "acme_corp"}

    Runtime->>Core: Import SystemConfig
    Core-->>Runtime: SystemConfig class

    Runtime->>Core: config = SystemConfig(tenant_id="acme_corp")
    Core-->>Runtime: config with tenant isolation

    Runtime->>Agents: Import RoutingAgent
    Agents-->>Runtime: RoutingAgent class

    Runtime->>Agents: agent = RoutingAgent(config)
    Agents->>Core: Initialize telemetry for tenant
    Core-->>Agents: TelemetryManager(tenant="acme_corp")

    Runtime->>Agents: result = agent.route_query(query)
    Agents->>Agents: GLiNER entity extraction
    Agents->>Agents: LLM-based routing decision

    Agents->>Core: Record routing span
    Core->>Core: Attach tenant_id attribute
    Core->>Core: Send to Phoenix: acme_corp_project

    Agents-->>Runtime: {modality: "video", strategy: "hybrid"}
    Runtime-->>User: Routing response
```

### Search Flow Across Packages

```mermaid
sequenceDiagram
    participant User as User Query
    participant Runtime as cogniverse_runtime
    participant Core as cogniverse_core
    participant Agents as cogniverse_agents
    participant Vespa as cogniverse_vespa

    User->>Runtime: POST /search {"query": "ML tutorial", "tenant_id": "acme_corp"}

    Runtime->>Core: config = SystemConfig(tenant_id="acme_corp")
    Core-->>Runtime: Tenant config

    Runtime->>Agents: agent = VideoSearchAgent(config)
    Agents->>Core: Initialize telemetry
    Core-->>Agents: Telemetry manager

    Runtime->>Agents: results = agent.search(query, profile="frame_based")

    Agents->>Agents: Generate query embedding

    Agents->>Vespa: backend = VespaBackend(config)
    Vespa->>Core: Use tenant_id for schema

    Agents->>Vespa: docs = backend.search(query, schema="video_colpali_mv_frame_acme_corp")
    Vespa->>Vespa: Execute Vespa query with tenant schema
    Vespa-->>Agents: Search results

    Agents->>Agents: Multi-modal reranking
    Agents->>Core: Record search span with results
    Core->>Core: Send to Phoenix: acme_corp_project

    Agents-->>Runtime: Reranked results
    Runtime-->>User: Search response
```

---

## Import Patterns

### Correct Import Patterns by Package

```mermaid
graph TB
    subgraph "Scripts Layer"
        Script[scripts/run_ingestion.py]
    end

    subgraph "cogniverse_core Imports"
        CoreConfig["from cogniverse_core.config import SystemConfig"]
        CoreTelemetry["from cogniverse_core.telemetry import TelemetryManager"]
        CoreEval["from cogniverse_core.evaluation import ExperimentTracker"]
    end

    subgraph "cogniverse_agents Imports"
        AgentsPipeline["from cogniverse_agents.ingestion import VideoIngestionPipeline"]
        AgentsRouting["from cogniverse_agents.routing import RoutingAgent"]
        AgentsSearch["from cogniverse_agents.search import MultiModalReranker"]
    end

    subgraph "cogniverse_vespa Imports"
        VespaBackend["from cogniverse_vespa.backends import VespaBackend"]
        VespaSchema["from cogniverse_vespa.backends import VespaSchemaManager"]
    end

    Script --> CoreConfig
    Script --> CoreTelemetry
    Script --> CoreEval
    Script --> AgentsPipeline
    Script --> AgentsRouting
    Script --> AgentsSearch
    Script --> VespaBackend
    Script --> VespaSchema

    style Script fill:#f0f0f0
    style CoreConfig fill:#e1f5ff
    style CoreTelemetry fill:#e1f5ff
    style CoreEval fill:#e1f5ff
    style AgentsPipeline fill:#fff4e1
    style AgentsRouting fill:#fff4e1
    style AgentsSearch fill:#fff4e1
    style VespaBackend fill:#ffe1f5
    style VespaSchema fill:#ffe1f5
```

### Package Import Dependencies (Valid Paths)

```mermaid
graph LR
    subgraph "cogniverse_core CAN import"
        CoreStdLib[Python stdlib]
        CoreThirdParty[3rd party:<br/>pydantic, httpx,<br/>arize-phoenix-otel]
    end

    subgraph "cogniverse_agents CAN import"
        AgentsCore[cogniverse_core.*]
        AgentsThirdParty[3rd party:<br/>litellm, PIL,<br/>transformers]
    end

    subgraph "cogniverse_vespa CAN import"
        VespaCore[cogniverse_core.*]
        VespaThirdParty[3rd party:<br/>pyvespa, requests]
    end

    subgraph "cogniverse_runtime CAN import"
        RuntimeCore[cogniverse_core.*]
        RuntimeAgents[cogniverse_agents.*]
        RuntimeVespa[cogniverse_vespa.*]
        RuntimeThirdParty[3rd party:<br/>fastapi, uvicorn]
    end

    Core[cogniverse_core] --> CoreStdLib
    Core --> CoreThirdParty

    Agents[cogniverse_agents] --> AgentsCore
    Agents --> AgentsThirdParty

    Vespa[cogniverse_vespa] --> VespaCore
    Vespa --> VespaThirdParty

    Runtime[cogniverse_runtime] --> RuntimeCore
    Runtime --> RuntimeAgents
    Runtime --> RuntimeVespa
    Runtime --> RuntimeThirdParty

    style Core fill:#e1f5ff
    style Agents fill:#fff4e1
    style Vespa fill:#ffe1f5
    style Runtime fill:#f5e1ff
```

### INVALID Import Patterns (Circular Dependencies)

```mermaid
graph LR
    Core[cogniverse_core]
    Agents[cogniverse_agents]
    Vespa[cogniverse_vespa]

    Core -.->|❌ INVALID| Agents
    Core -.->|❌ INVALID| Vespa
    Agents -.->|❌ INVALID| Vespa
    Vespa -.->|❌ INVALID| Agents

    style Core fill:#ffcccc
    style Agents fill:#ffcccc
    style Vespa fill:#ffcccc
```

---

## Build and Deployment

### Package Build Pipeline

```mermaid
graph TB
    subgraph "Development"
        Source[Source Code<br/>libs/*/cogniverse_*]
        Tests[Tests<br/>tests/*]
    end

    subgraph "Build Process"
        Lint[uv run ruff check .]
        Format[uv run ruff format .]
        Test[uv run pytest tests/ -v]
        Build[uv build]
    end

    subgraph "Artifacts"
        Wheel[*.whl<br/>Binary distribution]
        Tarball[*.tar.gz<br/>Source distribution]
    end

    subgraph "Distribution"
        PyPI[PyPI<br/>Public registry]
        Private[Private registry<br/>Artifactory/Nexus]
        Local[Local installation<br/>pip install dist/*.whl]
    end

    Source --> Lint
    Tests --> Test
    Lint --> Format
    Format --> Test
    Test --> Build

    Build --> Wheel
    Build --> Tarball

    Wheel --> PyPI
    Wheel --> Private
    Wheel --> Local
    Tarball --> PyPI

    style Source fill:#e1f5ff
    style Tests fill:#e1f5ff
    style Lint fill:#fff4e1
    style Format fill:#fff4e1
    style Test fill:#fff4e1
    style Build fill:#ffe1f5
    style Wheel fill:#e1ffe1
    style Tarball fill:#e1ffe1
    style PyPI fill:#f5e1ff
    style Private fill:#f5e1ff
    style Local fill:#f5e1ff
```

### Workspace Sync Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant UV as uv Package Manager
    participant Lock as uv.lock
    participant VEnv as .venv
    participant Packages as libs/*/

    Dev->>UV: uv sync
    UV->>Lock: Read uv.lock
    Lock-->>UV: Exact dependency versions

    UV->>VEnv: Create/update virtual environment
    VEnv-->>UV: Environment ready

    UV->>Packages: Install libs/core in editable mode
    Packages-->>UV: cogniverse_core installed

    UV->>Packages: Install libs/agents in editable mode
    Packages-->>UV: cogniverse_agents installed

    UV->>Packages: Install libs/vespa in editable mode
    Packages-->>UV: cogniverse_vespa installed

    UV->>Packages: Install libs/runtime in editable mode
    Packages-->>UV: cogniverse_runtime installed

    UV->>Packages: Install libs/dashboard in editable mode
    Packages-->>UV: cogniverse_dashboard installed

    UV->>VEnv: Install 3rd-party dependencies
    VEnv-->>UV: Dependencies installed

    UV-->>Dev: ✅ Workspace synced
```

### Package Release Flow

```mermaid
graph TB
    subgraph "Version Update"
        UpdateVersion[Update version in<br/>pyproject.toml]
        UpdateChangelog[Update CHANGELOG.md]
        UpdateDeps[Update inter-package<br/>dependencies]
    end

    subgraph "Build & Test"
        BuildPkg[uv build]
        TestBuild[Test built packages<br/>in clean environment]
    end

    subgraph "Git Operations"
        CommitChanges[git commit -m<br/>"Release v0.2.0"]
        CreateTag[git tag -a v0.2.0]
        PushTag[git push origin v0.2.0]
    end

    subgraph "Publish"
        PublishCore[Publish cogniverse-core]
        PublishVespa[Publish cogniverse-vespa]
        PublishAgents[Publish cogniverse-agents]
        PublishRuntime[Publish cogniverse-runtime]
        PublishDashboard[Publish cogniverse-dashboard]
    end

    UpdateVersion --> UpdateChangelog
    UpdateChangelog --> UpdateDeps
    UpdateDeps --> BuildPkg
    BuildPkg --> TestBuild
    TestBuild --> CommitChanges
    CommitChanges --> CreateTag
    CreateTag --> PushTag

    PushTag --> PublishCore
    PublishCore --> PublishVespa
    PublishVespa --> PublishAgents
    PublishAgents --> PublishRuntime
    PublishRuntime --> PublishDashboard

    style UpdateVersion fill:#e1f5ff
    style UpdateChangelog fill:#e1f5ff
    style UpdateDeps fill:#e1f5ff
    style BuildPkg fill:#fff4e1
    style TestBuild fill:#fff4e1
    style CommitChanges fill:#ffe1f5
    style CreateTag fill:#ffe1f5
    style PushTag fill:#ffe1f5
    style PublishCore fill:#e1ffe1
    style PublishVespa fill:#e1ffe1
    style PublishAgents fill:#e1ffe1
    style PublishRuntime fill:#e1ffe1
    style PublishDashboard fill:#e1ffe1
```

### Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DevWorkspace[UV Workspace<br/>uv sync<br/>Editable installs]
        DevTests[Local Tests<br/>pytest]
    end

    subgraph "CI/CD Pipeline"
        GitHub[GitHub Actions]
        BuildAll[Build all packages]
        TestAll[Test all packages]
        PublishPyPI[Publish to PyPI]
    end

    subgraph "Production Deployment"
        Docker[Docker Container]
        Modal[Modal Serverless]
        Kubernetes[Kubernetes]
    end

    subgraph "Package Installation"
        ProdInstall[pip install<br/>cogniverse-runtime]
        AllDeps[Auto-installs:<br/>• cogniverse-core<br/>• cogniverse-agents<br/>• cogniverse-vespa]
    end

    DevWorkspace --> DevTests
    DevTests --> GitHub
    GitHub --> BuildAll
    BuildAll --> TestAll
    TestAll --> PublishPyPI

    PublishPyPI --> Docker
    PublishPyPI --> Modal
    PublishPyPI --> Kubernetes

    Docker --> ProdInstall
    Modal --> ProdInstall
    Kubernetes --> ProdInstall
    ProdInstall --> AllDeps

    style DevWorkspace fill:#e1f5ff
    style DevTests fill:#e1f5ff
    style GitHub fill:#fff4e1
    style BuildAll fill:#fff4e1
    style TestAll fill:#fff4e1
    style PublishPyPI fill:#ffe1f5
    style Docker fill:#e1ffe1
    style Modal fill:#e1ffe1
    style Kubernetes fill:#e1ffe1
    style ProdInstall fill:#f5e1ff
    style AllDeps fill:#f5e1ff
```

---

## Summary

This diagram collection provides comprehensive visual documentation of:

1. **Package Dependencies**: Clear hierarchy with core as foundation
2. **Internal Structure**: Detailed breakdown of each package's modules
3. **Data Flow**: Cross-package interactions during ingestion, routing, and search
4. **Import Patterns**: Valid and invalid import paths
5. **Build & Deploy**: Complete pipeline from development to production

**Key Principles:**
- Unidirectional dependencies (agents/vespa → core, runtime → all)
- UV workspace enables editable installs for development
- Build artifacts (wheels) for production deployment
- Tenant isolation maintained across all package interactions

**Related Documentation:**
- [SDK Architecture](../architecture/sdk-architecture.md)
- [Package Development](../development/package-dev.md)
- [Multi-Tenant Architecture](../architecture/multi-tenant.md)
