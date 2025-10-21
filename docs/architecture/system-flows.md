# Cogniverse System Flows & Architecture Scenarios

**Last Updated:** 2025-10-15
**Purpose:** Comprehensive system flows with SDK architecture and multi-tenant patterns
**Format:** Mermaid diagrams showing package boundaries and tenant context

---

## Table of Contents
1. [Overall System Architecture](#overall-system-architecture)
2. [Query Processing Flows](#query-processing-flows)
3. [Agent Orchestration Flows](#agent-orchestration-flows)
4. [Multi-Tenant Flows](#multi-tenant-flows)
5. [Optimization & Learning Flows](#optimization--learning-flows)
6. [Evaluation & Experiment Flows](#evaluation--experiment-flows)
7. [Memory & Context Flows](#memory--context-flows)
8. [Ingestion & Dataset Flows](#ingestion--dataset-flows)
9. [Production Deployment Flows](#production-deployment-flows)

---

## Overall System Architecture

### SDK Package Architecture

```mermaid
graph TB
    subgraph "cogniverse_runtime Package"
        Runtime[FastAPI Server<br/>Tenant Middleware<br/>Ingestion Pipeline]
    end

    subgraph "cogniverse_agents Package"
        Orchestrator[Multi-Agent Orchestrator]
        Routing[Routing Agent]
        VideoAgent[Video Search Agent]
        Summarizer[Summarizer Agent]
        ReportAgent[Report Agent]
    end

    subgraph "cogniverse_core Package"
        Memory[Mem0 Memory Manager]
        Config[Configuration Manager]
        Telemetry[Phoenix Telemetry]
        Evaluation[Evaluation Module]
    end

    subgraph "cogniverse_vespa Package"
        TenantMgr[TenantSchemaManager]
        VespaBackend[Vespa Search Backend]
        EmbedProc[Embedding Processor]
    end

    subgraph "cogniverse_dashboard Package"
        Dashboard[Streamlit UI<br/>Phoenix Dashboard]
    end

    Runtime --> Orchestrator
    Runtime --> TenantMgr

    Orchestrator --> Routing
    Orchestrator --> VideoAgent
    Orchestrator --> Summarizer
    Orchestrator --> ReportAgent

    Routing --> Memory
    Routing --> Telemetry
    VideoAgent --> VespaBackend
    VideoAgent --> Telemetry

    Memory --> VespaBackend
    Telemetry --> Evaluation

    VespaBackend --> TenantMgr
    TenantMgr --> EmbedProc

    Dashboard --> Telemetry
    Dashboard --> VespaBackend

    style Runtime fill:#e1f5ff
    style Orchestrator fill:#fff4e1
    style TenantMgr fill:#e1ffe1
    style Telemetry fill:#f5e1ff
```

### Component Interaction with Package Boundaries

```mermaid
graph LR
    subgraph "cogniverse_runtime"
        API[FastAPI Endpoints<br/>Tenant Middleware]
    end

    subgraph "cogniverse_agents"
        ORC[Orchestrator]
        RT[Routing Agent]
        VA[Video Agent]
        SUM[Summarizer]
    end

    subgraph "cogniverse_core"
        DSPy[DSPy Integration]
        GRPO[GRPO Optimizer]
        Memory[Mem0 Memory]
        Phoenix[Phoenix Telemetry]
    end

    subgraph "cogniverse_vespa"
        TenantSchema[TenantSchemaManager]
        Vespa[Vespa Backends]
    end

    API --> ORC
    ORC --> RT
    ORC --> VA
    ORC --> SUM

    RT --> DSPy
    RT --> GRPO
    RT --> Memory

    VA --> Vespa
    VA --> Phoenix

    Vespa --> TenantSchema

    GRPO --> Phoenix
    Memory --> Vespa

    style API fill:#e1f5ff
    style ORC fill:#fff4e1
```

---

## Query Processing Flows

### Scenario 1: Simple Video Search with Tenant Context

```mermaid
sequenceDiagram
    participant User
    participant Runtime as cogniverse_runtime<br/>FastAPI
    participant Middleware as Tenant Middleware
    participant Routing as Routing Agent<br/>cogniverse_agents
    participant VideoAgent as Video Search Agent<br/>cogniverse_agents
    participant TenantMgr as TenantSchemaManager<br/>cogniverse_vespa
    participant Vespa as Vespa Backend<br/>cogniverse_vespa
    participant Phoenix as Phoenix Telemetry<br/>cogniverse_core

    User->>Runtime: GET /search?query="cooking videos"<br/>Header: X-Tenant-ID: acme
    Runtime->>Middleware: inject_tenant_context()

    activate Middleware
    Middleware->>Middleware: Extract tenant_id from header
    Middleware->>TenantMgr: ensure_tenant_schemas_exist("acme")
    TenantMgr-->>Middleware: Schemas ready
    Middleware->>Runtime: request.state.tenant_id = "acme"
    deactivate Middleware

    Runtime->>Routing: route_query(query, tenant_id="acme")

    activate Routing
    Routing->>Routing: Extract entities (GLiNER)
    Note over Routing: Entities: cooking, videos
    Routing->>Routing: Determine modality
    Note over Routing: Modality: VIDEO
    Routing->>Phoenix: Record routing span (tenant project)
    Routing-->>Runtime: {modality: VIDEO, agent: video_search}
    deactivate Routing

    Runtime->>VideoAgent: search(query="cooking videos", tenant_id="acme")

    activate VideoAgent
    VideoAgent->>TenantMgr: get_tenant_schema_name("acme", "video")
    TenantMgr-->>VideoAgent: "video_frames_acme"
    VideoAgent->>VideoAgent: Encode query (ColPali)
    VideoAgent->>Vespa: nearest_neighbor_search(schema="video_frames_acme")
    Vespa-->>VideoAgent: Top 10 results (tenant-scoped)
    VideoAgent->>VideoAgent: Rerank results
    VideoAgent->>Phoenix: Record search span (tenant project)
    VideoAgent-->>Runtime: [ranked_results]
    deactivate VideoAgent

    Runtime-->>User: Display video results<br/>Header: X-Tenant-ID: acme
```

### Scenario 2: Multi-Modal Query with Fusion

```mermaid
sequenceDiagram
    participant User
    participant Runtime as cogniverse_runtime
    participant Orchestrator as Orchestrator<br/>cogniverse_agents
    participant Routing as Routing Agent<br/>cogniverse_agents
    participant VideoAgent as Video Agent<br/>cogniverse_agents
    participant TextAgent as Text Agent<br/>cogniverse_agents
    participant Fusion as Cross-Modal Fusion<br/>cogniverse_agents
    participant Phoenix as Phoenix<br/>cogniverse_core

    User->>Runtime: "How does photosynthesis work?"<br/>tenant_id="startup"

    Runtime->>Orchestrator: process_query(tenant_id="startup")
    Orchestrator->>Routing: route_query(tenant_id="startup")
    Routing-->>Orchestrator: {modalities: [VIDEO, TEXT]}

    par Parallel Execution
        Orchestrator->>VideoAgent: search(tenant_id="startup")
        and
        Orchestrator->>TextAgent: search(tenant_id="startup")
    end

    VideoAgent-->>Orchestrator: video_results
    TextAgent-->>Orchestrator: text_results

    Orchestrator->>Fusion: fuse_results(video, text)
    activate Fusion
    Fusion->>Fusion: Calculate cross-modal consistency
    Fusion->>Fusion: Merge and deduplicate
    Fusion->>Fusion: Apply fusion strategy
    Fusion-->>Orchestrator: fused_results
    deactivate Fusion

    Orchestrator->>Phoenix: Record orchestration span<br/>(project: startup_orchestrator)
    Orchestrator-->>Runtime: Combined results
    Runtime-->>User: Results with metadata
```

### Scenario 3: Memory-Enhanced Routing

```mermaid
sequenceDiagram
    participant User
    participant Routing as Routing Agent<br/>cogniverse_agents
    participant Memory as Mem0 Manager<br/>cogniverse_core
    participant Vespa as Vespa Memory<br/>agent_memories_acme
    participant VideoAgent as Video Agent<br/>cogniverse_agents
    participant Phoenix as Phoenix<br/>cogniverse_core

    User->>Routing: "Show me more like the last one"<br/>tenant_id="acme"

    Routing->>Memory: get_relevant_context(tenant_id="acme")
    activate Memory
    Memory->>Vespa: Vector search in agent_memories_acme
    Vespa-->>Memory: [previous_context]
    Memory-->>Routing: Context: "pasta cooking tutorial"
    deactivate Memory

    Note over Routing: Context: User previously searched<br/>"pasta cooking tutorial"

    Routing->>Routing: Enhance query with context
    Note over Routing: Enhanced: "Show me more<br/>pasta cooking tutorials"

    Routing->>VideoAgent: search(enhanced_query, tenant_id="acme")
    VideoAgent-->>Routing: results

    Routing->>Memory: add_memory(result_context, tenant_id="acme")
    Routing->>Phoenix: Record memory-enhanced span<br/>(project: acme_routing_agent)

    Routing-->>User: Contextual results
```

---

## Agent Orchestration Flows

### Scenario 4: Complex Multi-Agent Workflow with SDK Packages

```mermaid
graph TB
    Start[User Query<br/>tenant_id="acme"] --> Runtime[cogniverse_runtime<br/>FastAPI + Middleware]

    Runtime --> Orchestrator[Orchestrator<br/>cogniverse_agents]

    Orchestrator --> T1[Task 1: Route Query]
    Orchestrator --> T2[Task 2: Parallel Search]
    Orchestrator --> T3[Task 3: Summarize]
    Orchestrator --> T4[Task 4: Generate Report]

    T1 --> Routing[Routing Agent<br/>cogniverse_agents]
    Routing --> T2

    T2 --> ParallelBlock{Parallel Execution}
    ParallelBlock --> Video[Video Search<br/>cogniverse_agents]
    ParallelBlock --> Text[Text Search<br/>cogniverse_agents]

    Video --> VespaV[video_frames_acme<br/>cogniverse_vespa]
    Text --> VespaT[document_content_acme<br/>cogniverse_vespa]

    VespaV --> T3
    VespaT --> T3

    T3 --> Summarizer[Summarizer Agent<br/>cogniverse_agents]
    Summarizer --> T4

    T4 --> Reporter[Report Agent<br/>cogniverse_agents]
    Reporter --> Result[Final Report]

    style Start fill:#e1f5ff
    style Result fill:#e1ffe1
    style ParallelBlock fill:#fff4e1
    style Runtime fill:#ffe1e1
```

### Scenario 5: Task Dependency Resolution

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator as Orchestrator<br/>cogniverse_agents
    participant TaskGraph as Task Graph Builder
    participant Routing as Routing Agent
    participant VideoAgent as Video Agent
    participant Summarizer as Summarizer Agent

    User->>Orchestrator: Complex query<br/>tenant_id="acme"

    Orchestrator->>TaskGraph: build_dependency_graph()
    activate TaskGraph
    TaskGraph->>TaskGraph: Identify required tasks
    TaskGraph->>TaskGraph: Determine dependencies
    TaskGraph-->>Orchestrator: task_graph
    deactivate TaskGraph

    Note over Orchestrator: Dependency Order:<br/>1. Routing (no deps)<br/>2. Search (depends on routing)<br/>3. Summarize (depends on search)

    Orchestrator->>Routing: Execute Task 1 (tenant_id="acme")
    Routing-->>Orchestrator: routing_result

    Orchestrator->>VideoAgent: Execute Task 2 (tenant_id="acme")
    VideoAgent-->>Orchestrator: search_results

    Orchestrator->>Summarizer: Execute Task 3 (tenant_id="acme")
    Summarizer-->>Orchestrator: summary

    Orchestrator-->>User: Final result with all components
```

### Scenario 6: Agent-to-Agent Communication (A2A Protocol)

```mermaid
sequenceDiagram
    participant VideoAgent as Video Agent<br/>cogniverse_agents
    participant Gateway as A2A Gateway<br/>cogniverse_agents
    participant Summarizer as Summarizer<br/>cogniverse_agents
    participant ReportAgent as Report Agent<br/>cogniverse_agents

    VideoAgent->>Gateway: Send A2A Message
    Note over Gateway: Message Format:<br/>{type: "task",<br/>sender: "video_agent",<br/>target: "summarizer",<br/>tenant_id: "acme",<br/>data: results}

    Gateway->>Gateway: Validate message format
    Gateway->>Gateway: Route to target agent

    Gateway->>Summarizer: Forward task (tenant_id="acme")

    activate Summarizer
    Summarizer->>Summarizer: Process results
    Summarizer-->>Gateway: A2A Response
    deactivate Summarizer

    Gateway->>ReportAgent: Chain to next agent
    activate ReportAgent
    ReportAgent->>ReportAgent: Generate detailed report
    ReportAgent-->>Gateway: Final response
    deactivate ReportAgent

    Gateway-->>VideoAgent: Complete A2A workflow
```

---

## Multi-Tenant Flows

### Scenario 7: Tenant Schema Lifecycle

```mermaid
sequenceDiagram
    participant Admin
    participant Runtime as cogniverse_runtime
    participant TenantMgr as TenantSchemaManager<br/>cogniverse_vespa
    participant Vespa as Vespa (port 19071)
    participant ConfigMgr as ConfigManager<br/>cogniverse_core

    Admin->>Runtime: POST /tenants<br/>{tenant_id: "newcorp"}

    Runtime->>TenantMgr: deploy_tenant_schemas("newcorp")

    activate TenantMgr
    TenantMgr->>TenantMgr: Generate schema names:<br/>- video_frames_newcorp<br/>- document_content_newcorp<br/>- agent_memories_newcorp

    loop For each schema
        TenantMgr->>TenantMgr: Clone from base template
        TenantMgr->>Vespa: Deploy schema (HTTP POST)
        Vespa-->>TenantMgr: Deployment successful
    end

    TenantMgr-->>Runtime: All schemas deployed
    deactivate TenantMgr

    Runtime->>ConfigMgr: create_tenant_config("newcorp")
    ConfigMgr-->>Runtime: Config initialized

    Runtime-->>Admin: Tenant "newcorp" created<br/>Ready for use
```

### Scenario 8: Multi-Tenant Request Isolation

```mermaid
graph TB
    subgraph "Incoming Requests"
        ReqA[Request A<br/>tenant_id: acme]
        ReqB[Request B<br/>tenant_id: startup]
    end

    subgraph "cogniverse_runtime"
        Middleware[Tenant Middleware]
    end

    subgraph "Schema Routing cogniverse_vespa"
        TenantMgr[TenantSchemaManager]
    end

    subgraph "Tenant A Isolation"
        SchemaA1[video_frames_acme]
        SchemaA2[agent_memories_acme]
        ConfigA[Config: acme]
        PhoenixA[Phoenix Project: acme_*]
    end

    subgraph "Tenant B Isolation"
        SchemaB1[video_frames_startup]
        SchemaB2[agent_memories_startup]
        ConfigB[Config: startup]
        PhoenixB[Phoenix Project: startup_*]
    end

    ReqA --> Middleware
    ReqB --> Middleware

    Middleware --> TenantMgr

    TenantMgr -->|Route A| SchemaA1
    TenantMgr -->|Route A| SchemaA2
    TenantMgr -->|Route B| SchemaB1
    TenantMgr -->|Route B| SchemaB2

    SchemaA1 -.->|No cross-tenant access| SchemaB1
    ConfigA -.->|Isolated| ConfigB
    PhoenixA -.->|Isolated| PhoenixB

    style ReqA fill:#e1f5ff
    style ReqB fill:#fff4e1
    style TenantMgr fill:#e1ffe1
```

### Scenario 9: Tenant Memory Isolation

```mermaid
graph LR
    subgraph "Tenant: acme"
        UserA[User Query<br/>tenant_id="acme"]
        MemA[Mem0Manager<br/>instance for acme]
        SchemaA[agent_memories_acme]
    end

    subgraph "Tenant: startup"
        UserB[User Query<br/>tenant_id="startup"]
        MemB[Mem0Manager<br/>instance for startup]
        SchemaB[agent_memories_startup]
    end

    subgraph "cogniverse_core"
        MemSingleton[Mem0MemoryManager<br/>Per-tenant singletons<br/>_instances['acme']<br/>_instances['startup']]
    end

    subgraph "cogniverse_vespa"
        VespaCore[Vespa Core<br/>Physical isolation]
    end

    UserA --> MemA
    UserB --> MemB

    MemA --> MemSingleton
    MemB --> MemSingleton

    MemA --> SchemaA
    MemB --> SchemaB

    SchemaA --> VespaCore
    SchemaB --> VespaCore

    SchemaA -.->|No cross-access| SchemaB

    style UserA fill:#e1f5ff
    style UserB fill:#fff4e1
    style VespaCore fill:#e1ffe1
```

---

## Optimization & Learning Flows

### Scenario 10: GRPO Optimization Cycle with Packages

```mermaid
graph TB
    Start[Query Execution<br/>tenant_id="acme"] --> Telemetry[Phoenix Records Spans<br/>cogniverse_core<br/>project: acme_routing_agent]

    Telemetry --> SpanEval[Span Evaluator<br/>cogniverse_core]

    SpanEval --> Extract{Extract Experiences<br/>per tenant}
    Extract --> Quality[Quality Signals]
    Extract --> Latency[Latency Metrics]
    Extract --> UserFeedback[User Feedback]

    Quality --> ExpReplay[Experience Replay Buffer<br/>cogniverse_core]
    Latency --> ExpReplay
    UserFeedback --> ExpReplay

    ExpReplay --> GRPO[GRPO Optimizer<br/>cogniverse_core]

    GRPO --> UpdateDSPy[Update DSPy Module<br/>cogniverse_agents]
    UpdateDSPy --> NewModel[Optimized Routing Model<br/>tenant-specific]

    NewModel --> Deploy[Deploy to Routing Agent<br/>cogniverse_agents]
    Deploy --> Monitor[Monitor Performance<br/>cogniverse_core]

    Monitor --> Telemetry

    style Start fill:#e1f5ff
    style NewModel fill:#e1ffe1
    style ExpReplay fill:#fff4e1
```

### Scenario 11: Experience Collection & Optimization

```mermaid
sequenceDiagram
    participant Routing as Routing Agent<br/>cogniverse_agents
    participant Phoenix as Phoenix<br/>cogniverse_core
    participant SpanEval as Span Evaluator<br/>cogniverse_core
    participant GRPO as GRPO Optimizer<br/>cogniverse_core
    participant Deploy as Deployment<br/>cogniverse_runtime

    loop Continuous Operation (per tenant)
        Routing->>Phoenix: Record routing decisions<br/>(project: tenant_routing_agent)
        Phoenix->>Phoenix: Store spans with tenant context
    end

    Note over SpanEval: Triggered every 1000 queries<br/>or on-demand<br/>(per tenant)

    SpanEval->>Phoenix: Fetch recent spans (tenant-scoped)
    Phoenix-->>SpanEval: routing_spans[] (filtered by tenant)

    SpanEval->>SpanEval: Extract experiences per tenant
    Note over SpanEval: Experience = {<br/>  query: str,<br/>  prediction: modality,<br/>  reward: float,<br/>  tenant_id: str,<br/>  context: dict<br/>}

    SpanEval->>GRPO: feed_experiences(experiences, tenant_id)

    activate GRPO
    GRPO->>GRPO: Update tenant-specific replay buffer
    GRPO->>GRPO: Sample mini-batch
    GRPO->>GRPO: Compute gradients
    GRPO->>GRPO: Update policy
    GRPO-->>Deploy: optimization_metrics + tenant_id
    deactivate GRPO

    Deploy->>Routing: Deploy optimized module (tenant-aware)
    Routing->>Phoenix: Record deployment event<br/>(project: tenant_routing_agent)
```

### Scenario 12: Cross-Modal Optimization

```mermaid
graph LR
    subgraph "Data Collection (cogniverse_core)"
        V[Video Queries] --> VMetrics[Video Metrics<br/>per tenant]
        T[Text Queries] --> TMetrics[Text Metrics<br/>per tenant]
        M[Multi-Modal] --> MMetrics[Multi-Modal Metrics<br/>per tenant]
    end

    subgraph "Analysis (cogniverse_core)"
        VMetrics --> Analyzer[Cross-Modal Analyzer<br/>tenant-aware]
        TMetrics --> Analyzer
        MMetrics --> Analyzer

        Analyzer --> Patterns[Pattern Detection]
        Patterns --> Insights[Insights per tenant]
    end

    subgraph "Optimization (cogniverse_agents)"
        Insights --> VidOpt[Video Optimizer]
        Insights --> TextOpt[Text Optimizer]
        Insights --> FusionOpt[Fusion Optimizer]

        VidOpt --> Deploy[Deploy Updates<br/>per tenant]
        TextOpt --> Deploy
        FusionOpt --> Deploy
    end

    Deploy --> V
    Deploy --> T
    Deploy --> M

    style Analyzer fill:#fff4e1
    style Deploy fill:#e1ffe1
```

---

## Evaluation & Experiment Flows

### Scenario 13: Phoenix Experiment Workflow

```mermaid
sequenceDiagram
    participant Script as Experiment Script
    participant Tracker as Experiment Tracker<br/>cogniverse_core
    participant Dataset as Dataset Manager
    participant Routing as Routing Agent<br/>cogniverse_agents
    participant Phoenix as Phoenix<br/>cogniverse_core
    participant Eval as Evaluator<br/>cogniverse_core

    Script->>Tracker: create_experiment("routing_eval_v1")<br/>tenant_id="acme"
    Tracker->>Phoenix: Register experiment<br/>(project: acme_experiments)

    Script->>Dataset: load_golden_dataset()
    Dataset-->>Script: queries_with_labels

    loop For each query
        Script->>Routing: route_query(query, tenant_id="acme")
        Routing->>Phoenix: Record span (project: acme_routing_agent)
        Routing-->>Script: prediction

        Script->>Script: Compare with gold label
        Script->>Tracker: log_prediction(query, pred, gold)
    end

    Script->>Eval: evaluate_experiment(tenant_id="acme")

    activate Eval
    Eval->>Phoenix: Fetch experiment spans<br/>(project: acme_experiments)
    Eval->>Eval: Calculate metrics
    Note over Eval: Accuracy: 0.92<br/>Precision: 0.89<br/>Recall: 0.94<br/>F1: 0.91
    Eval->>Phoenix: Store metrics (tenant-scoped)
    Eval-->>Script: evaluation_results
    deactivate Eval

    Script->>Tracker: finalize_experiment()
```

### Scenario 14: Routing Evaluator Integration

```mermaid
graph TB
    Start[Evaluation Request<br/>tenant_id="acme"] --> LoadDataset[Load Golden Dataset<br/>cogniverse_core]
    LoadDataset --> PrepQueries[Prepare Test Queries]

    PrepQueries --> Loop{For Each Query}

    Loop --> Execute[Execute Routing<br/>cogniverse_agents]
    Execute --> Predict[Get Prediction]
    Predict --> Compare[Compare with Gold Label]

    Compare --> StoreResult[Store Result<br/>cogniverse_core]
    StoreResult --> Loop

    Loop --> Aggregate[Aggregate Results]

    Aggregate --> CalcMetrics[Calculate Metrics<br/>cogniverse_core]
    CalcMetrics --> Accuracy[Accuracy]
    CalcMetrics --> Precision[Precision/Recall]
    CalcMetrics --> Confusion[Confusion Matrix]

    Accuracy --> Report[Generate Report]
    Precision --> Report
    Confusion --> Report

    Report --> Visualize[Create Visualizations<br/>cogniverse_dashboard]
    Visualize --> Dashboard[Phoenix Dashboard<br/>project: acme_evaluation]

    style Start fill:#e1f5ff
    style Dashboard fill:#e1ffe1
```

### Scenario 15: Quality Evaluator for Experiments

```mermaid
sequenceDiagram
    participant Exp as Experiment Script
    participant Phoenix as Phoenix<br/>cogniverse_core
    participant QualityEval as Quality Evaluator<br/>cogniverse_core
    participant LLM as LLM Judge<br/>Ollama
    participant Metrics as Metrics Store<br/>cogniverse_core

    Exp->>Phoenix: Run experiment with queries<br/>tenant_id="acme"
    Phoenix->>Phoenix: Record all spans<br/>(project: acme_experiments)

    Exp->>QualityEval: evaluate_quality(experiment_id, tenant_id="acme")

    activate QualityEval
    QualityEval->>Phoenix: Fetch experiment spans<br/>(project: acme_experiments)
    Phoenix-->>QualityEval: spans_with_results (tenant-scoped)

    loop For each result
        QualityEval->>LLM: Evaluate relevance
        Note over LLM: Prompt: "Rate result relevance<br/>for query X on scale 1-5"
        LLM-->>QualityEval: relevance_score

        QualityEval->>QualityEval: Calculate quality metrics
    end

    QualityEval->>Metrics: Store quality scores (tenant-scoped)
    QualityEval-->>Exp: {avg_relevance: 4.2,<br/>quality_distribution: {...}}
    deactivate QualityEval
```

---

## Memory & Context Flows

### Scenario 16: Conversation Memory Integration

```mermaid
graph TB
    Query[New Query<br/>tenant_id="acme"] --> CheckMemory{Check Tenant Memory<br/>cogniverse_core}

    CheckMemory -->|Memory Found| Retrieve[Retrieve Context<br/>agent_memories_acme]
    CheckMemory -->|No Memory| Direct[Direct Processing]

    Retrieve --> Relevant[Filter Relevant Memories<br/>Mem0 semantic search]
    Relevant --> Enhance[Enhance Query<br/>cogniverse_agents]

    Enhance --> Process[Process Enhanced Query]
    Direct --> Process

    Process --> Execute[Execute Search<br/>cogniverse_agents]
    Execute --> Results[Get Results from<br/>video_frames_acme]

    Results --> Store[Store New Memory<br/>agent_memories_acme]
    Store --> Update[Update Tenant Context]

    Update --> Return[Return Results]

    style Query fill:#e1f5ff
    style Return fill:#e1ffe1
    style Relevant fill:#fff4e1
```

### Scenario 17: Memory Lifecycle with Tenant Isolation

```mermaid
sequenceDiagram
    participant User
    participant Agent as Agent<br/>cogniverse_agents
    participant Memory as Mem0 Manager<br/>cogniverse_core<br/>instance for "acme"
    participant Vespa as Vespa<br/>agent_memories_acme
    participant Cleanup as Cleanup Service

    User->>Agent: Initial query<br/>tenant_id="acme"
    Agent->>Memory: add_memory(content, tenant_id="acme")

    activate Memory
    Memory->>Memory: Generate embeddings (Ollama)
    Memory->>Vespa: Store in agent_memories_acme
    Memory-->>Agent: memory_id
    deactivate Memory

    Note over Memory: Time passes...

    User->>Agent: Follow-up query<br/>tenant_id="acme"
    Agent->>Memory: search_memory(query, tenant_id="acme")

    activate Memory
    Memory->>Memory: Encode search query
    Memory->>Vespa: Vector search in agent_memories_acme
    Vespa-->>Memory: Relevant memories (tenant-scoped)
    Memory->>Memory: Filter by recency
    Memory-->>Agent: context_memories
    deactivate Memory

    Note over Cleanup: Scheduled maintenance (per tenant)

    Cleanup->>Memory: cleanup_old_memories(tenant_id="acme")
    Memory->>Vespa: Delete expired from agent_memories_acme
```

### Scenario 18: Per-Tenant Memory Singleton Pattern

```mermaid
graph TB
    subgraph "Agent Initialization"
        AgentA[Agent A requests memory<br/>tenant_id="acme"]
        AgentB[Agent B requests memory<br/>tenant_id="acme"]
        AgentC[Agent C requests memory<br/>tenant_id="startup"]
    end

    subgraph "cogniverse_core Mem0MemoryManager"
        Singleton[Per-Tenant Singleton<br/>_instances = {<br/>  'acme': manager_instance_1,<br/>  'startup': manager_instance_2<br/>}]
    end

    subgraph "Vespa Schemas cogniverse_vespa"
        SchemaA[agent_memories_acme]
        SchemaB[agent_memories_startup]
    end

    AgentA --> Singleton
    AgentB --> Singleton
    AgentC --> Singleton

    Singleton -->|Same instance| SchemaA
    Singleton -->|Same instance| SchemaA
    Singleton -->|Different instance| SchemaB

    SchemaA -.->|Isolated| SchemaB

    style AgentA fill:#e1f5ff
    style AgentB fill:#e1f5ff
    style AgentC fill:#fff4e1
    style Singleton fill:#e1ffe1
```

---

## Ingestion & Dataset Flows

### Scenario 19: Video Ingestion Pipeline with Tenant Routing

```mermaid
graph TB
    Input[Video Files<br/>tenant_id="acme"] --> Runtime[Ingestion Pipeline<br/>cogniverse_runtime]

    Runtime --> Strategy[Strategy Factory<br/>cogniverse_runtime]

    Strategy --> Profile{Select Profile}
    Profile -->|Frame-Based| FrameProc[Frame Processor<br/>ColPali]
    Profile -->|Chunk-Based| ChunkProc[Chunk Processor<br/>ColQwen]
    Profile -->|Global| GlobalProc[Global Processor<br/>VideoPrism]

    FrameProc --> Embed[Generate Embeddings<br/>cogniverse_runtime]
    ChunkProc --> Embed
    GlobalProc --> Embed

    Embed --> Format[Format Conversion<br/>cogniverse_vespa<br/>Binary + Float]

    Format --> TenantMgr[TenantSchemaManager<br/>get_tenant_schema_name<br/>'acme', 'video']

    TenantMgr --> Build[Build Vespa Documents<br/>cogniverse_vespa]

    Build --> Upload[Bulk Upload<br/>to video_frames_acme]
    Upload --> Verify[Verify Upload Success]

    style Input fill:#e1f5ff
    style Verify fill:#e1ffe1
```

### Scenario 20: Dataset Extraction for Evaluation

```mermaid
sequenceDiagram
    participant Script
    participant DatasetMgr as Dataset Manager<br/>cogniverse_core
    participant Phoenix as Phoenix<br/>cogniverse_core
    participant Vespa as Vespa<br/>cogniverse_vespa
    participant Export as Export Service

    Script->>DatasetMgr: create_dataset("golden_eval_v1")<br/>tenant_id="acme"

    DatasetMgr->>Phoenix: Fetch production spans<br/>(project: acme_routing_agent)
    Note over Phoenix: Filter by:<br/>- Date range<br/>- Quality threshold<br/>- User feedback<br/>- Tenant isolation
    Phoenix-->>DatasetMgr: high_quality_spans (tenant-scoped)

    DatasetMgr->>DatasetMgr: Extract queries & labels

    loop For each span
        DatasetMgr->>DatasetMgr: Parse routing decision
        DatasetMgr->>DatasetMgr: Validate gold label
        DatasetMgr->>DatasetMgr: Verify tenant_id="acme"
    end

    DatasetMgr->>DatasetMgr: Deduplicate queries
    DatasetMgr->>DatasetMgr: Balance modalities

    DatasetMgr->>Vespa: Store dataset<br/>(tenant-scoped storage)
    DatasetMgr->>Export: Export to CSV with tenant_id

    Export-->>Script: dataset_file_path
    Script->>Script: Validate dataset quality
```

### Scenario 21: Ingestion Strategy Resolution

```mermaid
graph TB
    Start[Video Input<br/>tenant_id="acme"] --> Runtime[cogniverse_runtime<br/>Ingestion Service]

    Runtime --> Analyzer[Analyze Video Properties]

    Analyzer --> Duration{Duration?}
    Analyzer --> Resolution{Resolution?}
    Analyzer --> Format{Format?}

    Duration -->|Short < 30s| ShortStrategy[Frame-Based Strategy<br/>ColPali]
    Duration -->|Medium 30s-5m| MediumStrategy[Chunk-Based Strategy<br/>ColQwen]
    Duration -->|Long > 5m| LongStrategy[Hybrid Strategy<br/>Multi-model]

    Resolution -->|Low < 720p| LowRes[Basic Processing]
    Resolution -->|High >= 720p| HighRes[Advanced Processing]

    Format -->|MP4| DirectProcess[Direct Processing]
    Format -->|Other| Convert[Convert Format]

    ShortStrategy --> Combine[Combine Strategies<br/>cogniverse_runtime]
    MediumStrategy --> Combine
    LongStrategy --> Combine
    LowRes --> Combine
    HighRes --> Combine
    DirectProcess --> Combine
    Convert --> Combine

    Combine --> TenantRoute[TenantSchemaManager<br/>cogniverse_vespa<br/>Route to video_frames_acme]

    TenantRoute --> Execute[Execute Ingestion]

    style Start fill:#e1f5ff
    style Execute fill:#e1ffe1
```

---

## Production Deployment Flows

### Scenario 22: SDK Package Deployment

```mermaid
graph TB
    Code[Code Changes<br/>in libs/ packages] --> Tests[Run Tests<br/>JAX_PLATFORM_NAME=cpu uv run pytest]

    Tests --> UnitPass{Unit Tests Pass?}

    UnitPass -->|No| Fix[Fix Issues]
    Fix --> Tests

    UnitPass -->|Yes| Integration[Integration Tests<br/>Multi-tenant + SDK]
    Integration --> IntPass{Integration Pass?}

    IntPass -->|No| Fix
    IntPass -->|Yes| BuildPkg[Build Packages<br/>uv build libs/runtime<br/>uv build libs/dashboard]

    BuildPkg --> Containers[Build Docker Containers<br/>cogniverse_runtime<br/>cogniverse_dashboard]

    Containers --> Stage[Deploy to Staging<br/>With tenant isolation]

    Stage --> StageTest[Staging Tests<br/>Multi-tenant validation]
    StageTest --> StagePass{Tests Pass?}

    StagePass -->|No| Rollback[Rollback]
    StagePass -->|Yes| Prod[Deploy to Production<br/>Blue-Green deployment]

    Prod --> Monitor[Monitor Metrics<br/>Per-tenant Phoenix projects]
    Monitor --> Health{Healthy?}

    Health -->|No| Rollback
    Health -->|Yes| Complete[Deployment Complete]

    style Code fill:#e1f5ff
    style Complete fill:#e1ffe1
    style Rollback fill:#ffe1e1
```

### Scenario 23: Blue-Green Deployment with Multi-Tenancy

```mermaid
sequenceDiagram
    participant Traffic
    participant LB as Load Balancer
    participant Blue as Blue Environment<br/>cogniverse_runtime v1.0
    participant Green as Green Environment<br/>cogniverse_runtime v2.0
    participant Monitor as Monitor<br/>cogniverse_core
    participant Evaluator

    Note over Blue: Current Production<br/>All tenants: acme, startup, enterprise
    Note over Green: New Version Deployed<br/>Ready for A/B testing

    Traffic->>LB: User requests (all tenants)
    LB->>Blue: 90% traffic (all tenants)
    LB->>Green: 10% traffic (sampled tenants)

    Blue-->>Monitor: Metrics (baseline)<br/>Per-tenant Phoenix projects
    Green-->>Monitor: Metrics (new version)<br/>Per-tenant Phoenix projects

    Monitor->>Evaluator: Compare performance<br/>Aggregate across tenants

    alt Performance Improved (all tenants)
        Evaluator->>LB: Increase Green traffic
        LB->>Blue: 50% traffic
        LB->>Green: 50% traffic

        Note over Evaluator: Continue monitoring<br/>per tenant...

        Evaluator->>LB: Full cutover
        LB->>Green: 100% traffic (all tenants)

        Evaluator->>Blue: Decommission old version
    else Performance Degraded (any tenant)
        Evaluator->>LB: Rollback
        LB->>Blue: 100% traffic (all tenants)

        Evaluator->>Green: Debug and fix<br/>Tenant-specific issues
    end
```

---

## Key Takeaways

### SDK Architecture Principles
1. **Package Boundaries**: Clear separation between core, agents, vespa, runtime, dashboard
2. **Dependency Flow**: Core is foundational → Vespa builds on Core → Agents uses both → Runtime consumes all
3. **UV Workspace**: Monorepo with independent package versioning
4. **Import Paths**: All imports use `cogniverse_*` package names

### Multi-Tenant Design Patterns
1. **Schema-Per-Tenant**: Physical isolation via dedicated Vespa schemas
2. **Tenant Context Injection**: Middleware layer extracts and validates tenant_id
3. **Per-Tenant Singletons**: Mem0MemoryManager maintains isolated instances
4. **Tenant-Scoped Telemetry**: Phoenix projects per tenant for complete observability

### Critical Integration Points
1. **Runtime ↔ TenantSchemaManager**: Automatic schema routing and lazy creation
2. **Agents ↔ Vespa**: Tenant-aware search clients with schema resolution
3. **Memory ↔ Vespa**: Tenant-specific memory schemas (agent_memories_{tenant_id})
4. **Phoenix ↔ All Packages**: Tenant-scoped span collection and metrics

### Data Flow Patterns
1. **Request Flow**: User → Runtime (Middleware extracts tenant_id) → Agents (tenant-aware) → Vespa (tenant schema)
2. **Optimization Flow**: Phoenix (tenant project) → Evaluator → GRPO (tenant-specific) → Agents
3. **Memory Flow**: Query → Mem0 (tenant singleton) → Vespa (agent_memories_{tenant_id}) → Context
4. **Ingestion Flow**: Video → Runtime → TenantSchemaManager → Vespa (video_frames_{tenant_id})

---

**Related Guides:**
- [architecture/overview.md](./overview.md) - SDK and multi-tenant architecture
- [architecture/sdk-architecture.md](./sdk-architecture.md) - UV workspace deep dive
- [architecture/multi-tenant.md](./multi-tenant.md) - Tenant isolation guide
- [modules/](../modules/) - Per-package technical details

---

**Version**: 2.0 (SDK Architecture + Multi-Tenancy)
**Last Updated**: 2025-10-15
**Status**: Production-Ready

**Note:** All diagrams use Mermaid syntax and render in:
- GitHub markdown files
- IDEs with Mermaid support (VS Code, IntelliJ)
- Documentation sites (GitBook, Docusaurus)
- Mermaid Live Editor (https://mermaid.live)
