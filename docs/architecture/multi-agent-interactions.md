# Multi-Agent Interactions and Communication Flows

## Overview

This document describes how agents communicate with each other in the Cogniverse multi-agent system using Google's A2A (Agent-to-Agent) protocol. It provides comprehensive sequence diagrams showing the complete flow of messages between agents, the orchestrator, and the agent registry.

## A2A Protocol Basics

### Message Format

All agents communicate using standardized A2A Task messages:

```json
{
  "id": "task_uuid",
  "messages": [
    {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "User query or instruction"
        },
        {
          "type": "data",
          "data": {
            "key": "value",
            "additional_context": {}
          }
        }
      ]
    }
  ]
}
```

### Standard Endpoints

Every A2A agent exposes three standard endpoints:

1. **`POST /tasks/send`** - Main task processing endpoint
2. **`GET /.well-known/agent-card.json`** - Agent card with capabilities (Google A2A standard)
3. **`GET /health`** - Health check endpoint

Additional endpoints:
- **`GET /agent.json`** - Legacy agent card endpoint (alias for well-known URI)

### Agent Lifecycle

```text
Initialize → Self-Register → Start Server → Process Tasks → Shutdown
```

## 1. Agent Registry and Discovery Flow

### Agent Self-Registration

Agents can register themselves with the AgentRegistry using the Curated Registry pattern. The `A2AEndpointsMixin` provides a `register_with_registry(registry_url)` method that agents can use to self-register via HTTP POST to the registry's `/register` endpoint.

```mermaid
sequenceDiagram
    participant Agent as New Agent
    participant Registry as AgentRegistry
    participant Runtime as Runtime Server

    Note over Agent: Agent Initialization
    Agent->>Agent: A2AAgent.__init__(deps, config)
    Agent->>Agent: Load configuration
    Note over Agent: Optional self-registration<br/>(if using A2AEndpointsMixin)
    Agent->>Agent: register_with_registry(registry_url)

    Note over Agent,Registry: Self-Registration via HTTP
    Agent->>Registry: POST /register
    Note right of Agent: Registration Data:<br/>{name, url, capabilities,<br/>health_endpoint,<br/>process_endpoint, agent_card_url}

    Registry->>Registry: register_agent(agent_endpoint)
    Registry->>Registry: Update capability mapping
    Registry-->>Agent: 200 OK {status: "success", message: "registered"}

    Note over Agent: Start A2A Server
    Agent->>Agent: uvicorn.run(app, host, port)
    Agent->>Agent: Ready to process tasks

    Note over Runtime,Registry: Agent Discovery
    Runtime->>Registry: GET /capabilities/{capability}
    Registry->>Registry: Query capability mapping
    Registry-->>Runtime: {status, capability, agents: [{name, url, health_status}]}

    Note over Runtime,Agent: Task Execution
    Runtime->>Agent: POST /tasks/send (A2A message)
    Agent->>Agent: process(input: InputT) -> OutputT
    Agent-->>Runtime: A2A response
```

### Key Points

- **Self-Registration**: Agents using `A2AEndpointsMixin` can register themselves by calling `register_with_registry(registry_url)` which sends a POST request to the registry's `/register` endpoint
- **Capability-Based Discovery**: Runtime discovers agents by querying the registry endpoint `/capabilities/{capability}` (e.g., `/capabilities/search`, `/capabilities/profile_selection`)
- **Health Monitoring**: Registry periodically checks agent health via their `/health` endpoints
- **Dynamic Updates**: Agents can re-register to update capabilities by sending updated registration data

---

## 2. Complete Query Flow with Orchestrator

This diagram shows the full end-to-end flow from user query to final results, including parallel planning and ensemble search execution.

```mermaid
sequenceDiagram
    participant User
    participant Orch as OrchestratorAgent
    participant Registry as AgentRegistry
    participant Prof as ProfileSelectionAgent
    participant Entity as EntityExtractionAgent
    participant Search as SearchAgent
    participant Backend as Search Backend

    User->>Orch: POST /tasks/send<br/>{query: "show me robots playing soccer"}

    Note over Orch: PLANNING PHASE (Parallel Execution)

    par Profile Selection
        Orch->>Registry: GET /capabilities/profile_selection
        Registry-->>Orch: [ProfileSelectionAgent @ http://localhost:8011]

        Orch->>Prof: POST /tasks/send<br/>{query, available_profiles}
        Note over Prof: LLM Reasoning<br/>(DSPy ChainOfThought)
        Prof->>Prof: Analyze query complexity
        Prof->>Prof: Match to profile strengths
        Prof-->>Orch: {selected_profile: "colpali",<br/>confidence: 0.85, alternatives: [...], modality: "video"}
    and Entity Extraction
        Orch->>Registry: GET /capabilities/entity_extraction
        Registry-->>Orch: [EntityExtractionAgent @ http://localhost:8010]

        Orch->>Entity: POST /tasks/send<br/>{query}
        Entity->>Entity: Extract entities (DSPy ChainOfThought)
        Entity->>Entity: Classify entity types
        Entity-->>Orch: {entities: ["robots", "soccer"],<br/>entity_count: 2, dominant_types: ["CONCEPT"]}
    end

    Note over Orch: Planning Complete (< 500ms total)

    Note over Orch: ACTION PHASE

    Orch->>Registry: GET /capabilities/search
    Registry-->>Orch: [SearchAgent @ http://localhost:8002]

    alt Ensemble Mode (profiles specified)
        Orch->>Search: POST /tasks/send<br/>{query, profiles: ["colpali", "videoprism"], modality: "video"}

        Note over Search: Parallel Profile Execution
        par Profile 1: ColPali
            Search->>Backend: POST /search/<br/>{profile: "colpali", query_embedding}
            Backend-->>Search: Results 1 (ranked by ColPali similarity)
        and Profile 2: VideoPrism
            Search->>Backend: POST /search/<br/>{profile: "videoprism", query_embedding}
            Backend-->>Search: Results 2 (ranked by VideoPrism similarity)
        end

        Search->>Search: RRF Fusion<br/>score(doc) = Σ 1/(k+rank)
        Search-->>Orch: {results: [...fused_results...],<br/>metadata: {profiles_used, rrf_scores}}
    else Single Profile Mode (single profile)
        Orch->>Search: POST /tasks/send<br/>{query, modality: "video"}
        Search->>Backend: POST /search/
        Backend-->>Search: Results
        Search-->>Orch: {results: [...]}
    end

    Orch->>Orch: Aggregate results + metadata
    Orch-->>User: {results: [...],<br/>metadata: {planning_time, search_time, profiles_used}}
```

### Timeline Analysis

**Planning Phase (Parallel)**:

- Profile Selection: ~100-150ms (DSPy LLM inference)
- Entity Extraction: ~50-100ms (DSPy LLM inference)
- **Total**: ~150-200ms (limited by slowest agent)

**Action Phase (Sequential or Parallel)**:

- Single Profile Search: ~400-600ms
- Ensemble Search (2-3 profiles): ~500-700ms (parallel execution)
- RRF Fusion: ~5-10ms
- **Total**: ~500-700ms

**End-to-End Latency**: ~650-900ms

---

## 3. ProfileSelectionAgent Decision Flow

This flowchart shows how the ProfileSelectionAgent makes intelligent decisions about which profile to select based on query characteristics.

```mermaid
flowchart TD
    Start[<span style='color:#000'>A2A Task Received</span>] --> Parse[<span style='color:#000'>Parse Task Message</span>]
    Parse --> Extract[<span style='color:#000'>Extract Query Features</span>]

    Extract --> Entities[<span style='color:#000'>Count Entities</span>]
    Extract --> EntityTypes[<span style='color:#000'>Entity Types</span>]
    Extract --> Length[<span style='color:#000'>Query Length</span>]
    Extract --> Keywords[<span style='color:#000'>Visual/Temporal Keywords</span>]

    Entities --> Features[<span style='color:#000'>Aggregate Features</span>]
    EntityTypes --> Features
    Length --> Features
    Keywords --> Features

    Features --> LLM[<span style='color:#000'>LLM Reasoning via DSPy ChainOfThought</span>]

    LLM --> Reasoning{<span style='color:#000'>LLM Output</span>}
    Reasoning --> Profile[<span style='color:#000'>Selected Profile<br/>selected_profile</span>]
    Reasoning --> Confidence[<span style='color:#000'>Confidence Score</span>]
    Reasoning --> Explanation[<span style='color:#000'>Reasoning Text</span>]
    Reasoning --> Intent[<span style='color:#000'>Query Intent</span>]
    Reasoning --> Modality[<span style='color:#000'>Modality</span>]

    Profile --> Alternatives[<span style='color:#000'>Generate Alternatives<br/>alternatives list</span>]
    Confidence --> Alternatives
    Modality --> Alternatives

    Alternatives --> Response[<span style='color:#000'>Build A2A Response</span>]
    Intent --> Response
    Explanation --> Response

    Response --> Return[<span style='color:#000'>Return to Orchestrator</span>]

    style Start fill:#90caf9,stroke:#1565c0,color:#000
    style Parse fill:#ffcc80,stroke:#ef6c00,color:#000
    style Extract fill:#ffcc80,stroke:#ef6c00,color:#000
    style Entities fill:#ffcc80,stroke:#ef6c00,color:#000
    style EntityTypes fill:#ffcc80,stroke:#ef6c00,color:#000
    style Length fill:#ffcc80,stroke:#ef6c00,color:#000
    style Keywords fill:#ffcc80,stroke:#ef6c00,color:#000
    style Features fill:#ffcc80,stroke:#ef6c00,color:#000
    style LLM fill:#81d4fa,stroke:#0288d1,color:#000
    style Reasoning fill:#ffcc80,stroke:#ef6c00,color:#000
    style Profile fill:#a5d6a7,stroke:#388e3c,color:#000
    style Confidence fill:#a5d6a7,stroke:#388e3c,color:#000
    style Explanation fill:#a5d6a7,stroke:#388e3c,color:#000
    style Intent fill:#a5d6a7,stroke:#388e3c,color:#000
    style Modality fill:#a5d6a7,stroke:#388e3c,color:#000
    style Alternatives fill:#ffcc80,stroke:#ef6c00,color:#000
    style Response fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Return fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### Decision Criteria

**Profile Selection Factors**:

- Query complexity (simple/medium/complex)
- Query intent (text_search, video_search, image_search, etc.)
- Target modality (video, image, text, audio)
- Entity count and types
- LLM confidence score (0.0-1.0)

**Profile Selection Output**:

- Single selected_profile (best match)
- Confidence score
- Alternative profiles (ranked by relevance)
- Query analysis (intent, modality, complexity)

---

## 4. Ensemble Search with RRF Fusion

This diagram shows the detailed execution flow for ensemble search with Reciprocal Rank Fusion.

```mermaid
flowchart TB
    Start[<span style='color:#000'>User Query</span>] --> Encode{<span style='color:#000'>For Each Selected Profile</span>}

    Encode -->|Profile 1| Enc1[<span style='color:#000'>Encode with ColPali</span>]
    Encode -->|Profile 2| Enc2[<span style='color:#000'>Encode with VideoPrism</span>]
    Encode -->|Profile 3| Enc3[<span style='color:#000'>Encode with Qwen</span>]

    Enc1 --> Search1[<span style='color:#000'>Backend Search<br/>Profile: colpali</span>]
    Enc2 --> Search2[<span style='color:#000'>Backend Search<br/>Profile: videoprism</span>]
    Enc3 --> Search3[<span style='color:#000'>Backend Search<br/>Profile: qwen</span>]

    Search1 --> Results1[<span style='color:#000'>Results 1<br/>Ranked by ColPali</span>]
    Search2 --> Results2[<span style='color:#000'>Results 2<br/>Ranked by VideoPrism</span>]
    Search3 --> Results3[<span style='color:#000'>Results 3<br/>Ranked by Qwen</span>]

    Results1 --> RRF[<span style='color:#000'>RRF Fusion Algorithm</span>]
    Results2 --> RRF
    Results3 --> RRF

    RRF --> Calc[<span style='color:#000'>For each document:<br/>score = Σ 1/(k + rank_in_profile)</span>]
    Calc --> Sort[<span style='color:#000'>Sort by RRF Score<br/>Descending</span>]
    Sort --> TopN[<span style='color:#000'>Select Top N Results</span>]

    TopN --> Metadata[<span style='color:#000'>Add Fusion Metadata</span>]
    Metadata --> Final[<span style='color:#000'>Return Fused Results</span>]

    style Start fill:#90caf9,stroke:#1565c0,color:#000
    style Encode fill:#ffcc80,stroke:#ef6c00,color:#000
    style Enc1 fill:#81d4fa,stroke:#0288d1,color:#000
    style Enc2 fill:#81d4fa,stroke:#0288d1,color:#000
    style Enc3 fill:#81d4fa,stroke:#0288d1,color:#000
    style Search1 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Search2 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Search3 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Results1 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Results2 fill:#a5d6a7,stroke:#388e3c,color:#000
    style Results3 fill:#a5d6a7,stroke:#388e3c,color:#000
    style RRF fill:#ffcc80,stroke:#ef6c00,color:#000
    style Calc fill:#ffcc80,stroke:#ef6c00,color:#000
    style Sort fill:#ffcc80,stroke:#ef6c00,color:#000
    style TopN fill:#ffcc80,stroke:#ef6c00,color:#000
    style Metadata fill:#b0bec5,stroke:#546e7a,color:#000
    style Final fill:#a5d6a7,stroke:#388e3c,color:#000
```

### RRF Algorithm Details

**Formula**:
```text
score(document) = Σ_profiles (1 / (k + rank_in_profile))
```

**Parameters**:

- `k = 60` (default) - Controls weight of top-ranked documents
- Profiles: 2-3 profiles maximum for optimal performance
- Rank: 0-indexed position in each profile's result list

**Example Calculation**:

Document appears in 3 profiles at ranks [2, 5, 1]:

```text
RRF score = 1/(60+2) + 1/(60+5) + 1/(60+1)
         = 1/62 + 1/65 + 1/61
         = 0.0161 + 0.0154 + 0.0164
         = 0.0479
```

**Complexity**: O(n_profiles × n_results) ~ 5-10ms for typical case

---

## 5. Agent Registry Health Monitoring

The AgentRegistry continuously monitors agent health and updates status.

```mermaid
sequenceDiagram
    participant Registry as AgentRegistry
    participant Agent1 as SearchAgent
    participant Agent2 as ProfileSelectionAgent
    participant Agent3 as EntityExtractionAgent

    Note over Registry: Health Check Cycle (every 60s)

    Registry->>Registry: health_check_all()

    par Health Check Agent 1
        Registry->>Agent1: GET /health
        Agent1-->>Registry: {status: "healthy", metrics: {...}}
        Registry->>Registry: Update: search_agent = "healthy"
    and Health Check Agent 2
        Registry->>Agent2: GET /health
        Agent2-->>Registry: {status: "healthy", metrics: {...}}
        Registry->>Registry: Update: profile_selection_agent = "healthy"
    and Health Check Agent 3
        Registry->>Agent3: GET /health
        Note over Agent3: Agent Unavailable
        Agent3--xAgent3: Timeout (5s)
        Registry->>Registry: Update: entity_extraction_agent = "unreachable"
    end

    Note over Registry: Update Registry Stats
    Registry->>Registry: get_registry_stats()
    Registry->>Registry: {total_agents: 3,<br/>healthy_agents: 2,<br/>unhealthy_agents: 1}

    Note over Registry: Load Balancing Decision
    Registry->>Registry: get_load_balanced_agent("profile_selection")
    Registry->>Registry: Filter: healthy_agents only
    Registry->>Registry: Return: ProfileSelectionAgent
```

### Health Status States

- **healthy**: Agent responded with 200 OK
- **unhealthy**: Agent responded with non-200 status
- **unreachable**: Agent timed out or connection failed

### Load Balancing Strategy

1. **Filter**: Only healthy agents
2. **Round-Robin**: Simple rotation for multiple healthy agents
3. **Fallback**: If no healthy agents, use any registered agent
4. **Retry**: Automatic retry with backoff for failed requests

---

## 6. Complete Agent Lifecycle

This diagram shows the complete lifecycle of an agent from initialization to shutdown.

```mermaid
flowchart TB
    Start((<span style='color:#000'>Start</span>)) --> Initializing[<span style='color:#000'>Initializing</span>]

    Initializing --> ConfigLoading[<span style='color:#000'>Config Loading</span>]
    ConfigLoading --> DSPySetup[<span style='color:#000'>DSPy Setup</span>]
    DSPySetup --> SelfRegistration[<span style='color:#000'>Self Registration</span>]

    SelfRegistration --> Registering[<span style='color:#000'>Registering</span>]
    Registering --> ServerStartup[<span style='color:#000'>Server Startup</span>]
    ServerStartup --> Ready[<span style='color:#000'>Ready</span>]

    Ready --> Processing[<span style='color:#000'>Processing</span>]
    Processing --> DSPyProcessing[<span style='color:#000'>DSPy Processing</span>]
    DSPyProcessing --> ResponseSent[<span style='color:#000'>Response Sent</span>]
    ResponseSent --> Ready

    Ready --> HealthCheck[<span style='color:#000'>Health Check</span>]
    HealthCheck --> Ready

    Ready --> Shutdown[<span style='color:#000'>Shutdown</span>]
    Shutdown --> End((<span style='color:#000'>End</span>))

    style Start fill:#b0bec5,stroke:#546e7a,color:#000
    style Initializing fill:#90caf9,stroke:#1565c0,color:#000
    style ConfigLoading fill:#90caf9,stroke:#1565c0,color:#000
    style DSPySetup fill:#90caf9,stroke:#1565c0,color:#000
    style SelfRegistration fill:#ffcc80,stroke:#ef6c00,color:#000
    style Registering fill:#ffcc80,stroke:#ef6c00,color:#000
    style ServerStartup fill:#ffcc80,stroke:#ef6c00,color:#000
    style Ready fill:#a5d6a7,stroke:#388e3c,color:#000
    style Processing fill:#ce93d8,stroke:#7b1fa2,color:#000
    style DSPyProcessing fill:#ce93d8,stroke:#7b1fa2,color:#000
    style ResponseSent fill:#ce93d8,stroke:#7b1fa2,color:#000
    style HealthCheck fill:#a5d6a7,stroke:#388e3c,color:#000
    style Shutdown fill:#b0bec5,stroke:#546e7a,color:#000
    style End fill:#b0bec5,stroke:#546e7a,color:#000
```

**State Notes:**

- **SelfRegistration**: Only if agent_registry_url configured (Curated Registry pattern)
- **Processing**: Type-safe processing, multi-tenant isolation, telemetry tracking

---

## 7. Error Handling and Retry Logic

```mermaid
sequenceDiagram
    participant Orch as OrchestratorAgent
    participant Registry as AgentRegistry
    participant Agent as Target Agent

    Orch->>Registry: GET /capabilities/search
    Registry-->>Orch: [SearchAgent @ http://localhost:8002]

    Orch->>Agent: POST /tasks/send {query}

    alt Agent Healthy
        Agent-->>Orch: 200 OK {results}
        Orch->>Orch: Process results
    else Agent Timeout
        Agent--xOrch: Timeout (5s)
        Orch->>Registry: Mark agent unhealthy
        Orch->>Registry: GET /capabilities/search (retry)
        Registry-->>Orch: [SearchAgent (alternate) @ http://localhost:8002]
        Orch->>Agent: POST /tasks/send {query} (retry)
    else Agent Error
        Agent-->>Orch: 500 Internal Server Error
        Orch->>Orch: Log error to telemetry
        Orch->>Registry: Update agent status
        Orch->>Orch: Try fallback agent or return error
    end
```

---

## 8. Multi-Tenant Isolation

All agents support multi-tenant isolation using tenant_id for data and telemetry separation.

```mermaid
flowchart TD
    Request[<span style='color:#000'>A2A Task with tenant_id</span>] --> Extract[<span style='color:#000'>Extract tenant_id from task</span>]

    Extract --> Validate{<span style='color:#000'>tenant_id valid?</span>}
    Validate -->|No| Error[<span style='color:#000'>Return 400 Bad Request</span>]
    Validate -->|Yes| TenantContext[<span style='color:#000'>Create tenant context</span>]

    TenantContext --> Telemetry[<span style='color:#000'>Telemetry<br/>Project: cogniverse-tenant</span>]
    TenantContext --> Config[<span style='color:#000'>Load tenant configuration</span>]
    TenantContext --> Memory[<span style='color:#000'>Memory isolation</span>]

    Telemetry --> Process[<span style='color:#000'>Process with DSPy</span>]
    Config --> Process
    Memory --> Process

    Process --> Results[<span style='color:#000'>Return results</span>]
    Results --> Trace[<span style='color:#000'>Export telemetry trace<br/>to tenant-specific project</span>]

    style Request fill:#90caf9,stroke:#1565c0,color:#000
    style Extract fill:#ffcc80,stroke:#ef6c00,color:#000
    style Validate fill:#ffcc80,stroke:#ef6c00,color:#000
    style Error fill:#e53935,stroke:#c62828,color:#000
    style TenantContext fill:#90caf9,stroke:#1565c0,color:#000
    style Telemetry fill:#a5d6a7,stroke:#388e3c,color:#000
    style Config fill:#b0bec5,stroke:#546e7a,color:#000
    style Memory fill:#90caf9,stroke:#1565c0,color:#000
    style Process fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Results fill:#a5d6a7,stroke:#388e3c,color:#000
    style Trace fill:#a5d6a7,stroke:#388e3c,color:#000
```

---

## Performance Characteristics

### Latency Breakdown (Typical Query)

| Phase | Component | Latency | Parallelizable |
|-------|-----------|---------|----------------|
| Planning | Profile Selection | 100-150ms | ✅ Yes |
| Planning | Entity Extraction | 50-100ms | ✅ Yes |
| **Planning Total** | **Both agents** | **150-200ms** | **Parallel** |
| Action | Single Profile Search | 400-600ms | ❌ No |
| Action | Ensemble Search (2-3) | 500-700ms | ✅ Yes (profiles) |
| Action | RRF Fusion | 5-10ms | ❌ No |
| **Action Total** | **Search + Fusion** | **500-710ms** | **Profiles parallel** |
| **End-to-End** | **Complete flow** | **650-910ms** | **Planning + Action** |

### Network Calls

**Minimum** (Single profile, simple query):

- 1× Registry lookup (profile selection agent)
- 1× Registry lookup (search agent)
- 1× Profile selection task
- 1× Search task
- **Total**: 4 HTTP calls

**Maximum** (Ensemble with 3 profiles, complex query):

- 3× Registry lookups (planning agents + search agent)
- 2× Planning tasks (profile selection + entity extraction)
- 1× Search task → 3× backend queries (parallel)
- **Total**: 9 HTTP calls (6 parallel)

---

## See Also

- [Agents Module Documentation](../modules/agents.md) - Implementation details for all agents
- [Ensemble Composition](./ensemble-composition.md) - Deep dive into RRF algorithm
- [A2A Protocol Specification](https://github.com/google/a2a) - Google's A2A protocol spec
- [Multi-Tenant Architecture](./multi-tenant.md) - Tenant isolation design
