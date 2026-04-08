# A2A Architecture Restructuring — Design Spec

## Goal

Restructure the cogniverse multi-agent system so that preprocessing (entity extraction, query enhancement, profile selection) moves from inline code in `routing_agent.py` into dedicated A2A agents, `OrchestratorAgent` becomes the top-level coordinator (replacing `MultiAgentOrchestrator`), `routing_agent` becomes a thin decision-maker, and all optimization loops move from inline to Argo batch jobs via telemetry spans.

## Architecture

Three-layer A2A architecture: **Gateway** (fast triage) → **Orchestrator** (LLM-planned pipeline) → **Preprocessing + Routing + Execution agents** (each a standalone A2A service).

All agents emit telemetry spans. All learning/optimization runs as Argo CronWorkflow batch jobs reading those spans from Phoenix. No inline optimization loops.

## Tech Stack

- **A2A protocol**: Existing `A2AAgent` base class + `AgentRegistry` for discovery + `httpx` for inter-agent HTTP calls
- **DSPy**: `ChainOfThought` for agent decisions, teleprompters (GEPA/MIPROv2/SIMBA/BootstrapFewShot) for batch optimization
- **GLiNER**: `urchade/gliner_large-v2.1` for fast entity classification (GatewayAgent + EntityExtractionAgent)
- **SpaCy**: Dependency analysis for relationship extraction (EntityExtractionAgent)
- **Phoenix/OpenTelemetry**: Telemetry spans as the single source of truth for all learning
- **Argo Workflows**: CronWorkflows for batch optimization jobs
- **Mem0 + Vespa**: Persistent memory (interactions, strategies) — unchanged
- **ArtifactManager**: Stores optimized DSPy modules, workflow templates — loaded by agents at startup

---

## 1. Query Flow

```
User Query → agent_dispatcher.dispatch()
  → GatewayAgent (GLiNER classification, <100ms, no LLM)
    ├── SIMPLE → execution agent directly (search_agent, summarizer_agent, etc.)
    └── COMPLEX → OrchestratorAgent (A2A HTTP)
                    │
                    │  DSPy LLM planner decides steps dynamically:
                    ├── EntityExtractionAgent (A2A HTTP)
                    ├── QueryEnhancementAgent (A2A HTTP)
                    ├── RoutingAgent (A2A HTTP) — "which execution agent?"
                    ├── ProfileSelectionAgent (A2A HTTP) — "which Vespa profile?"
                    └── Execution agents (A2A HTTP)
                    │
                    │  Cross-cutting features:
                    ├── Streaming (SSE events as each agent completes)
                    ├── Checkpoint/resume (save/restore workflow state)
                    ├── Cross-modal fusion (DSPy ResultAggregatorSignature)
                    ├── Workflow intelligence (templates loaded from artifacts)
                    └── Cancellation (EventQueue-based)
```

### GatewayAgent decision logic

The GatewayAgent classifies queries using the GLiNER fast path extracted from `ComprehensiveRouter`. It outputs:
- `complexity`: `simple` or `complex`
- `modality`: `VIDEO`, `TEXT`, `AUDIO`, `IMAGE`, `DOCUMENT`, `BOTH`
- `generation_type`: `RAW_RESULTS`, `SUMMARY`, `DETAILED_REPORT`

Simple path mapping:
- `VIDEO` + `RAW_RESULTS` → `search_agent`
- `TEXT` + `RAW_RESULTS` → `search_agent`
- `*` + `SUMMARY` → `summarizer_agent`
- `*` + `DETAILED_REPORT` → `detailed_report_agent`
- `BOTH` or multi-modal or low confidence → `COMPLEX` → OrchestratorAgent

### OrchestratorAgent planning

Uses `DSPy ChainOfThought(OrchestrationPlannerSignature)` with inputs:
- `query`: the user query
- `available_agents`: comma-separated list of registered agents
- `gateway_context`: modality + generation_type from GatewayAgent
- `conversation_context`: multi-turn history (if any)

Outputs:
- `agent_sequence`: ordered list of agents to call
- `parallel_groups`: which steps can run concurrently
- `reasoning`: explanation of the plan

The planner uses workflow templates (loaded from artifacts at startup) as few-shot examples when available.

---

## 2. Agent Inventory

### 2.1 GatewayAgent (NEW)

**File**: `libs/agents/cogniverse_agents/gateway_agent.py`

**Source**: Extracted from `ComprehensiveRouter._fast_path_route()` in `routing/router.py` and `GLiNERRoutingStrategy` from `routing/strategies.py`.

**Responsibilities**:
- Load GLiNER model (`urchade/gliner_large-v2.1`)
- Classify query modality and generation type
- Determine simple vs complex
- For simple queries, map to execution agent name
- Emit `cogniverse.gateway` span

**A2A interface**:
- Input: `GatewayInput(query: str, tenant_id: Optional[str])`
- Output: `GatewayOutput(complexity: str, modality: str, generation_type: str, routed_to: str, confidence: float, reasoning: str)`

**Does NOT use MemoryAwareMixin** — pure classification, no memory needed.

### 2.2 OrchestratorAgent (ENHANCED)

**File**: `libs/agents/cogniverse_agents/orchestrator_agent.py` (existing, enhanced)

**What it gains from MultiAgentOrchestrator** (`orchestrator/multi_agent_orchestrator.py`):
1. **Streaming**: `_process_complex_query_stream()` — SSE events via `emit_progress()` as each A2A agent completes
2. **Checkpoint/resume**: `WorkflowCheckpointStorage` — save workflow state after each step, resume from last checkpoint on failure
3. **Cross-modal fusion**: `ResultAggregatorSignature` with `FusionStrategy` enum — DSPy-powered merging of results from multiple execution agents
4. **Workflow intelligence**: Load workflow templates from artifacts at startup. Use templates as few-shot examples in the planner. Record orchestration spans for batch optimization.
5. **Cancellation**: `EventQueue`-based cancellation tokens — cancel in-flight agent calls

**What it loses**:
- Hardcoded `AgentType` enum (6 types) → replaced by dynamic agent discovery via `AgentRegistry`
- In-process dispatch → replaced by A2A HTTP calls via `httpx`

**What changes**:
- `OrchestrationSignature` enhanced with `gateway_context` input field
- `_execute_step()` calls agents via A2A HTTP instead of in-process
- Loads workflow templates from `ArtifactManager` at startup (no inline DSPy optimization)

**Emits**: `cogniverse.orchestration` span with attributes: `workflow_id`, `query`, `agent_sequence`, `execution_time`, `success`, `parallel_efficiency`, `tasks_completed`

### 2.3 EntityExtractionAgent (ENHANCED)

**File**: `libs/agents/cogniverse_agents/entity_extraction_agent.py` (existing, enhanced)

**What it gains from routing_agent.py**:
- `GLiNERRelationshipExtractor` (`routing/dspy_relationship_router.py:35-101`) — real NER model with typed entity labels and confidence scores
- `SpaCyDependencyAnalyzer` — syntactic/dependency-based relationship inference
- LLM relationship extraction (slow path) — DSPy ChainOfThought for complex relationships that GLiNER/SpaCy miss

**Tiered approach**:
1. **Fast path**: GLiNER entity detection + SpaCy dependency analysis (no LLM, <200ms)
2. **Slow path**: DSPy ChainOfThought for relationship extraction (LLM call, when GLiNER confidence is low)

**A2A interface**:
- Input: `EntityExtractionInput(query: str, tenant_id: Optional[str])`
- Output: `EntityExtractionOutput(query: str, entities: List[Entity], relationships: List[Relationship], entity_count: int, has_entities: bool, dominant_types: List[str])`

**Emits**: `cogniverse.entity_extraction` span with attributes: `query`, `entity_count`, `relationship_count`, `entities_json`, `path_used` (fast/slow)

**Uses MemoryAwareMixin**: Yes — retrieves strategies and context for entity extraction.

### 2.4 QueryEnhancementAgent (ENHANCED)

**File**: `libs/agents/cogniverse_agents/query_enhancement_agent.py` (existing, enhanced)

**What it gains**:
- DSPy `ChainOfThought(QueryEnhancementSignature)` with optimized module loaded from artifacts at startup
- RRF query variant generation (multiple reformulations for fusion search)
- Synonym expansion, context additions

**What it does NOT have** (removed anti-patterns):
- No inline SIMBA pattern bank (eliminated — was 800+ LOC)
- No inline `SIMBA.compile()` calls
- No in-memory embedding cache
- No `record_enhancement_outcome()` — outcomes are telemetry spans

**A2A interface**:
- Input: `QueryEnhancementInput(query: str, entities: List[Entity], relationships: List[Relationship], tenant_id: Optional[str])`
- Output: `QueryEnhancementOutput(original_query: str, enhanced_query: str, expansion_terms: List[str], synonyms: List[str], query_variants: List[str], confidence: float, reasoning: str)`

**Emits**: `cogniverse.query_enhancement` span with attributes: `original_query`, `enhanced_query`, `strategy_used`, `variant_count`, `confidence`

**Uses MemoryAwareMixin**: Yes — retrieves strategies for enhancement guidance.

### 2.5 ProfileSelectionAgent (WIRED IN)

**File**: `libs/agents/cogniverse_agents/profile_selection_agent.py` (existing, wired into dispatch)

**No code changes needed** — already has DSPy profile selection with intent/modality/complexity classification and 4 Vespa profile candidates.

**Wiring changes**: Register in `config_loader.py`, add config.json entry, OrchestratorAgent calls it via A2A HTTP.

**Emits**: `cogniverse.profile_selection` span with attributes: `query`, `selected_profile`, `modality`, `complexity`, `reasoning`

### 2.6 RoutingAgent (GUTTED)

**File**: `libs/agents/cogniverse_agents/routing_agent.py` (existing, gutted from ~1200 to ~200 LOC)

**What remains**:
- DSPy `ChainOfThought` routing decision (~40 LOC of actual decision logic)
- `AdvancedRoutingOptimizer` integration — loads optimized DSPy module from artifacts
- `AdaptiveThresholdLearner` — loads learned thresholds from artifacts
- Confidence calibration
- A2A server boilerplate

**What is removed** (~1000 LOC):
- GLiNER initialization + `_initialize_gliner()` + entity extraction
- SpaCy dependency analysis initialization
- `ComposableQueryAnalysisModule` initialization and usage
- `QueryEnhancementPipeline` initialization and usage
- SIMBA initialization (`SIMBAQueryEnhancer`)
- `_analyze_and_enhance_query()` method
- All inline query enrichment logic
- `_prepare_routing_context()` (entities/relationships now come as input)
- GLiNER/SpaCy model loading in `__init__`

**New A2A interface**:
- Input: `RoutingInput(query: str, enhanced_query: str, entities: List[Entity], relationships: List[Relationship], tenant_id: str)`
- Output: `RoutingOutput(recommended_agent: str, confidence: float, reasoning: str, fallback_agents: List[str], metadata: Dict)`

**Emits**: `cogniverse.routing` span (already exists — keep as-is)

### 2.7 Execution Agents (UNCHANGED)

No changes: `search_agent`, `summarizer_agent`, `detailed_report_agent`, `image_search_agent`, `audio_analysis_agent`, `document_agent`, `deep_research_agent`, `coding_agent`, `text_analysis_agent`.

---

## 3. Telemetry-First Learning Architecture

### Principle

Agents emit telemetry spans. Argo batch jobs read spans from Phoenix and optimize. No inline optimization loops. No in-memory pattern banks.

### Request Path (what agents do)

1. Agent receives request
2. Loads optimized DSPy module from artifacts (loaded once at startup, refreshed periodically)
3. Retrieves strategies from Mem0 via `MemoryAwareMixin.get_strategies(query)`
4. Does its work (entity extraction, query enhancement, routing decision, etc.)
5. Emits telemetry span with attributes
6. Returns result

Agents do NOT:
- Run `SIMBA.compile()` or any DSPy teleprompter inline
- Maintain in-memory pattern banks or embedding caches
- Record outcomes to separate stores (telemetry spans ARE the record)

### Batch Path (Argo CronWorkflow jobs)

Each optimization job:
1. Reads spans from Phoenix for a time window (last N hours)
2. QualityMonitor scores the spans
3. Runs the appropriate optimization
4. Saves artifacts (optimized DSPy modules, templates, thresholds)
5. Optionally runs StrategyLearner to distill strategies into Mem0

**Optimization jobs and their span sources:**

| Argo Job | Reads Spans | Produces | Consumed By |
|---|---|---|---|
| SIMBA optimizer | `cogniverse.query_enhancement` | Optimized DSPy enhancement module | QueryEnhancementAgent (artifact) |
| Routing optimizer (GEPA/MIPROv2) | `cogniverse.routing` | Optimized DSPy routing module | RoutingAgent (artifact) |
| Workflow optimizer | `cogniverse.orchestration` | Workflow templates + agent performance profiles | OrchestratorAgent (artifact) |
| Threshold optimizer | `cogniverse.routing` + `cogniverse.gateway` | Adaptive threshold configs | RoutingAgent + GatewayAgent (artifact) |
| Strategy distiller | All scored spans | Strategies | All agents via Mem0 |
| Profile optimizer | `cogniverse.profile_selection` | Optimized profile selection module | ProfileSelectionAgent (artifact) |

### Span Definitions

Each agent emits a span with specific attributes that batch jobs read:

**`cogniverse.gateway`**:
- `gateway.query`, `gateway.complexity` (simple/complex), `gateway.modality`, `gateway.generation_type`, `gateway.routed_to`, `gateway.confidence`

**`cogniverse.entity_extraction`**:
- `entity_extraction.query`, `entity_extraction.entity_count`, `entity_extraction.relationship_count`, `entity_extraction.entities` (JSON), `entity_extraction.path_used` (fast/slow)

**`cogniverse.query_enhancement`**:
- `query_enhancement.original_query`, `query_enhancement.enhanced_query`, `query_enhancement.strategy_used`, `query_enhancement.variant_count`, `query_enhancement.confidence`

**`cogniverse.routing`** (already exists):
- `routing.query`, `routing.recommended_agent`, `routing.confidence`, `routing.reasoning`

**`cogniverse.profile_selection`**:
- `profile_selection.query`, `profile_selection.selected_profile`, `profile_selection.modality`, `profile_selection.reasoning`

**`cogniverse.orchestration`** (already exists):
- `orchestration.workflow_id`, `orchestration.query`, `orchestration.agent_sequence`, `orchestration.execution_time`, `orchestration.success`, `orchestration.parallel_efficiency`, `orchestration.tasks_completed`

---

## 4. Memory & Learning Systems

### 4.1 Mem0 Persistent Memory

**No architectural changes.** `MemoryAwareMixin` works via inheritance — any agent that extends it gets per-tenant, per-agent memory. The `agent_dispatcher._init_agent_memory()` handles initialization.

New agents that should use `MemoryAwareMixin`: `EntityExtractionAgent`, `QueryEnhancementAgent`, `OrchestratorAgent` (already has it), `RoutingAgent` (already has it).

`GatewayAgent` and `ProfileSelectionAgent` do NOT need memory — they're pure classifiers.

### 4.2 Strategy Learner

**No architectural changes.** Already runs as Argo batch job via `optimization_cli --mode triggered`. Reads Phoenix scored spans, distills into strategies, stores in Mem0. Agents retrieve via `MemoryAwareMixin.get_strategies(query)`.

### 4.3 DSPy Module Optimization

**Architectural change: all optimization moves to Argo batch jobs.**

Current state:
- SIMBA runs `SIMBA.compile()` inline during requests (every 50 patterns) — **eliminated**
- Routing optimizer already runs via Argo (`optimization_cli --mode dspy`) — **kept**
- Workflow Intelligence runs `_dspy_optimize_workflow()` inline — **eliminated**

New state: All DSPy optimization is Argo-only. Agents load optimized modules from `ArtifactManager` at startup.

### 4.4 Workflow Intelligence

**Architectural change: becomes a read-only template loader.**

Current `WorkflowIntelligence` does:
- Record workflow executions (inline) → **eliminated** — spans replace this
- DSPy workflow optimization (inline) → **eliminated** — Argo batch job
- Template matching and application → **kept** — OrchestratorAgent loads templates at startup
- Agent performance profiling → **kept** — batch job produces profiles, orchestrator loads them

The `WorkflowIntelligence` class is simplified to a template/profile loader. The batch job (Argo) reads orchestration spans, generates templates, and produces agent performance profiles.

---

## 5. Dispatch Layer Changes

### 5.1 config_loader.py

Add 5 new entries to `AGENT_CLASSES`:

```python
AGENT_CLASSES = {
    # NEW — gateway + preprocessing + orchestration
    "gateway_agent": "cogniverse_agents.gateway_agent:GatewayAgent",
    "entity_extraction_agent": "cogniverse_agents.entity_extraction_agent:EntityExtractionAgent",
    "query_enhancement_agent": "cogniverse_agents.query_enhancement_agent:QueryEnhancementAgent",
    "profile_selection_agent": "cogniverse_agents.profile_selection_agent:ProfileSelectionAgent",
    "orchestrator_agent": "cogniverse_agents.orchestrator_agent:OrchestratorAgent",
    # EXISTING — routing + execution
    "routing_agent": "cogniverse_agents.routing_agent:RoutingAgent",
    "search_agent": "cogniverse_agents.search_agent:SearchAgent",
    # ... rest unchanged
}
```

### 5.2 configs/config.json

Add agent entries with capabilities and URLs. All agents share the unified runtime URL (`http://localhost:8000`) and are routed by the runtime's agent router.

```json
{
  "agents": {
    "gateway_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["gateway", "classification"]
    },
    "orchestrator_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["orchestration", "planning"]
    },
    "entity_extraction_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["entity_extraction", "ner"]
    },
    "query_enhancement_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["query_enhancement", "expansion"]
    },
    "profile_selection_agent": {
      "enabled": true,
      "url": "http://localhost:8000",
      "capabilities": ["profile_selection"]
    }
  }
}
```

### 5.3 agent_dispatcher.py

**Major rewrite of `dispatch()` method:**

Current flow:
```
dispatch() → routing_agent.route_query()
  → if needs_orchestration: MultiAgentOrchestrator.process_complex_query()
  → else: call downstream agent directly
```

New flow:
```
dispatch() → GatewayAgent.classify()
  → if SIMPLE: call execution agent directly
  → if COMPLEX: OrchestratorAgent.orchestrate()
      → OrchestratorAgent calls A2A agents via HTTP (entity extraction, query enhancement, routing, profile selection, execution)
```

New capability branches needed:
```python
if "gateway" in capabilities:
    result = await self._execute_gateway_task(query, context, tenant_id)
elif "orchestration" in capabilities:
    result = await self._execute_orchestration_task(query, context, tenant_id)
elif "entity_extraction" in capabilities:
    result = await self._execute_entity_extraction_task(query, context, tenant_id)
elif "query_enhancement" in capabilities:
    result = await self._execute_query_enhancement_task(query, context, tenant_id)
elif "profile_selection" in capabilities:
    result = await self._execute_profile_selection_task(query, context, tenant_id)
```

The `_execute_routing_task()` method is simplified — no longer creates `MultiAgentOrchestrator`, no longer has the `needs_orchestration` branch.

---

## 6. Files Deleted

| File | Reason |
|---|---|
| `libs/agents/cogniverse_agents/orchestrator/multi_agent_orchestrator.py` | Replaced by enhanced OrchestratorAgent |
| `libs/agents/cogniverse_agents/routing/router.py` | ComprehensiveRouter extracted into GatewayAgent |
| `libs/agents/cogniverse_agents/routing/strategies.py` | GLiNERRoutingStrategy moves to GatewayAgent, other strategies unused |
| `libs/agents/cogniverse_agents/routing/query_enhancement_engine.py` | Logic moves into QueryEnhancementAgent |
| `libs/agents/cogniverse_agents/routing/simba_query_enhancer.py` | Inline pattern bank eliminated; SIMBA optimization is Argo batch-only |

## 7. Files Gutted

| File | Before | After | What's removed |
|---|---|---|---|
| `libs/agents/cogniverse_agents/routing_agent.py` | ~1200 LOC | ~200 LOC | GLiNER init, SpaCy init, ComposableQueryAnalysisModule, QueryEnhancementPipeline, SIMBA init, all inline enrichment |
| `libs/agents/cogniverse_agents/workflow/intelligence.py` | ~700 LOC | ~200 LOC | Inline DSPy optimization, runtime recording. Becomes template/profile loader only. |

---

## 8. Testing Strategy

Each agent gets its own test file following existing patterns:

- **Unit tests**: Mock A2A HTTP calls, test agent logic in isolation
- **Integration tests**: Real Phoenix telemetry, real A2A HTTP between agents, verify spans are emitted with correct attributes
- **End-to-end tests**: Full query flow from GatewayAgent through to execution agent, verify correct routing

Key test scenarios:
1. Simple query → GatewayAgent → direct to search_agent (no orchestration)
2. Complex query → GatewayAgent → OrchestratorAgent → full pipeline
3. OrchestratorAgent streaming (SSE events from each agent step)
4. OrchestratorAgent checkpoint/resume (simulate failure mid-workflow)
5. Cross-modal fusion (video + text results merged)
6. Artifact loading (agents load optimized DSPy modules at startup)
7. Span emission (each agent emits correct span attributes)
8. Argo batch job reads spans and produces artifacts

---

## 9. Argo Batch Job Changes

### Existing jobs (unchanged)
- `optimization_cli --mode once` — runs OptimizationOrchestrator
- `optimization_cli --mode triggered` — runs StrategyLearner on trigger datasets
- `optimization_cli --mode dspy` — runs DSPy routing optimization

### New jobs needed
- `optimization_cli --mode simba` — reads `cogniverse.query_enhancement` spans, builds training set, runs SIMBA.compile(), saves optimized module artifact
- `optimization_cli --mode workflow` — reads `cogniverse.orchestration` spans, generates workflow templates + agent performance profiles, saves artifacts
- `optimization_cli --mode gateway-thresholds` — reads `cogniverse.gateway` spans, updates GLiNER confidence thresholds, saves config artifact
- `optimization_cli --mode profile` — reads `cogniverse.profile_selection` spans, optimizes profile selection DSPy module, saves artifact
