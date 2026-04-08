# A2A Plan 2: Core Restructuring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gut routing_agent.py to a thin DSPy decision-maker, enhance OrchestratorAgent with streaming/checkpoints/fusion/intelligence/cancellation from MultiAgentOrchestrator, and rewire agent_dispatcher.py to use GatewayAgent as entry point.

**Architecture:** RoutingAgent becomes thin (~200 LOC, accepts enriched query, outputs agent recommendation). OrchestratorAgent gains 5 features from MultiAgentOrchestrator and coordinates all A2A agents via HTTP. Dispatcher routes through GatewayAgent → OrchestratorAgent for complex queries.

**Tech Stack:** Python 3.12, DSPy 3.0, httpx (A2A HTTP), OpenTelemetry, Pydantic v2, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-08-a2a-architecture-restructuring-design.md`

**Depends on:** Plan 1 (Foundation Agents) — all 5 tasks complete.

---

## File Structure

### Modified files (major changes)
| File | What changes |
|------|-------------|
| `libs/agents/cogniverse_agents/routing_agent.py` | Gut from ~1865 to ~400 LOC. Delete all inline preprocessing. Keep thin DSPy routing decision. |
| `libs/agents/cogniverse_agents/orchestrator_agent.py` | Port 5 features from multi_agent_orchestrator.py. Replace hardcoded AgentType with dynamic registry. |
| `libs/runtime/cogniverse_runtime/agent_dispatcher.py` | Rewire dispatch: GatewayAgent entry → OrchestratorAgent for complex. Remove MultiAgentOrchestrator usage. |
| `libs/runtime/cogniverse_runtime/config_loader.py` | Add orchestrator_agent entry |
| `configs/config.json` | Add orchestrator_agent config |

### Test files
| File | What changes |
|------|-------------|
| `tests/agents/unit/test_routing_agent.py` (or equivalent) | Update tests for thin RoutingAgent interface |
| `tests/agents/unit/test_orchestrator_agent.py` | Add tests for 5 ported features |
| `tests/agents/unit/test_complete_multi_agent_orchestration.py` | Update for new architecture |

---

### Task 1: Gut RoutingAgent — Delete Inline Preprocessing

**Files:**
- Modify: `libs/agents/cogniverse_agents/routing_agent.py`
- Modify: tests for routing_agent

This is the largest single change. routing_agent.py goes from ~1865 LOC to ~400 LOC.

- [ ] **Step 1: Read the current routing_agent.py and map what stays vs goes**

Read `libs/agents/cogniverse_agents/routing_agent.py` fully. Identify:
- **KEEP:** `_initialize_telemetry_manager`, `_configure_dspy`, `_initialize_routing_module`, `_create_fallback_routing_module`, `_get_telemetry_provider`, `_ensure_memory_for_tenant`, `_make_routing_decision`, `_extract_reasoning`, `_process_impl`, `_get_agent_skills`, standalone FastAPI app
- **DELETE:** `_initialize_enhancement_pipeline`, `_initialize_advanced_optimizer`, `_get_optimizer`, `_get_cross_modal_optimizer`, `_initialize_mlflow_tracking`, `_initialize_production_components`, `_agent_to_modality`, `_analyze_and_enhance_query`, `_apply_grpo_optimization`, `_prepare_routing_context`, `_calibrate_confidence`, `_get_fallback_agents`, `_assess_orchestration_need`, `_get_orchestration_signals`, `_create_fallback_decision`, `_update_routing_stats`, `get_routing_statistics`, `record_routing_outcome`, `analyze_and_route_with_relationships`, `get_grpo_status`, `reset_grpo_optimization`, `get_mlflow_status`, `start_mlflow_run`, `log_optimization_metrics`, `save_dspy_model`, `cleanup_mlflow`

- [ ] **Step 2: Update RoutingInput to accept pre-enriched data**

The thin RoutingAgent receives entities, relationships, and enhanced_query from upstream agents. Update `RoutingInput`:

```python
class RoutingInput(AgentInput):
    """Input for thin routing decision."""
    query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query from QueryEnhancementAgent")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Entities from EntityExtractionAgent")
    relationships: Optional[List[Dict[str, Any]]] = Field(None, description="Relationships from EntityExtractionAgent")
    tenant_id: str = Field(..., description="Tenant identifier (required)")
```

- [ ] **Step 3: Simplify RoutingOutput**

Remove fields that the routing agent no longer produces (it no longer does entity extraction or query enhancement):

```python
class RoutingOutput(AgentOutput):
    """Output from thin routing decision."""
    query: str = Field(..., description="Original query")
    recommended_agent: str = Field(..., description="Selected execution agent")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Routing confidence")
    reasoning: str = Field("", description="Why this agent was selected")
    fallback_agents: List[str] = Field(default_factory=list, description="Fallback agents if primary fails")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Routing metadata")
```

Remove: `enhanced_query`, `entities`, `relationships`, `query_variants`, `timestamp` — these are now produced by upstream agents, not the routing agent.

- [ ] **Step 4: Simplify RoutingDeps**

Remove all preprocessing-related deps. Keep only routing decision deps:

```python
class RoutingDeps(AgentDeps):
    """Dependencies for thin routing agent."""
    telemetry_config: Any = Field(None, description="Telemetry configuration")
    llm_config: Any = Field(None, description="LLM endpoint config for DSPy")
    confidence_threshold: float = Field(0.5, description="Minimum confidence for routing")
    enable_memory: bool = Field(False, description="Enable memory for strategy retrieval")
    # Memory-related fields (only used when enable_memory=True)
    memory_backend_host: Optional[str] = Field(None)
    memory_backend_port: Optional[int] = Field(None)
    memory_llm_model: Optional[str] = Field(None)
    memory_embedding_model: Optional[str] = Field(None)
    memory_llm_base_url: Optional[str] = Field(None)
    memory_config_manager: Optional[Any] = Field(None)
    memory_schema_loader: Optional[Any] = Field(None)
```

Remove: `relationship_weight`, `enhancement_weight`, `entity_confidence_threshold`, `relationship_confidence_threshold`, `enable_relationship_extraction`, `enable_query_enhancement`, `enable_fallback_routing`, `enable_confidence_calibration`, `enable_advanced_optimization`, `enable_caching`, `enable_parallel_execution`, `cache_ttl_seconds`, `query_fusion_config`, `agent_capabilities` mapping.

- [ ] **Step 5: Delete all inline preprocessing imports**

Remove these imports from the top of the file:
```python
# DELETE these imports:
from cogniverse_agents.routing.advanced_optimizer import ...
from cogniverse_agents.routing.contextual_analyzer import ...
from cogniverse_agents.routing.cross_modal_optimizer import ...
from cogniverse_agents.routing.dspy_relationship_router import ...
from cogniverse_agents.routing.lazy_executor import ...
from cogniverse_agents.routing.mlflow_integration import ...
from cogniverse_agents.routing.modality_cache import ...
from cogniverse_agents.routing.modality_metrics import ...
from cogniverse_agents.routing.parallel_executor import ...
from cogniverse_agents.routing.query_enhancement_engine import ...
from cogniverse_agents.routing.relationship_extraction_tools import ...
from cogniverse_agents.search.multi_modal_reranker import ...
```

- [ ] **Step 6: Gut the `__init__` method**

New `__init__` should only:
1. Call super().__init__()
2. Initialize telemetry manager
3. Configure DSPy LM
4. Initialize routing module (DSPy)
5. Set up memory (if enabled)

Remove all calls to: `_initialize_enhancement_pipeline`, `_initialize_advanced_optimizer`, `_initialize_mlflow_tracking`, `_initialize_production_components`.

- [ ] **Step 7: Simplify `route_query` method**

The new `route_query` should:
1. Validate tenant_id
2. Set up telemetry span
3. Set tenant for memory context
4. Call `_make_routing_decision` with the pre-enriched data from input
5. Return RoutingOutput

Remove all: cache checking, entity extraction, query enhancement, GRPO optimization, confidence calibration, orchestration assessment, parallel execution, MLflow tracking.

- [ ] **Step 8: Simplify `_make_routing_decision`**

The routing decision now receives entities/relationships/enhanced_query as parameters (from upstream agents), not extracted inline. Update to accept these as parameters and pass to the DSPy module:

```python
async def _make_routing_decision(
    self, query: str, enhanced_query: Optional[str],
    entities: Optional[List[Dict]], relationships: Optional[List[Dict]],
    context: Optional[str], tenant_id: str,
) -> Dict[str, Any]:
    # Build routing context from pre-enriched data
    routing_context = ""
    if entities:
        entity_str = ", ".join(e.get("text", "") for e in entities[:5])
        routing_context += f"Entities: {entity_str}. "
    if relationships:
        rel_str = "; ".join(
            f"{r.get('subject','')} {r.get('relation','')} {r.get('object','')}"
            for r in relationships[:3]
        )
        routing_context += f"Relationships: {rel_str}. "

    # Inject memory strategies if available
    if self.is_memory_enabled():
        routing_context = self.inject_context_into_prompt(routing_context, query)

    effective_query = enhanced_query or query
    available_agents = "search_agent, summarizer_agent, detailed_report_agent, image_search_agent, audio_analysis_agent, document_agent, deep_research_agent"

    with dspy.context(lm=self._dspy_lm):
        dspy_result = await self.call_dspy(
            self.routing_module,
            output_field="recommended_agent",
            query=effective_query,
            context=routing_context,
            available_agents=available_agents,
        )

    return {
        "recommended_agent": getattr(dspy_result, "recommended_agent", "search_agent"),
        "confidence": self._parse_confidence(getattr(dspy_result, "confidence", "0.5")),
        "reasoning": self._extract_reasoning(dspy_result),
    }
```

- [ ] **Step 9: Delete all removed methods**

Delete every method marked DELETE in Step 1. This should remove ~1400 LOC.

- [ ] **Step 10: Update `_process_impl` and `_get_agent_skills`**

`_process_impl` delegates to `route_query`. Update to pass the new input fields (entities, relationships, enhanced_query).

Update `_get_agent_skills` output_schema to match the new slimmer RoutingOutput.

- [ ] **Step 11: Update tests**

Read existing routing agent tests. Update:
- Mock setup: no longer need to mock GLiNER, SpaCy, enhancement pipeline
- Input: pass entities/relationships/enhanced_query in RoutingInput
- Output: assert on slimmer RoutingOutput (no entities/relationships/query_variants)
- Remove tests for deleted methods (GRPO, MLflow, enhancement pipeline, etc.)

- [ ] **Step 12: Run tests and lint**

```bash
uv run pytest tests/agents/unit/ -k "routing" --tb=long -q
uv run ruff check libs/agents/cogniverse_agents/routing_agent.py
```

- [ ] **Step 13: Commit**

```bash
git add libs/agents/cogniverse_agents/routing_agent.py tests/
git commit -m "Gut RoutingAgent to thin DSPy decision-maker, remove inline preprocessing"
```

---

### Task 2: Enhance OrchestratorAgent — Streaming Support

**Files:**
- Modify: `libs/agents/cogniverse_agents/orchestrator_agent.py`
- Modify: `tests/agents/unit/test_orchestrator_agent.py`

Port streaming from `multi_agent_orchestrator.py` lines 271-600.

- [ ] **Step 1: Read existing orchestrator_agent.py and multi_agent_orchestrator.py streaming code**

Understand the current `_execute_plan` method and how streaming events work in `_process_complex_query_stream`.

- [ ] **Step 2: Add EventQueue support to OrchestratorAgent**

Add `event_queue` parameter to constructor:
```python
def __init__(
    self, deps: OrchestratorDeps, registry: "AgentRegistry",
    config_manager=None, port: int = 8013,
    event_queue: Optional[EventQueue] = None,
):
    ...
    self.event_queue = event_queue
```

Add `_emit_event` method:
```python
async def _emit_event(self, event: Dict[str, Any]) -> None:
    """Emit event to event queue (for streaming consumers)."""
    if self.event_queue is not None:
        await self.event_queue.enqueue(event)
```

- [ ] **Step 3: Add streaming to `_process_impl`**

When the agent is processing in streaming mode (detected via `self._progress_queue`), emit progress events at each phase:
- `emit_progress("planning", "Creating orchestration plan...")`
- `emit_progress("executing", f"Executing step {i}: {agent_name}...", {"step": i})`
- `emit_progress("aggregating", "Aggregating results...")`
- `emit_progress("complete", "Orchestration complete", {"result": final_output})`

Also emit events via `_emit_event` for external consumers (EventQueue).

- [ ] **Step 4: Write tests for streaming**

```python
@pytest.mark.asyncio
async def test_orchestrator_emits_progress_events(self):
    """OrchestratorAgent should emit progress events during processing."""
    agent = _make_orchestrator()
    # Mock registry to return fake agents
    # Process with stream=True
    # Collect events from async generator
    # Assert planning, executing, aggregating, complete events emitted
```

- [ ] **Step 5: Run tests, commit**

```bash
uv run pytest tests/agents/unit/test_orchestrator_agent.py --tb=long -q
git commit -m "Add streaming support to OrchestratorAgent"
```

---

### Task 3: Enhance OrchestratorAgent — Checkpoint/Resume

**Files:**
- Modify: `libs/agents/cogniverse_agents/orchestrator_agent.py`
- Modify: `tests/agents/unit/test_orchestrator_agent.py`

Port checkpointing from `multi_agent_orchestrator.py` lines 1109-1283.

- [ ] **Step 1: Add checkpoint types to OrchestratorAgent**

Import checkpoint types:
```python
from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig, CheckpointStatus, TaskCheckpoint, WorkflowCheckpoint,
)
from cogniverse_agents.orchestrator.checkpoint_storage import WorkflowCheckpointStorage
```

Add to constructor:
```python
self.checkpoint_config = checkpoint_config
self.checkpoint_storage = checkpoint_storage
```

- [ ] **Step 2: Add `_save_checkpoint` method**

Port from multi_agent_orchestrator.py lines 1109-1172. Creates TaskCheckpoint for each step, builds WorkflowCheckpoint, saves via storage.

- [ ] **Step 3: Add `resume_workflow` method**

Port from multi_agent_orchestrator.py lines 1174-1282. Loads latest checkpoint, reconstructs plan, resumes from last successful step.

- [ ] **Step 4: Integrate checkpointing into `_execute_plan`**

After each step completes, call `_save_checkpoint` if checkpoint_storage is configured.

- [ ] **Step 5: Write tests for checkpointing**

```python
@pytest.mark.asyncio
async def test_checkpoint_saved_after_each_step(self):
    """Checkpoint should be saved after each step completes."""

@pytest.mark.asyncio
async def test_resume_workflow_from_checkpoint(self):
    """Should resume from last successful step."""
```

- [ ] **Step 6: Run tests, commit**

```bash
uv run pytest tests/agents/unit/test_orchestrator_agent.py --tb=long -q
git commit -m "Add checkpoint/resume support to OrchestratorAgent"
```

---

### Task 4: Enhance OrchestratorAgent — Cross-Modal Fusion

**Files:**
- Modify: `libs/agents/cogniverse_agents/orchestrator_agent.py`
- Modify: `tests/agents/unit/test_orchestrator_agent.py`

Port cross-modal fusion from `multi_agent_orchestrator.py` lines 1341-1840.

- [ ] **Step 1: Add FusionStrategy enum and ResultAggregatorSignature**

Port from multi_agent_orchestrator.py:
```python
class FusionStrategy(Enum):
    SCORE_BASED = "score"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    SIMPLE = "simple"
```

And the DSPy signature:
```python
class ResultAggregatorSignature(dspy.Signature):
    """Aggregate results from multiple agents with cross-modal fusion."""
    original_query = dspy.InputField()
    task_results = dspy.InputField()
    fusion_strategy = dspy.InputField()
    agent_modalities = dspy.InputField()
    aggregated_result = dspy.OutputField()
    confidence_score = dspy.OutputField()
    fusion_quality = dspy.OutputField()
    cross_modal_consistency = dspy.OutputField()
```

- [ ] **Step 2: Replace `_aggregate_results` with fusion-aware version**

Port the full `_aggregate_results` from multi_agent_orchestrator.py lines 1341-1427, including:
- Modality detection per task result
- Fusion strategy selection
- Dispatch to fusion methods
- Quality calculation

- [ ] **Step 3: Port fusion methods**

Port: `_select_fusion_strategy`, `_fuse_by_score`, `_fuse_by_temporal_alignment`, `_fuse_by_semantic_similarity`, `_fuse_hierarchically`, `_fuse_simple`, `_check_cross_modal_consistency`, `_calculate_fusion_quality`.

- [ ] **Step 4: Write tests for fusion**

```python
@pytest.mark.asyncio
async def test_cross_modal_fusion_video_and_text(self):
    """Fusion should merge video and text results."""

def test_fusion_strategy_selection(self):
    """Should select appropriate fusion strategy based on modalities."""
```

- [ ] **Step 5: Run tests, commit**

```bash
uv run pytest tests/agents/unit/test_orchestrator_agent.py --tb=long -q
git commit -m "Add cross-modal fusion to OrchestratorAgent"
```

---

### Task 5: Enhance OrchestratorAgent — Workflow Intelligence + Cancellation

**Files:**
- Modify: `libs/agents/cogniverse_agents/orchestrator_agent.py`
- Modify: `tests/agents/unit/test_orchestrator_agent.py`

Port workflow intelligence (template loading) and cancellation.

- [ ] **Step 1: Add workflow intelligence integration**

Import and initialize WorkflowIntelligence:
```python
from cogniverse_agents.workflow.intelligence import WorkflowIntelligence, create_workflow_intelligence

# In __init__:
if telemetry_provider and tenant_id:
    self.workflow_intelligence = create_workflow_intelligence(
        telemetry_provider=telemetry_provider,
        tenant_id=tenant_id,
    )
else:
    self.workflow_intelligence = None
```

Use templates in planning: load templates at startup, use as few-shot examples when matching templates are found.

- [ ] **Step 2: Add cancellation support**

Port `cancel_workflow` from multi_agent_orchestrator.py:
```python
async def cancel_workflow(self, workflow_id: str) -> bool:
    """Cancel an in-flight orchestration workflow."""
    workflow = self.active_workflows.get(workflow_id)
    if not workflow:
        return False
    workflow.status = WorkflowStatus.CANCELLED
    workflow.end_time = datetime.now()
    for task in workflow.tasks:
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.SKIPPED
    return True
```

Add `active_workflows` dict to track in-flight workflows.

- [ ] **Step 3: Add orchestration span emission**

Emit `cogniverse.orchestration` span with attributes:
```python
def _emit_orchestration_span(
    self, query, tenant_id, workflow_id, agent_sequence,
    execution_time, success, parallel_efficiency, tasks_completed,
):
    with self.telemetry_manager.span(
        name="cogniverse.orchestration",
        tenant_id=tenant_id,
        attributes={
            "orchestration.workflow_id": workflow_id,
            "orchestration.query": query[:200],
            "orchestration.agent_sequence": ",".join(agent_sequence),
            "orchestration.execution_time": execution_time,
            "orchestration.success": success,
            "orchestration.parallel_efficiency": parallel_efficiency,
            "orchestration.tasks_completed": tasks_completed,
        },
    ):
        pass
```

- [ ] **Step 4: Replace hardcoded AgentType enum with dynamic registry discovery**

Remove `AgentType` enum. In `_create_plan`, get available agents from `self.registry.list_agents()` instead of hardcoded enum values. Pass them as comma-separated string to the DSPy planner.

- [ ] **Step 5: Update OrchestrationSignature with gateway_context**

Add `gateway_context` input field:
```python
class OrchestrationPlannerSignature(dspy.Signature):
    query = dspy.InputField(desc="User query")
    available_agents = dspy.InputField(desc="Comma-separated registered agents")
    gateway_context = dspy.InputField(desc="Modality + generation type from GatewayAgent")
    conversation_context = dspy.InputField(desc="Previous conversation turns")
    agent_sequence = dspy.OutputField(desc="Ordered agent sequence")
    parallel_steps = dspy.OutputField(desc="Parallel step groups")
    reasoning = dspy.OutputField(desc="Plan explanation")
```

- [ ] **Step 6: Register OrchestratorAgent**

Add to config_loader.py AGENT_CLASSES:
```python
"orchestrator_agent": "cogniverse_agents.orchestrator_agent:OrchestratorAgent",
```

Add to config.json:
```json
"orchestrator_agent": {
    "enabled": true,
    "url": "http://localhost:8000",
    "capabilities": ["orchestration", "planning"],
    "timeout": 120
}
```

- [ ] **Step 7: Write tests for intelligence, cancellation, span, registry**

```python
@pytest.mark.asyncio
async def test_workflow_intelligence_templates_loaded(self):
    """Templates should be available from WorkflowIntelligence."""

@pytest.mark.asyncio
async def test_cancel_workflow(self):
    """Should cancel in-flight workflow."""

@pytest.mark.asyncio
async def test_orchestration_span_emitted(self):
    """Should emit cogniverse.orchestration span."""

def test_dynamic_agent_discovery(self):
    """Should discover agents from registry, not hardcoded enum."""
```

- [ ] **Step 8: Run tests, commit**

```bash
uv run pytest tests/agents/unit/test_orchestrator_agent.py --tb=long -q
git commit -m "Add workflow intelligence, cancellation, span emission, dynamic registry to OrchestratorAgent"
```

---

### Task 6: Rewire agent_dispatcher.py — GatewayAgent Entry Point

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/agent_dispatcher.py`
- Modify: tests for dispatcher

- [ ] **Step 1: Read current agent_dispatcher.py**

Understand the full dispatch flow, especially `_execute_routing_task` (lines 424-566).

- [ ] **Step 2: Add gateway capability branch to `dispatch()`**

In the capability routing logic, add:
```python
if "gateway" in capabilities:
    result = await self._execute_gateway_task(query, context, tenant_id)
```

- [ ] **Step 3: Implement `_execute_gateway_task`**

```python
async def _execute_gateway_task(
    self, query: str, context: Dict[str, Any], tenant_id: str,
) -> Dict[str, Any]:
    """Route query through GatewayAgent for triage."""
    from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps, GatewayInput

    deps = GatewayDeps()
    agent = await asyncio.to_thread(GatewayAgent, deps=deps)

    input_data = GatewayInput(query=query, tenant_id=tenant_id)
    result = await agent._process_impl(input_data)

    if result.complexity == "complex":
        # Forward to OrchestratorAgent
        return await self._execute_orchestration_task(
            query, context, tenant_id,
            gateway_context={
                "modality": result.modality,
                "generation_type": result.generation_type,
                "confidence": result.confidence,
            },
        )
    else:
        # Simple: dispatch directly to the routed agent
        return await self._execute_downstream_agent(
            agent_name=result.routed_to,
            query=query,
            tenant_id=tenant_id,
            top_k=context.get("top_k", 10),
            conversation_history=context.get("conversation_history", []),
        )
```

- [ ] **Step 4: Implement `_execute_orchestration_task`**

```python
async def _execute_orchestration_task(
    self, query: str, context: Dict[str, Any], tenant_id: str,
    gateway_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute full orchestration pipeline via OrchestratorAgent."""
    from cogniverse_agents.orchestrator_agent import (
        OrchestratorAgent, OrchestratorDeps, OrchestratorInput,
    )

    deps = OrchestratorDeps()
    agent = OrchestratorAgent(
        deps=deps,
        registry=self._registry,
        config_manager=self._config_manager,
    )

    input_data = OrchestratorInput(
        query=query,
        tenant_id=tenant_id,
        session_id=context.get("session_id"),
        conversation_history=context.get("conversation_history"),
    )

    result = await agent._process_impl(input_data)

    return {
        "status": "success",
        "agent": "orchestrator_agent",
        "message": f"Orchestrated '{query}' via A2A pipeline",
        "orchestration_result": result.model_dump() if hasattr(result, "model_dump") else result,
        "gateway_context": gateway_context,
    }
```

- [ ] **Step 5: Simplify `_execute_routing_task`**

Remove the `MultiAgentOrchestrator` branch (lines 498-539). The simplified version:
1. Create thin RoutingAgent
2. Call `route_query` (which now just makes the DSPy decision)
3. Call downstream agent with the result

Remove the `needs_orchestration` check — that logic is now in GatewayAgent.

- [ ] **Step 6: Update `create_streaming_agent` for gateway and orchestration**

Add branches for `"gateway"` and `"orchestration"` capabilities in `create_streaming_agent()`.

- [ ] **Step 7: Update the default dispatch entry point**

Change the "routing" capability branch to use the new simplified `_execute_routing_task` (no inline MultiAgentOrchestrator). In the main API router, the entry point should call GatewayAgent instead of RoutingAgent for unqualified queries.

- [ ] **Step 8: Write tests for new dispatch flow**

```python
@pytest.mark.asyncio
async def test_gateway_simple_routes_directly(self):
    """Simple query via gateway should route directly to execution agent."""

@pytest.mark.asyncio  
async def test_gateway_complex_routes_to_orchestrator(self):
    """Complex query via gateway should route to OrchestratorAgent."""

@pytest.mark.asyncio
async def test_simplified_routing_task_no_orchestrator(self):
    """_execute_routing_task should not create MultiAgentOrchestrator."""
```

- [ ] **Step 9: Run tests, commit**

```bash
uv run pytest tests/agents/unit/ -k "dispatch" --tb=long -q
uv run ruff check libs/runtime/cogniverse_runtime/agent_dispatcher.py
git commit -m "Rewire agent_dispatcher: GatewayAgent entry, OrchestratorAgent for complex"
```

---

### Task 7: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all agents unit tests**

```bash
uv run pytest tests/agents/unit/ --tb=long -q
```
Expected: All pass, 0 failed, 0 skipped.

- [ ] **Step 2: Run all routing unit tests**

```bash
uv run pytest tests/routing/unit/ --tb=long -q
```
Expected: All pass (may have failures from removed code — fix them).

- [ ] **Step 3: Run ruff on all changed files**

```bash
uv run ruff check libs/agents/cogniverse_agents/routing_agent.py libs/agents/cogniverse_agents/orchestrator_agent.py libs/runtime/cogniverse_runtime/agent_dispatcher.py
```

- [ ] **Step 4: Verify all imports**

```bash
uv run python -c "
from cogniverse_agents.routing_agent import RoutingAgent, RoutingInput, RoutingOutput
from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.gateway_agent import GatewayAgent
print('All core agents importable')
"
```

- [ ] **Step 5: Commit any remaining fixes**

```bash
git add -A && git status
# If changes: git commit -m "Fix verification issues from Plan 2"
```
