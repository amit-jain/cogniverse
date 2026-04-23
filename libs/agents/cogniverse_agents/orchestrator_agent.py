"""
OrchestratorAgent - Autonomous A2A agent for coordinating multi-agent query processing.

Implements two-phase orchestration:
1. Planning Phase: Analyze query and create execution plan
2. Action Phase: Execute plan by coordinating specialized agents via A2A HTTP

Features:
- Streaming progress events per step
- Checkpoint/resume for durable execution
- Cross-modal fusion for multi-agent result aggregation
- Workflow intelligence (template matching + execution recording)
- Cancellation of in-flight workflows
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dspy
import httpx
from pydantic import BaseModel, Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig,
    CheckpointStatus,
    TaskCheckpoint,
    WorkflowCheckpoint,
)
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID

if TYPE_CHECKING:
    from cogniverse_agents.orchestrator.checkpoint_storage import (
        WorkflowCheckpointStorage,
    )
    from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
    from cogniverse_core.events import EventQueue
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_foundation.config.manager import ConfigManager

logger = logging.getLogger(__name__)


# Module-level shared HTTP client. httpx.AsyncClient holds a connection
# pool, DNS resolver state and async dispatch workers; creating a fresh
# one per sub-agent call turns each orchestration into 5+ client spin-ups
# that stack thread/socket pressure across concurrent orchestrations.
# The client is lazily initialised per running event loop — at test time
# the loop gets torn down between sessions, so a cached client bound to
# a dead loop would wedge; keying by id(loop) avoids that.
_HTTP_CLIENT_TIMEOUT = httpx.Timeout(240.0, connect=10.0)
_http_clients: Dict[int, httpx.AsyncClient] = {}
_http_clients_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Return a loop-scoped shared AsyncClient, building on first use."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    client = _http_clients.get(key)
    if client is not None and not client.is_closed:
        return client
    async with _http_clients_lock:
        client = _http_clients.get(key)
        if client is not None and not client.is_closed:
            return client
        client = httpx.AsyncClient(timeout=_HTTP_CLIENT_TIMEOUT)
        _http_clients[key] = client
        return client


# Cap concurrent orchestrations. Each complex query fans out to 5+
# sub-agent HTTP calls; without a cap, a handful of concurrent complex
# queries saturates both the shared httpx pool and the runtime's
# FastAPI worker pool, starving /health/live past the readiness probe.
# Hard-coded here rather than env-driven because agent modules must not
# read process environment at import time (project rule: env lookups
# happen at startup boundaries only). If dynamic tuning is ever needed,
# read the value in the runtime startup hook and thread it through the
# constructor.
_ORCH_CONCURRENCY = 4
_orch_semaphores: Dict[int, asyncio.Semaphore] = {}


def _get_orchestration_semaphore() -> asyncio.Semaphore:
    """Return a loop-scoped semaphore (see httpx client comment for why)."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _orch_semaphores.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_ORCH_CONCURRENCY)
        _orch_semaphores[key] = sem
    return sem


class OrchestratorInput(AgentInput):
    """Type-safe input for orchestration"""

    query: str = Field(..., description="Query to orchestrate")
    tenant_id: str = Field(
        ..., description="Tenant identifier (per-request, required)"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session identifier (per-request)"
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Previous conversation turns for multi-turn context"
    )


class OrchestratorOutput(AgentOutput):
    """Type-safe output from orchestration"""

    query: str = Field(..., description="Original query")
    workflow_id: str = Field("", description="Unique workflow identifier")
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Orchestration plan steps"
    )
    parallel_groups: List[List[int]] = Field(
        default_factory=list, description="Parallel execution groups"
    )
    plan_reasoning: str = Field("", description="Plan reasoning")
    agent_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results from each agent"
    )
    final_output: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated final output"
    )
    execution_summary: str = Field("", description="Summary of execution")


class OrchestratorDeps(AgentDeps):
    """Dependencies for orchestrator agent (tenant-agnostic at startup)."""

    pass


class FusionStrategy(Enum):
    """Strategies for combining results from multiple agents across modalities"""

    SCORE_BASED = "score"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    SIMPLE = "simple"


class AgentStep(BaseModel):
    """Single step in orchestration plan"""

    agent_name: str = Field(description="Name of the agent to invoke")
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input for this agent"
    )
    depends_on: List[int] = Field(
        default_factory=list, description="Indices of steps this depends on"
    )
    reasoning: str = Field(description="Why this step is needed")


class OrchestrationPlan(BaseModel):
    """Plan for query processing workflow"""

    query: str
    steps: List[AgentStep] = Field(description="Ordered sequence of agent invocations")
    parallel_groups: List[List[int]] = Field(
        default_factory=list,
        description="Groups of step indices that can run in parallel",
    )
    reasoning: str = Field(default="", description="Overall plan reasoning")
    unavailable_agents: List[str] = Field(
        default_factory=list,
        description="Agent names proposed by LLM but not in registry",
    )


class OrchestrationResult(BaseModel):
    """Result of orchestrated query processing"""

    query: str
    plan: OrchestrationPlan
    agent_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results from each agent"
    )
    final_output: Dict[str, Any] = Field(description="Aggregated final output")
    execution_summary: str = Field(description="Summary of execution")


class OrchestrationSignature(dspy.Signature):
    """Create execution plan for query processing"""

    query: str = dspy.InputField(desc="User query to process")
    available_agents: str = dspy.InputField(
        desc="Comma-separated list of available agents"
    )
    conversation_context: str = dspy.InputField(
        desc="Summary of previous conversation turns. Empty string if first turn."
    )
    gateway_context: str = dspy.InputField(
        desc="Classification context from gateway (intent, entities, modality). Empty string if not available."
    )

    agent_sequence: str = dspy.OutputField(
        desc="Comma-separated sequence of agents to invoke (e.g., 'entity_extraction_agent,profile_selection_agent,search_agent')"
    )
    parallel_steps: str = dspy.OutputField(
        desc="Indices of steps that can run in parallel (e.g., '0,1|2,3' means 0&1 parallel, then 2&3 parallel)"
    )
    reasoning: str = dspy.OutputField(desc="Explanation of orchestration plan")


class ResultAggregatorSignature(dspy.Signature):
    """Cross-modal fusion of multi-agent results"""

    original_query: str = dspy.InputField(desc="Original user query")
    task_results: str = dspy.InputField(
        desc="JSON string of individual task results with modalities"
    )
    fusion_strategy: str = dspy.InputField(
        desc="Fusion strategy: score, temporal, semantic, hierarchical"
    )
    agent_modalities: str = dspy.InputField(
        desc="Modalities of each agent result (video, image, audio, document, text)"
    )

    aggregated_result: str = dspy.OutputField(desc="Synthesized final result")
    confidence_score: float = dspy.OutputField(desc="Confidence in aggregated result")
    fusion_quality: str = dspy.OutputField(
        desc="Fusion quality metrics: coverage, consistency, coherence"
    )


class OrchestrationModule(dspy.Module):
    """DSPy module for orchestration planning"""

    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(OrchestrationSignature)

    def forward(
        self,
        query: str,
        available_agents: str,
        conversation_context: str = "",
        gateway_context: str = "",
    ) -> dspy.Prediction:
        """Create orchestration plan using LLM reasoning"""
        return self.planner(
            query=query,
            available_agents=available_agents,
            conversation_context=conversation_context,
            gateway_context=gateway_context,
        )


class OrchestratorAgent(
    MemoryAwareMixin, A2AAgent[OrchestratorInput, OrchestratorOutput, OrchestratorDeps]
):
    """
    Type-safe autonomous A2A agent for multi-agent orchestration.

    Implements two-phase orchestration:
    1. Planning Phase: Analyze query and create execution plan
    2. Action Phase: Execute plan by coordinating agents

    Features:
    - Dynamic agent discovery from AgentRegistry (no hardcoded enum)
    - Streaming progress events per step
    - Checkpoint/resume for durable execution
    - Cross-modal fusion for multi-agent result aggregation
    - Workflow intelligence (template matching + execution recording)
    - Cancellation of in-flight workflows
    """

    def __init__(
        self,
        deps: OrchestratorDeps,
        registry: "AgentRegistry",
        config_manager: "ConfigManager" = None,
        port: int = 8013,
        checkpoint_config: Optional[CheckpointConfig] = None,
        checkpoint_storage: Optional["WorkflowCheckpointStorage"] = None,
        event_queue: Optional["EventQueue"] = None,
        workflow_intelligence: Optional["WorkflowIntelligence"] = None,
    ):
        """
        Initialize OrchestratorAgent with real AgentRegistry.

        Args:
            deps: Typed dependencies (tenant-agnostic)
            registry: AgentRegistry for dynamic agent discovery
            config_manager: ConfigManager instance (REQUIRED)
            port: Port for A2A server
            checkpoint_config: Configuration for durable execution checkpoints
            checkpoint_storage: Storage backend for checkpoints
            event_queue: Optional EventQueue for real-time notifications
            workflow_intelligence: Optional WorkflowIntelligence for template matching

        Raises:
            TypeError: If deps is not OrchestratorDeps
            ValueError: If registry or config_manager is not provided
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for OrchestratorAgent. "
                "Dependency injection is mandatory - pass ConfigManager instance explicitly."
            )
        self.registry = registry
        self._config_manager = config_manager

        # Checkpoint support
        self.checkpoint_config = checkpoint_config or CheckpointConfig(enabled=False)
        self.checkpoint_storage = checkpoint_storage

        # Event queue for external consumers
        self.event_queue = event_queue

        # Workflow intelligence for template matching
        self.workflow_intelligence = workflow_intelligence

        # Active workflows for cancellation support
        self.active_workflows: Dict[str, OrchestrationPlan] = {}
        self._cancelled_workflows: set = set()

        orchestration_module = OrchestrationModule()

        config = A2AAgentConfig(
            agent_name="orchestrator_agent",
            agent_description="Type-safe orchestration with planning and action phases",
            capabilities=[
                "orchestration",
                "planning",
                "multi_agent_coordination",
                "parallel_execution",
                "result_aggregation",
            ],
            port=port,
            version="1.0.0",
        )

        super().__init__(deps=deps, config=config, dspy_module=orchestration_module)

        # Track which tenants have memory initialized
        self._memory_initialized_tenants: set = set()

        logger.info(
            f"OrchestratorAgent initialized with {len(self.registry.agents)} registered agents"
        )

    def _load_artifact(self) -> None:
        """Load workflow templates and historical data from artifact store.

        Called by the dispatcher after telemetry_manager and _artifact_tenant_id
        are injected — not from __init__ (telemetry_manager is not yet available).

        Delegates to WorkflowIntelligence.load_historical_data() which loads
        templates, agent profiles, and query patterns from the artifact store.
        """
        if not self.workflow_intelligence:
            return
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            async def _load():
                await self.workflow_intelligence.load_historical_data()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, _load())
                    future.result()
            else:
                asyncio.run(_load())

            logger.info(
                "OrchestratorAgent loaded %d workflow templates from artifact",
                len(self.workflow_intelligence.workflow_templates),
            )
        except Exception as e:
            logger.debug("No workflow artifact to load (using defaults): %s", e)

    def _ensure_memory_for_tenant(self, tenant_id: str) -> None:
        """Lazily initialize memory for a tenant (first request only)."""
        if tenant_id in self._memory_initialized_tenants:
            return

        try:
            from cogniverse_foundation.config.utils import get_config

            config = get_config(
                tenant_id=SYSTEM_TENANT_ID, config_manager=self._config_manager
            )

            llm_config = config.get_llm_config()
            resolved = llm_config.resolve("orchestrator_agent")

            # Read backend URL/port from BootstrapConfig (reads env vars set
            # at the startup boundary — authoritative source of truth).
            from cogniverse_foundation.config.bootstrap import BootstrapConfig

            bootstrap = BootstrapConfig.from_environment()
            backend_url = bootstrap.backend_url
            backend_port = bootstrap.backend_port

            # Extract provider from model string (e.g., "ollama/smollm3:3b" -> "ollama")
            provider = (
                resolved.model.split("/")[0] if "/" in resolved.model else "local"
            )
            llm_model = resolved.model
            llm_base_url = resolved.api_base or "http://localhost:11434"
            embedding_model = config.get("embedding_model", "nomic-embed-text")

            from pathlib import Path

            from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

            schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

            from cogniverse_vespa.config_utils import calculate_config_port

            self.initialize_memory(
                agent_name="orchestrator_agent",
                tenant_id=tenant_id,
                backend_host=backend_url,
                backend_port=backend_port,
                backend_config_port=calculate_config_port(backend_port),
                llm_model=llm_model,
                embedding_model=embedding_model,
                llm_base_url=llm_base_url,
                config_manager=self._config_manager,
                schema_loader=schema_loader,
                provider=provider,
            )
            self._memory_initialized_tenants.add(tenant_id)
        except Exception as e:
            logger.warning(
                f"Memory initialization failed for tenant '{tenant_id}': {e}. "
                "Continuing without memory support."
            )

    async def _emit_event(self, event) -> None:
        """Emit event to EventQueue if configured."""
        if self.event_queue is not None:
            await self.event_queue.enqueue(event)

    async def _process_impl(
        self, input: Union[OrchestratorInput, Dict[str, Any]]
    ) -> OrchestratorOutput:
        """
        Process orchestration request with typed input/output.

        Args:
            input: Typed input with query, tenant_id, session_id (or dict)

        Returns:
            OrchestratorOutput with plan, agent results, and final output
        """
        # Coerce dict to typed input (A2A sends JSON dicts)
        if isinstance(input, dict):
            input = OrchestratorInput(**input)

        query = input.query
        tenant_id = input.tenant_id
        session_id = input.session_id
        self._current_tenant_id = tenant_id
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"

        if not query:
            return OrchestratorOutput(
                query="",
                workflow_id=workflow_id,
                plan_steps=[],
                parallel_groups=[],
                plan_reasoning="Empty query, no orchestration needed",
                agent_results={},
                final_output={"status": "error", "message": "Empty query"},
                execution_summary="No execution performed",
            )

        # Bound concurrent orchestrations — each fans out to 5+ sub-agent
        # calls, and unrestricted concurrency saturates the shared httpx
        # pool, the FastAPI worker pool, and the event loop. Waiters queue
        # here instead of stacking load on downstream services.
        sem = _get_orchestration_semaphore()
        async with sem:
            return await self._process_impl_locked(
                input, workflow_id, query, tenant_id, session_id
            )

    async def _process_impl_locked(
        self,
        input: "OrchestratorInput",
        workflow_id: str,
        query: str,
        tenant_id: str,
        session_id: Optional[str],
    ) -> "OrchestratorOutput":
        """Orchestration body — runs under ``_orch_semaphores`` cap."""
        start_time = time.monotonic()

        # Lazy memory initialization for this tenant
        self._ensure_memory_for_tenant(tenant_id)

        # Get relevant context from memory (cross-session)
        self.emit_progress("memory_context", "Retrieving memory context...")
        memory_context = self.get_relevant_context(query)
        if memory_context:
            logger.info(f"Retrieved memory context for query: {query[:50]}...")

        # Format conversation history for DSPy planner
        conversation_context = self._format_conversation_context(
            input.conversation_history
        )

        # Phase 1: Planning (with workflow intelligence template matching)
        self.emit_progress("planning", "Creating execution plan...")
        gateway_context = ""
        if self.workflow_intelligence:
            template = self.workflow_intelligence._find_matching_template(query)
            if template:
                gateway_context = (
                    f"Matched template: {template.name}. "
                    f"Suggested sequence: {json.dumps(template.task_sequence)}"
                )
                logger.info(f"Workflow intelligence matched template: {template.name}")

        plan = await self._create_plan(query, conversation_context, gateway_context)

        # Track active workflow for cancellation
        self.active_workflows[workflow_id] = plan

        try:
            # Phase 2: Action -- execute via A2A HTTP, passing tenant_id/session_id
            self.emit_progress("execution", "Executing agent plan...")
            agent_results = await self._execute_plan(
                plan,
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )

            # Record error entries for agents the LLM proposed but aren't registered
            for agent_name in plan.unavailable_agents:
                agent_results[agent_name] = {
                    "status": "error",
                    "message": f"Agent '{agent_name}' is not available in the registry",
                }

            self.emit_progress("aggregating", "Merging results from all agents")
            final_output = self._aggregate_results(query, agent_results)
            execution_summary = self._generate_summary(plan, agent_results)
            self.remember_success(query, execution_summary)

            execution_time = time.monotonic() - start_time

            # Emit orchestration telemetry span
            self._emit_orchestration_span(
                workflow_id=workflow_id,
                query=query,
                agent_sequence=[step.agent_name for step in plan.steps],
                execution_time=execution_time,
                success=True,
                tasks_completed=len(agent_results),
            )

            # Record execution for workflow intelligence
            if self.workflow_intelligence:
                try:
                    from datetime import datetime, timedelta

                    from cogniverse_agents.workflow_types import (
                        WorkflowPlan as WFPlan,
                    )
                    from cogniverse_agents.workflow_types import (
                        WorkflowStatus,
                        WorkflowTask,
                    )

                    wf_plan = WFPlan(
                        workflow_id=workflow_id,
                        original_query=query,
                        tasks=[
                            WorkflowTask(
                                task_id=f"task_{i}",
                                agent_name=step.agent_name,
                                query=query,
                            )
                            for i, step in enumerate(plan.steps)
                        ],
                        status=WorkflowStatus.COMPLETED,
                        start_time=datetime.now() - timedelta(seconds=execution_time),
                        end_time=datetime.now(),
                        metadata={"agent_results": {}},
                    )
                    await self.workflow_intelligence.record_workflow_execution(wf_plan)
                except Exception as e:
                    logger.warning(f"Failed to record workflow execution: {e}")

            self.emit_progress("complete", "Orchestration finished")

            return OrchestratorOutput(
                query=query,
                workflow_id=workflow_id,
                plan_steps=[
                    {
                        "agent_name": step.agent_name,
                        "reasoning": step.reasoning,
                        "depends_on": step.depends_on,
                    }
                    for step in plan.steps
                ],
                parallel_groups=plan.parallel_groups,
                plan_reasoning=plan.reasoning,
                agent_results=agent_results,
                final_output=final_output,
                execution_summary=execution_summary,
            )
        finally:
            self.active_workflows.pop(workflow_id, None)

    async def _create_plan(
        self,
        query: str,
        conversation_context: str = "",
        gateway_context: str = "",
    ) -> OrchestrationPlan:
        """
        Planning Phase: Create execution plan using LLM reasoning.

        Uses dynamic agent discovery from AgentRegistry instead of a hardcoded enum.

        Args:
            query: User query to analyze
            conversation_context: Formatted previous conversation turns
            gateway_context: Classification context from gateway agent

        Returns:
            OrchestrationPlan with agent sequence and parallelization
        """
        # Dynamic agent discovery from registry
        registered_agents = self.registry.list_agents()
        available_agents = ", ".join(registered_agents)

        result = await self.call_dspy(
            self.dspy_module,
            output_field="agent_sequence",
            query=query,
            available_agents=available_agents,
            conversation_context=conversation_context,
            gateway_context=gateway_context,
        )

        raw_sequence = result.agent_sequence or ""
        agent_sequence = [
            a.strip() for a in raw_sequence.split(",") if a.strip()
        ]
        if not agent_sequence:
            # DSPy planner returned empty/None — fall back to search
            logger.warning("DSPy planner returned empty agent_sequence, falling back to search_agent")
            agent_sequence = ["search_agent"]

        # Parse parallel groups (LLM may return "None" or non-numeric strings)
        parallel_groups = []
        if result.parallel_steps and result.parallel_steps.strip().lower() != "none":
            for group in result.parallel_steps.split("|"):
                indices = []
                for i in group.split(","):
                    token = i.strip()
                    if token.isdigit():
                        indices.append(int(token))
                if indices:
                    parallel_groups.append(indices)

        steps = []
        unavailable_agents = []
        # Build lookup for agent name normalization (LLM may add/omit _agent suffix)
        _agent_lookup = {name: name for name in registered_agents}
        for name in registered_agents:
            _agent_lookup[f"{name}_agent"] = name
            if name.endswith("_agent"):
                _agent_lookup[name[: -len("_agent")]] = name

        for i, agent_name in enumerate(agent_sequence):
            # Normalize: match with or without _agent suffix
            agent_name = _agent_lookup.get(agent_name, agent_name)
            # Validate against registry (skip unknown agents)
            if agent_name not in registered_agents:
                logger.warning(
                    f"LLM proposed unknown agent '{agent_name}', "
                    f"not in registry ({registered_agents}), skipping"
                )
                unavailable_agents.append(agent_name)
                continue
            step = AgentStep(
                agent_name=agent_name,
                input_data={"query": query},
                depends_on=self._calculate_dependencies(i, parallel_groups),
                reasoning=f"Step {i + 1}: {agent_name} processing",
            )
            steps.append(step)

        return OrchestrationPlan(
            query=query,
            steps=steps,
            parallel_groups=parallel_groups,
            reasoning=result.reasoning or "",
            unavailable_agents=unavailable_agents,
        )

    def _calculate_dependencies(
        self, step_index: int, parallel_groups: List[List[int]]
    ) -> List[int]:
        """Calculate which steps this step depends on"""
        depends_on = []

        # Find which parallel group this step belongs to
        current_group = None
        for group in parallel_groups:
            if step_index in group:
                current_group = group
                break

        if current_group:
            # Parallel step - check what comes before this group
            min_index_in_group = min(current_group)
            if min_index_in_group > 0:
                prev_step = min_index_in_group - 1

                # Check if previous step is in a parallel group
                prev_in_group = None
                for group in parallel_groups:
                    if prev_step in group:
                        prev_in_group = group
                        break

                if prev_in_group:
                    # Previous parallel group - depend on entire group
                    depends_on.extend(prev_in_group)
                else:
                    # Sequential step before this group - depend on it
                    depends_on.append(prev_step)
        else:
            # Sequential step - check if previous step is in a parallel group
            if step_index > 0:
                prev_step = step_index - 1

                # Check if previous step is in a parallel group
                prev_in_group = None
                for group in parallel_groups:
                    if prev_step in group:
                        prev_in_group = group
                        break

                if prev_in_group:
                    # Previous step is in a parallel group, depend on entire group
                    depends_on.extend(prev_in_group)
                else:
                    # Previous step is sequential, just depend on it
                    depends_on.append(prev_step)

        return depends_on

    async def _execute_plan(
        self,
        plan: OrchestrationPlan,
        tenant_id: str,
        workflow_id: str = "",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Action Phase: Execute orchestration plan via A2A HTTP calls.

        Discovers agents from the real AgentRegistry, calls them via HTTP,
        and passes tenant_id/session_id through to each agent.
        Emits streaming progress events per step, saves checkpoints after
        each step, and checks for cancellation between steps.

        Args:
            plan: OrchestrationPlan to execute
            workflow_id: Workflow identifier for tracking
            tenant_id: Tenant identifier (per-request)
            session_id: Session identifier (per-request)

        Returns:
            Dictionary of agent results
        """
        import asyncio

        agent_results = {}
        executed = [False] * len(plan.steps)
        step_count = 0

        # Execute steps respecting dependencies and parallelism
        while not all(executed):
            # Check for cancellation (explicit cancel via cancel_workflow)
            if workflow_id and workflow_id in self._cancelled_workflows:
                logger.info(f"Workflow {workflow_id} cancelled, stopping execution")
                self._cancelled_workflows.discard(workflow_id)
                break

            # Find steps ready to execute (all dependencies met)
            ready_steps = []
            for i, step in enumerate(plan.steps):
                if executed[i]:
                    continue
                deps_met = all(executed[dep_idx] for dep_idx in step.depends_on)
                if deps_met:
                    ready_steps.append((i, step))

            if not ready_steps:
                logger.error("No steps ready to execute but execution incomplete")
                break

            logger.info(
                f"Executing {len(ready_steps)} steps in parallel: "
                f"{[s[1].agent_name for s in ready_steps]}"
            )

            async def execute_step(step_index: int, step: AgentStep):
                """Execute a single step via A2A HTTP."""
                agent_name = step.agent_name

                # Emit progress before execution
                self.emit_progress(
                    "executing",
                    f"Step {step_index}: {agent_name}",
                    {"step": step_index, "agent": agent_name},
                )

                # Discover agent from registry by name
                agent_endpoint = self.registry.get_agent(agent_name)
                if not agent_endpoint:
                    candidates = self.registry.find_agents_by_capability(agent_name)
                    if candidates:
                        agent_endpoint = candidates[0]

                if not agent_endpoint:
                    logger.warning(f"Agent '{agent_name}' not found in registry")
                    return agent_name, {
                        "status": "error",
                        "message": f"Agent '{agent_name}' not available in registry",
                    }

                # Prepare input (merge query with previous results if needed)
                agent_input = step.input_data.copy()
                context: Dict[str, Any] = {"tenant_id": tenant_id}
                if session_id:
                    context["session_id"] = session_id

                for dep_idx in step.depends_on:
                    if dep_idx < len(plan.steps):
                        dep_agent = plan.steps[dep_idx].agent_name
                        if dep_agent in agent_results:
                            agent_input[f"{dep_agent}_result"] = agent_results[
                                dep_agent
                            ]

                # Call agent via HTTP — payload must satisfy AgentTask schema:
                # agent_name + query required, tenant_id goes inside context.
                try:
                    query = agent_input.pop("query", "")
                    payload = {
                        "agent_name": agent_name,
                        "query": query,
                        "context": context,
                        **agent_input,
                    }
                    http_client = await _get_http_client()
                    response = await http_client.post(
                        f"{agent_endpoint.url}{agent_endpoint.process_endpoint}",
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Emit completion event
                    self.emit_progress(
                        "step_complete",
                        f"Step {step_index} complete",
                        {
                            "step": step_index,
                            "result_preview": str(result)[:200],
                        },
                    )
                    return agent_name, result
                except Exception as e:
                    # Include the exception type — httpx.ReadTimeout and
                    # friends have empty str() and render as bare "failed: "
                    # without it, flattening every failure in the logs.
                    err_detail = f"{type(e).__name__}: {e}" if str(e) else repr(e)
                    logger.error(
                        f"Agent {agent_name} at {agent_endpoint.url} failed: "
                        f"{err_detail}"
                    )
                    return agent_name, {
                        "status": "error",
                        "message": err_detail,
                    }

            # Execute all ready steps concurrently
            results = await asyncio.gather(
                *[execute_step(idx, step) for idx, step in ready_steps]
            )

            # Store results and mark as executed
            for (step_idx, _), (agent_name, result) in zip(ready_steps, results):
                agent_results[agent_name] = result
                executed[step_idx] = True
                step_count += 1

            # Save checkpoint after each batch of steps (if configured)
            if self._should_checkpoint():
                await self._save_checkpoint(
                    plan,
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                    current_step=step_count,
                    status=CheckpointStatus.ACTIVE,
                    agent_results=agent_results,
                )

        return agent_results

    def _format_conversation_context(
        self, conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Format conversation history as text for the DSPy planner.

        Truncates to last 5 turns, 200 chars each, to keep prompt size manageable.
        """
        if not conversation_history:
            return ""

        lines = []
        for turn in conversation_history[-5:]:
            role = turn.get("role", "user")
            content = str(turn.get("content", ""))[:200]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _aggregate_results(
        self, query: str, agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cross-modal fusion of results from all agents.

        Detects modality per result, selects fusion strategy, and dispatches
        to the appropriate fusion method. Falls back to simple aggregation
        for single-modality results.
        """
        if not agent_results:
            return {"query": query, "status": "success", "results": {}}

        # Detect modalities and build task_results structure
        task_results = {}
        agent_modalities = {}

        for agent_name, result in agent_results.items():
            modality = self._detect_agent_modality(agent_name)
            agent_modalities[agent_name] = modality

            if isinstance(result, BaseModel):
                result_data = result.model_dump()
            else:
                result_data = result

            confidence = (
                result_data.get("confidence", 0.5)
                if isinstance(result_data, dict)
                else 0.5
            )

            task_results[agent_name] = {
                "agent": agent_name,
                "modality": modality,
                "result": result_data,
                "confidence": confidence,
            }

        # Select fusion strategy
        fusion_strategy = self._select_fusion_strategy(query, agent_modalities)

        # Dispatch to fusion method
        if fusion_strategy == FusionStrategy.SCORE_BASED:
            fused = self._fuse_by_score(task_results)
        elif fusion_strategy == FusionStrategy.HIERARCHICAL:
            fused = self._fuse_hierarchically(task_results, agent_modalities)
        else:
            fused = self._fuse_simple(task_results)

        # Calculate fusion quality
        modalities = set(agent_modalities.values())
        fusion_quality = {
            "strategy": fusion_strategy.value,
            "modality_count": len(modalities),
            "modalities": list(modalities),
            "confidence": fused["confidence"],
        }

        return {
            "query": query,
            "status": "success",
            "results": {
                name: tr["result"] for name, tr in task_results.items()
            },
            "fusion_strategy": fusion_strategy.value,
            "fusion_quality": fusion_quality,
            "aggregated_content": fused["content"],
        }

    def _detect_agent_modality(self, agent_name: str) -> str:
        """Detect modality from agent name."""
        name_lower = agent_name.lower()
        if "video" in name_lower:
            return "video"
        elif "image" in name_lower:
            return "image"
        elif "audio" in name_lower:
            return "audio"
        elif "document" in name_lower:
            return "document"
        return "text"

    def _select_fusion_strategy(
        self, query: str, agent_modalities: Dict[str, str]
    ) -> FusionStrategy:
        """Select fusion strategy based on query and modalities."""
        modalities = set(agent_modalities.values())
        query_lower = query.lower()

        # Temporal fusion for time-related queries with multiple modalities
        temporal_keywords = ["timeline", "sequence", "chronological", "when", "duration"]
        if (
            any(kw in query_lower for kw in temporal_keywords)
            and len(modalities) > 1
        ):
            return FusionStrategy.TEMPORAL

        # Hierarchical fusion for comparison queries
        hierarchical_keywords = ["compare", "contrast", "difference", "versus", "vs"]
        if any(kw in query_lower for kw in hierarchical_keywords):
            return FusionStrategy.HIERARCHICAL

        # Score-based for multi-modality queries
        if len(modalities) > 1:
            return FusionStrategy.SCORE_BASED

        return FusionStrategy.SIMPLE

    def _fuse_by_score(self, task_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Score-based fusion: weight results by confidence scores."""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        total_confidence = sum(tr["confidence"] for tr in task_results.values())

        if total_confidence == 0:
            weights = {tid: 1.0 / len(task_results) for tid in task_results}
        else:
            weights = {
                tid: tr["confidence"] / total_confidence
                for tid, tr in task_results.items()
            }

        content_parts = []
        for tid, tr in sorted(
            task_results.items(), key=lambda x: weights[x[0]], reverse=True
        ):
            modality = tr["modality"]
            content_parts.append(
                f"[{modality.upper()} - confidence: {weights[tid]:.2f}]\n{str(tr['result'])}"
            )

        aggregated_confidence = sum(
            tr["confidence"] * weights[tid] for tid, tr in task_results.items()
        )

        return {
            "content": "\n\n".join(content_parts),
            "confidence": aggregated_confidence,
        }

    def _fuse_hierarchically(
        self, task_results: Dict[str, Dict], agent_modalities: Dict[str, str]
    ) -> Dict[str, Any]:
        """Hierarchical fusion: structured combination by modality groups."""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        modality_groups: Dict[str, List] = {}
        for tid, tr in task_results.items():
            modality = tr["modality"]
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append((tid, tr))

        content_parts = []
        total_confidence = 0.0
        modality_count = 0

        for modality in ["video", "image", "audio", "document", "text"]:
            if modality not in modality_groups:
                continue
            modality_count += 1
            tasks = modality_groups[modality]
            content_parts.append(
                f"## {modality.upper()} RESULTS ({len(tasks)} sources)"
            )
            modality_conf = 0.0
            for tid, tr in tasks:
                content_parts.append(f"- {str(tr['result'])}")
                modality_conf += tr["confidence"]
            avg = modality_conf / len(tasks)
            total_confidence += avg
            content_parts.append("")

        avg_confidence = total_confidence / modality_count if modality_count > 0 else 0.0
        return {"content": "\n".join(content_parts), "confidence": avg_confidence}

    def _fuse_simple(self, task_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Simple fusion: basic concatenation."""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        content_parts = []
        total_confidence = 0.0
        for tr in task_results.values():
            content_parts.append(str(tr["result"]))
            total_confidence += tr["confidence"]

        return {
            "content": "\n\n".join(content_parts),
            "confidence": total_confidence / len(task_results),
        }

    # Checkpoint methods

    def _should_checkpoint(self) -> bool:
        """Check if checkpointing is enabled and configured."""
        return (
            self.checkpoint_config.enabled
            and self.checkpoint_storage is not None
        )

    async def _save_checkpoint(
        self,
        plan: OrchestrationPlan,
        tenant_id: str,
        workflow_id: str,
        current_step: int,
        status: CheckpointStatus,
        agent_results: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save a checkpoint of the current workflow state."""
        if self.checkpoint_storage is None:
            return None

        try:
            task_states = {}
            for i, step in enumerate(plan.steps):
                completed = i < current_step
                task_states[f"step_{i}"] = TaskCheckpoint(
                    task_id=f"step_{i}",
                    agent_name=step.agent_name,
                    query=step.input_data.get("query", ""),
                    dependencies=[str(d) for d in step.depends_on],
                    status="completed" if completed else "pending",
                    result=(
                        agent_results.get(step.agent_name)
                        if agent_results and completed
                        else None
                    ),
                )

            from datetime import datetime

            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"ckpt_{uuid.uuid4().hex[:12]}",
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                workflow_status="running" if status == CheckpointStatus.ACTIVE else status.value,
                current_phase=current_step,
                original_query=plan.query,
                execution_order=[[f"step_{i}"] for i in range(len(plan.steps))],
                metadata={"reasoning": plan.reasoning},
                task_states=task_states,
                checkpoint_time=datetime.now(),
                checkpoint_status=status,
            )

            checkpoint_id = await self.checkpoint_storage.save_checkpoint(checkpoint)
            logger.info(
                f"Saved checkpoint {checkpoint_id} for workflow {workflow_id} "
                f"(step {current_step}, status: {status.value})"
            )
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    async def resume_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Resume a workflow from its latest checkpoint.

        Args:
            workflow_id: ID of the workflow to resume

        Returns:
            Checkpoint data if found, None if no checkpoint exists
        """
        if self.checkpoint_storage is None:
            return None

        checkpoint = await self.checkpoint_storage.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            logger.warning(f"No checkpoint found for workflow {workflow_id}")
            return None

        logger.info(
            f"Resuming workflow {workflow_id} from checkpoint {checkpoint.checkpoint_id} "
            f"(step {checkpoint.current_phase})"
        )
        return checkpoint.to_dict()

    # Cancellation

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            True if workflow was cancelled, False if not found
        """
        if workflow_id not in self.active_workflows:
            return False

        self._cancelled_workflows.add(workflow_id)
        self.active_workflows.pop(workflow_id)
        logger.info(f"Workflow {workflow_id} cancelled")
        return True

    # Telemetry

    def _emit_orchestration_span(
        self,
        workflow_id: str,
        query: str,
        agent_sequence: List[str],
        execution_time: float,
        success: bool,
        tasks_completed: int,
    ) -> None:
        """Emit cogniverse.orchestration telemetry span."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        tenant_id = getattr(self, "_current_tenant_id", None)
        if not tenant_id:
            raise RuntimeError(
                f"{type(self).__name__}._emit_orchestration_span called before "
                f"_process_impl set self._current_tenant_id"
            )
        try:
            with self.telemetry_manager.span(
                name="cogniverse.orchestration",
                tenant_id=tenant_id,
                attributes={
                    "orchestration.workflow_id": str(workflow_id),
                    "orchestration.query": query[:200],
                    "orchestration.agent_sequence": ",".join(agent_sequence) if agent_sequence else "",
                    "orchestration.execution_time": float(execution_time),
                    "orchestration.success": bool(success),
                    "orchestration.tasks_completed": int(tasks_completed),
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit orchestration span: %s", e)

    def _generate_summary(
        self, plan: OrchestrationPlan, agent_results: Dict[str, Any]
    ) -> str:
        """Generate execution summary"""
        executed_steps = len(agent_results)
        successful_steps = sum(
            1
            for result in agent_results.values()
            if not (isinstance(result, dict) and result.get("status") == "error")
        )
        total_steps = len(plan.steps)

        return (
            f"Executed {executed_steps}/{total_steps} steps "
            f"({successful_steps} successful). "
            f"Plan: {plan.reasoning}"
        )

    def _dspy_to_a2a_output(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Convert OrchestrationResult to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "query": result.query,
            "plan": {
                "steps": [
                    {
                        "agent_name": step.agent_name,
                        "reasoning": step.reasoning,
                        "depends_on": step.depends_on,
                    }
                    for step in result.plan.steps
                ],
                "parallel_groups": result.plan.parallel_groups,
                "reasoning": result.plan.reasoning,
            },
            "agent_results": result.agent_results,
            "final_output": result.final_output,
            "execution_summary": result.execution_summary,
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "orchestrate",
                "description": "Orchestrate multi-agent query processing with planning and execution",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "plan": "object",
                    "agent_results": "object",
                    "final_output": "object",
                    "execution_summary": "string",
                },
                "examples": [
                    {
                        "input": {"query": "Show me machine learning videos"},
                        "output": {
                            "plan": {
                                "steps": [
                                    {"agent_type": "query_enhancement"},
                                    {"agent_type": "search"},
                                ]
                            },
                            "execution_summary": "Executed 2/2 steps (2 successful)",
                        },
                    }
                ],
            }
        ]


# FastAPI app for standalone deployment
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Global agent instance
orchestrator_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global orchestrator_agent

    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()

    # AgentRegistry reads agents from config.json > agents section.
    # Startup is cluster-scope (no request tenant); per-request flow
    # still routes through the dispatcher with its own tenant.
    registry = AgentRegistry(
        tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager
    )

    deps = OrchestratorDeps()
    orchestrator_agent = OrchestratorAgent(
        deps=deps, registry=registry, config_manager=config_manager
    )
    logger.info("OrchestratorAgent started")
    yield


app = FastAPI(
    title="OrchestratorAgent",
    description="Autonomous orchestration agent with planning and action phases",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not orchestrator_agent:
        return {"status": "initializing"}
    return orchestrator_agent.app.routes[2].endpoint()


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not orchestrator_agent:
        return {"error": "Agent not initialized"}
    return orchestrator_agent.app.routes[0].endpoint()


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not orchestrator_agent:
        return {"error": "Agent not initialized"}
    return await orchestrator_agent.app.routes[1].endpoint(task)


if __name__ == "__main__":
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    registry = AgentRegistry(
        tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager
    )
    deps = OrchestratorDeps()
    agent = OrchestratorAgent(
        deps=deps, registry=registry, config_manager=config_manager, port=8013
    )
    logger.info("Starting OrchestratorAgent on port 8013...")
    agent.start()
