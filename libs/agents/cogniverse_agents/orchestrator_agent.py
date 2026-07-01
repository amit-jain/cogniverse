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
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dspy
import httpx
from pydantic import BaseModel, Field

from cogniverse_agents._confidence import parse_confidence
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig,
    CheckpointStatus,
    TaskCheckpoint,
    WorkflowCheckpoint,
)
from cogniverse_agents.orchestrator.sufficient_context_signature import (
    SufficientContextSignature,
)
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.common.utils.async_bridge import run_coro_blocking

# Per-session inbound messaging — looked up lazily so the orchestrator
# can run without the runtime layer wired (e.g. agent-only unit tests
# don't import cogniverse_runtime). When the import fails we operate
# as if no inbound channel is available — drain becomes a no-op.
try:
    from cogniverse_runtime.messaging import (
        InboundQueue as _InboundQueue,
    )
    from cogniverse_runtime.messaging import (
        get_inbound_queue_registry as _get_inbound_queue_registry,
    )
except ImportError:  # pragma: no cover — runtime not wired in agent-only contexts
    _InboundQueue = None  # type: ignore[assignment]
    _get_inbound_queue_registry = None  # type: ignore[assignment]


async def _resolve_inbound_registry_for_orchestrator(redis_url: str = ""):
    """Pick in-pod vs Redis backend from a caller-supplied URL.

    Mirrors the routers/agents.py helper so the HTTP route and the
    orchestrator hit the same registry instance per pod. When
    ``redis_url`` is non-empty, both go through the cross-pod Redis
    backend and messages routed across pods land where the
    orchestrator drains them.

    The caller MUST source ``redis_url`` from
    ``SystemConfig.redis_url`` (populated at runtime startup from
    the ``REDIS_URL`` env var). This function does not read env.
    """
    if _get_inbound_queue_registry is None:
        return None
    if redis_url:
        try:
            from cogniverse_runtime.messaging_redis import (
                get_redis_inbound_queue_registry as _get_redis_reg,
            )

            return await _get_redis_reg(redis_url)
        except ImportError:  # pragma: no cover
            pass
    return _get_inbound_queue_registry()


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


# Preprocessing agent outputs → execution agent input fields. Maps the
# source field name on the producer's result to the target field name on
# the consumer's typed input schema.
_ENRICHMENT_FIELD_MAP: Dict[str, Dict[str, str]] = {
    "entity_extraction_agent": {
        "entities": "entities",
        "relationships": "relationships",
    },
    "query_enhancement_agent": {
        "enhanced_query": "enhanced_query",
        "query_variants": "query_variants",
    },
}


def _normalize_query_variants(
    variants: List[Any],
) -> List[Dict[str, str]]:
    """Normalise query variants to the dict form consumers expect.

    QueryEnhancementAgent emits variants as raw strings; SearchInput
    declares them as ``List[Dict[str, str]]`` with ``name``/``query`` keys.
    """
    normalized: List[Dict[str, str]] = []
    for i, variant in enumerate(variants):
        if isinstance(variant, dict):
            normalized.append(
                {
                    "name": str(variant.get("name", f"variant_{i}")),
                    "query": str(variant.get("query", "")),
                }
            )
        elif isinstance(variant, str):
            normalized.append(
                {"name": f"variant_{i}" if i > 0 else "original", "query": variant}
            )
    return normalized


from cogniverse_agents._rlm_promotion import (
    maybe_promote_to_rlm as _maybe_promote_to_rlm,
)


def _merge_enrichment(
    agent_input: Dict[str, Any],
    dep_agent: str,
    dep_result: Dict[str, Any],
) -> None:
    """Copy named enrichment fields from a preprocessing result onto the
    next step's agent input. No-op for dependencies not in the field map.

    Agent name lookup is suffix-tolerant: the registry may expose agents
    with or without the ``_agent`` suffix, and both forms map to the same
    field rules.

    ``profile_selection`` wraps ``selected_profile`` as ``profiles=[selected]``
    because ``SearchInput.profiles`` is a list (single-element = single-profile
    override; multi-element = ensemble).
    """
    canonical = dep_agent if dep_agent.endswith("_agent") else f"{dep_agent}_agent"
    field_map = _ENRICHMENT_FIELD_MAP.get(canonical)
    if field_map:
        for src_field, dst_field in field_map.items():
            value = dep_result.get(src_field)
            if not value:
                continue
            if dst_field == "query_variants":
                value = _normalize_query_variants(value)
            agent_input[dst_field] = value
        return

    if canonical == "profile_selection_agent":
        selected = dep_result.get("selected_profile")
        if selected:
            agent_input["profiles"] = [selected]


class OrchestratorInput(AgentInput):
    """Type-safe input for orchestration"""

    query: str = Field(..., description="Query to orchestrate")
    tenant_id: str = Field(..., description="Tenant identifier (per-request, required)")
    session_id: Optional[str] = Field(
        default=None, description="Session identifier (per-request)"
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Previous conversation turns for multi-turn context"
    )
    modality: Optional[str] = Field(
        default=None,
        description=(
            "Content modality hint from GatewayAgent GLiNER classification "
            "(video/image/audio/document/text). Fed to the DSPy planner "
            "as a prior."
        ),
    )
    generation_type: Optional[str] = Field(
        default=None,
        description=(
            "Output type hint from GatewayAgent classification "
            "(raw_results/summary/detailed_report)."
        ),
    )
    synthesis_depth: Optional[str] = Field(
        default=None,
        description=(
            "Opt-in deep-synthesis switch. When set to ``deep`` the "
            "orchestrator dispatches via :class:`DeepSynthesisWorkflow` "
            "(recursive multi-agent loop with hard rate + call caps) "
            "instead of the default plan-then-act path. Any other value "
            "(or omitted) keeps the default behaviour."
        ),
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
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Side-channel structured metadata. Currently carries an "
            "``iterative_loop`` mirror of ``final_output['iterative_loop']`` "
            "so downstream consumers (eval harnesses, the BRIGHT probe "
            "harness, evaluation dashboards) can read the loop trajectory "
            "without depending on the fusion-shaped ``final_output`` "
            "envelope. Set by ``_process_impl_locked`` after the iterative "
            "retrieval loop returns."
        ),
    )


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


# Caps for the iterative retrieval loop. The loop is the retrieval path,
# not an optional mode, so every request runs at least one iteration and
# at most these caps allow. The numbers are tuned for reasoning-intensive
# probe sets and per-segment KG provenance flows; anything heavier should
# adjust caps via the orchestrator constructor, not by carving out a
# separate code path.
# Iterative-loop tuning knobs are now plumbed through SystemConfig
# (see ``iter_retrieval_max_iter`` / ``iter_retrieval_token_budget``
# / ``iter_retrieval_wall_clock_ms``). The runtime startup reads the
# corresponding env vars at the main entry point and stores them on
# SystemConfig; the orchestrator reads them via
# ``self._config_manager.get_system_config()`` (no env access here).
# Threshold (in characters of JSON-serialized evidence) above which the
# sufficient-context gate is promoted from a single-prompt CoT to the
# iterative RLM substrate. Mirrors the RLM-promotion rule for other large
# prompts.
_ITER_GATE_RLM_PROMOTION_CHARS = 6000
# Approximate character budget consumed by the dspy chat adapter's
# wrapping of ``SufficientContextSignature`` (input field descriptions,
# output field schemas, CoT scaffolding, role markers). Measured by
# rendering the signature through ``ChatAdapter.format`` with an empty
# evidence list. Used by ``_evidence_token_estimate`` so the cumulative
# token budget reflects the real LM prompt size, not just the evidence
# JSON. See ``test_d9_token_budget_breach_exits_at_iter1``.
_GATE_PROMPT_SCAFFOLDING_CHARS = 2400

# Test-hook overrides — when not None, take precedence over
# SystemConfig.iter_retrieval_* for the iterative-retrieval loop. Used
# by tests to override caps without constructing a per-test
# SystemConfig. Production deploys leave these at None and rely on the
# SystemConfig values supplied at runtime startup.
_ITER_RETRIEVAL_MAX_ITER: Optional[int] = None
_ITER_RETRIEVAL_TOKEN_BUDGET: Optional[int] = None
_ITER_RETRIEVAL_WALL_CLOCK_MS: Optional[int] = None


@dataclass
class AccumulatedEvidence:
    """Output of the orchestrator's iterative retrieval loop.

    ``evidence`` is the running list of snippets collected across all
    iterations. Each snippet is a dict with at least
    ``{source_doc_id, segment_id, ts_start, ts_end, text}`` plus optional
    ranking / modality fields. ``final_gate_output`` mirrors the last
    sufficient-context gate decision (with keys ``sufficient``,
    ``missing_aspects``, ``confidence``, ``rationale``) so callers can
    surface why the loop stopped without re-running it.

    ``trace_id`` is the OTEL trace id of the orchestration span; tests
    use it to fetch the ``retrieval_iteration`` child spans from Phoenix.

    The partial flags are mutually exclusive markers for the cap-driven
    exit reasons — they let downstream UIs warn the user that the answer
    is best-effort, not authoritative.
    """

    evidence: List[Dict[str, Any]] = field(default_factory=list)
    iterations_executed: int = 0
    exit_reason: str = ""
    final_gate_output: Dict[str, Any] = field(default_factory=dict)
    partial_due_to_budget: bool = False
    partial_due_to_timeout: bool = False
    trace_id: str = ""
    # Inbound-channel constraints the loop drained from
    # ``InboundQueue`` while running. Each entry is the literal
    # ``content`` of a message tagged ``constraint`` or ``interrupt``,
    # in submission order. Empty for sessions without an inbound queue
    # or with no constraint messages enqueued.
    inbound_constraints_applied: List[str] = field(default_factory=list)
    # Per-iteration record: list of dicts with keys
    # ``iteration_idx``, ``missing_aspects``, ``reformulated_query``,
    # ``evidence_added_count``, ``duration_ms``. Lets E2E tests
    # assert byte-equal against per-iter goldens (baseline vs
    # with-constraint) so the LM's actual consumption of the
    # constraint surfaces — not just that the channel buffered it.
    # The last iteration's ``evidence_added_count`` is the slice
    # index for cooperative-stop partial-evidence comparisons.
    loop_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Total loop wall-clock duration in ms. Used by cooperative-stop
    # tests to assert the stop actually short-circuited rather than
    # waiting for the full loop.
    duration_ms: float = 0.0
    # Per-iteration wall-clock durations in ms. Indexed by iteration.
    per_iter_duration_ms: List[float] = field(default_factory=list)


class OrchestrationSignature(dspy.Signature):
    """Plan a multi-agent workflow for a user query.

    Rules the plan MUST follow:
      1. Include preprocessing agents BEFORE the execution agent when they
         are available and useful:
           - `entity_extraction_agent` for queries naming specific entities,
             topics, or relationships.
           - `query_enhancement_agent` for vague, short, or ambiguous queries
             that benefit from expansion or rewriting.
           - `profile_selection_agent` when multiple search profiles are
             available and the query's modality/domain should pick one.
         These preprocessing agents can run in parallel with each other; the
         execution agent depends on them.
      2. End with exactly one execution agent (`search_agent`,
         `summarizer_agent`, `detailed_report_agent`, `image_search_agent`,
         `audio_analysis_agent`, `document_agent`, `coding_agent`,
         `deep_research_agent`) matching the query's modality and
         generation_type.
      3. Only include agents from `available_agents`. Never hallucinate.
      4. For trivial queries where preprocessing adds no value, a single
         execution agent is acceptable.
    """

    query: str = dspy.InputField(desc="User query to process")
    available_agents: str = dspy.InputField(
        desc="Comma-separated list of available agents"
    )
    conversation_context: str = dspy.InputField(
        desc="Summary of previous conversation turns. Empty string if first turn."
    )
    gateway_context: str = dspy.InputField(
        desc=(
            "Classification hints from the gateway: modality, generation_type, "
            "matched workflow template. Use these as priors when choosing "
            "preprocessing and execution agents. Empty string if not available."
        )
    )

    agent_sequence: str = dspy.OutputField(
        desc=(
            "Comma-separated sequence of agents to invoke. Preprocessing "
            "first, then execution, e.g. "
            "'entity_extraction_agent,query_enhancement_agent,search_agent'."
        )
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
        http_client: Optional[httpx.AsyncClient] = None,
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
            http_client: Optional httpx client used for A2A sub-agent calls.
                When provided (e.g. from
                ``SandboxManager.make_http_client("orchestrator_agent")``)
                the policy-enforcing transport applies; when omitted, the
                loop-scoped shared client is used.

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
        self._http_client_override = http_client

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

            async def _load():
                await self.workflow_intelligence.load_historical_data()

            run_coro_blocking(_load())

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

            # Extract provider from model string (e.g., "openai/gpt-4o" -> "openai")
            provider = (
                resolved.model.split("/")[0] if "/" in resolved.model else "local"
            )
            llm_model = resolved.model
            llm_base_url = resolved.api_base or "http://localhost:11434"

            sys_cfg = self._config_manager.get_system_config()
            denseon_url = sys_cfg.inference_service_urls.get("denseon")
            if not denseon_url:
                raise RuntimeError(
                    "orchestrator_agent memory init requires the denseon "
                    "inference service. Available: "
                    f"{sorted(sys_cfg.inference_service_urls)}"
                )

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
                embedding_model="lightonai/DenseOn",
                llm_base_url=llm_base_url,
                embedder_base_url=denseon_url,
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

    def _build_deep_synthesis_workflow(self):
        """Construct a :class:`DeepSynthesisWorkflow` for opt-in deep mode.

        Used by ``_process_impl`` when ``input.synthesis_depth == "deep"``.
        Returns ``None`` when the workflow's prerequisites are not yet
        wired (no LLM config or no in-process registry to dispatch
        through). Kept as a helper so tests can swap in a stub workflow
        without monkey-patching the full constructor.
        """
        try:
            from cogniverse_agents.deep_synthesis_workflow import (
                DeepSynthesisConfig,
                DeepSynthesisWorkflow,
            )
            from cogniverse_agents.inference.rlm_inference import (
                RLMInference,
                route_rlm_endpoint,
            )
            from cogniverse_foundation.config.utils import get_config
        except Exception as exc:
            logger.debug("Deep synthesis prerequisites missing: %s", exc)
            return None

        try:
            tenant_id = getattr(self.deps, "tenant_id", None) or "__system__"
            cfg = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
            llm_primary = cfg.get_llm_config().primary
        except Exception as exc:
            logger.debug("Could not resolve LLM config for deep synthesis: %s", exc)
            return None

        llm_primary = route_rlm_endpoint(llm_primary, self._config_manager, tenant_id)
        rlm = RLMInference(llm_config=llm_primary, tenant_id=tenant_id)

        async def _dispatcher(query: str, sub_agent_name: str) -> str:
            """Send a sub-query to a registered sub-agent over the
            orchestrator's HTTP path.

            Uses the orchestrator's existing http_client (which carries
            the OpenShell policy-enforcing transport when configured)
            to POST the sub-query to the agent's process_endpoint. The
            response's ``answer``/``output``/``content``/``result`` field
            is returned as the snippet — DeepSynthesisWorkflow appends
            these per-round to its trajectory.

            The previous version of this dispatcher returned a STUB
            string ("stub dispatch to ...") regardless of registration —
            the deep-synthesis loop ran but synthesised against placeholder
            content, never real sub-agent output.
            """
            try:
                ep = self.registry.get_agent(sub_agent_name)
            except Exception:
                ep = None
            if ep is None:
                return f"(sub-agent {sub_agent_name} not registered)"

            try:
                client = self._http_client_override or await _get_http_client()
                process_url = ep.url.rstrip("/") + (
                    ep.process_endpoint or f"/agents/{sub_agent_name}/process"
                )
                tenant_id_for_call = (
                    getattr(self.deps, "tenant_id", None) or "__system__"
                )
                resp = await client.post(
                    process_url,
                    json={
                        "query": query,
                        "context": {"tenant_id": tenant_id_for_call},
                    },
                    timeout=getattr(ep, "timeout", 30),
                )
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            except Exception as exc:
                logger.debug(
                    "deep-synth sub-dispatch %s failed: %s", sub_agent_name, exc
                )
                return f"(sub-agent {sub_agent_name} call failed: {type(exc).__name__})"

            if isinstance(payload, dict):
                for key in ("answer", "output", "content", "result", "summary"):
                    val = payload.get(key)
                    if isinstance(val, str) and val:
                        return val
                # No known answer field — surface the JSON so the RLM
                # step can still extract something, rather than dropping
                # the response entirely.
                return json.dumps(payload, default=str)[:2000]
            return str(payload)[:2000]

        return DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher,
            config=DeepSynthesisConfig(),
        )

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

        # Workflow owns its own per-tenant rate limit + hard call cap; on
        # any failure fall back to plan-then-act so the request still completes.
        if (input.synthesis_depth or "").lower() == "deep":
            workflow = self._build_deep_synthesis_workflow()
            if workflow is not None:
                try:
                    seed_subagents = list(self.registry.list_agents())[:6]
                    result = await workflow.run(
                        query=input.query,
                        tenant_id=input.tenant_id,
                        seed_subagents=seed_subagents,
                    )
                    return OrchestratorOutput(
                        query=input.query,
                        workflow_id="deep_synthesis",
                        plan_steps=[],
                        plan_reasoning="Deep synthesis workflow selected",
                        agent_results={},
                        final_output={
                            "answer": result.answer,
                            "iterations_used": result.iterations_used,
                            "subagent_calls_made": result.subagent_calls_made,
                            "llm_calls_used": result.llm_calls_used,
                            "was_capped": result.was_capped,
                            "was_submitted": result.was_submitted,
                            "was_rate_limited": result.was_rate_limited,
                        },
                        execution_summary=(
                            f"deep_synthesis: iter={result.iterations_used}, "
                            f"submitted={result.was_submitted}, "
                            f"rate_limited={result.was_rate_limited}"
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "Deep synthesis workflow failed (%s); falling back "
                        "to plan-then-act",
                        exc,
                    )

        query = input.query
        # Canonicalize so every downstream span / artifact lookup in
        # this request uses the same form. Without this the orchestrator
        # emits ``cogniverse.orchestration`` under the raw project and
        # later request-handler callers query under the canonical one.
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        tenant_id = canonical_tenant_id(input.tenant_id)
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

        sem = _get_orchestration_semaphore()
        async with sem:
            with self._gateway_lm_context(tenant_id):
                return await self._process_impl_locked(
                    input, workflow_id, query, tenant_id, session_id
                )

    def _gateway_lm_context(self, tenant_id: str):
        """Per-request LM routing for the orchestrator's DSPy calls.

        The orchestrator serves every tenant from one instance and relies on
        the global ``dspy.settings.lm``, so per-tenant gateway routing has to
        happen per request. When ``gateway_routing`` is enabled this returns a
        ``dspy.context(lm=...)`` whose LM is built from the request tenant and
        the ``orchestrator_agent`` endpoint; the context wraps the whole
        ``_process_impl_locked`` body so the planner, the retrieval-loop gate,
        and the aggregator all inherit it (contextvars propagate through the
        ``await``/``asyncio.to_thread`` calls inside).

        Disabled (the default) or on any resolution error it returns a
        ``nullcontext`` — the global LM path, byte-for-byte unchanged.
        """
        from cogniverse_foundation.config.gateway_routing import (
            routed_lm_context_for,
        )

        return routed_lm_context_for(
            self._config_manager, tenant_id, "orchestrator_agent"
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

        # Register the inbound-messaging session before any agent work
        # starts so concurrent callers hitting ``POST
        # /agents/{name}/message`` see 202 instead of 404. Closed in
        # the ``finally`` block below; closure makes subsequent POSTs
        # 404 and raises ``QueueClosedError`` on any in-flight enqueue.
        inbound_queue: Optional["_InboundQueue"] = None
        if session_id and _get_inbound_queue_registry is not None:
            try:
                _reg = await _resolve_inbound_registry_for_orchestrator(
                    redis_url=self._config_manager.get_system_config().redis_url
                )
                if _reg is None:
                    raise RuntimeError("no inbound registry available")
                inbound_queue = await _reg.get_or_create_queue(session_id, tenant_id)
            except ValueError as exc:
                # Cross-tenant session collision — surface as a routing
                # bug. The orchestrator can still run without the inbound
                # channel; we just won't accept inbound messages.
                logger.warning(
                    "Inbound session register failed for session_id=%s "
                    "tenant_id=%s: %s",
                    session_id,
                    tenant_id,
                    exc,
                )
                inbound_queue = None

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

        # Planning: fold gateway classification + workflow template
        # matches into a single hints string for the DSPy planner.
        self.emit_progress("planning", "Creating execution plan...")
        gateway_hints: List[str] = []
        if input.modality:
            gateway_hints.append(f"modality={input.modality}")
        if input.generation_type:
            gateway_hints.append(f"generation_type={input.generation_type}")
        if self.workflow_intelligence:
            template = self.workflow_intelligence._find_matching_template(query)
            if template:
                gateway_hints.append(
                    f"matched_template={template.name} "
                    f"suggested_sequence={json.dumps(template.task_sequence)}"
                )
                logger.info(f"Workflow intelligence matched template: {template.name}")
        gateway_context = "; ".join(gateway_hints)

        plan = await self._create_plan(query, conversation_context, gateway_context)

        # Track active workflow for cancellation
        self.active_workflows[workflow_id] = plan

        try:
            # Action: drive the iterative retrieval loop.
            # The loop IS the retrieval path — it calls ``_execute_plan``
            # internally, once per iteration, and feeds the
            # sufficient-context gate between iterations. There is no
            # flag and no single-shot fallback.
            self.emit_progress("execution", "Executing iterative retrieval loop...")
            agent_results: Dict[str, Any] = {}
            loop_result = await self._iterative_retrieval_loop(
                query=query,
                plan=plan,
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                session_id=session_id,
                agent_results_sink=agent_results,
                inbound_queue=inbound_queue,
            )

            # Record error entries for agents the LLM proposed but aren't registered
            for agent_name in plan.unavailable_agents:
                agent_results[agent_name] = {
                    "status": "error",
                    "message": f"Agent '{agent_name}' is not available in the registry",
                }

            self.emit_progress("aggregating", "Merging results from all agents")
            final_output = self._aggregate_results(query, agent_results)
            # Cap the ranked evidence list at top-5 — anything beyond is
            # noise for the downstream eval harness, and a tighter cap
            # keeps the metadata payload bounded for callers that fan
            # results into telemetry. The BRIGHT probe harness only
            # asserts on the top-1 hit per query.
            ranked_hits = self._rank_evidence_for_metadata(loop_result.evidence)
            top_hits = ranked_hits[:5]
            missing_aspects = list(
                loop_result.final_gate_output.get("missing_aspects") or []
            )
            if top_hits:
                first = top_hits[0]
                final_answer_id = (
                    f"{first.get('source_doc_id', '')}::{first.get('segment_id', '')}"
                )
            else:
                final_answer_id = ""
            iterative_loop = {
                "iterations_executed": loop_result.iterations_executed,
                "exit_reason": loop_result.exit_reason,
                "evidence_count": len(loop_result.evidence),
                "final_gate": loop_result.final_gate_output,
                "partial_due_to_budget": loop_result.partial_due_to_budget,
                "partial_due_to_timeout": loop_result.partial_due_to_timeout,
                "trace_id": loop_result.trace_id,
                "top_hits": top_hits,
                "missing_aspects": missing_aspects,
                "final_answer_id": final_answer_id,
                "inbound_constraints_applied": list(
                    loop_result.inbound_constraints_applied
                ),
                "loop_trajectory": list(loop_result.loop_trajectory),
                "duration_ms": loop_result.duration_ms,
                "per_iter_duration_ms": list(loop_result.per_iter_duration_ms),
                "accumulated_evidence": list(loop_result.evidence),
            }
            final_output["iterative_loop"] = iterative_loop
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
                    from datetime import datetime, timedelta, timezone

                    from cogniverse_agents.workflow_types import (
                        WorkflowPlan as WFPlan,
                    )
                    from cogniverse_agents.workflow_types import (
                        WorkflowStatus,
                        WorkflowTask,
                    )

                    _now = datetime.now(timezone.utc)
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
                        start_time=_now - timedelta(seconds=execution_time),
                        end_time=_now,
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
                metadata={"iterative_loop": iterative_loop},
            )
        finally:
            self.active_workflows.pop(workflow_id, None)
            # Close the inbound-messaging session — subsequent POSTs to
            # /agents/{name}/message for this session_id return 404 and
            # any race with an in-flight enqueue surfaces
            # QueueClosedError (handled by the HTTP route as 404).
            if session_id and _get_inbound_queue_registry is not None:
                try:
                    _reg = await _resolve_inbound_registry_for_orchestrator(
                        redis_url=self._config_manager.get_system_config().redis_url
                    )
                    if _reg is not None:
                        await _reg.close_queue(session_id)
                except Exception as exc:  # noqa: BLE001 — log + degrade
                    logger.warning(
                        "Inbound session close failed for session_id=%s: %s",
                        session_id,
                        exc,
                    )

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
        agent_sequence = [a.strip() for a in raw_sequence.split(",") if a.strip()]
        if not agent_sequence:
            # DSPy planner returned empty/None — fall back to search
            logger.warning(
                "DSPy planner returned empty agent_sequence, falling back to search_agent"
            )
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

        unavailable_agents = []
        # Build lookup for agent name normalization (LLM may add/omit _agent suffix)
        _agent_lookup = {name: name for name in registered_agents}
        for name in registered_agents:
            _agent_lookup[f"{name}_agent"] = name
            if name.endswith("_agent"):
                _agent_lookup[name[: -len("_agent")]] = name

        # Pass 1: filter to registered agents, keeping the raw sequence index so
        # parallel-group / dependency indices can be remapped to the surviving
        # step positions below (identity when nothing is filtered).
        surviving = []  # (raw_sequence_index, agent_name)
        for i, agent_name in enumerate(agent_sequence):
            agent_name = _agent_lookup.get(agent_name, agent_name)
            if agent_name not in registered_agents:
                logger.warning(
                    f"LLM proposed unknown agent '{agent_name}', "
                    f"not in registry ({registered_agents}), skipping"
                )
                unavailable_agents.append(agent_name)
                continue
            surviving.append((i, agent_name))

        raw_to_step = {raw: pos for pos, (raw, _) in enumerate(surviving)}

        # parallel_steps and the dependency calculation index into the raw
        # sequence; remap them to surviving-step positions (which is what
        # _execute_plan's executed[] / depends_on space expects), dropping any
        # index pointing at a filtered agent and any group emptied by filtering.
        parallel_groups = [
            [raw_to_step[idx] for idx in group if idx in raw_to_step]
            for group in parallel_groups
        ]
        parallel_groups = [g for g in parallel_groups if g]

        # Pass 2: build steps with dependencies in the surviving-step index space.
        steps = [
            AgentStep(
                agent_name=agent_name,
                input_data={"query": query},
                depends_on=self._calculate_dependencies(step_index, parallel_groups),
                reasoning=f"Step {step_index + 1}: {agent_name} processing",
            )
            for step_index, (_, agent_name) in enumerate(surviving)
        ]

        # Terminal fallback: when the planner proposed only unknown agents
        # (or proposed nothing), the executor would otherwise stall with
        # "No steps ready to execute but execution incomplete". Drop in
        # a single ``search`` step against the registered search agent so
        # the loop still produces a retrieval pass. If ``search`` itself
        # isn't registered, leave ``steps`` empty and let the caller
        # surface the empty-plan condition.
        if not steps:
            search_name = next(
                (
                    candidate
                    for candidate in ("search", "search_agent", "video_search_agent")
                    if candidate in registered_agents
                ),
                None,
            )
            if search_name is not None:
                logger.warning(
                    f"Planner produced 0 executable steps after filtering "
                    f"{unavailable_agents}; falling back to a single "
                    f"'{search_name}' step"
                )
                steps.append(
                    AgentStep(
                        agent_name=search_name,
                        input_data={"query": query},
                        depends_on=[],
                        reasoning="Terminal fallback after planner produced no executable steps",
                    )
                )

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

                # Merge enrichment from every preprocessing agent that has
                # completed so far, not just direct dependencies. The DSPy
                # planner often wires search to depend on
                # [query_enhancement, profile_selection] only, which would
                # otherwise drop entity_extraction's entities. Walking all
                # completed results keeps enrichment additive.
                for dep_agent, dep_result in agent_results.items():
                    if dep_agent == agent_name:
                        continue
                    if dep_result:
                        _merge_enrichment(agent_input, dep_agent, dep_result)

                # Skips if caller set explicit ``rlm`` field, or if the agent
                # is the orchestrator itself (no recursive promotion).
                _maybe_promote_to_rlm(agent_name, agent_input)

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
                    http_client = (
                        self._http_client_override
                        if self._http_client_override is not None
                        else await _get_http_client()
                    )
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

    # ------------------------------------------------------------------
    # Iterative retrieval loop
    # ------------------------------------------------------------------

    def _get_query_analysis_module(self):
        """Lazily build the ``ComposableQueryAnalysisModule`` reused per
        iteration of the retrieval loop. Cached on the instance so the
        GLiNER + spaCy extractors load once."""
        if getattr(self, "_query_analysis_module", None) is not None:
            return self._query_analysis_module
        from cogniverse_agents.routing.dspy_relationship_router import (
            create_composable_query_analysis_module,
        )

        # Resolve the configured GLiNER model + remote endpoint so the slim
        # runtime image (no in-process gliner/torch) routes extraction through
        # the inference service instead of failing to import gliner.
        gliner_model = None
        gliner_inference_url = None
        try:
            sys_cfg = self._config_manager.get_system_config()
            gliner_model = getattr(sys_cfg, "gliner_model", None)
            gliner_inference_url = (sys_cfg.inference_service_urls or {}).get("gliner")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("GLiNER config lookup failed for query analysis: %s", exc)

        self._query_analysis_module = create_composable_query_analysis_module(
            gliner_model=gliner_model,
            gliner_inference_url=gliner_inference_url,
        )
        return self._query_analysis_module

    @staticmethod
    def _coerce_evidence_snippet(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reshape a search/KG result dict into the gate's evidence schema.

        Returns ``None`` when the raw dict can't be coerced (no usable
        text + no doc id); the loop just skips such entries.
        """
        if not isinstance(raw, dict):
            return None

        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        temporal = (
            raw.get("temporal_info")
            if isinstance(raw.get("temporal_info"), dict)
            else {}
        )

        source_doc_id = (
            raw.get("source_doc_id")
            or raw.get("document_id")
            or raw.get("documentid")
            or metadata.get("source_doc_id")
            or metadata.get("video_id")
            or raw.get("id")
            or ""
        )
        segment_id = (
            raw.get("segment_id")
            or metadata.get("segment_id")
            or metadata.get("frame_id")
            or ""
        )
        ts_start = (
            raw.get("ts_start")
            if raw.get("ts_start") is not None
            else temporal.get("start_time", metadata.get("start_time", 0.0))
        )
        ts_end = (
            raw.get("ts_end")
            if raw.get("ts_end") is not None
            else temporal.get("end_time", metadata.get("end_time", 0.0))
        )
        text = (
            raw.get("text")
            or raw.get("transcript")
            or metadata.get("transcript")
            or metadata.get("description")
            or metadata.get("text")
            or ""
        )

        if not source_doc_id and not text:
            return None

        snippet: Dict[str, Any] = {
            "source_doc_id": str(source_doc_id),
            "segment_id": str(segment_id),
            "ts_start": float(ts_start) if ts_start is not None else 0.0,
            "ts_end": float(ts_end) if ts_end is not None else 0.0,
            "text": str(text),
        }
        if raw.get("score") is not None:
            snippet["score"] = raw["score"]
        if metadata.get("modality"):
            snippet["modality"] = metadata["modality"]
        return snippet

    def _extract_evidence_from_results(
        self, agent_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Pull evidence snippets out of agent results.

        Walks every agent result, looks for the ``results`` list shape
        produced by the search/document/coding agents and the ``nodes``
        list shape produced by ``kg_traversal_agent``, and coerces each
        hit through :meth:`_coerce_evidence_snippet`.
        """
        snippets: List[Dict[str, Any]] = []
        for result in agent_results.values():
            if not isinstance(result, dict):
                continue
            for hit in result.get("results") or []:
                snippet = self._coerce_evidence_snippet(hit)
                if snippet is not None:
                    snippets.append(snippet)
            for node in result.get("nodes") or []:
                snippet = self._coerce_evidence_snippet(node)
                if snippet is not None:
                    snippets.append(snippet)
        return snippets

    @staticmethod
    def _rank_evidence_for_metadata(
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank evidence snippets for the ``iterative_loop.top_hits`` field.

        Sorts by descending ``score`` (snippets without a score sort
        after scored ones) and reshapes each entry to the
        ``{source_doc_id, segment_id, ts_start, ts_end, score}`` schema
        the BRIGHT probe harness asserts against. Keeps the input list
        immutable so callers retain the full snippet for other paths.
        """
        scored: List[Dict[str, Any]] = []
        unscored: List[Dict[str, Any]] = []
        for hit in evidence:
            if not isinstance(hit, dict):
                continue
            entry = {
                "source_doc_id": str(hit.get("source_doc_id") or ""),
                "video_id": str(hit.get("source_doc_id") or hit.get("video_id") or ""),
                "segment_id": str(hit.get("segment_id") or ""),
                "ts_start": float(hit.get("ts_start") or 0.0),
                "ts_end": float(hit.get("ts_end") or 0.0),
                "score": hit.get("score"),
            }
            if entry["score"] is None:
                unscored.append(entry)
            else:
                scored.append(entry)
        scored.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
        return scored + unscored

    @staticmethod
    def _deduplicate_evidence(
        snippets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop snippets the loop has already seen (same doc+segment+text)."""
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for snippet in snippets:
            key = (
                snippet.get("source_doc_id", ""),
                snippet.get("segment_id", ""),
                snippet.get("text", "")[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(snippet)
        return unique

    async def _reformulate_query(
        self, query: str, missing_aspects: List[str]
    ) -> tuple[str, str]:
        """Run the query analysis module and return (reformulated, rationale).

        On the first iteration ``missing_aspects`` is empty and the
        reformulator works from the original query alone. On subsequent
        iterations the missing aspects are appended so the reformulator
        targets the gaps the gate identified.
        """
        seeded_query = query
        if missing_aspects:
            joined = "; ".join(str(a) for a in missing_aspects if a)
            if joined:
                seeded_query = f"{query} (focus on: {joined})"

        analysis_module = self._get_query_analysis_module()
        try:
            prediction = await asyncio.to_thread(
                analysis_module.forward,
                query=seeded_query,
                search_context="general",
            )
        except Exception as exc:
            logger.warning(
                "Query reformulation failed in iterative loop (%s); "
                "falling back to seeded query",
                exc,
            )
            return seeded_query, ""

        reformulated_query = getattr(prediction, "enhanced_query", None) or seeded_query
        rationale = getattr(prediction, "reasoning", "") or ""
        return reformulated_query, rationale

    def _build_gate_module(self, evidence_chars: int):
        """Return the DSPy module that drives the sufficient-context gate.

        Promotes from a single-prompt ``ChainOfThought`` to ``InstrumentedRLM``
        when the JSON-serialized evidence exceeds the promotion threshold.
        The RLM substrate breaks the gate decision into iterative
        reasoning steps so the prompt budget doesn't blow up.
        """
        if evidence_chars > _ITER_GATE_RLM_PROMOTION_CHARS:
            from cogniverse_agents.inference.instrumented_rlm import InstrumentedRLM

            tenant_id = (
                getattr(self, "_current_tenant_id", None)
                or getattr(self.deps, "tenant_id", None)
                or SYSTEM_TENANT_ID
            )
            return InstrumentedRLM(
                SufficientContextSignature,
                event_queue=self.event_queue,
                task_id=f"sufficient_context_gate_{uuid.uuid4().hex[:8]}",
                tenant_id=tenant_id,
            )
        return dspy.ChainOfThought(SufficientContextSignature)

    async def _run_sufficiency_gate(
        self,
        original_query: str,
        accumulated_evidence: List[Dict[str, Any]],
        iteration_idx: int,
    ) -> Dict[str, Any]:
        """Invoke the sufficient-context gate and return its decision dict.

        When the gate is promoted to ``InstrumentedRLM`` (evidence JSON
        above the promotion threshold), the call is wrapped in an
        ``InstrumentedRLM.run`` telemetry span carrying the number of
        REPL iterations the RLM actually used, so callers can correlate
        the heavier substrate with its iteration cost.
        """
        from cogniverse_agents.inference.instrumented_rlm import InstrumentedRLM

        evidence_chars = len(json.dumps(accumulated_evidence, default=str))
        module = self._build_gate_module(evidence_chars)
        is_rlm = isinstance(module, InstrumentedRLM)

        async def _invoke() -> Any:
            return await asyncio.to_thread(
                module,
                original_query=original_query,
                accumulated_evidence=accumulated_evidence,
                iteration_idx=iteration_idx,
            )

        prediction: Any = None
        gate_error: Optional[Exception] = None
        if is_rlm and getattr(self, "telemetry_manager", None) is not None:
            tenant_id = (
                getattr(self, "_current_tenant_id", None)
                or getattr(self.deps, "tenant_id", None)
                or SYSTEM_TENANT_ID
            )
            try:
                with self.telemetry_manager.span(
                    name="InstrumentedRLM.run",
                    tenant_id=tenant_id,
                    attributes={
                        "iteration_idx": int(iteration_idx),
                        "evidence_chars": int(evidence_chars),
                        "max_iterations": int(getattr(module, "max_iterations", 0)),
                    },
                ) as rlm_span:
                    try:
                        prediction = await _invoke()
                    except Exception as exc:
                        gate_error = exc
                    # Record the RLM's actual REPL iteration count from
                    # the returned Prediction.trajectory (dspy.RLM populates
                    # this with one entry per REPL step). Falls back to
                    # max_iterations when trajectory isn't available
                    # (e.g. early failure).
                    rlm_iterations = 0
                    if prediction is not None:
                        traj = getattr(prediction, "trajectory", None)
                        if isinstance(traj, list):
                            rlm_iterations = len(traj)
                    try:
                        rlm_span.set_attribute("rlm_iterations", int(rlm_iterations))
                    except Exception:
                        logger.debug(
                            "Failed to set rlm_iterations attribute on RLM span"
                        )
            except Exception as exc:  # pragma: no cover - telemetry best-effort
                logger.debug("InstrumentedRLM.run span emission failed: %s", exc)
                if prediction is None and gate_error is None:
                    try:
                        prediction = await _invoke()
                    except Exception as exc2:
                        gate_error = exc2
        else:
            try:
                prediction = await _invoke()
            except Exception as exc:
                gate_error = exc

        if gate_error is not None:
            logger.warning(
                "Sufficient-context gate failed at iter=%d (%s); "
                "defaulting to insufficient to keep loop honest",
                iteration_idx,
                gate_error,
            )
            return {
                "sufficient": False,
                "missing_aspects": [],
                "confidence": 0.0,
                "rationale": f"gate_error: {type(gate_error).__name__}",
            }

        missing = getattr(prediction, "missing_aspects", None) or []
        if not isinstance(missing, list):
            missing = [str(missing)]
        try:
            confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "sufficient": bool(getattr(prediction, "sufficient", False)),
            "missing_aspects": [str(a) for a in missing],
            "confidence": confidence,
            "rationale": str(getattr(prediction, "rationale", "") or ""),
        }

    @staticmethod
    def _evidence_token_estimate(evidence: List[Dict[str, Any]]) -> int:
        """Estimate gate-prompt tokens including dspy adapter scaffolding.

        The token budget for the iterative loop is a budget on what gets
        sent to the LM at the *next* gate call. dspy's chat adapter wraps
        ``SufficientContextSignature`` with the full input/output schema
        descriptions, field markers, and CoT instructions — roughly
        ``_GATE_PROMPT_SCAFFOLDING_CHARS`` characters before any user
        content. Counting only the evidence JSON (as a naive chars/4)
        understates the actual LM prompt by an order of magnitude and
        lets the loop keep iterating long after the per-call prompt has
        outgrown the budget. We add the scaffolding constant to the
        evidence JSON length and divide by 4 for an approximate token
        count.
        """
        evidence_chars = len(json.dumps(evidence, default=str))
        return (_GATE_PROMPT_SCAFFOLDING_CHARS + evidence_chars) // 4

    @staticmethod
    def _evidence_video_anchor(
        evidence: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the first evidence snippet with a usable doc + timestamp
        anchor — used to scope the KG traversal expansion call."""
        for snippet in evidence:
            doc_id = snippet.get("source_doc_id")
            if not doc_id:
                continue
            ts_start = snippet.get("ts_start", 0.0) or 0.0
            ts_end = snippet.get("ts_end", 0.0) or 0.0
            return {
                "source_doc_id": doc_id,
                "ts_start": float(ts_start),
                "ts_end": float(ts_end),
            }
        return None

    async def _expand_via_kg_traversal(
        self,
        evidence: List[Dict[str, Any]],
        missing_aspects: List[str],
        tenant_id: str,
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Call ``kg_traversal_agent`` (if registered) to expand the
        evidence frontier into the knowledge graph, scoped to the video
        + time window of the strongest existing evidence snippet."""
        try:
            endpoint = self.registry.get_agent("kg_traversal_agent")
        except Exception:
            endpoint = None
        if endpoint is None:
            return []

        anchor = self._evidence_video_anchor(evidence)
        if anchor is None:
            return []

        context: Dict[str, Any] = {"tenant_id": tenant_id}
        if session_id:
            context["session_id"] = session_id

        seed_subject = (
            missing_aspects[0] if missing_aspects else anchor["source_doc_id"]
        )
        payload = {
            "agent_name": "kg_traversal_agent",
            "query": "; ".join(missing_aspects) if missing_aspects else "",
            "context": context,
            "start_subject_key": seed_subject,
            "max_depth": 2,
            "max_edges": 50,
        }

        client = (
            self._http_client_override
            if self._http_client_override is not None
            else await _get_http_client()
        )
        try:
            response = await client.post(
                f"{endpoint.url}{endpoint.process_endpoint}",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            logger.debug(
                "kg_traversal expansion failed (%s: %s); continuing without it",
                type(exc).__name__,
                exc,
            )
            return []

        self._emit_kg_traversal_span(
            tenant_id=tenant_id,
            node_name=str(seed_subject),
            anchor=anchor,
            result=result,
        )
        return self._extract_evidence_from_results({"kg_traversal_agent": result})

    def _emit_kg_traversal_span(
        self,
        *,
        tenant_id: str,
        node_name: str,
        anchor: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Emit a ``KnowledgeGraphTraversalAgent.traverse`` span carrying the
        seed subject, the evidence time window, and the traversed node ids."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            node_ids = [
                n.get("name") for n in (result.get("nodes") or []) if n.get("name")
            ]
            with self.telemetry_manager.span(
                name="KnowledgeGraphTraversalAgent.traverse",
                tenant_id=tenant_id,
                attributes={
                    "node_name": node_name,
                    "filter_ts_start": float(anchor["ts_start"]),
                    "filter_ts_end": float(anchor["ts_end"]),
                    "result_node_ids": json.dumps(node_ids),
                },
            ):
                pass
        except Exception as exc:
            logger.debug("Failed to emit kg traversal span: %s", exc)

    def _emit_retrieval_iteration_span(
        self,
        *,
        tenant_id: str,
        iteration_idx: int,
        sufficiency_score: float,
        exit_reason: str,
        evidence_count: int,
        session_id: Optional[str] = None,
        inbound_constraints_applied: Optional[List[str]] = None,
    ) -> None:
        """Emit a ``retrieval_iteration`` Phoenix span carrying
        ``iteration_idx``, ``sufficiency_score``, ``exit_reason``, and
        ``evidence_count`` so trajectory-level evaluation can grade each
        loop iteration independently.

        ``session_id`` + ``inbound_constraints_applied`` are added as
        attributes so end-to-end tests can query Phoenix spans by
        session_id (deterministic across runs) to verify the inbound
        channel reached the orchestrator's loop even when the
        OTEL-context trace_id isn't propagated to the response.
        """
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            attrs: Dict[str, Any] = {
                "iteration_idx": int(iteration_idx),
                "sufficiency_score": float(sufficiency_score),
                "exit_reason": exit_reason,
                "evidence_count": int(evidence_count),
            }
            if session_id:
                attrs["session_id"] = session_id
            if inbound_constraints_applied:
                # Phoenix expects string-valued attributes; serialize
                # the list compactly so consumers can split if needed.
                attrs["inbound_constraints_applied"] = "|".join(
                    inbound_constraints_applied
                )
            with self.telemetry_manager.span(
                name="retrieval_iteration",
                tenant_id=tenant_id,
                attributes=attrs,
            ):
                pass
        except Exception as exc:
            logger.debug("Failed to emit retrieval_iteration span: %s", exc)

    async def _iterative_retrieval_loop(
        self,
        query: str,
        plan: OrchestrationPlan,
        *,
        tenant_id: str,
        workflow_id: str,
        session_id: Optional[str],
        agent_results_sink: Dict[str, Any],
        inbound_queue: Optional["_InboundQueue"] = None,
    ) -> AccumulatedEvidence:
        """Run the retrieve → gate → reformulate loop bounded by hard caps.

        The loop IS the retrieval path — there is no fallback to a
        single-shot execution. Bounded by ``SystemConfig.iter_retrieval_max_iter``
        iterations, ``SystemConfig.iter_retrieval_token_budget`` cumulative tokens
        across all evidence, and ``SystemConfig.iter_retrieval_wall_clock_ms``
        wall-clock milliseconds.

        ``agent_results_sink`` receives the per-agent results from each
        iteration so the caller can keep emitting them (the orchestration
        envelope still surfaces agent_results) without re-running steps.
        """
        loop_started = time.monotonic()
        # ``raw_accumulated`` keeps every snippet that came back from a
        # sub-agent so the gate sees the full weight of what was retrieved
        # (e.g. multiple corroborating hits for the same fact). The dedup'd
        # ``accumulated`` is what we hand back to the caller — that's the
        # clean, presentation-ready evidence set the answer is grounded on.
        # The split matters for RLM promotion: when many corroborating hits
        # come back, the raw JSON pushes ``evidence_chars`` above the
        # promotion threshold and the gate runs through the RLM substrate
        # instead of a single CoT call.
        raw_accumulated: List[Dict[str, Any]] = []
        accumulated: List[Dict[str, Any]] = []
        gate_output: Dict[str, Any] = {}
        exit_reason = "max_iter"
        partial_due_to_budget = False
        partial_due_to_timeout = False
        trace_id = ""

        # Capture the active trace id for the AccumulatedEvidence return.
        # OpenTelemetry exposes the current span via ``opentelemetry.trace``;
        # if that import fails or no span is active, trace_id stays empty.
        try:  # pragma: no cover - opentelemetry optional in some envs
            from opentelemetry import trace as _otel_trace

            current_span = _otel_trace.get_current_span()
            ctx = current_span.get_span_context()
            if ctx and ctx.trace_id:
                trace_id = f"{ctx.trace_id:032x}"
        except Exception:
            trace_id = ""

        # Constraint messages drained from the inbound queue across the
        # loop, prepended to ``missing_aspects`` at each iteration so
        # the reformulator sees the caller's steering. We keep them
        # across iterations (rather than re-applying on each drain)
        # so a constraint sent at iter 0 still influences iter 2's
        # reformulation. The list grows monotonically until loop exit.
        accumulated_inbound_constraints: List[str] = []
        # Per-iter trajectory: missing_aspects (input to reformulate),
        # reformulated_query (LM output), evidence_added_count (snippets
        # this iter contributed before dedup), duration_ms. Tests assert
        # byte-equal against per-iter goldens.
        loop_trajectory: List[Dict[str, Any]] = []
        iterations_executed = 0
        # Read loop-tuning caps from SystemConfig (set at runtime
        # startup from ITER_RETRIEVAL_* env vars). Read once per
        # /process call so a live ConfigManager update between
        # requests takes effect.
        _sys_cfg = self._config_manager.get_system_config()
        # Module-level overrides (set by tests via monkeypatch) take
        # precedence over SystemConfig. Production leaves them None.
        _max_iter = (
            _ITER_RETRIEVAL_MAX_ITER
            if _ITER_RETRIEVAL_MAX_ITER is not None
            else _sys_cfg.iter_retrieval_max_iter
        )
        _token_budget = (
            _ITER_RETRIEVAL_TOKEN_BUDGET
            if _ITER_RETRIEVAL_TOKEN_BUDGET is not None
            else _sys_cfg.iter_retrieval_token_budget
        )
        _wall_clock_ms = (
            _ITER_RETRIEVAL_WALL_CLOCK_MS
            if _ITER_RETRIEVAL_WALL_CLOCK_MS is not None
            else _sys_cfg.iter_retrieval_wall_clock_ms
        )
        for iter_idx in range(_max_iter):
            iteration_started = time.monotonic()

            # Drain inbound messages BEFORE building the next iteration's
            # query so user-injected constraints land in this iter's
            # reformulation, not the one after. A ``stop`` tag triggers
            # cooperative cancellation with the partial evidence
            # accumulated so far.
            if inbound_queue is not None:
                inbound_msgs = await inbound_queue.drain()
                for msg in inbound_msgs:
                    if "stop" in msg.tags:
                        exit_reason = "user_stop"
                        iterations_executed = iter_idx
                        # Also signal the EventQueue's cancellation
                        # token so any InstrumentedRLM currently
                        # running inside a sub-agent's chain observes
                        # the stop at its next REPL iteration and
                        # raises RLMCancelledError(reason="user_stop").
                        # The outer return path returns the partial
                        # evidence already accumulated.
                        event_queue = getattr(self, "event_queue", None)
                        if event_queue is not None:
                            try:
                                event_queue.cancel(reason="user_stop")
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "EventQueue.cancel failed during user_stop: %s",
                                    exc,
                                )
                        loop_duration_ms = (time.monotonic() - loop_started) * 1000.0
                        return AccumulatedEvidence(
                            evidence=accumulated,
                            iterations_executed=iterations_executed,
                            exit_reason=exit_reason,
                            final_gate_output=gate_output,
                            partial_due_to_budget=partial_due_to_budget,
                            partial_due_to_timeout=partial_due_to_timeout,
                            trace_id=trace_id,
                            inbound_constraints_applied=list(
                                accumulated_inbound_constraints
                            ),
                            loop_trajectory=list(loop_trajectory),
                            duration_ms=loop_duration_ms,
                            per_iter_duration_ms=[
                                t["duration_ms"] for t in loop_trajectory
                            ],
                        )
                    if (
                        "constraint" in msg.tags or "interrupt" in msg.tags
                    ) and msg.content:
                        accumulated_inbound_constraints.append(msg.content)

            base_missing = gate_output.get("missing_aspects", []) if gate_output else []
            # Inbound constraints take precedence over gate-derived
            # missing_aspects — the caller's explicit steering is more
            # specific than the gate's inferred gaps.
            missing_aspects = list(accumulated_inbound_constraints) + list(base_missing)
            reformulated_query, _ = await self._reformulate_query(
                query, missing_aspects
            )

            # Inject the reformulated query into every plan step so the
            # execution path re-runs against the targeted question.
            for step in plan.steps:
                step.input_data["query"] = reformulated_query

            iter_results = await self._execute_plan(
                plan,
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            agent_results_sink.update(iter_results)

            new_snippets = self._extract_evidence_from_results(iter_results)
            raw_accumulated = raw_accumulated + new_snippets
            accumulated = self._deduplicate_evidence(raw_accumulated)

            gate_output = await self._run_sufficiency_gate(
                original_query=query,
                accumulated_evidence=raw_accumulated,
                iteration_idx=iter_idx,
            )

            iterations_executed = iter_idx + 1

            # Record per-iter trajectory snapshot. Captured here so
            # both the early-break paths (sufficient / max_iter /
            # token_budget / wall_clock) and the loop-completes-
            # naturally path see the iteration's inputs + outputs.
            iter_duration_ms = (time.monotonic() - iteration_started) * 1000.0
            loop_trajectory.append(
                {
                    "iteration_idx": iter_idx,
                    "missing_aspects": list(missing_aspects),
                    "reformulated_query": reformulated_query,
                    "evidence_added_count": len(new_snippets),
                    "duration_ms": iter_duration_ms,
                }
            )

            self._emit_retrieval_iteration_span(
                tenant_id=tenant_id,
                iteration_idx=iterations_executed,
                sufficiency_score=gate_output.get("confidence", 0.0),
                exit_reason="in_progress",
                evidence_count=len(accumulated),
                session_id=session_id,
                inbound_constraints_applied=list(accumulated_inbound_constraints),
            )

            # Honor a ``sufficient`` gate decision only from the second
            # iteration onwards. Real-world LMs (especially smaller
            # student models) can flip between sufficient/insufficient on
            # the same evidence run-to-run; declaring "done" on the first
            # iteration with only a single retrieval pass behind us makes
            # the loop dependent on whichever way the model leans this
            # call. Requiring iter_idx >= 1 forces at least one
            # corroborating retrieval pass before we trust the gate.
            if iter_idx >= 1 and gate_output.get("sufficient"):
                exit_reason = "sufficient"
                break

            # Convergence heuristic — small / cautious LMs (gemma-class
            # student models) keep returning ``sufficient=False`` on the
            # gate question even when 3+ evidence snippets already span
            # the query's main aspects. The orchestrator must not run
            # unbounded retrievals chasing the gate's perfectionism. From
            # the second iteration onward, if every sub-agent in the plan
            # contributed at least one usable evidence snippet this
            # round, treat the loop as converged: continuing would just
            # re-query the same agents with the same reformulated query
            # and return the same evidence. The gate's missing-aspects
            # list is retained in ``final_gate_output`` so the caller can
            # still surface what the gate flagged.
            all_agents_returned_evidence = bool(iter_results) and all(
                bool(self._extract_evidence_from_results({agent: result}))
                for agent, result in iter_results.items()
            )
            # A constraint is folded into THIS iteration's reformulation
            # (above) and the evidence below is gathered against it before this
            # check, so it never blocks convergence — it keeps steering via
            # missing_aspects. The loop converges once every sub-agent
            # contributed evidence this round.
            if iter_idx >= 1 and all_agents_returned_evidence:
                exit_reason = "sufficient"
                break

            if iterations_executed >= _max_iter:
                exit_reason = "max_iter"
                break

            # Token budget reflects what the *next* gate call would send to
            # the LM, so we measure against the raw accumulated set (the
            # same set fed to the gate). Otherwise dedup hides the real
            # prompt cost and the loop keeps iterating past the cap.
            if self._evidence_token_estimate(raw_accumulated) > _token_budget:
                exit_reason = "token_budget"
                partial_due_to_budget = True
                break

            elapsed_ms = (time.monotonic() - loop_started) * 1000.0
            if elapsed_ms > _wall_clock_ms:
                exit_reason = "wall_clock"
                partial_due_to_timeout = True
                break

            # Otherwise keep going — expand the frontier via KG traversal
            # when an anchor is available, then loop back into the next
            # reformulation pass.
            kg_snippets = await self._expand_via_kg_traversal(
                accumulated,
                gate_output.get("missing_aspects", []),
                tenant_id=tenant_id,
                session_id=session_id,
            )
            if kg_snippets:
                raw_accumulated = raw_accumulated + kg_snippets
                accumulated = self._deduplicate_evidence(raw_accumulated)

            # Defensive guard against runaway iterations from clock skew.
            del iteration_started

        loop_duration_ms = (time.monotonic() - loop_started) * 1000.0
        return AccumulatedEvidence(
            evidence=accumulated,
            iterations_executed=iterations_executed,
            exit_reason=exit_reason,
            final_gate_output=gate_output,
            partial_due_to_budget=partial_due_to_budget,
            partial_due_to_timeout=partial_due_to_timeout,
            trace_id=trace_id,
            inbound_constraints_applied=list(accumulated_inbound_constraints),
            loop_trajectory=list(loop_trajectory),
            duration_ms=loop_duration_ms,
            per_iter_duration_ms=[t["duration_ms"] for t in loop_trajectory],
        )

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

            confidence = parse_confidence(
                result_data.get("confidence")
                if isinstance(result_data, dict)
                else None,
                default=0.5,
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
            "results": {name: tr["result"] for name, tr in task_results.items()},
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
        temporal_keywords = [
            "timeline",
            "sequence",
            "chronological",
            "when",
            "duration",
        ]
        if any(kw in query_lower for kw in temporal_keywords) and len(modalities) > 1:
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

        avg_confidence = (
            total_confidence / modality_count if modality_count > 0 else 0.0
        )
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
        return self.checkpoint_config.enabled and self.checkpoint_storage is not None

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

            from datetime import datetime, timezone

            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"ckpt_{uuid.uuid4().hex[:12]}",
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                workflow_status="running"
                if status == CheckpointStatus.ACTIVE
                else status.value,
                current_phase=current_step,
                original_query=plan.query,
                execution_order=[[f"step_{i}"] for i in range(len(plan.steps))],
                metadata={"reasoning": plan.reasoning},
                task_states=task_states,
                checkpoint_time=datetime.now(timezone.utc),
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
                    "orchestration.agent_sequence": ",".join(agent_sequence)
                    if agent_sequence
                    else "",
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
