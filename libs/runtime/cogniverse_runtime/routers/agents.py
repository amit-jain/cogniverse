"""Agent endpoints - unified interface for all agent operations."""

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_agents.routing.annotation_queue import AnnotationQueue

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from cogniverse_agents.search.vespa_query import VespaSearchDegraded
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.messaging import (
    InboundMessage,
    QueueClosedError,
    get_inbound_queue_registry,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader


async def _resolve_inbound_registry():
    """Pick the in-pod or Redis-backed inbound registry from config.

    Non-empty ``SystemConfig.redis_url`` → cross-pod durable Redis
    backend. Empty → in-pod singleton. The two paths share the same
    surface (``get_or_create_queue`` / ``get_queue`` /
    ``close_queue``) so the route logic below doesn't branch.

    Env reads for ``REDIS_URL`` happen at the runtime startup
    boundary (see ``main.py``); this route never touches env directly.
    Falls back to the in-pod registry when ``_config_manager`` hasn't
    been wired (test harnesses that mount the router directly without
    running the runtime lifespan).
    """
    redis_url = ""
    if _config_manager is not None:
        redis_url = _config_manager.get_system_config().redis_url
    if redis_url:
        from cogniverse_runtime.messaging_redis import (
            get_redis_inbound_queue_registry,
        )

        return await get_redis_inbound_queue_registry(redis_url)
    return get_inbound_queue_registry()


logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level dependencies (injected from main.py)
_agent_registry: Optional[AgentRegistry] = None
_config_manager: Optional[ConfigManager] = None
_schema_loader: Optional[SchemaLoader] = None
_dispatcher: Optional[AgentDispatcher] = None
_sandbox_manager = None


def set_agent_registry(registry: AgentRegistry) -> None:
    """Inject AgentRegistry dependency for router endpoints"""
    global _agent_registry, _dispatcher
    _agent_registry = registry
    _dispatcher = None  # Reset dispatcher so it picks up the new registry
    logger.info("AgentRegistry injected into agents router")


def set_sandbox_manager(sandbox_mgr) -> None:
    """Inject SandboxManager for agent dispatch."""
    global _sandbox_manager, _dispatcher
    _sandbox_manager = sandbox_mgr
    _dispatcher = None
    logger.info("SandboxManager injected into agents router")


def set_agent_dependencies(
    config_manager: ConfigManager, schema_loader: SchemaLoader
) -> None:
    """Inject config_manager and schema_loader for in-process agent execution."""
    global _config_manager, _schema_loader, _dispatcher
    _config_manager = config_manager
    _schema_loader = schema_loader
    _dispatcher = None  # Reset dispatcher so it picks up new dependencies
    logger.info("Agent dependencies (config_manager, schema_loader) injected")


def _build_artifact_manager_factory():
    """Per-tenant ArtifactManager factory for canary-aware artefact routing.

    Returns ``None`` when no telemetry manager is configured — the dispatcher
    then serves active artefacts to every request (canary disabled). When a
    manager exists, the factory hands the dispatcher a tenant-scoped
    ArtifactManager so ``resolve_artefact_for_request`` can read the canary
    state machine and split live traffic.
    """
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    tm = get_telemetry_manager()
    if tm is None:
        return None

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

    # Reuse one ArtifactManager per tenant so its 5s-TTL request cache (artefact
    # state + prompts, invalidated on promote/retire) spans dispatches. A fresh
    # manager per request discarded that cache, re-reading Phoenix every time.
    cache: dict[str, ArtifactManager] = {}
    lock = threading.Lock()

    def factory(tenant_id: str):
        with lock:
            am = cache.get(tenant_id)
            if am is None:
                am = ArtifactManager(tm.get_provider(tenant_id=tenant_id), tenant_id)
                cache[tenant_id] = am
            return am

    return factory


def _ensure_dispatcher() -> AgentDispatcher:
    """Lazily create the dispatcher once registry + deps are wired.

    A partial-startup call (lifespan hasn't finished wiring the registry
    or config_manager yet) surfaces as a 503, not the default 500 from a
    naked ``RuntimeError``.
    """
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    if _agent_registry is None or _config_manager is None or _schema_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Agent dependencies not configured; runtime initialising",
        )
    _dispatcher = AgentDispatcher(
        agent_registry=_agent_registry,
        config_manager=_config_manager,
        schema_loader=_schema_loader,
        sandbox_manager=_sandbox_manager,
        artifact_manager_factory=_build_artifact_manager_factory(),
    )
    return _dispatcher


def get_registry() -> AgentRegistry:
    """Get the injected registry or raise error"""
    if _agent_registry is None:
        raise RuntimeError(
            "AgentRegistry not initialized. Call set_agent_registry() first."
        )
    return _agent_registry


def get_dispatcher() -> AgentDispatcher:
    """Get the agent dispatcher (public accessor for A2A executor)."""
    return _ensure_dispatcher()


class AgentTask(BaseModel):
    """Task request for agent processing.

    Multi-turn support: Pass context_id and conversation_history for
    multi-turn conversations via REST (mirrors A2A contextId semantics).

    Enrichment fields (``enhanced_query``, ``entities``, ``relationships``,
    ``query_variants``, ``profiles``) are populated by the orchestrator
    from preprocessing agent outputs (QueryEnhancementAgent,
    EntityExtractionAgent, ProfileSelectionAgent) and forwarded to the
    execution agent so it can skip redundant preprocessing.
    """

    agent_name: str
    query: str
    context: Dict[str, Any] = {}
    top_k: int = 10
    context_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

    enhanced_query: Optional[str] = None
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    query_variants: List[Dict[str, str]] = []
    profiles: Optional[List[str]] = None
    # opt-in deep-synthesis switch propagated down to the
    # OrchestratorInput. Kept top-level so HTTP callers don't have to
    # nest it under context. Any value other than "deep" is ignored.
    synthesis_depth: Optional[str] = None
    session_id: Optional[str] = None


class AgentRegistrationData(BaseModel):
    """Agent self-registration data for Curated Registry pattern"""

    name: str
    url: str
    capabilities: List[str] = []
    health_endpoint: str = "/health"
    process_endpoint: str = "/tasks/send"
    timeout: int = 30


@router.post("/register", status_code=201)
async def register_agent(data: AgentRegistrationData) -> Dict[str, Any]:
    """
    Register an agent in the curated registry (A2A pattern).

    Agents call this endpoint during startup to self-register.
    """
    registry = get_registry()

    success = registry.register_agent_from_data(data.model_dump())

    if not success:
        raise HTTPException(
            status_code=400, detail=f"Failed to register agent '{data.name}'"
        )

    return {
        "status": "registered",
        "agent": data.name,
        "url": data.url,
        "capabilities": data.capabilities,
    }


@router.get("/")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents."""
    registry = get_registry()
    agents = registry.list_agents()

    return {
        "count": len(agents),
        "agents": agents,
    }


@router.get("/stats")
async def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics including health status"""
    registry = get_registry()
    return registry.get_registry_stats()


_annotation_queue: Optional["AnnotationQueue"] = None


def get_annotation_queue():
    """Lazily create or return the singleton AnnotationQueue."""
    global _annotation_queue
    if _annotation_queue is None:
        from cogniverse_agents.routing.annotation_queue import AnnotationQueue

        _annotation_queue = AnnotationQueue()
    return _annotation_queue


class AssignRequest(BaseModel):
    reviewer: str
    sla_hours: Optional[int] = None


class CompleteRequest(BaseModel):
    label: Optional[str] = None
    reasoning: str = ""
    annotator: str = "human"


class EnqueueBatchRequest(BaseModel):
    """Batch of annotation requests in ``AnnotationRequest.to_dict`` shape."""

    requests: List[Dict[str, Any]]


@router.get("/annotations/queue")
async def get_annotation_queue_status() -> Dict[str, Any]:
    """Get annotation queue statistics and pending items."""
    queue = get_annotation_queue()
    pending = queue.get_pending()
    assigned = queue.get_assigned()
    expired = queue.get_expired()
    return {
        "statistics": queue.statistics(),
        "pending": [r.to_dict() for r in pending[:50]],
        "assigned": [r.to_dict() for r in assigned[:50]],
        "expired": [r.to_dict() for r in expired[:50]],
    }


@router.post("/annotations/queue/{span_id}/assign")
async def assign_annotation(span_id: str, body: AssignRequest) -> Dict[str, Any]:
    """Assign a pending annotation to a reviewer."""
    queue = get_annotation_queue()
    try:
        request = queue.assign(
            span_id=span_id, reviewer=body.reviewer, sla_hours=body.sla_hours
        )
        return {"status": "assigned", "annotation": request.to_dict()}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotations/queue/enqueue")
async def enqueue_annotations(body: EnqueueBatchRequest) -> Dict[str, Any]:
    """Enqueue a batch of annotation requests (worklist ingress).

    Called by the scheduled annotation-identification cycle; also usable by
    external systems. Duplicated span_ids (already in the queue) are skipped.
    """
    from cogniverse_agents.routing.annotation_agent import AnnotationRequest

    queue = get_annotation_queue()
    try:
        requests = [AnnotationRequest.from_dict(item) for item in body.requests]
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid request payload: {e}")

    enqueued = queue.enqueue_batch(requests)
    return {
        "enqueued": enqueued,
        "skipped": len(requests) - enqueued,
        "queue_total": queue.statistics()["total"],
    }


@router.post("/annotations/queue/{span_id}/complete")
async def complete_annotation(span_id: str, body: CompleteRequest) -> Dict[str, Any]:
    """Mark an annotation as completed, persisting the label durably.

    The label is the whole value of the review — persistence happens BEFORE
    the in-memory completion, so a telemetry outage leaves the item open for
    retry (502) instead of silently discarding the reviewer's work. Items
    enqueued without a tenant_id can't be persisted; they complete in-memory
    with ``persisted: false``.
    """
    from cogniverse_agents.routing.annotation_agent import AnnotationStatus

    queue = get_annotation_queue()
    request = queue.get(span_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    # Guard status BEFORE persisting so a re-complete of a finished item can't
    # write a duplicate annotation.
    if request.status not in (AnnotationStatus.PENDING, AnnotationStatus.ASSIGNED):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot complete span {span_id}: status is {request.status.value}"
            ),
        )

    persisted = False
    if body.label is not None:
        import cogniverse_agents.routing.annotation_storage as annotation_storage_mod
        from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel

        try:
            label_enum = AnnotationLabel(body.label)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown annotation label '{body.label}'; expected one of "
                    f"{sorted(m.value for m in AnnotationLabel)}"
                ),
            )
        if request.tenant_id:
            storage = annotation_storage_mod.AnnotationStorage(
                tenant_id=request.tenant_id, agent_type=request.agent_type
            )
            try:
                await storage.store_human_annotation(
                    span_id=span_id,
                    label=label_enum,
                    reasoning=body.reasoning,
                    annotator_id=body.annotator,
                )
                persisted = True
            except Exception as e:
                logger.error("Annotation persist failed for span %s: %r", span_id, e)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Annotation could not be persisted to the telemetry "
                        "backend; the item remains open for retry."
                    ),
                )
        else:
            logger.warning(
                "Annotation for span %s completed without tenant_id — label "
                "kept in-memory only",
                span_id,
            )

    try:
        request = queue.complete(span_id=span_id, label=body.label)
        return {
            "status": "completed",
            "persisted": persisted,
            "annotation": request.to_dict(),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/by-capability/{capability}")
async def find_agents_by_capability(capability: str) -> Dict[str, Any]:
    """
    Find agents by capability (A2A Curated Registry pattern).

    Enables capability-based agent discovery.
    """
    registry = get_registry()
    agents = registry.find_agents_by_capability(capability)

    return {
        "capability": capability,
        "count": len(agents),
        "agents": [
            {
                "name": agent.name,
                "url": agent.url,
                "capabilities": agent.capabilities,
                "health_status": agent.health_status,
            }
            for agent in agents
        ],
    }


@router.get("/{agent_name}")
async def get_agent_info(agent_name: str) -> Dict[str, Any]:
    """Get information about a specific agent."""
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "name": agent.name,
        "url": agent.url,
        "capabilities": agent.capabilities,
        "health_status": agent.health_status,
        "health_endpoint": agent.health_endpoint,
        "process_endpoint": agent.process_endpoint,
    }


@router.delete("/{agent_name}", status_code=200)
async def unregister_agent(agent_name: str) -> Dict[str, Any]:
    """Unregister an agent from the registry"""
    registry = get_registry()

    success = registry.unregister_agent(agent_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "status": "unregistered",
        "agent": agent_name,
    }


@router.get("/{agent_name}/card")
async def get_agent_card(agent_name: str) -> Dict[str, Any]:
    """Get agent card (A2A protocol) for a specific agent."""
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "name": agent.name,
        "url": agent.url,
        "version": "1.0",
        "capabilities": agent.capabilities,
        "endpoints": {
            "health": agent.health_endpoint,
            "process": agent.process_endpoint,
            "info": f"/agents/{agent_name}",
        },
    }


@router.post("/{agent_name}/process")
async def process_agent_task(agent_name: str, task: AgentTask) -> Dict[str, Any]:
    """
    Process a task with a specific agent.

    In unified runtime mode, this endpoint executes agent logic in-process
    by routing to the appropriate service (search, LLM, etc.).
    """
    dispatcher = _ensure_dispatcher()

    # Merge multi-turn + enrichment fields into context dict for dispatcher.
    dispatch_context = dict(task.context)
    if task.context_id is not None:
        dispatch_context["context_id"] = task.context_id
    if task.conversation_history is not None:
        dispatch_context["conversation_history"] = task.conversation_history
    if task.enhanced_query is not None:
        dispatch_context["enhanced_query"] = task.enhanced_query
    if task.entities:
        dispatch_context["entities"] = task.entities
    if task.relationships:
        dispatch_context["relationships"] = task.relationships
    if task.query_variants:
        dispatch_context["query_variants"] = task.query_variants
    if task.profiles is not None:
        dispatch_context["profiles"] = task.profiles
    if task.synthesis_depth is not None:
        dispatch_context["synthesis_depth"] = task.synthesis_depth
    if task.session_id is not None:
        dispatch_context["session_id"] = task.session_id

    # Stable per-request seed for canary/variant bucketing — session-sticky
    # when a session/context id is present, else a fresh id so one-shot calls
    # are still split. The dispatcher reads ``context["request_id"]``.
    if not dispatch_context.get("request_id"):
        dispatch_context["request_id"] = (
            task.session_id or task.context_id or uuid.uuid4().hex
        )

    try:
        return await dispatcher.dispatch(
            agent_name=agent_name,
            query=task.query,
            context=dispatch_context,
            top_k=task.top_k,
        )
    except VespaSearchDegraded as e:
        # Vespa soft-timeout (HTTP 200 + root.errors): the backend is up but
        # degraded — 503 tells the caller to retry, instead of an opaque 500.
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(status_code=404, detail=detail)
        elif "no supported execution path" in detail:
            raise HTTPException(status_code=501, detail=detail)
        raise HTTPException(status_code=400, detail=detail)


# --------------------------------------------------------------------------- #
# Inbound messaging — per-session steering into running agents.               #
# Mirrors the design in libs/runtime/cogniverse_runtime/messaging.py.         #
# Multi-pod + durability via libs/runtime/cogniverse_runtime/messaging_redis. #
# --------------------------------------------------------------------------- #


_ALLOWED_INBOUND_ROLES = {"user", "system", "agent"}


class InboundMessageRequest(BaseModel):
    """Request body for ``POST /agents/{name}/message``.

    Pydantic-validated at intake; mismatches surface as 422 (role,
    types) or 400 (deadline already past). Successful intake returns
    202 with ``message_id`` + ``queued_at`` so the caller can
    correlate the enqueue with the agent's downstream consumption.

    ``tenant_id`` scopes the message — the route checks it matches
    the session's registered tenant and returns 404 (not 403) on
    mismatch so a cross-tenant probe can't enumerate other tenants'
    session ids. Required, never optional.
    """

    session_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    role: str
    content: str = ""
    tags: List[str] = Field(default_factory=list)
    deadline_ms: Optional[int] = None

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in _ALLOWED_INBOUND_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_ALLOWED_INBOUND_ROLES)}, got {v!r}"
            )
        return v


@router.post("/{agent_name}/message", status_code=202)
async def post_agent_message(
    agent_name: str, request: InboundMessageRequest
) -> Dict[str, Any]:
    """Enqueue an inbound message for a running agent session.

    The agent's running ``process()`` registered the session via
    ``InboundQueueRegistry.get_or_create_queue`` at loop entry. Until
    its ``finally`` block calls ``close_queue``, this route delivers
    messages to the same queue. Tags drive agent behaviour:

      * ``"stop"`` — cooperative cancellation, agent drains and exits
        with ``exit_reason="user_stop"`` returning partial state.
      * ``"constraint"`` / ``"interrupt"`` — content is prepended to
        the next iteration's ``missing_aspects`` and feeds the
        reformulator.

    ``agent_name`` is reserved for future per-agent routing (so an
    operator can address a specific running agent by name); today
    the registry is keyed only by session_id and the param is
    accepted for URL symmetry with ``/process``.
    """
    _ = agent_name  # reserved for future per-agent routing

    if request.deadline_ms is not None and request.deadline_ms < int(
        time.time() * 1000
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"deadline_ms {request.deadline_ms} is already in the past; "
                "reject at intake rather than buffer a message no consumer "
                "would ever drain"
            ),
        )

    registry = await _resolve_inbound_registry()
    queue = await registry.get_queue(request.session_id)
    if queue is None:
        raise HTTPException(
            status_code=404,
            detail=f"session {request.session_id!r} not active",
        )
    # Cross-tenant guard: deliberately return 404 (not 403) so a probe
    # cannot distinguish "session exists under different tenant" from
    # "no such session" — denies tenant enumeration via the message
    # route. Caller-supplied tenant_id MUST match the session's
    # registered tenant.
    if queue.tenant_id != request.tenant_id:
        raise HTTPException(
            status_code=404,
            detail=f"session {request.session_id!r} not active",
        )

    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    queued_at = datetime.now(timezone.utc).isoformat()
    msg = InboundMessage(
        session_id=request.session_id,
        role=request.role,
        content=request.content,
        tags=tuple(request.tags),
        created_at=queued_at,
        deadline_ms=request.deadline_ms,
    )
    try:
        await queue.enqueue(msg)
    except QueueClosedError as exc:
        # Race: queue closed between get_queue and enqueue. Surface as
        # 404 (consistent with "not active") rather than a 5xx — the
        # caller's mental model is "session ended."
        raise HTTPException(
            status_code=404,
            detail=f"session {request.session_id!r} not active",
        ) from exc
    return {"message_id": message_id, "queued_at": queued_at}


@router.get("/{agent_name}/sessions/{session_id}")
async def get_agent_session(
    agent_name: str, session_id: str, tenant_id: str
) -> Dict[str, Any]:
    """Return 200 + session metadata when active, 404 when not.

    Used by clients (and the E2E test harness) to poll whether the
    agent's ``process()`` has reached its loop body. ``tenant_id`` is
    required (query string) and scoped the same way the message
    route is — cross-tenant peek returns 404, never reveals the
    session's actual tenant.
    """
    _ = agent_name
    registry = await _resolve_inbound_registry()
    queue = await registry.get_queue(session_id)
    if queue is None or queue.tenant_id != tenant_id:
        raise HTTPException(
            status_code=404,
            detail=f"session {session_id!r} not active",
        )
    return {
        "session_id": queue.session_id,
        "tenant_id": queue.tenant_id,
        "created_at": queue.created_at.isoformat(),
    }
