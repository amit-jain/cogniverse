"""Agent endpoints - unified interface for all agent operations."""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from cogniverse_agents.routing.annotation_queue import (
    AnnotationCompletionInProgressError,
    AnnotationQueue,
    AnnotationQueueFullError,
    AnnotationQueueUnavailableError,
)
from cogniverse_agents.search.vespa_query import VespaSearchDegraded
from cogniverse_core.query.encoders import (
    EncoderNotConfiguredError,
    EncoderUnavailableError,
)
from cogniverse_core.registries.agent_registry import (
    AgentRegistry,
    AgentRegistryUnavailableError,
    endpoint_from_data,
)
from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.telemetry.context import request_trace_context
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.http_errors import (
    failure_response,
    query_encoder_failure,
    record_failure,
)
from cogniverse_runtime.llm_dependency import llm_dependency_failure
from cogniverse_runtime.messaging import (
    InboundMessage,
    QueueClosedError,
    get_inbound_queue_registry,
)
from cogniverse_runtime.session_state import (
    ConversationLedger,
    SessionStateUnavailable,
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

# Bound on cached per-tenant ArtifactManagers. Least-recently-dispatched
# tenants rebuild on their next request; tenant delete evicts eagerly via
# the registered-cache hook.
ARTIFACT_MANAGER_CACHE_CAPACITY = 64

# Module-level dependencies (injected from main.py)
_agent_registry: Optional[AgentRegistry] = None
_config_manager: Optional[ConfigManager] = None
_schema_loader: Optional[SchemaLoader] = None
_dispatcher: Optional[AgentDispatcher] = None
_sandbox_manager = None
_conversation_ledger: Optional[ConversationLedger] = None


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


def set_conversation_ledger(ledger: Optional[ConversationLedger]) -> None:
    """Inject the shared conversation ledger into the dispatcher, built or not."""
    global _conversation_ledger
    _conversation_ledger = ledger
    if _dispatcher is not None:
        _dispatcher.set_conversation_ledger(ledger)
    logger.info(
        "Conversation ledger %s agents router",
        "injected into" if ledger is not None else "removed from",
    )


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
    from cogniverse_foundation.caching import TenantLRUCache, register_tenant_cache

    # Reuse one ArtifactManager per tenant so its 5s-TTL request cache (artefact
    # state + prompts, invalidated on promote/retire) spans dispatches. A fresh
    # manager per request discarded that cache, re-reading Phoenix every time.
    # LRU-bounded; tenant delete evicts eagerly via the registered-cache hook.
    cache: TenantLRUCache[ArtifactManager] = register_tenant_cache(
        TenantLRUCache(capacity=ARTIFACT_MANAGER_CACHE_CAPACITY)
    )

    def factory(tenant_id: str):
        return cache.get_or_set(
            tenant_id,
            lambda: ArtifactManager(tm.get_provider(tenant_id=tenant_id), tenant_id),
        )

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
        conversation_ledger=_conversation_ledger,
    )
    return _dispatcher


def get_registry() -> AgentRegistry:
    """Get the injected registry or raise error"""
    if _agent_registry is None:
        raise RuntimeError(
            "AgentRegistry not initialized. Call set_agent_registry() first."
        )
    return _agent_registry


async def _current_registry() -> AgentRegistry:
    """The registry with every process's registrations applied; 503 when the
    shared store cannot be read."""
    registry = get_registry()
    try:
        await registry.refresh()
    except AgentRegistryUnavailableError as exc:
        raise _registry_unavailable(exc) from exc
    return registry


def _registry_unavailable(
    exc: AgentRegistryUnavailableError, **fields: Any
) -> HTTPException:
    """503 for a shared agent registry operation that did not complete."""
    return failure_response(
        503,
        "agent_registry_unavailable",
        "The shared agent registry did not answer; retry.",
        exc,
        **fields,
    )


def get_dispatcher() -> AgentDispatcher:
    """Get the agent dispatcher (public accessor for A2A executor)."""
    return _ensure_dispatcher()


async def drain_conversation_saves() -> bool:
    """Land conversation turns still persisting in the background.

    Called from the runtime's shutdown so a turn answered moments before
    SIGTERM is not lost. True when nothing was pending or everything landed;
    a process that never built a dispatcher has nothing to drain.
    """
    if _dispatcher is None:
        return True
    return await _dispatcher.drain_conversation_saves()


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
    # RLM promotion stamped by the orchestrator (or an explicit caller
    # opt-in); threaded into the dispatch context and consumed by the
    # promotable agents' typed inputs as RLMOptions.
    rlm: Optional[Dict[str, Any]] = None


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

    Agents call this endpoint during startup to self-register. The
    registration is served by every runtime process from the next request on.
    """
    registry = get_registry()

    try:
        await registry.add_registration(endpoint_from_data(data.model_dump()))
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Failed to register agent '{data.name}'"
        )
    except AgentRegistryUnavailableError as exc:
        raise _registry_unavailable(exc, agent=data.name) from exc

    return {
        "status": "registered",
        "agent": data.name,
        "url": data.url,
        "capabilities": data.capabilities,
    }


@router.get("/")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents."""
    registry = await _current_registry()
    agents = registry.list_agents()

    return {
        "count": len(agents),
        "agents": agents,
    }


@router.get("/stats")
async def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics including health status"""
    registry = await _current_registry()
    return registry.get_registry_stats()


_annotation_queue: Optional[AnnotationQueue] = None


def set_annotation_queue(queue: Optional[AnnotationQueue]) -> None:
    """Inject the annotation queue every runtime process shares."""
    global _annotation_queue
    _annotation_queue = queue


def get_annotation_queue() -> AnnotationQueue:
    """The injected annotation queue; 503 until startup has wired it."""
    if _annotation_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Annotation queue not configured; runtime initialising",
        )
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


def _queue_unavailable(
    exc: AnnotationQueueUnavailableError, **fields: Any
) -> HTTPException:
    """503 for an annotation queue operation that did not complete."""
    return failure_response(
        503,
        "annotation_queue_unavailable",
        "The annotation queue did not answer; retry.",
        exc,
        **fields,
    )


def _review_label_values() -> List[str]:
    # Imported here and called off the loop: the module loads litellm.
    from cogniverse_agents.routing.llm_auto_annotator import REVIEW_LABELS

    return [label.value for label in REVIEW_LABELS]


@router.get("/annotations/labels")
async def list_review_labels() -> Dict[str, List[str]]:
    """The labels a reviewer can complete an annotation with."""
    return {"labels": await asyncio.to_thread(_review_label_values)}


@router.get("/annotations/queue")
async def get_annotation_queue_status() -> Dict[str, Any]:
    """Get annotation queue statistics and the first 50 requests of each list.

    Assigned requests past their SLA deadline turn expired on this read.
    """
    queue = get_annotation_queue()
    try:
        snapshot = await queue.snapshot(limit=50)
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc) from exc
    return {
        "statistics": snapshot.statistics,
        "pending": [r.to_dict() for r in snapshot.pending],
        "assigned": [r.to_dict() for r in snapshot.assigned],
        "expired": [r.to_dict() for r in snapshot.expired],
    }


@router.get("/annotations/queue/{span_id}")
async def get_annotation_request(span_id: str) -> Dict[str, Any]:
    """Return one annotation request by span id."""
    try:
        request = await get_annotation_queue().get(span_id)
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc, span_id=span_id) from exc
    if request is None:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    return request.to_dict()


@router.post("/annotations/queue/{span_id}/assign")
async def assign_annotation(span_id: str, body: AssignRequest) -> Dict[str, Any]:
    """Assign a pending annotation to a reviewer."""
    queue = get_annotation_queue()
    try:
        request = await queue.assign(
            span_id=span_id, reviewer=body.reviewer, sla_hours=body.sla_hours
        )
        return {"status": "assigned", "annotation": request.to_dict()}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc, span_id=span_id) from exc


@router.post("/annotations/queue/enqueue")
async def enqueue_annotations(body: EnqueueBatchRequest) -> Dict[str, Any]:
    """Enqueue a batch of annotation requests (worklist ingress).

    Called by the scheduled annotation-identification cycle; also usable by
    external systems. Duplicated span_ids (already in the queue) are skipped.
    A batch that would take the open requests past the queue's limit is
    refused whole with 429.
    """
    from cogniverse_agents.routing.annotation_agent import AnnotationRequest

    queue = get_annotation_queue()
    try:
        requests = [AnnotationRequest.from_dict(item) for item in body.requests]
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid request payload: {e}")

    try:
        outcome = await queue.enqueue_batch(requests)
    except AnnotationQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc) from exc
    return {
        "enqueued": outcome.enqueued,
        "skipped": len(requests) - outcome.enqueued,
        "queue_total": outcome.total,
    }


@router.post("/annotations/queue/{span_id}/complete")
async def complete_annotation(span_id: str, body: CompleteRequest) -> Dict[str, Any]:
    """Mark an annotation as completed, persisting the label durably.

    The request is claimed first, so of concurrent completions on any
    processes exactly one persists a label (the others get 409). The label is
    the whole value of the review — persistence happens BEFORE the
    completion, so a telemetry outage releases the claim and leaves the item
    open for retry (502) instead of silently discarding the reviewer's work.
    Items enqueued without a tenant_id can't be persisted; they complete with
    ``persisted: false``.
    """
    queue = get_annotation_queue()
    try:
        claim = await queue.begin_completion(span_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Span {span_id} not in queue")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AnnotationCompletionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc, span_id=span_id) from exc

    try:
        persisted = await _persist_label(claim.request, span_id, body)
    except HTTPException:
        await _release_claim(queue, claim)
        raise

    try:
        request = await queue.finish_completion(claim, label=body.label)
    except AnnotationCompletionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AnnotationQueueUnavailableError as exc:
        raise _queue_unavailable(exc, span_id=span_id) from exc
    return {
        "status": "completed",
        "persisted": persisted,
        "annotation": request.to_dict(),
    }


def _label_persistence():
    """The telemetry label writer and the label types.

    Their first import loads litellm, which fetches its model cost map over
    the network (five-second timeout), so callers import them off the event
    loop.
    """
    import cogniverse_agents.routing.annotation_storage as annotation_storage_mod
    from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel

    return annotation_storage_mod, AnnotationLabel


async def _persist_label(request, span_id: str, body: CompleteRequest) -> bool:
    """Write the reviewer's label to telemetry; True when it was written."""
    if body.label is None:
        return False
    annotation_storage_mod, AnnotationLabel = await asyncio.to_thread(
        _label_persistence
    )

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
    if not request.tenant_id:
        logger.warning(
            "Annotation for span %s completed without tenant_id — label kept "
            "in the queue only",
            span_id,
        )
        return False
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
    except Exception as e:
        logger.error("Annotation persist failed for span %s: %r", span_id, e)
        raise HTTPException(
            status_code=502,
            detail=(
                "Annotation could not be persisted to the telemetry "
                "backend; the item remains open for retry."
            ),
        )
    return True


async def _release_claim(queue: AnnotationQueue, claim) -> None:
    try:
        await queue.abandon_completion(claim)
    except AnnotationQueueUnavailableError as exc:
        logger.error(
            "Could not release the completion claim on span %s (%s); it lapses "
            "on its own",
            claim.span_id,
            exc,
        )


@router.get("/by-capability/{capability}")
async def find_agents_by_capability(capability: str) -> Dict[str, Any]:
    """
    Find agents by capability (A2A Curated Registry pattern).

    Enables capability-based agent discovery.
    """
    registry = await _current_registry()
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
    registry = await _current_registry()

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
    """Unregister an agent from every runtime process's registry."""
    registry = get_registry()

    try:
        success = await registry.remove_registration(agent_name)
    except AgentRegistryUnavailableError as exc:
        raise _registry_unavailable(exc, agent=agent_name) from exc

    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "status": "unregistered",
        "agent": agent_name,
    }


@router.get("/{agent_name}/card")
async def get_agent_card(agent_name: str) -> Dict[str, Any]:
    """Get agent card (A2A protocol) for a specific agent."""
    registry = await _current_registry()

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
async def process_agent_task(
    agent_name: str, task: AgentTask, request: Request
) -> Dict[str, Any]:
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
    if task.rlm is not None:
        dispatch_context["rlm"] = task.rlm

    # Stable per-request seed for canary/variant bucketing — session-sticky
    # when a session/context id is present, else a fresh id so one-shot calls
    # are still split. The dispatcher reads ``context["request_id"]``.
    if not dispatch_context.get("request_id"):
        dispatch_context["request_id"] = (
            task.session_id or task.context_id or uuid.uuid4().hex
        )

    try:
        with request_trace_context(request.headers):
            return await dispatcher.dispatch(
                agent_name=agent_name,
                query=task.query,
                context=dispatch_context,
                top_k=task.top_k,
            )
    except (EncoderNotConfiguredError, EncoderUnavailableError) as e:
        # Checked before ValueError: a missing encoder setting is not the
        # caller's bad input, and its text names the sidecar URL.
        record_failure(e, "query_encoder")
        status, body, headers = query_encoder_failure(
            e,
            profile=None,
            strategy=None,
            agent=agent_name,
            request_id=dispatch_context["request_id"],
        )
        raise HTTPException(status_code=status, detail=body, headers=headers)
    except VespaSearchDegraded as e:
        # Vespa soft-timeout (HTTP 200 + root.errors): the backend is up but
        # degraded — 503 tells the caller to retry, instead of an opaque 500.
        raise failure_response(
            503,
            "search_degraded",
            f"Agent '{agent_name}' could not complete: the search backend "
            "answered with degraded coverage; retry.",
            e,
            agent=agent_name,
            request_id=dispatch_context["request_id"],
        )
    except AgentRegistryUnavailableError as e:
        raise _registry_unavailable(
            e, agent=agent_name, request_id=dispatch_context["request_id"]
        )
    except SessionStateUnavailable as e:
        # The shared ledger that orders this context's turns did not answer;
        # the turn is refused rather than answered and never stored.
        raise failure_response(
            503,
            "session_state_unavailable",
            f"Agent '{agent_name}' could not complete: the session state store "
            "did not answer; retry.",
            e,
            agent=agent_name,
            context_id=task.context_id,
            request_id=dispatch_context["request_id"],
        )
    except InferenceServiceUnavailableError as e:
        # The sidecar backing this capability isn't provisioned in this
        # deployment, or is unreachable. 503 names the service to configure;
        # the exception text, which can carry the sidecar URL, stays in the log.
        raise failure_response(
            503,
            "inference_service_unavailable",
            f"Agent '{agent_name}' could not complete: inference service "
            f"'{e.service}' is unavailable.",
            e,
            agent=agent_name,
            service=e.service,
            module=e.module,
            request_id=dispatch_context["request_id"],
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(status_code=404, detail=detail)
        elif "no supported execution path" in detail:
            raise failure_response(
                501,
                "no_execution_path",
                f"Agent '{agent_name}' has no supported execution path in this "
                "runtime.",
                e,
                agent=agent_name,
                request_id=dispatch_context["request_id"],
            )
        raise HTTPException(status_code=400, detail=detail)
    except Exception as e:
        request_id = dispatch_context["request_id"]
        llm_failure = llm_dependency_failure(e)
        if llm_failure is not None:
            # The chat LLM failed: a 503/502 naming it, never an opaque 500.
            logger.warning(
                "Agent '%s' dispatch failed on the chat LLM (request_id=%s): %s",
                agent_name,
                request_id,
                e,
            )
            raise HTTPException(
                status_code=llm_failure.http_status,
                detail=llm_failure.body(agent=agent_name, request_id=request_id),
                headers=llm_failure.headers(),
            )
        # Anything else is a 500 whose JSON body names the agent, the error
        # type and the request id; the traceback stays in the log, since the
        # exception text can carry backend URLs with credentials.
        logger.exception(
            "Agent '%s' dispatch failed (request_id=%s)", agent_name, request_id
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Agent '{agent_name}' failed with {type(e).__name__} "
                f"(request_id={request_id}). See runtime logs for detail."
            ),
        )


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
