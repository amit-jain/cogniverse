"""Tenant self-service endpoints.

Allows tenants to customize their agent experience:
- Instructions: per-tenant system prompt stored in ConfigStore
- Memory management: browse and delete Mem0 memories
- Scheduled jobs: create/list/delete Argo CronWorkflow-backed jobs
"""

import asyncio
import logging
import re
import uuid
import weakref
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from cogniverse_core.common.tenant_utils import (
    canonical_tenant_id,
    sanitize_k8s_label_value,
)
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_foundation.common.argo_client import build_argo_async_client
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.config_loader import get_workflow_settings
from cogniverse_runtime.http_errors import failure_response, upstream_rejection
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER

logger = logging.getLogger(__name__)

# One client per running event loop for the Argo submit/status calls.
# Building a client per request paid ~10ms of TLS-context setup plus a fresh
# TCP handshake on every call. Keyed by the LOOP OBJECT in a weak map — an
# ``id(loop)`` key collides when a torn-down loop's id is reused by a new
# loop (CPython reuses freed addresses), handing the new loop a client bound
# to the dead one; the weak map reaps entries with their loop instead. No
# lock: the check-construct-set sequence has no await, so it is atomic on
# its loop.
_ARGO_CLIENT_TIMEOUT = httpx.Timeout(10.0)
# Bound on finishing a compensating delete after the request was cancelled,
# above the Argo client's own timeout so a slow delete still completes.
_CLEANUP_COMPLETION_TIMEOUT_S = 15.0
_argo_clients: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


async def _shared_argo_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _argo_clients.get(loop)
    if client is None or client.is_closed:
        client = build_argo_async_client(_ARGO_CLIENT_TIMEOUT)
        _argo_clients[loop] = client
    return client


router = APIRouter()

# Module-level config manager — set by main.py at startup
_config_manager: Optional[ConfigManager] = None


def set_config_manager(config_manager: ConfigManager) -> None:
    """Inject ConfigManager (called from main.py lifespan)."""
    global _config_manager
    _config_manager = config_manager


def _require_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialised")
    return _config_manager


# Path k8s injects the runtime pod's ServiceAccount token at. argo-server
# in ``--auth-mode=server`` validates the bearer token via the TokenReview
# API, so the runtime forwards its own SA token; no separate Argo client
# secret is needed.
_K8S_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def _argo_auth_headers() -> Dict[str, str]:
    """Bearer-auth headers for Argo API calls.

    Reads the runtime pod's ServiceAccount token at request time so a token
    rotation by kubelet doesn't strand a long-lived header. Returns an empty
    dict outside the cluster (no token file → tests / local dev).
    """
    try:
        with open(_K8S_SA_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        return {}
    return {"Authorization": f"Bearer {token}"} if token else {}


class InstructionsRequest(BaseModel):
    text: str


class InstructionsResponse(BaseModel):
    text: str
    updated_at: str


class MemoryItem(BaseModel):
    id: str
    memory: str
    type: str
    owned: bool
    category: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    score: Optional[float] = Field(
        None, description="Similarity to the search query; null when listing"
    )


class MemoryListResponse(BaseModel):
    memories: List[MemoryItem]
    count: int


class StatusResponse(BaseModel):
    status: str
    agent: Optional[str] = None


class JobCreateRequest(BaseModel):
    name: str
    schedule: str
    query: str
    post_actions: List[str] = []


class JobResponse(BaseModel):
    job_id: str
    name: str
    schedule: str
    query: str
    post_actions: List[str]
    status: str
    created_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobResponse]


_INSTRUCTIONS_SERVICE = "tenant_instructions"
_INSTRUCTIONS_KEY = "system_prompt"


@router.put("/{tenant_id}/instructions", response_model=InstructionsResponse)
async def set_instructions(tenant_id: str, body: InstructionsRequest):
    """Store tenant-level agent instructions (SOUL.md equivalent)."""
    cm = _require_config_manager()
    now = datetime.now(timezone.utc).isoformat()
    value = {"text": body.text, "updated_at": now}
    # The config store is a synchronous pyvespa call (60s timeout); offload it
    # so a slow Vespa doesn't stall the whole pod's event loop.
    await asyncio.to_thread(
        cm.set_config_value,
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
        config_value=value,
    )
    logger.info("Updated instructions for tenant %s", tenant_id)
    return InstructionsResponse(text=body.text, updated_at=now)


@router.get("/{tenant_id}/instructions", response_model=InstructionsResponse)
async def get_instructions(tenant_id: str):
    """Retrieve the stored tenant instructions."""
    cm = _require_config_manager()
    # Read through the manager so the key gets the same tenant
    # canonicalization as set_instructions' write — a raw store read looks
    # up a key the write never produced and 404s on every stored value.
    value = await asyncio.to_thread(
        cm.get_config_value,
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
    )
    if not value:
        raise HTTPException(
            status_code=404, detail="No instructions found for this tenant"
        )
    return InstructionsResponse(
        text=value.get("text", ""),
        updated_at=value.get("updated_at", ""),
    )


@router.delete("/{tenant_id}/instructions")
async def delete_instructions(tenant_id: str):
    """Clear the stored tenant instructions."""
    cm = _require_config_manager()
    await asyncio.to_thread(
        cm.set_config_value,
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
        config_value={"text": "", "updated_at": datetime.now(timezone.utc).isoformat()},
    )
    logger.info("Cleared instructions for tenant %s", tenant_id)
    return {"status": "cleared"}


def _get_memory_manager(tenant_id: str):
    """Return an initialised Mem0MemoryManager for the given tenant.

    Lazily initializes Mem0 from the system config if the singleton
    exists but was never initialized (common on k3d where memory isn't
    wired at startup).
    """
    from cogniverse_runtime.memory_init import lazy_init_memory

    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        try:
            lazy_init_memory(mgr, tenant_id, _require_config_manager())
        except Exception as exc:
            raise failure_response(
                503,
                "memory_unavailable",
                f"Memory backend not initialised for tenant {tenant_id}.",
                exc,
                tenant_id=tenant_id,
            ) from exc
    if not mgr.memory:
        raise HTTPException(
            status_code=503,
            detail="Memory backend not initialised for this tenant",
        )
    return mgr


_USER_MEMORY_AGENT = "_user_memories"

_TYPE_TO_NAMESPACE: Dict[str, str] = {
    "preference": "_user_memories",
    "strategy": "_strategy_store",
}

_SYSTEM_NAMESPACES = {"_strategy_store"}

_ALL_NAMESPACES = ["_user_memories", "_strategy_store"]


def _namespace_to_type(agent_name: str) -> str:
    """Map internal agent_name to user-facing type."""
    for type_name, ns in _TYPE_TO_NAMESPACE.items():
        if ns == agent_name:
            return type_name
    return "interaction"


def _is_owned(agent_name: str) -> bool:
    """User-owned memories are in _user_memories; everything else is system."""
    return agent_name == _USER_MEMORY_AGENT


def _is_writable(agent_name: str) -> bool:
    """User memories and agents' own namespaces; ``_``-prefixed partitions
    other than the user's belong to the runtime."""
    return agent_name == _USER_MEMORY_AGENT or not agent_name.startswith("_")


def _memory_read_failed(
    exc: Exception, tenant_id: str, namespaces: List[str]
) -> HTTPException:
    """503 for a memory read the store did not answer."""
    return failure_response(
        503,
        "memory_unavailable",
        f"Could not read the memories of {', '.join(namespaces)} for tenant "
        f"{tenant_id}.",
        exc,
        tenant_id=tenant_id,
    )


def _require_writable(agent_name: str) -> None:
    if not _is_writable(agent_name):
        raise HTTPException(
            status_code=403,
            detail=f"{agent_name} is a system memory namespace; the runtime manages it.",
        )


class MemoryCreateRequest(BaseModel):
    text: str
    category: Optional[str] = None
    # Optional admin/import fields. ``kind`` flows into the
    # KnowledgeRegistry-keyed retention contract — the daily-cleanup
    # workflow respects per-kind TTLs, so importers and e2e tests need
    # to set the right kind. ``metadata`` is a free-form dict merged on
    # top (e.g. ``created_at`` for backdated historical imports). Both
    # are optional so the original {text, category} caller is unaffected.
    kind: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # The namespace the memory lands in: the user's memories, or an agent's
    # own (its mem0 agent_id). System namespaces answer 403.
    agent_name: str = _USER_MEMORY_AGENT


class MemoryStats(BaseModel):
    agent_name: str
    user_id: str = Field(..., description="The tenant partition the count reads")
    total: int
    archived: int
    writable: bool


class MemoryHealth(BaseModel):
    tenant_id: str
    agent_name: str
    healthy: bool
    problem: Optional[str] = Field(
        None, description="Why the memory store is not healthy"
    )


@router.post("/{tenant_id}/memories")
async def create_memory(tenant_id: str, request: MemoryCreateRequest):
    """Save a memory with optional category, kind, metadata."""
    tenant_id = canonical_tenant_id(tenant_id)
    _require_writable(request.agent_name)
    mgr = await asyncio.to_thread(_get_memory_manager, tenant_id)
    metadata: Dict[str, Any] = {}
    if request.category:
        metadata["category"] = request.category
    if request.kind:
        metadata["kind"] = request.kind
    if request.metadata:
        # Caller-supplied metadata wins over the derived fields above.
        metadata.update(request.metadata)

    # Blocking mem0 write (embedder HTTP + Vespa) — off the event loop, like
    # the list_memories sibling below.
    memory_id = await asyncio.to_thread(
        mgr.add_memory,
        content=request.text,
        tenant_id=tenant_id,
        agent_name=request.agent_name,
        metadata=metadata,
        infer=False,
    )
    return {
        "status": "saved",
        "id": str(memory_id),
        "type": _namespace_to_type(request.agent_name),
        "agent_name": request.agent_name,
        "category": request.category,
        "kind": request.kind,
    }


def _entry_to_item(entry: dict, agent_name: str) -> Optional[MemoryItem]:
    """Convert a raw Mem0 result dict to a MemoryItem with type/owned."""
    if not isinstance(entry, dict):
        return None
    # Mem0 may emit ``metadata: None`` (or omit it) when the row has no
    # caller-set metadata. ``entry.get("metadata", {})`` returns None in
    # the explicit-None case and crashes the next ``.get("category")``.
    meta = entry.get("metadata") or {}
    return MemoryItem(
        id=str(entry.get("id", "")),
        memory=entry.get("memory", entry.get("text", "")),
        type=_namespace_to_type(agent_name),
        owned=_is_owned(agent_name),
        category=meta.get("category"),
        metadata=meta,
        created_at=str(entry.get("created_at", "")) or None,
        updated_at=str(entry.get("updated_at") or "") or None,
        score=entry.get("score"),
    )


@router.get("/{tenant_id}/memories", response_model=MemoryListResponse)
async def list_memories(
    tenant_id: str,
    q: Optional[str] = Query(default=None, description="Search query"),
    type: Optional[str] = Query(
        default=None, description="Filter by type: preference, strategy"
    ),
    agent_name: Optional[str] = Query(
        default=None,
        description="Scope to one agent's memories (its mem0 agent_id)",
    ),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    limit: int = Query(default=20, ge=1, le=200, description="Max results"),
):
    """List or search tenant memories across all types.

    Without ``q``, lists all memories.  With ``q``, performs semantic search.
    Use ``type`` to restrict to a single memory type and ``category`` to
    filter user-created memories by their category tag.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    mgr = await asyncio.to_thread(_get_memory_manager, tenant_id)

    if agent_name:
        # Agents store their learned memories under agent_id=agent_name, so
        # scope directly to that store rather than the default namespaces.
        namespaces = [agent_name]
    elif type:
        ns = _TYPE_TO_NAMESPACE.get(type)
        if ns is None:
            raise HTTPException(status_code=400, detail=f"Unknown memory type: {type}")
        namespaces = [ns]
    else:
        namespaces = list(_ALL_NAMESPACES)

    items: List[MemoryItem] = []
    for ns in namespaces:
        try:
            if q:
                raw = await asyncio.to_thread(
                    mgr.search_memory,
                    query=q,
                    tenant_id=tenant_id,
                    agent_name=ns,
                    top_k=limit,
                )
            else:
                # A category filter is applied in Python below, so the whole
                # partition must be walked (limit=None) — a capped read would
                # drop matches sitting past the store's 100-row page. Without
                # a category, the display limit bounds the read.
                raw = await asyncio.to_thread(
                    mgr.get_all_memories,
                    tenant_id=tenant_id,
                    agent_name=ns,
                    limit=None if category else limit,
                )
        except Exception as exc:
            raise _memory_read_failed(exc, tenant_id, namespaces) from exc

        for entry in raw:
            item = _entry_to_item(entry, ns)
            if item is None:
                continue
            if category and item.category != category:
                continue
            items.append(item)

    items = items[:limit]
    return MemoryListResponse(memories=items, count=len(items))


@router.get("/{tenant_id}/memories/stats", response_model=MemoryStats)
async def memory_stats(
    tenant_id: str,
    agent_name: str = Query(
        default=_USER_MEMORY_AGENT, description="The namespace to count"
    ),
):
    """Count a namespace's live and archived memories over its whole partition.

    The count reads the store, so a backend outage answers 503 rather than 0.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    mgr = await asyncio.to_thread(_get_memory_manager, tenant_id)
    try:
        stats = await asyncio.to_thread(
            mgr.get_memory_stats, tenant_id=tenant_id, agent_name=agent_name
        )
    except Exception as exc:
        raise _memory_read_failed(exc, tenant_id, [agent_name]) from exc
    return MemoryStats(
        agent_name=agent_name,
        user_id=tenant_id,
        total=stats["total_memories"],
        archived=stats["archived_memories"],
        writable=_is_writable(agent_name),
    )


@router.get("/{tenant_id}/memories/health", response_model=MemoryHealth)
async def memory_health(
    tenant_id: str,
    agent_name: str = Query(
        default=_USER_MEMORY_AGENT, description="The namespace to read"
    ),
):
    """Whether the tenant's memory manager is up and its store answers a read
    of ``agent_name``; an unhealthy answer names the step that failed."""
    tenant_id = canonical_tenant_id(tenant_id)

    def _probe() -> Optional[str]:
        try:
            mgr = _get_memory_manager(tenant_id)
        except Exception as exc:
            return f"The memory manager could not start ({type(exc).__name__})."
        if not mgr.health_check():
            return "The memory manager is not initialized."
        try:
            mgr.get_all_memories(tenant_id=tenant_id, agent_name=agent_name, limit=1)
        except Exception as exc:
            return f"The memory store did not answer a read ({type(exc).__name__})."
        return None

    problem = await asyncio.to_thread(_probe)
    return MemoryHealth(
        tenant_id=tenant_id,
        agent_name=agent_name,
        healthy=problem is None,
        problem=problem,
    )


@router.delete("/{tenant_id}/memories/{memory_id}")
async def delete_memory(
    tenant_id: str,
    memory_id: str,
    agent_name: str = Query(
        default=_USER_MEMORY_AGENT,
        description="The namespace the memory must belong to",
    ),
):
    """Delete one memory of ``agent_name`` by ID.

    Answers 404 unless the memory is this tenant's and in that namespace,
    and 403 for a system namespace.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    _require_writable(agent_name)
    mgr = await asyncio.to_thread(_get_memory_manager, tenant_id)

    def _delete() -> bool:
        # The store deletes by ID alone, so membership is checked first.
        row = mgr.memory.get(memory_id)
        if (
            row is None
            or row.get("agent_id") != agent_name
            or canonical_tenant_id(str(row.get("user_id") or "")) != tenant_id
        ):
            return False
        return mgr.delete_memory(
            memory_id=memory_id, tenant_id=tenant_id, agent_name=agent_name
        )

    if not await asyncio.to_thread(_delete):
        raise HTTPException(
            status_code=404,
            detail=f"Memory {memory_id} not found among the memories of {agent_name}",
        )
    return {"status": "deleted"}


@router.delete("/{tenant_id}/memories")
async def clear_memories(
    tenant_id: str,
    category: Optional[str] = Query(
        default=None,
        description="Clear only this category, or the whole namespace if omitted",
    ),
    agent_name: str = Query(
        default=_USER_MEMORY_AGENT, description="The namespace to clear"
    ),
):
    """Clear a namespace's memories, optionally only one category of them.

    System namespaces answer 403.
    """
    tenant_id = canonical_tenant_id(tenant_id)
    _require_writable(agent_name)
    mgr = await asyncio.to_thread(_get_memory_manager, tenant_id)

    if category:
        # The get + per-memory deletes are N blocking Vespa round-trips — run
        # the whole sweep off the event loop in one worker.
        def _clear_category() -> int:
            # Walk every page (limit=None): the category filter runs in
            # Python, so a capped read would leave matches past the store's
            # 100-row page undeleted while reporting success. Archived rows
            # are cleared too — both branches of this route mean the same
            # thing by "cleared".
            results = mgr.get_all_memories(
                tenant_id=tenant_id,
                agent_name=agent_name,
                limit=None,
                include_archived=True,
            )
            deleted = 0
            for r in results:
                if not isinstance(r, dict):
                    continue
                meta = r.get("metadata") or {}
                if meta.get("category") == category:
                    mid = r.get("id")
                    if mid:
                        mgr.delete_memory(
                            memory_id=str(mid),
                            tenant_id=tenant_id,
                            agent_name=agent_name,
                        )
                        deleted += 1
            return deleted

        deleted = await asyncio.to_thread(_clear_category)
        logger.info(
            "Cleared %d '%s' memories of %s for tenant=%s",
            deleted,
            category,
            agent_name,
            tenant_id,
        )
        return {
            "status": "cleared",
            "agent_name": agent_name,
            "category": category,
            "deleted": deleted,
        }

    await asyncio.to_thread(
        mgr.clear_agent_memory, tenant_id=tenant_id, agent_name=agent_name
    )
    logger.info("Cleared all memories of %s for tenant=%s", agent_name, tenant_id)
    return {"status": "cleared", "agent_name": agent_name}


_JOBS_SERVICE = "tenant_jobs"


def _build_cron_workflow(
    tenant_id: str, job_id: str, schedule: str, namespace: str
) -> dict:
    """Build an Argo CronWorkflow that runs the job via the job WorkflowTemplate."""
    if not get_workflow_settings().job_template:
        raise HTTPException(
            status_code=503,
            detail="Job WorkflowTemplate is not configured; cannot schedule jobs.",
        )
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "CronWorkflow",
        "metadata": {
            "name": _cron_workflow_name(tenant_id, job_id),
            "namespace": namespace,
            "labels": {
                "app": "cogniverse",
                "tenant": _sanitize_label_value(tenant_id),
                "job-id": job_id,
            },
        },
        "spec": {
            "schedule": schedule,
            "concurrencyPolicy": "Forbid",
            "workflowSpec": {
                "serviceAccountName": get_workflow_settings().service_account,
                "workflowTemplateRef": {"name": get_workflow_settings().job_template},
                "arguments": {
                    "parameters": [
                        {"name": "job-id", "value": job_id},
                        {"name": "tenant-id", "value": tenant_id},
                    ],
                },
            },
        },
    }


async def _submit_cron_workflow(manifest: dict) -> None:
    """Submit a CronWorkflow to Argo. Raises ``HTTPException(503)`` on
    network or non-2xx failure so callers can roll back the persisted
    ConfigStore entry instead of returning ``status="created"`` with no
    schedule ever firing on the cluster.
    """
    namespace = manifest["metadata"]["namespace"]
    name = manifest["metadata"]["name"]
    try:
        client = await _shared_argo_client()
        response = await client.post(
            f"{get_workflow_settings().api_url}/api/v1/cron-workflows/{namespace}",
            # Argo's CreateCronWorkflowRequest wraps the manifest.
            json={"namespace": namespace, "cronWorkflow": manifest},
            headers=_argo_auth_headers(),
        )
    except Exception as exc:
        raise failure_response(
            503,
            "argo_unavailable",
            f"Argo did not answer while scheduling job {name}; retry.",
            exc,
            job=name,
        ) from exc

    if response.status_code not in (200, 201):
        raise upstream_rejection(
            503,
            "argo_rejected",
            f"Argo rejected CronWorkflow {name} (HTTP {response.status_code}).",
            upstream_status=response.status_code,
            upstream_body=response.text,
            job=name,
        )

    logger.info("Submitted CronWorkflow: %s", name)


async def _delete_cron_workflow(name: str, namespace: str) -> None:
    """Delete a CronWorkflow from Argo. Raises on failure so the caller
    does NOT tombstone the ConfigStore entry while the schedule keeps
    firing on the cluster.

    A 404 from Argo means the CronWorkflow is already gone, which is the
    desired end state, so it counts as success.
    """
    try:
        client = await _shared_argo_client()
        response = await client.delete(
            f"{get_workflow_settings().api_url}/api/v1/cron-workflows/{namespace}/{name}",
            headers=_argo_auth_headers(),
        )
    except Exception as exc:
        raise failure_response(
            503,
            "argo_unavailable",
            f"Argo did not answer while deleting job {name}; retry.",
            exc,
            job=name,
        ) from exc

    if response.status_code not in (200, 404):
        raise upstream_rejection(
            503,
            "argo_rejected",
            f"Argo rejected the delete of CronWorkflow {name} "
            f"(HTTP {response.status_code}).",
            upstream_status=response.status_code,
            upstream_body=response.text,
            job=name,
        )

    logger.info("Deleted CronWorkflow: %s", name)


# Modes accepted by POST /{tenant_id}/optimize: the `--mode` choices of
# cogniverse_runtime.optimization_cli meant for an operator to start.
_MANUAL_OPTIMIZE_MODES = {
    "gateway-thresholds",
    "simba",
    "workflow",
    "profile",
    "entity-extraction",
    "llm-annotate",
    "synthetic",
}
_SYNTHETIC_MODE = "synthetic"
_DEFAULT_LOOKBACK_HOURS = 48.0


def _sanitize_label_value(value: str) -> str:
    """Shared K8s label sanitizer. The raw tenant_id is still passed through
    the ``--tenant-id`` CLI arg, so the sanitized label is for
    grouping/filtering only."""
    return sanitize_k8s_label_value(value)


_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]")


def _sanitize_resource_name(value: str) -> str:
    """K8s/Argo ``metadata.name`` must be an RFC-1123 segment: lowercase
    alphanumeric and '-' only, ≤63 chars, edge-trimmed. Tenant ids like
    ``org:env`` contain colons (and may be uppercased) that Argo rejects, so
    lowercase and replace unsupported chars with '-'."""
    cleaned = _NAME_SAFE_RE.sub("-", value.lower()).strip("-")[:63].strip("-")
    return cleaned or "x"


def _cron_workflow_name(tenant_id: str, job_id: str) -> str:
    """Argo CronWorkflow name for a scheduled job. Both segments are sanitized
    to a valid RFC-1123 name so a colon-form tenant ('acme:prod') — or any
    upper/underscore job id — produces a valid, stable name. create and delete
    MUST derive it the same way. The raw tenant_id still flows through the
    ``--tenant-id`` CLI arg."""
    return (
        f"tenant-job-{_sanitize_resource_name(tenant_id)}-"
        f"{_sanitize_resource_name(job_id)}"
    )


def _build_optimization_workflow_manifest(
    tenant_id: str,
    mode: str,
    namespace: str,
    *,
    lookback_hours: float = _DEFAULT_LOOKBACK_HOURS,
    agents: Optional[List[str]] = None,
) -> dict:
    """Build a one-off Argo Workflow that runs ``optimization_cli --mode``;
    ``agents`` becomes its ``--agents`` (the synthetic mode's optimizer
    types)."""
    parameters = [
        {"name": "mode", "value": mode},
        {"name": "tenant-id", "value": tenant_id},
        {"name": "lookback-hours", "value": f"{lookback_hours:g}"},
    ]
    if agents:
        parameters.append({"name": "agents", "value": ",".join(agents)})
    if not get_workflow_settings().optimization_template:
        raise HTTPException(
            status_code=503,
            detail="Optimization WorkflowTemplate is not configured.",
        )
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"manual-optimize-{mode}-",
            "namespace": namespace,
            "labels": {
                "app": "cogniverse",
                "cogniverse.ai/trigger": "manual",
                "cogniverse.ai/mode": mode,
                "cogniverse.ai/tenant": _sanitize_label_value(tenant_id),
            },
        },
        "spec": {
            # Argo Emissary posts ``workflowtaskresults`` under the pod's SA.
            # The default namespace SA lacks that permission in typical
            # installs; bind the Workflow to the runtime SA which the chart
            # RBAC grants.
            "serviceAccountName": get_workflow_settings().service_account,
            # Auto-delete completed workflows after 1 hour so the
            # namespace doesn't fill with dashboard-triggered runs.
            "ttlStrategy": {
                "secondsAfterCompletion": 3600,
                "secondsAfterSuccess": 3600,
                "secondsAfterFailure": 3600,
            },
            "workflowTemplateRef": {
                "name": get_workflow_settings().optimization_template,
            },
            "arguments": {"parameters": parameters},
        },
    }


async def _submit_workflow(manifest: dict) -> dict:
    """Submit a one-off Workflow to Argo. Returns the server response."""
    namespace = manifest["metadata"]["namespace"]
    try:
        client = await _shared_argo_client()
        response = await client.post(
            f"{get_workflow_settings().api_url}/api/v1/workflows/{namespace}",
            json={"workflow": manifest},
            headers=_argo_auth_headers(),
        )
    except httpx.HTTPError as exc:
        raise failure_response(
            502, "argo_unavailable", "The Argo API did not answer; retry.", exc
        ) from exc
    if response.status_code not in (200, 201):
        raise upstream_rejection(
            502,
            "argo_rejected",
            f"Argo rejected the Workflow submit (HTTP {response.status_code}).",
            upstream_status=response.status_code,
            upstream_body=response.text,
        )
    return response.json()


class ManualOptimizeRequest(BaseModel):
    mode: str
    lookback_hours: float = Field(
        _DEFAULT_LOOKBACK_HOURS,
        gt=0,
        le=8760,
        description="Hours of span history the run reads",
    )
    optimizers: Optional[List[str]] = Field(
        None,
        description=(
            "The synthetic mode's optimizer types to generate training data "
            "for; required for it and refused for every other mode"
        ),
    )


class ManualOptimizeResponse(BaseModel):
    workflow_name: str
    namespace: str
    mode: str
    status_url: str


class OptimizeRunStatus(BaseModel):
    workflow_name: str
    phase: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    message: Optional[str]
    # Per-step phase, so a step Argo omitted (its ``when`` did not fire, e.g.
    # the profile step for a tenant with no uploaded ground truth) reads as
    # Skipped rather than disappearing into a green workflow.
    steps: Dict[str, str] = {}
    # ``blocked_reason`` is populated when phase is ``Pending`` specifically
    # because the per-tenant optimization mutex is held by another Workflow.
    # The dashboard surfaces this so users don't confuse mutex-wait with
    # ordinary scheduler pending.
    blocked_reason: Optional[str] = None


def _step_phases(status_block: Dict[str, Any]) -> Dict[str, str]:
    """Phase per named workflow step.

    Argo records a step whose ``when`` did not fire with phase ``Omitted`` and
    node type ``Skipped``; both render here as ``Skipped``.
    """
    phases: Dict[str, str] = {}
    for node in (status_block.get("nodes") or {}).values():
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if node_type not in ("Pod", "Skipped"):
            continue
        name = node.get("displayName")
        if not name:
            continue
        phase = node.get("phase") or "Pending"
        if node_type == "Skipped" or phase == "Omitted":
            phase = "Skipped"
        phases[name] = phase
    return phases


def _extract_blocked_reason(status_block: Dict[str, Any]) -> Optional[str]:
    """Return a user-readable reason if the Workflow is Pending because
    the per-tenant optimization mutex is held by another Workflow,
    otherwise ``None``.

    Argo 3.x records mutex waits under ``status.synchronization.mutex.waiting``
    as ``[{mutex: "<ns>/<name>", holder: "<ns>/<workflow>"}, ...]``. Older
    versions just put the hint in ``status.message``, so we also scan
    that as a fallback."""
    if status_block.get("phase") != "Pending":
        return None
    sync = status_block.get("synchronization") or {}
    mutex = sync.get("mutex") or {}
    waiting = mutex.get("waiting") or []
    mutex_names = sorted(
        {entry.get("mutex", "") for entry in waiting if isinstance(entry, dict)}
    )
    mutex_names = [n for n in mutex_names if n]
    if mutex_names:
        return "Waiting for another optimization to release the mutex: " + ", ".join(
            mutex_names
        )
    message = (status_block.get("message") or "").lower()
    if "waiting for" in message and ("lock" in message or "mutex" in message):
        return status_block.get("message")
    return None


class OptimizeModes(BaseModel):
    modes: List[str]
    synthetic_optimizers: List[str]


@router.get("/optimize-modes", response_model=OptimizeModes)
async def list_optimization_modes():
    """The modes ``POST /{tenant_id}/optimize`` accepts and the optimizer
    types its synthetic mode generates data for, sorted."""
    return OptimizeModes(
        modes=sorted(_MANUAL_OPTIMIZE_MODES),
        synthetic_optimizers=sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER),
    )


def _synthetic_optimizers(body: ManualOptimizeRequest) -> Optional[List[str]]:
    """The request's optimizer types, validated against its mode."""
    supported = sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
    if body.mode != _SYNTHETIC_MODE:
        if body.optimizers is not None:
            raise HTTPException(
                status_code=400,
                detail="optimizers apply only to the synthetic mode",
            )
        return None
    if not body.optimizers:
        raise HTTPException(
            status_code=400,
            detail=f"The synthetic mode needs optimizers, from: {supported}",
        )
    unknown = sorted(set(body.optimizers) - set(supported))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown optimizers {unknown}; supported: {supported}",
        )
    return sorted(set(body.optimizers))


@router.post("/{tenant_id}/optimize", response_model=ManualOptimizeResponse)
async def run_manual_optimization(tenant_id: str, body: ManualOptimizeRequest):
    """Manually trigger an optimization run for a tenant via Argo.

    Submits a one-off Workflow that invokes ``optimization_cli --mode <mode>``
    in a fresh pod. Mirrors what the scheduled ``agent-optimization``
    CronWorkflow runs weekly — just on demand instead of on a schedule.
    """
    if get_workflow_settings().api_url is None:
        raise HTTPException(
            status_code=503,
            detail="Argo is not configured on this deployment.",
        )
    if body.mode not in _MANUAL_OPTIMIZE_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown optimization mode: {body.mode!r}. "
                f"Supported: {sorted(_MANUAL_OPTIMIZE_MODES)}"
            ),
        )

    optimizers = _synthetic_optimizers(body)
    manifest = _build_optimization_workflow_manifest(
        tenant_id,
        body.mode,
        get_workflow_settings().namespace,
        lookback_hours=body.lookback_hours,
        agents=optimizers,
    )
    response = await _submit_workflow(manifest)

    # Argo assigns the final name after generateName expansion.
    workflow_name = response.get("metadata", {}).get("name", "")
    if not workflow_name:
        logger.error("Argo returned no workflow name: %s", response)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "argo_no_workflow_name",
                "message": "Argo accepted the submission but returned no "
                "workflow name.",
            },
        )
    return ManualOptimizeResponse(
        workflow_name=workflow_name,
        namespace=get_workflow_settings().namespace,
        mode=body.mode,
        status_url=f"/admin/tenant/{tenant_id}/optimize/runs/{workflow_name}",
    )


def _workflow_tenant_tag(data: Dict[str, Any]) -> Optional[str]:
    """Return the tenant a Workflow was submitted for, or None if untagged.

    The submit path tags each Workflow two ways: a ``tenant-id`` spec
    argument carrying the raw tenant id, and a ``cogniverse.ai/tenant``
    label carrying the colon-sanitized (lossy) form. Prefer the argument;
    fall back to the label only when the argument is absent.
    """
    arguments = (data.get("spec") or {}).get("arguments") or {}
    for param in arguments.get("parameters") or []:
        if isinstance(param, dict) and param.get("name") == "tenant-id":
            value = param.get("value")
            if value:
                return value
    label = ((data.get("metadata") or {}).get("labels") or {}).get(
        "cogniverse.ai/tenant"
    )
    return label or None


def _workflow_belongs_to_tenant(data: Dict[str, Any], tenant_id: str) -> bool:
    """Whether the Workflow was submitted for ``tenant_id``.

    The ``cogniverse.ai/tenant`` label is the colon-sanitized (lossy) form, so
    two tenant ids can share one label value; ownership is decided on the raw
    ``tenant-id`` argument whenever the Workflow carries it.
    """
    workflow_tenant = _workflow_tenant_tag(data)
    return workflow_tenant is not None and canonical_tenant_id(
        workflow_tenant
    ) == canonical_tenant_id(tenant_id)


def _assert_workflow_belongs_to_tenant(data: Dict[str, Any], tenant_id: str) -> None:
    """Raise 404 unless the Workflow was submitted for ``tenant_id``.

    Tenant isolation: an operation scoped to ``/{tenant_id}/...`` must only
    touch that tenant's Workflows. A cross-tenant reference — or a Workflow
    with no tenant tag at all — answers 404 (not 403) so a caller cannot
    probe another tenant's run names.
    """
    if not _workflow_belongs_to_tenant(data, tenant_id):
        raise HTTPException(status_code=404, detail="Workflow not found")


async def _argo_get_workflow_data(workflow_name: str, tenant_id: str) -> Dict[str, Any]:
    """GET a Workflow from Argo and confirm it belongs to ``tenant_id``.

    Shared by the status/cancel/retry endpoints so each per-tenant action
    verifies ownership against the same Argo record before proceeding.
    Raises 404 when the Workflow is missing or owned by another tenant,
    502 on any other Argo error, 503 when Argo is not configured.
    """
    if get_workflow_settings().api_url is None:
        raise HTTPException(
            status_code=503,
            detail="Argo is not configured on this deployment.",
        )
    try:
        client = await _shared_argo_client()
        response = await client.get(
            f"{get_workflow_settings().api_url}/api/v1/workflows/{get_workflow_settings().namespace}/{workflow_name}",
            headers=_argo_auth_headers(),
        )
    except httpx.HTTPError as exc:
        raise failure_response(
            502,
            "argo_unavailable",
            "The Argo API did not answer; retry.",
            exc,
            workflow=workflow_name,
        ) from exc
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if response.status_code != 200:
        raise upstream_rejection(
            502,
            "argo_rejected",
            f"Argo answered HTTP {response.status_code} for workflow {workflow_name}.",
            upstream_status=response.status_code,
            upstream_body=response.text,
            workflow=workflow_name,
        )
    data = response.json()
    _assert_workflow_belongs_to_tenant(data, tenant_id)
    return data


class OptimizeRunSummary(BaseModel):
    workflow_name: str
    # ``None`` for a scheduled pipeline run: the CronWorkflow passes a mode per
    # step, so the Workflow itself carries no single mode.
    mode: Optional[str]
    trigger: str
    phase: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]


class OptimizeRunList(BaseModel):
    runs: List[OptimizeRunSummary]


class ArgoListUnavailableError(RuntimeError):
    """Argo could not be listed. Reported as 503 — never as an empty list."""


# Label Argo's CronWorkflow controller stamps on each Workflow it spawns.
_CRON_WORKFLOW_LABEL = "workflows.argoproj.io/cron-workflow"
# Response page size for the run listing.
_OPTIMIZE_RUNS_DEFAULT_LIMIT = 20
_OPTIMIZE_RUNS_MAX_LIMIT = 100
# Ceiling on what a single Argo list may return into runtime memory. Completed
# Workflows are reaped by ttlStrategy and by each CronWorkflow's history
# limits, so a namespace holding more than this is already misconfigured.
_ARGO_LIST_CEILING = 500


async def _argo_list_workflows(label_selector: str) -> List[Dict[str, Any]]:
    """List namespace Workflows matching ``label_selector``.

    Raises ``ArgoListUnavailableError`` when Argo is unreachable or answers
    anything but 200, so the caller reports the outage instead of an empty
    list that reads as "no runs".
    """
    settings = get_workflow_settings()
    try:
        client = await _shared_argo_client()
        response = await client.get(
            f"{settings.api_url}/api/v1/workflows/{settings.namespace}",
            params={
                "listOptions.labelSelector": label_selector,
                "listOptions.limit": _ARGO_LIST_CEILING,
            },
            headers=_argo_auth_headers(),
        )
    except httpx.HTTPError as exc:
        raise ArgoListUnavailableError(f"Argo API unreachable: {exc}") from exc
    if response.status_code != 200:
        raise ArgoListUnavailableError(
            f"Argo list failed ({response.status_code}): {response.text[:200]}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ArgoListUnavailableError(
            f"Argo list returned a non-JSON body: {response.text[:200]}"
        ) from exc
    return [item for item in (body.get("items") or []) if isinstance(item, dict)]


# Phases after which Argo runs nothing more for a Workflow.
_FINISHED_WORKFLOW_PHASES = frozenset({"Succeeded", "Failed", "Error"})


async def delete_finished_tenant_workflows(
    tenant_id: str,
) -> tuple[List[str], Dict[str, str]]:
    """Delete every finished Workflow ``tenant_id`` owns.

    Found as ``list_manual_optimization_runs`` finds a tenant's runs: by the
    tenant label and, for scheduled runs, the cron-workflow label, owned by
    the raw ``tenant-id`` argument. A deployment without Argo has none.
    Returns the names deleted (a Workflow already gone counts) and, for each
    one Argo did not delete, why.

    Raises:
        ArgoListUnavailableError: Argo could not be listed.
    """
    settings = get_workflow_settings()
    if settings.api_url is None:
        return [], {}
    listed = await _argo_list_workflows(
        f"cogniverse.ai/tenant={_sanitize_label_value(tenant_id)}"
    )
    listed += await _argo_list_workflows(_CRON_WORKFLOW_LABEL)
    names = sorted(
        {
            item["metadata"]["name"]
            for item in listed
            if (item.get("metadata") or {}).get("name")
            and (item.get("status") or {}).get("phase") in _FINISHED_WORKFLOW_PHASES
            and _workflow_belongs_to_tenant(item, tenant_id)
        }
    )
    deleted: List[str] = []
    failed: Dict[str, str] = {}
    client = await _shared_argo_client()
    for name in names:
        try:
            response = await client.delete(
                f"{settings.api_url}/api/v1/workflows/{settings.namespace}/{name}",
                headers=_argo_auth_headers(),
            )
        except httpx.HTTPError as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        if response.status_code in (200, 404):
            deleted.append(name)
        else:
            failed[name] = f"HTTP {response.status_code}: {response.text[:200]}"
    return deleted, failed


def _references_template(node: Any, template_name: str) -> bool:
    """Whether any ``templateRef``/``workflowTemplateRef`` in ``node`` names
    ``template_name``. A manual run references it at the spec root; a
    CronWorkflow-spawned run references it from a step or DAG task."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("templateRef", "workflowTemplateRef"):
                if isinstance(value, dict) and value.get("name") == template_name:
                    return True
            if _references_template(value, template_name):
                return True
        return False
    if isinstance(node, list):
        return any(_references_template(item, template_name) for item in node)
    return False


def _is_optimization_run(data: Dict[str, Any]) -> bool:
    """Whether a Workflow runs the optimization WorkflowTemplate.

    Scheduled tenant *jobs* carry a ``tenant-id`` argument too, so the tenant
    tag alone cannot tell them apart from optimization runs.
    """
    template = get_workflow_settings().optimization_template
    if not template:
        return False
    return _references_template(data.get("spec") or {}, template)


def _workflow_parameter(data: Dict[str, Any], name: str) -> Optional[str]:
    """Value of a workflow-level spec argument, or ``None``."""
    arguments = (data.get("spec") or {}).get("arguments") or {}
    for param in arguments.get("parameters") or []:
        if isinstance(param, dict) and param.get("name") == name:
            return param.get("value")
    return None


def _optimize_run_summary(data: Dict[str, Any]) -> OptimizeRunSummary:
    metadata = data.get("metadata") or {}
    labels = metadata.get("labels") or {}
    status_block = data.get("status") or {}
    return OptimizeRunSummary(
        workflow_name=metadata.get("name") or "",
        mode=labels.get("cogniverse.ai/mode") or _workflow_parameter(data, "mode"),
        trigger=labels.get("cogniverse.ai/trigger")
        or ("scheduled" if labels.get(_CRON_WORKFLOW_LABEL) else "unknown"),
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
    )


def _run_sort_key(data: Dict[str, Any]) -> str:
    """Start time, falling back to creation time for a run Argo has not
    started yet. Argo emits RFC-3339 UTC, which sorts lexicographically."""
    status_block = data.get("status") or {}
    metadata = data.get("metadata") or {}
    return status_block.get("startedAt") or metadata.get("creationTimestamp") or ""


@router.get("/{tenant_id}/optimize/runs", response_model=OptimizeRunList)
async def list_optimization_runs(
    tenant_id: str,
    limit: int = Query(_OPTIMIZE_RUNS_DEFAULT_LIMIT, ge=1, le=_OPTIMIZE_RUNS_MAX_LIMIT),
):
    """List a tenant's optimization Workflows, newest first.

    Two selectors, because Argo does not copy a CronWorkflow's labels onto the
    Workflows it spawns: manual runs carry ``cogniverse.ai/tenant`` from the
    submit path, scheduled runs are found by the cron-workflow label the
    controller stamps. Both are then narrowed to this tenant's optimization
    runs by the raw ``tenant-id`` argument and the optimization
    WorkflowTemplate reference.
    """
    if get_workflow_settings().api_url is None:
        raise HTTPException(
            status_code=503,
            detail="Argo is not configured on this deployment.",
        )
    try:
        listed = await _argo_list_workflows(
            f"cogniverse.ai/tenant={_sanitize_label_value(tenant_id)}"
        )
        listed += await _argo_list_workflows(_CRON_WORKFLOW_LABEL)
    except ArgoListUnavailableError as exc:
        raise failure_response(
            503,
            "argo_unavailable",
            f"Argo could not list the workflows of tenant {tenant_id}; retry.",
            exc,
            tenant_id=tenant_id,
        ) from exc

    by_name: Dict[str, Dict[str, Any]] = {}
    for item in listed:
        name = (item.get("metadata") or {}).get("name")
        if not name or name in by_name:
            continue
        if not _workflow_belongs_to_tenant(item, tenant_id):
            continue
        if not _is_optimization_run(item):
            continue
        by_name[name] = item

    ordered = sorted(by_name.values(), key=_run_sort_key, reverse=True)
    return OptimizeRunList(
        runs=[_optimize_run_summary(item) for item in ordered[:limit]]
    )


@router.get(
    "/{tenant_id}/optimize/runs/{workflow_name}",
    response_model=OptimizeRunStatus,
)
async def get_manual_optimization_status(tenant_id: str, workflow_name: str):
    """Return current phase + timestamps for a dashboard-triggered run."""
    data = await _argo_get_workflow_data(workflow_name, tenant_id)
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
        steps=_step_phases(status_block),
    )


async def _argo_workflow_action(
    verb: str, workflow_name: str, action_path: str
) -> dict:
    """Proxy an Argo ``/<action>`` request (``terminate`` / ``retry``) and
    unwrap the response body. Centralised so cancel and retry share the
    same error-handling shape: 404 if the Workflow doesn't exist, 502 on
    any other Argo error, raw JSON on success."""
    if get_workflow_settings().api_url is None:
        raise HTTPException(
            status_code=503,
            detail="Argo is not configured on this deployment.",
        )
    url = (
        f"{get_workflow_settings().api_url}/api/v1/workflows/"
        f"{get_workflow_settings().namespace}/{workflow_name}/{action_path}"
    )
    try:
        client = await _shared_argo_client()
        response = await client.put(
            url,
            json={"name": workflow_name},
            headers=_argo_auth_headers(),
        )
    except httpx.HTTPError as exc:
        raise failure_response(
            502,
            "argo_unavailable",
            f"The Argo API did not answer the {verb}; retry.",
            exc,
            workflow=workflow_name,
        ) from exc
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if response.status_code not in (200, 201):
        raise upstream_rejection(
            502,
            "argo_rejected",
            f"Argo rejected the {verb} of workflow {workflow_name} "
            f"(HTTP {response.status_code}).",
            upstream_status=response.status_code,
            upstream_body=response.text,
            workflow=workflow_name,
        )
    return response.json()


@router.post(
    "/{tenant_id}/optimize/runs/{workflow_name}/cancel",
    response_model=OptimizeRunStatus,
)
async def cancel_manual_optimization(tenant_id: str, workflow_name: str):
    """Terminate an in-flight optimize Workflow.

    Argo's ``terminate`` verb stops the main container immediately; TTL
    still applies so the Workflow resource auto-deletes after the
    configured grace window. Returns the post-terminate status block so
    the dashboard can surface the ``Failed`` phase without polling again.
    """
    await _argo_get_workflow_data(workflow_name, tenant_id)
    data = await _argo_workflow_action("cancel", workflow_name, "terminate")
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
        steps=_step_phases(status_block),
    )


@router.post(
    "/{tenant_id}/optimize/runs/{workflow_name}/retry",
    response_model=OptimizeRunStatus,
)
async def retry_manual_optimization(tenant_id: str, workflow_name: str):
    """Retry a ``Failed``/``Error`` optimize Workflow.

    Argo's ``retry`` verb restarts only the failed nodes, reusing the
    successful ones — cheaper than resubmitting. The per-tenant mutex
    on the WorkflowTemplate still applies, so a retry that lands while
    another Workflow holds the mutex will queue behind it.
    """
    await _argo_get_workflow_data(workflow_name, tenant_id)
    data = await _argo_workflow_action("retry", workflow_name, "retry")
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
        steps=_step_phases(status_block),
    )


@router.post("/{tenant_id}/jobs", response_model=JobResponse)
async def create_job(tenant_id: str, body: JobCreateRequest):
    """Create a scheduled agent job.

    Stores the job config in ConfigStore and, if Argo is available, submits
    a CronWorkflow that will run the job_executor on the given schedule.
    """
    cm = _require_config_manager()
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    config_value = {
        "job_id": job_id,
        "name": body.name,
        "schedule": body.schedule,
        "query": body.query,
        "post_actions": body.post_actions,
        "created_at": now,
    }
    # Submit the CronWorkflow first when Argo is available: _submit_cron_workflow
    # propagates a submit failure, so the ConfigStore job row is persisted only
    # once the cluster has accepted the schedule — no visible row with nothing
    # firing.
    submitted_namespace = None
    if get_workflow_settings().api_url:
        namespace = get_workflow_settings().namespace
        manifest = _build_cron_workflow(tenant_id, job_id, body.schedule, namespace)
        await _submit_cron_workflow(manifest)
        submitted_namespace = namespace

    try:
        await asyncio.to_thread(
            cm.set_config_value,
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service=_JOBS_SERVICE,
            config_key=f"job_{job_id}",
            config_value=config_value,
        )
    except BaseException as write_error:
        # The schedule is live on the cluster and nothing describes it: list_jobs
        # reads config rows, delete_job 404s without one, and job_executor raises
        # on every tick. Remove it so the failed create leaves nothing behind.
        if submitted_namespace is None:
            raise
        name = _cron_workflow_name(tenant_id, job_id)
        cleanup = asyncio.ensure_future(
            _delete_cron_workflow(name, submitted_namespace)
        )
        try:
            # Shielded: the failure being compensated is often this task's own
            # cancellation (the client disconnected), and an unshielded await
            # would abandon the delete at its first suspension point.
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.wait({cleanup}, timeout=_CLEANUP_COMPLETION_TIMEOUT_S)
            raise
        except BaseException as cleanup_error:
            # Nothing anywhere records the orphan: the config row that would
            # describe it is what failed to write. The cluster's own labels
            # (app, tenant, job-id on the CronWorkflow) are the record, so name
            # what an operator has to look for.
            logger.error(
                "Job %s for tenant %s failed to commit (%r) and its CronWorkflow "
                "%s in namespace %s could not be removed (%r): the schedule is "
                "still firing",
                job_id,
                tenant_id,
                write_error,
                name,
                submitted_namespace,
                cleanup_error,
            )
            raise
        raise
    logger.info(
        "Created job %s for tenant %s (schedule=%s)", job_id, tenant_id, body.schedule
    )

    return JobResponse(
        job_id=job_id,
        name=body.name,
        schedule=body.schedule,
        query=body.query,
        post_actions=body.post_actions,
        status="created",
        created_at=now,
    )


@router.get("/{tenant_id}/jobs", response_model=JobListResponse)
async def list_jobs(tenant_id: str):
    """List all scheduled jobs for a tenant."""
    cm = _require_config_manager()
    # The manager has no list-by-service read, so canonicalize explicitly —
    # create_job writes through the manager, and listing with the raw path
    # param would read an empty namespace for any non-canonical tenant id.
    entries = await asyncio.to_thread(
        cm.store.list_configs,
        tenant_id=canonical_tenant_id(tenant_id),
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
    )

    jobs: List[JobResponse] = []
    for entry in entries or []:
        v = entry.config_value if hasattr(entry, "config_value") else entry
        if not isinstance(v, dict) or "job_id" not in v or v.get("deleted"):
            continue
        jobs.append(
            JobResponse(
                job_id=v["job_id"],
                name=v.get("name", ""),
                schedule=v.get("schedule", ""),
                query=v.get("query", ""),
                post_actions=v.get("post_actions", []),
                status="active",
                created_at=v.get("created_at"),
            )
        )

    return JobListResponse(jobs=jobs)


@router.delete("/{tenant_id}/jobs/{job_id}")
async def delete_job(tenant_id: str, job_id: str):
    """Delete a scheduled job by ID.

    Removes the backing Argo CronWorkflow (so it stops firing) and then
    tombstones the ConfigStore entry. An already-deleted job returns 404.
    """
    cm = _require_config_manager()
    # Manager read for the same canonicalized key create_job wrote —
    # a raw store read never finds the job, so every delete 404s.
    value = await asyncio.to_thread(
        cm.get_config_value,
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
    )
    if not isinstance(value, dict) or not value or value.get("deleted"):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Stop the schedule on the cluster before tombstoning the config — a
    # config-only delete leaves the CronWorkflow firing indefinitely.
    if get_workflow_settings().api_url:
        await _delete_cron_workflow(
            _cron_workflow_name(tenant_id, job_id), get_workflow_settings().namespace
        )

    await asyncio.to_thread(
        cm.set_config_value,
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
        config_value={"job_id": job_id, "deleted": True},
    )
    logger.info("Deleted job %s for tenant %s", job_id, tenant_id)
    return {"status": "deleted", "job_id": job_id}
