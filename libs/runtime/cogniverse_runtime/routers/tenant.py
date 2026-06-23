"""Tenant self-service endpoints.

Allows tenants to customize their agent experience:
- Instructions: per-tenant system prompt stored in ConfigStore
- Memory management: browse and delete Mem0 memories
- Scheduled jobs: create/list/delete Argo CronWorkflow-backed jobs
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.config_loader import get_workflow_settings
from cogniverse_sdk.interfaces.config_store import ConfigScope

logger = logging.getLogger(__name__)

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
    cm.set_config_value(
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
    entry = cm.store.get_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
    )
    if entry is None or not entry.config_value:
        raise HTTPException(
            status_code=404, detail="No instructions found for this tenant"
        )
    value = entry.config_value
    return InstructionsResponse(
        text=value.get("text", ""),
        updated_at=value.get("updated_at", ""),
    )


@router.delete("/{tenant_id}/instructions")
async def delete_instructions(tenant_id: str):
    """Clear the stored tenant instructions."""
    cm = _require_config_manager()
    cm.set_config_value(
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
        lazy_init_memory(mgr, tenant_id, _require_config_manager())
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


@router.post("/{tenant_id}/memories")
async def create_memory(tenant_id: str, request: MemoryCreateRequest):
    """Save a user-defined memory with optional category, kind, metadata."""
    mgr = _get_memory_manager(tenant_id)
    metadata: Dict[str, Any] = {}
    if request.category:
        metadata["category"] = request.category
    if request.kind:
        metadata["kind"] = request.kind
    if request.metadata:
        # Caller-supplied metadata wins over the derived fields above.
        metadata.update(request.metadata)

    memory_id = mgr.add_memory(
        content=request.text,
        tenant_id=tenant_id,
        agent_name=_USER_MEMORY_AGENT,
        metadata=metadata,
        infer=False,
    )
    return {
        "status": "saved",
        "id": str(memory_id),
        "type": "preference",
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
    )


@router.get("/{tenant_id}/memories", response_model=MemoryListResponse)
async def list_memories(
    tenant_id: str,
    q: Optional[str] = Query(default=None, description="Search query"),
    type: Optional[str] = Query(
        default=None, description="Filter by type: preference, strategy"
    ),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    limit: int = Query(default=20, ge=1, le=200, description="Max results"),
):
    """List or search tenant memories across all types.

    Without ``q``, lists all memories.  With ``q``, performs semantic search.
    Use ``type`` to restrict to a single memory type and ``category`` to
    filter user-created memories by their category tag.
    """
    mgr = _get_memory_manager(tenant_id)

    if type:
        ns = _TYPE_TO_NAMESPACE.get(type)
        if ns is None:
            raise HTTPException(status_code=400, detail=f"Unknown memory type: {type}")
        namespaces = [ns]
    else:
        namespaces = list(_ALL_NAMESPACES)

    items: List[MemoryItem] = []
    for ns in namespaces:
        if q:
            raw = mgr.search_memory(
                query=q,
                tenant_id=tenant_id,
                agent_name=ns,
                top_k=limit,
            )
        else:
            raw = mgr.get_all_memories(tenant_id=tenant_id, agent_name=ns)

        for entry in raw:
            item = _entry_to_item(entry, ns)
            if item is None:
                continue
            if category and item.category != category:
                continue
            items.append(item)

    items = items[:limit]
    return MemoryListResponse(memories=items, count=len(items))


@router.delete("/{tenant_id}/memories/{memory_id}")
async def delete_memory(tenant_id: str, memory_id: str):
    """Delete a single user-owned memory by ID.

    Returns 403 if the memory belongs to a system namespace.
    """
    mgr = _get_memory_manager(tenant_id)

    success = mgr.delete_memory(
        memory_id=memory_id,
        tenant_id=tenant_id,
        agent_name=_USER_MEMORY_AGENT,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return {"status": "deleted"}


@router.delete("/{tenant_id}/memories")
async def clear_memories(
    tenant_id: str,
    category: Optional[str] = Query(
        default=None,
        description="Clear only this category, or all user memories if omitted",
    ),
):
    """Clear user-owned memories. System memories (strategies) are not affected.

    Optionally filter by category to only clear a subset.
    """
    mgr = _get_memory_manager(tenant_id)

    if category:
        results = mgr.get_all_memories(
            tenant_id=tenant_id,
            agent_name=_USER_MEMORY_AGENT,
        )
        deleted = 0
        for r in results:
            if not isinstance(r, dict):
                continue
            meta = r.get("metadata", {})
            if meta.get("category") == category:
                mid = r.get("id")
                if mid:
                    mgr.delete_memory(
                        memory_id=str(mid),
                        tenant_id=tenant_id,
                        agent_name=_USER_MEMORY_AGENT,
                    )
                    deleted += 1
        logger.info(
            "Cleared %d '%s' memories for tenant=%s", deleted, category, tenant_id
        )
        return {"status": "cleared", "category": category, "deleted": deleted}

    mgr.clear_agent_memory(tenant_id=tenant_id, agent_name=_USER_MEMORY_AGENT)
    logger.info("Cleared all user memories for tenant=%s", tenant_id)
    return {"status": "cleared"}


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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{get_workflow_settings().api_url}/api/v1/cron-workflows/{namespace}",
                # Argo's CreateCronWorkflowRequest wraps the manifest.
                json={"namespace": namespace, "cronWorkflow": manifest},
                headers=_argo_auth_headers(),
            )
    except Exception as exc:
        logger.error("Failed to submit CronWorkflow %s to Argo: %s", name, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Argo unreachable while scheduling job {name}: {exc}",
        ) from exc

    if response.status_code not in (200, 201):
        logger.error(
            "Argo CronWorkflow submit failed (%s): %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Argo rejected CronWorkflow {name}: "
                f"HTTP {response.status_code} {response.text[:200]}"
            ),
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(
                f"{get_workflow_settings().api_url}/api/v1/cron-workflows/{namespace}/{name}",
                headers=_argo_auth_headers(),
            )
    except Exception as exc:
        logger.error("Failed to delete CronWorkflow %s from Argo: %s", name, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Argo unreachable while deleting job {name}: {exc}",
        ) from exc

    if response.status_code not in (200, 404):
        logger.error(
            "Argo CronWorkflow delete failed (%s): %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Argo rejected CronWorkflow delete for {name}: "
                f"HTTP {response.status_code} {response.text[:200]}"
            ),
        )

    logger.info("Deleted CronWorkflow: %s", name)


# Modes accepted by POST /{tenant_id}/optimize. Matches the `--mode` choices
# declared by cogniverse_runtime.optimization_cli (minus `triggered` and
# `cleanup`, which aren't intended for interactive dashboard use, and
# `synthetic` which has its own scheduled CronWorkflow).
_MANUAL_OPTIMIZE_MODES = {
    "gateway-thresholds",
    "simba",
    "workflow",
    "profile",
    "entity-extraction",
}


_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """K8s label values must match ``([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]``
    and be ≤63 chars. Tenant IDs like ``org:env`` contain colons that violate
    this, so replace unsupported chars with ``-`` and trim edges. The raw
    tenant_id is still passed through the ``--tenant-id`` CLI arg, so the
    sanitized label is for grouping/filtering only."""
    cleaned = _LABEL_SAFE_RE.sub("-", value).strip("-_.")[:63]
    return cleaned or "unknown"


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
    tenant_id: str, mode: str, namespace: str
) -> dict:
    """Build a one-off Argo Workflow that runs ``optimization_cli --mode``."""
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
            "arguments": {
                "parameters": [
                    {"name": "mode", "value": mode},
                    {"name": "tenant-id", "value": tenant_id},
                    {"name": "lookback-hours", "value": "48"},
                ],
            },
        },
    }


async def _submit_workflow(manifest: dict) -> dict:
    """Submit a one-off Workflow to Argo. Returns the server response."""
    namespace = manifest["metadata"]["namespace"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{get_workflow_settings().api_url}/api/v1/workflows/{namespace}",
                json={"workflow": manifest},
                headers=_argo_auth_headers(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Argo API unreachable: {exc}"
        ) from exc
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Argo Workflow submit failed ({response.status_code}): "
                f"{response.text[:500]}"
            ),
        )
    return response.json()


class ManualOptimizeRequest(BaseModel):
    mode: str


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
    # ``blocked_reason`` is populated when phase is ``Pending`` specifically
    # because the per-tenant optimization mutex is held by another Workflow.
    # The dashboard surfaces this so users don't confuse mutex-wait with
    # ordinary scheduler pending.
    blocked_reason: Optional[str] = None


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

    manifest = _build_optimization_workflow_manifest(
        tenant_id, body.mode, get_workflow_settings().namespace
    )
    response = await _submit_workflow(manifest)

    # Argo assigns the final name after generateName expansion.
    workflow_name = response.get("metadata", {}).get("name", "")
    if not workflow_name:
        raise HTTPException(
            status_code=502,
            detail=f"Argo returned no workflow name: {response}",
        )
    return ManualOptimizeResponse(
        workflow_name=workflow_name,
        namespace=get_workflow_settings().namespace,
        mode=body.mode,
        status_url=f"/admin/tenant/{tenant_id}/optimize/runs/{workflow_name}",
    )


@router.get(
    "/{tenant_id}/optimize/runs/{workflow_name}",
    response_model=OptimizeRunStatus,
)
async def get_manual_optimization_status(tenant_id: str, workflow_name: str):
    """Return current phase + timestamps for a dashboard-triggered run."""
    if get_workflow_settings().api_url is None:
        raise HTTPException(
            status_code=503,
            detail="Argo is not configured on this deployment.",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{get_workflow_settings().api_url}/api/v1/workflows/{get_workflow_settings().namespace}/{workflow_name}",
                headers=_argo_auth_headers(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Argo API unreachable: {exc}"
        ) from exc
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Argo API error ({response.status_code}): {response.text[:500]}",
        )
    data = response.json()
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.put(
                url,
                json={"name": workflow_name},
                headers=_argo_auth_headers(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Argo API unreachable during {verb}: {exc}",
        ) from exc
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Argo {verb} failed ({response.status_code}): {response.text[:500]}"
            ),
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
    data = await _argo_workflow_action("cancel", workflow_name, "terminate")
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
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
    data = await _argo_workflow_action("retry", workflow_name, "retry")
    status_block = data.get("status", {}) or {}
    return OptimizeRunStatus(
        workflow_name=workflow_name,
        phase=status_block.get("phase"),
        started_at=status_block.get("startedAt"),
        finished_at=status_block.get("finishedAt"),
        message=status_block.get("message"),
        blocked_reason=_extract_blocked_reason(status_block),
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
    # Submit the CronWorkflow first when Argo is available — a failure here
    # used to be swallowed, leaving the ConfigStore row visible but no
    # schedule firing. Now we let _submit_cron_workflow propagate, and only
    # persist the job once the cluster has accepted it.
    if get_workflow_settings().api_url:
        manifest = _build_cron_workflow(
            tenant_id, job_id, body.schedule, get_workflow_settings().namespace
        )
        await _submit_cron_workflow(manifest)

    cm.set_config_value(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
        config_value=config_value,
    )
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
    entries = cm.store.list_configs(
        tenant_id=tenant_id,
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
    entry = cm.store.get_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
    )
    value = entry.config_value if entry is not None else None
    if not isinstance(value, dict) or not value or value.get("deleted"):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Stop the schedule on the cluster before tombstoning the config — a
    # config-only delete leaves the CronWorkflow firing indefinitely.
    if get_workflow_settings().api_url:
        await _delete_cron_workflow(
            _cron_workflow_name(tenant_id, job_id), get_workflow_settings().namespace
        )

    cm.set_config_value(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
        config_value={"job_id": job_id, "deleted": True},
    )
    logger.info("Deleted job %s for tenant %s", job_id, tenant_id)
    return {"status": "deleted", "job_id": job_id}
