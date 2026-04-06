"""Tenant self-service endpoints.

Allows tenants to customize their agent experience:
- Instructions: per-tenant system prompt stored in ConfigStore
- Memory management: browse and delete Mem0 memories
- Scheduled jobs: create/list/delete Argo CronWorkflow-backed jobs
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.config_store import ConfigScope

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level config manager — set by main.py at startup
_config_manager: Optional[ConfigManager] = None

# Optional Argo API URL — set by main.py if Argo is available
_argo_api_url: Optional[str] = None
_argo_namespace: str = "cogniverse"


def set_config_manager(config_manager: ConfigManager) -> None:
    """Inject ConfigManager (called from main.py lifespan)."""
    global _config_manager
    _config_manager = config_manager


def set_argo_config(api_url: Optional[str], namespace: str = "cogniverse") -> None:
    """Inject Argo API URL (called from main.py lifespan, optional)."""
    global _argo_api_url, _argo_namespace
    _argo_api_url = api_url
    _argo_namespace = namespace


def _require_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialised")
    return _config_manager


# ── Pydantic models ───────────────────────────────────────────────────────


class InstructionsRequest(BaseModel):
    text: str


class InstructionsResponse(BaseModel):
    text: str
    updated_at: str


class MemoryItem(BaseModel):
    id: str
    memory: str
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


# ── Instruction endpoints ─────────────────────────────────────────────────

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
        raise HTTPException(status_code=404, detail="No instructions found for this tenant")
    value = entry.config_value
    return InstructionsResponse(
        text=value.get("text", ""),
        updated_at=value.get("updated_at", ""),
    )


@router.delete("/{tenant_id}/instructions")
async def delete_instructions(tenant_id: str):
    """Clear the stored tenant instructions."""
    cm = _require_config_manager()
    # Overwrite with empty value to signal "cleared"
    cm.set_config_value(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
        config_value={"text": "", "updated_at": datetime.now(timezone.utc).isoformat()},
    )
    logger.info("Cleared instructions for tenant %s", tenant_id)
    return {"status": "cleared"}


# ── Memory endpoints ──────────────────────────────────────────────────────


def _get_memory_manager(tenant_id: str):
    """Return an initialised Mem0MemoryManager for the given tenant.

    Raises HTTPException(503) if the manager is not yet initialised (no
    memory backend wired at runtime — this is expected in unit tests with
    mocked managers).
    """
    mgr = Mem0MemoryManager(tenant_id)
    if not mgr.memory:
        raise HTTPException(
            status_code=503,
            detail="Memory backend not initialised for this tenant",
        )
    return mgr


@router.get("/{tenant_id}/memories", response_model=MemoryListResponse)
async def list_memories(
    tenant_id: str,
    agent: str = Query(default="_strategy_store", description="Agent namespace to query"),
    q: Optional[str] = Query(default=None, description="Semantic search query"),
    limit: int = Query(default=20, ge=1, le=200, description="Max results"),
):
    """List or search memories for a tenant's agent namespace."""
    mgr = _get_memory_manager(tenant_id)

    if q:
        raw = mgr.search_memory(query=q, tenant_id=tenant_id, agent_name=agent, top_k=limit)
    else:
        raw = mgr.get_all_memories(tenant_id=tenant_id, agent_name=agent)

    items: List[MemoryItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        items.append(
            MemoryItem(
                id=str(entry.get("id", "")),
                memory=entry.get("memory", entry.get("text", "")),
                metadata=entry.get("metadata", {}),
                created_at=str(entry.get("created_at", "")) or None,
            )
        )

    return MemoryListResponse(memories=items, count=len(items))


@router.delete("/{tenant_id}/memories/{memory_id}")
async def delete_memory(tenant_id: str, memory_id: str):
    """Delete a single memory by ID."""
    mgr = _get_memory_manager(tenant_id)
    success = mgr.delete_memory(memory_id=memory_id, tenant_id=tenant_id, agent_name="*")
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found or delete failed")
    return {"status": "deleted"}


@router.delete("/{tenant_id}/memories")
async def clear_memories(
    tenant_id: str,
    agent: str = Query(default="_strategy_store", description="Agent namespace to clear"),
):
    """Clear all memories for a tenant's agent namespace."""
    mgr = _get_memory_manager(tenant_id)
    success = mgr.clear_agent_memory(tenant_id=tenant_id, agent_name=agent)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear memories")
    logger.info("Cleared memories for tenant=%s agent=%s", tenant_id, agent)
    return {"status": "cleared", "agent": agent}


# ── Job endpoints ─────────────────────────────────────────────────────────

_JOBS_SERVICE = "tenant_jobs"


def _build_cron_workflow(tenant_id: str, job_id: str, schedule: str, namespace: str) -> dict:
    """Build an Argo CronWorkflow manifest for the given job."""
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "CronWorkflow",
        "metadata": {
            "name": f"tenant-job-{tenant_id}-{job_id}",
            "namespace": namespace,
            "labels": {
                "app": "cogniverse",
                "tenant": tenant_id,
                "job-id": job_id,
            },
        },
        "spec": {
            "schedule": schedule,
            "concurrencyPolicy": "Forbid",
            "workflowSpec": {
                "entrypoint": "run-job",
                "templates": [
                    {
                        "name": "run-job",
                        "container": {
                            "image": "cogniverse-runtime:latest",
                            "command": ["python", "-m", "cogniverse_runtime.job_executor"],
                            "args": [
                                "--job-id", job_id,
                                "--tenant-id", tenant_id,
                                "--runtime-url", "http://cogniverse-runtime:28000",
                            ],
                        },
                    }
                ],
            },
        },
    }


async def _submit_cron_workflow(manifest: dict) -> None:
    """Submit a CronWorkflow to Argo. Logs on failure; does not raise."""
    namespace = manifest["metadata"]["namespace"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{_argo_api_url}/api/v1/cronworkflows/{namespace}",
                json=manifest,
            )
            if response.status_code in (200, 201):
                name = manifest["metadata"]["name"]
                logger.info("Submitted CronWorkflow: %s", name)
            else:
                logger.error(
                    "Argo CronWorkflow submit failed (%s): %s",
                    response.status_code,
                    response.text[:500],
                )
    except Exception as exc:
        logger.error("Failed to submit CronWorkflow to Argo: %s", exc)


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
    cm.set_config_value(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
        config_value=config_value,
    )
    logger.info("Created job %s for tenant %s (schedule=%s)", job_id, tenant_id, body.schedule)

    if _argo_api_url:
        manifest = _build_cron_workflow(tenant_id, job_id, body.schedule, _argo_namespace)
        await _submit_cron_workflow(manifest)

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
        if not isinstance(v, dict) or "job_id" not in v:
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
    """Delete a scheduled job by ID."""
    cm = _require_config_manager()
    entry = cm.store.get_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
    )
    if entry is None or not entry.config_value:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cm.delete_config_value(
        tenant_id=tenant_id,
        scope=ConfigScope.SYSTEM,
        service=_JOBS_SERVICE,
        config_key=f"job_{job_id}",
    )
    logger.info("Deleted job %s for tenant %s", job_id, tenant_id)
    return {"status": "deleted", "job_id": job_id}
