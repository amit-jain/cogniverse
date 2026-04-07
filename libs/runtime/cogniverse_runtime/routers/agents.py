"""Agent endpoints - unified interface for all agent operations."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_agents.routing.annotation_queue import AnnotationQueue

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

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


def _ensure_dispatcher() -> AgentDispatcher:
    """Lazily create the dispatcher once registry + deps are wired."""
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    if _agent_registry is None or _config_manager is None or _schema_loader is None:
        raise RuntimeError(
            "Agent dependencies not configured. Runtime not fully initialized."
        )
    _dispatcher = AgentDispatcher(
        agent_registry=_agent_registry,
        config_manager=_config_manager,
        schema_loader=_schema_loader,
        sandbox_manager=_sandbox_manager,
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
    """

    agent_name: str
    query: str
    context: Dict[str, Any] = {}
    top_k: int = 10
    context_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


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

    success = registry.register_agent_from_data(data.dict())

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


@router.post("/annotations/queue/{span_id}/complete")
async def complete_annotation(span_id: str, body: CompleteRequest) -> Dict[str, Any]:
    """Mark an annotation as completed."""
    queue = get_annotation_queue()
    try:
        request = queue.complete(span_id=span_id, label=body.label)
        return {"status": "completed", "annotation": request.to_dict()}
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

    # Merge multi-turn fields into context dict for dispatcher
    dispatch_context = dict(task.context)
    if task.context_id is not None:
        dispatch_context["context_id"] = task.context_id
    if task.conversation_history is not None:
        dispatch_context["conversation_history"] = task.conversation_history

    try:
        return await dispatcher.dispatch(
            agent_name=agent_name,
            query=task.query,
            context=dispatch_context,
            top_k=task.top_k,
        )
    except ValueError as e:
        detail = str(e)
        if "not found" in detail:
            raise HTTPException(status_code=404, detail=detail)
        elif "no supported execution path" in detail:
            raise HTTPException(status_code=501, detail=detail)
        raise HTTPException(status_code=400, detail=detail)


# Audit fix #13 — POST /agents/{name}/upload was a 501 stub that had no
# implementation path and tested only its own 501 response. File uploads
# already have a real home at POST /ingestion/upload, so the stub has been
# removed entirely. Tests that targeted /agents/upload now expect 404.
