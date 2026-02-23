"""Agent endpoints - unified interface for all agent operations."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level dependencies (injected from main.py)
_agent_registry: Optional[AgentRegistry] = None
_config_manager: Optional[ConfigManager] = None
_schema_loader: Optional[SchemaLoader] = None


def set_agent_registry(registry: AgentRegistry) -> None:
    """Inject AgentRegistry dependency for router endpoints"""
    global _agent_registry
    _agent_registry = registry
    logger.info("AgentRegistry injected into agents router")


def set_agent_dependencies(
    config_manager: ConfigManager, schema_loader: SchemaLoader
) -> None:
    """Inject config_manager and schema_loader for in-process agent execution."""
    global _config_manager, _schema_loader
    _config_manager = config_manager
    _schema_loader = schema_loader
    logger.info("Agent dependencies (config_manager, schema_loader) injected")


def get_registry() -> AgentRegistry:
    """Get the injected registry or raise error"""
    if _agent_registry is None:
        raise RuntimeError(
            "AgentRegistry not initialized. Call set_agent_registry() first."
        )
    return _agent_registry


class AgentTask(BaseModel):
    """Task request for agent processing."""

    agent_name: str
    query: str
    context: Dict[str, Any] = {}
    top_k: int = 10


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

    # Try to get agent
    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # Return agent endpoint info
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

    # Return A2A agent card
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
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    if _config_manager is None or _schema_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Agent dependencies not configured. Runtime not fully initialized.",
        )

    tenant_id = task.context.get("tenant_id", "default")
    capabilities = set(agent.capabilities)

    # Route based on agent capabilities
    if capabilities & {"search", "video_search", "retrieval", "routing"}:
        return await _execute_search_task(task, tenant_id)
    elif capabilities & {"summarization", "text_generation"}:
        return _execute_text_generation_task(task, agent_name)
    elif capabilities & {"text_analysis", "sentiment", "classification"}:
        return _execute_text_analysis_task(task, agent_name)
    elif capabilities & {"detailed_report", "analysis"}:
        return _execute_text_generation_task(task, agent_name)
    else:
        return {
            "status": "success",
            "agent": agent_name,
            "message": f"Agent '{agent_name}' acknowledged query: {task.query}",
            "capabilities": agent.capabilities,
        }


async def _execute_search_task(task: AgentTask, tenant_id: str) -> Dict[str, Any]:
    """Execute a search task using the SearchService."""
    from cogniverse_foundation.config.utils import get_config
    from cogniverse_runtime.search.service import SearchService

    config = get_config(tenant_id=tenant_id, config_manager=_config_manager)
    search_service = SearchService(
        config=config,
        config_manager=_config_manager,
        schema_loader=_schema_loader,
    )

    profile = config.get("default_profile", "video_colpali_smol500_mv_frame")

    results = search_service.search(
        query=task.query,
        profile=profile,
        tenant_id=tenant_id,
        top_k=task.top_k,
        ranking_strategy="float_float",
    )

    result_list = [r.to_dict() for r in results]
    result_count = len(result_list)

    if result_count > 0:
        message = f"Found {result_count} results for '{task.query}'"
    else:
        message = f"No results found for '{task.query}'"

    return {
        "status": "success",
        "agent": "search_agent",
        "message": message,
        "results_count": result_count,
        "results": result_list,
        "profile": profile,
    }


def _execute_text_generation_task(
    task: AgentTask, agent_name: str
) -> Dict[str, Any]:
    """Execute a text generation task (summarization, reports)."""
    return {
        "status": "success",
        "agent": agent_name,
        "message": (
            f"Text generation via '{agent_name}' requires an LLM backend. "
            f"Query received: {task.query}"
        ),
    }


def _execute_text_analysis_task(
    task: AgentTask, agent_name: str
) -> Dict[str, Any]:
    """Execute a text analysis task (sentiment, classification)."""
    return {
        "status": "success",
        "agent": agent_name,
        "message": (
            f"Text analysis via '{agent_name}' requires an LLM backend. "
            f"Query received: {task.query}"
        ),
    }


@router.post("/{agent_name}/upload")
async def upload_file_to_agent(
    agent_name: str,
    file: UploadFile = File(...),
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Upload a file for agent processing (e.g., video/image search).

    Note: This endpoint is for routing to agents. For direct agent execution,
    agents should expose their own upload endpoints.
    """
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # In curated registry pattern, agents are remote services
    # This endpoint would forward the request to the agent's URL
    raise HTTPException(
        status_code=501,
        detail=f"Direct file upload not supported. Call agent at {agent.url}/upload",
    )
