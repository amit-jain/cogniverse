"""Agent endpoints - unified interface for all agent operations."""

import logging
from typing import Any, Dict

from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentTask(BaseModel):
    """Task request for agent processing."""

    agent_name: str
    query: str
    context: Dict[str, Any] = {}
    top_k: int = 10


@router.get("/")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents."""
    config_manager = ConfigManager()
    agent_registry = AgentRegistry(config_manager=config_manager)
    agents = agent_registry.list_agents()

    return {
        "count": len(agents),
        "agents": agents,
    }


@router.get("/{agent_name}")
async def get_agent_info(agent_name: str) -> Dict[str, Any]:
    """Get information about a specific agent."""
    config_manager = ConfigManager()
    agent_registry = AgentRegistry(config_manager=config_manager)

    # Try to get agent
    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # Return agent card/info
    return {
        "name": agent_name,
        "type": agent.__class__.__name__,
        "capabilities": getattr(agent, "capabilities", []),
        "description": getattr(agent, "description", ""),
    }


@router.get("/{agent_name}/card")
async def get_agent_card(agent_name: str) -> Dict[str, Any]:
    """Get agent card (A2A protocol) for a specific agent."""
    config_manager = ConfigManager()
    agent_registry = AgentRegistry(config_manager=config_manager)

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    # If agent has get_agent_card method, use it
    if hasattr(agent, "get_agent_card"):
        return agent.get_agent_card()

    # Otherwise return basic card
    return {
        "name": agent_name,
        "version": "1.0",
        "capabilities": getattr(agent, "capabilities", []),
        "endpoints": {
            "process": f"/agents/{agent_name}/process",
            "info": f"/agents/{agent_name}",
        },
    }


@router.post("/{agent_name}/process")
async def process_agent_task(agent_name: str, task: AgentTask) -> Dict[str, Any]:
    """Process a task with a specific agent."""
    config_manager = ConfigManager()
    agent_registry = AgentRegistry(config_manager=config_manager)

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    try:
        # Call agent's process method
        if hasattr(agent, "process"):
            result = await agent.process(
                query=task.query, context=task.context, top_k=task.top_k
            )
            return result
        elif hasattr(agent, "forward"):
            # For DSPy agents
            result = agent.forward(
                query=task.query, context=task.context, top_k=task.top_k
            )
            return {"result": result}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Agent '{agent_name}' does not support processing",
            )
    except Exception as e:
        logger.error(f"Error processing task with agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_name}/upload")
async def upload_file_to_agent(
    agent_name: str,
    file: UploadFile = File(...),
    top_k: int = 10,
) -> Dict[str, Any]:
    """Upload a file for agent processing (e.g., video/image search)."""
    config_manager = ConfigManager()
    agent_registry = AgentRegistry(config_manager=config_manager)

    agent = agent_registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    try:
        # Read file content
        content = await file.read()

        # Call agent's upload handler if available
        if hasattr(agent, "process_upload"):
            result = await agent.process_upload(
                file_content=content, filename=file.filename, top_k=top_k
            )
            return result
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{agent_name}' does not support file uploads",
            )
    except Exception as e:
        logger.error(f"Error uploading file to agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
