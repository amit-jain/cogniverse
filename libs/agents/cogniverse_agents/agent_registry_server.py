"""
FastAPI server for AgentRegistry with REST API endpoints for dynamic registration.
"""

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cogniverse_agents.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentRegistrationRequest(BaseModel):
    """Agent registration request payload"""

    name: str
    url: str
    capabilities: List[str]
    health_endpoint: str = "/health"
    process_endpoint: str = "/tasks/send"
    timeout: int = 30


class AgentDiscoveryRequest(BaseModel):
    """Agent discovery request payload"""

    agent_urls: List[str]


class AgentRegistryServer:
    """
    FastAPI server wrapper for AgentRegistry with REST API endpoints.
    """

    def __init__(self):
        """Initialize registry server"""
        self.app = FastAPI(title="Agent Registry", version="1.0.0")
        self.registry = AgentRegistry()

        # Setup endpoints
        self._setup_endpoints()

        logger.info("AgentRegistryServer initialized")

    def _setup_endpoints(self):
        """Setup REST API endpoints"""

        @self.app.post("/register")
        async def register_agent(request: AgentRegistrationRequest) -> Dict[str, Any]:
            """
            Register a new agent with the registry.

            Args:
                request: Agent registration data

            Returns:
                Registration result
            """
            try:
                success = self.registry.register_agent_from_data(request.dict())

                if success:
                    return {
                        "status": "success",
                        "message": f"Agent {request.name} registered successfully",
                        "agent_name": request.name,
                    }
                else:
                    raise HTTPException(
                        status_code=400, detail="Failed to register agent"
                    )

            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/unregister/{agent_name}")
        async def unregister_agent(agent_name: str) -> Dict[str, Any]:
            """
            Unregister an agent from the registry.

            Args:
                agent_name: Name of agent to unregister

            Returns:
                Unregistration result
            """
            success = self.registry.unregister_agent(agent_name)

            if success:
                return {
                    "status": "success",
                    "message": f"Agent {agent_name} unregistered successfully",
                }
            else:
                raise HTTPException(status_code=404, detail="Agent not found")

        @self.app.post("/discover")
        async def discover_agents(request: AgentDiscoveryRequest) -> Dict[str, Any]:
            """
            Discover and auto-register agents from URLs.

            Args:
                request: Agent discovery request with URLs

            Returns:
                Discovery results
            """
            try:
                results = await self.registry.auto_register_from_urls(
                    request.agent_urls
                )

                successful = sum(results.values())
                total = len(results)

                return {
                    "status": "success",
                    "message": f"Discovered and registered {successful}/{total} agents",
                    "results": results,
                }

            except Exception as e:
                logger.error(f"Discovery failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/agents")
        async def list_agents() -> Dict[str, Any]:
            """
            List all registered agents.

            Returns:
                List of agent names and details
            """
            agents = self.registry.list_agents()
            return {
                "status": "success",
                "total_agents": len(agents),
                "agents": agents,
            }

        @self.app.get("/agents/{agent_name}")
        async def get_agent(agent_name: str) -> Dict[str, Any]:
            """
            Get details for a specific agent.

            Args:
                agent_name: Name of agent

            Returns:
                Agent details
            """
            agent = self.registry.get_agent(agent_name)

            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            return {
                "status": "success",
                "agent": {
                    "name": agent.name,
                    "url": agent.url,
                    "capabilities": agent.capabilities,
                    "health_endpoint": agent.health_endpoint,
                    "process_endpoint": agent.process_endpoint,
                    "health_status": agent.health_status,
                    "last_health_check": (
                        agent.last_health_check.isoformat()
                        if agent.last_health_check
                        else None
                    ),
                },
            }

        @self.app.get("/capabilities/{capability}")
        async def find_agents_by_capability(capability: str) -> Dict[str, Any]:
            """
            Find agents that support a specific capability.

            Args:
                capability: Required capability

            Returns:
                List of agents with the capability
            """
            agents = self.registry.find_agents_by_capability(capability)

            return {
                "status": "success",
                "capability": capability,
                "total_agents": len(agents),
                "agents": [
                    {
                        "name": agent.name,
                        "url": agent.url,
                        "health_status": agent.health_status,
                    }
                    for agent in agents
                ],
            }

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """
            Registry health check.

            Returns:
                Health status
            """
            health_results = await self.registry.health_check_all()

            return {
                "status": "healthy",
                "registry": "operational",
                "agents": health_results,
            }

        @self.app.get("/stats")
        async def get_stats() -> Dict[str, Any]:
            """
            Get registry statistics.

            Returns:
                Registry statistics
            """
            stats = self.registry.get_registry_stats()

            return {"status": "success", "stats": stats}


# Create server instance
app = AgentRegistryServer().app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
