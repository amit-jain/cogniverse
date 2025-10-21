"""A2A Protocol Standard Endpoints Mixin"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class A2AEndpointsMixin:
    """
    Mixin to provide standard A2A protocol endpoints.

    Implements:
    - /.well-known/agent-card.json (Google A2A standard)
    - Auto-registration helper

    Usage:
        class MyAgent(A2AEndpointsMixin):
            def __init__(self):
                self.agent_name = "MyAgent"
                self.agent_description = "Does something useful"
                self.agent_version = "1.0.0"
                self.agent_url = "http://localhost:8001"
                self.agent_capabilities = ["search", "analyze"]

                app = FastAPI()
                self.setup_a2a_endpoints(app)
    """

    def get_agent_card_data(self) -> Dict[str, Any]:
        """
        Override this to provide custom agent card data.
        Default implementation uses instance attributes.
        """
        return {
            "name": getattr(self, "agent_name", self.__class__.__name__),
            "description": getattr(self, "agent_description", "A2A-compliant agent"),
            "url": getattr(self, "agent_url", "http://localhost:8000"),
            "version": getattr(self, "agent_version", "1.0.0"),
            "protocol": "a2a",
            "protocol_version": "1.0",
            "capabilities": getattr(self, "agent_capabilities", []),
            "skills": getattr(self, "agent_skills", []),
        }

    def setup_a2a_endpoints(self, app):
        """
        Setup standard A2A protocol endpoints on FastAPI app.

        Adds:
        - /.well-known/agent-card.json (Google A2A standard)

        Args:
            app: FastAPI application instance
        """

        @app.get("/.well-known/agent-card.json")
        async def get_well_known_agent_card():
            """
            Google A2A standard well-known URI for agent discovery.

            Returns agent card in standard A2A format.
            """
            return self.get_agent_card_data()

        logger.info(
            f"A2A endpoints configured for {getattr(self, 'agent_name', 'agent')}"
        )

    def get_registry_registration_data(self) -> Dict[str, Any]:
        """
        Get data for registering with agent registry.

        Returns:
            Registration data with agent endpoint information
        """
        card = self.get_agent_card_data()
        return {
            "name": card["name"],
            "url": card["url"],
            "capabilities": card["capabilities"],
            "health_endpoint": "/health",
            "process_endpoint": getattr(self, "process_endpoint", "/tasks/send"),
            "agent_card_url": f"{card['url']}/.well-known/agent-card.json",
        }

    async def register_with_registry(self, registry_url: str) -> Dict[str, Any]:
        """
        Register this agent with a central agent registry.

        Args:
            registry_url: URL of the agent registry service

        Returns:
            Registration response from registry
        """
        import httpx

        registration_data = self.get_registry_registration_data()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{registry_url}/register",
                    json=registration_data,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info(
                    f"Agent {registration_data['name']} registered with registry at {registry_url}"
                )
                return response.json()
        except Exception as e:
            logger.error(f"Failed to register with registry: {e}")
            return {"error": str(e), "status": "failed"}
