"""Shared health check endpoint mixin for FastAPI agents"""

from typing import Dict, Any


class HealthCheckMixin:
    """Mixin to provide standard health check endpoint"""

    def get_health_status(self) -> Dict[str, Any]:
        """
        Override this method to provide custom health check logic.
        Default implementation returns healthy status.
        """
        return {
            "status": "healthy",
            "agent": getattr(self, "agent_name", self.__class__.__name__),
        }

    def setup_health_endpoint(self, app):
        """
        Setup health check endpoint on FastAPI app.
        Call this from your agent's __init__ after creating the app.

        Example:
            app = FastAPI()
            self.setup_health_endpoint(app)
        """

        @app.get("/health")
        async def health():
            return self.get_health_status()
