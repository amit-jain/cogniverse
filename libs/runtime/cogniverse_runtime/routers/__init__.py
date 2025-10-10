"""Runtime routers - all API endpoints."""

from cogniverse_runtime.routers import health, agents, search, ingestion, admin

__all__ = ["health", "agents", "search", "ingestion", "admin"]
