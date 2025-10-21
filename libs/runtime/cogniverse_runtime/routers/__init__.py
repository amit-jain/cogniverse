"""Runtime routers - all API endpoints."""

from cogniverse_runtime.routers import admin, agents, health, ingestion, search

__all__ = ["health", "agents", "search", "ingestion", "admin"]
