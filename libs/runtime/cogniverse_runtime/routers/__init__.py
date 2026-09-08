"""Runtime routers - all API endpoints."""

from cogniverse_runtime.routers import (
    admin,
    agents,
    health,
    ingestion,
    openai_compat,
    search,
    wiki,
)

__all__ = [
    "health",
    "agents",
    "search",
    "ingestion",
    "admin",
    "wiki",
    "openai_compat",
]
