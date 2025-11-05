"""Unified FastAPI Runtime - Single entry point for all Cogniverse services.

This replaces 10+ scattered FastAPI apps with a single, unified runtime that:
- Dynamically loads backends/agents from config.yml
- Consolidates all endpoints under one service
- Enables clean deployment and scaling
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from cogniverse_core.config.utils import get_config
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cogniverse_runtime.config_loader import get_config_loader

# Import routers
from cogniverse_runtime.routers import admin, agents, health, ingestion, search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app - handles startup and shutdown."""

    # Startup
    logger.info("Starting Cogniverse Runtime...")

    # 1. Load configuration
    from cogniverse_core.config.manager import ConfigManager
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from pathlib import Path

    config_manager = ConfigManager()
    config = get_config(tenant_id="default", config_manager=config_manager)
    logger.info(f"Loaded configuration for tenant: {config.tenant_id}")

    # 2. Initialize SchemaLoader
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    logger.info("SchemaLoader initialized")

    # 3. Set dependencies on routers
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    ingestion.set_config_manager(config_manager)
    ingestion.set_schema_loader(schema_loader)
    logger.info("Router dependencies configured")

    # 4. Initialize registries
    backend_registry = BackendRegistry(config_manager=config_manager)
    agent_registry = AgentRegistry(config_manager=config_manager)
    logger.info("Registries initialized")

    # 5. Use config loader to dynamically load backends and agents
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents()

    logger.info(
        f"Loaded {len(backend_registry.list_backends())} backends, "
        f"{len(agent_registry.list_agents())} agents"
    )

    logger.info("Cogniverse Runtime started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Cogniverse Runtime...")
    # Add cleanup logic here if needed
    logger.info("Cogniverse Runtime shut down successfully")


# Create FastAPI app
app = FastAPI(
    title="Cogniverse Runtime",
    description="Unified multi-agent RAG system for video content analysis and search",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Cogniverse Runtime",
        "version": "1.0.0",
        "description": "Unified multi-agent RAG system",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    from cogniverse_core.config.manager import ConfigManager

    # Load config to get port
    config_manager = ConfigManager()
    config = get_config(tenant_id="default", config_manager=config_manager)

    port = config.get("runtime", {}).get("port", 8000)
    host = config.get("runtime", {}).get("host", "0.0.0.0")

    uvicorn.run(
        "cogniverse_runtime.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
