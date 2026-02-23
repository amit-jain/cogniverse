"""Unified FastAPI Runtime - Single entry point for all Cogniverse services.

This replaces 10+ scattered FastAPI apps with a single, unified runtime that:
- Dynamically loads backends/agents from configs/config.json
- Consolidates all endpoints under one service
- Enables clean deployment and scaling
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import get_config

# Import routers
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.config_loader import get_config_loader
from cogniverse_runtime.routers import admin, agents, events, health, ingestion, search
from cogniverse_synthetic.api import router as synthetic_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app - handles startup and shutdown."""

    # Startup
    logger.info("Starting Cogniverse Runtime...")

    # 1. Load configuration
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)
    logger.info(f"Loaded configuration for tenant: {config.tenant_id}")

    # 2. Initialize SchemaLoader
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    logger.info("SchemaLoader initialized")

    # 3. Set dependencies on routers
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)

    # Wire ingestion and search routers via FastAPI dependency overrides
    app.dependency_overrides[ingestion.get_config_manager_dependency] = (
        lambda: config_manager
    )
    app.dependency_overrides[ingestion.get_schema_loader_dependency] = (
        lambda: schema_loader
    )
    app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )
    logger.info("Router dependencies configured")

    # 4. Initialize registries
    backend_registry = BackendRegistry.get_instance()
    agent_registry = AgentRegistry(config_manager=config_manager)
    logger.info("Registries initialized")

    # 5. Wire agent registry and dependencies to agents router
    agents.set_agent_registry(agent_registry)
    agents.set_agent_dependencies(config_manager, schema_loader)
    logger.info("AgentRegistry and dependencies wired to agents router")

    # 6. Use config loader to dynamically load backends and agents
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents(agent_registry=agent_registry)

    logger.info(
        f"Loaded {len(backend_registry.list_backends())} backends, "
        f"{len(agent_registry.list_agents())} agents"
    )

    # 7. Create system backend and deploy metadata schemas
    from cogniverse_foundation.config.bootstrap import BootstrapConfig

    bootstrap = BootstrapConfig.from_environment()
    system_backend = BackendRegistry.get_instance().get_ingestion_backend(
        name=bootstrap.backend_type,
        tenant_id="system",
        config={
            "backend": {"url": bootstrap.backend_url, "port": bootstrap.backend_port}
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    # Deploy metadata schemas once at startup (not in every VespaBackend.__init__)
    system_config = config_manager.get_system_config()
    system_backend.schema_manager.upload_metadata_schemas(
        app_name=system_config.application_name
    )
    logger.info("Metadata schemas deployed via system backend")

    # 8. Wire tenant manager dependencies
    tenant_manager.set_config_manager(config_manager)
    tenant_manager.set_schema_loader(schema_loader)
    tenant_manager.backend = system_backend
    logger.info("Tenant manager wired to Runtime")

    # 9. Configure DSPy LM and synthetic data service
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import (
        AgentMappingRule,
        DSPyModuleConfig,
        OptimizerGenerationConfig,
        SyntheticGeneratorConfig,
    )
    from cogniverse_synthetic.api import configure_service as configure_synthetic

    llm_config = config.get_llm_config()
    primary_lm = create_dspy_lm(llm_config.primary)
    dspy.configure(lm=primary_lm)
    logger.info(f"DSPy configured with LM: {llm_config.primary.model}")

    modality_config = OptimizerGenerationConfig(
        optimizer_type="modality",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
                module_type="ChainOfThought",
            ),
        },
        agent_mappings=[
            AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
            AgentMappingRule(modality="DOCUMENT", agent_name="document_agent"),
            AgentMappingRule(modality="IMAGE", agent_name="image_search_agent"),
            AgentMappingRule(modality="AUDIO", agent_name="audio_analysis_agent"),
        ],
    )
    routing_config = OptimizerGenerationConfig(
        optimizer_type="routing",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                module_type="ChainOfThought",
            ),
        },
    )
    synthetic_gen_config = SyntheticGeneratorConfig(
        optimizer_configs={
            "modality": modality_config,
            "routing": routing_config,
        },
    )
    configure_synthetic(generator_config=synthetic_gen_config)
    logger.info("Synthetic data service configured")

    logger.info("Cogniverse Runtime started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Cogniverse Runtime...")
    # Add cleanup logic here if needed
    logger.info("Cogniverse Runtime shut down successfully")


# Create FastAPI app
app = FastAPI(
    title="Cogniverse Runtime",
    description="Multi-agent AI platform for content intelligence",
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
app.include_router(tenant_manager.router, prefix="/admin", tags=["tenant-management"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(synthetic_router, tags=["synthetic-data"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Cogniverse Runtime",
        "version": "1.0.0",
        "description": "Multi-agent AI platform for content intelligence",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    from cogniverse_foundation.config.utils import create_default_config_manager

    # Load config to get port
    config_manager = create_default_config_manager()
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
