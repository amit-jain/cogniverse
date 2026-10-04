"""The runtime the web client's operations views talk to, for browser tests.

The tenant admin, tenant self-service, profile admin, agents and ingestion
routers are mounted at the paths the runtime mounts them on, over the
caller's real config store and schema loader, with a cluster-events channel
of their own so tenant deletes and session closes reach this worker the way
they reach a runtime replica.
Given an ingest processor, the ingestion worker's claim loop runs on the
server's loop against ``REDIS_URL``, as a worker pod runs it.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Awaitable, Callable, Iterator, Optional

from fastapi import FastAPI

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.ingestion_worker import status_api
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis
from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _claim_loop
from cogniverse_runtime.routers import admin, agents, ingestion, tenant
from tests.utils.web_client import serve_app


@contextmanager
def serve_ops_runtime(
    config_manager,
    schema_loader,
    redis_url: str,
    *,
    ingest_processor: Optional[Callable[..., Awaitable[dict]]] = None,
) -> Iterator[str]:
    """Serve the operations routes on a real uvicorn socket; yields its URL."""
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    tenant.set_config_manager(config_manager)
    agents.set_agent_registry(
        AgentRegistry(tenant_id="default", config_manager=config_manager)
    )

    @asynccontextmanager
    async def cluster_events(_app):
        # Started on the server's loop, as the runtime starts its own.
        events = ClusterEvents(
            redis_url,
            f"web-ops-test-{uuid.uuid4().hex[:8]}",
            {
                "tenant_deleted": tm.release_deleted_tenant,
                "session_closed": admin.sweep_closed_session,
            },
            channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        )
        await events.start()
        tm.set_cluster_events(events)
        admin.set_cluster_events(events)
        stop = asyncio.Event()
        worker = None
        if ingest_processor is not None:
            config = WorkerConfig()
            config.claim_block_ms = 200
            worker = asyncio.create_task(
                _claim_loop(
                    await get_redis(config.redis_url),
                    config,
                    stop,
                    processor=ingest_processor,
                )
            )
        try:
            yield
        finally:
            stop.set()
            if worker is not None:
                await asyncio.wait_for(worker, timeout=20)
            await close_redis()
            tm.set_cluster_events(None)
            admin.set_cluster_events(None)
            await events.close()

    app = FastAPI(lifespan=cluster_events)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    app.include_router(tenant.router, prefix="/admin/tenant")
    app.include_router(agents.router, prefix="/agents")
    app.include_router(ingestion.router, prefix="/ingestion")
    app.include_router(status_api.router, prefix="/ingestion")
    app.dependency_overrides[ingestion.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[ingestion.get_schema_loader_dependency] = lambda: (
        schema_loader
    )
    try:
        with serve_app(app) as url:
            yield url
    finally:
        tm.set_config_manager(previous[0])
        tm.set_schema_loader(previous[1])
        admin.reset_dependencies()
        tenant.set_config_manager(None)
        BackendRegistry.get_instance().clear_instances()
