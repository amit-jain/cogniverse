"""The runtime the web client's operations views talk to, for browser tests.

The tenant admin, profile admin and agents routers are mounted at the paths
the runtime mounts them on, over the caller's real config store and schema
loader, with a cluster-events channel of their own so tenant deletes and
session closes reach this worker the way they reach a runtime replica.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Iterator

from fastapi import FastAPI

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.routers import admin, agents
from tests.utils.web_client import serve_app


@contextmanager
def serve_ops_runtime(config_manager, schema_loader, redis_url: str) -> Iterator[str]:
    """Serve the operations routes on a real uvicorn socket; yields its URL."""
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
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
        try:
            yield
        finally:
            tm.set_cluster_events(None)
            admin.set_cluster_events(None)
            await events.close()

    app = FastAPI(lifespan=cluster_events)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    app.include_router(agents.router, prefix="/agents")
    try:
        with serve_app(app) as url:
            yield url
    finally:
        tm.set_config_manager(previous[0])
        tm.set_schema_loader(previous[1])
        admin.reset_dependencies()
        BackendRegistry.get_instance().clear_instances()
