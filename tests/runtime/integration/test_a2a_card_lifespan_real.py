"""The A2A agent card must advertise the real loaded agents.

Regression: the card was built from agent_registry.list_agents() BEFORE
config_loader.load_agents() ran, so the registry was empty and skills fell back
to a single 'default' skill. This drives the real main.py lifespan and reads the
mounted card over ASGI.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_a2a_card_advertises_loaded_agents(monkeypatch, workflow_state_redis_url):
    # Keep the lifespan light: no sandbox connect, no memory lifecycle scan.
    monkeypatch.setenv("REDIS_URL", workflow_state_redis_url)
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
    monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
    # dspy.configure is once-per-task; stub so a re-run in the module is safe.
    import dspy

    monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)

    from cogniverse_runtime.main import lifespan

    app = FastAPI()
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/a2a/.well-known/agent-card.json")

        assert resp.status_code == 200, resp.text[:300]
        body = resp.json()
        skill_ids = {s["id"] for s in body["skills"]}
        # The card must reflect the populated registry, not the 'default' stub.
        assert "search_agent" in skill_ids, skill_ids
        assert "default" not in skill_ids, skill_ids
        assert len(skill_ids) > 1, skill_ids


@pytest.mark.asyncio
async def test_the_schema_migration_runs_after_startup_without_holding_it(
    monkeypatch, workflow_state_redis_url
):
    """The provenance schema migration deploys one package per drifted tenant,
    so it runs once startup completes: the runtime serves while it runs."""
    import asyncio

    import dspy

    from cogniverse_runtime import main as runtime_main

    monkeypatch.setenv("REDIS_URL", workflow_state_redis_url)
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
    monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
    monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)
    running = asyncio.Event()
    release = asyncio.Event()
    migrated = []

    async def migration_in_progress(resolve_registry, base_schema_name, stop):
        registry = await asyncio.to_thread(resolve_registry)
        migrated.append((type(registry).__name__, base_schema_name))
        running.set()
        await release.wait()

    monkeypatch.setattr(runtime_main, "_migrate_drifted_schemas", migration_in_progress)

    app = FastAPI()
    async with runtime_main.lifespan(app):
        await asyncio.wait_for(running.wait(), timeout=5)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            served = await client.get("/a2a/.well-known/agent-card.json")
        release.set()

    assert migrated == [("SchemaRegistry", "provenance")]
    assert served.status_code == 200
    assert served.json()["name"] == "Cogniverse Runtime"


@pytest.mark.asyncio
async def test_shutdown_stops_the_background_deploys_before_its_drains(
    monkeypatch, workflow_state_redis_url
):
    """No background deploy may start while shutdown drains: the migration is
    cancelled before the first drain runs."""
    import asyncio

    import dspy

    from cogniverse_runtime import main as runtime_main
    from cogniverse_runtime.routers import admin as admin_router

    monkeypatch.setenv("REDIS_URL", workflow_state_redis_url)
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
    monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
    monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)
    running = asyncio.Event()
    order = []

    async def migration_never_done(resolve_registry, base_schema_name, stop):
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append(f"migration cancelled, stop set: {stop.is_set()}")
            raise

    async def recording_blob_drain(timeout_s: float = 60.0) -> bool:
        order.append("blob drain")
        return True

    monkeypatch.setattr(runtime_main, "_migrate_drifted_schemas", migration_never_done)
    monkeypatch.setattr(admin_router, "drain_blob_writes", recording_blob_drain)

    async with runtime_main.lifespan(FastAPI()):
        await asyncio.wait_for(running.wait(), timeout=5)

    assert order == ["migration cancelled, stop set: True", "blob drain"]
