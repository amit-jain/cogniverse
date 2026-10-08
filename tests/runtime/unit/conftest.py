"""Unit-test isolation for ``tests/runtime/unit``.

These are pure-unit tests (dependencies mocked / injected via
``InMemoryConfigStore``). They must not touch Vespa. The one leak is the global
telemetry singleton: ``get_telemetry_manager()`` (foundation/telemetry/manager)
falls back to ``create_default_config_manager()`` → ``VespaConfigStore`` on its
first call, and the project-wide dead-port default then makes that read fail.

Seed the singleton with a default in-memory ``TelemetryManager`` before every
test (and reset after) so no code path triggers the Vespa fallback — the same
seed-the-singleton pattern ``telemetry_manager_with_phoenix`` uses, minus
Phoenix.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_telemetry_singleton():
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
    telemetry_manager_module._telemetry_manager = TelemetryManager(TelemetryConfig())
    yield
    TelemetryManager.reset()


@pytest.fixture
def no_injected_agent_registry():
    """Run with no registry injected into the agents router, then restore it.

    /health prefers the registry injected into the agents router over the one
    it builds itself. A registry an earlier test left there answers instead of
    the one a test patches in, and its clients may belong to a closed loop."""
    from cogniverse_runtime.routers import agents as agents_router

    saved = (agents_router._agent_registry, agents_router._dispatcher)
    agents_router._agent_registry = None
    agents_router._dispatcher = None
    yield
    agents_router._agent_registry, agents_router._dispatcher = saved


@pytest.fixture
def harness_key_config_store(monkeypatch):
    """Bind tenant retirement to an in-memory credential store."""
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_runtime.admin import tenant_manager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    monkeypatch.setattr(tenant_manager, "_config_manager", ConfigManager(store=store))
    assert tenant_manager._config_manager.store is store
    return store


class InProcessClusterEvents:
    """The cluster-events channel with this process as its only worker: each
    event runs its handler here and answers as one acknowledgement."""

    worker_id = "unit-worker"

    def __init__(self, handlers):
        self._handlers = handlers
        self.published: list = []

    async def publish(self, kind, payload, *, timeout_s):
        import asyncio

        self.published.append((kind, payload))
        return {self.worker_id: await asyncio.to_thread(self._handlers[kind], payload)}


@pytest.fixture
def in_process_cluster_events(monkeypatch):
    """Wire tenant deletes, tier sets and session closes to an in-process
    channel."""
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.routers import admin

    events = InProcessClusterEvents(
        {
            "tenant_deleted": tenant_manager.release_deleted_tenant,
            "tenant_tier_set": tenant_manager.release_tenant_tier,
            "session_closed": admin.sweep_closed_session,
        }
    )
    monkeypatch.setattr(tenant_manager, "_cluster_events", events)
    monkeypatch.setattr(admin, "_cluster_events", events)
    return events


@pytest.fixture
async def tenant_task_events(monkeypatch, shared_state_redis):
    """Wire tenant deletes to a task event store on the test Redis, under
    keys of their own."""
    import uuid

    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.task_events import TaskEventStore

    prefix = f"test:task-events:{uuid.uuid4().hex}"
    store = TaskEventStore(
        shared_state_redis,
        key_prefix=prefix,
        ingestion_stream_prefix=f"{prefix}:ingest:",
    )
    monkeypatch.setattr(tenant_manager, "_task_events", store)
    return store
