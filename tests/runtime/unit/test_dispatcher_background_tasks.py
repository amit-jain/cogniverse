"""Background tasks spawned by AgentDispatcher must be strong-referenced.

CPython's ``asyncio.create_task`` documents that the runtime keeps only a
weak reference to the task: if the caller does not hold a strong ref,
the coroutine may be garbage-collected before it runs. The dispatcher
spawns wiki auto-file work on every dispatch and previously dropped the
``create_task`` return value.
"""

from __future__ import annotations

import asyncio

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher


def _bare_dispatcher() -> AgentDispatcher:
    # Bypass __init__ — it needs an AgentRegistry, ConfigManager, SchemaLoader.
    # Just the background-task plumbing is exercised here.
    d = object.__new__(AgentDispatcher)
    d._background_tasks = set()
    return d


@pytest.mark.asyncio
async def test_spawn_background_strong_references_until_complete() -> None:
    """The real AgentDispatcher._spawn_background must hold the task until it
    completes, then release it via the done callback."""
    d = _bare_dispatcher()

    async def long_work():
        await asyncio.sleep(0.05)
        return "done"

    task = d._spawn_background(long_work())

    # While running, the dispatcher holds a strong reference.
    assert task in d._background_tasks
    await task
    # After completion the discard callback runs.
    await asyncio.sleep(0)
    assert task not in d._background_tasks
    assert task.result() == "done"


@pytest.mark.asyncio
async def test_spawn_background_survives_gc_pressure() -> None:
    """A task whose only reference is the dispatcher's set must still run to
    completion under GC pressure."""
    d = _bare_dispatcher()
    called = asyncio.Event()

    async def fake_wiki_work():
        called.set()

    d._spawn_background(fake_wiki_work())
    # Drop our local handle entirely; only the dispatcher's set holds it.
    import gc

    gc.collect()

    await asyncio.wait_for(called.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert d._background_tasks == set()


@pytest.mark.asyncio
async def test_phoenix_provider_spawn_background_tracks_and_releases() -> None:
    """PhoenixEvaluationProvider._spawn_background tracks then releases."""
    from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
        PhoenixEvaluationProvider,
    )

    provider = object.__new__(PhoenixEvaluationProvider)
    provider._background_tasks = set()

    async def fake_annotate():
        await asyncio.sleep(0.01)

    task = provider._spawn_background(fake_annotate())
    assert task in provider._background_tasks
    await task
    await asyncio.sleep(0)
    assert provider._background_tasks == set()


# ---------------------------------------------------------------------------
# Naive datetime sweep regression — agent registry, workflow intelligence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_registry_health_check_writes_aware_utc() -> None:
    """``AgentRegistry.health_check_agent`` must stamp the registered
    endpoint's ``last_health_check`` as UTC-aware so cross-pod compares work."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    from cogniverse_core.common.agent_models import AgentEndpoint
    from cogniverse_core.registries.agent_registry import AgentRegistry

    registry = AgentRegistry(tenant_id="acme:production", config_manager=MagicMock())
    ep = AgentEndpoint(name="x", url="http://x", capabilities=["video_search"])
    assert registry.register_agent(ep) is True
    assert ep.last_health_check is None

    response = MagicMock(status_code=200)
    registry._http_client = MagicMock()
    registry._http_client.get = AsyncMock(return_value=response)

    before = datetime.now(timezone.utc)
    healthy = await registry.health_check_agent("x")
    after = datetime.now(timezone.utc)

    assert healthy is True
    registry._http_client.get.assert_awaited_once_with("http://x/health", timeout=5.0)
    stored = registry.agents["x"]
    assert stored is ep
    assert stored.health_status == "healthy"
    assert stored.last_health_check is not None
    assert stored.last_health_check.tzinfo is timezone.utc
    assert before <= stored.last_health_check <= after


@pytest.mark.asyncio
async def test_agent_registry_health_check_failure_writes_aware_utc() -> None:
    """An unreachable agent still gets a UTC-aware ``last_health_check``."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from cogniverse_core.common.agent_models import AgentEndpoint
    from cogniverse_core.registries.agent_registry import AgentRegistry

    registry = AgentRegistry(tenant_id="acme:production", config_manager=MagicMock())
    ep = AgentEndpoint(name="x", url="http://x", capabilities=["video_search"])
    assert registry.register_agent(ep) is True

    registry._http_client = MagicMock()
    registry._http_client.get = AsyncMock(
        side_effect=httpx.TimeoutException("health probe timed out")
    )

    before = datetime.now(timezone.utc)
    healthy = await registry.health_check_agent("x")
    after = datetime.now(timezone.utc)

    assert healthy is False
    stored = registry.agents["x"]
    assert stored.health_status == "unreachable"
    assert stored.last_health_check.tzinfo is timezone.utc
    assert before <= stored.last_health_check <= after


def test_agent_endpoint_needs_health_check_tolerates_legacy_naive_stamp() -> None:
    """A registry written before the aware-utc switch must not crash."""
    from datetime import datetime, timedelta

    from cogniverse_core.common.agent_models import AgentEndpoint

    ep = AgentEndpoint(name="x", url="http://x", capabilities=[])
    ep.last_health_check = datetime.utcnow() - timedelta(seconds=120)
    # Must not raise ``TypeError: can't compare offset-naive and offset-aware``.
    needs = ep.needs_health_check()
    assert isinstance(needs, bool)
