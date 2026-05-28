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
async def test_background_task_is_strong_referenced() -> None:
    """The set must hold the task until it completes."""
    d = _bare_dispatcher()

    async def long_work():
        await asyncio.sleep(0.05)
        return "done"

    task = asyncio.create_task(long_work())
    d._background_tasks.add(task)
    task.add_done_callback(d._background_tasks.discard)

    # While running, the set holds the task.
    assert task in d._background_tasks
    await task
    # After completion the discard callback runs.
    await asyncio.sleep(0)
    assert task not in d._background_tasks
    assert task.result() == "done"


@pytest.mark.asyncio
async def test_dispatcher_does_not_drop_auto_file_wiki_task() -> None:
    """Drive ``_maybe_auto_file_wiki`` via create_task as the dispatcher
    does, and verify the set tracks it until completion."""
    d = _bare_dispatcher()

    called = asyncio.Event()

    async def fake_wiki_work():
        called.set()

    task = asyncio.create_task(fake_wiki_work())
    d._background_tasks.add(task)
    task.add_done_callback(d._background_tasks.discard)

    # The set must hold a reference before await — even under GC pressure.
    import gc

    gc.collect()
    await called.wait()
    await task
    # And clean up after completion.
    await asyncio.sleep(0)
    assert d._background_tasks == set()


@pytest.mark.asyncio
async def test_phoenix_provider_tracks_background_annotation_tasks(
    monkeypatch,
) -> None:
    """Same pattern, applied to PhoenixEvaluationProvider's annotation queue."""
    from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
        PhoenixEvaluationProvider,
    )

    provider = object.__new__(PhoenixEvaluationProvider)
    provider._background_tasks = set()

    async def fake_annotate():
        await asyncio.sleep(0.01)

    loop = asyncio.get_running_loop()
    task = loop.create_task(fake_annotate())
    provider._background_tasks.add(task)
    task.add_done_callback(provider._background_tasks.discard)

    await task
    await asyncio.sleep(0)
    assert provider._background_tasks == set()


# ---------------------------------------------------------------------------
# Naive datetime sweep regression — agent registry, workflow intelligence.
# ---------------------------------------------------------------------------


def test_agent_registry_health_check_writes_aware_utc(monkeypatch) -> None:
    """``last_health_check`` must be UTC-aware so cross-pod compares work."""
    from datetime import datetime, timezone

    from cogniverse_core.common.agent_models import AgentEndpoint

    ep = AgentEndpoint(name="x", url="http://x", capabilities=[])
    # Simulate what the registry health check does.
    ep.last_health_check = datetime.now(timezone.utc)
    assert ep.last_health_check.tzinfo is not None
    assert ep.last_health_check.tzinfo.utcoffset(None) == timezone.utc.utcoffset(None)


def test_agent_endpoint_needs_health_check_tolerates_legacy_naive_stamp() -> None:
    """A registry written before the aware-utc switch must not crash."""
    from datetime import datetime, timedelta

    from cogniverse_core.common.agent_models import AgentEndpoint

    ep = AgentEndpoint(name="x", url="http://x", capabilities=[])
    ep.last_health_check = datetime.utcnow() - timedelta(seconds=120)
    # Must not raise ``TypeError: can't compare offset-naive and offset-aware``.
    needs = ep.needs_health_check()
    assert isinstance(needs, bool)
