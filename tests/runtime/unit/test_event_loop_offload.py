"""Sync Vespa I/O on async paths runs off the event loop.

The wiki routes, the ingestion worker's graph upsert, and the lifecycle
scheduler's pin lookup call synchronous manager methods that do blocking Vespa
round-trips (the graph feed can even ``time.sleep`` during a convergence race).
Run inline, each stalls the whole loop. These tests prove a concurrent ticker
keeps firing while the blocking call is in flight — i.e. it was offloaded.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


async def _ticks_during(slow_coro_factory) -> int:
    """Run slow_coro_factory() while a 10ms ticker counts; return tick count.

    If the awaited work blocks the loop, the ticker is starved (~0-1 ticks);
    if it was offloaded via to_thread, the ticker keeps firing (many ticks).
    """
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await slow_coro_factory()
    stop.set()
    await t
    return ticks


def _blocking(duration: float):
    def _fn(*a, **k):
        time.sleep(duration)
        return MagicMock()

    return _fn


@pytest.mark.asyncio
async def test_wiki_search_route_offloads_blocking_search(monkeypatch):
    from cogniverse_runtime.routers import wiki

    wm = MagicMock()
    wm.search = _blocking(0.3)
    monkeypatch.setattr(wiki, "get_wiki_manager_for_tenant", lambda t: wm)
    req = wiki.WikiSearchRequest(query="q", tenant_id="acme:acme", top_k=5)

    ticks = await _ticks_during(lambda: wiki.search_wiki(req))

    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s wiki search — the blocking Vespa "
        "search ran on the event loop"
    )


@pytest.mark.asyncio
async def test_lifecycle_pin_lookup_offloaded(monkeypatch):
    from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler

    warm = MagicMock()
    warm.tenant_id = "acme:acme"
    warm.cleanup_with_schema.return_value = {}

    sched = LifecycleScheduler(
        get_warm_managers=lambda: [warm],
        registry=MagicMock(),
        pin_lookup=_blocking(0.3),
    )

    ticks = await _ticks_during(sched.tick_once)

    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s pin lookup — it ran on the loop"
    )
