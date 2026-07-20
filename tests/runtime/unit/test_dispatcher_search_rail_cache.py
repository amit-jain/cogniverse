"""The thread-invoked search-agent and rail-chain caches build once per key.

Both ``_get_search_agent`` and ``_get_rail_chains`` are called from
``asyncio.to_thread`` OS threads, not the event loop, so their per-key
cold-builds are guarded by ``threading.Lock`` (an asyncio Future guard, as the
gateway/orchestrator caches use, would not serialize concurrent threads). N
concurrent first-touches for one key must run exactly ONE build — otherwise each
re-runs a heavyweight SearchAgent init (encoder + Vespa read) or a rails
get_config Vespa read.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_N = 12


def _make_dispatcher():
    d = object.__new__(AgentDispatcher)
    d._search_agent_cache = {}
    d._search_agent_cache_lock = threading.Lock()
    d._rail_chains_cache = {}
    d._rail_chains_cache_lock = threading.Lock()
    d._schema_loader = SimpleNamespace()
    d._config_manager = SimpleNamespace(
        get_system_config=lambda: SimpleNamespace(
            backend_url="http://vespa", backend_port=8080
        )
    )
    return d


@pytest.mark.asyncio
async def test_concurrent_thread_first_touches_build_search_agent_once(monkeypatch):
    import cogniverse_agents.search_agent as sa_mod

    builds = {"n": 0}
    count_lock = threading.Lock()

    class _StubSearchAgent:
        def __init__(self, **kwargs):
            with count_lock:
                builds["n"] += 1
            time.sleep(0.03)  # widen the build window

    monkeypatch.setattr(sa_mod, "SearchAgent", _StubSearchAgent)
    monkeypatch.setattr(sa_mod, "SearchAgentDeps", lambda **k: SimpleNamespace(**k))

    d = _make_dispatcher()
    barrier = threading.Barrier(_N)

    def _call():
        # All N threads arrive together, THEN hit the cache miss at once — the
        # lock must funnel them into a single build.
        barrier.wait(timeout=5)
        return d._get_search_agent("profile_x")

    results = await asyncio.gather(*(asyncio.to_thread(_call) for _ in range(_N)))

    assert builds["n"] == 1
    assert all(r is results[0] for r in results)
    assert d._search_agent_cache["profile_x"] is results[0]


@pytest.mark.asyncio
async def test_concurrent_thread_first_touches_build_rail_chains_once(monkeypatch):
    import cogniverse_foundation.config.utils as cfg_utils

    reads = {"n": 0}
    count_lock = threading.Lock()

    def _fake_get_config(tenant_id, config_manager):
        with count_lock:
            reads["n"] += 1
        time.sleep(0.03)  # widen the read window
        return {}  # no rails block -> chains is None, cached as None

    monkeypatch.setattr(cfg_utils, "get_config", _fake_get_config)

    d = _make_dispatcher()
    barrier = threading.Barrier(_N)

    def _call():
        barrier.wait(timeout=5)
        return d._get_rail_chains("acme:acme")

    results = await asyncio.gather(*(asyncio.to_thread(_call) for _ in range(_N)))

    # One Vespa get_config read for the tenant, and every caller sees the same
    # (None) cached result.
    assert reads["n"] == 1
    assert all(r is None for r in results)
    assert "acme:acme" in d._rail_chains_cache
