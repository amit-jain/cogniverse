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
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_foundation.caching import evict_tenant_from_registered_caches
from cogniverse_runtime.agent_dispatcher import (
    SEARCH_AGENT_CACHE_CAPACITY,
    AgentDispatcher,
    _new_rail_chains_cache,
    _new_search_agent_cache,
    _RailChainsEntry,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_N = 12


def _make_dispatcher():
    d = object.__new__(AgentDispatcher)
    d._search_agent_cache = _new_search_agent_cache()
    d._search_agent_cache_lock = threading.Lock()
    d._rail_chains_cache = _new_rail_chains_cache()
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
        return d._get_search_agent("profile_x", "acme:acme")

    # asyncio.to_thread shares the default executor, whose worker ceiling
    # (cpu_count + 4) is below _N on small CI hosts and starves the barrier.
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=_N) as pool:
        results = await asyncio.gather(
            *(loop.run_in_executor(pool, _call) for _ in range(_N))
        )

    assert builds["n"] == 1
    assert all(r is results[0] for r in results)
    assert d._search_agent_cache.get("acme:acme") == {"profile_x": results[0]}


def test_each_tenant_gets_a_search_agent_built_from_its_own_profiles(monkeypatch):
    """A profile name resolves against the requesting tenant's config, which
    carries that tenant's own profiles, so one tenant's agent never serves
    another's."""
    import cogniverse_agents.search_agent as sa_mod

    class _StubSearchAgent:
        def __init__(self, **kwargs):
            self.deps = kwargs["deps"]

    monkeypatch.setattr(sa_mod, "SearchAgent", _StubSearchAgent)
    monkeypatch.setattr(sa_mod, "SearchAgentDeps", lambda **k: SimpleNamespace(**k))

    d = _make_dispatcher()
    acme = d._get_search_agent("profile_x", "acme:acme")
    beta = d._get_search_agent("profile_x", "beta:beta")

    assert acme is not beta
    assert d._get_search_agent("profile_x", "acme:acme") is acme
    assert vars(acme.deps) == {
        "profile": "profile_x",
        "tenant_id": "acme:acme",
        "backend_url": "http://vespa",
        "backend_port": 8080,
    }
    assert vars(beta.deps)["tenant_id"] == "beta:beta"
    assert d._search_agent_cache.keys() == ["beta:beta", "acme:acme"]
    assert d._search_agent_cache.get("beta:beta") == {"profile_x": beta}


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

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=_N) as pool:
        results = await asyncio.gather(
            *(loop.run_in_executor(pool, _call) for _ in range(_N))
        )

    # One Vespa get_config read for the tenant, and every caller sees the same
    # (None) cached result.
    assert reads["n"] == 1
    assert all(r is None for r in results)
    assert d._rail_chains_cache.get("acme:acme") == _RailChainsEntry(None)


def _stub_search_agents(monkeypatch) -> dict:
    import cogniverse_agents.search_agent as sa_mod

    builds = {"n": 0}

    class _StubSearchAgent:
        def __init__(self, **kwargs):
            builds["n"] += 1
            self.deps = kwargs["deps"]

    monkeypatch.setattr(sa_mod, "SearchAgent", _StubSearchAgent)
    monkeypatch.setattr(sa_mod, "SearchAgentDeps", lambda **k: SimpleNamespace(**k))
    return builds


def test_search_agents_are_held_for_a_bounded_number_of_tenants(monkeypatch):
    """Past the capacity the least recently used tenant's agents are dropped,
    and its next search builds them again."""
    builds = _stub_search_agents(monkeypatch)
    d = _make_dispatcher()
    tenants = [f"org{i}:t" for i in range(SEARCH_AGENT_CACHE_CAPACITY + 1)]

    for tenant in tenants:
        d._get_search_agent("profile_x", tenant)

    assert d._search_agent_cache.keys() == tenants[1:]
    d._get_search_agent("profile_x", tenants[0])
    assert builds["n"] == SEARCH_AGENT_CACHE_CAPACITY + 2


def test_a_deleted_tenant_s_search_agents_and_rails_are_released(monkeypatch):
    import cogniverse_foundation.config.utils as cfg_utils

    builds = _stub_search_agents(monkeypatch)
    monkeypatch.setattr(cfg_utils, "get_config", lambda tenant_id, cm: {})
    d = _make_dispatcher()
    for tenant in ("acme:acme", "beta:beta"):
        d._get_search_agent("profile_x", tenant)
        d._get_rail_chains(tenant)

    evict_tenant_from_registered_caches("acme:acme")

    assert (d._search_agent_cache.keys(), d._rail_chains_cache.keys()) == (
        ["beta:beta"],
        ["beta:beta"],
    )
    d._get_search_agent("profile_x", "acme:acme")
    assert builds["n"] == 3


def test_a_rails_config_that_cannot_be_read_raises_and_is_read_again(monkeypatch):
    """A failed read must not leave the tenant without its rails: it raises,
    caches nothing, and the next request compiles the rails it reads."""
    import cogniverse_foundation.config.utils as cfg_utils
    from cogniverse_core.agents.rails import RailBlockedError
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    answers = [
        ConfigStoreUnavailableError("config store did not answer"),
        {
            "rails": {
                "enabled": True,
                "input_rails": [
                    {
                        "type": "content_safety",
                        "params": {"blocked_patterns": ["<script>"]},
                    }
                ],
                "output_rails": [],
            }
        },
    ]

    def _get_config(tenant_id, config_manager):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(cfg_utils, "get_config", _get_config)
    d = _make_dispatcher()

    with pytest.raises(ConfigStoreUnavailableError) as caught:
        d._get_rail_chains("acme:acme")
    assert str(caught.value) == "config store did not answer"
    assert d._rail_chains_cache.keys() == []

    input_chain, _ = d._get_rail_chains("acme:acme")
    with pytest.raises(RailBlockedError) as blocked:
        input_chain.check({"query": "find <script> clips"})
    assert blocked.value.rail_name == "content_safety"
    assert d._get_rail_chains("acme:acme")[0] is input_chain
