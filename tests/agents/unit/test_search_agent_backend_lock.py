"""Concurrent first-touches of _get_backend resolve to one built instance.

SearchAgent is cross-tenant shared and _get_backend is reached from
asyncio.to_thread OS threads. The agent keeps no handle — the registry owns
the instance's lifetime and closes what it evicts — so every touch resolves
through ``get_search_backend``. The registry's ``set_if_absent`` is what
funnels N concurrent cold starts into one build and closes the losers, so N
threads arriving together must still leave exactly one built instance with
its Vespa session open, and every thread must get that one.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_agents.search_agent import SearchAgent
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.unified_config import SystemConfig

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_N = 12


def _config_manager():
    """A config manager bound to the endpoint the registry keys instances on."""
    return SimpleNamespace(
        get_system_config=lambda: SystemConfig(
            backend_url="http://localhost", backend_port=8080
        )
    )


def _bare_agent():
    agent = object.__new__(SearchAgent)
    agent._backend_type = "vespa"
    agent._backend_config = {}
    agent.bind_config_manager(_config_manager())
    agent.schema_loader = SimpleNamespace()
    return agent


@pytest.mark.asyncio
async def test_concurrent_get_backend_resolves_to_one_instance(monkeypatch):
    """N threads through the real registry: one build, one instance, no leak."""
    builds: list[object] = []
    count_lock = threading.Lock()

    def _build(*args, **kwargs):
        instance = SimpleNamespace(
            name="shared-backend", closed=False, schema_registry=None
        )
        instance.close = lambda inst=instance: setattr(inst, "closed", True)
        with count_lock:
            builds.append(instance)
        time.sleep(0.03)  # widen the build window
        return instance

    monkeypatch.setattr(
        "cogniverse_core.factories.backend_factory.BackendFactory."
        "create_backend_with_dependencies",
        staticmethod(_build),
    )
    monkeypatch.setitem(BackendRegistry._search_backends, "vespa", SimpleNamespace)
    BackendRegistry.clear_instances()

    agent = _bare_agent()
    monkeypatch.setattr(
        SearchAgent,
        "_get_backend",
        lambda self: BackendRegistry.get_search_backend(
            "vespa",
            {"url": "http://localhost", "port": 8080},
            config_manager=self.config_manager,
            schema_loader=self.schema_loader,
        ),
    )

    barrier = threading.Barrier(_N)

    def _call():
        # All N threads arrive together, THEN hit the cache miss at once.
        barrier.wait(timeout=5)
        return agent._get_backend()

    # asyncio.to_thread shares the default executor, whose worker ceiling
    # (cpu_count + 4) is below _N on small CI hosts and starves the barrier.
    loop = asyncio.get_running_loop()
    try:
        with ThreadPoolExecutor(max_workers=_N) as pool:
            results = await asyncio.gather(
                *(loop.run_in_executor(pool, _call) for _ in range(_N))
            )

        winner = BackendRegistry._backend_instances.get(
            "search_vespa@http://localhost:8080"
        )
        # Every thread got the one cached instance...
        assert results == [winner] * _N
        # ...and every candidate the race built but did not cache was closed,
        # so no losing Vespa session leaks.
        assert [b.closed for b in builds] == [b is not winner for b in builds]
        assert builds.count(winner) == 1
    finally:
        BackendRegistry.clear_instances()


@pytest.mark.asyncio
async def test_get_backend_holds_no_instance_between_calls():
    """The removed holder: no attribute survives a call to go stale."""
    agent = _bare_agent()
    assert hasattr(agent, "_shared_backend") is False
    assert hasattr(agent, "_shared_backend_lock") is False
