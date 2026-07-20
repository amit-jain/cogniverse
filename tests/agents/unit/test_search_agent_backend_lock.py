"""_get_backend builds the shared backend once under concurrent thread touches.

SearchAgent is cross-tenant shared and _get_backend is reached from
asyncio.to_thread OS threads. Without the lock, N concurrent first-touches each
call registry.get_search_backend — each builds a backend candidate and N-1 lose
set_if_absent and are dropped with their Vespa session still open. The lock must
funnel them into a single registry build.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from cogniverse_agents.search_agent import SearchAgent

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_N = 12


@pytest.mark.asyncio
async def test_concurrent_get_backend_builds_once(monkeypatch):
    import cogniverse_agents.search_agent as sa_mod

    builds = {"n": 0}
    count_lock = threading.Lock()
    winner = SimpleNamespace(name="shared-backend")

    def _get_search_backend(
        backend_type, backend_config, *, config_manager, schema_loader
    ):
        with count_lock:
            builds["n"] += 1
        time.sleep(0.03)  # widen the build window
        return winner

    monkeypatch.setattr(
        sa_mod,
        "get_backend_registry",
        lambda: SimpleNamespace(get_search_backend=_get_search_backend),
    )

    agent = object.__new__(SearchAgent)
    agent._shared_backend = None
    agent._shared_backend_lock = threading.Lock()
    agent._backend_type = "vespa"
    agent._backend_config = {}
    agent.config_manager = SimpleNamespace()
    agent.schema_loader = SimpleNamespace()

    barrier = threading.Barrier(_N)

    def _call():
        # All N threads arrive together, THEN hit the None check at once.
        barrier.wait(timeout=5)
        return agent._get_backend()

    results = await asyncio.gather(*(asyncio.to_thread(_call) for _ in range(_N)))

    assert builds["n"] == 1
    assert all(r is winner for r in results)
    assert agent._shared_backend is winner
