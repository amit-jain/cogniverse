"""Singleton lifecycle of ``get_telemetry_manager`` across ``reset()``.

``reset()`` shuts the instance down and clears the class singleton; it must
also clear the module-global ``get_telemetry_manager`` short-circuits on, or a
later ``get_telemetry_manager()`` hands back the already-shut-down instance
instead of rebuilding. The rebuild path is a process-global cold build, so it
must also be single-flight under a race.
"""

from __future__ import annotations

import threading

import pytest

from cogniverse_foundation.telemetry import manager as manager_mod
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import (
    TelemetryManager,
    get_telemetry_manager,
)


@pytest.fixture(autouse=True)
def _cold_singleton(monkeypatch):
    """Force a cold singleton before and after each test, offline."""
    monkeypatch.delenv("TELEMETRY_OTLP_ENDPOINT", raising=False)
    TelemetryManager.reset()
    manager_mod._telemetry_manager = None
    yield
    TelemetryManager.reset()
    manager_mod._telemetry_manager = None


class _StubConfigManager:
    """Returns an offline telemetry config; counts config loads."""

    def __init__(self) -> None:
        self.load_count = 0

    def get_telemetry_config(self, tenant_id: str) -> TelemetryConfig:
        self.load_count += 1
        return TelemetryConfig(enabled=True, otlp_enabled=False, provider=None)


def test_reset_forces_get_to_rebuild_a_live_instance() -> None:
    cfg_mgr = _StubConfigManager()

    m1 = get_telemetry_manager(cfg_mgr)
    m1._project_configs["proj-a"] = {"marker": 1}

    TelemetryManager.reset()
    m2 = get_telemetry_manager(cfg_mgr)

    # A brand-new instance, not the shut-down one reset() cleared from the global.
    assert m2 is not m1
    assert isinstance(m2, TelemetryManager)
    # The rebuild is live and fresh: initialized, empty project registry.
    assert m2._initialized is True
    assert m2._project_configs == {}
    # get_telemetry_manager rebuilt (loaded config) exactly twice: once per get.
    assert cfg_mgr.load_count == 2


def test_concurrent_get_after_reset_builds_exactly_one() -> None:
    # Prime then reset so the concurrent gets race a genuine cold build.
    priming = _StubConfigManager()
    get_telemetry_manager(priming)
    TelemetryManager.reset()

    n = 16
    barrier = threading.Barrier(n)
    cfg_mgr = _StubConfigManager()
    results: list[TelemetryManager | None] = [None] * n

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = get_telemetry_manager(cfg_mgr)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread ran the cold build; the rest reused the global.
    assert cfg_mgr.load_count == 1
    # Every racer observed the same single live instance.
    assert all(r is not None for r in results)
    assert len({id(r) for r in results}) == 1
