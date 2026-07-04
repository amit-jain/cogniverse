"""Tracer-cache eviction: LRU count cap, orphaned-provider cleanup, and TTL.

``_evict_old_tracers`` must bound the provider cache too, not just tracers
(before the fix the provider map grew without limit). Cached entries must
also expire after ``tenant_cache_ttl_seconds`` — the TTL was documented on
the config but eviction was count-based only, so a tracer built once was
served forever regardless of age.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from cogniverse_foundation.telemetry import manager as manager_mod
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager


def _manager(max_cached_tenants: int) -> TelemetryManager:
    # Bypass the singleton/__init__; _evict_old_tracers only touches the caches
    # and config.max_cached_tenants.
    m = object.__new__(TelemetryManager)
    m.config = SimpleNamespace(max_cached_tenants=max_cached_tenants)
    m._tenant_tracers = {}
    m._tenant_providers = {}
    m._tracer_provider_keys = {}
    m._tracer_created_at = {}
    return m


def test_orphaned_provider_is_shutdown_and_dropped():
    m = _manager(max_cached_tenants=1)
    p_old, p_new = MagicMock(), MagicMock()
    m._tenant_providers = {"t1:proj": p_old, "t2:proj": p_new}
    m._tenant_tracers = {"t1:proj": MagicMock(), "t2:proj": MagicMock()}
    m._tracer_provider_keys = {"t1:proj": "t1:proj", "t2:proj": "t2:proj"}

    m._evict_old_tracers()

    assert "t1:proj" not in m._tenant_tracers
    assert "t1:proj" not in m._tenant_providers
    assert "t1:proj" not in m._tracer_provider_keys
    p_old.shutdown.assert_called_once()
    # The still-referenced provider is untouched.
    assert "t2:proj" in m._tenant_providers
    p_new.shutdown.assert_not_called()


def test_shared_tenant_provider_survives_until_last_tracer():
    # One provider keyed by a colon-bearing tenant_id, shared by two projects.
    m = _manager(max_cached_tenants=1)
    provider = MagicMock()
    m._tenant_providers = {"acme:prod": provider}
    m._tenant_tracers = {
        "acme:prod:projA": MagicMock(),
        "acme:prod:projB": MagicMock(),
    }
    m._tracer_provider_keys = {
        "acme:prod:projA": "acme:prod",
        "acme:prod:projB": "acme:prod",
    }

    # Evict the oldest tracer; the provider is still referenced by projB.
    m._evict_old_tracers()
    assert "acme:prod:projA" not in m._tenant_tracers
    assert "acme:prod" in m._tenant_providers
    provider.shutdown.assert_not_called()

    # Drop the last tracer too — now the provider is orphaned.
    m.config.max_cached_tenants = 0
    m._evict_old_tracers()
    assert m._tenant_tracers == {}
    assert "acme:prod" not in m._tenant_providers
    provider.shutdown.assert_called_once()


def test_no_eviction_keeps_everything():
    m = _manager(max_cached_tenants=10)
    provider = MagicMock()
    m._tenant_providers = {"t1:proj": provider}
    m._tenant_tracers = {"t1:proj": MagicMock()}
    m._tracer_provider_keys = {"t1:proj": "t1:proj"}

    m._evict_old_tracers()

    assert "t1:proj" in m._tenant_tracers
    assert "t1:proj" in m._tenant_providers
    provider.shutdown.assert_not_called()


def _live_manager(ttl_seconds: int, max_cached_tenants: int = 10) -> TelemetryManager:
    """Manager with real config + lock, exercising the full
    ``_get_tracer_for_project`` cache path against stubbed providers."""
    m = object.__new__(TelemetryManager)
    m.config = TelemetryConfig(
        tenant_cache_ttl_seconds=ttl_seconds,
        max_cached_tenants=max_cached_tenants,
    )
    m._tenant_providers = {}
    m._tenant_tracers = {}
    m._tracer_provider_keys = {}
    m._tracer_created_at = {}
    m._lock = threading.RLock()
    m._project_configs = {}
    m._cache_hits = 0
    m._cache_misses = 0
    m._failed_initializations = 0
    return m


def _stub_provider_factory(m: TelemetryManager) -> list:
    """Each provider creation yields a distinct provider + tracer."""
    providers = []

    def _mk(tenant_id, project_name):
        p = MagicMock(name=f"provider{len(providers)}")
        p.get_tracer.return_value = MagicMock(name=f"tracer{len(providers)}")
        providers.append(p)
        return p

    m._create_tenant_provider_for_project = MagicMock(side_effect=_mk)
    return providers


def _freeze_clock(monkeypatch, start: float = 1000.0) -> dict:
    clock = {"now": start}
    monkeypatch.setattr(
        manager_mod, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    return clock


def test_entry_older_than_ttl_is_rebuilt(monkeypatch):
    clock = _freeze_clock(monkeypatch)
    m = _live_manager(ttl_seconds=3600)
    providers = _stub_provider_factory(m)

    first = m._get_tracer_for_project("acme", "search")
    assert first is providers[0].get_tracer.return_value
    assert m._cache_misses == 1

    clock["now"] += 3601
    second = m._get_tracer_for_project("acme", "search")

    assert len(providers) == 2
    assert second is providers[1].get_tracer.return_value
    assert second is not first
    # Stale provider was flushed via shutdown() and replaced.
    providers[0].shutdown.assert_called_once()
    providers[1].shutdown.assert_not_called()
    assert m._tenant_providers == {"acme:cogniverse-acme-search": providers[1]}
    assert m._tracer_created_at == {"acme:cogniverse-acme-search": clock["now"]}
    assert m._cache_hits == 0
    assert m._cache_misses == 2


def test_entry_younger_than_ttl_is_reused(monkeypatch):
    clock = _freeze_clock(monkeypatch)
    m = _live_manager(ttl_seconds=3600)
    providers = _stub_provider_factory(m)

    first = m._get_tracer_for_project("acme", "search")
    clock["now"] += 3599
    second = m._get_tracer_for_project("acme", "search")

    assert second is first
    assert len(providers) == 1
    providers[0].shutdown.assert_not_called()
    assert m._cache_hits == 1
    assert m._cache_misses == 1
    # Reuse does not refresh the insert timestamp.
    assert m._tracer_created_at == {"acme:cogniverse-acme-search": 1000.0}


def test_ttl_zero_disables_expiry(monkeypatch):
    clock = _freeze_clock(monkeypatch)
    m = _live_manager(ttl_seconds=0)
    providers = _stub_provider_factory(m)

    first = m._get_tracer_for_project("acme", "search")
    clock["now"] += 10_000_000
    second = m._get_tracer_for_project("acme", "search")

    assert second is first
    assert len(providers) == 1
    assert m._cache_hits == 1


def test_count_cap_still_evicts_with_ttl_active(monkeypatch):
    _freeze_clock(monkeypatch)
    m = _live_manager(ttl_seconds=3600, max_cached_tenants=1)
    providers = _stub_provider_factory(m)

    m._get_tracer_for_project("acme", "search")
    m._get_tracer_for_project("acme", "routing")

    assert set(m._tenant_tracers) == {"acme:cogniverse-acme-routing"}
    assert set(m._tracer_created_at) == {"acme:cogniverse-acme-routing"}
    assert set(m._tenant_providers) == {"acme:cogniverse-acme-routing"}
    providers[0].shutdown.assert_called_once()
