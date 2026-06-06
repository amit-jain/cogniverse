"""``_evict_old_tracers`` must bound the provider cache too, not just tracers.

Before the fix the provider map grew without limit (tracers were LRU-capped
but their backing ``TracerProvider`` entries were never dropped). Eviction now
drops providers no remaining tracer references and flushes them via
``shutdown()`` first.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cogniverse_foundation.telemetry.manager import TelemetryManager


def _manager(max_cached_tenants: int) -> TelemetryManager:
    # Bypass the singleton/__init__; _evict_old_tracers only touches the caches
    # and config.max_cached_tenants.
    m = object.__new__(TelemetryManager)
    m.config = SimpleNamespace(max_cached_tenants=max_cached_tenants)
    m._tenant_tracers = {}
    m._tenant_providers = {}
    m._tracer_provider_keys = {}
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
