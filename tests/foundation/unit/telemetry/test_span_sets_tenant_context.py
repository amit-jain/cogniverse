"""TelemetryManager.span must publish the tenant to the context variable.

DSPy (and any instrumented library) resolves its target project from
``current_tenant_id()``. That only works if entering a tenant span makes the
tenant observable for the span's lifetime -- on every branch, including the
telemetry-disabled no-op path -- and clears it on exit.
"""

from __future__ import annotations

from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.tenant_context import current_tenant_id


def _manager(enabled: bool) -> TelemetryManager:
    TelemetryManager.reset()
    return TelemetryManager(config=TelemetryConfig(enabled=enabled))


def test_span_publishes_tenant_for_its_lifetime_and_clears_it():
    manager = _manager(enabled=True)
    assert current_tenant_id() is None
    with manager.span("cogniverse.profile_selection", tenant_id="acme:production"):
        assert current_tenant_id() == "acme:production"
    assert current_tenant_id() is None
    TelemetryManager.reset()


def test_tenant_is_published_even_when_telemetry_is_disabled():
    # The DSPy provider is independent of this manager's enabled flag; a
    # disabled manager still yields a no-op span, but the tenant MUST be in
    # context so a nested DSPy span routes to the right project.
    manager = _manager(enabled=False)
    with manager.span("cogniverse.gateway", tenant_id="globex:production"):
        assert current_tenant_id() == "globex:production"
    assert current_tenant_id() is None
    TelemetryManager.reset()


def test_nested_tenant_spans_restore_the_outer_tenant():
    manager = _manager(enabled=True)
    with manager.span("outer", tenant_id="acme:production"):
        with manager.span("inner", tenant_id="acme:production"):
            assert current_tenant_id() == "acme:production"
        assert current_tenant_id() == "acme:production"
    assert current_tenant_id() is None
    TelemetryManager.reset()
