"""Span leases across provider retirement.

A tracer is resolved under the manager lock but the span it starts is started
after that lock is released: ``TelemetryManager.get_tracer`` hands the tracer
to the caller, and ``TenantRoutingTracerProvider`` resolves one per span. The
provider behind that tracer can be retired and drained in between, so the lease
processor must not assume its provider still holds a lease entry, and the drain
must not wait on a lease that never comes back.
"""

from __future__ import annotations

import logging
import threading

import pytest
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import (
    TelemetryManager,
    _ProviderSpanLease,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture(autouse=True)
def _reset_manager():
    TelemetryManager.reset()
    yield
    TelemetryManager.reset()


def _manager(lease_timeout: float = 30.0) -> TelemetryManager:
    return TelemetryManager(
        TelemetryConfig(
            max_cached_tenants=1,
            tenant_cache_ttl_seconds=0,
            retirement_lease_timeout_seconds=lease_timeout,
        )
    )


def _leased_provider(manager: TelemetryManager, key: str) -> SdkTracerProvider:
    """A cached provider wired exactly as the manager wires an optional one."""
    provider = SdkTracerProvider()
    manager._provider_leases[provider] = 0
    SdkTracerProvider.add_span_processor(
        provider, _ProviderSpanLease(manager, provider)
    )
    manager._tenant_providers[key] = provider
    manager._tenant_tracers[key] = provider.get_tracer(key)
    manager._tracer_provider_keys[key] = key
    manager._tracer_created_at[key] = 0.0
    return provider


def _await_drain(manager: TelemetryManager, timeout: float = 10.0) -> None:
    with manager._retirement_condition:
        assert (
            manager._retirement_condition.wait_for(
                lambda: not manager._retired_providers, timeout=timeout
            )
            is True
        )


class TestLeaseSurvivesDrainBetweenResolveAndStart:
    def test_span_started_on_a_drained_provider_records_no_lease(self):
        """The interleaving executed: resolve the tracer, let the drain finish,
        then start the span the caller was going to start."""
        manager = _manager()
        provider = _leased_provider(manager, "t1:proj")
        tracer = manager._tenant_tracers["t1:proj"]

        manager.config.max_cached_tenants = 0
        manager._evict_old_tracers()
        _await_drain(manager)
        assert provider not in manager._provider_leases

        with tracer.start_as_current_span("late") as span:
            assert span.name == "late"
            assert manager._provider_leases == {}

        assert manager._provider_leases == {}
        assert manager._retired_providers == {}
        assert manager.get_stats()["retired_providers"] == 0

    def test_concurrent_resolve_and_retire_never_raises(self):
        """Sixteen threads start spans on tracers they resolved while another
        thread retires every provider under them."""
        manager = _manager()
        errors: list[BaseException] = []
        start = threading.Barrier(17)
        keys = [f"t{index}:proj" for index in range(16)]
        providers = [_leased_provider(manager, key) for key in keys]
        tracers = [manager._tenant_tracers[key] for key in keys]

        def record(tracer):
            try:
                start.wait(timeout=10)
                with tracer.start_as_current_span("dispatch"):
                    pass
            except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                errors.append(exc)

        def retire():
            try:
                start.wait(timeout=10)
                manager.config.max_cached_tenants = 0
                manager._evict_old_tracers()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=record, args=(t,)) for t in tracers]
        threads.append(threading.Thread(target=retire))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert errors == []
        _await_drain(manager)
        assert manager._provider_leases == {}
        assert manager._retired_providers == {}
        assert manager._draining_providers == set()
        assert manager.get_stats()["retirement_workers"] == 0
        assert [provider for provider in providers if provider is None] == []


class TestLeaseWaitIsBounded:
    def test_span_that_never_ends_releases_the_retirement_slot(self, caplog):
        manager = _manager(lease_timeout=0.2)
        provider = _leased_provider(manager, "t1:proj")
        tracer = manager._tenant_tracers["t1:proj"]

        leaked = tracer.start_span("never-ended")
        assert manager._provider_leases[provider] == 1

        manager.config.max_cached_tenants = 0
        with caplog.at_level(logging.WARNING):
            manager._evict_old_tracers()
            _await_drain(manager)

        assert manager._retired_providers == {}
        assert manager._provider_leases == {}
        assert manager._draining_providers == set()
        assert manager.get_stats()["retired_providers"] == 0
        assert [
            record.getMessage()
            for record in caplog.records
            if "retirement deadline" in record.getMessage()
        ] == [
            "Telemetry provider retirement deadline passed with 1 span(s) still "
            "recording; shutting down provider=t1:proj"
        ]

        # The leaked span ending after its provider was dropped must not raise
        # and must not resurrect a lease entry.
        leaked.end()
        assert manager._provider_leases == {}

    def test_capacity_is_released_for_the_next_tenant(self):
        """At capacity a stuck retirement stops tracer creation process-wide;
        the deadline must hand the slot back."""
        manager = _manager(lease_timeout=0.2)
        _leased_provider(manager, "t1:proj")
        tracer = manager._tenant_tracers["t1:proj"]
        tracer.start_span("never-ended")

        manager.config.max_cached_tenants = 0
        manager._evict_old_tracers()
        manager.config.max_cached_tenants = 1
        assert manager._can_create_provider() is False

        _await_drain(manager)
        assert manager._can_create_provider() is True
