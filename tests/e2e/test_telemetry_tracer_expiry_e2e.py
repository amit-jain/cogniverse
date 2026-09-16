"""Tenant tracer expiry inside the deployed runtime.

Pins that retiring a tenant's tracer provider detaches it and drains it on a
background worker: the caller that triggers the expiry returns with the
provider retired rather than shut down, a provider whose spans are still
recording is kept alive by its lease, unrelated tenants keep getting tracers
while that drain is parked, and the runtime keeps answering its liveness probe
throughout.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
import threading
import time

import httpx
import pytest

from tests.e2e.conftest import (
    IN_POD_TELEMETRY_PRELUDE,
    KUBECTL_CONTEXT,
    RUNTIME,
    unique_id,
)

NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"

# Sibling tenants served while the retired provider's drain is parked on its
# lease. Five is enough to show the manager lock is free without lengthening
# the window the health poller has to cover.
SIBLING_TENANTS = 5
HEALTH_POLL_INTERVAL_S = 0.1
IN_POD_TIMEOUT_S = 300


def _expiry_script(tenant_a: str, tenant_b: str) -> str:
    return IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
        f"""\
        import contextlib
        import json
        import time

        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        manager = get_telemetry_manager()
        config = manager.config
        tenant_a = {tenant_a!r}
        tenant_b = {tenant_b!r}
        key_a = tenant_a + ":" + config.get_project_name(tenant_a)
        key_b = tenant_b + ":" + config.get_project_name(tenant_b)

        with manager.span("e2e-tracer-expiry", tenant_id=tenant_b):
            pass

        stack = contextlib.ExitStack()
        stack.enter_context(manager.span("e2e-tracer-expiry", tenant_id=tenant_a))
        provider_a = manager._tenant_providers[key_a]
        providers_before = sorted(manager._tenant_providers)
        leases_while_recording = manager._provider_leases.get(provider_a)

        with manager._lock:
            manager._tracer_created_at[key_a] = time.monotonic() - (
                config.tenant_cache_ttl_seconds + 1
            )

        evict_started = time.monotonic()
        with manager.span("e2e-tracer-expiry", tenant_id=tenant_a):
            pass
        evict_elapsed = time.monotonic() - evict_started

        with manager._lock:
            retired_key = manager._retired_providers.get(provider_a)
            retired_leases = manager._provider_leases.get(provider_a)
            providers_after_expiry = sorted(manager._tenant_providers)
        stats_during_drain = manager.get_stats()

        sibling_started = time.monotonic()
        siblings = [tenant_b + "-s" + str(i) for i in range({SIBLING_TENANTS})]
        for sibling in siblings:
            with manager.span("e2e-tracer-expiry", tenant_id=sibling):
                pass
        sibling_elapsed = time.monotonic() - sibling_started
        with manager._lock:
            still_retired = manager._retired_providers.get(provider_a)

        stack.close()
        with manager._retirement_condition:
            drained = manager._retirement_condition.wait_for(
                lambda: not manager._retired_providers,
                timeout=config.retirement_lease_timeout_seconds * 2,
            )
        deadline = time.monotonic() + 30.0
        while manager.get_stats()["retirement_workers"] and time.monotonic() < deadline:
            time.sleep(0.05)
        stats_after_drain = manager.get_stats()
        with manager._lock:
            leases_after = list(manager._provider_leases.values())
            draining_after = len(manager._draining_providers)
            providers_after_drain = sorted(manager._tenant_providers)

        print("__EXPIRY__" + json.dumps({{
            "key_a": key_a,
            "key_b": key_b,
            "sibling_keys": sorted(
                sibling + ":" + config.get_project_name(sibling)
                for sibling in siblings
            ),
            "lease_timeout_seconds": config.retirement_lease_timeout_seconds,
            "tenant_cache_ttl_seconds": config.tenant_cache_ttl_seconds,
            "leases_while_recording": leases_while_recording,
            "providers_before": providers_before,
            "providers_after_expiry": providers_after_expiry,
            "providers_after_drain": providers_after_drain,
            "retired_key": retired_key,
            "retired_leases": retired_leases,
            "still_retired": still_retired,
            "evict_elapsed": evict_elapsed,
            "sibling_elapsed": sibling_elapsed,
            "drained": drained,
            "stats_during_drain": stats_during_drain,
            "stats_after_drain": stats_after_drain,
            "leases_after": leases_after,
            "draining_after": draining_after,
        }}))
        """
    )


def _run_in_pod(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=IN_POD_TIMEOUT_S,
    )


class _LivenessPoller:
    """Polls the runtime's liveness probe from the host until stopped."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.samples: list[tuple[float, int]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with httpx.Client(timeout=10.0) as client:
            while not self._stop.is_set():
                status = client.get(f"{RUNTIME}/health/live").status_code
                self.samples.append((time.monotonic(), status))
                self._stop.wait(HEALTH_POLL_INTERVAL_S)

    def __enter__(self) -> "_LivenessPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=30.0)


@pytest.mark.e2e
class TestTenantTracerExpiry:
    """Expiry detaches the provider; a background worker drains it."""

    def test_expiry_detaches_the_provider_and_keeps_the_runtime_answering(self):
        tenant_a = unique_id("opt_trexp") + ":t1"
        tenant_b = unique_id("opt_trexp") + ":t2"

        with _LivenessPoller() as poller:
            started = time.monotonic()
            result = _run_in_pod(_expiry_script(tenant_a, tenant_b))
            finished = time.monotonic()
        assert result.returncode == 0, result.stderr[-4000:]
        line = next(
            ln for ln in result.stdout.splitlines() if ln.startswith("__EXPIRY__")
        )
        report = json.loads(line[len("__EXPIRY__") :])

        # The span held open while the provider was retired kept exactly one
        # lease on it, which is what stops the drain from shutting it down.
        assert report["leases_while_recording"] == 1, report
        assert report["retired_key"] == report["key_a"], report
        assert report["retired_leases"] == 1, report

        # The expiry replaced the tenant's provider rather than shutting the
        # old one down on the caller's thread.
        assert report["providers_before"] == sorted(
            [report["key_a"], report["key_b"]]
        ), report
        assert report["providers_after_expiry"] == sorted(
            [report["key_a"], report["key_b"]]
        ), report
        assert report["stats_during_drain"]["retired_providers"] == 1, report
        assert report["stats_during_drain"]["retirement_workers"] == 1, report
        assert report["evict_elapsed"] < report["lease_timeout_seconds"], report

        # Unrelated tenants were served while that drain was parked: the drain
        # does not hold the manager lock.
        assert report["providers_after_drain"] == sorted(
            [report["key_a"], report["key_b"], *report["sibling_keys"]]
        ), report
        assert report["still_retired"] == report["key_a"], report
        assert report["sibling_elapsed"] < report["lease_timeout_seconds"], report

        # Ending the recording span released the lease and the drain finished,
        # leaving no retired provider, no lease and no worker behind.
        assert report["drained"] is True, report
        assert report["leases_after"] == [0] * len(report["providers_after_drain"]), (
            report
        )
        assert report["draining_after"] == 0, report
        assert report["stats_after_drain"]["retired_providers"] == 0, report
        assert report["stats_after_drain"]["retirement_workers"] == 0, report

        # The serving process answered every poll taken across the retirement,
        # and the polls bracket the retirement rather than following it.
        assert {status for _, status in poller.samples} == {200}, poller.samples
        assert poller.samples[0][0] < finished, poller.samples[0]
        assert poller.samples[-1][0] > started, poller.samples[-1]
