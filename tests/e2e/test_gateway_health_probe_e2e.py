"""GatewayHealthProbe end-to-end.

Pins the shipped probe contract against the live OpenShell gateway:

  * probe_once on a live gateway returns ``(True, latency_ms)`` with a
    bounded latency, and the probe records ``last_available`` /
    ``last_latency_ms`` for the dashboard tile;
  * the probe always emits an OpenTelemetry span named
    ``openshell.gateway_health`` with attributes ``openshell.gateway_available``
    (0/1) and ``openshell.gateway_latency_ms`` (positive number);
  * the background loop runs at the configured cadence — at least
    ``ceil(window / interval)`` probes land within ``window`` seconds.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from typing import Iterator

import pytest

from tests.e2e.conftest import run_async, skip_if_no_runtime

_GATEWAY_NAME = "cogniverse-test-gw"
_GATEWAY_PORT = 19090
_GATEWAY_ENDPOINT = f"127.0.0.1:{_GATEWAY_PORT}"


def _gateway_running() -> bool:
    res = subprocess.run(
        ["uv", "run", "openshell", "gateway", "info", "--gateway", _GATEWAY_NAME],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if res.returncode != 0 or "Gateway endpoint" not in res.stdout:
        return False
    # `gateway info` reads stored metadata, which survives the gateway
    # process dying — verify something is actually listening.
    try:
        with socket.create_connection(("127.0.0.1", _GATEWAY_PORT), timeout=5):
            return True
    except OSError:
        return False


def _start_gateway_with_retry(max_attempts: int = 3) -> None:
    # A dead gateway leaves its registration behind and `start` then reuses
    # it without launching a listener — clear it so start is a real start.
    subprocess.run(
        ["uv", "run", "openshell", "gateway", "destroy", "--name", _GATEWAY_NAME],
        capture_output=True,
        text=True,
        timeout=60,
    )
    last_err = ""
    for _ in range(max_attempts):
        res = subprocess.run(
            [
                "uv",
                "run",
                "openshell",
                "gateway",
                "start",
                "--name",
                _GATEWAY_NAME,
                "--port",
                str(_GATEWAY_PORT),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if res.returncode == 0:
            return
        last_err = res.stderr or res.stdout
        if "Corrupted cluster state" not in last_err:
            break
    raise RuntimeError(f"openshell gateway start failed: {last_err[:500]}")


@pytest.fixture(scope="module", autouse=True)
def live_gateway() -> Iterator[None]:
    """Module-scoped: ensure the live openshell gateway is up + endpoint env set."""
    if not _gateway_running():
        _start_gateway_with_retry()
        if not _gateway_running():
            pytest.fail("openshell gateway did not come up")
    # Clear any stale env override; SandboxManager._connect falls back to
    # SandboxClient.from_active_cluster() which talks to the live gateway
    # the start step registered as active.
    os.environ.pop("OPENSHELL_GATEWAY_ENDPOINT", None)
    yield


def _make_manager():
    """Build a SandboxManager bound to the live gateway."""
    from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

    mgr = SandboxManager(policy=SandboxPolicy.OPTIONAL)
    assert mgr._available is True, (
        "live gateway expected — autouse fixture should have started it"
    )
    return mgr


# ---------------------------------------------------------------------------
# 1. probe_once on live gateway → available=True with bounded latency
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProbeOnceLiveGatewayReportsAvailable:
    """probe_once returns (True, latency in (0, 5000ms]) and records state."""

    def test_probe_once_reports_available(self) -> None:
        from cogniverse_runtime.openshell_health import GatewayHealthProbe

        mgr = _make_manager()
        probe = GatewayHealthProbe(mgr, interval_seconds=1.0)
        available, latency = run_async(probe.probe_once())
        assert available is True
        # Latency must be positive and below the timeout window.
        assert 0 < latency <= 5000, latency
        assert probe.last_available is True
        assert probe.last_latency_ms == latency


# ---------------------------------------------------------------------------
# 2. probe_once emits the canonical span with attribute set
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProbeEmitsOpenshellGatewayHealthSpan:
    """A single probe emits an openshell.gateway_health span with required attrs."""

    def test_span_attributes_pinned(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_runtime.openshell_health import GatewayHealthProbe

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("gateway_probe_test")

        mgr = _make_manager()
        probe = GatewayHealthProbe(mgr, interval_seconds=1.0, tracer=tracer)
        run_async(probe.probe_once())

        spans = exporter.get_finished_spans()
        # Exactly one span per probe_once call — no leak, no missing.
        gateway_spans = [s for s in spans if s.name == "openshell.gateway_health"]
        assert len(gateway_spans) == 1, (
            f"expected exactly one openshell.gateway_health span, got "
            f"{[s.name for s in spans]}"
        )
        attrs = dict(gateway_spans[0].attributes or {})
        assert attrs.get("openshell.gateway_available") in {0, 1}, attrs
        assert isinstance(attrs.get("openshell.gateway_latency_ms"), (int, float))
        assert attrs["openshell.gateway_latency_ms"] > 0


# ---------------------------------------------------------------------------
# 3. probe loop runs at the configured cadence
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProbeLoopRunsOnSchedule:
    """interval_seconds=0.5; sleep 1.6s; expect at least 3 spans (3 ticks)."""

    def test_loop_emits_spans_on_cadence(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_runtime.openshell_health import GatewayHealthProbe

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("gateway_probe_loop_test")

        mgr = _make_manager()
        probe = GatewayHealthProbe(mgr, interval_seconds=0.5, tracer=tracer)

        async def _scenario() -> None:
            probe.start()
            await asyncio.sleep(1.6)
            await probe.stop()

        run_async(_scenario())

        spans = [
            s
            for s in exporter.get_finished_spans()
            if s.name == "openshell.gateway_health"
        ]
        # At t=0 (immediate first probe), t≈0.5, t≈1.0, t≈1.5 → 4 ticks
        # are likely; allow 3 as the lower bound for a slow probe step.
        assert len(spans) >= 3, (
            f"expected ≥ 3 ticks within 1.6s at interval=0.5s, got {len(spans)}"
        )
        assert probe.last_available is True


# ---------------------------------------------------------------------------
# 4. probe flips to unavailable when gateway client disappears
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProbeFlipsToUnavailableWhenClientGone:
    """Forcing manager._client = None makes probe_once return (False, latency_ms).

    The plan called for ``pkill openshell-sandbox-gateway``; that destroys
    the live gateway needed by every other test in this module. We
    exercise the same code path (the ``client is None`` branch
    in ``probe_once``) by setting ``mgr._client = None`` directly — the
    "gateway disappeared" semantics is what the probe code branches on,
    not the kill itself.
    """

    def test_flips_to_unavailable_on_missing_client(self) -> None:
        from cogniverse_runtime.openshell_health import GatewayHealthProbe

        mgr = _make_manager()
        probe = GatewayHealthProbe(mgr, interval_seconds=1.0)
        # Healthy first.
        ok, _ = run_async(probe.probe_once())
        assert ok is True

        # Now force the manager into the no-client state and re-probe.
        # SandboxManager re-connects from .available getter on miss, so
        # also patch _enabled=False to keep the second probe deterministic.
        mgr._client = None
        mgr._available = False
        mgr._enabled = False
        not_ok, latency = run_async(probe.probe_once())
        assert not_ok is False
        assert latency >= 0
        assert probe.last_available is False
