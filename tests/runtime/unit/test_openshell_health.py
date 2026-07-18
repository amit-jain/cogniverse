"""Unit tests for the OpenShell gateway health probe.

These tests exercise the probe against a fake client (no real gateway) to
verify the contract: emits spans with correct attributes, tolerates probe
errors without killing the loop, honours stop() promptly. Real-gateway
coverage lives in ``tests/agents/integration/test_sandbox_integration.py``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from openshell._proto import openshell_pb2
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_runtime.openshell_health import GatewayHealthProbe


@pytest.fixture
def captured_spans():
    """OpenTelemetry exporter that captures spans in-memory for assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


def _manager_with_client(client):
    """Build a minimal SandboxManager-shaped object with the given client."""
    mgr = MagicMock()
    mgr._client = client
    return mgr


class TestProbeOnce:
    @pytest.mark.asyncio
    async def test_records_available_when_health_ok(self, captured_spans):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse(
            status=openshell_pb2.SERVICE_STATUS_HEALTHY, version="0.0.13"
        )

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, latency = await probe.probe_once()

        assert available is True
        assert latency >= 0
        assert probe.last_available is True
        assert probe.last_latency_ms == latency

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["openshell.gateway_available"] == 1
        assert attrs["openshell.gateway_latency_ms"] == latency
        assert "openshell.gateway_error" not in attrs

    @pytest.mark.asyncio
    async def test_unhealthy_status_records_unavailable(self, captured_spans):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse(
            status=openshell_pb2.SERVICE_STATUS_UNHEALTHY, version="0.0.13"
        )

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, latency = await probe.probe_once()

        assert available is False
        assert probe.last_available is False
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["openshell.gateway_available"] == 0
        assert attrs["openshell.gateway_error"] == "SERVICE_STATUS_UNHEALTHY"

    @pytest.mark.asyncio
    async def test_degraded_status_records_unavailable(self, captured_spans):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse(
            status=openshell_pb2.SERVICE_STATUS_DEGRADED
        )

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, _ = await probe.probe_once()

        assert available is False
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["openshell.gateway_available"] == 0
        assert attrs["openshell.gateway_error"] == "SERVICE_STATUS_DEGRADED"

    @pytest.mark.asyncio
    async def test_empty_response_message_records_available(self, captured_spans):
        """Some gateways answer health with an empty message (status left at
        the proto3 default SERVICE_STATUS_UNSPECIFIED); that still counts as
        available — the gateway answered and did not report itself sick."""
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse()

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, _ = await probe.probe_once()

        assert available is True
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["openshell.gateway_available"] == 1
        assert "openshell.gateway_error" not in attrs

    @pytest.mark.asyncio
    async def test_response_without_status_attribute_records_available(
        self, captured_spans
    ):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = object()

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, _ = await probe.probe_once()

        assert available is True
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["openshell.gateway_available"] == 1
        assert "openshell.gateway_error" not in attrs

    @pytest.mark.asyncio
    async def test_records_unavailable_on_health_failure(self, captured_spans):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.side_effect = ConnectionRefusedError("gateway down")

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, latency = await probe.probe_once()

        assert available is False
        assert latency >= 0
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["openshell.gateway_available"] == 0
        assert attrs["openshell.gateway_error"] == "ConnectionRefusedError"

    @pytest.mark.asyncio
    async def test_no_client_records_unavailable_with_error_attr(self, captured_spans):
        tracer, exporter = captured_spans
        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client=None),
            interval_seconds=30.0,
            tracer=tracer,
        )
        available, _ = await probe.probe_once()

        assert available is False
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["openshell.gateway_available"] == 0
        assert attrs["openshell.gateway_error"] == "no_client"


class TestStartStopLifecycle:
    @pytest.mark.asyncio
    async def test_start_runs_periodic_probes_and_stop_clean(self, captured_spans):
        tracer, exporter = captured_spans
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse(
            status=openshell_pb2.SERVICE_STATUS_HEALTHY
        )

        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=0.05,  # fast for the test
            tracer=tracer,
        )
        probe.start()

        # Allow several probes to fire.
        await asyncio.sleep(0.18)
        await probe.stop()

        spans = exporter.get_finished_spans()
        assert len(spans) >= 2, (
            f"expected multiple probe spans during 0.18s @ 0.05s interval; "
            f"got {len(spans)}"
        )
        # All recorded as available since client.health succeeds.
        for span in spans:
            assert dict(span.attributes)["openshell.gateway_available"] == 1

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_never_started(self):
        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(MagicMock()),
            interval_seconds=30.0,
        )
        # Must not raise.
        await probe.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        client = MagicMock()
        client.health.return_value = openshell_pb2.HealthResponse(
            status=openshell_pb2.SERVICE_STATUS_HEALTHY
        )
        probe = GatewayHealthProbe(
            sandbox_manager=_manager_with_client(client),
            interval_seconds=0.05,
        )
        probe.start()
        first_task = probe._task
        probe.start()  # second start is a no-op
        assert probe._task is first_task
        await probe.stop()
