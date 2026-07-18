"""Periodic OpenShell gateway health probe with Phoenix span export.

The probe runs as a background asyncio task on a configurable interval. Each
probe:

  1. Calls ``SandboxClient.health()`` on the live gateway client.
  2. Records latency in ms and an availability boolean.
  3. Emits an OpenTelemetry span (``openshell.gateway_health``) with
     attributes ``openshell.gateway_available`` (0/1) and
     ``openshell.gateway_latency_ms``. The Phoenix dashboard reads these
     spans for the gateway-status dashboard tile.

Lifecycle: ``GatewayHealthProbe.start()`` schedules the loop on the running
event loop; ``stop()`` cancels it and awaits clean shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from opentelemetry import trace

if TYPE_CHECKING:
    from cogniverse_runtime.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 30.0
_PROBE_TIMEOUT_SECONDS = 5.0


def _response_availability(response: object) -> tuple[bool, Optional[str]]:
    """Map a gateway ``HealthResponse`` to ``(available, error_attr)``.

    The gateway can answer without raising while reporting itself sick
    (``status=SERVICE_STATUS_UNHEALTHY`` / ``DEGRADED``), so a non-raising
    ``health()`` call alone does not mean available. A response lacking a
    status — or an empty message left at the proto3 default
    ``SERVICE_STATUS_UNSPECIFIED`` — counts as available: the gateway
    answered and did not report a problem.
    """
    status = getattr(response, "status", None)
    if status is None:
        return True, None
    from openshell._proto import openshell_pb2 as pb

    if status in (pb.SERVICE_STATUS_UNSPECIFIED, pb.SERVICE_STATUS_HEALTHY):
        return True, None
    try:
        name = pb.ServiceStatus.Name(status)
    except (TypeError, ValueError):
        name = f"status_{status}"
    return False, name


class GatewayHealthProbe:
    """Background probe that records gateway availability + latency to Phoenix.

    Args:
        sandbox_manager: SandboxManager whose underlying client we probe.
        interval_seconds: Probe cadence. Defaults to 30 s; tuneable for tests.
        tracer: Optional opentelemetry tracer; when None, the global tracer
            is used. Tests can inject a recording tracer to assert spans.
    """

    def __init__(
        self,
        sandbox_manager: "SandboxManager",
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        tracer: Optional[trace.Tracer] = None,
    ) -> None:
        self._mgr = sandbox_manager
        self._interval = interval_seconds
        self._tracer = tracer or trace.get_tracer(__name__)
        self._task: Optional[asyncio.Task] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._last_available: Optional[bool] = None
        self._last_latency_ms: Optional[float] = None

    @property
    def last_available(self) -> Optional[bool]:
        """Latest availability signal (None before first probe)."""
        return self._last_available

    @property
    def last_latency_ms(self) -> Optional[float]:
        """Latest probe latency in milliseconds (None before first probe)."""
        return self._last_latency_ms

    async def probe_once(self) -> tuple[bool, float]:
        """Run a single probe synchronously. Returns (available, latency_ms).

        Always emits a span — both success and failure paths are observable.
        """
        client = self._mgr._client
        start = time.monotonic()
        available = False

        if client is None:
            # Manager is disabled or never connected; record availability=0
            # without raising so the tile still updates.
            latency_ms = (time.monotonic() - start) * 1000
            self._record_span(available=False, latency_ms=latency_ms, error="no_client")
            self._last_available = False
            self._last_latency_ms = latency_ms
            return False, latency_ms

        try:
            # ``client.health()`` is sync (gRPC). Run on a thread to keep the
            # asyncio loop free; bound by _PROBE_TIMEOUT_SECONDS so a stuck
            # gateway can never block the probe forever.
            response = await asyncio.wait_for(
                asyncio.to_thread(client.health),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
            available, error_attr = _response_availability(response)
            if not available:
                logger.debug("OpenShell gateway reports sick status: %s", error_attr)
        except Exception as exc:  # connection refused, timeout, gRPC error
            error_attr = type(exc).__name__
            logger.debug("OpenShell gateway probe failed: %s", exc)
        latency_ms = (time.monotonic() - start) * 1000

        self._record_span(available=available, latency_ms=latency_ms, error=error_attr)
        self._last_available = available
        self._last_latency_ms = latency_ms
        return available, latency_ms

    def _record_span(
        self,
        available: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Emit a Phoenix-compatible OpenTelemetry span for this probe."""
        with self._tracer.start_as_current_span("openshell.gateway_health") as span:
            span.set_attribute("openshell.gateway_available", 1 if available else 0)
            span.set_attribute("openshell.gateway_latency_ms", latency_ms)
            if error is not None:
                span.set_attribute("openshell.gateway_error", error)

    def start(self) -> None:
        """Start the background probe loop on the running event loop."""
        if self._task is not None and not self._task.done():
            logger.debug("GatewayHealthProbe already running; start() is a no-op")
            return
        loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        self._task = loop.create_task(self._run_loop(), name="openshell_health_probe")
        logger.info(
            "OpenShell gateway health probe started (interval=%.1fs)",
            self._interval,
        )

    async def stop(self) -> None:
        """Stop the probe loop and await clean shutdown."""
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("OpenShell gateway health probe stopped")

    async def _run_loop(self) -> None:
        """Probe loop: probe → wait interval (or stop) → repeat."""
        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            try:
                await self.probe_once()
            except Exception:  # never let a probe error kill the loop
                logger.exception("Unhandled error during OpenShell gateway probe")
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
