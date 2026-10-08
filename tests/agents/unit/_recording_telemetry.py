"""Recording telemetry-manager seam for agent unit tests.

Agents receive a telemetry manager from ``AgentDispatcher`` at every
construction site. Unit tests build agents directly, so they attach this
stand-in to match how production builds them and to assert exactly which
span each agent emits.

``span()`` mirrors the enqueue-only shape of ``TelemetryManager.span``:
name, tenant and an optional start time, recorded only when given. Passing
``require_export`` (synchronous export on the request path) is a
``TypeError`` by construction.
"""

from __future__ import annotations

from typing import Any


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.status: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.status = status


class SpanContext:
    def __init__(self, span: RecordingSpan) -> None:
        self._span = span

    def __enter__(self) -> RecordingSpan:
        return self._span

    def __exit__(self, *exc: Any) -> bool:
        return False


def _call(name: str, tenant_id: str, start_time: int | None) -> dict[str, Any]:
    call: dict[str, Any] = {"name": name, "tenant_id": tenant_id}
    if start_time is not None:
        call["start_time"] = start_time
    return call


class RecordingTelemetryManager:
    """Records every span() call. Enqueue-only: no export keyword exists."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.spans: list[RecordingSpan] = []

    def span(
        self, name: str, tenant_id: str, start_time: int | None = None
    ) -> SpanContext:
        self.calls.append(_call(name, tenant_id, start_time))
        span = RecordingSpan()
        self.spans.append(span)
        return SpanContext(span)


class FailingTelemetryManager:
    """A manager whose enqueue raises — the boundary-failure double."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or ConnectionError("collector unreachable")
        self.calls: list[dict[str, Any]] = []

    def span(
        self, name: str, tenant_id: str, start_time: int | None = None
    ) -> SpanContext:
        self.calls.append(_call(name, tenant_id, start_time))
        raise self.exc
