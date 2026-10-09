"""Hold a test's spans in the batch exporter and watch the runtime look them
up while it waits for them, through an ``InterceptFaultProxy`` in front of
Phoenix."""

from __future__ import annotations

import time
from contextlib import contextmanager

from cogniverse_foundation.telemetry.config import BatchExportConfig

# Long enough that no span of a test is exported before the test flushes it.
HELD_EXPORT_MS = 60_000


@contextmanager
def held_span_export(telemetry):
    """Spans of tenants first traced inside the block wait in a batch
    exporter that sends them only when the test flushes it, as a deployed
    runtime's exporter holds a span for its schedule delay; the runtime's
    waits read the same configuration. Yields that configuration."""
    shipped = telemetry.config.batch_config
    telemetry.config.batch_config = BatchExportConfig(
        use_sync_export=False, schedule_delay_millis=HELD_EXPORT_MS
    )
    try:
        yield telemetry.config.batch_config
    finally:
        telemetry.force_flush(timeout_millis=10000)
        telemetry.config.batch_config = shipped


def span_lookups(proxy, span_id):
    """How many lookups naming ``span_id`` the proxy forwarded."""
    return sum(
        1
        for method, path, body in proxy.requests
        if method == "POST" and path == "/graphql" and span_id.encode() in body
    )


def answers(pending):
    """``(status, body)`` of each of the ``pending`` request futures that
    already returned."""
    return [
        (future.result().status_code, future.result().json())
        for future in pending
        if future.done()
    ]


def await_span_lookups(proxy, span_id, count, answered=list, timeout=30.0):
    """Return once the proxy has seen ``count`` lookups naming ``span_id``,
    while ``answered()`` (the requests already answered) stays empty."""
    deadline = time.monotonic() + timeout
    while span_lookups(proxy, span_id) < count:
        assert answered() == [], "answered before the span was exported"
        assert time.monotonic() < deadline, (
            f"{span_lookups(proxy, span_id)} lookups of {span_id}, not {count}"
        )
        time.sleep(0.05)
    assert answered() == [], "answered before the span was exported"
