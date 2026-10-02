"""A failure answers with typed fields; its cause stays in the log and span."""

from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from cogniverse_runtime.http_errors import failure_response

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

SECRET = "http://admin:hunter2@vespa-0.internal:8080/document/v1"


def _cause() -> RuntimeError:
    try:
        raise RuntimeError(f"connection refused by {SECRET}")
    except RuntimeError as exc:
        return exc


def test_body_carries_typed_fields_and_never_the_exception_text():
    response = failure_response(
        503,
        "backend_unavailable",
        "The tenant registry did not answer; retry.",
        _cause(),
        headers={"Retry-After": "5"},
        tenant_id="acme:prod",
    )

    assert (response.status_code, response.headers) == (503, {"Retry-After": "5"})
    assert response.detail == {
        "error": "backend_unavailable",
        "message": "The tenant registry did not answer; retry.",
        "failure": "RuntimeError",
        "tenant_id": "acme:prod",
    }
    assert "hunter2" not in repr(response.detail)


def test_cause_is_logged_with_its_traceback_and_recorded_on_the_span(caplog):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("http-errors-test")

    with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.http_errors"):
        with tracer.start_as_current_span("api.request"):
            failure_response(500, "search_failed", "Search failed.", _cause())
        with tracer.start_as_current_span("api.request.invalid"):
            failure_response(400, "invalid_request", "Rejected.", ValueError("x"))

    failed, invalid = exporter.get_finished_spans()
    assert (failed.status.status_code, failed.status.description) == (
        StatusCode.ERROR,
        "search_failed: RuntimeError",
    )
    assert [
        (event.name, event.attributes["exception.type"]) for event in failed.events
    ] == [("exception", "RuntimeError")]
    assert SECRET in failed.events[0].attributes["exception.message"]
    assert invalid.status.description == "invalid_request: ValueError"
    assert [
        (record.levelno, record.getMessage(), record.exc_info[1].args[0])
        for record in caplog.records
    ] == [
        (
            logging.ERROR,
            f"search_failed: RuntimeError: connection refused by {SECRET}",
            f"connection refused by {SECRET}",
        ),
        (logging.WARNING, "invalid_request: ValueError: x", "x"),
    ]
