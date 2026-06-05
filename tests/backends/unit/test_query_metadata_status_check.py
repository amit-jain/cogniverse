"""Regression test for VespaBackend.query_metadata_documents
status_code handling.

pyvespa does NOT raise on a non-2xx response — it returns a response
object with ``.status_code`` set and ``.json`` holding the error body.
The pre-fix code path read ``results.json.get("root", {}).get("children",
[])`` directly, which evaluates to ``[]`` on a 4xx error body, so the
function silently returned an empty list. Callers (``BackendVectorStore.
list``, ``ProvenanceStore.fetch``, the admin tenant routes) cannot
distinguish "no matches" from "Vespa rejected the query."

Siblings ``get_metadata_document`` (returns ``None``) and
``delete_metadata_document`` (returns ``False``) both check
``status_code``; ``query_metadata_documents`` was the odd one out.

The fix raises ``RuntimeError`` on non-2xx inside the method's existing
try/except, so the outer log line now includes the HTTP status and the
response body — actionable signal for operators.

These tests mock pyvespa's response object directly (the CONTRACT side
of the boundary, not the SUT side).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cogniverse_vespa.backend import VespaBackend


@pytest.fixture
def backend() -> VespaBackend:
    # query_metadata_documents reads self._url/_port and the lazily-cached
    # metadata client (_metadata_app / _metadata_app_key); the constructor
    # demands a full BackendConfig + schema_loader + config_manager for real
    # init. Bypass __init__ to keep the unit test focused on the method-level
    # status_code branch (the real-Vespa integration suite exercises full
    # construction).
    b = object.__new__(VespaBackend)
    b._url = "http://test-vespa"
    b._port = 8080
    b._metadata_app = None
    b._metadata_app_key = None
    return b


class _FakeVespaClient:
    """Stand-in for the pyvespa client returned by make_vespa_app.

    Returns a configurable ``SimpleNamespace`` response so we can drive
    both 200-with-body and 4xx-with-error-body branches.
    """

    def __init__(self, status_code: int, body: dict) -> None:
        self._status_code = status_code
        self._body = body

    def query(self, **kwargs):
        return SimpleNamespace(status_code=self._status_code, json=self._body)


def test_non_200_raises_logged_runtime_error_and_returns_empty(
    backend: VespaBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Vespa 400 must be logged at error level with the response body,
    not silently masked as 'zero results'."""
    error_body = {
        "root": {"errors": [{"code": 4, "summary": "BadRequest", "message": "x"}]}
    }
    fake = _FakeVespaClient(status_code=400, body=error_body)
    with caplog.at_level(logging.ERROR, logger="cogniverse_vespa.backend"):
        with patch("cogniverse_vespa.backend.make_vespa_app", return_value=fake):
            result = backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
            )
    # Caller-facing return preserved (list) — the outer except still
    # swallows to [] so existing callers don't break — but the log now
    # carries actionable detail.
    assert result == []
    assert any("HTTP 400" in rec.message for rec in caplog.records), (
        f"Expected an error log mentioning 'HTTP 400'; got: "
        f"{[r.message for r in caplog.records]}"
    )
    assert any("organization_metadata" in rec.message for rec in caplog.records), (
        "Expected the schema name in the error log"
    )


def test_non_200_log_includes_response_body(
    backend: VespaBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operators need the response body to diagnose 4xx (bad YQL etc.)."""
    error_body = {"root": {"errors": [{"summary": "Parse error at position 17"}]}}
    fake = _FakeVespaClient(status_code=400, body=error_body)
    with caplog.at_level(logging.ERROR, logger="cogniverse_vespa.backend"):
        with patch("cogniverse_vespa.backend.make_vespa_app", return_value=fake):
            backend.query_metadata_documents(schema="tenant_metadata", yql="bad yql")
    assert any("Parse error at position 17" in rec.message for rec in caplog.records), (
        f"Expected the Vespa error body in the log; got: "
        f"{[r.message for r in caplog.records]}"
    )


def test_200_with_results_returns_fields(backend: VespaBackend) -> None:
    """Happy path: 200 with a populated root.children returns fields."""
    body = {
        "root": {
            "children": [
                {"fields": {"org_id": "acme", "name": "Acme"}},
                {"fields": {"org_id": "beta", "name": "Beta"}},
            ]
        }
    }
    fake = _FakeVespaClient(status_code=200, body=body)
    with patch("cogniverse_vespa.backend.make_vespa_app", return_value=fake):
        result = backend.query_metadata_documents(
            schema="organization_metadata",
            yql="select * from organization_metadata where true",
        )
    assert result == [
        {"org_id": "acme", "name": "Acme"},
        {"org_id": "beta", "name": "Beta"},
    ]


def test_200_with_empty_children_returns_empty_list(
    backend: VespaBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A LEGITIMATE empty result (200 + no children) must NOT log an error —
    it must be distinguishable from the 4xx silent-mask path."""
    body = {"root": {"children": []}}
    fake = _FakeVespaClient(status_code=200, body=body)
    with caplog.at_level(logging.ERROR, logger="cogniverse_vespa.backend"):
        with patch("cogniverse_vespa.backend.make_vespa_app", return_value=fake):
            result = backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
            )
    assert result == []
    # The bug class this test guards: legitimate empty results must NOT
    # write to the error log, or operators will drown in noise.
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records), (
        f"Empty 200 result wrongly logged an error: {[r.message for r in caplog.records]}"
    )


def test_uninitialized_backend_raises(backend: VespaBackend) -> None:
    """A separate-from-this-fix sanity check: the initialize() guard still
    fires when the backend was not initialized."""
    backend._url = None
    with pytest.raises(RuntimeError, match="not initialized"):
        backend.query_metadata_documents(schema="x", yql="y")
