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

A non-2xx raises ``RuntimeError`` carrying the HTTP status and response
body, and a transport failure propagates as-is — matching the raise-on-
backend-failure contract the config/adapter stores enforce
(test_store_read_outage_raises.py). Callers that deliberately degrade
(ProvenanceStore.fetch, BackendVectorStore.list) catch and log; everyone
else fails loudly instead of misreading an outage as "no rows".

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


def test_non_200_raises_runtime_error_with_status(
    backend: VespaBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Vespa 400 must raise (with the status in the message) — never be
    masked as 'zero results' that a caller reads as a valid empty state."""
    error_body = {
        "root": {"errors": [{"code": 4, "summary": "BadRequest", "message": "x"}]}
    }
    fake = _FakeVespaClient(status_code=400, body=error_body)
    with caplog.at_level(logging.ERROR, logger="cogniverse_vespa.backend"):
        with patch(
            "cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake
        ):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                backend.query_metadata_documents(
                    schema="organization_metadata",
                    yql="select * from organization_metadata where true",
                )
    assert any("organization_metadata" in rec.message for rec in caplog.records), (
        "Expected the schema name in the error log"
    )


def test_non_200_raise_includes_response_body(
    backend: VespaBackend,
) -> None:
    """Operators need the response body to diagnose 4xx (bad YQL etc.)."""
    error_body = {"root": {"errors": [{"summary": "Parse error at position 17"}]}}
    fake = _FakeVespaClient(status_code=400, body=error_body)
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        with pytest.raises(RuntimeError, match="Parse error at position 17"):
            backend.query_metadata_documents(schema="tenant_metadata", yql="bad yql")


def test_transport_failure_propagates(backend: VespaBackend) -> None:
    """A dead Vespa must surface as the original transport error, not []."""

    class _DeadVespaClient:
        def query(self, **kwargs):
            raise ConnectionError("vespa unreachable")

    with patch(
        "cogniverse_vespa.backend.make_persistent_vespa_ops",
        return_value=_DeadVespaClient(),
    ):
        with pytest.raises(ConnectionError, match="vespa unreachable"):
            backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
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
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
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
        with patch(
            "cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake
        ):
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
