"""Regression test for VespaBackend.query_metadata_documents
status_code handling.

Real pyvespa RAISES VespaError on non-2xx (raise_for_status; only a 404
returns), so the status_code branch here is a belt-and-braces net for the
rare returns-without-raising shape — the raise path is covered by
test_vespa_error_propagates below. The pre-fix code path read
``results.json.get("root", {}).get("children", [])`` directly, which
evaluates to ``[]`` on an error body, so the function silently returned
an empty list. Callers (``BackendVectorStore.
list``, ``ProvenanceStore.fetch``, the admin tenant routes) cannot
distinguish "no matches" from "Vespa rejected the query."

Siblings ``get_metadata_document`` (returns ``None`` on a genuine 404,
raises on a backend failure) and ``delete_metadata_document`` (returns
``False``) both check ``status_code``; ``query_metadata_documents`` was
the odd one out.

A non-2xx raises ``RuntimeError`` carrying the HTTP status and response
body, and a transport failure propagates as-is — matching the raise-on-
backend-failure contract the config/adapter stores enforce
(test_store_read_outage_raises.py). Callers that deliberately degrade
(ProvenanceStore.fetch, BackendVectorStore.list) catch and log; everyone
else fails loudly instead of misreading an outage as "no rows".

A soft timeout or degraded content-node coverage is the sneakier case: it
returns HTTP 200 with ``root.errors`` (and only partial children) or a
``root.coverage.degraded`` marker. Consuming the children then returns a
partial listing recorded as success. Both now raise, matching the
convergence probe and ``vespa_search_children``.

These tests mock pyvespa's response object directly (the CONTRACT side
of the boundary, not the SUT side).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vespa.exceptions import VespaError

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


class _CapturingVespaClient:
    """Records the query parameters so paging kwargs can be asserted."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(status_code=200, json={"root": {"children": []}})


def test_offset_forwarded_as_native_query_parameter(backend: VespaBackend) -> None:
    """A non-zero paging offset must ride as Vespa's native ``offset``
    parameter. A YQL offset alone is bounded by ``hits``, so the second page
    of a walk lands outside the window and comes back empty — the walk then
    silently truncates at the first page."""
    fake = _CapturingVespaClient()
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        backend.query_metadata_documents(
            schema="agent_memories_acme",
            yql="select * from agent_memories_acme where true order by id desc",
            hits=100,
            offset=100,
        )
    assert fake.calls[0]["body"]["offset"] == 100
    assert fake.calls[0]["body"]["hits"] == 100


def test_offset_zero_omits_the_parameter(backend: VespaBackend) -> None:
    """Page one (offset 0) must not send an offset — the default is 0 and an
    explicit 0 is redundant noise on the hot first-page read."""
    fake = _CapturingVespaClient()
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        backend.query_metadata_documents(
            schema="agent_memories_acme",
            yql="select * from agent_memories_acme where true order by id desc",
            hits=100,
            offset=0,
        )
    assert "offset" not in fake.calls[0]["body"]


@pytest.mark.parametrize(
    ("query_kwargs", "expected_query"),
    [
        (
            {"hits": 500},
            {
                "hits": 500,
                "maxHits": 500,
                "maxOffset": 500,
                "yql": "select * from agent_memories_acme where true order by id desc",
            },
        ),
        (
            {"hits": 500, "offset": 500},
            {
                "hits": 500,
                "maxHits": 500,
                "maxOffset": 1000,
                "offset": 500,
                "yql": "select * from agent_memories_acme where true order by id desc",
            },
        ),
    ],
)
def test_hits_raise_native_query_limits(
    backend: VespaBackend,
    query_kwargs: dict[str, int],
    expected_query: dict[str, int | str],
) -> None:
    """Paging requests must raise Vespa's native query caps with the same
    requested window, including later pages."""
    fake = _CapturingVespaClient()
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        backend.query_metadata_documents(
            schema="agent_memories_acme",
            yql="select * from agent_memories_acme where true order by id desc",
            **query_kwargs,
        )
    assert fake.calls == [{"body": expected_query}]


def test_vespa_error_propagates(backend: VespaBackend) -> None:
    """The REAL non-2xx shape: pyvespa raise_for_status raises VespaError on
    4xx/5xx (only 404 returns) — it must propagate, not flatten to []."""

    class _RaisingVespaClient:
        def query(self, **kwargs):
            raise VespaError("400 Could not create query from YQL")

    with patch(
        "cogniverse_vespa.backend.make_persistent_vespa_ops",
        return_value=_RaisingVespaClient(),
    ):
        with pytest.raises(VespaError, match="Could not create query"):
            backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
            )


def test_200_with_root_errors_raises(backend: VespaBackend) -> None:
    """A soft timeout returns HTTP 200 with root.errors and only partial
    children — it must raise, not return the partial list as a complete
    result that a caller reads as the full set of tenants/memories."""
    body = {
        "root": {
            "errors": [
                {"code": 12, "summary": "Timeout", "message": "Timed out 1 of 3 groups"}
            ],
            "children": [{"fields": {"org_id": "acme", "name": "Acme"}}],
        }
    }
    fake = _FakeVespaClient(status_code=200, body=body)
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        with pytest.raises(RuntimeError, match="Timed out 1 of 3 groups"):
            backend.query_metadata_documents(
                schema="organization_metadata",
                yql="select * from organization_metadata where true",
            )


def test_200_with_degraded_coverage_raises(backend: VespaBackend) -> None:
    """Degraded content-node coverage on a 200 (a partial scan) must raise —
    a partial scan is not a complete result, even with no root.errors."""
    body = {
        "root": {
            "coverage": {"coverage": 42, "full": False, "degraded": {"timeout": True}},
            "children": [{"fields": {"org_id": "acme"}}],
        }
    }
    fake = _FakeVespaClient(status_code=200, body=body)
    with patch("cogniverse_vespa.backend.make_persistent_vespa_ops", return_value=fake):
        with pytest.raises(RuntimeError, match="coverage degraded"):
            backend.query_metadata_documents(
                schema="tenant_metadata",
                yql="select * from tenant_metadata where true",
            )


def test_metadata_query_returns_empty_when_tenant_schema_is_missing() -> None:
    backend = object.__new__(VespaBackend)
    backend._url = "http://vespa"
    backend._port = 8080
    backend.get_tenant_schema_name = MagicMock(return_value="wiki_pages_acme_acme")
    backend._read_path_deployed = MagicMock(return_value=False)
    backend._metadata_vespa_app = MagicMock()

    rows = backend.query_metadata_documents(
        schema="wiki_pages",
        yql="select * from sources wiki_pages where true limit 2",
        tenant_id="acme:acme",
        hits=2,
    )

    assert rows == []
    backend._read_path_deployed.assert_called_once_with("acme:acme", "wiki_pages")
    backend._metadata_vespa_app.assert_not_called()


def test_metadata_query_raises_when_tenant_schema_lookup_fails() -> None:
    backend = object.__new__(VespaBackend)
    backend._url = "http://vespa"
    backend._port = 8080
    backend.get_tenant_schema_name = MagicMock(return_value="wiki_pages_acme_acme")
    backend._read_path_deployed = MagicMock(
        side_effect=RuntimeError("schema registry unavailable")
    )
    backend._metadata_vespa_app = MagicMock()

    with pytest.raises(RuntimeError, match="schema registry unavailable"):
        backend.query_metadata_documents(
            schema="wiki_pages",
            yql="select * from sources wiki_pages where true limit 2",
            tenant_id="acme:acme",
            hits=2,
        )

    backend._read_path_deployed.assert_called_once_with("acme:acme", "wiki_pages")
    backend._metadata_vespa_app.assert_not_called()


def _stale_backend(query_error: Exception, stored_deployed: bool):
    backend = object.__new__(VespaBackend)
    backend._url = "http://vespa"
    backend._port = 8080
    backend.get_tenant_schema_name = MagicMock(return_value="wiki_pages_acme_acme")
    backend._read_path_deployed = MagicMock(return_value=True)
    backend._deployed_schema_names = MagicMock()
    backend.schema_exists = MagicMock(return_value=stored_deployed)
    client = MagicMock()
    client.query.side_effect = query_error
    backend._metadata_vespa_app = MagicMock(return_value=client)
    return backend


_UNRESOLVED = VespaError(
    [
        {
            "code": 4,
            "summary": "Invalid query parameter",
            "message": "Could not resolve source ref 'wiki_pages_acme_acme'. "
            "Valid source refs are cogniverse_content.tenant_metadata.",
        }
    ]
)


def test_a_query_on_a_schema_a_peer_dropped_answers_no_rows() -> None:
    """Vespa no longer resolves the schema and the stored row no longer
    registers it: no rows, and the read-path answer is dropped."""
    backend = _stale_backend(_UNRESOLVED, stored_deployed=False)

    rows = backend.query_metadata_documents(
        schema="wiki_pages", tenant_id="acme:acme", hits=2
    )

    assert rows == []
    backend.schema_exists.assert_called_once_with("wiki_pages", tenant_id="acme:acme")
    backend._deployed_schema_names.invalidate.assert_called_once_with("acme:acme")


def test_an_unresolved_schema_the_store_still_registers_raises() -> None:
    backend = _stale_backend(_UNRESOLVED, stored_deployed=True)

    with pytest.raises(VespaError) as caught:
        backend.query_metadata_documents(
            schema="wiki_pages", tenant_id="acme:acme", hits=2
        )

    assert caught.value is _UNRESOLVED
    backend._deployed_schema_names.invalidate.assert_not_called()


def test_any_other_query_failure_raises_without_reading_the_store() -> None:
    other = VespaError([{"code": 12, "summary": "Timed out", "message": "timeout"}])
    backend = _stale_backend(other, stored_deployed=False)

    with pytest.raises(VespaError) as caught:
        backend.query_metadata_documents(
            schema="wiki_pages", tenant_id="acme:acme", hits=2
        )

    assert caught.value is other
    backend.schema_exists.assert_not_called()
