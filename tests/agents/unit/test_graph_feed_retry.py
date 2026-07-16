"""GraphManager._feed_with_retry survives the schema-convergence window.

Right after a fresh per-tenant KG deploy, the first feeds race Vespa's
content-distributor convergence: the backend put raises a RuntimeError whose
text carries Vespa's "does not exist" / "No handler for document type". The
retry loop must back off and retry those, but fail fast on anything else. This
loop had zero committed test reach — only the first-attempt-succeeds path ran.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_agents.graph.graph_manager import GraphManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _manager(backend):
    mgr = GraphManager.__new__(GraphManager)
    mgr._backend = backend
    mgr._schema_name = "knowledge_graph_acme"
    mgr._tenant_id = "acme:acme"
    return mgr


def test_transient_convergence_error_is_retried_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    backend = MagicMock()
    backend.put_document_fields.side_effect = [
        RuntimeError("... Document type knowledge_graph_acme does not exist"),
        RuntimeError("... No handler for document type ..."),
        None,  # third attempt succeeds
    ]
    mgr = _manager(backend)

    assert mgr._feed_with_retry("kg_node_x", {"f": 1}, "node") is True
    assert backend.put_document_fields.call_count == 3
    assert len(sleeps) == 2  # backed off before each retry


def test_non_transient_error_fails_fast_without_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    backend = MagicMock()
    backend.put_document_fields.side_effect = RuntimeError(
        "Could not parse field 'embedding': type mismatch"
    )
    mgr = _manager(backend)

    assert mgr._feed_with_retry("kg_node_x", {"f": 1}, "node") is False
    assert backend.put_document_fields.call_count == 1  # no retry
    assert sleeps == []


def test_persistent_transient_gives_up_after_the_attempt_budget(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    backend = MagicMock()
    backend.put_document_fields.side_effect = RuntimeError("... does not exist ...")
    mgr = _manager(backend)

    assert mgr._feed_with_retry("kg_node_x", {"f": 1}, "node") is False
    assert backend.put_document_fields.call_count == 8  # the loop's attempt cap


class TestGetEdgeById:
    """get_edge_by_id (2 prod callers: audit_explanation + citation_tracing)
    reads through the backend document API by the deterministic edge doc_id."""

    def test_returns_fields_for_an_existing_edge(self):
        backend = MagicMock()
        backend.get_document_fields.return_value = {"relation": "cites", "weight": 1}
        mgr = _manager(backend)

        out = mgr.get_edge_by_id("edge42")

        assert out == {"relation": "cites", "weight": 1}
        (doc_id,), kwargs = backend.get_document_fields.call_args
        assert doc_id == "kg_edge_acme_acme_edge42"
        assert kwargs["namespace"] == "graph_content"
        assert kwargs["schema_name"] == "knowledge_graph_acme"

    def test_missing_edge_returns_none(self):
        backend = MagicMock()
        backend.get_document_fields.return_value = None
        assert _manager(backend).get_edge_by_id("nope") is None

    def test_backend_error_raises_not_none(self):
        """An outage must not be mistaken for a missing edge."""
        backend = MagicMock()
        backend.get_document_fields.side_effect = RuntimeError("vespa down")
        with pytest.raises(RuntimeError, match="vespa down"):
            _manager(backend).get_edge_by_id("edge42")
