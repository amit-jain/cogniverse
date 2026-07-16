"""Graph upsert surfaces feed failures instead of a silent zero-count success.

A total feed failure (schema missing/unconverged, backend down) returned counts
of zero with no error, so the /graph/upsert route replied HTTP 200 "upserted"
having persisted nothing; and a persistent failure spent the full ~40s
convergence retry budget on every document in the batch. These pin the failed-id
reporting and the batch-level retry cap.
"""

from unittest.mock import MagicMock

import pytest

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import ExtractionResult, Node

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _manager(backend):
    mgr = GraphManager.__new__(GraphManager)
    mgr._backend = backend
    mgr._schema_name = "knowledge_graph_acme_acme"
    mgr._tenant_id = "acme:acme"
    # No encoder — nodes ship without embeddings via the encoder-down fallback.
    mgr._encode_documents = lambda texts: [{} for _ in texts]
    return mgr


def _nodes(*names):
    return [
        Node(tenant_id="acme:acme", name=n, description=f"about {n}", mentions=[])
        for n in names
    ]


def test_upsert_reports_failed_ids_and_caps_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    backend = MagicMock()
    backend.put_document_fields.side_effect = RuntimeError(
        "... Document type knowledge_graph_acme_acme does not exist"
    )
    mgr = _manager(backend)

    nodes = _nodes("alpha", "beta", "gamma")
    out = mgr.upsert(ExtractionResult(source_doc_id="d1", nodes=nodes, edges=[]))

    # Every feed failed — counts zero AND the failed ids are named.
    assert out["nodes_upserted"] == 0
    assert out["failed_ids"] == [n.doc_id for n in nodes]
    # First node retries the full convergence budget (8 attempts); the rest fail
    # fast (1 attempt each) — 8 + 1 + 1 = 10, not 3 * 8 = 24.
    assert backend.put_document_fields.call_count == 10


def test_upsert_partial_failure_names_only_the_failed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    nodes = _nodes("alpha", "beta")
    backend = MagicMock()
    # alpha succeeds, beta fails.
    backend.put_document_fields.side_effect = [
        None,
        RuntimeError("... does not exist"),
    ]
    mgr = _manager(backend)

    out = mgr.upsert(ExtractionResult(source_doc_id="d1", nodes=nodes, edges=[]))

    assert out["nodes_upserted"] == 1
    assert out["failed_ids"] == [nodes[1].doc_id]
