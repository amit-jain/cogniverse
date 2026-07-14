"""BackendVectorStore.update_many issues ONE batched backend feed.

The per-hit update() path costs one HTTP round-trip per document; the
batched path must build the same partial-update documents (metadata_
JSON-serialized, embedding omitted when vector is None so the stored
tensor survives) and hand them to the backend in a single
ingest_documents(operation_type="update") call.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from cogniverse_core.memory.backend_vector_store import BackendVectorStore


class _CapturingBackend:
    def __init__(self):
        self.ingest_calls: List[Dict[str, Any]] = []
        self.update_document_calls: List[str] = []

    def ingest_documents(self, documents, schema_name, operation_type="feed"):
        self.ingest_calls.append(
            {
                "documents": list(documents),
                "schema_name": schema_name,
                "operation_type": operation_type,
            }
        )
        return {"success_count": len(documents), "failed_docs": []}

    def update_document(self, doc_id, doc, schema_name):
        self.update_document_calls.append(doc_id)


def _store() -> tuple[BackendVectorStore, _CapturingBackend]:
    backend = _CapturingBackend()
    store = object.__new__(BackendVectorStore)
    store.backend = backend
    store.collection_name = "agent_memories_acme"
    store.profile = "agent_memories"
    store.is_telemetry = False
    return store, backend


def test_update_many_issues_single_batched_update_feed():
    store, backend = _store()

    store.update_many(
        [
            ("mem-1", None, {"data": "x", "metadata": {"last_accessed": "t1"}}),
            ("mem-2", None, {"data": "y", "metadata": {"last_accessed": "t2"}}),
            ("mem-3", None, {"data": "z", "metadata": {"last_accessed": "t3"}}),
        ]
    )

    assert backend.update_document_calls == []
    assert len(backend.ingest_calls) == 1
    call = backend.ingest_calls[0]
    assert call["operation_type"] == "update"
    assert call["schema_name"] == "agent_memories"
    docs = call["documents"]
    assert [d.id for d in docs] == ["mem-1", "mem-2", "mem-3"]
    for doc, stamp in zip(docs, ["t1", "t2", "t3"]):
        assert doc.embeddings == {}, "vector=None must not write an embedding"
        assert json.loads(doc.metadata["metadata_"]) == {"last_accessed": stamp}


def test_update_many_raises_when_feed_drops_documents():
    """Sibling-parity with insert(): a partially-dropped batch must raise,
    never report the dropped writes as stamped."""
    import pytest

    store, backend = _store()

    def dropping_ingest(documents, schema_name, operation_type="feed"):
        return {
            "success_count": len(documents) - 1,
            "failed_docs": [documents[-1].id],
        }

    backend.ingest_documents = dropping_ingest

    with pytest.raises(RuntimeError, match=r"persisted only 1/2"):
        store.update_many(
            [
                ("mem-1", None, {"data": "x", "metadata": {"last_accessed": "t1"}}),
                ("mem-2", None, {"data": "y", "metadata": {"last_accessed": "t2"}}),
            ]
        )


def test_update_many_skips_empty_items_and_noops_on_nothing():
    store, backend = _store()

    store.update_many([("mem-1", None, None)])
    store.update_many([])

    assert backend.ingest_calls == []


def test_update_many_builds_same_document_shape_as_update():
    """The batched path must serialize payloads exactly like update()."""
    store, backend = _store()
    payload = {
        "data": "remember the tabby cat",
        "user_id": "u1",
        "agent_id": "search_agent",
        "metadata": {"topic": "pets", "last_accessed": "2026-07-14T00:00:00+00:00"},
    }

    store.update("mem-solo", vector=None, payload=payload)
    store.update_many([("mem-batch", None, dict(payload))])

    assert backend.update_document_calls == ["mem-solo"]
    batched_doc = backend.ingest_calls[0]["documents"][0]
    assert batched_doc.text_content == "remember the tabby cat"
    assert batched_doc.metadata["user_id"] == "u1"
    assert batched_doc.metadata["agent_id"] == "search_agent"
    assert json.loads(batched_doc.metadata["metadata_"]) == {
        "topic": "pets",
        "last_accessed": "2026-07-14T00:00:00+00:00",
    }
