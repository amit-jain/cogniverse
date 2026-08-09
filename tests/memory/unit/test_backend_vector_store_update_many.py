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
        self.search_calls: List[Dict[str, Any]] = []
        self.update_document_calls: List[str] = []

    def ingest_documents(self, documents, schema_name, operation_type="feed"):
        self.ingest_calls.append(
            {
                "documents": list(documents),
                "schema_name": schema_name,
                "operation_type": operation_type,
            }
        )
        return {"success_count": len(documents), "failed_documents": []}

    def update_document(self, doc_id, doc, schema_name):
        self.update_document_calls.append(doc_id)

    def search(self, query_dict):
        self.search_calls.append(query_dict)
        return []


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
            "failed_documents": [documents[-1].id],
        }

    backend.ingest_documents = dropping_ingest

    # The diagnostic must NAME the dropped ids — the read used a key the
    # real ingest_documents never returns, so it always printed None.
    with pytest.raises(RuntimeError, match=r"persisted only 1/2.*mem-2"):
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


def test_memory_search_requests_exact_nearest_neighbor():
    backend = _CapturingBackend()
    store = BackendVectorStore(
        collection_name="agent_memories_acme_acme",
        backend_client=backend,
        embedding_model_dims=3,
        tenant_id="acme:acme",
        profile="agent_memories",
    )

    assert (
        store.search(
            "Marie Curie discovered radium",
            vectors=[1.0, 0.0, 0.0],
            limit=1,
            filters={"user_id": "scientists", "agent_id": "research"},
        )
        == []
    )

    assert len(backend.search_calls) == 1
    query = backend.search_calls[0]
    assert set(query) == {
        "query",
        "type",
        "profile",
        "schema_name",
        "strategy",
        "top_k",
        "filters",
        "query_embeddings",
        "tenant_id",
        "nearest_neighbor_approximate",
    }
    assert query["query"] == "Marie Curie discovered radium"
    assert query["type"] == "memory"
    assert query["profile"] == "agent_memories"
    assert query["schema_name"] == "agent_memories_acme_acme"
    assert query["strategy"] == "semantic_search"
    assert query["top_k"] == 1
    assert query["filters"] == {
        "user_id": "scientists",
        "agent_id": "research",
    }
    assert query["query_embeddings"].tolist() == [1.0, 0.0, 0.0]
    assert query["tenant_id"] == "acme:acme"
    assert query["nearest_neighbor_approximate"] is False


import pytest  # noqa: E402


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDimensionValidation:
    """embedding_model_dims is a REAL contract, not bookkeeping: a vector of
    the wrong dimension fails fast with a clear error instead of a downstream
    Vespa 400 (or silently garbage ANN scores), and the attribute mem0 reads
    (``embedding_model_dims``) is exposed with the true value."""

    def _store(self, dims=768):
        from unittest.mock import MagicMock

        from cogniverse_core.memory.backend_vector_store import BackendVectorStore

        return BackendVectorStore(
            collection_name="agent_memories_t",
            backend_client=MagicMock(),
            embedding_model_dims=dims,
            tenant_id="t",
            profile="agent_memories",
        )

    def test_constructor_rejects_non_positive_dims(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="embedding_model_dims"):
            self._store(dims=0)
        with _pytest.raises(ValueError, match="embedding_model_dims"):
            self._store(dims=-1)

    def test_mem0_visible_attribute_carries_true_dims(self):
        store = self._store(dims=1024)
        # mem0 reads getattr(vector_store, "embedding_model_dims", 1536); the
        # store previously only set vector_size, so mem0 saw the 1536 default.
        assert store.embedding_model_dims == 1024
        assert store.vector_size == 1024
        store.create_col("agent_memories_t2", 512, "cosine")
        assert store.embedding_model_dims == 512
        assert store.vector_size == 512

    def test_insert_rejects_wrong_dimension_vector(self):
        import pytest as _pytest

        store = self._store(dims=768)
        with _pytest.raises(ValueError, match="512.*768|768.*512"):
            store.insert(vectors=[[0.1] * 512], payloads=[{"data": "x"}], ids=["m1"])
        store.backend.ingest_documents.assert_not_called()

    def test_search_rejects_wrong_dimension_query(self):
        import pytest as _pytest

        store = self._store(dims=768)
        with _pytest.raises(ValueError, match="512.*768|768.*512"):
            store.search("q", vectors=[0.1] * 512, limit=3)

    def test_correct_dimension_flows_through(self):
        store = self._store(dims=8)
        store.backend.ingest_documents.return_value = {"success_count": 1}
        store.insert(vectors=[[0.1] * 8], payloads=[{"data": "x"}], ids=["m1"])
        store.backend.ingest_documents.assert_called_once()


class TestCreatedAtNormalization:
    """Stored created_at epochs must normalize to ISO on every read path —
    np.int64 is not an int subclass, so the plain isinstance gate missed
    numpy epochs and stringified them to bare digits instead of ISO."""

    def test_numpy_epoch_normalizes_to_iso(self):
        import numpy as np

        from cogniverse_core.memory._timestamps import epoch_to_iso_utc
        from cogniverse_core.memory.backend_vector_store import _created_at_iso

        epoch = 1700000000
        expected = epoch_to_iso_utc(epoch)
        assert _created_at_iso(np.int64(epoch)) == expected
        assert _created_at_iso(np.float64(epoch)) == expected
        assert _created_at_iso(epoch) == expected
        assert _created_at_iso(float(epoch)) == expected

    def test_string_and_none_pass_through(self):
        from cogniverse_core.memory.backend_vector_store import _created_at_iso

        assert _created_at_iso("2024-01-01T00:00:00+00:00") == (
            "2024-01-01T00:00:00+00:00"
        )
        assert _created_at_iso(None) is None


class TestReadFaultContract:
    """Backend failures during reads must raise — search() returning [] and
    get() returning None on an outage are indistinguishable from genuine
    absence, silently disabling memory retrieval. delete() already raises."""

    def _store(self):
        from unittest.mock import MagicMock

        return BackendVectorStore(
            collection_name="agent_memories_t",
            backend_client=MagicMock(),
            embedding_model_dims=768,
            tenant_id="t",
            profile="agent_memories",
        )

    @pytest.mark.unit
    def test_search_raises_on_backend_failure(self):
        store = self._store()
        store.backend.search.side_effect = ConnectionError("backend down")

        with pytest.raises(ConnectionError):
            store.search("q", vectors=[0.1] * 768, limit=3)

    @pytest.mark.unit
    def test_get_raises_on_backend_failure(self):
        store = self._store()
        store.backend.get_document.side_effect = ConnectionError("backend down")

        with pytest.raises(ConnectionError):
            store.get("mem-1")

    @pytest.mark.unit
    def test_get_returns_none_for_genuine_not_found(self):
        store = self._store()
        store.backend.get_document.return_value = None

        assert store.get("missing-id") is None

    @pytest.mark.unit
    def test_get_unwraps_canonical_document_embedding(self):
        from cogniverse_sdk.document import Document

        store = self._store()
        document = Document(
            id="mem-embedded",
            text_content="exact memory text",
            metadata={
                "user_id": "user-7",
                "agent_id": "agent-4",
                "created_at": 1_700_000_000,
            },
        )
        document.add_embedding(
            "embedding",
            [0.125, -0.25, 0.5],
            metadata={"model": "exact-test-model"},
        )
        store.backend.get_document.return_value = document

        record = store.get("mem-embedded")

        assert record.id == "mem-embedded"
        assert record.vector == [0.125, -0.25, 0.5]
        assert record.payload["data"] == "exact memory text"
        assert record.payload["created_at"] == "2023-11-14T22:13:20+00:00"
        store.backend.get_document.assert_called_once_with(
            "mem-embedded", schema_name="agent_memories"
        )

    @pytest.mark.unit
    def test_list_raises_on_backend_failure(self):
        """A swallowed list() outage reads as an empty partition, so every
        enumerate-and-filter caller mistakes it for no-data and truncates."""
        store = self._store()
        store.backend.query_metadata_documents.side_effect = ConnectionError(
            "backend down"
        )

        with pytest.raises(ConnectionError):
            store.list(limit=100)
        with pytest.raises(ConnectionError):
            store.list(limit=None)


class _PagingBackend:
    """Backend double that honors YQL ``limit``/``offset`` over an ordered
    row set — models Vespa's paged query so the store's walk can be driven
    deterministically without a live cluster.
    """

    def __init__(self, rows: List[Dict[str, Any]], *, ignore_offset: bool = False):
        self._by_id = sorted(rows, key=lambda r: r["id"], reverse=True)
        self._by_recency = sorted(
            rows, key=lambda r: (r["created_at"], r["id"]), reverse=True
        )
        self.ignore_offset = ignore_offset
        self.query_count = 0
        self.yqls: List[str] = []

    def query_metadata_documents(self, schema, yql=None, **kwargs):
        self.query_count += 1
        self.yqls.append(yql)
        # Page size and offset ride as hits/offset, not in the YQL.
        limit = int(kwargs["hits"])
        offset = 0 if self.ignore_offset else int(kwargs.get("offset", 0))
        # Serve the order the query asked for so the walk's unique-id ordering
        # is exercised, not just the loop.
        ordered = self._by_recency if "created_at" in yql else self._by_id
        return ordered[offset : offset + limit]


def _paging_store(backend: _PagingBackend) -> BackendVectorStore:
    store = object.__new__(BackendVectorStore)
    store.backend = backend
    store.collection_name = "agent_memories_acme"
    store.profile = "agent_memories"
    store.is_telemetry = False
    return store


def _rows(n: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"m{i:04d}",
            "created_at": 1_700_000_000 + i,
            "text": f"row {i}",
            "user_id": "u",
            "agent_id": "a",
            "metadata_": json.dumps({"n": i}),
        }
        for i in range(n)
    ]


@pytest.mark.unit
class TestListPagination:
    def test_walks_every_page_with_no_duplicates(self):
        """limit=None returns the whole partition across page boundaries."""
        backend = _PagingBackend(_rows(250))
        store = _paging_store(backend)

        results, next_offset = store.list(limit=None)

        assert next_offset is None
        ids = [r.id for r in results]
        assert len(ids) == 250
        assert len(set(ids)) == 250  # no row seen twice across pages
        # Newest first: m0249 down to m0000.
        assert ids[0] == "m0249"
        assert ids[-1] == "m0000"
        assert backend.query_count == 3  # 100 + 100 + 50

    def test_bounded_read_is_the_newest_slice(self):
        backend = _PagingBackend(_rows(250))
        store = _paging_store(backend)

        results, next_offset = store.list(limit=10)

        assert [r.id for r in results] == [f"m{249 - i:04d}" for i in range(10)]
        assert next_offset == 10

    def test_walk_orders_by_unique_id_bounded_by_recency(self):
        """The full walk must page by the unique id (a total order offset
        paging cannot skip across), while a bounded read stays most-recent
        first — a walk ordered by the second-granularity created_at would
        shuffle tied rows between page queries and drop them."""
        walk_backend = _PagingBackend(_rows(150))
        _paging_store(walk_backend).list(limit=None)
        assert all("order by id desc" in q for q in walk_backend.yqls)
        assert all("created_at" not in q for q in walk_backend.yqls)

        bounded_backend = _PagingBackend(_rows(150))
        _paging_store(bounded_backend).list(limit=10)
        assert "order by created_at desc, id desc" in bounded_backend.yqls[0]

    def test_pagination_guard_raises_when_offset_ignored(self):
        """A backend that ignores offset would loop forever or silently
        truncate; the walk must fail loud instead."""
        backend = _PagingBackend(_rows(250), ignore_offset=True)
        store = _paging_store(backend)

        with pytest.raises(RuntimeError, match="did not advance"):
            store.list(limit=None)

    def test_concurrent_walks_are_isolated(self):
        """N threads walking one store see the complete partition each, with
        no cross-talk — the walk holds only call-local state."""
        import threading

        backend = _PagingBackend(_rows(250))
        store = _paging_store(backend)

        barrier = threading.Barrier(8)
        results: Dict[int, List[str]] = {}
        errors: List[Exception] = []

        def worker(tid: int):
            try:
                barrier.wait()
                rows, _ = store.list(limit=None)
                results[tid] = [r.id for r in rows]
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        expected = [f"m{249 - i:04d}" for i in range(250)]
        assert len(results) == 8
        for tid, ids in results.items():
            assert ids == expected, f"thread {tid} saw an incomplete/reordered walk"
