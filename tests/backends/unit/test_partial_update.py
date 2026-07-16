"""Vespa update path must use a partial update, not a full PUT-replace.

A metadata-only update (mem0 ``vector=None``) builds a Document with no
embedding field. With the old full-PUT feed that wiped the stored embedding /
embedding_binary tensors, making the memory invisible to vector search. The
fix routes updates through ``operation_type="update"`` (partial assign — only
present fields are written), so embeddings survive. These pin the wiring;
Vespa's documented partial-update semantics guarantee the field preservation.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.ingestion_client import VespaPyClient


@pytest.mark.unit
class TestVespaPartialUpdate:
    def test_feed_prepared_batch_forwards_operation_type(self):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        client = VespaPyClient(
            {
                "schema_name": "video_colpali_smol500_mv_frame",
                "url": "http://localhost",
                "port": 8080,
                "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
            }
        )
        client.app = MagicMock()
        client._connected = True

        client._feed_prepared_batch(
            [{"put": "id:content:s::d1", "fields": {"video_id": "v1"}}],
            operation_type="update",
        )

        assert client.app.feed_iterable.call_args.kwargs["operation_type"] == "update"

    def test_feed_defaults_to_full_feed(self):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        client = VespaPyClient(
            {
                "schema_name": "video_colpali_smol500_mv_frame",
                "url": "http://localhost",
                "port": 8080,
                "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
            }
        )
        client.app = MagicMock()
        client._connected = True

        client._feed_prepared_batch(
            [{"put": "id:content:s::d1", "fields": {"video_id": "v1"}}]
        )

        assert client.app.feed_iterable.call_args.kwargs["operation_type"] == "feed"

    def test_update_document_requests_partial_update(self):
        backend = object.__new__(VespaBackend)
        backend.config = {"schema_name": "s"}
        backend.ingest_documents = MagicMock(return_value={"success_count": 1})
        doc = MagicMock()
        doc.id = "d1"

        ok = backend.update_document("d1", doc, schema_name="s")

        assert ok is True
        assert backend.ingest_documents.call_args.kwargs["operation_type"] == "update"

    def test_update_document_id_mismatch_returns_false(self):
        """If the caller's document_id disagrees with document.id, the partial
        update would land on the wrong doc id — must fail loudly, not silently."""
        backend = object.__new__(VespaBackend)
        backend.config = {"schema_name": "s"}
        backend.ingest_documents = MagicMock(return_value={"success_count": 1})
        doc = MagicMock()
        doc.id = "actual-doc-id"

        ok = backend.update_document("wrong-id", doc, schema_name="s")

        assert ok is False
        backend.ingest_documents.assert_not_called()

    def test_delete_metadata_document_returns_false_on_non_200(self, monkeypatch):
        """Belt-and-braces branch: real pyvespa delete_data raise_for_status
        RAISES VespaError on every 4xx/5xx except 404, so a returned non-200
        response only occurs for a 404 (or a pyvespa contract change). The
        status check must still map it to False."""
        from cogniverse_vespa import backend as vespa_backend_mod

        backend = object.__new__(VespaBackend)
        backend._url = "http://localhost"
        backend._port = 8080
        backend._metadata_app = None
        backend._metadata_app_key = None

        client = MagicMock()
        client.delete_data = MagicMock(return_value=MagicMock(status_code=500))
        monkeypatch.setattr(
            vespa_backend_mod, "make_persistent_vespa_ops", lambda **_kw: client
        )

        ok = backend.delete_metadata_document(schema="s", doc_id="d1")

        assert ok is False
        client.delete_data.assert_called_once()

    def test_delete_metadata_document_returns_false_when_pyvespa_raises(
        self, monkeypatch
    ):
        """The REAL error shape: pyvespa raise_for_status raises VespaError on
        4xx/5xx — the raise path must map to False, same as the status check.
        Previously only the unreal returns-non-200 shape was covered."""
        from vespa.exceptions import VespaError

        from cogniverse_vespa import backend as vespa_backend_mod

        backend = object.__new__(VespaBackend)
        backend._url = "http://localhost"
        backend._port = 8080
        backend._metadata_app = None
        backend._metadata_app_key = None

        client = MagicMock()
        client.delete_data = MagicMock(side_effect=VespaError("500 backend error"))
        monkeypatch.setattr(
            vespa_backend_mod, "make_persistent_vespa_ops", lambda **_kw: client
        )

        ok = backend.delete_metadata_document(schema="s", doc_id="d1")

        assert ok is False
        client.delete_data.assert_called_once()


@pytest.mark.unit
class TestVespaTimestampOnPartialUpdate:
    """process() stamped created_at unconditionally; on a partial update that
    assign overwrote the original creation time on every metadata-only mem0
    update. It must omit an absent timestamp on update, stamp it on a full
    feed, and always honour a caller-supplied value.
    """

    def _memory_client(self):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        return VespaPyClient(
            {
                "schema_name": "agent_memories",
                "url": "http://localhost",
                "port": 8080,
                "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
            }
        )

    def _doc(self, **metadata):
        from cogniverse_sdk.document import ContentType, Document

        doc = Document(id="mem1", content_type=ContentType.TEXT, content_id="mem1")
        for key, value in metadata.items():
            doc.add_metadata(key, value)
        return doc

    def test_partial_update_omits_unset_created_at(self):
        client = self._memory_client()
        assert "created_at" in client.schema_fields  # guards the branch under test
        fields = client.process(self._doc(), operation_type="update")["fields"]
        # Omitted -> the partial assign leaves the stored created_at untouched.
        assert "created_at" not in fields

    def test_full_feed_stamps_created_at(self):
        client = self._memory_client()
        fields = client.process(self._doc(), operation_type="feed")["fields"]
        assert isinstance(fields["created_at"], int)
        assert fields["created_at"] > 0

    def test_caller_supplied_created_at_honoured_on_update(self):
        client = self._memory_client()
        fields = client.process(self._doc(created_at=12345), operation_type="update")[
            "fields"
        ]
        assert fields["created_at"] == 12345

    def test_numpy_supplied_created_at_honoured(self):
        """np.int64 is not an int subclass — the timestamp gate missed numpy
        epochs and stamped now() instead of the supplied value (back-dating,
        re-ingest, and migration callers)."""
        import numpy as np

        client = self._memory_client()
        fields = client.process(
            self._doc(created_at=np.int64(12345)), operation_type="feed"
        )["fields"]
        assert fields["created_at"] == 12345


@pytest.mark.unit
class TestYqlQuote:
    """YQL ``contains "value"`` interpolations must escape ``\\`` and ``"`` or
    a value with either character either breaks the query (HTTP 400) or opens
    a YQL injection. ``yql_quote`` is the single shared escape used by the
    search backend, adapter store, and config store."""

    def test_plain_value_is_double_quoted(self):
        from cogniverse_vespa._yql import yql_quote

        assert yql_quote("plain") == '"plain"'

    def test_double_quote_is_escaped(self):
        from cogniverse_vespa._yql import yql_quote

        # foo"bar -> "foo\"bar"
        assert yql_quote('foo"bar') == '"foo\\"bar"'

    def test_backslash_is_escaped(self):
        from cogniverse_vespa._yql import yql_quote

        # foo\bar -> "foo\\bar"
        assert yql_quote("foo\\bar") == '"foo\\\\bar"'

    def test_non_string_value_is_stringified(self):
        from cogniverse_vespa._yql import yql_quote

        assert yql_quote(42) == '"42"'


@pytest.mark.unit
class TestVisibilitySweep:
    """The post-feed visibility wait must probe docs in sweeps over one
    keep-alive session — one backoff sleep per sweep, per-batch deadline —
    instead of a fresh connection and its own sleep loop per document."""

    def _backend(self, docs_visible_after: dict) -> tuple:
        """Build a VespaBackend whose feed succeeds for every doc, plus a
        fake requests module recording probes. ``docs_visible_after[doc_id]``
        is how many 404s the probe returns before the doc turns 200."""
        from unittest.mock import MagicMock

        backend = object.__new__(VespaBackend)
        backend.config = {"wait_for_indexing": True, "indexing_timeout": 30.0}
        backend._url = "http://localhost"
        backend._port = 8080
        backend._tenant_id = None

        client = MagicMock()
        client.namespace = "content"
        client.process.side_effect = lambda doc, operation_type: {
            "put": doc.id,
            "fields": {},
        }
        client._feed_prepared_batch.return_value = (len(docs_visible_after), [])
        backend._get_or_create_ingestion_client = MagicMock(return_value=client)

        probes: list[str] = []
        remaining_404s = dict(docs_visible_after)

        class _Resp:
            def __init__(self, status_code):
                self.status_code = status_code
                self.text = ""

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, timeout=None):
                doc_id = url.rsplit("/", 1)[-1]
                probes.append(doc_id)
                if remaining_404s.get(doc_id, 0) > 0:
                    remaining_404s[doc_id] -= 1
                    return _Resp(404)
                return _Resp(200)

        return backend, _Session, probes

    def _docs(self, *ids):
        from unittest.mock import MagicMock

        docs = []
        for doc_id in ids:
            d = MagicMock()
            d.id = doc_id
            docs.append(d)
        return docs

    def test_all_visible_probes_each_doc_exactly_once_without_sleeping(
        self, monkeypatch
    ):
        import time

        import requests

        backend, session_cls, probes = self._backend({"d1": 0, "d2": 0, "d3": 0})
        monkeypatch.setattr(requests, "Session", session_cls)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        result = backend.ingest_documents(self._docs("d1", "d2", "d3"), "schema_x")

        assert result["success_count"] == 3
        assert sorted(probes) == ["d1", "d2", "d3"]
        assert sleeps == []

    def test_lagging_doc_sleeps_per_sweep_not_per_doc(self, monkeypatch):
        import time

        import requests

        backend, session_cls, probes = self._backend({"d1": 0, "d2": 0, "d3": 2})
        monkeypatch.setattr(requests, "Session", session_cls)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        backend.ingest_documents(self._docs("d1", "d2", "d3"), "schema_x")

        # Sweep 1 probes all three (d3 404s); sweeps 2-3 probe only d3.
        assert probes == ["d1", "d2", "d3", "d3", "d3"]
        # Two backoffs total — shared by the sweep, not one loop per doc.
        assert sleeps == [0.5, 0.5]
