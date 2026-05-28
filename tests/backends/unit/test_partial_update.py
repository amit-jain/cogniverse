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
        """pyvespa.delete_data does not raise on 4xx/5xx; the backend must check
        the status_code, not assume success on no-exception."""
        from cogniverse_vespa import backend as vespa_backend_mod

        backend = object.__new__(VespaBackend)
        backend._url = "http://localhost"
        backend._port = 8080

        client = MagicMock()
        client.delete_data = MagicMock(return_value=MagicMock(status_code=500))
        monkeypatch.setattr(vespa_backend_mod, "make_vespa_app", lambda **_kw: client)

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
