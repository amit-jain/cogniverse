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

        ok = backend.update_document("d1", MagicMock(), schema_name="s")

        assert ok is True
        assert (
            backend.ingest_documents.call_args.kwargs["operation_type"] == "update"
        )
