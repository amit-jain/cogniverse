"""Real-Vespa round-trip for VespaPyClient feed operation_type semantics.

tests/backends/unit/test_partial_update.py only asserts the operation_type
kwarg forwarded to pyvespa. These prove the field-level effect on a live
Vespa: operation_type="update" assigns only the present fields and leaves the
stored embedding tensor intact (the mem0 metadata-only update case), while
operation_type="feed" replaces the whole document and drops an omitted field.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document
from cogniverse_vespa.ingestion_client import VespaPyClient

# Re-export the canonical session-scoped Vespa.
from tests.conftest import shared_vespa  # noqa: F401
from tests.utils.vespa_test_helpers import deploy_tenant_schema

logger = logging.getLogger(__name__)

TENANT_ID = "partial_update_rt"
EMBED = np.full((768,), 0.05, dtype=np.float32)


def _memory_doc(doc_id: str, text: str, *, with_embedding: bool) -> Document:
    doc = Document(id=doc_id, content_type=ContentType.TEXT, content_id=doc_id)
    doc.text_content = text
    doc.add_metadata("user_id", "u1")
    if with_embedding:
        doc.add_embedding("embedding", EMBED)
    return doc


def _embedding_values(emb) -> list:
    """Normalise a Document v1 tensor field to a flat list of values."""
    if emb is None:
        return []
    if isinstance(emb, dict):
        if "values" in emb:
            return list(emb["values"])
        if "cells" in emb:
            return [c["value"] for c in emb["cells"]]
        return list(emb.values())
    return list(emb)


@pytest.fixture(scope="module")
def memory_client(shared_vespa):  # noqa: F811
    full_name = deploy_tenant_schema(
        shared_vespa, tenant_id=TENANT_ID, base_schema_name="agent_memories"
    )
    client = VespaPyClient(
        {
            "schema_name": full_name,
            "base_schema_name": "agent_memories",
            "url": "http://localhost",
            "port": shared_vespa["http_port"],
            "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
        }
    )
    assert client.connect(), "VespaPyClient failed to connect to shared_vespa"

    # Vespa's prepareandactivate returns before content nodes finish activating
    # the new schema; retry the first feed until it lands.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        prepared = client.process(
            _memory_doc("__ready__", "ready", with_embedding=True)
        )
        success, _failed = client._feed_prepared_batch([prepared])
        if success == 1 and client.get_document_data("__ready__") is not None:
            break
        time.sleep(2)
    else:
        pytest.fail(f"{full_name} not feedable within 90s of deploy")

    return client


@pytest.mark.integration
class TestPartialUpdateRoundTrip:
    def test_partial_update_preserves_embedding(self, memory_client):
        c = memory_client

        prepared = c.process(_memory_doc("mem-pu", "original", with_embedding=True))
        success, failed = c._feed_prepared_batch([prepared], operation_type="feed")
        assert success == 1, failed

        before = c.get_document_data("mem-pu")
        assert before["text"] == "original"
        assert len(_embedding_values(before["embedding"])) == 768

        # Metadata-only update (no embedding field), partial assign.
        update = c.process(
            _memory_doc("mem-pu", "updated", with_embedding=False),
            operation_type="update",
        )
        assert "embedding" not in update["fields"]
        success, failed = c._feed_prepared_batch([update], operation_type="update")
        assert success == 1, failed

        after = c.get_document_data("mem-pu")
        assert after["text"] == "updated"
        # The stored embedding survived the metadata-only update.
        survived = _embedding_values(after["embedding"])
        assert len(survived) == 768
        assert survived == pytest.approx([0.05] * 768, abs=1e-3)

    def test_full_feed_replaces_and_drops_omitted_embedding(self, memory_client):
        c = memory_client

        prepared = c.process(_memory_doc("mem-ff", "original", with_embedding=True))
        success, failed = c._feed_prepared_batch([prepared], operation_type="feed")
        assert success == 1, failed
        assert len(_embedding_values(c.get_document_data("mem-ff")["embedding"])) == 768

        # Full feed of the same id with no embedding replaces the whole document.
        replace = c.process(_memory_doc("mem-ff", "replaced", with_embedding=False))
        assert "embedding" not in replace["fields"]
        success, failed = c._feed_prepared_batch([replace], operation_type="feed")
        assert success == 1, failed

        after = c.get_document_data("mem-ff")
        assert after["text"] == "replaced"
        # A full PUT-replace dropped the embedding the new payload omitted.
        assert _embedding_values(after.get("embedding")) == []
