"""Real-Vespa round-trip for VespaBackend.ingest_stream.

ingest_stream batches an iterator of Documents and yields one result per batch,
threading the target schema into every ingest_documents call. The prior
signature took no schema, and its internal ingest_documents(batch) call omitted
the required schema_name argument, so streaming raised TypeError and no document
could ever be fed this way. This feeds 3 docs with batch_size=2 (one full batch
plus a remainder — exercising both the mid-stream and final-batch feeds) and
asserts every document lands and reads back from a live Vespa.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document

pytestmark = pytest.mark.integration

_EMBED = np.full((768,), 0.05, dtype=np.float32)


def _memory_doc(doc_id: str, text: str) -> Document:
    doc = Document(id=doc_id, content_type=ContentType.TEXT, content_id=doc_id)
    doc.text_content = text
    doc.add_metadata("user_id", "u1")
    doc.add_embedding("embedding", _EMBED)
    return doc


@pytest.fixture
def ready_backend(vespa_instance):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )

    tenant = f"istream{uuid.uuid4().hex[:6]}"
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    backend.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name="agent_memories"
    )

    # prepareandactivate returns before content nodes activate the new schema;
    # feed a probe (via the working single-batch path) until it lands.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        result = backend.ingest_documents(
            [_memory_doc("__ready__", "ready")], "agent_memories"
        )
        if (
            result["success_count"] == 1
            and backend.get_document("__ready__", schema_name="agent_memories")
            is not None
        ):
            break
        time.sleep(2)
    else:
        pytest.fail("agent_memories not feedable within 90s of deploy")

    return backend


@pytest.mark.integration
def test_ingest_stream_threads_schema_into_every_batch(ready_backend):
    backend = ready_backend
    docs = [_memory_doc(f"istream-{i}", f"memory number {i}") for i in range(3)]

    # batch_size=2 → a full batch (istream-0, istream-1) then a remainder
    # (istream-2). Both yields call ingest_documents, which needs the schema.
    results = list(backend.ingest_stream(iter(docs), "agent_memories", batch_size=2))

    assert len(results) == 2, results
    assert sum(r["success_count"] for r in results) == 3
    assert all(r["failed_count"] == 0 for r in results)

    for i in range(3):
        stored = backend.get_document(f"istream-{i}", schema_name="agent_memories")
        assert stored is not None, f"istream-{i} missing after ingest_stream"
        assert stored.text_content == f"memory number {i}"
