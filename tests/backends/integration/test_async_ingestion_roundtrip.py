"""Real-Vespa round-trip for use_async_ingestion.

The flag existed but did nothing: constructing a VespaBackend with
use_async_ingestion=True raised ImportError (it gated on a non-existent
async_ingestion_client module), so the async feed path could never run. Async
feed is built into pyvespa (feed_async_iterable, an HTTP/2 async feeder callable
from sync code). This asserts that with the flag set, the backend constructs,
routes feeds through feed_async_iterable, and the documents land in Vespa.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import vespa.application as vespa_app

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


def _build_async_backend(vespa_instance):
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

    tenant = f"async{uuid.uuid4().hex[:6]}"
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "use_async_ingestion": True,
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            },
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    backend.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name="agent_memories"
    )
    return backend


@pytest.mark.integration
def test_async_ingestion_feeds_via_pyvespa_async_and_lands(vespa_instance, monkeypatch):
    async_calls = {"n": 0}
    original = vespa_app.Vespa.feed_async_iterable

    def counting_async_feed(self, *args, **kwargs):
        async_calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(vespa_app.Vespa, "feed_async_iterable", counting_async_feed)

    backend = _build_async_backend(vespa_instance)

    # prepareandactivate returns before content nodes activate; retry until the
    # first async feed lands.
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
        pytest.fail("async ingestion never landed within 90s of deploy")

    docs = [_memory_doc(f"async-{i}", f"memory number {i}") for i in range(3)]
    result = backend.ingest_documents(docs, "agent_memories")
    assert result["success_count"] == 3, result
    assert result["failed_count"] == 0, result

    # The async feeder was actually exercised (not the sync feed_iterable).
    assert async_calls["n"] >= 1, "feed_async_iterable was never called"

    for i in range(3):
        stored = backend.get_document(f"async-{i}", schema_name="agent_memories")
        assert stored is not None, f"async-{i} missing after async ingest"
        assert stored.text_content == f"memory number {i}"
