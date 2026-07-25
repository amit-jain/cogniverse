"""Real-Vespa round-trip for VespaPyClient feed operation_type semantics.

tests/backends/unit/test_partial_update.py only asserts the operation_type
kwarg forwarded to pyvespa. These prove the field-level effect on a live
Vespa: operation_type="update" assigns only the present fields and leaves the
stored embedding tensor intact (the mem0 metadata-only update case), while
operation_type="feed" replaces the whole document and drops an omitted field.
"""

import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import vespa.application as vespa_app

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document
from cogniverse_vespa.ingestion_client import VespaPyClient

# shared_vespa resolves via the conftest re-export. Importing it here would
# define a second module-level FixtureDef with its own session cache — pytest
# then boots a SECOND Vespa container mid-sweep, and cross-container schema
# wiring breaks every later multi-tenant test in the run.
from tests.utils.vespa_test_helpers import deploy_tenant_schema

logger = logging.getLogger(__name__)

TENANT_ID = "partial_update_rt"
EMBED = np.full((768,), 0.05, dtype=np.float32)


def _build_backend(shared_vespa):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=shared_vespa["http_port"]
    )
    config_manager = ConfigManager(store=store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_vespa["http_port"],
        )
    )
    tenant = f"feedfault{uuid.uuid4().hex[:6]}"
    return BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "wait_for_indexing": False,
            "backend": {
                "url": "http://localhost",
                "config_port": shared_vespa["config_port"],
                "port": shared_vespa["http_port"],
            },
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )


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

    def test_mid_batch_connection_loss_reports_exact_nonpersisted_ids(
        self, shared_vespa, monkeypatch
    ):
        """Returned failed IDs must match real Document-v1 state after an
        out-of-order concurrent feed loses its Vespa connection."""
        backend = _build_backend(shared_vespa)
        client = backend._get_or_create_ingestion_client("agent_memories")

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            ready = _memory_doc("__fault_ready__", "ready", with_embedding=True)
            result = backend.ingest_documents([ready], "agent_memories")
            if (
                result["success_count"] == 1
                and client.get_document_data(ready.id) is not None
            ):
                break
            time.sleep(2)
        else:
            pytest.fail("fault-test tenant schema was not feedable within 90s")

        ids = [f"fault-{i:03d}" for i in range(80)]
        payload = "x" * (256 * 1024)
        docs = [
            _memory_doc(doc_id, f"{doc_id}:{payload}", with_embedding=True)
            for doc_id in ids
        ]

        original = vespa_app.Vespa.feed_iterable
        pause_lock = threading.Lock()
        paused = threading.Event()
        release_thread = None

        def feed_nonprefix_then_abort(self, *args, **kwargs):
            callback = kwargs["callback"]
            all_docs = list(kwargs["iter"])
            # Submit a deliberately non-prefix subset to the real concurrent
            # feeder. The remaining IDs model work not yet submitted when the
            # connection-level batch abort occurs.
            kwargs["iter"] = iter(all_docs[::4])

            def callback_then_pause(response, doc_id):
                nonlocal release_thread
                callback(response, doc_id)
                if response.is_successful() and not paused.is_set():
                    with pause_lock:
                        if not paused.is_set():
                            result = subprocess.run(
                                [
                                    "docker",
                                    "pause",
                                    shared_vespa["container_name"],
                                ],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            assert result.returncode == 0, result.stderr
                            paused.set()

                            def release_after_transport_timeout():
                                time.sleep(5)
                                subprocess.run(
                                    [
                                        "docker",
                                        "unpause",
                                        shared_vespa["container_name"],
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                )

                            release_thread = threading.Thread(
                                target=release_after_transport_timeout,
                                daemon=True,
                            )
                            release_thread.start()

            kwargs["callback"] = callback_then_pause
            original(self, *args, **kwargs)
            raise ConnectionError("connection lost before remaining submissions")

        monkeypatch.setattr(vespa_app.Vespa, "feed_iterable", feed_nonprefix_then_abort)

        try:
            result = backend.ingest_documents(docs, "agent_memories")
            assert paused.is_set(), "fault injection never paused Vespa"
        finally:
            if paused.is_set():
                subprocess.run(
                    ["docker", "unpause", shared_vespa["container_name"]],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            if release_thread is not None:
                release_thread.join(timeout=30)

        persisted = {
            doc_id for doc_id in ids if client.get_document_data(doc_id) is not None
        }
        assert 0 < len(persisted) < len(ids)
        assert result["success_count"] == len(persisted)
        assert set(result["failed_documents"]) == set(ids) - persisted

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
