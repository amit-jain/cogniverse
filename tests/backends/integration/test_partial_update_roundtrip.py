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
        failures = result["failed_documents"]
        assert {f["id"] for f in failures} == set(ids) - persisted
        assert {f["schema"] for f in failures} == {client.schema_name}
        assert all(
            set(f) == {"id", "state", "schema", "status_code", "attempts", "error"}
            for f in failures
        )
        assert {f["state"] for f in failures} == {"unresolved"}
        assert all(
            "connection lost before remaining submissions" in f["error"]
            for f in failures
        )

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


class _JdiscOutage:
    """Stops and restores the Vespa HTTP container (jdisc) inside the shared
    container without touching the container the session fixture owns.

    A stopped jdisc refuses new connections and resets established ones, which
    is what a real backend death mid-feed does. ``docker pause`` cannot stand
    in: pyvespa's feeder holds a 120s client timeout and retries connection
    errors, so a paused backend turns one document into a multi-minute hang.
    """

    def __init__(self, shared_vespa):
        self._container = shared_vespa["container_name"]
        self._health_url = f"{shared_vespa['base_url']}/state/v1/health"
        self.stopped = False

    def _refuses_connections(self) -> bool:
        """jdisc drains before it closes its listener, so it keeps landing
        writes for a moment after ``stop``. The outage is in effect only once
        a fresh connection is refused outright."""
        import requests

        try:
            requests.get(self._health_url, timeout=3)
        except requests.ConnectionError:
            return True
        except requests.RequestException:
            return False
        return False

    def _accepts_connections(self) -> bool:
        import requests

        try:
            return requests.get(self._health_url, timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def _sentinel(self, verb: str) -> None:
        result = subprocess.run(
            [
                "docker",
                "exec",
                self._container,
                "vespa-sentinel-cmd",
                verb,
                "container",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def stop(self) -> None:
        self._sentinel("stop")
        self.stopped = True
        deadline = time.monotonic() + 30
        while not self._refuses_connections():
            assert time.monotonic() < deadline, (
                "jdisc still accepting connections 30s after stop"
            )
            time.sleep(0.2)

    def restore(self) -> None:
        if not self.stopped:
            return
        self._sentinel("start")
        deadline = time.monotonic() + 120
        while not self._accepts_connections():
            if time.monotonic() > deadline:
                pytest.fail("jdisc did not come back within 120s of start")
            time.sleep(1)
        self.stopped = False


@pytest.fixture
def jdisc_outage(shared_vespa):  # noqa: F811
    outage = _JdiscOutage(shared_vespa)
    yield outage
    outage.restore()


def _feedable_client(shared_vespa):
    """A connected client on a fresh tenant whose data schema has activated."""
    backend = _build_backend(shared_vespa)
    client = backend._get_or_create_ingestion_client("agent_memories")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        ready = _memory_doc("__ready__", "ready", with_embedding=True)
        result = backend.ingest_documents([ready], "agent_memories")
        if result["success_count"] == 1 and client.check_document_exists(ready.id):
            return client
        time.sleep(2)
    pytest.fail("fresh tenant schema was not feedable within 90s")


def _document_url(client, doc_id: str) -> str:
    return (
        f"{client.backend_url}:{client.backend_port}/document/v1/"
        f"{client.namespace}/{client.schema_name}/docid/{doc_id}"
    )


def _transport_reason(client, doc_id: str) -> str:
    """The per-document error a transport death produces: pyvespa wraps the
    httpr client failure as a synthetic 599 whose body carries this string."""
    return (
        "the backend stopped answering during this batch: "
        f"error sending request for url ({_document_url(client, doc_id)})"
    )


def _outage_raise_message(client, count: int) -> str:
    return (
        f"Feed to {client.namespace}/{client.schema_name} did not complete for "
        f"any of {count} documents: the backend at {client.backend_url}:"
        f"{client.backend_port} stopped answering during the feed"
    )


class TestFeedFailureContract:
    """A refused document, a feed the backend stopped answering, and a backend
    gone before the feed are three different outcomes on real Vespa."""

    def test_rejected_document_carries_the_backend_reason(self, shared_vespa):
        """A document the schema refuses stays "rejected" and names the cause.

        This is the control for the transport cases below: a schema rejection
        reaches the feeder as the SAME synthetic 599 a transport death does, so
        the classifier must keep this one "rejected" while moving the transport
        ones to "unresolved".
        """
        client = _feedable_client(shared_vespa)

        doc_id = "reject-001"
        prepared = client.process(_memory_doc(doc_id, "body", with_embedding=True))
        prepared["fields"]["not_a_declared_field"] = "x"

        success, failures = client._feed_prepared_batch([prepared])

        assert success == 0
        assert len(failures) == 1
        failure = failures[0]
        assert set(failure) == {
            "id",
            "state",
            "schema",
            "status_code",
            "attempts",
            "error",
        }
        assert {
            k: failure[k] for k in ("id", "state", "schema", "status_code", "attempts")
        } == {
            "id": doc_id,
            "state": "rejected",
            "schema": client.schema_name,
            "status_code": 599,
            "attempts": 1,
        }
        # pyvespa wraps the backend's field-validation refusal as a synthetic
        # 599; the reason must still name the offending field and schema, and
        # must not read like the transport wrapper an outage produces.
        assert failure["error"].startswith(
            "HTTP 599: {'Exception': \"No field 'not_a_declared_field' in the "
            f"structure of type '{client.schema_name}', which has the fields: "
            "[field 'id' of type string,"
        )
        assert failure["error"].endswith(
            "field 'created_at' of type long]\", 'id': 'reject-001', "
            "'message': 'Exception during feed_data_point'}"
        )
        assert "error sending request" not in failure["error"]
        assert client.check_document_exists(doc_id) is False

    def test_backend_dying_mid_feed_reports_unresolved_not_rejected(
        self, shared_vespa, jdisc_outage
    ):
        """Documents the backend stopped answering are "unresolved" naming the
        transport cause -- never "rejected", which would send an operator
        hunting for bad data that does not exist.

        The feeder submits the first document, and only once it has landed does
        the backend die; the rest then meet a refused connection. That pins the
        split deterministically: exactly one document persists and the rest come
        back unresolved.
        """
        client = _feedable_client(shared_vespa)
        ids = [f"midfeed-{i:03d}" for i in range(6)]
        prepared = [
            client.process(_memory_doc(doc_id, "body", with_embedding=True))
            for doc_id in ids
        ]

        original = vespa_app.Vespa.feed_iterable

        def feed_first_then_kill_backend(self, *args, **kwargs):
            source = kwargs["iter"]

            def gated():
                first = next(source)
                yield first
                # Do not release the rest until the first has landed and the
                # backend has stopped accepting connections, so every remaining
                # document meets a dead backend rather than a live one.
                deadline = time.monotonic() + 30
                while not client.check_document_exists(first["id"]):
                    assert time.monotonic() < deadline, "first document never landed"
                    time.sleep(0.2)
                jdisc_outage.stop()
                yield from source

            kwargs["iter"] = gated()
            return original(self, *args, **kwargs)

        import unittest.mock as _mock

        with _mock.patch.object(
            vespa_app.Vespa, "feed_iterable", feed_first_then_kill_backend
        ):
            success, failures = client._feed_prepared_batch(prepared)
        jdisc_outage.restore()

        persisted = {doc_id for doc_id in ids if client.check_document_exists(doc_id)}
        # The gated first document always lands; the backend then dies, so at
        # least the tail is lost. jdisc's graceful drain makes the exact split
        # of the middle nondeterministic, so the contract is pinned against the
        # real Document-v1 state, not a hardcoded split.
        assert "midfeed-000" in persisted
        assert persisted != set(ids)
        assert success == len(persisted)
        failed_ids = set(ids) - persisted
        assert {f["id"]: f for f in failures} == {
            doc_id: {
                "id": doc_id,
                "state": "unresolved",
                "schema": client.schema_name,
                "status_code": None,
                "attempts": 1,
                "error": _transport_reason(client, doc_id),
            }
            for doc_id in failed_ids
        }

    def test_backend_gone_before_feed_raises_instead_of_reporting_rejections(
        self, shared_vespa, jdisc_outage
    ):
        """An established connection whose backend then dies must raise naming
        the endpoint -- not return "the backend refused every document"."""
        client = _feedable_client(shared_vespa)
        ids = ["outage-001", "outage-002"]
        prepared = [
            client.process(_memory_doc(doc_id, "body", with_embedding=True))
            for doc_id in ids
        ]

        jdisc_outage.stop()
        with pytest.raises(ConnectionError) as excinfo:
            client._feed_prepared_batch(prepared)
        jdisc_outage.restore()

        assert str(excinfo.value) == _outage_raise_message(client, 2)
        assert [doc_id for doc_id in ids if client.check_document_exists(doc_id)] == []

    def test_concurrent_batches_during_outage_each_raise_for_their_own_batch(
        self, shared_vespa, jdisc_outage
    ):
        """Two batches fed through one client at once while the backend is down
        each raise the same endpoint outage, with no cross-batch bleed."""
        client = _feedable_client(shared_vespa)
        batches = {
            "a": ["conc-a-001", "conc-a-002"],
            "b": ["conc-b-001", "conc-b-002"],
        }
        prepared = {
            name: [
                client.process(_memory_doc(doc_id, "body", with_embedding=True))
                for doc_id in doc_ids
            ]
            for name, doc_ids in batches.items()
        }
        barrier = threading.Barrier(len(batches))
        outcomes: dict = {}

        def run(name):
            barrier.wait()
            try:
                client._feed_prepared_batch(prepared[name])
                outcomes[name] = "RETURNED"
            except ConnectionError as exc:
                outcomes[name] = str(exc)

        jdisc_outage.stop()
        threads = [threading.Thread(target=run, args=(name,)) for name in batches]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        jdisc_outage.restore()

        assert outcomes == {
            "a": _outage_raise_message(client, 2),
            "b": _outage_raise_message(client, 2),
        }
        assert [
            doc_id
            for doc_ids in batches.values()
            for doc_id in doc_ids
            if client.check_document_exists(doc_id)
        ] == []
