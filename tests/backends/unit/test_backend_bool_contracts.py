"""VespaBackend's bool-contracted surfaces return honest booleans.

``validate_schema`` used to be a validator that never validated (its try body
was a comment plus an unconditional ``return True``), and ``health_check``
(typed ``-> bool`` by the SearchBackend ABC) leaked the search backend's
status DICT — always truthy, even when degraded.
"""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
import requests

from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_backend():
    backend = object.__new__(VespaBackend)
    backend.schema_manager = MagicMock()
    backend._vespa_search_backend = None
    return backend


def _dead_port() -> int:
    """A local TCP port with nothing listening on it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_validate_schema_checks_deployed_document_types():
    backend = _bare_backend()
    backend.schema_manager.list_deployed_document_types.return_value = [
        "agent_memories",
        "video_colpali_smol500_mv_frame",
    ]

    assert backend.validate_schema("agent_memories") is True
    assert backend.validate_schema("nonexistent_schema") is False


def test_validate_schema_raises_on_listing_failure():
    """An enumeration outage is not "schema invalid" — flattening it to False
    collapses the two for every caller keying off the bool.

    The failure is injected at the real config-server HTTP seam (a dead port)
    rather than via a mock side_effect: the mock raises regardless of how
    ``list_deployed_document_types`` is called, hiding that the real method
    swallows the probe failure to ``[]`` unless ``raise_on_failure=True`` is
    passed. Against the real seam the pre-fix default-arg call returns False,
    so this fails until ``validate_schema`` opts into the raising contract.
    """
    backend = object.__new__(VespaBackend)
    backend._vespa_search_backend = None
    backend.schema_manager = VespaSchemaManager(
        backend_endpoint="http://127.0.0.1", backend_port=_dead_port()
    )

    with pytest.raises(requests.exceptions.ConnectionError):
        backend.validate_schema("agent_memories")


def test_schema_exists_tenant_branch_raises_on_lookup_failure():
    """The tenant branch must surface a lookup/enumeration failure, not flatten
    it to a 'schema missing' False that a caller (e.g. the deploy route) acts on.

    A real ``VespaSchemaManager`` with no registry raises ``ValueError`` from
    ``tenant_schema_exists``; the branch must propagate it, not mask it.
    """
    backend = object.__new__(VespaBackend)
    backend._tenant_id = "acme:acme"
    backend.schema_manager = VespaSchemaManager(
        backend_endpoint="http://127.0.0.1",
        backend_port=19071,
        schema_registry=None,
    )

    with pytest.raises(ValueError, match="schema_registry required"):
        backend.schema_exists("video_colpali_smol500_mv_frame", tenant_id="acme:acme")


def test_health_check_coerces_status_dict_to_bool():
    backend = _bare_backend()
    search = MagicMock()
    backend._vespa_search_backend = search

    search.health_check.return_value = {"status": "healthy", "components": {}}
    assert backend.health_check() is True

    search.health_check.return_value = {"status": "degraded", "components": {}}
    assert backend.health_check() is False


def test_get_schema_info_reports_initialization_flags():
    backend = _bare_backend()
    backend.config = {"schema_name": "video_colpali_smol500_mv_frame"}
    backend._initialized_as_search = True
    backend._initialized_as_ingestion = True
    backend._vespa_search_backend = MagicMock()

    info = backend.get_schema_info()

    assert info == {
        "name": "video_colpali_smol500_mv_frame",
        "backend": "vespa",
        "initialized": True,
        "search_enabled": True,
        "ingestion_enabled": True,
    }


def test_get_schema_info_requires_initialization():
    backend = object.__new__(VespaBackend)
    backend.schema_manager = None
    with pytest.raises(RuntimeError, match="not initialized"):
        backend.get_schema_info()


class TestConfigStoreHealthCheck:
    def _store(self, app):
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = object.__new__(VespaConfigStore)
        store.vespa_app = app
        store.schema_name = "config_metadata"
        return store

    def test_healthy_when_query_succeeds(self):
        app = MagicMock()
        app.query.return_value = MagicMock(hits=[])
        assert self._store(app).health_check() is True
        assert "config_metadata" in app.query.call_args.kwargs["yql"]

    def test_unhealthy_when_query_raises(self):
        app = MagicMock()
        app.query.side_effect = ConnectionError("vespa down")
        assert self._store(app).health_check() is False


class TestEmbeddingRequirements:
    def _search_backend(self):
        from cogniverse_vespa.search_backend import VespaSearchBackend

        return object.__new__(VespaSearchBackend)

    def test_requirements_derived_from_schema_strategies(self):
        import cogniverse_vespa.search_backend as sb

        backend = self._search_backend()
        original = sb._RANKING_STRATEGIES_CACHE
        sb._RANKING_STRATEGIES_CACHE = {
            "video_probe_schema": {
                "float_float": {
                    "needs_float_embeddings": True,
                    "embedding_field": "embedding",
                },
                "binary_binary": {
                    "needs_binary_embeddings": True,
                    "embedding_field": "embedding_binary",
                },
            }
        }
        try:
            reqs = backend.get_embedding_requirements("video_probe_schema")
        finally:
            sb._RANKING_STRATEGIES_CACHE = original

        assert reqs == {
            "needs_float": True,
            "needs_binary": True,
            "float_field": "embedding",
            "binary_field": "embedding_binary",
        }

    def test_unknown_schema_raises_with_available_list(self):
        import cogniverse_vespa.search_backend as sb

        backend = self._search_backend()
        original = sb._RANKING_STRATEGIES_CACHE
        sb._RANKING_STRATEGIES_CACHE = {"known_schema": {"s": {}}}
        try:
            with pytest.raises(ValueError, match="known_schema"):
                backend.get_embedding_requirements("missing_schema")
        finally:
            sb._RANKING_STRATEGIES_CACHE = original


def test_feed_wraps_single_document_ingest():
    """feed() delegates one document to ingest_documents and maps the result
    to (success_count, failed_ids) — incl. both failed-document shapes."""
    from cogniverse_sdk.document import Document

    backend = object.__new__(VespaBackend)
    doc = Document(id="d1", text_content="x", metadata={})

    backend.ingest_documents = lambda docs, schema: {
        "success_count": 1,
        "failed_documents": [],
    }
    assert backend.feed(doc, "some_schema") == (1, [])

    backend.ingest_documents = lambda docs, schema: {
        "success_count": 0,
        "failed_documents": ["d1"],
    }
    assert backend.feed(doc, "some_schema") == (0, ["d1"])

    backend.ingest_documents = lambda docs, schema: {
        "success_count": 0,
        "failed_documents": [{"id": "d1", "reason": "400"}],
    }
    assert backend.feed(doc, "some_schema") == (0, ["d1"])


def test_factory_builds_configured_search_backend():
    from cogniverse_vespa.search_backend import (
        VespaSearchBackend,
        create_vespa_search_backend,
    )

    backend = create_vespa_search_backend(
        "video_colpali_smol500_mv_frame",
        backend_url="http://localhost:9",
        enable_connection_pool=False,
    )
    assert isinstance(backend, VespaSearchBackend)
    assert backend.schema_name == "video_colpali_smol500_mv_frame"


def test_metadata_app_lazy_init_is_thread_safe():
    """Concurrent first-touches of _metadata_vespa_app must build exactly ONE
    PersistentVespaOps — an unlocked lazy init let two threads each construct
    one and leak the loser's session pool."""
    import threading
    from unittest.mock import patch

    from cogniverse_vespa.backend import VespaBackend

    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = 8080
    backend._metadata_app = None
    backend._metadata_app_key = None
    backend._metadata_app_lock = threading.Lock()

    built = []

    def fake_make(**kwargs):
        m = MagicMock()
        built.append(m)
        return m

    barrier = threading.Barrier(12)
    apps = []
    alock = threading.Lock()

    def worker():
        barrier.wait()
        app = backend._metadata_vespa_app()
        with alock:
            apps.append(app)

    with patch(
        "cogniverse_vespa.backend.make_persistent_vespa_ops", side_effect=fake_make
    ):
        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(built) == 1, f"{len(built)} PersistentVespaOps built — lazy init raced"
    assert len({id(a) for a in apps}) == 1, "threads got different metadata apps"


def test_document_field_numpy_scalars_are_coerced():
    """np.int64 in a field must not TypeError at the JSON step — the primitive
    coerces numpy scalars to native Python before feeding."""
    import json

    import numpy as np

    coerced = VespaBackend._coerce_field_values(
        {"count": np.int64(7), "score": np.float64(0.5), "name": "x", "ids": ["a"]}
    )
    assert coerced == {"count": 7, "score": 0.5, "name": "x", "ids": ["a"]}
    assert type(coerced["count"]) is int
    json.dumps(coerced)  # must not raise


@pytest.mark.unit
@pytest.mark.ci_fast
def test_document_field_nested_numpy_values_are_coerced():
    """The shallow pass coerced only TOP-LEVEL numpy scalars — an ndarray
    value or numpy scalars nested in a list/dict reached pyvespa's json.dumps
    un-serializable. The coercion must recurse into containers."""
    import json

    import numpy as np

    from cogniverse_vespa.backend import VespaBackend

    fields = {
        "vec": np.asarray([0.25, 0.5], dtype=np.float32),
        "scores": [np.float32(0.1), np.int64(7)],
        "meta": {"count": np.int64(3), "flags": [np.bool_(True)]},
        "plain": "text",
        "n": np.int64(42),
    }

    out = VespaBackend._coerce_field_values(fields)

    # Exact native shapes — and the whole dict must be JSON-serializable.
    assert out["vec"] == [0.25, 0.5]
    assert out["scores"] == [pytest.approx(0.1), 7]
    assert out["meta"] == {"count": 3, "flags": [True]}
    assert out["plain"] == "text"
    assert out["n"] == 42
    json.dumps(out)


@pytest.mark.unit
@pytest.mark.ci_fast
def test_check_document_response_raises_on_non_2xx():
    """The document primitives' defensive net: a returned non-2xx (pyvespa's
    rare returns-without-raising shape) maps to RuntimeError naming the op and
    document id; 2xx passes through silently."""
    from types import SimpleNamespace

    from cogniverse_vespa.backend import VespaBackend

    with pytest.raises(RuntimeError) as exc:
        VespaBackend._check_document_response(
            SimpleNamespace(status_code=507, json={"message": "disk full"}),
            "put",
            "doc-1",
        )
    msg = str(exc.value)
    assert "put" in msg
    assert "doc-1" in msg
    assert "507" in msg

    VespaBackend._check_document_response(
        SimpleNamespace(status_code=200, json={}), "put", "doc-1"
    )


def test_get_metadata_document_returns_none_on_genuine_404():
    """pyvespa returns a 404 as a non-raising response; a missing doc is None,
    not an error."""
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    client = MagicMock()
    client.get_data = MagicMock(return_value=MagicMock(status_code=404))
    backend._metadata_vespa_app = MagicMock(return_value=client)

    assert (
        backend.get_metadata_document(schema="tenant_metadata", doc_id="missing")
        is None
    )


def test_get_metadata_document_raises_on_backend_failure():
    """A backend failure (connection error / 5xx that pyvespa raises) must NOT
    flatten to None — that made an outage indistinguishable from 'not found',
    so assert_tenant_exists 404'd every request during a Vespa blip."""
    from vespa.exceptions import VespaError

    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    client = MagicMock()
    client.get_data = MagicMock(side_effect=VespaError("connection refused"))
    backend._metadata_vespa_app = MagicMock(return_value=client)

    with pytest.raises(VespaError):
        backend.get_metadata_document(schema="tenant_metadata", doc_id="acme:acme")


class TestWriteFaultContracts:
    """Backend write failures must raise — a False return reads as "the
    write was rejected" and callers silently drop or mis-report the write."""

    def test_update_document_raises_on_backend_failure(self):
        from cogniverse_sdk.document import Document

        backend = _bare_backend()
        backend.config = {"schema_name": "agent_memories"}
        backend.ingest_documents = MagicMock(
            side_effect=ConnectionError("backend down")
        )

        doc = Document(id="m1", text_content="x", metadata={})
        with pytest.raises(ConnectionError):
            backend.update_document("m1", doc)

    def test_update_document_rejects_id_mismatch_loudly(self):
        from cogniverse_sdk.document import Document

        backend = _bare_backend()
        backend.config = {"schema_name": "agent_memories"}
        backend.ingest_documents = MagicMock()

        doc = Document(id="OTHER", text_content="x", metadata={})
        with pytest.raises(ValueError, match="does not match"):
            backend.update_document("m1", doc)
        backend.ingest_documents.assert_not_called()

    def test_update_document_requires_a_schema_name(self):
        from cogniverse_sdk.document import Document

        backend = _bare_backend()
        backend.config = {}

        doc = Document(id="m1", text_content="x", metadata={})
        with pytest.raises(ValueError, match="schema_name"):
            backend.update_document("m1", doc)

    def test_create_metadata_document_raises_on_outage(self):
        backend = _bare_backend()
        backend._url = "http://localhost"
        app = MagicMock()
        app.feed_data_point.side_effect = ConnectionError("backend down")
        backend._metadata_vespa_app = MagicMock(return_value=app)

        with pytest.raises(ConnectionError):
            backend.create_metadata_document("tenant_metadata", "t1", {"a": 1})

    def test_delete_metadata_document_raises_on_outage(self):
        backend = _bare_backend()
        backend._url = "http://localhost"
        app = MagicMock()
        app.delete_data.side_effect = ConnectionError("backend down")
        backend._metadata_vespa_app = MagicMock(return_value=app)

        with pytest.raises(ConnectionError):
            backend.delete_metadata_document("tenant_metadata", "t1")

    def test_create_metadata_document_false_on_rejected_status(self):
        """A non-200 the client surfaces WITHOUT raising stays a clean False
        (a rejected write, distinct from an outage)."""
        backend = _bare_backend()
        backend._url = "http://localhost"
        response = MagicMock()
        response.status_code = 400
        app = MagicMock()
        app.feed_data_point.return_value = response
        backend._metadata_vespa_app = MagicMock(return_value=app)

        assert backend.create_metadata_document("tenant_metadata", "t1", {}) is False

    def test_document_delete_is_idempotent_only_for_a_genuine_404(self):
        from types import SimpleNamespace

        from vespa.exceptions import VespaError

        from cogniverse_vespa.ingestion_client import VespaPyClient

        client = object.__new__(VespaPyClient)
        client._connected = True
        client.schema_name = "wiki_pages_acme_acme"
        client.namespace = "content"
        client.logger = MagicMock()
        client.app = MagicMock()

        client.app.delete_data.return_value = SimpleNamespace(status_code=404)
        assert client.delete_document("missing") is True

        client.app.delete_data.side_effect = VespaError("HTTP 404: not found")
        assert client.delete_document("missing") is True

    def test_document_delete_transport_failure_raises_with_route_context(self):
        from cogniverse_vespa.ingestion_client import VespaPyClient

        client = object.__new__(VespaPyClient)
        client._connected = True
        client.schema_name = "wiki_pages_acme_acme"
        client.namespace = "content"
        client.logger = MagicMock()
        client.app = MagicMock()
        client.app.delete_data.side_effect = ConnectionError("vespa unreachable")

        with pytest.raises(
            RuntimeError,
            match=r"content/wiki_pages_acme_acme/doc-7.*vespa unreachable",
        ):
            client.delete_document("doc-7")

    def test_backend_document_delete_propagates_client_failure(self):
        backend = _bare_backend()
        backend.config = {"schema_name": "wiki_pages"}
        client = MagicMock()
        client.delete_document.side_effect = RuntimeError(
            "content/wiki_pages_acme_acme/doc-7: vespa unreachable"
        )
        backend._get_or_create_ingestion_client = MagicMock(return_value=client)

        with pytest.raises(RuntimeError, match="vespa unreachable"):
            backend.delete_document("doc-7", schema_name="wiki_pages")


def _lazy_backend() -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._tenant_id = ""
    backend._url = "http://vespa"
    backend._port = 8080
    backend._schema_loader_instance = MagicMock()
    backend._config_manager_instance = None
    backend._vespa_ingestion_clients = {}
    backend._vespa_search_backend = None
    backend._ingestion_clients_lock = threading.Lock()
    backend._search_backend_lock = threading.Lock()
    backend._initialized_as_search = False
    backend.use_async_ingestion = False
    backend.config = {
        "profiles": {"wiki": {"type": "wiki", "schema_name": "wiki_pages"}},
        "default_profiles": {"wiki": {"profile": "wiki"}},
    }
    return backend


def test_concurrent_ingestion_first_touch_builds_one_client(monkeypatch):
    from cogniverse_vespa import backend as backend_module

    backend = _lazy_backend()
    built = []

    def build_client(*, config, logger):
        time.sleep(0.02)
        client = MagicMock()
        client.config = config
        client.connect.return_value = True
        built.append(client)
        return client

    monkeypatch.setattr(backend_module, "VespaPyClient", build_client)
    start = threading.Barrier(12)

    def get_client():
        start.wait()
        return backend._get_or_create_ingestion_client("wiki_pages")

    with ThreadPoolExecutor(max_workers=12) as executor:
        clients = list(executor.map(lambda _: get_client(), range(12)))

    assert len(built) == 1
    assert {id(client) for client in clients} == {id(built[0])}
    assert backend._vespa_ingestion_clients == {"wiki_pages": built[0]}


def test_failed_ingestion_first_touch_is_not_cached(monkeypatch):
    from cogniverse_vespa import backend as backend_module

    backend = _lazy_backend()
    failed = MagicMock()
    failed.connect.return_value = False
    healthy = MagicMock()
    healthy.connect.return_value = True
    monkeypatch.setattr(
        backend_module,
        "VespaPyClient",
        MagicMock(side_effect=[failed, healthy]),
    )

    with pytest.raises(ConnectionError, match="wiki_pages"):
        backend._get_or_create_ingestion_client("wiki_pages")
    assert backend._vespa_ingestion_clients == {}

    assert backend._get_or_create_ingestion_client("wiki_pages") is healthy
    assert backend._vespa_ingestion_clients == {"wiki_pages": healthy}


def test_concurrent_search_first_touch_builds_one_backend(monkeypatch):
    from cogniverse_vespa import backend as backend_module

    backend = _lazy_backend()
    search_backend = MagicMock()
    search_backend.search.side_effect = lambda query: query["request_id"]
    built = []

    def build_search_backend(**kwargs):
        time.sleep(0.02)
        built.append(kwargs)
        return search_backend

    monkeypatch.setattr(
        backend_module,
        "VespaSearchBackend",
        build_search_backend,
    )
    start = threading.Barrier(12)

    def search(request_id):
        start.wait()
        return backend.search(
            {
                "request_id": request_id,
                "query": "tenant content",
                "type": "wiki",
                "tenant_id": "acme:acme",
            }
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(search, range(12)))

    assert results == list(range(12))
    assert len(built) == 1
    assert backend._vespa_search_backend is search_backend
    assert search_backend.search.call_count == 12


def test_failed_search_first_touch_is_retried_cleanly(monkeypatch):
    from cogniverse_vespa import backend as backend_module

    backend = _lazy_backend()
    healthy = MagicMock()
    healthy.search.return_value = ["doc-1"]
    factory = MagicMock(
        side_effect=[RuntimeError("search construction failed"), healthy]
    )
    monkeypatch.setattr(backend_module, "VespaSearchBackend", factory)
    query = {
        "query": "tenant content",
        "type": "wiki",
        "tenant_id": "acme:acme",
    }

    with pytest.raises(RuntimeError, match="construction failed"):
        backend.search(query)
    assert backend._vespa_search_backend is None
    assert backend._initialized_as_search is False

    assert backend.search(query) == ["doc-1"]
    assert backend._vespa_search_backend is healthy
    assert factory.call_count == 2
