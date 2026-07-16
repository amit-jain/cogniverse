"""VespaBackend's bool-contracted surfaces return honest booleans.

``validate_schema`` used to be a validator that never validated (its try body
was a comment plus an unconditional ``return True``), and ``health_check``
(typed ``-> bool`` by the SearchBackend ABC) leaked the search backend's
status DICT — always truthy, even when degraded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.backend import VespaBackend

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_backend():
    backend = object.__new__(VespaBackend)
    backend.schema_manager = MagicMock()
    backend._vespa_search_backend = None
    return backend


def test_validate_schema_checks_deployed_document_types():
    backend = _bare_backend()
    backend.schema_manager.list_deployed_document_types.return_value = [
        "agent_memories",
        "video_colpali_smol500_mv_frame",
    ]

    assert backend.validate_schema("agent_memories") is True
    assert backend.validate_schema("nonexistent_schema") is False


def test_validate_schema_returns_false_on_listing_failure():
    backend = _bare_backend()
    backend.schema_manager.list_deployed_document_types.side_effect = RuntimeError(
        "config server down"
    )

    assert backend.validate_schema("agent_memories") is False


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
