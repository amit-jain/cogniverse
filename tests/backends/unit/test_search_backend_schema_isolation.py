"""VespaSearchBackend.batch_get_documents must read the schema it is told to,
not the shared per-request ``self.schema_name``.

The search backend is a single process-global instance shared across every
tenant, and each search request rewrites ``self.schema_name``. A document fetch
that read that shared attribute would, after tenant A's search, read tenant A's
schema for a tenant B fetch that interleaved between the write and the read.
Passing the schema explicitly closes that cross-tenant window.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_vespa import search_backend as search_module
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.search_backend import ConnectionPool, VespaSearchBackend

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingHandle:
    """Stands in for a pyvespa sync session and records point-read routing."""

    def __init__(self):
        self.routes_seen = []

    def get_data(self, schema, data_id, namespace, raise_on_not_found):
        self.routes_seen.append((schema, namespace))
        return SimpleNamespace(status_code=404, json={})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _backend_with_shared_schema(shared_schema: str, handle: _RecordingHandle):
    backend = VespaSearchBackend.__new__(VespaSearchBackend)
    backend.pool = None
    backend.schema_name = shared_schema
    backend.vespa = SimpleNamespace(syncio=lambda: handle)
    return backend


def test_batch_get_uses_explicit_schema_not_shared_attr():
    handle = _RecordingHandle()
    # The shared attribute holds tenant A's schema (A searched last)...
    backend = _backend_with_shared_schema("video_tenant_a", handle)

    # ...but tenant B fetches with its own schema explicitly.
    backend.batch_get_documents(
        ["doc-1"], schema_name="video_tenant_b", namespace="content"
    )

    assert handle.routes_seen == [("video_tenant_b", "content")]
    assert ("video_tenant_a", "content") not in handle.routes_seen


def test_get_document_threads_schema_and_namespace_through():
    handle = _RecordingHandle()
    backend = _backend_with_shared_schema("video_tenant_a", handle)

    backend.get_document(
        "doc-9",
        schema_name="agent_memories_acme_acme",
        namespace="memory_content",
    )

    assert handle.routes_seen == [("agent_memories_acme_acme", "memory_content")]


def test_point_read_without_explicit_route_is_rejected():
    handle = _RecordingHandle()
    backend = _backend_with_shared_schema("video_default", handle)

    with pytest.raises(TypeError):
        backend.batch_get_documents(["doc-1"])

    assert handle.routes_seen == []


def test_closed_pool_rejects_new_acquisitions_without_creating_clients():
    pool = ConnectionPool.__new__(ConnectionPool)
    pool.url = "http://vespa:8080"
    pool.config = SimpleNamespace(max_connections=1, connection_timeout=0.1)
    pool._connections = []
    pool._available = []
    pool._removing = set()
    pool._lock = threading.Lock()
    pool._returned = threading.Condition(pool._lock)
    pool._closed = True

    with (
        patch.object(search_module, "VespaConnection") as connection_factory,
        pytest.raises(RuntimeError, match="closed"),
    ):
        with pool.get_connection():
            pytest.fail("a closed pool yielded a connection")

    connection_factory.assert_not_called()


def test_tenant_profiles_and_defaults_override_same_named_global_entries(
    monkeypatch,
):
    backend = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": 8080,
            "profiles": {
                "shared": {
                    "type": "wiki",
                    "schema_name": "global_wiki",
                }
            },
            "default_profiles": {
                "wiki": {"profile": "shared", "strategy": "global_rank"}
            },
        },
        config_manager=MagicMock(),
        enable_connection_pool=False,
    )
    backend.vespa = MagicMock()
    backend.vespa.query.return_value = SimpleNamespace(
        status_code=200,
        hits=[],
        get_json=lambda: {
            "root": {
                "coverage": {"coverage": 100, "documents": 0},
                "children": [],
            }
        },
    )
    monkeypatch.setattr(
        backend,
        "_load_tenant_profiles",
        lambda _tenant: (
            {
                "shared": {
                    "type": "wiki",
                    "schema_name": "tenant_wiki",
                }
            },
            {"wiki": {"profile": "shared", "strategy": "tenant_rank"}},
        ),
    )
    monkeypatch.setattr(
        search_module,
        "_RANKING_STRATEGIES_CACHE",
        {
            "global_wiki": {
                "global_rank": {"needs_text_query": True},
                "other": {"needs_text_query": True},
            },
            "tenant_wiki": {
                "tenant_rank": {"needs_text_query": True},
                "other": {"needs_text_query": True},
            },
        },
    )

    assert (
        backend.search(
            {
                "query": "tenant-local article",
                "type": "wiki",
                "tenant_id": "acme",
            }
        )
        == []
    )

    body = backend.vespa.query.call_args.kwargs["body"]
    assert body["yql"] == (
        "select * from tenant_wiki_acme_acme where userInput(@userQuery)"
    )
    assert body["model.restrict"] == "tenant_wiki_acme_acme"
    assert body["ranking"] == "tenant_rank"


def test_metadata_query_scopes_direct_yql_and_forwards_query_options():
    backend = object.__new__(VespaBackend)
    backend._url = "http://vespa"
    backend._port = 8080
    backend._metadata_app = None
    backend._metadata_app_key = None
    backend.get_tenant_schema_name = MagicMock(return_value="wiki_pages_acme_acme")
    client = MagicMock()
    client.query.return_value = SimpleNamespace(
        status_code=200,
        json={"root": {"children": [{"fields": {"id": "best"}}]}},
    )

    with patch.object(backend, "_metadata_vespa_app", return_value=client):
        rows = backend.query_metadata_documents(
            schema="wiki_pages",
            yql="select * from sources wiki_pages where true limit 2",
            tenant_id="acme:acme",
            hits=2,
            ranking="random",
        )

    assert rows == [{"id": "best"}]
    assert client.query.call_args.kwargs == {
        "hits": 2,
        "ranking": "random",
        "yql": ("select * from sources wiki_pages_acme_acme where true limit 2"),
    }


def test_metadata_query_rejects_yql_that_cannot_be_tenant_scoped():
    backend = object.__new__(VespaBackend)
    backend._url = "http://vespa"
    backend._port = 8080
    backend.get_tenant_schema_name = MagicMock(return_value="wiki_pages_acme_acme")
    backend._metadata_vespa_app = MagicMock()

    with pytest.raises(ValueError, match="wiki_pages"):
        backend.query_metadata_documents(
            schema="wiki_pages",
            yql="select * from sources another_schema where true",
            tenant_id="acme:acme",
        )

    backend._metadata_vespa_app.assert_not_called()
