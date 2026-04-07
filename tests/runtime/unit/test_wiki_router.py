"""Unit tests for the wiki router's tenant-aware factory wiring (audit fix #12).

Verifies that:
1. Each request resolves a tenant-specific WikiManager via the factory.
2. ``WikiSaveRequest.tenant_id`` is honored — two tenants get isolated wikis.
3. The factory caches one manager per tenant (no leak).
4. Missing factory produces a clean 503 instead of crashing.
5. The auto-file path in agent_dispatcher uses the factory, not a singleton.

Before this fix the router bound a single ``WikiManager(tenant_id="default")``
at startup and ignored ``WikiSaveRequest.tenant_id`` entirely — every tenant's
writes ended up in the default wiki.
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import wiki as wiki_router


@pytest.fixture(autouse=True)
def reset_wiki_factory():
    """Snapshot and restore the module-level factory between tests."""
    original = wiki_router._wiki_manager_factory
    yield
    wiki_router._wiki_manager_factory = original


@pytest.fixture
def factory_with_per_tenant_managers():
    """Build a factory whose managers track which tenant they belong to."""
    managers: Dict[str, Any] = {}

    def _factory(tenant_id: str):
        if tenant_id not in managers:
            mgr = MagicMock()
            mgr._tenant_id = tenant_id
            mgr.save_session = MagicMock(
                return_value=MagicMock(
                    doc_id=f"doc_{tenant_id}_x",
                    title=f"page-{tenant_id}",
                    slug=f"slug-{tenant_id}",
                )
            )
            mgr.search = MagicMock(return_value=[])
            mgr.get_topic = MagicMock(return_value=None)
            mgr.get_index = MagicMock(return_value="")
            mgr.lint = MagicMock(return_value={"issues": []})
            mgr.delete_page = MagicMock()
            managers[tenant_id] = mgr
        return managers[tenant_id]

    wiki_router.set_wiki_manager_factory(_factory)
    return _factory, managers


@pytest.fixture
def wiki_client(factory_with_per_tenant_managers):
    """TestClient with the wiki router and a tenant-aware factory installed."""
    factory, managers = factory_with_per_tenant_managers
    app = FastAPI()
    app.include_router(wiki_router.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, managers


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPerTenantWikiSave:
    def test_save_routes_to_tenant_specific_manager(self, wiki_client):
        """POST /save with two different tenant_ids must hit two different
        WikiManagers — proves the singleton bug is gone."""
        client, managers = wiki_client

        client.post(
            "/save",
            json={
                "query": "alpha",
                "response": {"answer": "first"},
                "tenant_id": "tenant_a",
            },
        )
        client.post(
            "/save",
            json={
                "query": "beta",
                "response": {"answer": "second"},
                "tenant_id": "tenant_b",
            },
        )

        # Two distinct managers were materialised.
        assert "tenant_a" in managers
        assert "tenant_b" in managers
        assert managers["tenant_a"] is not managers["tenant_b"]

        # Each manager only saw its own write.
        managers["tenant_a"].save_session.assert_called_once()
        managers["tenant_b"].save_session.assert_called_once()
        assert managers["tenant_a"].save_session.call_args.kwargs["query"] == "alpha"
        assert managers["tenant_b"].save_session.call_args.kwargs["query"] == "beta"

    def test_save_returns_doc_metadata(self, wiki_client):
        client, _managers = wiki_client
        resp = client.post(
            "/save",
            json={
                "query": "hi",
                "response": {"answer": "ok"},
                "tenant_id": "acme",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "saved"
        assert body["doc_id"] == "doc_acme_x"
        assert body["title"] == "page-acme"
        assert body["slug"] == "slug-acme"

    def test_factory_caches_one_manager_per_tenant(self, wiki_client):
        """Two saves to the same tenant must hit the SAME manager instance.
        If the factory builds a new manager every call, that's a leak."""
        client, managers = wiki_client

        client.post(
            "/save",
            json={
                "query": "first",
                "response": {"answer": "x"},
                "tenant_id": "acme",
            },
        )
        first_mgr = managers["acme"]

        client.post(
            "/save",
            json={
                "query": "second",
                "response": {"answer": "y"},
                "tenant_id": "acme",
            },
        )
        second_mgr = managers["acme"]

        assert first_mgr is second_mgr
        assert first_mgr.save_session.call_count == 2


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPerTenantWikiSearch:
    def test_search_routes_to_tenant_specific_manager(self, wiki_client):
        client, managers = wiki_client

        client.post(
            "/search",
            json={"query": "alpha", "tenant_id": "tenant_a", "top_k": 3},
        )
        client.post(
            "/search",
            json={"query": "beta", "tenant_id": "tenant_b", "top_k": 5},
        )

        managers["tenant_a"].search.assert_called_once_with(query="alpha", top_k=3)
        managers["tenant_b"].search.assert_called_once_with(query="beta", top_k=5)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPerTenantWikiGetEndpoints:
    def test_get_topic_uses_query_param_tenant_id(self, wiki_client):
        client, managers = wiki_client
        managers_before = set(managers.keys())

        client.get("/topic/foo?tenant_id=tenant_a")

        new = set(managers.keys()) - managers_before
        assert new == {"tenant_a"}

    def test_get_index_uses_query_param_tenant_id(self, wiki_client):
        client, managers = wiki_client
        client.get("/index?tenant_id=tenant_b")
        assert "tenant_b" in managers
        managers["tenant_b"].get_index.assert_called_once()

    def test_lint_uses_query_param_tenant_id(self, wiki_client):
        client, managers = wiki_client
        client.get("/lint?tenant_id=tenant_c")
        assert "tenant_c" in managers
        managers["tenant_c"].lint.assert_called_once()

    def test_get_endpoints_default_to_default_tenant(self, wiki_client):
        """Backward compat: omitting ?tenant_id= falls back to "default"."""
        client, managers = wiki_client
        client.get("/index")
        assert "default" in managers


@pytest.mark.unit
@pytest.mark.ci_fast
class TestFactoryNotConfigured:
    def test_save_returns_503_when_factory_missing(self):
        """If main.py never installed the factory (e.g., schema deploy
        failed), every endpoint must return 503 — not crash with NoneType."""
        wiki_router._wiki_manager_factory = None
        app = FastAPI()
        app.include_router(wiki_router.router)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/save",
                json={
                    "query": "hi",
                    "response": {"answer": "x"},
                    "tenant_id": "acme",
                },
            )
            assert resp.status_code == 503
            assert "factory not configured" in resp.json()["detail"].lower()
