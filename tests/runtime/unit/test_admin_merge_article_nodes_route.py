"""Route contract for ``POST /admin/graph/merge-article-nodes``.

The route canonicalizes the tenant scope, defaults to a dry run, resolves the
cluster-wide Vespa backend the migration leases per write, and maps a tenant
with no graph to 404. The migration itself runs against real Vespa in
tests/agents/integration/test_article_node_migration_vespa.py; here it is a
recording double so the route's own decisions are what is asserted.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.graph import article_node_migration
from cogniverse_agents.graph.article_node_migration import (
    ArticleMerge,
    TenantArticleMergeReport,
    UnknownGraphTenantError,
)
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_CONFIG_MANAGER = object()
_SCHEMA_LOADER = object()
_BACKEND = object()


class _Registry:
    def __init__(self) -> None:
        self.requests = []

    def get_ingestion_backend(self, name, **kwargs):
        self.requests.append((name, kwargs))
        return _BACKEND


@pytest.fixture
def route(monkeypatch):
    calls = []
    registry = _Registry()

    def fake_merge(backend_resolver, *, tenant_ids=None, apply=False, exclude=None):
        calls.append(
            {
                "backend": backend_resolver(),
                "tenant_ids": tenant_ids,
                "apply": apply,
                "exclude": exclude,
            }
        )
        if tenant_ids == ["ghost:ghost"]:
            raise UnknownGraphTenantError("no deployed knowledge_graph schema")
        return [
            TenantArticleMergeReport(
                tenant_id="acme:acme",
                applied=apply,
                merges=[
                    ArticleMerge(
                        source_id="the_sorbonne",
                        target_id="sorbonne",
                        node_doc=True,
                        edges_repointed=2,
                        edges_deduped=1,
                    )
                ],
                skipped=["the_louvre"],
                excluded=list(exclude or []),
                edges_repointed=2,
                edges_deduped=1,
                mentions_added=1,
                content_docs_updated=3,
            )
        ]

    monkeypatch.setattr(article_node_migration, "merge_article_nodes", fake_merge)
    monkeypatch.setattr(
        admin_router.BackendRegistry, "get_instance", staticmethod(lambda: registry)
    )
    admin_router.set_config_manager(_CONFIG_MANAGER)
    admin_router.set_schema_loader(_SCHEMA_LOADER)
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    yield TestClient(app), calls, registry
    admin_router.reset_dependencies()


_REPORT = {
    "tenant_id": "acme:acme",
    "merges": [
        {
            "from": "the_sorbonne",
            "into": "sorbonne",
            "node_doc": True,
            "edges_repointed": 2,
            "edges_deduped": 1,
        }
    ],
    "skipped": ["the_louvre"],
    "excluded": [],
    "edges_repointed": 2,
    "edges_deduped": 1,
    "mentions_added": 1,
    "content_docs_updated": 3,
}


def test_defaults_to_a_dry_run_over_every_tenant(route):
    client, calls, registry = route

    resp = client.post("/admin/graph/merge-article-nodes")

    assert resp.status_code == 200
    assert resp.json() == {
        "dry_run": True,
        "tenants": [{**_REPORT, "applied": False}],
    }
    assert calls == [
        {"backend": _BACKEND, "tenant_ids": None, "apply": False, "exclude": []}
    ]
    assert registry.requests == [
        (
            "vespa",
            {
                "tenant_id": SYSTEM_TENANT_ID,
                "config_manager": _CONFIG_MANAGER,
                "schema_loader": _SCHEMA_LOADER,
            },
        )
    ]


def test_apply_with_a_simple_form_tenant_runs_its_canonical_graph(route):
    client, calls, _ = route

    resp = client.post(
        "/admin/graph/merge-article-nodes",
        params={"dry_run": "false", "tenant_id": "acme"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"dry_run": False, "tenants": [{**_REPORT, "applied": True}]}
    assert calls == [
        {"backend": _BACKEND, "tenant_ids": ["acme:acme"], "apply": True, "exclude": []}
    ]


def test_tenant_without_a_graph_is_404(route):
    client, calls, _ = route

    resp = client.post(
        "/admin/graph/merge-article-nodes", params={"tenant_id": "ghost:ghost"}
    )

    assert resp.status_code == 404
    assert resp.json() == {"detail": "no deployed knowledge_graph schema"}
    assert [c["tenant_ids"] for c in calls] == [["ghost:ghost"]]


def test_malformed_tenant_is_400_before_any_migration(route):
    client, calls, _ = route

    resp = client.post(
        "/admin/graph/merge-article-nodes", params={"tenant_id": "a:b:c"}
    )

    assert resp.status_code == 400
    assert calls == []


def test_exclude_accepts_repeated_and_comma_separated_ids(route):
    client, calls, _ = route

    resp = client.post(
        "/admin/graph/merge-article-nodes",
        params=[
            ("exclude", "the_who"),
            ("exclude", "a_team, an_apple_pie"),
            ("exclude", "the_who"),
        ],
    )

    assert resp.status_code == 200
    assert calls == [
        {
            "backend": _BACKEND,
            "tenant_ids": None,
            "apply": False,
            "exclude": ["the_who", "a_team", "an_apple_pie"],
        }
    ]
    assert resp.json()["tenants"][0]["excluded"] == [
        "the_who",
        "a_team",
        "an_apple_pie",
    ]


@pytest.mark.parametrize("value", ["The Who", "who", "the_", "the__who"])
def test_exclude_that_is_not_an_article_node_id_is_400(route, value):
    client, calls, _ = route

    resp = client.post("/admin/graph/merge-article-nodes", params={"exclude": value})

    assert resp.status_code == 400
    assert "article node id" in resp.json()["detail"]
    assert calls == []
