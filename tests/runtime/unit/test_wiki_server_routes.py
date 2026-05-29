"""Server-side coverage for /wiki/topic GET, /wiki/lint, /wiki/topic DELETE.

The existing tests in tests/messaging/ exercise the runtime client (mocks
the HTTP boundary client-side and asserts the URL it would post to). The
server route handlers were untested — the doc_id construction, the
``RuntimeError → 500`` branch, the 200/404 split for the topic GET.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import wiki as wiki_router


@pytest.fixture
def fake_wm():
    wm = MagicMock()
    wm._tenant_id = "acme"
    return wm


@pytest.fixture
def client(fake_wm) -> TestClient:
    app = FastAPI()
    app.include_router(wiki_router.router, prefix="/wiki")

    # Install factory that returns our preconfigured mock manager.
    wiki_router.set_wiki_manager_factory(lambda tenant_id: fake_wm)
    yield TestClient(app)
    # Reset factory between tests so a previous one cannot leak.
    wiki_router._wiki_manager_factory = None


def test_get_topic_returns_200_when_found(client: TestClient, fake_wm) -> None:
    fake_wm.get_topic.return_value = {
        "title": "Acme Architecture",
        "content": "# Architecture\n",
    }
    r = client.get("/wiki/topic/acme-architecture?tenant_id=acme")
    assert r.status_code == 200
    assert r.json() == {
        "title": "Acme Architecture",
        "content": "# Architecture\n",
    }
    fake_wm.get_topic.assert_called_once_with("acme-architecture")


def test_get_topic_returns_404_when_missing(client: TestClient, fake_wm) -> None:
    fake_wm.get_topic.return_value = None
    r = client.get("/wiki/topic/missing?tenant_id=acme")
    assert r.status_code == 404
    assert "missing" in r.json()["detail"]


def test_lint_returns_manager_dict(client: TestClient, fake_wm) -> None:
    fake_wm.lint.return_value = {"errors": 2, "warnings": 7, "issues": []}
    r = client.get("/wiki/lint?tenant_id=acme")
    assert r.status_code == 200
    assert r.json() == {"errors": 2, "warnings": 7, "issues": []}


def test_lint_requires_tenant_id(client: TestClient) -> None:
    r = client.get("/wiki/lint")
    assert r.status_code == 422


def test_delete_topic_constructs_correct_doc_id(client: TestClient, fake_wm) -> None:
    r = client.delete("/wiki/topic/sample-page?tenant_id=acme")
    assert r.status_code == 200
    body = r.json()
    # The route MUST build doc_id as f"wiki_topic_{safe}_{slug}" so the
    # tombstone matches the schema's id format.
    assert body["doc_id"] == "wiki_topic_acme_sample-page"
    assert body["slug"] == "sample-page"
    fake_wm.delete_page.assert_called_once_with("wiki_topic_acme_sample-page")


def test_delete_topic_colon_in_tenant_id_is_sanitised(
    client: TestClient, fake_wm
) -> None:
    """Tenant_ids carry a colon (org:tenant); the doc id replaces ``:`` with ``_``."""
    fake_wm._tenant_id = "acme:east"
    r = client.delete("/wiki/topic/foo?tenant_id=acme:east")
    assert r.status_code == 200
    assert r.json()["doc_id"] == "wiki_topic_acme_east_foo"


def test_delete_topic_runtime_error_becomes_500(client: TestClient, fake_wm) -> None:
    fake_wm.delete_page.side_effect = RuntimeError("vespa down")
    r = client.delete("/wiki/topic/foo?tenant_id=acme")
    assert r.status_code == 500
    assert "vespa down" in r.json()["detail"]


def test_factory_not_configured_returns_503(monkeypatch) -> None:
    """No factory installed → 503 (typed), not 500."""
    app = FastAPI()
    app.include_router(wiki_router.router, prefix="/wiki")
    monkeypatch.setattr(wiki_router, "_wiki_manager_factory", None)
    with TestClient(app) as c:
        r = c.get("/wiki/lint?tenant_id=acme")
    assert r.status_code == 503
    assert "WikiManager factory" in r.json()["detail"]
