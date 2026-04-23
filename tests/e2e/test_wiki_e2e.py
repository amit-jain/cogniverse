"""E2E tests for wiki knowledge base on k3d."""

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime


@pytest.mark.e2e
@skip_if_no_runtime
class TestWikiEndpoints:
    def test_wiki_save(self):
        """POST /wiki/save creates wiki pages."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/wiki/save",
                json={
                    "query": "e2e test wiki save",
                    "response": {"message": "This is an e2e test of wiki"},
                    "entities": ["e2e_testing", "wiki_feature"],
                    "agent_name": "gateway_agent",
                    "tenant_id": TENANT_ID,
                },
            )
        assert resp.status_code == 200, f"Wiki save failed: {resp.text[:300]}"
        data = resp.json()
        assert data["status"] == "saved"
        assert "doc_id" in data

    def test_wiki_search(self):
        """POST /wiki/search returns results."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/wiki/search",
                json={"query": "e2e test", "tenant_id": TENANT_ID},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "count" in data

    def test_wiki_index(self):
        """GET /wiki/index returns content."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/wiki/index", params={"tenant_id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
