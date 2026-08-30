"""E2E tests for wiki knowledge base on k3d."""

import json
import re
import uuid

import httpx
import pytest

from cogniverse_agents.wiki.wiki_schema import generate_slug
from tests.e2e.conftest import RUNTIME, TENANT_ID

SAFE_TENANT = TENANT_ID.replace(":", "_")


@pytest.mark.e2e
class TestWikiEndpoints:
    def test_wiki_save(self):
        """POST /wiki/save persists a session page and one topic page per
        entity; both are read back exactly by id."""
        marker = uuid.uuid4().hex[:12]
        query = f"e2e test wiki save {marker}"
        answer = f"This is an e2e test of wiki ({marker})"
        entity = f"wiki feature {marker}"
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/wiki/save",
                json={
                    "query": query,
                    "response": {"answer": answer},
                    "entities": [entity],
                    "agent_name": "gateway_agent",
                    "tenant_id": TENANT_ID,
                },
            )
            assert resp.status_code == 200, f"Wiki save failed: {resp.text[:300]}"
            data = resp.json()
            title = f"Session — {query[:60]}"
            assert data == {
                "status": "saved",
                "doc_id": data["doc_id"],
                "title": title,
                "slug": generate_slug(title),
            }, data
            assert re.fullmatch(
                rf"wiki_session_{SAFE_TENANT}_\d{{20}}", data["doc_id"]
            ), data["doc_id"]

            topic_slug = generate_slug(entity)
            topic = client.get(
                f"/wiki/topic/{topic_slug}", params={"tenant_id": TENANT_ID}
            )
            assert topic.status_code == 200, topic.text[:300]
            assert topic.json() == {
                "doc_id": f"wiki_topic_{SAFE_TENANT}_{topic_slug}",
                "title": entity,
                "content": answer,
                "page_type": "topic",
                "entities": json.dumps([entity]),
                "sources": "[]",
                "update_count": 1,
            }, topic.json()

    def test_wiki_search(self):
        """POST /wiki/search returns the WikiManager.search row shape."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/wiki/search",
                json={"query": "e2e test", "tenant_id": TENANT_ID, "top_k": 5},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {"results", "count"}, data
        assert data["count"] == len(data["results"]), data
        assert data["count"] <= 5, data
        for row in data["results"]:
            assert set(row) == {"doc_id", "title", "content", "page_type", "score"}, row
            assert row["page_type"] in {"topic", "session", "index"}, row
            assert row["doc_id"].startswith(f"wiki_{row['page_type']}_"), row
        scores = [row["score"] for row in data["results"]]
        assert scores == sorted(scores, reverse=True), scores

    def test_wiki_index(self):
        """GET /wiki/index returns the rendered WikiIndex markdown."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/wiki/index", params={"tenant_id": TENANT_ID})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {"content"}, data
        header = re.match(
            rf"# Wiki Index — {re.escape(TENANT_ID)}\n\n"
            r"_(\d+) pages: (\d+) topics, (\d+) sessions\._\n\n## Topics\n",
            data["content"],
        )
        assert header, data["content"][:200]
        pages, topics, sessions = (int(group) for group in header.groups())
        assert pages == topics + sessions, header.groups()
        assert "\n\n## Sessions\n" in data["content"], data["content"][-200:]
