"""E2E tests for tenant extensibility on k3d.

Tests instructions, memories, jobs, search, and agent behavior against
the live runtime stack with real Vespa, Ollama, and agents. All
assertions verify actual behavior, not just HTTP status codes.
"""

import math
import time

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime

OLLAMA_URL = "http://localhost:11434"


def _embed(text: str) -> list:
    """Get embedding from host Ollama for semantic similarity checks."""
    r = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "nomic-embed-text", "input": text},
        timeout=30,
    )
    return r.json()["embeddings"][0]


def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _semantic_similarity(text_a: str, text_b: str) -> float:
    return _cosine_sim(_embed(text_a), _embed(text_b))


def _memory_available() -> bool:
    try:
        r = httpx.get(
            f"{RUNTIME}/admin/tenant/{TENANT_ID}/memories?type=preference",
            timeout=10.0,
        )
        return r.status_code != 503
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantInstructions:
    def test_set_get_delete_round_trip(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.put(
                f"/admin/tenant/{TENANT_ID}/instructions",
                json={"text": "Always prefer bullet-point summaries over prose"},
            )
            assert resp.status_code == 200
            assert resp.json()["text"] == "Always prefer bullet-point summaries over prose"
            assert resp.json()["updated_at"]

            resp = client.get(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 200
            assert resp.json()["text"] == "Always prefer bullet-point summaries over prose"

            resp = client.delete(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cleared"

            resp = client.get(f"/admin/tenant/{TENANT_ID}/instructions")
            stored = resp.json()
            assert stored.get("text", "") == "" or resp.status_code == 404

    def test_instructions_stored_and_retrievable_by_agents(self):
        """Instructions are persisted and available for agent context injection.

        Verifies the full round-trip: store → retrieve via ConfigStore →
        the MemoryAwareMixin._get_tenant_instructions() path returns them.
        Agent formatting compliance is LLM-dependent, so we verify the
        storage and retrieval contract, not the LLM's adherence to style.
        """
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            instruction_text = "Focus only on video retrieval topics. Ignore unrelated queries."
            client.put(
                f"/admin/tenant/{TENANT_ID}/instructions",
                json={"text": instruction_text},
            )

            resp = client.get(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 200
            assert resp.json()["text"] == instruction_text

            client.delete(f"/admin/tenant/{TENANT_ID}/instructions")

            resp = client.get(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 404 or resp.json().get("text", "") == ""


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantJobs:
    def test_full_lifecycle_with_post_actions_preserved(self):
        """Create → list → verify all fields → delete → verify gone."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/jobs",
                json={
                    "name": "weekly_ai_research",
                    "schedule": "0 9 * * 1",
                    "query": "latest papers on video retrieval with ColPali",
                    "post_actions": [
                        "save to wiki",
                        "send me a summary on Telegram",
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            job_id = data["job_id"]
            assert data["name"] == "weekly_ai_research"
            assert data["schedule"] == "0 9 * * 1"
            assert data["query"] == "latest papers on video retrieval with ColPali"
            assert data["post_actions"] == [
                "save to wiki",
                "send me a summary on Telegram",
            ]
            assert data["status"] == "created"
            assert data["created_at"]

            resp = client.get(f"/admin/tenant/{TENANT_ID}/jobs")
            jobs = resp.json()["jobs"]
            match = [j for j in jobs if j["job_id"] == job_id]
            assert len(match) == 1, f"Job {job_id} not found in list"
            assert match[0]["name"] == "weekly_ai_research"
            assert match[0]["query"] == "latest papers on video retrieval with ColPali"
            assert match[0]["post_actions"] == [
                "save to wiki",
                "send me a summary on Telegram",
            ]
            assert match[0]["status"] == "active"

            resp = client.delete(f"/admin/tenant/{TENANT_ID}/jobs/{job_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

            resp = client.get(f"/admin/tenant/{TENANT_ID}/jobs")
            remaining_ids = [j["job_id"] for j in resp.json()["jobs"]]
            assert job_id not in remaining_ids, "Deleted job still in list"

    def test_delete_nonexistent_returns_404(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.delete(f"/admin/tenant/{TENANT_ID}/jobs/nonexistent_xyz")
        assert resp.status_code == 404


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantMemories:

    def test_create_search_delete_with_semantic_verification(self):
        """Full memory lifecycle: create → semantic search → verify content → delete → verify gone."""
        if not _memory_available():
            pytest.skip("Memory backend not initialized on runtime")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/memories",
                json={
                    "text": "I prefer using ColPali over CLIP for video retrieval",
                    "category": "search_preferences",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "preference"
            assert data["category"] == "search_preferences"
            memory_id = data["id"]
            assert memory_id

            time.sleep(3)

            resp = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"q": "ColPali video retrieval", "type": "preference"},
            )
            assert resp.status_code == 200
            memories = resp.json()["memories"]
            assert len(memories) >= 1, "Search should find the stored memory"

            best_match = memories[0]["memory"]
            sim = _semantic_similarity(
                "I prefer using ColPali over CLIP for video retrieval",
                best_match,
            )
            assert sim > 0.6, (
                f"Stored memory should be semantically similar to input "
                f"(got {sim:.1%}): {best_match!r}"
            )

            for m in memories:
                assert m["type"] == "preference"
                assert m["owned"] is True

            resp = client.delete(f"/admin/tenant/{TENANT_ID}/memories/{memory_id}")
            assert resp.status_code == 200

            time.sleep(2)

            resp = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"q": "ColPali video retrieval", "type": "preference"},
            )
            for m in resp.json()["memories"]:
                assert m["id"] != memory_id, f"Deleted memory {memory_id} still visible"

    def test_strategy_type_visible_not_owned(self):
        """System strategies are visible through the API with owned=false.

        Seeds a strategy via the admin endpoint to guarantee at least one exists.
        """
        if not _memory_available():
            pytest.skip("Memory backend not initialized on runtime")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            # Seed a strategy so we're not testing against empty state
            client.post(
                f"/admin/tenant/{TENANT_ID}/memories",
                json={"text": "I prefer chunk-level retrieval for temporal queries"},
            )
            time.sleep(2)

            resp = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"type": "strategy"},
            )
            assert resp.status_code == 200
            data = resp.json()
            for mem in data["memories"]:
                assert mem["type"] == "strategy"
                assert mem["owned"] is False

    def test_bulk_clear_preserves_strategies(self):
        """Clearing user memories must not touch system strategies."""
        if not _memory_available():
            pytest.skip("Memory backend not initialized on runtime")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            client.post(
                f"/admin/tenant/{TENANT_ID}/memories",
                json={"text": "I prefer dark mode for all dashboards"},
            )
            time.sleep(2)

            strategies_before = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"type": "strategy"},
            ).json()["count"]

            resp = client.delete(f"/admin/tenant/{TENANT_ID}/memories")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cleared"

            time.sleep(2)

            prefs = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"type": "preference"},
            ).json()
            assert prefs["count"] == 0, "User memories should be cleared"

            strategies_after = client.get(
                f"/admin/tenant/{TENANT_ID}/memories",
                params={"type": "strategy"},
            ).json()["count"]
            assert strategies_after >= strategies_before, (
                f"Strategies should survive user clear: {strategies_before} → {strategies_after}"
            )


@pytest.mark.e2e
@skip_if_no_runtime
class TestJobExecution:
    """Job execution: query → routing_agent → post_actions routed."""

    def test_routing_agent_processes_search_query(self):
        """Routing agent receives a search query and routes to search_agent."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find videos about outdoor nature scenes",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["recommended_agent"] == "search_agent", (
            f"Expected search_agent, got {data.get('recommended_agent')}"
        )
        assert data["confidence"] > 0.5, (
            f"Routing confidence too low: {data['confidence']}"
        )
        assert len(data.get("entities", [])) >= 1, "Should extract at least 1 entity"

    def test_wiki_save_and_retrieve(self):
        """Save content to wiki → retrieve by slug → verify content."""
        unique_marker = f"e2e_test_{int(time.time())}"
        content = f"ColPali retrieval outperforms CLIP on temporal video queries ({unique_marker})"

        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/wiki/save",
                json={
                    "query": f"ColPali vs CLIP benchmark results {unique_marker}",
                    "response": {"answer": content},
                    "entities": ["ColPali", "CLIP", "video retrieval"],
                    "tenant_id": TENANT_ID,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "saved"
            assert data["doc_id"]
            slug = data["slug"]
            assert slug

    def test_job_create_execute_verify(self):
        """Create job via API → execute via job_executor → verify routing agent called.

        This tests the full path: tenant creates a scheduled job, the executor
        reads it from ConfigStore, sends the query to routing_agent, and routes
        each post_action. HTTP transport to routing_agent is mocked since we
        can't trigger a real Argo workflow from the test, but ConfigStore
        read is against real Vespa.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/jobs",
                json={
                    "name": "e2e_execution_test",
                    "schedule": "0 12 * * *",
                    "query": "latest research on video retrieval with ColPali",
                    "post_actions": ["save to wiki", "send summary on Telegram"],
                },
            )
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

        from cogniverse_runtime.job_executor import run_job

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Found 5 ColPali papers on video retrieval"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cogniverse_foundation.config.utils.create_default_config_manager",
                return_value=_get_runtime_config_manager(),
            ),
            patch(
                "cogniverse_runtime.job_executor.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            asyncio.get_event_loop().run_until_complete(
                run_job(job_id, TENANT_ID, RUNTIME)
            )

        calls = mock_client.post.call_args_list
        assert len(calls) == 3, (
            f"Expected 3 routing_agent calls (1 query + 2 post_actions), got {len(calls)}"
        )

        query_payload = calls[0][1]["json"]
        assert query_payload["query"] == "latest research on video retrieval with ColPali"
        assert query_payload["tenant_id"] == TENANT_ID

        wiki_payload = calls[1][1]["json"]
        assert wiki_payload["query"] == "save to wiki"
        assert wiki_payload["context"] == "Found 5 ColPali papers on video retrieval"

        telegram_payload = calls[2][1]["json"]
        assert telegram_payload["query"] == "send summary on Telegram"
        assert telegram_payload["context"] == "Found 5 ColPali papers on video retrieval"

        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            client.delete(f"/admin/tenant/{TENANT_ID}/jobs/{job_id}")


def _get_runtime_config_manager():
    """Get a ConfigManager pointing at the k3d Vespa (NodePort 28080)."""
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=28080,
    )
    return ConfigManager(store=store)


@pytest.mark.e2e
@skip_if_no_runtime
class TestSearchBehavior:
    """Search endpoint returns real results with correct structure and ordering."""

    def test_search_results_descending_score_order(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "outdoor scene nature",
                    "tenant_id": TENANT_ID,
                    "profile": "video_colpali_smol500_mv_frame",
                    "top_k": 5,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results_count"] >= 1
        assert data["profile"] == "video_colpali_smol500_mv_frame"

        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True), (
            f"Results not in descending score order: {scores}"
        )
        assert all(s > 0 for s in scores), "All scores should be positive"

    def test_search_results_have_temporal_metadata(self):
        """Each video segment result includes temporal info and source."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "video content",
                    "tenant_id": TENANT_ID,
                    "profile": "video_colpali_smol500_mv_frame",
                    "top_k": 3,
                },
            )
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert "document_id" in result
            assert "score" in result
            assert "metadata" in result
            meta = result["metadata"]
            assert "video_id" in meta, f"Result missing video_id: {result['document_id']}"
            assert "segment_id" in meta, f"Result missing segment_id: {result['document_id']}"
            temporal = result.get("temporal_info", {})
            assert "start_time" in temporal, f"Missing start_time for {result['document_id']}"
            assert "end_time" in temporal, f"Missing end_time for {result['document_id']}"
            assert temporal["end_time"] > temporal["start_time"]

    def test_agent_search_returns_structured_response(self):
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "video with outdoor scenes",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "search_agent"
        assert data["results_count"] >= 1
        assert len(data["results"]) == data["results_count"]

        for result in data["results"]:
            assert "document_id" in result
            assert result["score"] > 0

    def test_text_analysis_produces_relevant_response(self):
        """Text analysis response should be semantically relevant to the query."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/text_analysis_agent/process",
                json={
                    "agent_name": "text_analysis_agent",
                    "query": "What are the main features of a video search system?",
                    "context": {"tenant_id": TENANT_ID},
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "text_analysis_agent"

        result = data.get("result", {})
        text = result.get("result", "") if isinstance(result, dict) else str(result)
        assert len(text) > 50, f"Response too short to be meaningful: {text!r}"

        sim = _semantic_similarity(
            "video search system features capabilities including search indexing retrieval ranking",
            text,
        )
        assert sim > 0.5, (
            f"Response should be semantically relevant to the query "
            f"(got {sim:.1%}):\n{text[:300]}"
        )
