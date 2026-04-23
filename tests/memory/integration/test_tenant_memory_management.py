"""
Integration tests for tenant extensibility HTTP API (memories + jobs).

Uses shared_memory_vespa fixture — real Vespa + Ollama backend.
Tests the actual HTTP endpoints with real ConfigStore and Mem0.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.routers import tenant
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def tenant_api_client(shared_memory_vespa):
    """FastAPI TestClient wired to real Vespa + Ollama for memory tests."""
    Mem0MemoryManager._instances.clear()

    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
        )
    )

    mm = Mem0MemoryManager(tenant_id="test_tenant")
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="nomic-embed-text",
        llm_base_url="http://localhost:11434",
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )

    tenant.set_config_manager(cm)
    app = FastAPI()
    app.include_router(tenant.router)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, mm

    try:
        mm.clear_agent_memory("test_tenant", "_user_memories")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


@pytest.mark.integration
class TestTenantMemoryAPI:
    """Full HTTP round-trip: POST → GET → verify type/owned/category."""

    def test_create_returns_preference_type(self, tenant_api_client):
        client, _ = tenant_api_client
        resp = client.post(
            "/test_tenant/memories",
            json={
                "text": "I prefer using ColPali for all video searches",
                "category": "search",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["type"] == "preference"
        assert data["category"] == "search"
        assert data["id"]

    def test_create_then_search_returns_owned_memory(
        self, tenant_api_client, shared_memory_vespa
    ):
        client, _ = tenant_api_client
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        client.post(
            "/test_tenant/memories",
            json={"text": "I always prefer dark mode for all interfaces"},
        )

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=3, description="memory add"
        )

        resp = client.get("/test_tenant/memories?q=dark+mode&type=preference")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

        found = False
        for mem in data["memories"]:
            assert mem["type"] == "preference"
            assert mem["owned"] is True
            if "dark" in mem["memory"].lower() or "mode" in mem["memory"].lower():
                found = True
        assert found, (
            f"Expected dark mode memory, got: {[m['memory'] for m in data['memories']]}"
        )

    def test_category_preserved_in_get(self, tenant_api_client, shared_memory_vespa):
        client, _ = tenant_api_client
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        client.post(
            "/test_tenant/memories",
            json={
                "text": "I prefer chunk-level retrieval for temporal queries",
                "category": "retrieval",
            },
        )

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=3, description="memory add"
        )

        resp = client.get("/test_tenant/memories?q=chunk+retrieval&category=retrieval")
        assert resp.status_code == 200
        data = resp.json()
        for mem in data["memories"]:
            assert mem["category"] == "retrieval"


@pytest.mark.integration
class TestTenantMemoryDelete:
    """DELETE only affects user-owned memories."""

    def test_delete_user_memory_by_id(self, tenant_api_client, shared_memory_vespa):
        client, _ = tenant_api_client
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        resp = client.post(
            "/test_tenant/memories",
            json={"text": "I prefer using Python for all data science work"},
        )
        memory_id = resp.json()["id"]
        assert memory_id

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=2, description="memory add"
        )

        resp = client.delete(f"/test_tenant/memories/{memory_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=2, description="memory delete"
        )

        resp = client.get("/test_tenant/memories?q=Python+data+science&type=preference")
        assert resp.status_code == 200
        for mem in resp.json()["memories"]:
            assert mem["id"] != memory_id, f"Deleted memory {memory_id} still returned"

    def test_bulk_clear_only_removes_user_memories(
        self, tenant_api_client, shared_memory_vespa
    ):
        client, mm = tenant_api_client
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        mm.add_memory(
            content="I prefer using Rust for all systems programming tasks",
            tenant_id="test_tenant",
            agent_name="_strategy_store",
        )

        client.post(
            "/test_tenant/memories",
            json={"text": "I prefer TensorFlow over PyTorch for deep learning"},
        )

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=3, description="memory add"
        )

        before_strategies = client.get("/test_tenant/memories?type=strategy")
        strategy_count_before = before_strategies.json()["count"]

        resp = client.delete("/test_tenant/memories")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=3, description="memory clear"
        )

        resp = client.get("/test_tenant/memories?type=preference")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0, "User memories should be gone after clear"

        resp = client.get("/test_tenant/memories?type=strategy")
        assert resp.status_code == 200
        assert resp.json()["count"] >= strategy_count_before, (
            f"Strategies should survive user clear: had {strategy_count_before}, "
            f"now {resp.json()['count']}"
        )


@pytest.mark.integration
class TestTenantMemoryTypeFilter:
    """GET ?type= restricts results to one memory type."""

    def test_unknown_type_returns_400(self, tenant_api_client):
        client, _ = tenant_api_client
        resp = client.get("/test_tenant/memories?type=bogus")
        assert resp.status_code == 400

    def test_strategy_type_returns_not_owned(
        self, tenant_api_client, shared_memory_vespa
    ):
        """Strategies created by system are visible but not owned."""
        client, mm = tenant_api_client
        vespa_url = f"http://localhost:{shared_memory_vespa['http_port']}"

        mm.add_memory(
            content="For temporal queries use chunk-level video retrieval with 30s windows",
            tenant_id="test_tenant",
            agent_name="_strategy_store",
        )

        wait_for_vespa_indexing(
            backend_url=vespa_url, delay=3, description="strategy add"
        )

        resp = client.get("/test_tenant/memories?type=strategy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1, "Strategy memory should be found"
        for mem in data["memories"]:
            assert mem["type"] == "strategy"
            assert mem["owned"] is False


@pytest.mark.integration
class TestTenantJobsCRUD:
    """Job create → list → delete round-trip against real Vespa ConfigStore."""

    def test_create_job_persists_to_configstore(self, tenant_api_client):
        client, _ = tenant_api_client
        resp = client.post(
            "/test_tenant/jobs",
            json={
                "name": "weekly_ai_search",
                "schedule": "0 9 * * 1",
                "query": "latest AI research papers on video understanding",
                "post_actions": ["save to wiki", "send me a summary on Telegram"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "weekly_ai_search"
        assert data["schedule"] == "0 9 * * 1"
        assert data["query"] == "latest AI research papers on video understanding"
        assert data["post_actions"] == ["save to wiki", "send me a summary on Telegram"]
        assert data["status"] == "created"
        assert data["job_id"]
        assert data["created_at"]

    def test_list_retrieves_created_job(self, tenant_api_client):
        client, _ = tenant_api_client

        create_resp = client.post(
            "/test_tenant/jobs",
            json={
                "name": "daily_news_check",
                "schedule": "0 8 * * *",
                "query": "tech news about LLMs and agents",
                "post_actions": ["send me a summary on Telegram"],
            },
        )
        job_id = create_resp.json()["job_id"]

        resp = client.get("/test_tenant/jobs")
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]

        found = None
        for j in jobs:
            if j["job_id"] == job_id:
                found = j
                break
        assert found is not None, f"Job {job_id} not found in list"
        assert found["name"] == "daily_news_check"
        assert found["schedule"] == "0 8 * * *"
        assert found["query"] == "tech news about LLMs and agents"
        assert found["post_actions"] == ["send me a summary on Telegram"]
        assert found["status"] == "active"

    def test_delete_job_removes_from_list(self, tenant_api_client):
        client, _ = tenant_api_client

        create_resp = client.post(
            "/test_tenant/jobs",
            json={
                "name": "disposable_job",
                "schedule": "0 0 * * *",
                "query": "test query",
            },
        )
        job_id = create_resp.json()["job_id"]

        resp = client.delete(f"/test_tenant/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert resp.json()["job_id"] == job_id

        resp = client.get("/test_tenant/jobs")
        job_ids = [j["job_id"] for j in resp.json()["jobs"]]
        assert job_id not in job_ids, f"Deleted job {job_id} still appears in list"

    def test_delete_nonexistent_job_returns_404(self, tenant_api_client):
        client, _ = tenant_api_client
        resp = client.delete("/test_tenant/jobs/no-such-job")
        assert resp.status_code == 404


@pytest.mark.integration
class TestJobExecution:
    """Job executor reads config from real Vespa and routes through routing_agent."""

    def test_executor_reads_stored_job_and_routes_query(self, tenant_api_client):
        """Create job via API → job_executor reads from same ConfigStore → calls routing_agent.

        The job is stored in real Vespa via the tenant API. The job_executor
        reads it back from the same Vespa instance, then routes the query and
        each post_action through routing_agent (mocked HTTP transport only).
        """
        client, _ = tenant_api_client

        resp = client.post(
            "/test_tenant/jobs",
            json={
                "name": "exec_test_job",
                "schedule": "0 12 * * *",
                "query": "find recent papers on ColPali retrieval",
                "post_actions": ["save to wiki"],
            },
        )
        job_id = resp.json()["job_id"]

        from cogniverse_runtime.job_executor import run_job

        test_cm = tenant._config_manager

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Found 3 ColPali papers"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cogniverse_foundation.config.utils.create_default_config_manager",
                return_value=test_cm,
            ),
            patch(
                "cogniverse_runtime.job_executor.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            asyncio.run(run_job(job_id, "test_tenant", "http://localhost:28000"))

        calls = mock_client.post.call_args_list
        assert len(calls) == 2, (
            f"Expected 2 calls (query + 1 post_action), got {len(calls)}"
        )

        first_payload = calls[0][1]["json"]
        assert first_payload["query"] == "find recent papers on ColPali retrieval"
        assert first_payload["tenant_id"] == "test_tenant"
        assert "context" not in first_payload

        # Second call is _deliver_to_wiki (pure delivery skips routing_agent)
        second_payload = calls[1][1]["json"]
        assert second_payload["query"] == "find recent papers on ColPali retrieval"
        assert second_payload["tenant_id"] == "test_tenant"
        assert second_payload["response"] == {"answer": "Found 3 ColPali papers"}
        assert second_payload["agent_name"] == "job_executor"
