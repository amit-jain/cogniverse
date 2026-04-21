"""Unit tests for the tenant extensibility router.

All external dependencies (ConfigManager, Mem0MemoryManager) are mocked
so the tests run without a live Vespa or Ollama instance.
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import tenant


@pytest.fixture(autouse=True)
def reset_config_manager():
    """Ensure module-level config manager is cleared between tests."""
    original = tenant._config_manager
    yield
    tenant._config_manager = original


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager with a backing store."""
    cm = MagicMock()
    store = MagicMock()
    cm.store = store
    return cm


@pytest.fixture
def tenant_client(mock_config_manager):
    """TestClient with the tenant router mounted and a mock config manager."""
    tenant.set_config_manager(mock_config_manager)
    app = FastAPI()
    app.include_router(tenant.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, mock_config_manager


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSetInstructions:
    def test_stores_text_in_config_manager(self, tenant_client):
        client, cm = tenant_client
        resp = client.put(
            "/acme/instructions",
            json={"text": "Always respond in bullet points."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Always respond in bullet points."
        assert data["updated_at"]

        cm.set_config_value.assert_called_once()
        call_kwargs = cm.set_config_value.call_args.kwargs
        assert call_kwargs["tenant_id"] == "acme"
        assert call_kwargs["service"] == "tenant_instructions"
        assert call_kwargs["config_key"] == "system_prompt"
        assert call_kwargs["config_value"]["text"] == "Always respond in bullet points."

    def test_returns_text_and_updated_at(self, tenant_client):
        client, _ = tenant_client
        resp = client.put("/acme/instructions", json={"text": "Be concise."})
        assert resp.status_code == 200
        data = resp.json()
        assert "updated_at" in data
        datetime.fromisoformat(data["updated_at"])


@pytest.mark.unit
@pytest.mark.ci_fast
class TestGetInstructions:
    def test_returns_stored_instructions(self, tenant_client):
        client, cm = tenant_client
        entry = MagicMock()
        entry.config_value = {
            "text": "Focus on video search.",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
        cm.store.get_config.return_value = entry

        resp = client.get("/acme/instructions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Focus on video search."
        assert data["updated_at"] == "2024-01-01T00:00:00+00:00"

    def test_404_when_not_set(self, tenant_client):
        client, cm = tenant_client
        cm.store.get_config.return_value = None

        resp = client.get("/acme/instructions")
        assert resp.status_code == 404

    def test_404_when_empty_value(self, tenant_client):
        client, cm = tenant_client
        entry = MagicMock()
        entry.config_value = {}
        cm.store.get_config.return_value = entry

        resp = client.get("/acme/instructions")
        assert resp.status_code == 404

    def test_get_passes_correct_store_args(self, tenant_client):
        client, cm = tenant_client
        cm.store.get_config.return_value = None

        client.get("/myorg/instructions")

        cm.store.get_config.assert_called_once()
        call_kwargs = cm.store.get_config.call_args.kwargs
        assert call_kwargs["tenant_id"] == "myorg"
        assert call_kwargs["service"] == "tenant_instructions"
        assert call_kwargs["config_key"] == "system_prompt"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDeleteInstructions:
    def test_clears_instructions(self, tenant_client):
        client, cm = tenant_client
        resp = client.delete("/acme/instructions")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

    def test_writes_empty_text_on_delete(self, tenant_client):
        client, cm = tenant_client
        client.delete("/acme/instructions")

        cm.set_config_value.assert_called_once()
        call_kwargs = cm.set_config_value.call_args.kwargs
        assert call_kwargs["config_value"]["text"] == ""


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCreateMemory:
    def test_creates_user_memory(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.add_memory.return_value = "mem-123"

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.post("/acme/memories", json={"text": "I prefer dark mode"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["id"] == "mem-123"
        assert data["type"] == "preference"
        # User-posted memories pass infer=False so the LLM extraction step
        # is skipped (the user wrote exactly what they want stored).
        mgr.add_memory.assert_called_once_with(
            content="I prefer dark mode",
            tenant_id="acme",
            agent_name="_user_memories",
            metadata={},
            infer=False,
        )

    def test_creates_memory_with_category(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.add_memory.return_value = "mem-456"

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.post(
                "/acme/memories",
                json={"text": "Always use ColPali", "category": "search"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "search"
        mgr.add_memory.assert_called_once_with(
            content="Always use ColPali",
            tenant_id="acme",
            agent_name="_user_memories",
            metadata={"category": "search"},
            infer=False,
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestListMemories:
    def _mock_mgr(self, memories: List[Dict[str, Any]]):
        """Return a mock Mem0MemoryManager with pre-configured memories."""
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.get_all_memories.return_value = memories
        mgr.search_memory.return_value = memories
        return mgr

    def test_list_all_returns_memories_with_type(self, tenant_client):
        client, _ = tenant_client
        raw = [
            {"id": "m1", "memory": "User prefers dark mode", "metadata": {}, "created_at": "2024-01-01"},
            {"id": "m2", "memory": "User is in UTC+5", "metadata": {}, "created_at": "2024-01-02"},
        ]
        mgr = self._mock_mgr(raw)

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2
        mem = data["memories"][0]
        assert mem["id"] == "m1"
        assert mem["type"] in ("preference", "strategy")
        assert "owned" in mem

    def test_search_query_calls_search_memory(self, tenant_client):
        client, _ = tenant_client
        raw = [{"id": "m1", "memory": "strategy A", "metadata": {}}]
        mgr = self._mock_mgr(raw)

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories?q=strategy&limit=10")

        assert resp.status_code == 200
        assert mgr.search_memory.call_count >= 2

    def test_no_query_calls_get_all(self, tenant_client):
        client, _ = tenant_client
        mgr = self._mock_mgr([])

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories")

        assert resp.status_code == 200
        assert mgr.get_all_memories.call_count >= 2

    def test_type_filter_restricts_namespace(self, tenant_client):
        client, _ = tenant_client
        raw = [{"id": "s1", "memory": "Use chunk retrieval", "metadata": {}}]
        mgr = self._mock_mgr(raw)

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories?type=strategy")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["type"] == "strategy"
        assert data["memories"][0]["owned"] is False
        assert mgr.get_all_memories.call_count == 1

    def test_unknown_type_returns_400(self, tenant_client):
        client, _ = tenant_client
        mgr = self._mock_mgr([])

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories?type=bogus")

        assert resp.status_code == 400

    def test_preference_type_is_owned(self, tenant_client):
        client, _ = tenant_client
        raw = [{"id": "p1", "memory": "dark mode", "metadata": {"category": "ui"}}]
        mgr = self._mock_mgr(raw)

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories?type=preference")

        data = resp.json()
        assert data["memories"][0]["owned"] is True
        assert data["memories"][0]["type"] == "preference"
        assert data["memories"][0]["category"] == "ui"

    def test_503_when_memory_not_initialised(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = None  # not initialised

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.get("/acme/memories")

        assert resp.status_code == 503


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDeleteMemory:
    def test_deletes_user_owned_by_id(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.delete_memory.return_value = True

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.delete("/acme/memories/mem-abc123")

        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        mgr.delete_memory.assert_called_once_with(
            memory_id="mem-abc123", tenant_id="acme", agent_name="_user_memories",
        )

    def test_404_when_delete_returns_false(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.delete_memory.return_value = False

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.delete("/acme/memories/no-such-id")

        assert resp.status_code == 404


@pytest.mark.unit
@pytest.mark.ci_fast
class TestClearMemories:
    def test_clears_user_memories_only(self, tenant_client):
        """Bulk clear without category clears all user memories, not system."""
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.clear_agent_memory.return_value = True

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.delete("/acme/memories")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        mgr.clear_agent_memory.assert_called_once_with(
            tenant_id="acme", agent_name="_user_memories",
        )

    def test_clears_by_category(self, tenant_client):
        """Clear with category filter deletes matching user memories."""
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.get_all_memories.return_value = [
            {"id": "m1", "memory": "dark mode", "metadata": {"category": "ui"}},
            {"id": "m2", "memory": "UTC+5", "metadata": {"category": "locale"}},
        ]
        mgr.delete_memory.return_value = True

        with patch("cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr):
            resp = client.delete("/acme/memories?category=ui")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cleared"
        assert data["category"] == "ui"
        assert data["deleted"] == 1
        mgr.delete_memory.assert_called_once_with(
            memory_id="m1", tenant_id="acme", agent_name="_user_memories",
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMemoryAwareMixinInstructions:
    def _build_mixin(self, tenant_id: str = "acme"):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        class FakeAgent(MemoryAwareMixin):
            def __init__(self):
                self.memory_manager = MagicMock()
                self.memory_manager.memory = MagicMock()
                self._memory_agent_name = "test_agent"
                self._memory_tenant_id = tenant_id
                self._memory_initialized = True

        return FakeAgent()

    def test_instructions_injected_before_strategies(self):
        agent = self._build_mixin()

        with (
            patch.object(agent, "get_relevant_context", return_value=None),
            patch.object(agent, "get_strategies", return_value="## Learned Strategies\n- Try A"),
            patch.object(agent, "_get_tenant_instructions", return_value="Be concise."),
        ):
            result = agent.inject_context_into_prompt("Base prompt", "test query")

        assert "Tenant Instructions" in result
        assert "Be concise." in result
        assert result.index("Tenant Instructions") < result.index("Learned Strategies")

    def test_no_instructions_does_not_add_section(self):
        agent = self._build_mixin()

        with (
            patch.object(agent, "get_relevant_context", return_value=None),
            patch.object(agent, "get_strategies", return_value=None),
            patch.object(agent, "_get_tenant_instructions", return_value=None),
        ):
            result = agent.inject_context_into_prompt("Base prompt", "test query")

        assert result == "Base prompt"
        assert "Tenant Instructions" not in result

    def test_get_tenant_instructions_returns_text_from_config(self):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        class FakeAgent(MemoryAwareMixin):
            def __init__(self):
                self.memory_manager = None
                self._memory_agent_name = "a"
                self._memory_tenant_id = "acme"
                self._memory_initialized = False

        agent = FakeAgent()

        mock_entry = MagicMock()
        mock_entry.config_value = {"text": "Always be helpful.", "updated_at": "2024-01-01"}
        mock_cm = MagicMock()
        mock_cm.store.get_config.return_value = mock_entry

        with patch(
            "cogniverse_foundation.config.utils.create_default_config_manager",
            return_value=mock_cm,
        ):
            result = agent._get_tenant_instructions()

        assert result == "Always be helpful."

    def test_get_tenant_instructions_returns_none_on_error(self):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        class FakeAgent(MemoryAwareMixin):
            def __init__(self):
                self.memory_manager = None
                self._memory_agent_name = "a"
                self._memory_tenant_id = "acme"
                self._memory_initialized = False

        agent = FakeAgent()

        with patch(
            "cogniverse_foundation.config.utils.create_default_config_manager",
            side_effect=RuntimeError("store unavailable"),
        ):
            result = agent._get_tenant_instructions()

        assert result is None

    def test_get_tenant_instructions_returns_none_for_empty_text(self):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        class FakeAgent(MemoryAwareMixin):
            def __init__(self):
                self.memory_manager = None
                self._memory_agent_name = "a"
                self._memory_tenant_id = "acme"
                self._memory_initialized = False

        agent = FakeAgent()

        mock_entry = MagicMock()
        mock_entry.config_value = {"text": "", "updated_at": "2024-01-01"}
        mock_cm = MagicMock()
        mock_cm.store.get_config.return_value = mock_entry

        with patch(
            "cogniverse_foundation.config.utils.create_default_config_manager",
            return_value=mock_cm,
        ):
            result = agent._get_tenant_instructions()

        assert result is None


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCreateJob:
    def test_stores_job_and_returns_created(self, tenant_client):
        client, cm = tenant_client
        resp = client.post(
            "/acme/jobs",
            json={
                "name": "weekly_ai_search",
                "schedule": "0 9 * * 1",
                "query": "latest AI research papers",
                "post_actions": ["save to wiki", "send me a summary on Telegram"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "weekly_ai_search"
        assert data["schedule"] == "0 9 * * 1"
        assert data["query"] == "latest AI research papers"
        assert data["post_actions"] == ["save to wiki", "send me a summary on Telegram"]
        assert data["status"] == "created"
        assert "job_id" in data
        assert data["created_at"]

        cm.set_config_value.assert_called_once()
        call_kwargs = cm.set_config_value.call_args.kwargs
        assert call_kwargs["tenant_id"] == "acme"
        assert call_kwargs["service"] == "tenant_jobs"
        stored = call_kwargs["config_value"]
        assert stored["name"] == "weekly_ai_search"
        assert stored["query"] == "latest AI research papers"

    def test_job_id_is_set_in_stored_config(self, tenant_client):
        client, cm = tenant_client
        resp = client.post(
            "/acme/jobs",
            json={"name": "daily_check", "schedule": "0 8 * * *", "query": "news"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert job_id
        stored = cm.set_config_value.call_args.kwargs["config_value"]
        assert stored["job_id"] == job_id

    def test_no_argo_url_skips_workflow_submit(self, tenant_client):
        """When _argo_api_url is None (default), no HTTP call is made."""
        client, cm = tenant_client
        original = tenant._argo_api_url
        tenant._argo_api_url = None
        try:
            resp = client.post(
                "/acme/jobs",
                json={"name": "test", "schedule": "0 9 * * 1", "query": "test query"},
            )
            assert resp.status_code == 200
        finally:
            tenant._argo_api_url = original

    def test_argo_url_set_submits_cron_workflow(self, tenant_client):
        """When _argo_api_url is configured, _submit_cron_workflow is called."""
        client, cm = tenant_client
        original = tenant._argo_api_url
        tenant._argo_api_url = "http://argo-server:2746"
        try:
            with patch(
                "cogniverse_runtime.routers.tenant._submit_cron_workflow",
                new_callable=AsyncMock,
            ) as mock_submit:
                resp = client.post(
                    "/acme/jobs",
                    json={"name": "test", "schedule": "0 9 * * 1", "query": "test query"},
                )
                assert resp.status_code == 200
                mock_submit.assert_awaited_once()
                manifest = mock_submit.call_args[0][0]
                assert manifest["kind"] == "CronWorkflow"
                assert manifest["spec"]["schedule"] == "0 9 * * 1"
        finally:
            tenant._argo_api_url = original


@pytest.mark.unit
@pytest.mark.ci_fast
class TestListJobs:
    def test_returns_jobs_from_config_store(self, tenant_client):
        client, cm = tenant_client

        job_data = {
            "job_id": "abc12345",
            "name": "weekly_search",
            "schedule": "0 9 * * 1",
            "query": "AI papers",
            "post_actions": ["summarize"],
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        entry = MagicMock()
        entry.config_value = job_data
        cm.store.list_configs.return_value = [entry]

        resp = client.get("/acme/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 1
        j = data["jobs"][0]
        assert j["job_id"] == "abc12345"
        assert j["name"] == "weekly_search"
        assert j["post_actions"] == ["summarize"]

    def test_empty_list_when_no_jobs(self, tenant_client):
        client, cm = tenant_client
        cm.store.list_configs.return_value = []

        resp = client.get("/acme/jobs")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_skips_entries_without_job_id(self, tenant_client):
        client, cm = tenant_client
        entry_bad = MagicMock()
        entry_bad.config_value = {"some": "other_config"}
        entry_good = MagicMock()
        entry_good.config_value = {
            "job_id": "good1",
            "name": "ok",
            "schedule": "* * * * *",
            "query": "q",
            "post_actions": [],
            "created_at": None,
        }
        cm.store.list_configs.return_value = [entry_bad, entry_good]

        resp = client.get("/acme/jobs")
        assert resp.status_code == 200
        assert len(resp.json()["jobs"]) == 1
        assert resp.json()["jobs"][0]["job_id"] == "good1"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDeleteJob:
    def test_deletes_existing_job(self, tenant_client):
        client, cm = tenant_client
        entry = MagicMock()
        entry.config_value = {
            "job_id": "abc12345",
            "name": "test",
            "schedule": "0 9 * * 1",
            "query": "q",
            "post_actions": [],
        }
        cm.store.get_config.return_value = entry

        resp = client.delete("/acme/jobs/abc12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["job_id"] == "abc12345"

        cm.set_config_value.assert_called()
        last_call = cm.set_config_value.call_args
        assert last_call.kwargs["config_value"]["deleted"] is True

    def test_404_when_job_not_found(self, tenant_client):
        client, cm = tenant_client
        cm.store.get_config.return_value = None

        resp = client.delete("/acme/jobs/nonexistent")
        assert resp.status_code == 404


@pytest.mark.unit
@pytest.mark.ci_fast
class TestJobExecutor:
    def test_calls_routing_agent_with_query(self):
        """job_executor._call_agent sends the right payload to routing_agent."""
        import asyncio

        from cogniverse_runtime.job_executor import _call_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "here are the results"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = asyncio.run(
            _call_agent(mock_client, "http://localhost:28000", "acme", "latest AI papers")
        )

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert "/agents/routing_agent/process" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["query"] == "latest AI papers"
        assert payload["tenant_id"] == "acme"
        assert "context" not in payload

        assert result == "here are the results"

    def test_passes_context_for_post_actions(self):
        """When context is given, it is included in the payload."""
        import asyncio

        from cogniverse_runtime.job_executor import _call_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "summarized results"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        asyncio.run(
            _call_agent(
                mock_client,
                "http://localhost:28000",
                "acme",
                "summarize this",
                context="previous result text",
            )
        )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["context"] == "previous result text"

    def test_pure_delivery_detected(self):
        """Pure delivery actions skip agent processing."""
        from cogniverse_runtime.job_executor import _is_pure_delivery

        assert _is_pure_delivery("save to wiki") is True
        assert _is_pure_delivery("send on telegram") is True
        assert _is_pure_delivery("notify me") is True
        assert _is_pure_delivery("summarize and save to wiki") is False
        assert _is_pure_delivery("create a report and send on telegram") is False


@pytest.mark.unit
@pytest.mark.ci_fast
class TestArgoEnvironmentWiring:
    """Audit fix #3 — verify the runtime startup actually wires Argo.

    Before this fix ``set_argo_config()`` was defined but never called from
    ``main.py``, so ``tenant._argo_api_url`` stayed ``None`` and POST /jobs
    silently dropped the CronWorkflow submission. These tests pin the
    helper that the lifespan calls and the round-trip from env var → POST
    /jobs → submitted manifest.
    """

    @pytest.fixture(autouse=True)
    def reset_argo_state(self):
        """Snapshot and restore tenant._argo_api_url around each test."""
        original_url = tenant._argo_api_url
        original_ns = tenant._argo_namespace
        yield
        tenant._argo_api_url = original_url
        tenant._argo_namespace = original_ns

    def test_helper_sets_argo_url_when_env_var_present(self, monkeypatch):
        """When ARGO_API_URL is set, the helper must populate the module
        state via set_argo_config()."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        tenant._argo_api_url = None
        monkeypatch.setenv("ARGO_API_URL", "http://argo-server:2746")
        monkeypatch.setenv("ARGO_NAMESPACE", "production")

        _wire_argo_from_environment()

        assert tenant._argo_api_url == "http://argo-server:2746"
        assert tenant._argo_namespace == "production"

    def test_helper_leaves_argo_url_none_when_env_var_missing(
        self, monkeypatch
    ):
        """When ARGO_API_URL is unset, the helper must explicitly set
        _argo_api_url to None — not raise — so that POST /jobs degrades
        gracefully (persist without scheduling)."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        tenant._argo_api_url = "stale-value"
        monkeypatch.delenv("ARGO_API_URL", raising=False)
        monkeypatch.delenv("ARGO_NAMESPACE", raising=False)

        _wire_argo_from_environment()

        assert tenant._argo_api_url is None
        assert tenant._argo_namespace == "cogniverse"

    def test_helper_treats_empty_string_as_unset(self, monkeypatch):
        """Helm sometimes injects empty strings for unset values; the
        helper must coerce empty → None so the conditional in create_job
        still skips submission."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        monkeypatch.setenv("ARGO_API_URL", "")

        _wire_argo_from_environment()

        assert tenant._argo_api_url is None

    def test_round_trip_env_var_set_then_post_submits_workflow(
        self, monkeypatch, tenant_client
    ):
        """End-to-end round trip: set ARGO_API_URL, run the wire helper,
        POST /jobs, assert that the CronWorkflow submission was actually
        attempted. This is the test that would have caught the original
        bug — without the wire-up call, _argo_api_url would stay None and
        _submit_cron_workflow would never be reached."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        client, _cm = tenant_client
        monkeypatch.setenv("ARGO_API_URL", "http://argo-server:2746")

        _wire_argo_from_environment()
        assert tenant._argo_api_url == "http://argo-server:2746"

        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = client.post(
                "/acme/jobs",
                json={
                    "name": "daily_news",
                    "schedule": "0 8 * * *",
                    "query": "latest AI papers",
                },
            )
            assert resp.status_code == 200
            mock_submit.assert_awaited_once()
            manifest = mock_submit.call_args[0][0]
            assert manifest["kind"] == "CronWorkflow"
            assert manifest["spec"]["schedule"] == "0 8 * * *"

    def test_round_trip_env_var_unset_then_post_skips_workflow(
        self, monkeypatch, tenant_client
    ):
        """Symmetric round trip: when ARGO_API_URL is missing, POST /jobs
        must still succeed (persist) but NOT call _submit_cron_workflow."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        client, _cm = tenant_client
        monkeypatch.delenv("ARGO_API_URL", raising=False)

        _wire_argo_from_environment()
        assert tenant._argo_api_url is None

        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = client.post(
                "/acme/jobs",
                json={
                    "name": "daily_news",
                    "schedule": "0 8 * * *",
                    "query": "latest AI papers",
                },
            )
            assert resp.status_code == 200
            mock_submit.assert_not_awaited()
