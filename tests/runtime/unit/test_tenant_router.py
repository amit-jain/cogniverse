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


@pytest.fixture(autouse=True)
def reset_argo_module_state():
    """Reset Argo module-level state between tests.

    `set_argo_config` stores api_url/namespace/service_account in module
    globals; without a reset, one test's snapshot bleeds into the next."""
    snapshot = (
        tenant._argo_api_url,
        tenant._argo_namespace,
        tenant._runtime_service_account,
    )
    tenant._argo_api_url = None
    tenant._argo_namespace = "cogniverse"
    tenant._runtime_service_account = "default"
    yield
    (
        tenant._argo_api_url,
        tenant._argo_namespace,
        tenant._runtime_service_account,
    ) = snapshot


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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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
            {
                "id": "m1",
                "memory": "User prefers dark mode",
                "metadata": {},
                "created_at": "2024-01-01",
            },
            {
                "id": "m2",
                "memory": "User is in UTC+5",
                "metadata": {},
                "created_at": "2024-01-02",
            },
        ]
        mgr = self._mock_mgr(raw)

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.get("/acme/memories?q=strategy&limit=10")

        assert resp.status_code == 200
        assert mgr.search_memory.call_count >= 2

    def test_no_query_calls_get_all(self, tenant_client):
        client, _ = tenant_client
        mgr = self._mock_mgr([])

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.get("/acme/memories")

        assert resp.status_code == 200
        assert mgr.get_all_memories.call_count >= 2

    def test_type_filter_restricts_namespace(self, tenant_client):
        client, _ = tenant_client
        raw = [{"id": "s1", "memory": "Use chunk retrieval", "metadata": {}}]
        mgr = self._mock_mgr(raw)

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.get("/acme/memories?type=bogus")

        assert resp.status_code == 400

    def test_preference_type_is_owned(self, tenant_client):
        client, _ = tenant_client
        raw = [{"id": "p1", "memory": "dark mode", "metadata": {"category": "ui"}}]
        mgr = self._mock_mgr(raw)

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.get("/acme/memories?type=preference")

        data = resp.json()
        assert data["memories"][0]["owned"] is True
        assert data["memories"][0]["type"] == "preference"
        assert data["memories"][0]["category"] == "ui"

    def test_503_when_memory_not_initialised(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = None  # not initialised

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.delete("/acme/memories/mem-abc123")

        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        mgr.delete_memory.assert_called_once_with(
            memory_id="mem-abc123",
            tenant_id="acme",
            agent_name="_user_memories",
        )

    def test_404_when_delete_returns_false(self, tenant_client):
        client, _ = tenant_client
        mgr = MagicMock()
        mgr.memory = MagicMock()
        mgr.delete_memory.return_value = False

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.delete("/acme/memories")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        mgr.clear_agent_memory.assert_called_once_with(
            tenant_id="acme",
            agent_name="_user_memories",
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

        with patch(
            "cogniverse_runtime.routers.tenant.Mem0MemoryManager", return_value=mgr
        ):
            resp = client.delete("/acme/memories?category=ui")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cleared"
        assert data["category"] == "ui"
        assert data["deleted"] == 1
        mgr.delete_memory.assert_called_once_with(
            memory_id="m1",
            tenant_id="acme",
            agent_name="_user_memories",
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
            patch.object(
                agent, "get_strategies", return_value="## Learned Strategies\n- Try A"
            ),
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
        mock_entry.config_value = {
            "text": "Always be helpful.",
            "updated_at": "2024-01-01",
        }
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
                    json={
                        "name": "test",
                        "schedule": "0 9 * * 1",
                        "query": "test query",
                    },
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
    def test_calls_orchestrator_agent_with_query(self):
        """job_executor._call_agent sends the right payload to orchestrator_agent."""
        import asyncio

        from cogniverse_runtime.job_executor import _call_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "here are the results"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = asyncio.run(
            _call_agent(
                mock_client, "http://localhost:28000", "acme", "latest AI papers"
            )
        )

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert "/agents/orchestrator_agent/process" in call_args[0][0]
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

    def test_helper_leaves_argo_url_none_when_env_var_missing(self, monkeypatch):
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

    def test_helper_reads_service_account_from_env(self, monkeypatch):
        """``_wire_argo_from_environment`` reads ``RUNTIME_SERVICE_ACCOUNT``
        so the SA the chart RBAC-bound to ``workflowtaskresults`` lands on
        the submitted Workflow's ``spec.serviceAccountName``."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        monkeypatch.setenv("ARGO_API_URL", "http://argo:2746")
        monkeypatch.setenv("RUNTIME_SERVICE_ACCOUNT", "cogniverse-runner")

        _wire_argo_from_environment()

        assert tenant._runtime_service_account == "cogniverse-runner"

    def test_helper_defaults_service_account_when_missing(self, monkeypatch):
        """Without RUNTIME_SERVICE_ACCOUNT set, the default 'default' is
        used so local dev still works. Production Helm values always set
        it; defaulting is for dev/tests."""
        from cogniverse_runtime.main import _wire_argo_from_environment

        monkeypatch.setenv("ARGO_API_URL", "http://argo:2746")
        monkeypatch.delenv("RUNTIME_SERVICE_ACCOUNT", raising=False)

        _wire_argo_from_environment()

        assert tenant._runtime_service_account == "default"

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


@pytest.fixture
def argo_configured_client(mock_config_manager):
    """TestClient with Argo configured so /optimize endpoints are live."""
    tenant.set_config_manager(mock_config_manager)
    tenant.set_argo_config(
        api_url="http://argo.test:2746",
        namespace="cogniverse",
    )
    app = FastAPI()
    app.include_router(tenant.router)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        tenant.set_argo_config(api_url=None, namespace="cogniverse")


@pytest.mark.unit
@pytest.mark.ci_fast
class TestBuildOptimizationWorkflowManifest:
    """Manifest builder produces a runnable one-off Workflow spec."""

    def test_manifest_shape_and_ttl(self):
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="acme", mode="simba", namespace="cogniverse"
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

        assert manifest["apiVersion"] == "argoproj.io/v1alpha1"
        assert manifest["kind"] == "Workflow"
        assert manifest["metadata"]["generateName"] == "manual-optimize-simba-"
        assert manifest["metadata"]["namespace"] == "cogniverse"

        labels = manifest["metadata"]["labels"]
        assert labels["cogniverse.ai/trigger"] == "manual"
        assert labels["cogniverse.ai/mode"] == "simba"
        assert labels["cogniverse.ai/tenant"] == "acme"

        ttl = manifest["spec"]["ttlStrategy"]
        assert ttl["secondsAfterCompletion"] == 3600
        assert ttl["secondsAfterSuccess"] == 3600
        assert ttl["secondsAfterFailure"] == 3600

    def test_manifest_sets_service_account_name(self):
        """Workflow pods must run under the configured runtime SA so Argo
        Emissary has RBAC to post ``workflowtaskresults``. Without it Argo
        marks otherwise-successful Workflows as ``Error``."""
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
            service_account="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="acme", mode="simba", namespace="cogniverse"
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

        assert manifest["spec"]["serviceAccountName"] == "cogniverse"

    def test_manifest_falls_back_to_default_sa_when_unconfigured(self):
        """Local/dev/unit-test callers that never set the SA get a sane
        fallback — the Workflow still serialises."""
        # Reset to module defaults explicitly.
        tenant._runtime_service_account = "default"
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="acme", mode="simba", namespace="cogniverse"
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

        assert manifest["spec"]["serviceAccountName"] == "default"

    def test_tenant_label_sanitized_for_k8s(self):
        """K8s labels reject ``:`` (and other chars common in tenant IDs).
        The tenant label must be sanitized, while the CLI arg preserves
        the raw ID so the optimizer loads the right tenant config."""
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="flywheel_org:production",
                mode="simba",
                namespace="cogniverse",
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

        label = manifest["metadata"]["labels"]["cogniverse.ai/tenant"]
        assert ":" not in label
        assert label == "flywheel_org-production"
        # The template argument still carries the original tenant_id — the
        # optimizer needs it unmodified to resolve config / Phoenix project.
        params = {
            p["name"]: p["value"] for p in manifest["spec"]["arguments"]["parameters"]
        }
        assert params["tenant-id"] == "flywheel_org:production"

    def test_sanitize_label_value_edge_cases(self):
        assert tenant._sanitize_label_value("acme") == "acme"
        assert tenant._sanitize_label_value("acme:prod") == "acme-prod"
        assert tenant._sanitize_label_value(":::") == "unknown"
        assert tenant._sanitize_label_value("_leading-trail.") == "leading-trail"
        assert len(tenant._sanitize_label_value("x" * 100)) == 63

    def test_manifest_references_workflow_template_with_arguments(self):
        """The runtime delegates container spec to the cluster-installed
        WorkflowTemplate. What the runtime owns is the Workflow's identity
        (name, labels, TTL, SA) and the arguments passed to the template."""
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="acme:prod", mode="gateway-thresholds", namespace="ns1"
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

        spec = manifest["spec"]
        assert spec["workflowTemplateRef"] == {
            "name": "cogniverse-optimization-runner",
        }
        # Container image / env / resources live in the WorkflowTemplate now,
        # not the submitted Workflow.
        assert "templates" not in spec
        assert "entrypoint" not in spec

        params = {p["name"]: p["value"] for p in spec["arguments"]["parameters"]}
        assert params == {
            "mode": "gateway-thresholds",
            "tenant-id": "acme:prod",
            "lookback-hours": "48",
        }

    def test_manifest_template_ref_is_constant(self):
        """The WorkflowTemplate name is the contract between the chart and
        the runtime — hardcoded string, not a module global, so a typo in
        one side fails unit-test parity with the other."""
        tenant.set_argo_config(
            api_url="http://argo.test:2746",
            namespace="cogniverse",
        )
        try:
            manifest = tenant._build_optimization_workflow_manifest(
                tenant_id="t", mode="profile", namespace="cogniverse"
            )
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")
        # Must match the WorkflowTemplate metadata.name produced by
        # charts/cogniverse/templates/optimization-workflow-template.yaml
        # (which uses include "cogniverse.fullname" = release name).
        assert (
            manifest["spec"]["workflowTemplateRef"]["name"]
            == "cogniverse-optimization-runner"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPostManualOptimize:
    """Real FastAPI route → dispatcher → manifest → Argo POST, happy + error paths."""

    def test_submits_workflow_with_correct_body(self, argo_configured_client):
        """Happy path: route calls Argo POST /workflows with a Workflow manifest
        carrying the selected mode + tenant, and returns the name Argo assigned."""
        client = argo_configured_client
        captured: Dict[str, Any] = {}

        async def fake_post(self, url, json=None, **kwargs):
            captured["url"] = url
            captured["body"] = json
            resp = MagicMock()
            resp.status_code = 201
            resp.json = MagicMock(
                return_value={"metadata": {"name": "manual-optimize-simba-abcde"}}
            )
            return resp

        with patch("httpx.AsyncClient.post", new=fake_post):
            resp = client.post("/acme/optimize", json={"mode": "simba"})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["workflow_name"] == "manual-optimize-simba-abcde"
        assert data["namespace"] == "cogniverse"
        assert data["mode"] == "simba"
        assert (
            data["status_url"]
            == "/admin/tenant/acme/optimize/runs/manual-optimize-simba-abcde"
        )

        # Argo contract: endpoint + manifest wrapping
        assert captured["url"] == "http://argo.test:2746/api/v1/workflows/cogniverse"
        assert "workflow" in captured["body"]
        submitted = captured["body"]["workflow"]
        assert submitted["kind"] == "Workflow"
        assert submitted["metadata"]["labels"]["cogniverse.ai/mode"] == "simba"
        assert submitted["metadata"]["labels"]["cogniverse.ai/tenant"] == "acme"

    def test_rejects_unknown_mode(self, argo_configured_client):
        resp = argo_configured_client.post(
            "/acme/optimize", json={"mode": "not-a-real-mode"}
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "not-a-real-mode" in detail
        assert "entity-extraction" in detail  # shows supported list

    def test_503_when_argo_not_configured(self, mock_config_manager):
        tenant.set_config_manager(mock_config_manager)
        tenant.set_argo_config(api_url=None, namespace="cogniverse")
        app = FastAPI()
        app.include_router(tenant.router)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/acme/optimize", json={"mode": "simba"})
        assert resp.status_code == 503
        assert "Argo" in resp.json()["detail"]

    def test_502_when_argo_returns_error(self, argo_configured_client):
        client = argo_configured_client

        async def fake_post(self, url, json=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "internal server error"
            return resp

        with patch("httpx.AsyncClient.post", new=fake_post):
            resp = client.post("/acme/optimize", json={"mode": "simba"})
        assert resp.status_code == 502
        assert "500" in resp.json()["detail"]

    def test_502_when_argo_returns_no_workflow_name(self, argo_configured_client):
        """Defensive: if Argo accepts but omits the assigned name, fail loud."""
        client = argo_configured_client

        async def fake_post(self, url, json=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 201
            resp.json = MagicMock(return_value={"metadata": {}})
            return resp

        with patch("httpx.AsyncClient.post", new=fake_post):
            resp = client.post("/acme/optimize", json={"mode": "simba"})
        assert resp.status_code == 502


@pytest.mark.unit
@pytest.mark.ci_fast
class TestGetManualOptimizeStatus:
    """GET status proxies the Argo workflow phase + timestamps."""

    def test_extracts_phase_and_timestamps(self, argo_configured_client):
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            assert url == "http://argo.test:2746/api/v1/workflows/cogniverse/wf-abc"
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "status": {
                        "phase": "Succeeded",
                        "startedAt": "2025-01-01T12:00:00Z",
                        "finishedAt": "2025-01-01T12:04:00Z",
                        "message": "ok",
                    }
                }
            )
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-abc")

        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_name"] == "wf-abc"
        assert data["phase"] == "Succeeded"
        assert data["started_at"] == "2025-01-01T12:00:00Z"
        assert data["finished_at"] == "2025-01-01T12:04:00Z"
        assert data["message"] == "ok"

    def test_pending_workflow_has_no_timestamps(self, argo_configured_client):
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            # Argo omits startedAt/finishedAt while pending; phase may also be missing.
            resp.json = MagicMock(return_value={"status": {}})
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] is None
        assert data["started_at"] is None
        assert data["finished_at"] is None

    def test_404_from_argo_propagates(self, argo_configured_client):
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 404
            resp.text = "not found"
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/missing")
        assert resp.status_code == 404

    def test_502_when_argo_returns_other_error(self, argo_configured_client):
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "down"
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf1")
        assert resp.status_code == 502

    def test_pending_due_to_mutex_wait_surfaces_reason(self, argo_configured_client):
        """When Argo records a mutex wait under ``synchronization.mutex.waiting``,
        the status endpoint must propagate it as ``blocked_reason`` so the
        dashboard can distinguish mutex-pending from ordinary scheduler-pending."""
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "status": {
                        "phase": "Pending",
                        "synchronization": {
                            "mutex": {
                                "waiting": [
                                    {
                                        "mutex": "cogniverse/optimize-acme",
                                        "holder": "cogniverse/manual-optimize-gateway-abc",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-queued")

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "Pending"
        assert data["blocked_reason"] is not None
        assert "cogniverse/optimize-acme" in data["blocked_reason"]
        assert "Waiting for another optimization" in data["blocked_reason"]

    def test_pending_without_mutex_wait_returns_no_blocked_reason(
        self, argo_configured_client
    ):
        """Ordinary scheduler-pending (no mutex wait) must NOT produce a
        blocked_reason — the dashboard would mis-label a cold-start pod
        pull as a concurrency wait."""
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value={"status": {"phase": "Pending"}})
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-cold")

        assert resp.status_code == 200
        assert resp.json()["blocked_reason"] is None

    def test_running_phase_never_returns_blocked_reason(self, argo_configured_client):
        """A Running Workflow cannot be mutex-blocked (it already acquired).
        Even if Argo's older versions accidentally leave synchronization
        metadata around, we must not surface it once the phase flipped."""
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "status": {
                        "phase": "Running",
                        "synchronization": {
                            "mutex": {
                                "waiting": [
                                    {"mutex": "cogniverse/optimize-acme", "holder": "x"}
                                ]
                            }
                        },
                    }
                }
            )
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-running")

        assert resp.status_code == 200
        assert resp.json()["blocked_reason"] is None

    def test_pending_with_message_hint_falls_back_to_message_text(
        self, argo_configured_client
    ):
        """Older Argo versions put the wait hint in ``status.message`` instead
        of the structured ``synchronization`` block. The extractor must
        still surface it so users on those versions aren't stuck guessing."""
        client = argo_configured_client

        async def fake_get(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "status": {
                        "phase": "Pending",
                        "message": "Waiting for cogniverse/optimize-acme lock",
                    }
                }
            )
            return resp

        with patch("httpx.AsyncClient.get", new=fake_get):
            resp = client.get("/acme/optimize/runs/wf-old-argo")

        data = resp.json()
        assert data["blocked_reason"] == "Waiting for cogniverse/optimize-acme lock"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCancelManualOptimize:
    """POST /optimize/runs/{wf}/cancel proxies ``terminate`` to Argo and
    returns the post-terminate status block."""

    def test_terminates_workflow_and_returns_phase(self, argo_configured_client):
        client = argo_configured_client
        captured: Dict[str, Any] = {}

        async def fake_put(self, url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "metadata": {"name": "wf-running"},
                    "status": {
                        "phase": "Failed",
                        "message": "Terminated by user",
                        "startedAt": "2025-01-01T12:00:00Z",
                        "finishedAt": "2025-01-01T12:01:00Z",
                    },
                }
            )
            return resp

        with patch("httpx.AsyncClient.put", new=fake_put):
            resp = client.post("/acme/optimize/runs/wf-running/cancel")

        assert resp.status_code == 200, resp.text
        assert (
            captured["url"]
            == "http://argo.test:2746/api/v1/workflows/cogniverse/wf-running/terminate"
        )
        assert captured["json"] == {"name": "wf-running"}
        data = resp.json()
        assert data["phase"] == "Failed"
        assert data["message"] == "Terminated by user"
        assert data["workflow_name"] == "wf-running"

    def test_404_when_workflow_missing(self, argo_configured_client):
        client = argo_configured_client

        async def fake_put(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 404
            resp.text = "not found"
            return resp

        with patch("httpx.AsyncClient.put", new=fake_put):
            resp = client.post("/acme/optimize/runs/gone/cancel")
        assert resp.status_code == 404

    def test_502_when_argo_returns_error(self, argo_configured_client):
        client = argo_configured_client

        async def fake_put(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "down"
            return resp

        with patch("httpx.AsyncClient.put", new=fake_put):
            resp = client.post("/acme/optimize/runs/wf1/cancel")
        assert resp.status_code == 502

    def test_503_when_argo_not_configured(self, tenant_client):
        client, _cm = tenant_client
        resp = client.post("/acme/optimize/runs/wf/cancel")
        assert resp.status_code == 503


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRetryManualOptimize:
    """POST /optimize/runs/{wf}/retry proxies ``retry`` to Argo (restarts
    only the failed nodes, reusing successful ones)."""

    def test_retry_calls_argo_retry_endpoint(self, argo_configured_client):
        client = argo_configured_client
        captured: Dict[str, Any] = {}

        async def fake_put(self, url, json=None, **kwargs):
            captured["url"] = url
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(
                return_value={
                    "metadata": {"name": "wf-failed"},
                    "status": {"phase": "Running"},
                }
            )
            return resp

        with patch("httpx.AsyncClient.put", new=fake_put):
            resp = client.post("/acme/optimize/runs/wf-failed/retry")

        assert resp.status_code == 200
        # Must hit Argo's /retry (not /terminate).
        assert captured["url"].endswith("/wf-failed/retry")
        assert resp.json()["phase"] == "Running"

    def test_404_when_workflow_missing(self, argo_configured_client):
        client = argo_configured_client

        async def fake_put(self, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 404
            resp.text = "not found"
            return resp

        with patch("httpx.AsyncClient.put", new=fake_put):
            resp = client.post("/acme/optimize/runs/gone/retry")
        assert resp.status_code == 404


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSetArgoConfig:
    """set_argo_config stores api_url / namespace / service_account; default
    for service_account is preserved across calls that omit it."""

    def test_updates_all_fields(self):
        tenant.set_argo_config(
            api_url="http://x:1",
            namespace="ns",
            service_account="runner",
        )
        try:
            assert tenant._argo_api_url == "http://x:1"
            assert tenant._argo_namespace == "ns"
            assert tenant._runtime_service_account == "runner"
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")

    def test_omitting_service_account_preserves_prior_snapshot(self):
        """Calling set_argo_config without service_account keeps the previously
        snapshotted value — guards against clearing the SA on a later
        argo-disable call."""
        tenant.set_argo_config(
            api_url="http://x:1",
            namespace="ns",
            service_account="runner-keep",
        )
        try:
            tenant.set_argo_config(api_url="http://y:2", namespace="ns2")
            assert tenant._runtime_service_account == "runner-keep"
            assert tenant._argo_api_url == "http://y:2"
            assert tenant._argo_namespace == "ns2"
        finally:
            tenant.set_argo_config(api_url=None, namespace="cogniverse")
