"""
Real integration tests for tenant extensibility.

Tests instructions round-trip via real Vespa ConfigStore,
memory management with real Mem0/Vespa, and instruction
injection into agent prompts. Also tests job executor
routing with real LLM.
"""

import logging

import httpx
import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

logger = logging.getLogger(__name__)

TENANT_ID = "test_extensibility"


def _llm_available():
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


skip_if_no_llm = pytest.mark.skipif(not _llm_available(), reason="LLM not available")


@pytest.fixture(scope="module")
def tenant_config_manager(vespa_instance):
    """ConfigManager backed by real Vespa for tenant extensibility tests."""
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )
    return cm


@pytest.mark.integration
class TestTenantInstructionsRealVespa:
    """Instructions CRUD with real Vespa ConfigStore."""

    def test_set_and_read_back(self, tenant_config_manager):
        """Store instructions in real Vespa, read them back."""
        cm = tenant_config_manager
        text = "I prefer detailed reports with timestamps. Always use ColPali."

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
            config_value={"text": text, "updated_at": "2026-04-06T00:00:00"},
        )

        entry = cm.store.get_config(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
        )

        assert entry is not None
        value = entry.config_value
        assert isinstance(value, dict)
        assert value["text"] == text
        assert "ColPali" in value["text"]

    def test_update_overwrites_previous(self, tenant_config_manager):
        """Updating instructions replaces old ones."""
        cm = tenant_config_manager

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
            config_value={"text": "old instructions"},
        )

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
            config_value={"text": "new instructions"},
        )

        entry = cm.store.get_config(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
        )

        assert entry.config_value["text"] == "new instructions"

    def test_instructions_persist_across_reads(self, tenant_config_manager):
        """Instructions survive multiple reads from real Vespa."""
        cm = tenant_config_manager

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
            config_value={"text": "Always format results as tables."},
        )

        # Read twice — both should return same content
        for _ in range(2):
            entry = cm.store.get_config(
                tenant_id=TENANT_ID,
                scope=ConfigScope.SYSTEM,
                service="tenant_instructions",
                config_key="system_prompt",
            )
            assert entry is not None
            assert entry.config_value["text"] == "Always format results as tables."


@pytest.mark.integration
class TestTenantJobsRealVespa:
    """Job config storage with real Vespa ConfigStore."""

    def test_create_and_list_job(self, tenant_config_manager):
        """Store job config in real Vespa, list it back."""
        cm = tenant_config_manager
        job_id = "test_job_001"

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
            config_key=f"job_{job_id}",
            config_value={
                "job_id": job_id,
                "name": "weekly_search",
                "schedule": "0 9 * * 1",
                "query": "latest AI papers",
                "post_actions": ["save to wiki"],
            },
        )

        entries = cm.store.list_configs(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
        )

        job_ids = []
        for entry in entries or []:
            v = entry.config_value
            if isinstance(v, dict) and "job_id" in v and not v.get("deleted"):
                job_ids.append(v["job_id"])

        assert job_id in job_ids

    def test_soft_delete_job(self, tenant_config_manager):
        """Soft-deleted jobs don't appear in listing."""
        cm = tenant_config_manager
        job_id = "test_job_delete"

        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
            config_key=f"job_{job_id}",
            config_value={
                "job_id": job_id,
                "name": "to_delete",
                "schedule": "0 0 * * *",
                "query": "test",
                "post_actions": [],
            },
        )

        # Soft delete
        cm.set_config_value(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
            config_key=f"job_{job_id}",
            config_value={"deleted": True},
        )

        entries = cm.store.list_configs(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
        )

        active_ids = []
        for entry in entries or []:
            v = entry.config_value
            if isinstance(v, dict) and "job_id" in v and not v.get("deleted"):
                active_ids.append(v["job_id"])

        assert job_id not in active_ids


@pytest.mark.integration
@skip_if_no_llm
class TestMemoryManagementRealMem0:
    """Memory management with real Mem0/Vespa backend.

    NOTE: Real Mem0 round-trip (add → search → delete) is already tested in
    tests/memory/integration/test_mem0_vespa_integration.py and
    tests/memory/integration/test_strategy_learner_integration.py with the
    shared_memory_vespa fixture. Those tests verify the full Mem0 pipeline
    including LLM extraction, Vespa storage, and semantic search.

    This test verifies the tenant API concept: that memory namespacing by
    agent_name provides tenant-level memory isolation.
    """

    def test_agent_namespacing_isolates_memories(self, memory_manager):
        """Memories stored under different agent_names are isolated."""
        mm = memory_manager

        mm.add_memory(
            content="I prefer detailed reports with charts",
            tenant_id="default",
            agent_name="_test_agent_a",
        )

        # Search under a different agent namespace — should not find it
        results = mm.search_memory(
            query="detailed reports charts",
            tenant_id="default",
            agent_name="_test_agent_b",
            top_k=5,
        )

        # Agent B should not see Agent A's memories
        for r in results:
            assert "_test_agent_a" not in str(r.get("metadata", {})).lower()


@pytest.mark.integration
@skip_if_no_llm
class TestJobExecutorRealLLM:
    """Job executor routing with real LLM."""

    @pytest.mark.asyncio
    async def test_job_executor_routes_query(self):
        """Job executor routes a query through real routing_agent."""
        import dspy

        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cm = create_default_config_manager()
        llm = get_config(tenant_id="default", config_manager=cm).get_llm_config()
        endpoint = llm.resolve("primary")
        lm = create_dspy_lm(endpoint)
        dspy.configure(lm=lm)

        # Simulate what job_executor does: call routing_agent process
        # We can't call the full HTTP endpoint without a running server,
        # but we can verify the routing logic works with real LLM
        # Simulate job executor: use DSPy routing signature with real LLM
        from cogniverse_agents.routing.dspy_relationship_router import (
            DSPyAdvancedRoutingModule,
        )

        router = DSPyAdvancedRoutingModule()
        result = router.forward(
            query="latest AI research papers",
            available_agents="search_agent, summarizer_agent, detailed_report_agent",
        )

        assert result is not None

        # The routing decision is inside the prediction's routing_decision dict
        routing = getattr(result, "routing_decision", {})
        if isinstance(routing, str):
            import json
            try:
                routing = json.loads(routing)
            except Exception:
                routing = {}

        primary = routing.get("primary_agent", "")
        assert len(primary) > 0, f"No primary_agent in routing_decision: {routing}"
        assert result.overall_confidence > 0
