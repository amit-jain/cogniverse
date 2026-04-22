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
    """Memory management with real Mem0/Vespa.

    Full Mem0 add → search → delete round-trip is tested in
    tests/memory/integration/test_mem0_vespa_integration.py using the
    shared_memory_vespa fixture which correctly handles port assignment.

    The runtime integration conftest's memory_manager has a known
    VespaSearchBackend port issue for search (config.json port 8080 leaks).
    These tests verify the add_memory path (which uses the feed API on the
    correct port) and the clear_agent_memory path.
    """

    def test_add_then_search_round_trip(self, memory_manager):
        """Add memory via Mem0, search it back from real Vespa."""
        mm = memory_manager

        mm.add_memory(
            content="I always prefer using ColPali model for video searches",
            tenant_id="test:unit",
            agent_name="_test_tenant_ext_rt",
        )

        results = mm.search_memory(
            query="ColPali video search preference",
            tenant_id="test:unit",
            agent_name="_test_tenant_ext_rt",
            top_k=5,
        )

        assert len(results) >= 1, (
            "Should find the memory we just added via real Vespa search"
        )
        all_text = " ".join(r.get("memory", "") for r in results).lower()
        assert "colpali" in all_text or "video" in all_text or "search" in all_text, (
            f"Memory should reference ColPali/video/search, got: {all_text}"
        )

    def test_clear_agent_memory_removes_all(self, memory_manager):
        """clear_agent_memory removes all memories for that namespace."""
        mm = memory_manager

        # infer=False: Mem0's LLM extraction on a short synthetic sentence
        # is too brittle to rely on in a clear-then-verify test. The purpose
        # here is clear_agent_memory, not LLM-driven fact distillation.
        mm.add_memory(
            content="Temporary data for clear test",
            tenant_id="test:unit",
            agent_name="_test_tenant_ext_clear",
            infer=False,
        )

        success = mm.clear_agent_memory(
            tenant_id="test:unit",
            agent_name="_test_tenant_ext_clear",
        )
        assert success is True

        results = mm.search_memory(
            query="temporary data clear test",
            tenant_id="test:unit",
            agent_name="_test_tenant_ext_clear",
            top_k=5,
        )
        assert len(results) == 0, (
            f"After clear, should find 0 memories, got {len(results)}"
        )


@pytest.mark.integration
@skip_if_no_llm
class TestJobExecutorRealLLM:
    """Job executor routing with real LLM — verifies the routing module
    produces meaningful agent selections for different query types."""

    @pytest.mark.asyncio
    async def test_search_query_routes_to_search(self):
        """'Find AI papers' should route to a search-related agent."""
        import dspy

        from cogniverse_agents.routing.dspy_relationship_router import (
            DSPyAdvancedRoutingModule,
        )
        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cm = create_default_config_manager()
        llm = get_config(tenant_id="test:unit", config_manager=cm).get_llm_config()
        endpoint = llm.resolve("primary")
        lm = create_dspy_lm(endpoint)
        dspy.configure(lm=lm)

        router = DSPyAdvancedRoutingModule()
        result = router.forward(
            query="Find the latest AI research papers on transformers",
            available_agents="search_agent, summarizer_agent, detailed_report_agent",
        )

        assert result is not None
        assert result.overall_confidence > 0.3, (
            f"Confidence too low: {result.overall_confidence}"
        )

        # Verify the routing produced a meaningful analysis
        analysis = getattr(result, "query_analysis", {})
        if isinstance(analysis, str):
            import json
            try:
                analysis = json.loads(analysis)
            except Exception:
                analysis = {}

        intent = analysis.get("primary_intent", "")
        assert intent in ("search", "information_extraction", "content_discovery"), (
            f"Expected search-like intent for 'Find papers', got: {intent}"
        )

        # Verify entities were extracted
        entities = getattr(result, "extracted_entities", [])
        if isinstance(entities, str):
            import json
            try:
                entities = json.loads(entities)
            except Exception:
                entities = []
        assert len(entities) >= 1, (
            f"Should extract at least 1 entity from 'AI research papers on transformers', got: {entities}"
        )

    @pytest.mark.asyncio
    async def test_summary_query_produces_meaningful_analysis(self):
        """A summary query produces a non-trivial analysis with entities."""
        import dspy

        from cogniverse_agents.routing.dspy_relationship_router import (
            DSPyAdvancedRoutingModule,
        )
        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        cm = create_default_config_manager()
        llm = get_config(tenant_id="test:unit", config_manager=cm).get_llm_config()
        endpoint = llm.resolve("primary")
        lm = create_dspy_lm(endpoint)

        router = DSPyAdvancedRoutingModule()

        with dspy.context(lm=lm):
            result = router.forward(
                query="Write a comprehensive summary of all neural network architectures",
                available_agents="search_agent, summarizer_agent, detailed_report_agent",
            )

        assert result is not None
        assert result.overall_confidence > 0.3, (
            f"Confidence too low: {result.overall_confidence}"
        )

        # Should extract "neural network" as an entity
        entities = getattr(result, "extracted_entities", [])
        if isinstance(entities, str):
            import json
            try:
                entities = json.loads(entities)
            except Exception:
                entities = []

        assert len(entities) >= 1, (
            f"Should extract entities from 'neural network architectures', got: {entities}"
        )

        # Should produce an enhanced query that's different from the original
        enhanced = getattr(result, "enhanced_query", "")
        assert len(str(enhanced)) > 0, "Should produce an enhanced query"
