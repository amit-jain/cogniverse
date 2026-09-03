"""
Real integration tests for tenant extensibility.

Tests instructions round-trip via real Vespa ConfigStore,
memory management with real Mem0/Vespa, and instruction
injection into agent prompts. Also tests job executor
routing with real LLM.
"""

import logging
import time

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore

logger = logging.getLogger(__name__)

# Canonical org:tenant form. ConfigManager.set_config_value canonicalizes the
# tenant id before writing, while cm.store.get_config/list_configs match the
# passed id exactly — a bare id here would write under one scope and read
# from another, matching nothing.
TENANT_ID = "test_extensibility:test_extensibility"


def _llm_available():
    """Cheap reachability probe for the test LM provisioned by the
    session-scoped ``ensure_host_ollama`` fixture (tests/conftest.py).

    MUST NOT spawn: this feeds a module-level ``skipif`` evaluated at
    collection time, so a model-loading call here would block collection."""
    from tests.fixtures.llm import is_test_lm_available

    return is_test_lm_available()


# Runtime LM gate: the requires_lm marker is enforced per test by
# ``pytest_runtest_setup`` in tests/conftest.py (an import-time skipif
# latches the pre-session-fixture endpoint state).
skip_if_no_lm = pytest.mark.requires_lm


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
        """A created job appears in the listing until soft-deleted, then
        disappears — the presence check first, so the absence check below
        cannot pass vacuously against an empty listing."""
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

        entries = cm.store.list_configs(
            tenant_id=TENANT_ID,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
        )

        listed_ids = []
        for entry in entries or []:
            v = entry.config_value
            if isinstance(v, dict) and "job_id" in v and not v.get("deleted"):
                listed_ids.append(v["job_id"])

        assert job_id in listed_ids

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
@skip_if_no_lm
@pytest.mark.local_only
class TestMemoryManagementRealMem0:
    """Memory management with real Mem0/Vespa.

    Covers the add → search and add → clear paths through the runtime
    integration conftest's memory_manager, which shares one Vespa container
    and one agent_memories_test_unit schema with the rest of the session.
    """

    def test_add_then_search_round_trip(self, memory_manager):
        """Semantic search returns exactly the memories Mem0 persisted for the
        namespace, each with the exact stored text."""
        mm = memory_manager
        agent_name = "_test_tenant_ext_rt"

        # The agent_memories_test_unit schema is shared for the whole session,
        # so start from a namespace holding only what this test writes.
        mm.clear_agent_memory(tenant_id="test:unit", agent_name=agent_name)

        memory_id = mm.add_memory(
            content="I always prefer using ColPali model for video searches",
            tenant_id="test:unit",
            agent_name=agent_name,
        )
        assert memory_id, (
            "Mem0 persisted nothing for the content, so there is no memory to "
            "search back"
        )

        # Vespa indexes a feed asynchronously: the write is not necessarily
        # searchable the instant add_memory returns, and how far it lags
        # depends on what else is feeding the shared container. Poll to a
        # deadline instead of racing it.
        stored: dict[str, str] = {}
        found: dict[str, str] = {}
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            stored = {
                m["id"]: m["memory"]
                for m in mm.get_all_memories(
                    tenant_id="test:unit",
                    agent_name=agent_name,
                )
            }
            found = {
                r["id"]: r["memory"]
                for r in mm.search_memory(
                    query="ColPali video search preference",
                    tenant_id="test:unit",
                    agent_name=agent_name,
                    top_k=20,
                )
            }
            if stored and found == stored:
                break
            time.sleep(1)

        assert memory_id in stored, (
            f"add_memory returned {memory_id!r} but the namespace holds "
            f"{sorted(stored)}"
        )
        assert found == stored, (
            f"Vespa semantic search returned {found!r}, expected every stored "
            f"memory {stored!r}"
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
@skip_if_no_lm
@pytest.mark.local_only
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

        # dspy.configure is owned by whichever async task calls it first —
        # inside an async test body, scope the LM with dspy.context instead.
        router = DSPyAdvancedRoutingModule()
        with dspy.context(lm=lm):
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


@pytest.mark.integration
class TestTenantRoutesRealVespaRoundTrip:
    """Bare-tenant PUT->GET and create->list through the REAL tenant routes
    against a real Vespa ConfigStore. The routes canonicalize the tenant id
    before keying the store; a read that used the raw path param would miss
    every value the write just stored. This drives the real routes (not the
    store directly) with a bare, non-canonical tenant id — the true-boundary
    proof of the canonical-key contract."""

    @pytest.fixture
    def route_app(self, tenant_config_manager):
        from fastapi import FastAPI

        from cogniverse_runtime.config_loader import (
            WorkflowSettings,
            get_workflow_settings,
        )
        from cogniverse_runtime.routers import tenant as tenant_router

        saved_cm = tenant_router._config_manager
        tenant_router.set_config_manager(tenant_config_manager)
        # No Argo configured, so create_job just persists to the config store.
        get_workflow_settings._instance = WorkflowSettings(api_url=None)
        app = FastAPI()
        app.include_router(tenant_router.router)
        try:
            yield app
        finally:
            tenant_router._config_manager = saved_cm
            if hasattr(get_workflow_settings, "_instance"):
                del get_workflow_settings._instance

    @pytest.mark.asyncio
    async def test_instructions_round_trip_bare_tenant(self, route_app):
        import httpx

        bare = "roundtrip_org"  # bare, non-canonical form
        transport = httpx.ASGITransport(app=route_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime"
        ) as client:
            put = await client.put(
                f"/{bare}/instructions",
                json={"text": "always cite the source video"},
            )
            assert put.status_code == 200
            got = await client.get(f"/{bare}/instructions")
            assert got.status_code == 200
            assert got.json()["text"] == "always cite the source video"

    @pytest.mark.asyncio
    async def test_jobs_round_trip_bare_tenant(self, route_app):
        import httpx

        bare = "roundtrip_jobs_org"
        transport = httpx.ASGITransport(app=route_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime"
        ) as client:
            created = await client.post(
                f"/{bare}/jobs",
                json={
                    "name": "nightly summary",
                    "schedule": "0 2 * * *",
                    "query": "summarize new videos",
                    "post_actions": [],
                },
            )
            assert created.status_code == 200
            job_id = created.json()["job_id"]

            listed = await client.get(f"/{bare}/jobs")
            assert listed.status_code == 200
            assert job_id in [j["job_id"] for j in listed.json()["jobs"]]
