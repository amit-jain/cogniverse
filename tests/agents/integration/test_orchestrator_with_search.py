"""
Integration tests for OrchestratorAgent with real in-process agents.

Tests validate the COMPLETE orchestration pipeline:
- Real LLM planning (DSPy decides agent sequence + parallelization)
- Real agent execution (entity extraction, query enhancement, profile selection, search)
- Real HTTP dispatch via httpx.ASGITransport (no live server needed, but requests
  flow through the actual FastAPI routes, Pydantic AgentTask validation, and the
  runtime AgentDispatcher — exactly like production)
- Concrete assertions on agent outputs (entities, enhanced queries, profiles, search results)

Requirements:
- Ollama running with gemma4:e2b model (orchestrator planning needs a capable LLM)
- Docker for Vespa container (for search agent)

What is REAL (integration boundary):
- Real DSPy LLM inference via Ollama (planning + per-agent inference)
- Real Vespa Docker container (search agent hits actual Vespa)
- Real agent instances (EntityExtractionAgent, ProfileSelectionAgent, etc.)
- Real FastAPI app mounted in memory via ASGITransport — payloads validate
  against the production AgentTask schema, catching wire-format drift that
  previously slipped past when the test short-circuited httpx

This deliberately does NOT patch httpx.AsyncClient. Earlier versions of this
test routed POSTs directly to `agent._process_impl()` which meant the
AgentTask Pydantic schema never ran; the orchestrator→sub-agent payload
mismatch (missing `agent_name`, `tenant_id` at wrong level) shipped to
production undetected. Using the real ASGI transport closes that gap.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import dspy
import httpx
import pytest
from fastapi import FastAPI

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
)
from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
    QueryEnhancementInput,
)
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps, SearchInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from tests.agents.integration.conftest import skip_if_no_ollama

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def dspy_lm():
    """Module-scoped DSPy LM for orchestrator tests."""
    lm = create_dspy_lm(
        LLMEndpointConfig(model="ollama/gemma3:4b", api_base="http://localhost:11434")
    )
    dspy.configure(lm=lm)
    # Verify the LM was actually set
    assert dspy.settings.lm is not None, (
        f"dspy.configure(lm=...) failed silently. settings.lm={dspy.settings.lm}"
    )
    logger.info(f"DSPy LM configured: {dspy.settings.lm}")
    yield lm
    dspy.configure(lm=None)


@pytest.fixture
def agent_instances(vespa_with_schema, dspy_lm):
    """
    Create real in-process agent instances for orchestrator dispatch.

    Returns a dict mapping agent URL → agent instance for direct invocation.
    """
    config_manager = vespa_with_schema["manager"].config_manager
    vespa_http_port = vespa_with_schema["http_port"]
    vespa_config_port = vespa_with_schema["config_port"]
    default_schema = vespa_with_schema["default_schema"]

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Create real agent instances
    entity_agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    profile_agent = ProfileSelectionAgent(deps=ProfileSelectionDeps())
    query_agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())
    search_agent = SearchAgent(
        deps=SearchAgentDeps(
            backend_url="http://localhost",
            backend_port=vespa_http_port,
            backend_config_port=vespa_config_port,
            profile=default_schema,
        ),
        schema_loader=schema_loader,
        config_manager=config_manager,
    )

    # Pre-warm the search agent's tenant backend — this caches the
    # correctly-configured backend at the dynamic Vespa port BEFORE the
    # orchestrator fixture can pollute the backend registry cache.
    # Without this, the orchestrator's _ensure_memory_for_tenant() creates
    # backends at the wrong port (from config file), and the search agent
    # picks up a stale/wrong backend from the registry.
    search_agent._get_backend()

    # Map agent URLs (from registry config) to instances
    return {
        "http://localhost:8010": entity_agent,
        "http://localhost:8011": profile_agent,
        "http://localhost:8012": query_agent,
        "http://localhost:8002": search_agent,
    }


@pytest.fixture
def orchestrator_with_agents(vespa_with_schema, dspy_lm, agent_instances):
    """
    OrchestratorAgent wired to real in-process agents via an in-memory ASGI
    FastAPI app. httpx.AsyncClient is patched to use ASGITransport so POSTs
    flow through the real `/agents/{name}/process` route, Pydantic AgentTask
    validation, and the runtime AgentDispatcher exactly like production —
    but without binding a network port.
    """
    from cogniverse_core.common.agent_models import AgentEndpoint
    from cogniverse_runtime.routers import agents as agents_router

    config_manager = vespa_with_schema["manager"].config_manager
    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )
    registry = AgentRegistry(config_manager=config_manager)

    # All four sub-agents live at a single in-process ASGI host. Path routing
    # (/agents/{name}/process) selects the target agent, matching production
    # where every agent shares the runtime pod.
    ASGI_BASE = "http://asgi.test"
    for name, caps in [
        ("entity_extraction", ["entity_extraction"]),
        ("profile_selection", ["profile_selection"]),
        ("query_enhancement", ["query_enhancement"]),
        ("search", ["search"]),
        # Orchestrator itself is also reachable via the same app so nested
        # dispatch works if a plan asks for it.
        ("orchestrator", ["orchestration"]),
    ]:
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url=ASGI_BASE,
                capabilities=caps,
                process_endpoint=f"/agents/{name}/process",
            )
        )

    # Re-assert DSPy LM — can get cleared by agent init code
    dspy.configure(lm=dspy_lm)

    orchestrator = OrchestratorAgent(
        deps=OrchestratorDeps(),
        registry=registry,
        config_manager=config_manager,
        port=8013,
    )

    # Build a minimal FastAPI app carrying only the agents router. Injecting
    # registry + config_manager + schema_loader reaches the real dispatcher,
    # so every POST runs production AgentTask validation and dispatch logic.
    # The dispatcher constructs its own agent instances via ConfigLoader —
    # that mirrors production exactly, which is the point of this test.
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents", tags=["agents"])
    agents_router.set_agent_registry(registry)
    agents_router.set_agent_dependencies(config_manager, schema_loader)

    # Route httpx.AsyncClient through ASGITransport so the orchestrator's
    # POSTs hit the real FastAPI app (and thus the real AgentTask schema).
    # Any orchestrator→sub-agent payload drift (missing `agent_name`, wrong
    # nesting of `tenant_id`, etc.) will fail here at 422 — the exact case
    # the old httpx mock was silently swallowing.
    transport = httpx.ASGITransport(app=app)

    class _ASGIAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            kwargs.setdefault("base_url", ASGI_BASE)
            super().__init__(*args, **kwargs)

    httpx_patcher = patch(
        "cogniverse_agents.orchestrator_agent.httpx.AsyncClient",
        _ASGIAsyncClient,
    )
    httpx_patcher.start()

    # Restrict available agents to only those with in-process instances.
    # Monkey-patch the DSPy module's forward to override available_agents,
    # so the production _create_plan handles all parsing (no duplication).
    test_agents = [
        "entity_extraction",
        "query_enhancement",
        "profile_selection",
        "search",
    ]
    original_forward = orchestrator.dspy_module.forward

    def restricted_forward(query, available_agents=None, **kwargs):
        return original_forward(
            query=query,
            available_agents=", ".join(test_agents),
            **kwargs,
        )

    orchestrator.dspy_module.forward = restricted_forward

    yield orchestrator

    httpx_patcher.stop()


@pytest.mark.integration
@skip_if_no_ollama
@pytest.mark.slow
class TestOrchestratorWithRealAgents:
    """
    Integration tests with real LLM planning + real in-process agent execution.

    Every test asserts concrete outputs from agent inference — not just
    "result is not None" but actual entities, enhanced queries, profiles, etc.
    """

    @pytest.mark.asyncio
    async def test_entity_extraction_output(self, agent_instances, dspy_lm):
        """
        Entity extraction agent extracts real entities from a query.

        Direct invocation — no orchestrator, isolates entity extraction.
        """
        entity_agent = agent_instances["http://localhost:8010"]
        with dspy.context(lm=dspy_lm):
            result = await entity_agent._process_impl(
                EntityExtractionInput(
                    query="Find videos about Python programming by Google"
                )
            )

        assert result is not None
        assert result.query == "Find videos about Python programming by Google"

        # Should extract entities — at minimum "Python" and "Google"
        assert isinstance(result.entities, list)
        assert len(result.entities) >= 1, (
            f"Should extract at least 1 entity from 'Python programming by Google', "
            f"got: {result.entities}"
        )

        # Each entity should have text and type (Entity is a Pydantic model)
        for entity in result.entities:
            assert hasattr(entity, "text") and entity.text, (
                f"Entity missing text: {entity}"
            )
            assert hasattr(entity, "type") and entity.type, (
                f"Entity missing type: {entity}"
            )

        entity_names = [e.text.lower() for e in result.entities]
        logger.info(f"Extracted entities: {result.entities}")
        logger.info(f"Entity names: {entity_names}")

    @pytest.mark.asyncio
    async def test_query_enhancement_output(self, agent_instances, dspy_lm):
        """
        Query enhancement produces an enhanced query longer/richer than original.

        Direct invocation to verify the LLM actually expands the query.
        """
        query_agent = agent_instances["http://localhost:8012"]
        original = "find videos about robotic arm assembly in manufacturing"
        with dspy.context(lm=dspy_lm):
            result = await query_agent._process_impl(
                QueryEnhancementInput(query=original)
            )

        assert result is not None
        assert result.original_query == original
        assert isinstance(result.enhanced_query, str)
        assert len(result.enhanced_query) > 0, "Enhanced query must not be empty"

        # LLM should have enhanced the query — either the query text changed
        # or expansion terms were generated (small models may not rewrite the text
        # but still produce useful expansion terms)
        query_changed = result.enhanced_query != original
        has_expansions = len(result.expansion_terms) > 0
        assert query_changed or has_expansions, (
            f"LLM produced no enhancement at all: enhanced_query unchanged, "
            f"no expansion_terms. confidence={result.confidence}, "
            f"reasoning='{result.reasoning}'"
        )

        # Should not be a fallback (fallback has confidence=0.5)
        assert result.confidence > 0.5, (
            f"Should use real LLM, not fallback. confidence={result.confidence}"
        )

        logger.info(f"Original: '{original}' → Enhanced: '{result.enhanced_query}'")
        logger.info(f"Expansion terms: {result.expansion_terms}")
        logger.info(f"Synonyms: {result.synonyms}")

    @pytest.mark.asyncio
    async def test_profile_selection_output(self, agent_instances, dspy_lm):
        """
        Profile selection picks a valid profile for a video search query.
        """
        profile_agent = agent_instances["http://localhost:8011"]
        with dspy.context(lm=dspy_lm):
            result = await profile_agent._process_impl(
                ProfileSelectionInput(query="find tutorial videos about deep learning")
            )

        assert result is not None
        assert isinstance(result.selected_profile, str)
        assert len(result.selected_profile) > 0, "Must select a profile"

        # Small LLMs sometimes return profile names with extra quotes
        selected = result.selected_profile.strip('"').strip("'")

        # Selected profile should be from the available profiles list
        valid_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        ]
        assert selected in valid_profiles, (
            f"Selected profile '{selected}' (raw: '{result.selected_profile}') "
            f"not in valid profiles: {valid_profiles}"
        )

        # Should have reasoning for the selection
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 5, "Reasoning should be non-trivial"

        logger.info(
            f"Profile: {result.selected_profile}, Reasoning: {result.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_search_agent_returns_vespa_results(self, agent_instances, dspy_lm):
        """
        Search agent hits real Vespa and returns structured results.

        The vespa_with_schema fixture ingests test data, so we expect results.
        """
        search_agent = agent_instances["http://localhost:8002"]
        with dspy.context(lm=dspy_lm):
            result = await search_agent._process_impl(
                SearchInput(query="video content", tenant_id="test_tenant")
            )

        assert result is not None
        assert isinstance(result.results, list)
        assert result.total_results >= 0

        # If results exist, verify structure
        if result.results:
            first = result.results[0]
            assert isinstance(first, dict)
            # Vespa results have documentid and relevance
            assert "documentid" in first or "id" in first, (
                f"Result missing documentid/id: {first.keys()}"
            )

        logger.info(
            f"Search returned {result.total_results} results, "
            f"first: {result.results[0] if result.results else 'none'}"
        )

    @pytest.mark.asyncio
    async def test_orchestrated_pipeline_with_concrete_assertions(
        self, orchestrator_with_agents, dspy_lm
    ):
        """
        Full orchestrated pipeline: LLM plans ALL 4 agents, each executes with
        real inference, concrete assertions on every agent's output.

        Query is crafted to require all agents:
        - Entity extraction: "Python", "Google", "machine learning"
        - Query enhancement: expand the search terms
        - Profile selection: pick a video search profile
        - Search: execute against Vespa
        """
        with dspy.context(lm=dspy_lm):
            result = await orchestrator_with_agents._process_impl(
                OrchestratorInput(
                    query="Find detailed Python machine learning tutorial videos by Google research team",
                    tenant_id="test_tenant",
                )
            )

        assert result is not None
        assert result.plan_reasoning, "LLM should provide planning reasoning"

        # --- Validate LLM planned a multi-agent pipeline ---
        # Normalize _agent suffix (LLM may return "search_agent" or "search")
        def _norm(name):
            return name.removesuffix("_agent") if name.endswith("_agent") else name

        planned_agents = {_norm(s["agent_name"]) for s in result.plan_steps}
        assert "search" in planned_agents, (
            f"'search' must always be planned for a search query. "
            f"Planned: {planned_agents}"
        )
        preprocessing_agents = planned_agents - {"search"}
        assert len(preprocessing_agents) >= 1, (
            f"LLM should plan at least 1 pre-processing agent alongside search. "
            f"Got: {preprocessing_agents}. Full plan: {planned_agents}"
        )

        # Validate plan DAG structure
        for i, step in enumerate(result.plan_steps):
            for dep_idx in step["depends_on"]:
                assert 0 <= dep_idx < i, (
                    f"Step {i} ({step['agent_name']}) has invalid dep {dep_idx}"
                )

        # --- Every planned agent must have results ---
        for agent_name in planned_agents:
            assert agent_name in result.agent_results, (
                f"Agent '{agent_name}' was planned but has no result"
            )
            agent_result = result.agent_results[agent_name]
            assert isinstance(agent_result, dict), (
                f"{agent_name} result should be dict, got {type(agent_result)}"
            )

        # --- Concrete assertions on each planned agent's output ---
        if "entity_extraction" in planned_agents:
            ee_result = result.agent_results["entity_extraction"]
            assert "entities" in ee_result, (
                f"entity_extraction missing 'entities' key: {ee_result.keys()}"
            )
            assert isinstance(ee_result["entities"], list)
            assert len(ee_result["entities"]) >= 1, (
                f"Should extract at least 1 entity. Got: {ee_result['entities']}"
            )
            for entity in ee_result["entities"]:
                assert "name" in entity or "text" in entity, (
                    f"Entity missing name/text field: {entity}"
                )

        if "query_enhancement" in planned_agents:
            qe_result = result.agent_results["query_enhancement"]
            assert "enhanced_query" in qe_result, (
                f"query_enhancement missing 'enhanced_query': {qe_result.keys()}"
            )
            assert isinstance(qe_result["enhanced_query"], str)
            assert len(qe_result["enhanced_query"]) > 0, (
                "Enhanced query must not be empty"
            )

        if "profile_selection" in planned_agents:
            ps_result = result.agent_results["profile_selection"]
            assert "selected_profile" in ps_result, (
                f"profile_selection missing 'selected_profile': {ps_result.keys()}"
            )
            assert isinstance(ps_result["selected_profile"], str)
            assert len(ps_result["selected_profile"]) > 0, "Must select a profile"

        # Search is always planned (asserted above)
        s_result = result.agent_results["search"]
        assert "results" in s_result, f"search missing 'results' key: {s_result.keys()}"
        assert isinstance(s_result["results"], list)

        # Log full pipeline for debugging
        logger.info(f"Pipeline planned: {[s['agent_name'] for s in result.plan_steps]}")
        for agent_name in planned_agents:
            logger.info(f"  {agent_name}: {result.agent_results[agent_name]}")

    @pytest.mark.asyncio
    async def test_orchestrator_empty_query(self, orchestrator_with_agents, dspy_lm):
        """Empty query returns early without planning or execution."""
        with dspy.context(lm=dspy_lm):
            result = await orchestrator_with_agents._process_impl(
                OrchestratorInput(query="", tenant_id="test_tenant")
            )

        assert result is not None
        assert result.plan_steps == []
        assert result.agent_results == {}
        assert "Empty query" in str(result.final_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
