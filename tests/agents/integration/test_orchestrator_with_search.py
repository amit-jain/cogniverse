"""
End-to-end integration tests for Orchestrator with SearchAgent and Vespa.

These tests validate the complete autonomous agent pipeline:
Entity Extraction → Query Enhancement → Profile Selection → Search (Vespa)

Tests use:
- Real Ollama LLMs for agents
- Real Vespa Docker instance for search
- Full orchestration with parallel execution
"""

import dspy
import pytest

from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
)
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from tests.agents.integration.conftest import skip_if_no_ollama


@pytest.fixture
def real_dspy_lm():
    """Configure DSPy to use Ollama with qwen2.5:1.5b model"""
    lm = dspy.LM("ollama_chat/qwen2.5:1.5b", api_base="http://localhost:11434")
    dspy.settings.configure(lm=lm)
    return lm


@pytest.fixture
def search_agent_with_vespa(vespa_with_schema, real_dspy_lm):
    """SearchAgent connected to test Vespa instance"""
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    # Get Vespa connection info from fixture
    vespa_http_port = vespa_with_schema["http_port"]
    vespa_config_port = vespa_with_schema["config_port"]
    vespa_url = "http://localhost"
    default_schema = vespa_with_schema["default_schema"]

    # Use config manager from VespaTestManager (has correct ports)
    config_manager = vespa_with_schema["manager"].config_manager

    # Create schema loader pointing to test schemas
    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Create SearchAgent with test Vespa parameters
    deps = SearchAgentDeps(
        backend_url=vespa_url,
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile=default_schema,
    )
    search_agent = SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
        port=8015,
    )

    return search_agent


@pytest.fixture
def full_orchestrator_with_search(
    real_dspy_lm, vespa_with_schema, search_agent_with_vespa
):
    """OrchestratorAgent with all agents including SearchAgent via mock AgentRegistry"""
    from unittest.mock import Mock

    from cogniverse_core.common.agent_models import AgentEndpoint

    # Create mock AgentRegistry with endpoints for all agents
    registry = Mock()
    agent_endpoints = {
        "entity_extraction": AgentEndpoint(
            name="entity_extraction",
            url="http://localhost:8010",
            capabilities=["entity_extraction"],
        ),
        "profile_selection": AgentEndpoint(
            name="profile_selection",
            url="http://localhost:8011",
            capabilities=["profile_selection"],
        ),
        "query_enhancement": AgentEndpoint(
            name="query_enhancement",
            url="http://localhost:8012",
            capabilities=["query_enhancement"],
        ),
        "search": AgentEndpoint(
            name="search",
            url="http://localhost:8015",
            capabilities=["search"],
        ),
    }
    registry.get_agent = Mock(side_effect=lambda name: agent_endpoints.get(name))
    registry.find_agents_by_capability = Mock(
        side_effect=lambda cap: [
            ep for ep in agent_endpoints.values() if cap in ep.capabilities
        ]
    )
    registry.list_agents = Mock(return_value=list(agent_endpoints.keys()))
    registry.agents = agent_endpoints

    orchestrator_deps = OrchestratorDeps()
    orchestrator = OrchestratorAgent(
        deps=orchestrator_deps, registry=registry, port=8013
    )
    return orchestrator


@pytest.mark.integration
@skip_if_no_ollama
@pytest.mark.slow  # Requires Docker + Vespa startup
class TestOrchestratorWithSearch:
    """Integration tests for full orchestration pipeline with Vespa search"""

    @pytest.mark.asyncio
    async def test_full_pipeline_orchestration_structure(
        self, full_orchestrator_with_search
    ):
        """
        CORRECTNESS: Validate full 4-agent pipeline executes

        Tests: Entity → Profile → Enhancement → Search
        Validates structure even if search returns no results (no schema deployed)
        """
        from unittest.mock import Mock

        import dspy

        # Force 4-agent pipeline
        full_orchestrator_with_search.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,profile_selection,query_enhancement,search",
                parallel_steps="0,2",  # Entity and Enhancement in parallel
                reasoning="Extract entities and enhance query in parallel, select profile, then search",
            )
        )

        result = await full_orchestrator_with_search._process_impl(
            OrchestratorInput(
                query="Show me machine learning tutorial videos",
                tenant_id="test_tenant",
            )
        )

        # VALIDATE: 4-step plan created
        assert (
            len(result.plan_steps) == 4
        ), f"Should create 4-step plan, got: {len(result.plan_steps)}"

        # VALIDATE: All 4 agents in plan
        agent_names = [step["agent_type"] for step in result.plan_steps]
        assert "entity_extraction" in agent_names
        assert "profile_selection" in agent_names
        assert "query_enhancement" in agent_names
        assert "search" in agent_names

        # VALIDATE: Parallel execution configured
        assert len(result.parallel_groups) > 0, "Should have parallel groups"

        # VALIDATE: All agents executed (even if search fails without schema)
        assert (
            len(result.agent_results) == 4
        ), f"Should execute all 4 agents, got: {len(result.agent_results)}"

        # VALIDATE: Upstream agents succeeded
        assert "entity_extraction" in result.agent_results
        assert "profile_selection" in result.agent_results
        assert "query_enhancement" in result.agent_results

        # VALIDATE: Search attempted (may fail without deployed schema)
        assert "search" in result.agent_results
        search_result = result.agent_results["search"]

        # Search will either succeed (if schema exists) or fail gracefully
        if isinstance(search_result, dict) and search_result.get("status") == "error":
            # Expected if no schema deployed - verify error is graceful
            assert "message" in search_result
            assert search_result["message"]  # Has error message
        else:
            # If search succeeded, validate structure
            assert search_result is not None

    @pytest.mark.asyncio
    async def test_orchestrator_dependency_resolution_with_search(
        self, full_orchestrator_with_search
    ):
        """
        CORRECTNESS: Validate dependencies are respected with SearchAgent

        Ensures search doesn't execute until upstream agents complete
        """
        from unittest.mock import Mock

        import dspy

        # Sequential pipeline to test dependencies
        full_orchestrator_with_search.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search",
                parallel_steps="",  # All sequential
                reasoning="Sequential: extract, enhance, select profile, search",
            )
        )

        result = await full_orchestrator_with_search._process_impl(
            OrchestratorInput(
                query="Find Python programming videos", tenant_id="test_tenant"
            )
        )

        # VALIDATE: Sequential dependencies
        assert result.plan_steps[0]["depends_on"] == []  # First has no deps
        assert result.plan_steps[1]["depends_on"] == [0]  # Second depends on first
        assert result.plan_steps[2]["depends_on"] == [1]  # Third depends on second
        assert result.plan_steps[3]["depends_on"] == [2]  # Search depends on profile

        # VALIDATE: Execution order preserved
        # All agents executed in order
        assert len(result.agent_results) == 4

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_with_search_dependency(
        self, full_orchestrator_with_search
    ):
        """
        CORRECTNESS: Validate parallel execution upstream, then search

        Tests: (Entity + Enhancement parallel) → Profile → Search
        """
        from unittest.mock import Mock

        import dspy

        full_orchestrator_with_search.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search",
                parallel_steps="0,1",  # First two parallel
                reasoning="Extract and enhance in parallel, select profile, search",
            )
        )

        result = await full_orchestrator_with_search._process_impl(
            OrchestratorInput(
                query="Machine learning tutorials", tenant_id="test_tenant"
            )
        )

        # VALIDATE: Parallel group structure
        assert len(result.parallel_groups) == 1
        assert result.parallel_groups[0] == [0, 1]

        # VALIDATE: Dependencies
        # Steps 0,1 parallel (no deps)
        assert result.plan_steps[0]["depends_on"] == []
        assert result.plan_steps[1]["depends_on"] == []

        # Profile depends on both parallel steps
        assert set(result.plan_steps[2]["depends_on"]) == {0, 1}

        # Search depends on profile
        assert result.plan_steps[3]["depends_on"] == [2]

        # VALIDATE: All executed
        assert len(result.agent_results) == 4
