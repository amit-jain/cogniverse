"""
Unit tests for OrchestratorAgent A2A pipeline.

Validates:
- OrchestratorAgent uses AgentRegistry for discovery
- A2A task flow: plan → execute (via A2AClient) → aggregate
- tenant_id and session_id flow through to each agent call
- Parallel execution groups respect dependencies
- Graceful error handling when agents are unreachable
"""

from unittest.mock import AsyncMock, Mock, patch

import dspy
import pytest

from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
    OrchestratorOutput,
)


@pytest.fixture
def mock_registry():
    """Mock AgentRegistry with 4 agents."""
    registry = Mock()

    agent_endpoints = {}
    for name, port in [
        ("entity_extraction", 8010),
        ("profile_selection", 8011),
        ("query_enhancement", 8012),
        ("search", 8002),
    ]:
        endpoint = Mock()
        endpoint.name = name
        endpoint.url = f"http://localhost:{port}"
        endpoint.capabilities = [name]
        agent_endpoints[name] = endpoint

    registry.get_agent = Mock(side_effect=lambda n: agent_endpoints.get(n))
    registry.find_agents_by_capability = Mock(
        side_effect=lambda cap: [
            ep for ep in agent_endpoints.values() if cap in ep.capabilities
        ]
    )
    registry.list_agents = Mock(return_value=list(agent_endpoints.keys()))
    registry.agents = agent_endpoints
    return registry


@pytest.fixture
def orchestrator(mock_registry):
    """OrchestratorAgent with mock registry."""
    with patch("dspy.ChainOfThought"):
        deps = OrchestratorDeps()
        mock_config_manager = Mock()
        return OrchestratorAgent(
            deps=deps,
            registry=mock_registry,
            config_manager=mock_config_manager,
            port=8013,
        )


@pytest.mark.unit
class TestA2APipelineFlow:
    """Test the full A2A pipeline through OrchestratorAgent."""

    @pytest.mark.asyncio
    async def test_plan_execute_aggregate(self, orchestrator):
        """Full pipeline: plan → A2A execute → aggregate results."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,search",
                parallel_steps="",
                reasoning="Enhance then search",
            )
        )

        # Mock A2A client to return structured results
        call_count = {"n": 0}

        async def mock_send_task(url, **kwargs):
            call_count["n"] += 1
            if "8012" in url:
                return {"status": "success", "enhanced_query": "enhanced ML videos"}
            elif "8002" in url:
                return {"status": "success", "results": [{"title": "ML Tutorial"}]}
            return {"status": "error", "message": "Unknown agent"}

        orchestrator.a2a_client.send_task = AsyncMock(side_effect=mock_send_task)

        result = await orchestrator._process_impl(OrchestratorInput(query="ML videos"))

        assert isinstance(result, OrchestratorOutput)
        assert len(result.plan_steps) == 2
        assert "query_enhancement" in result.agent_results
        assert "search" in result.agent_results
        assert result.final_output["status"] == "success"

    @pytest.mark.asyncio
    async def test_tenant_id_flows_through(self, orchestrator):
        """tenant_id from input is passed to each A2A call."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Direct search",
            )
        )

        captured_kwargs = {}

        async def capture_send_task(url, **kwargs):
            captured_kwargs.update(kwargs)
            return {"status": "success", "results": []}

        orchestrator.a2a_client.send_task = AsyncMock(side_effect=capture_send_task)

        await orchestrator._process_impl(
            OrchestratorInput(
                query="test", tenant_id="acme_corp", session_id="sess-123"
            )
        )

        # Verify tenant_id and session_id propagated
        assert captured_kwargs.get("tenant_id") == "acme_corp"
        assert captured_kwargs.get("session_id") == "sess-123"


@pytest.mark.unit
class TestRegistryDiscovery:
    """Test agent discovery via AgentRegistry."""

    @pytest.mark.asyncio
    async def test_discovers_agent_by_capability(self, orchestrator, mock_registry):
        """Orchestrator finds agents via registry.find_agents_by_capability."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction",
                parallel_steps="",
                reasoning="Extract entities",
            )
        )
        orchestrator.a2a_client.send_task = AsyncMock(
            return_value={"status": "success", "entities": ["ML"]}
        )

        await orchestrator._process_impl(OrchestratorInput(query="machine learning"))

        # Verify registry was consulted
        mock_registry.get_agent.assert_called()

    @pytest.mark.asyncio
    async def test_unavailable_agent_returns_error(self, orchestrator, mock_registry):
        """Missing agent in registry returns structured error."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="summarizer",
                parallel_steps="",
                reasoning="Summarize results",
            )
        )

        # summarizer is NOT in the mock registry
        result = await orchestrator._process_impl(
            OrchestratorInput(query="summarize this")
        )

        assert "summarizer" in result.agent_results
        assert result.agent_results["summarizer"]["status"] == "error"
        assert "not available" in result.agent_results["summarizer"]["message"]


@pytest.mark.unit
class TestParallelExecution:
    """Test parallel execution groups in the pipeline."""

    @pytest.mark.asyncio
    async def test_parallel_group_structure(self, orchestrator):
        """Parallel groups generate correct dependencies."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search",
                parallel_steps="0,1",
                reasoning="Extract and enhance in parallel, then select profile, then search",
            )
        )
        orchestrator.a2a_client.send_task = AsyncMock(
            return_value={"status": "success"}
        )

        result = await orchestrator._process_impl(OrchestratorInput(query="ML videos"))

        # Steps 0,1 parallel (no deps), step 2 depends on 0,1, step 3 depends on 2
        assert result.plan_steps[0]["depends_on"] == []
        assert result.plan_steps[1]["depends_on"] == []
        assert set(result.plan_steps[2]["depends_on"]) == {0, 1}
        assert result.plan_steps[3]["depends_on"] == [2]

    @pytest.mark.asyncio
    async def test_all_agents_executed(self, orchestrator):
        """All 4 agents in pipeline get executed."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search",
                parallel_steps="0,1",
                reasoning="Full pipeline",
            )
        )
        orchestrator.a2a_client.send_task = AsyncMock(
            return_value={"status": "success", "result": "mock"}
        )

        result = await orchestrator._process_impl(OrchestratorInput(query="test query"))

        assert len(result.agent_results) == 4
        assert all(
            name in result.agent_results
            for name in [
                "entity_extraction",
                "query_enhancement",
                "profile_selection",
                "search",
            ]
        )


@pytest.mark.unit
class TestErrorHandling:
    """Test graceful error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_agent_exception_captured(self, orchestrator):
        """A2A call exception is captured as error result."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Search",
            )
        )
        orchestrator.a2a_client.send_task = AsyncMock(
            side_effect=ConnectionError("Agent unreachable")
        )

        result = await orchestrator._process_impl(OrchestratorInput(query="test"))

        assert result.agent_results["search"]["status"] == "error"
        assert "Agent unreachable" in result.agent_results["search"]["message"]

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, orchestrator):
        """Empty query returns error without executing pipeline."""
        result = await orchestrator._process_impl(OrchestratorInput(query=""))

        assert result.final_output["status"] == "error"
        assert "Empty query" in result.final_output["message"]
        assert len(result.plan_steps) == 0
