"""
Unit tests for OrchestratorAgent pipeline.

Validates:
- OrchestratorAgent uses AgentRegistry for discovery
- Task flow: plan → execute (via httpx) → aggregate
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


def _make_httpx_mock(response_factory):
    """Create an httpx client mock compatible with the orchestrator's
    loop-pooled client (``_get_http_client``).

    Args:
        response_factory: callable(url, json) -> dict to return as JSON response

    Returns:
        (patch_target, mock_client) — apply with
        ``patch("cogniverse_agents.orchestrator_agent._get_http_client",
                new=AsyncMock(return_value=mock_client))``.
    """

    async def mock_post(url, json=None, **kwargs):
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.return_value = response_factory(url, json)
        return resp

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.is_closed = False
    return mock_client


@pytest.mark.unit
class TestA2APipelineFlow:
    """Test the full pipeline through OrchestratorAgent."""

    @pytest.mark.asyncio
    async def test_plan_execute_aggregate(self, orchestrator):
        """Full pipeline: plan → execute → aggregate results."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,search",
                parallel_steps="",
                reasoning="Enhance then search",
            )
        )

        def response_factory(url, json_body):
            if "8012" in url:
                return {"status": "success", "enhanced_query": "enhanced ML videos"}
            elif "8002" in url:
                return {"status": "success", "results": [{"title": "ML Tutorial"}]}
            return {"status": "error", "message": "Unknown agent"}

        mock_client = _make_httpx_mock(response_factory)

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            result = await orchestrator._process_impl(
                OrchestratorInput(query="ML videos")
            )

        assert isinstance(result, OrchestratorOutput)
        assert len(result.plan_steps) == 2
        assert "query_enhancement" in result.agent_results
        assert "search" in result.agent_results
        assert result.final_output["status"] == "success"

    @pytest.mark.asyncio
    async def test_tenant_id_flows_through(self, orchestrator):
        """tenant_id from input is passed to each HTTP call."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Direct search",
            )
        )

        captured_bodies = []

        def response_factory(url, json_body):
            captured_bodies.append(json_body)
            return {"status": "success", "results": []}

        mock_client = _make_httpx_mock(response_factory)

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator._process_impl(
                OrchestratorInput(
                    query="test", tenant_id="acme_corp", session_id="sess-123"
                )
            )

        # Verify tenant_id and session_id propagated in the request body
        # (the orchestrator nests them under body["context"]).
        assert len(captured_bodies) >= 1
        ctx = captured_bodies[0].get("context", {})
        assert ctx.get("tenant_id") == "acme_corp"
        assert ctx.get("session_id") == "sess-123"


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

        mock_client = _make_httpx_mock(
            lambda url, json: {"status": "success", "entities": ["ML"]}
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator._process_impl(
                OrchestratorInput(query="machine learning")
            )

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

        mock_client = _make_httpx_mock(lambda url, json: {"status": "success"})

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            result = await orchestrator._process_impl(
                OrchestratorInput(query="ML videos")
            )

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

        mock_client = _make_httpx_mock(
            lambda url, json: {"status": "success", "result": "mock"}
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            result = await orchestrator._process_impl(
                OrchestratorInput(query="test query")
            )

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
        """HTTP call exception is captured as error result."""
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Search",
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("Agent unreachable"))
        mock_client.is_closed = False

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
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


@pytest.mark.unit
class TestConversationHistory:
    """Test conversation history flows through the orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_receives_conversation_history(self, orchestrator):
        """conversation_history from input is passed to the DSPy planner."""
        captured_kwargs = {}

        def mock_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Search based on context",
            )

        orchestrator.dspy_module.forward = mock_forward

        mock_client = _make_httpx_mock(lambda url, json: {"status": "success"})

        history = [
            {"role": "user", "content": "search for cat videos"},
            {"role": "agent", "content": "Found 5 results about cats"},
        ]

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator._process_impl(
                OrchestratorInput(
                    query="show me longer ones",
                    conversation_history=history,
                )
            )

        assert "conversation_context" in captured_kwargs
        ctx = captured_kwargs["conversation_context"]
        assert "cat videos" in ctx
        assert "Found 5 results" in ctx

    @pytest.mark.asyncio
    async def test_first_turn_empty_conversation_context(self, orchestrator):
        """First turn (no history) passes empty conversation_context to planner."""
        captured_kwargs = {}

        def mock_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return dspy.Prediction(
                agent_sequence="search",
                parallel_steps="",
                reasoning="Direct search",
            )

        orchestrator.dspy_module.forward = mock_forward

        mock_client = _make_httpx_mock(lambda url, json: {"status": "success"})

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator._process_impl(OrchestratorInput(query="search for dogs"))

        assert captured_kwargs["conversation_context"] == ""

    def test_format_conversation_context_truncates(self, orchestrator):
        """_format_conversation_context truncates to last 5 turns, 200 chars each."""
        history = [
            {"role": "user", "content": f"Turn {i}: {'x' * 300}"} for i in range(10)
        ]

        result = orchestrator._format_conversation_context(history)

        # Only last 5 turns
        lines = result.strip().split("\n")
        assert len(lines) == 5

        # Each content truncated to 200 chars
        for line in lines:
            # "user: Turn N: xxx..." — total line is "user: " + 200 chars max
            content_part = line.split(": ", 1)[1]
            assert len(content_part) <= 200

    def test_format_conversation_context_none(self, orchestrator):
        """None history returns empty string."""
        assert orchestrator._format_conversation_context(None) == ""

    def test_format_conversation_context_empty(self, orchestrator):
        """Empty list returns empty string."""
        assert orchestrator._format_conversation_context([]) == ""
