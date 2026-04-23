"""
Integration tests for RoutingAgent
Tests real interactions with routing system and configuration.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_agent(**dep_overrides) -> RoutingAgent:
    """Create a RoutingAgent with mocked DSPy LM."""
    defaults = {"telemetry_config": TelemetryConfig(enabled=False)}
    defaults.update(dep_overrides)
    deps = RoutingDeps(**defaults)

    def _mock_configure_dspy(self_agent, deps_arg):
        self_agent._dspy_lm = MagicMock()

    with patch.object(RoutingAgent, "_configure_dspy", _mock_configure_dspy):
        return RoutingAgent(deps=deps)


class TestRoutingAgentIntegration:
    """Integration tests for RoutingAgent with routing components."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_routing_decision_flow(self):
        """Test actual routing decision flow."""
        agent = _make_agent()

        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.85
        mock_prediction.reasoning_chain = ["Video content search"]

        test_queries = [
            "Show me videos about machine learning",
            "Summarize the latest AI research",
            "Provide detailed analysis of neural networks",
            "Find documents about deep learning",
        ]

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            for query in test_queries:
                result = await agent.route_query(query, tenant_id="test_tenant")

                assert result.query == query
                assert result.recommended_agent is not None
                assert result.confidence > 0
                assert result.reasoning is not None
                assert isinstance(result.fallback_agents, list)
                assert isinstance(result.metadata, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_routing_agent_context_propagation(self):
        """Test that context is properly propagated through routing."""
        agent = _make_agent()

        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.8
        mock_prediction.reasoning_chain = ["Search needed"]

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            result = await agent.route_query(
                "find videos",
                context="user prefers video content",
                tenant_id="test_tenant",
            )

        assert result.query == "find videos"
        assert result.recommended_agent is not None
        assert result.confidence > 0

    def test_agent_initialization(self):
        """Test agent initialization with different configurations."""
        agent = _make_agent()
        assert agent.config is not None
        assert hasattr(agent, "logger")
        assert hasattr(agent, "routing_module")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workflow_generation_consistency(self):
        """Test that routing is consistent across multiple calls."""
        agent = _make_agent()

        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.9
        mock_prediction.reasoning_chain = ["Training video search"]

        query = "Show me training videos"

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            results = []
            for _ in range(3):
                result = await agent.route_query(query, tenant_id="test_tenant")
                results.append(result)

        first_result = results[0]
        for result in results[1:]:
            assert result.query == first_result.query
            assert result.recommended_agent == first_result.recommended_agent
            assert result.confidence > 0
            assert result.reasoning is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_enriched_data_passthrough(self):
        """Test that pre-enriched data from upstream agents is used."""
        agent = _make_agent()

        forward_calls = []
        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "summarizer_agent"}
        mock_prediction.overall_confidence = 0.9
        mock_prediction.reasoning_chain = ["Summarization with entities"]

        def capture_forward(**kwargs):
            forward_calls.append(kwargs)
            return mock_prediction

        with patch.object(agent.routing_module, "forward", side_effect=capture_forward):
            result = await agent.route_query(
                query="summarize AI research",
                enhanced_query="summarize recent AI research papers from 2024",
                entities=[
                    {"text": "AI", "label": "TECH"},
                    {"text": "research", "label": "TOPIC"},
                ],
                relationships=[
                    {"subject": "AI", "relation": "part_of", "object": "research"}
                ],
                tenant_id="test_tenant",
            )

        assert result.recommended_agent == "summarizer_agent"
        assert len(forward_calls) == 1
        # Enriched query should be used
        assert "2024" in forward_calls[0]["query"]
        # Context should contain entities
        assert "AI" in forward_calls[0]["context"]
