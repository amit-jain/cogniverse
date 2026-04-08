"""
Unit tests for RoutingAgent — cache and metrics.

The gutted RoutingAgent no longer has inline caching, metrics tracking,
or production components (those features move to dedicated A2A agents
and infrastructure). These tests verify that the gutted agent works
correctly without those components.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps, RoutingOutput
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_agent(**dep_overrides) -> RoutingAgent:
    """Create a RoutingAgent with mocked DSPy LM."""
    defaults = {
        "telemetry_config": TelemetryConfig(enabled=False),
    }
    defaults.update(dep_overrides)
    deps = RoutingDeps(**defaults)

    def _mock_configure_dspy(self_agent, deps_arg):
        self_agent._dspy_lm = MagicMock()

    with patch.object(RoutingAgent, "_configure_dspy", _mock_configure_dspy):
        return RoutingAgent(deps=deps)


class TestGuttedAgentHasNoCacheOrMetrics:
    """Verify the gutted agent has no cache/metrics/production components."""

    @pytest.mark.ci_fast
    def test_no_cache_manager(self):
        agent = _make_agent()
        assert not hasattr(agent, "cache_manager")

    @pytest.mark.ci_fast
    def test_no_metrics_tracker(self):
        agent = _make_agent()
        assert not hasattr(agent, "metrics_tracker")

    @pytest.mark.ci_fast
    def test_no_parallel_executor(self):
        agent = _make_agent()
        assert not hasattr(agent, "parallel_executor")

    @pytest.mark.ci_fast
    def test_no_contextual_analyzer(self):
        agent = _make_agent()
        assert not hasattr(agent, "contextual_analyzer")


class TestGuttedAgentRouting:
    """Verify routing still works without cache/metrics."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_route_query_without_cache(self):
        agent = _make_agent()
        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.85
        mock_prediction.reasoning_chain = ["Direct routing"]

        with patch.object(
            agent.routing_module, "forward", return_value=mock_prediction
        ):
            result = await agent.route_query(query="test query", tenant_id="t1")

        assert isinstance(result, RoutingOutput)
        assert result.recommended_agent == "search_agent"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_fallback_on_dspy_failure(self):
        agent = _make_agent()
        with patch.object(
            agent.routing_module,
            "forward",
            side_effect=Exception("DSPy error"),
        ):
            result = await agent.route_query(query="test query", tenant_id="t1")

        assert result.recommended_agent == "search_agent"
        assert result.confidence <= 0.3
        assert "Fallback" in result.reasoning or "error" in result.reasoning.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
