"""
Unit tests for agent router endpoints.

Tests the routing→orchestration handoff in _execute_routing_task.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_telemetry_manager():
    """Create mock telemetry manager."""
    manager = MagicMock()

    @contextmanager
    def fake_span(*args, **kwargs):
        yield MagicMock()

    manager.span = fake_span
    return manager


@pytest.mark.unit
class TestRoutingOrchestrationHandoff:
    """Test that _execute_routing_task invokes orchestrator when needed."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_without_orchestration(self):
        """When needs_orchestration is False, return routing result directly."""
        from cogniverse_runtime.routers.agents import AgentTask, _execute_routing_task

        mock_result = Mock()
        mock_result.recommended_agent = "video_search_agent"
        mock_result.confidence = 0.9
        mock_result.reasoning = "Simple search"
        mock_result.enhanced_query = "test query enhanced"
        mock_result.entities = []
        mock_result.relationships = []
        mock_result.query_variants = []
        mock_result.metadata = {"needs_orchestration": False}

        task = AgentTask(
            agent_name="routing_agent",
            query="find videos of cats",
            context={"tenant_id": "test_tenant"},
        )

        with (
            patch(
                "cogniverse_runtime.routers.agents._config_manager",
                MagicMock(),
            ),
            patch(
                "cogniverse_runtime.routers.agents._schema_loader",
                MagicMock(),
            ),
            patch(
                "cogniverse_agents.routing_agent.RoutingAgent"
            ) as mock_agent_cls,
            patch(
                "cogniverse_agents.routing_agent.RoutingDeps"
            ),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value={
                    "llm_config": {"primary": {}},
                    "routing_agent": {},
                },
            ),
        ):
            mock_agent = mock_agent_cls.return_value
            mock_agent.route_query = AsyncMock(return_value=mock_result)

            result = await _execute_routing_task(task, "test_tenant")

        assert result["status"] == "success"
        assert result["recommended_agent"] == "video_search_agent"
        assert "orchestration_result" not in result

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_with_orchestration_handoff(self, mock_telemetry_manager):
        """When needs_orchestration is True, invoke MultiAgentOrchestrator."""
        from cogniverse_runtime.routers.agents import AgentTask, _execute_routing_task

        mock_result = Mock()
        mock_result.recommended_agent = "video_search_agent"
        mock_result.confidence = 0.5
        mock_result.reasoning = "Complex multi-step query"
        mock_result.enhanced_query = "enhanced complex query"
        mock_result.entities = [{"text": "robots"}]
        mock_result.relationships = []
        mock_result.query_variants = []
        mock_result.metadata = {"needs_orchestration": True}

        mock_orch_result = {
            "workflow_id": "wf_abc123",
            "status": "completed",
            "result": {"content": "orchestrated results"},
        }

        task = AgentTask(
            agent_name="routing_agent",
            query="find robots then summarize and create report",
            context={"tenant_id": "test_tenant"},
        )

        mock_registry = MagicMock()
        mock_registry.list_agents.return_value = ["search_agent", "summarizer_agent"]
        mock_agent_ep = MagicMock()
        mock_agent_ep.capabilities = ["video_content_search"]
        mock_agent_ep.timeout = 30
        mock_registry.get_agent.return_value = mock_agent_ep

        with (
            patch(
                "cogniverse_runtime.routers.agents._config_manager",
                MagicMock(),
            ),
            patch(
                "cogniverse_runtime.routers.agents._schema_loader",
                MagicMock(),
            ),
            patch(
                "cogniverse_runtime.routers.agents.get_registry",
                return_value=mock_registry,
            ),
            patch(
                "cogniverse_agents.routing_agent.RoutingAgent"
            ) as mock_agent_cls,
            patch(
                "cogniverse_agents.routing_agent.RoutingDeps"
            ),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value={
                    "llm_config": {"primary": {}},
                    "routing_agent": {},
                },
            ),
            patch(
                "cogniverse_agents.multi_agent_orchestrator.MultiAgentOrchestrator"
            ) as mock_orch_cls,
            patch(
                "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
                return_value=mock_telemetry_manager,
            ),
        ):
            mock_agent = mock_agent_cls.return_value
            mock_agent.route_query = AsyncMock(return_value=mock_result)

            mock_orch = mock_orch_cls.return_value
            mock_orch.process_complex_query = AsyncMock(
                return_value=mock_orch_result
            )

            result = await _execute_routing_task(task, "test_tenant")

        assert result["status"] == "success"
        assert result["needs_orchestration"] is True
        assert result["orchestration_result"] == mock_orch_result
        mock_orch.process_complex_query.assert_called_once()
