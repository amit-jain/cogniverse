"""
Unit tests for agent router endpoints.

Tests the routing→orchestration handoff via AgentDispatcher.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher


@pytest.fixture
def mock_telemetry_manager():
    """Create mock telemetry manager."""
    manager = MagicMock()

    @contextmanager
    def fake_span(*args, **kwargs):
        yield MagicMock()

    manager.span = fake_span
    return manager


@pytest.fixture
def dispatcher():
    """Create an AgentDispatcher with mock dependencies."""
    registry = MagicMock()
    config_manager = MagicMock()
    schema_loader = MagicMock()
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.mark.unit
class TestRoutingOrchestrationHandoff:
    """Test that AgentDispatcher routes to orchestrator when needed."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_without_orchestration(self, dispatcher):
        """When needs_orchestration is False, return routing result directly."""
        mock_result = Mock()
        mock_result.recommended_agent = "video_search_agent"
        mock_result.confidence = 0.9
        mock_result.reasoning = "Simple search"
        mock_result.enhanced_query = "test query enhanced"
        mock_result.entities = []
        mock_result.relationships = []
        mock_result.query_variants = []
        mock_result.metadata = {"needs_orchestration": False}

        # Configure registry: routing_agent has ["routing"],
        # video_search_agent has ["video_search"] for downstream execution
        routing_ep = MagicMock()
        routing_ep.capabilities = ["routing"]
        search_ep = MagicMock()
        search_ep.capabilities = ["video_search"]

        def get_agent_by_name(name):
            if name == "routing_agent":
                return routing_ep
            if name == "video_search_agent":
                return search_ep
            return None

        dispatcher._registry.get_agent.side_effect = get_agent_by_name

        # Mock the lazy imports inside _execute_routing_task
        mock_routing_agent_cls = MagicMock()
        mock_routing_agent = mock_routing_agent_cls.return_value
        mock_routing_agent.route_query = AsyncMock(return_value=mock_result)

        mock_routing_module = MagicMock()
        mock_routing_module.RoutingAgent = mock_routing_agent_cls
        mock_routing_module.RoutingDeps = MagicMock()

        mock_config_utils = MagicMock()
        mock_config_obj = MagicMock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            "llm_config": {"primary": {}},
            "routing_agent": {},
        }.get(key, default)
        mock_config_utils.get_config.return_value = mock_config_obj

        mock_telemetry_config_module = MagicMock()

        mock_unified_config = MagicMock()

        mock_downstream = {
            "status": "success",
            "agent": "search_agent",
            "message": "Found 3 results for 'test query enhanced'",
            "results_count": 3,
            "results": [],
            "profile": "test_profile",
        }

        with (
            patch.dict(
                "sys.modules",
                {
                    "cogniverse_agents": MagicMock(),
                    "cogniverse_agents.routing_agent": mock_routing_module,
                    "cogniverse_foundation.config.unified_config": mock_unified_config,
                    "cogniverse_foundation.config.utils": mock_config_utils,
                    "cogniverse_foundation.telemetry.config": mock_telemetry_config_module,
                },
            ),
            patch.object(
                dispatcher,
                "_execute_search_task",
                new_callable=AsyncMock,
                return_value=mock_downstream,
            ),
        ):
            result = await dispatcher.dispatch(
                agent_name="routing_agent",
                query="find videos of cats",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["recommended_agent"] == "video_search_agent"
        assert "orchestration_result" not in result
        assert result["downstream_result"] == mock_downstream

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_with_orchestration_handoff(
        self, dispatcher, mock_telemetry_manager
    ):
        """When needs_orchestration is True, invoke MultiAgentOrchestrator."""
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

        # Configure registry to find routing agent + list agents for orchestrator
        agent_ep = MagicMock()
        agent_ep.capabilities = ["routing"]

        other_ep = MagicMock()
        other_ep.capabilities = ["video_content_search"]
        other_ep.timeout = 30

        def get_agent_side_effect(name):
            if name == "routing_agent":
                return agent_ep
            return other_ep

        dispatcher._registry.get_agent.side_effect = get_agent_side_effect
        dispatcher._registry.list_agents.return_value = [
            "search_agent",
            "summarizer_agent",
        ]

        mock_routing_agent_cls = MagicMock()
        mock_routing_agent = mock_routing_agent_cls.return_value
        mock_routing_agent.route_query = AsyncMock(return_value=mock_result)

        mock_routing_module = MagicMock()
        mock_routing_module.RoutingAgent = mock_routing_agent_cls
        mock_routing_module.RoutingDeps = MagicMock()

        mock_orch_cls = MagicMock()
        mock_orch = mock_orch_cls.return_value
        mock_orch.process_complex_query = AsyncMock(return_value=mock_orch_result)

        mock_orch_module = MagicMock()
        mock_orch_module.MultiAgentOrchestrator = mock_orch_cls

        mock_config_utils = MagicMock()
        mock_config_obj2 = MagicMock()
        mock_config_obj2.get.side_effect = lambda key, default=None: {
            "llm_config": {"primary": {}},
            "routing_agent": {},
        }.get(key, default)
        mock_config_utils.get_config.return_value = mock_config_obj2

        mock_telemetry_manager_module = MagicMock()
        mock_telemetry_manager_module.get_telemetry_manager.return_value = (
            mock_telemetry_manager
        )

        with patch.dict(
            "sys.modules",
            {
                "cogniverse_agents": MagicMock(),
                "cogniverse_agents.routing_agent": mock_routing_module,
                "cogniverse_agents.multi_agent_orchestrator": mock_orch_module,
                "cogniverse_foundation.config.unified_config": MagicMock(),
                "cogniverse_foundation.config.utils": mock_config_utils,
                "cogniverse_foundation.telemetry.config": MagicMock(),
                "cogniverse_foundation.telemetry.manager": mock_telemetry_manager_module,
            },
        ):
            result = await dispatcher.dispatch(
                agent_name="routing_agent",
                query="find robots then summarize and create report",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["needs_orchestration"] is True
        assert result["orchestration_result"] == mock_orch_result
        mock_orch.process_complex_query.assert_called_once()


@pytest.mark.unit
class TestAgentDispatcherCapabilityRouting:
    """Test that dispatch routes to the correct _execute_* method by capability."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unknown_agent_raises(self, dispatcher):
        """Unknown agent name raises ValueError."""
        dispatcher._registry.get_agent.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await dispatcher.dispatch(agent_name="nonexistent", query="test")

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unsupported_capabilities_raises(self, dispatcher):
        """Agent with unrecognised capabilities raises ValueError."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["unknown_capability"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with pytest.raises(ValueError, match="no supported execution path"):
            await dispatcher.dispatch(agent_name="weird_agent", query="test")

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_search_capability(self, dispatcher):
        """Agent with 'search' capability routes to search handler."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["search"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with patch.object(
            dispatcher,
            "_execute_search_task",
            new_callable=AsyncMock,
            return_value={"status": "success", "agent": "search_agent"},
        ) as mock_search:
            result = await dispatcher.dispatch(
                agent_name="search_agent",
                query="find cats",
                context={"tenant_id": "t1"},
            )

        mock_search.assert_called_once_with(
            "find cats", "t1", 10, conversation_history=[]
        )
        assert result["status"] == "success"


@pytest.mark.unit
class TestRoutingOptimizationActions:
    """Test optimization action handling in dispatcher."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_optimize_routing_records_examples(self, dispatcher):
        """optimize_routing action records examples to optimizer."""
        mock_optimizer = AsyncMock()
        mock_optimizer.record_routing_experience = AsyncMock()

        mock_routing_agent = MagicMock()
        mock_routing_agent._get_optimizer.return_value = mock_optimizer
        mock_routing_agent.get_routing_statistics.return_value = {}

        routing_ep = MagicMock()
        routing_ep.capabilities = ["routing"]
        dispatcher._registry.get_agent.return_value = routing_ep

        examples = [
            {
                "query": "find cat videos",
                "chosen_agent": "search_agent",
                "confidence": 0.9,
                "search_quality": 0.8,
                "agent_success": True,
            },
            {
                "query": "summarize results",
                "chosen_agent": "summarizer_agent",
                "confidence": 0.85,
                "search_quality": 0.7,
                "agent_success": True,
            },
        ]

        mock_config_obj = MagicMock()
        mock_routing_module = MagicMock()
        mock_routing_module.RoutingAgent.return_value = mock_routing_agent
        mock_routing_module.RoutingDeps = MagicMock(return_value=MagicMock())

        mock_config_utils = MagicMock()
        mock_config_utils.get_config.return_value = mock_config_obj

        with (
            patch.dict(
                "sys.modules",
                {
                    "cogniverse_agents.routing_agent": mock_routing_module,
                    "cogniverse_foundation.config.utils": mock_config_utils,
                    "cogniverse_foundation.telemetry.config": MagicMock(),
                },
            ),
        ):
            result = await dispatcher._handle_routing_optimization(
                mock_routing_agent,
                {"action": "optimize_routing", "examples": examples},
                "default",
            )

        assert result["status"] == "optimization_triggered"
        assert result["training_examples"] == 2
        assert mock_optimizer.record_routing_experience.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_optimize_routing_no_examples(self, dispatcher):
        """optimize_routing with empty examples returns insufficient_data."""
        mock_agent = MagicMock()
        result = await dispatcher._handle_routing_optimization(
            mock_agent, {"action": "optimize_routing", "examples": []}, "default"
        )
        assert result["status"] == "insufficient_data"
        assert result["training_examples"] == 0

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_optimize_routing_no_optimizer(self, dispatcher):
        """optimize_routing when optimizer not enabled returns error."""
        mock_agent = MagicMock()
        mock_agent._get_optimizer.return_value = None
        result = await dispatcher._handle_routing_optimization(
            mock_agent,
            {"action": "optimize_routing", "examples": [{"query": "test"}]},
            "default",
        )
        assert result["status"] == "error"
        assert "not enabled" in result["message"]

    @pytest.mark.ci_fast
    def test_optimization_status_active(self, dispatcher):
        """get_optimization_status returns metrics when optimizer is active."""
        mock_agent = MagicMock()
        mock_agent._get_optimizer.return_value = MagicMock()
        mock_agent.get_routing_statistics.return_value = {
            "total_queries": 100,
            "successful_routes": 85,
        }
        result = dispatcher._handle_optimization_status(mock_agent, "default")
        assert result["status"] == "active"
        assert result["optimizer_ready"] is True
        assert result["metrics"]["total_queries"] == 100

    @pytest.mark.ci_fast
    def test_optimization_status_inactive(self, dispatcher):
        """get_optimization_status when optimizer not enabled."""
        mock_agent = MagicMock()
        mock_agent._get_optimizer.return_value = None
        result = dispatcher._handle_optimization_status(mock_agent, "default")
        assert result["status"] == "inactive"
        assert result["optimizer_ready"] is False
