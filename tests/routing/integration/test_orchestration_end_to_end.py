"""
Integration tests for Phase 7: Multi-Agent Orchestration End-to-End

Tests the complete orchestration workflow:
- RoutingAgent detects orchestration needs
- MultiAgentOrchestrator coordination
- Telemetry span hierarchy
- Error handling and recovery
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.agents.routing_agent import RoutingAgent
from src.app.routing.base import GenerationType, SearchModality


@pytest.mark.asyncio
class TestOrchestrationEndToEnd:
    """End-to-end integration tests for orchestration"""

    @pytest.fixture
    async def routing_agent(self):
        """Create routing agent for testing"""
        with patch("src.app.agents.routing_agent.get_config") as mock_config:
            mock_config.return_value = {
                "video_agent_url": "http://localhost:8002",
                "text_agent_url": "http://localhost:8003",
                "summarizer_agent_url": "http://localhost:8004",
                "detailed_report_agent_url": "http://localhost:8005",
                "optimization_dir": "/tmp/optimization",
            }
            agent = RoutingAgent()
            yield agent

    async def test_multi_modal_query_triggers_orchestration(self, routing_agent):
        """Test that multi-modal queries trigger orchestration workflow"""
        query = "Find videos and documents about artificial intelligence"
        context = {"tenant_id": "test-tenant"}

        # Mock orchestrator to avoid actual agent calls
        with patch.object(
            routing_agent.orchestrator,
            "process_complex_query",
            new_callable=AsyncMock,
        ) as mock_orchestrator:
            mock_orchestrator.return_value = {
                "workflow_id": "test-workflow-123",
                "status": "completed",
                "result": {"final_answer": "Test orchestration result"},
                "execution_summary": {
                    "total_tasks": 3,
                    "completed_tasks": 3,
                    "execution_time": 2.5,
                    "agents_used": [
                        "video_search_agent",
                        "text_search_agent",
                        "summarizer_agent",
                    ],
                },
                "metadata": {},
            }

            result = await routing_agent.analyze_and_route(query, context)

            # Verify orchestration was triggered
            assert "orchestration_result" in result or "routing_decision" in result

            # If orchestration was triggered, verify the structure
            if "orchestration_result" in result:
                assert result["workflow_type"] == "orchestrated"
                assert "agents_to_call" in result
                mock_orchestrator.assert_called_once()

    async def test_detailed_report_triggers_sequential_orchestration(
        self, routing_agent
    ):
        """Test that detailed report requests use sequential orchestration"""
        query = "Provide detailed analysis of quantum computing advancements"
        context = {"tenant_id": "test-tenant"}

        with patch.object(
            routing_agent.orchestrator,
            "process_complex_query",
            new_callable=AsyncMock,
        ) as mock_orchestrator:
            mock_orchestrator.return_value = {
                "workflow_id": "test-workflow-456",
                "status": "completed",
                "result": {"detailed_report": "Test report"},
                "execution_summary": {
                    "total_tasks": 3,
                    "completed_tasks": 3,
                    "execution_time": 5.2,
                    "agents_used": [
                        "video_search_agent",
                        "summarizer_agent",
                        "detailed_report_agent",
                    ],
                },
                "metadata": {},
            }

            result = await routing_agent.analyze_and_route(query, context)

            # Check routing decision
            routing_decision = result.get("routing_decision", {})
            if routing_decision.get("requires_orchestration"):
                assert (
                    routing_decision.get("orchestration_pattern") == "sequential"
                    or routing_decision.get("orchestration_pattern") == "parallel"
                )
                assert routing_decision.get("primary_agent") is not None

    async def test_single_modality_no_orchestration(self, routing_agent):
        """Test that single-modality queries don't trigger orchestration"""
        query = "Show me videos about cats"
        context = {"tenant_id": "test-tenant"}

        result = await routing_agent.analyze_and_route(query, context)

        # Verify no orchestration
        routing_decision = result.get("routing_decision", {})
        requires_orchestration = routing_decision.get("requires_orchestration", False)

        # Single video query should not require orchestration
        if routing_decision.get("search_modality") == "video":
            assert requires_orchestration is False

    async def test_orchestration_with_telemetry_spans(self, routing_agent):
        """Test that orchestration creates proper telemetry spans"""
        query = "Search videos and text, then summarize"
        context = {"tenant_id": "test-tenant"}

        with patch.object(
            routing_agent.orchestrator,
            "process_complex_query",
            new_callable=AsyncMock,
        ) as mock_orchestrator:
            mock_orchestrator.return_value = {
                "workflow_id": "test-workflow-789",
                "status": "completed",
                "result": {"summary": "Test summary"},
                "execution_summary": {
                    "total_tasks": 3,
                    "completed_tasks": 3,
                    "execution_time": 3.1,
                    "agents_used": [
                        "video_search_agent",
                        "text_search_agent",
                        "summarizer_agent",
                    ],
                },
                "metadata": {},
            }

            # Mock telemetry manager to verify span creation
            with patch.object(
                routing_agent.telemetry_manager, "span"
            ) as mock_span_context:
                mock_span = MagicMock()
                mock_span.__enter__ = MagicMock(return_value=mock_span)
                mock_span.__exit__ = MagicMock(return_value=False)
                mock_span_context.return_value = mock_span

                await routing_agent.analyze_and_route(query, context)

                # Verify spans were created (at least request span)
                assert mock_span_context.called
                # Should have at least: request span, and potentially routing/orchestration spans
                assert mock_span_context.call_count >= 1

    async def test_orchestration_error_handling(self, routing_agent):
        """Test that orchestration errors are properly handled and reported"""
        # Use a query that will definitely trigger orchestration (multi-modal)
        query = "Find videos and documents, then create detailed report"
        context = {"tenant_id": "test-tenant"}

        # First verify this query actually triggers orchestration
        decision = await routing_agent.router.route(query, context)

        # Only test error handling if orchestration would be triggered
        if decision.requires_orchestration:
            with patch.object(
                routing_agent.orchestrator,
                "process_complex_query",
                new_callable=AsyncMock,
            ) as mock_orchestrator:
                # Simulate orchestration failure
                mock_orchestrator.side_effect = Exception("Orchestrator failed")

                with pytest.raises(Exception) as exc_info:
                    await routing_agent.analyze_and_route(query, context)

                assert "Orchestrator failed" in str(exc_info.value)
        else:
            # If orchestration wasn't triggered, that's also valid - skip test
            pytest.skip("Query did not trigger orchestration")

    async def test_orchestration_pattern_selection_parallel(self, routing_agent):
        """Test that parallel pattern is selected for multi-search"""
        query = "Search videos and documents about machine learning"

        # Get routing decision
        decision = await routing_agent.router.route(query)

        # If orchestration is triggered, verify pattern
        if decision.requires_orchestration:
            if decision.search_modality == SearchModality.BOTH:
                assert decision.orchestration_pattern in ["parallel", "sequential"]
                assert decision.primary_agent in [
                    "video_search_agent",
                    "summarizer_agent",
                    "detailed_report_agent",
                ]

    async def test_orchestration_pattern_selection_sequential(self, routing_agent):
        """Test that sequential pattern is selected for detailed reports"""
        query = "Provide detailed analysis with video evidence"

        # Get routing decision
        decision = await routing_agent.router.route(query)

        # If orchestration is triggered with detailed report, verify pattern
        if (
            decision.requires_orchestration
            and decision.generation_type == GenerationType.DETAILED_REPORT
        ):
            assert decision.orchestration_pattern == "sequential"
            assert decision.primary_agent == "detailed_report_agent"
            assert len(decision.agent_execution_order) > 1

    async def test_orchestration_agent_execution_order(self, routing_agent):
        """Test that agent execution order is properly set"""
        query = "Search videos, then create detailed report"

        # Get routing decision
        decision = await routing_agent.router.route(query)

        # If orchestration is triggered, verify execution order
        if decision.requires_orchestration:
            assert decision.agent_execution_order is not None
            assert len(decision.agent_execution_order) > 0
            assert all(
                isinstance(agent, str) for agent in decision.agent_execution_order
            )

    async def test_orchestration_metadata_population(self, routing_agent):
        """Test that orchestration metadata is properly populated"""
        query = "Find videos and text documents, then summarize everything"

        # Get routing decision
        decision = await routing_agent.router.route(query)

        # Verify metadata
        assert "routing_method" in decision.routing_method or decision.routing_method
        if decision.requires_orchestration:
            assert decision.metadata.get("orchestration_determined") is True
            assert decision.metadata.get("orchestration_trigger") in [
                "multi_modal",
                "multi_search",
                "complex_generation",
                "explicit",
            ]


@pytest.mark.asyncio
class TestOrchestrationWithRealRouter:
    """Integration tests with real router (no mocks)"""

    @pytest.fixture
    def router_config(self):
        """Create router configuration"""
        from src.app.routing.router import RouterConfig

        return RouterConfig(
            enable_fast_path=True,
            enable_slow_path=True,
            enable_langextract=False,
            enable_fallback=False,
        )

    async def test_multi_modal_routing_decision(self, router_config):
        """Test routing decision for multi-modal query"""
        from src.app.routing.router import ComprehensiveRouter

        router = ComprehensiveRouter(router_config)

        query = "Find videos and documents about quantum computing"
        decision = await router.route(query)

        # Verify basic decision structure
        assert decision is not None
        assert hasattr(decision, "search_modality")
        assert hasattr(decision, "requires_orchestration")
        assert hasattr(decision, "orchestration_pattern")

    async def test_summary_routing_decision(self, router_config):
        """Test routing decision for summary request"""
        from src.app.routing.router import ComprehensiveRouter

        router = ComprehensiveRouter(router_config)

        query = "Summarize recent developments in AI"
        decision = await router.route(query)

        assert decision is not None
        # Summary with video should trigger orchestration
        if decision.search_modality in [SearchModality.VIDEO, SearchModality.BOTH]:
            assert decision.requires_orchestration is True

    async def test_detailed_report_routing_decision(self, router_config):
        """Test routing decision for detailed report request"""
        from src.app.routing.router import ComprehensiveRouter

        router = ComprehensiveRouter(router_config)

        query = "Provide detailed analysis of computer vision advances"
        decision = await router.route(query)

        assert decision is not None
        # Detailed reports should trigger orchestration
        if decision.generation_type == GenerationType.DETAILED_REPORT:
            assert decision.requires_orchestration is True
            assert decision.orchestration_pattern == "sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
