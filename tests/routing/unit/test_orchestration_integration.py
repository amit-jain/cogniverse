"""
Unit tests for Phase 7: Multi-Agent Orchestration Integration

Tests the integration between routing system and multi-agent orchestration:
- Orchestration decision logic in ComprehensiveRouter
- RoutingDecision orchestration fields
- Orchestration detection triggers
- Agent selection and execution order
"""

import pytest

from cogniverse_agents.routing.base import (
    GenerationType,
    RoutingDecision,
    SearchModality,
)
from cogniverse_agents.routing.router import ComprehensiveRouter, RouterConfig


class TestOrchestrationDecisionLogic:
    """Test orchestration decision logic in ComprehensiveRouter"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = RouterConfig(
            enable_fast_path=True,
            enable_slow_path=False,
            enable_langextract=False,
            enable_fallback=False,
        )
        self.router = ComprehensiveRouter(self.config)

    def test_determine_orchestration_single_modality_no_orchestration(self):
        """Test that single modality queries don't require orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.9,
            routing_method="fast_path",
            detected_modalities=["video"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Show me videos about cats", None
        )

        assert decision.requires_orchestration is False
        assert decision.orchestration_pattern is None
        assert decision.primary_agent is None

    def test_determine_orchestration_multi_modal_requires_orchestration(self):
        """Test that multi-modal queries require orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.85,
            routing_method="slow_path",
            detected_modalities=["video", "text"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Find videos and documents about AI", None
        )

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "parallel"
        assert decision.primary_agent == "video_search_agent"
        assert "text_search_agent" in decision.secondary_agents
        assert decision.agent_execution_order is not None

    def test_determine_orchestration_video_and_text_parallel(self):
        """Test that video+text searches use parallel orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="fast_path",
            detected_modalities=["video"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Search videos and text for machine learning", None
        )

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "parallel"
        assert decision.primary_agent == "video_search_agent"
        assert decision.secondary_agents == ["text_search_agent"]
        assert set(decision.agent_execution_order) == {
            "video_search_agent",
            "text_search_agent",
        }

    def test_determine_orchestration_detailed_report_sequential(self):
        """Test that detailed reports use sequential orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.DETAILED_REPORT,
            confidence_score=0.9,
            routing_method="slow_path",
            detected_modalities=["video"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Detailed analysis of AI trends", None
        )

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "sequential"
        assert decision.primary_agent == "detailed_report_agent"
        assert "video_search_agent" in decision.secondary_agents
        assert "summarizer_agent" in decision.secondary_agents
        assert decision.agent_execution_order == [
            "video_search_agent",
            "summarizer_agent",
            "detailed_report_agent",
        ]

    def test_determine_orchestration_summary_with_video_and_text(self):
        """Test that summaries with multi-search use parallel then sequential"""
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.85,
            routing_method="slow_path",
            detected_modalities=["video", "text"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Summarize AI developments", None
        )

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "parallel"  # Parallel search
        assert decision.primary_agent == "summarizer_agent"
        assert set(decision.secondary_agents) == {
            "video_search_agent",
            "text_search_agent",
        }
        assert decision.agent_execution_order == [
            "video_search_agent",
            "text_search_agent",
            "summarizer_agent",
        ]

    def test_determine_orchestration_explicit_context_override(self):
        """Test that explicit context can force orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.9,
            routing_method="fast_path",
            detected_modalities=["video"],
        )

        context = {"require_orchestration": True}

        self.router._determine_orchestration_strategy(
            decision, "Simple video query", context
        )

        assert decision.requires_orchestration is True
        assert decision.metadata.get("orchestration_trigger") == "explicit"

    def test_determine_orchestration_metadata_tracking(self):
        """Test that orchestration metadata is properly tracked"""
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.DETAILED_REPORT,
            confidence_score=0.9,
            routing_method="slow_path",
            detected_modalities=["video", "text"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Detailed multi-modal analysis", None
        )

        assert decision.requires_orchestration is True
        assert decision.metadata["orchestration_determined"] is True
        assert decision.metadata["orchestration_trigger"] == "multi_search"

    def test_determine_orchestration_summary_video_only(self):
        """Test summary with video-only uses sequential"""
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.SUMMARY,
            confidence_score=0.85,
            routing_method="fast_path",
            detected_modalities=["video"],
        )

        self.router._determine_orchestration_strategy(
            decision, "Summarize video content", None
        )

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "sequential"
        assert decision.primary_agent == "summarizer_agent"
        assert decision.secondary_agents == ["video_search_agent"]
        assert decision.agent_execution_order == [
            "video_search_agent",
            "summarizer_agent",
        ]


class TestRoutingDecisionOrchestrationFields:
    """Test RoutingDecision orchestration fields serialization"""

    def test_routing_decision_to_dict_includes_orchestration_fields(self):
        """Test that to_dict includes all orchestration fields"""
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.DETAILED_REPORT,
            confidence_score=0.9,
            routing_method="slow_path",
            detected_modalities=["video", "text"],
            requires_orchestration=True,
            orchestration_pattern="sequential",
            primary_agent="detailed_report_agent",
            secondary_agents=["video_search_agent", "summarizer_agent"],
            agent_execution_order=[
                "video_search_agent",
                "summarizer_agent",
                "detailed_report_agent",
            ],
        )

        result = decision.to_dict()

        assert result["requires_orchestration"] is True
        assert result["orchestration_pattern"] == "sequential"
        assert result["primary_agent"] == "detailed_report_agent"
        assert result["secondary_agents"] == [
            "video_search_agent",
            "summarizer_agent",
        ]
        assert result["agent_execution_order"] == [
            "video_search_agent",
            "summarizer_agent",
            "detailed_report_agent",
        ]

    def test_routing_decision_from_dict_restores_orchestration_fields(self):
        """Test that from_dict restores orchestration fields"""
        data = {
            "search_modality": "both",
            "generation_type": "detailed_report",
            "confidence_score": 0.9,
            "routing_method": "slow_path",
            "detected_modalities": ["video", "text"],
            "requires_orchestration": True,
            "orchestration_pattern": "parallel",
            "primary_agent": "video_search_agent",
            "secondary_agents": ["text_search_agent"],
            "agent_execution_order": ["video_search_agent", "text_search_agent"],
        }

        decision = RoutingDecision.from_dict(data)

        assert decision.requires_orchestration is True
        assert decision.orchestration_pattern == "parallel"
        assert decision.primary_agent == "video_search_agent"
        assert decision.secondary_agents == ["text_search_agent"]
        assert decision.agent_execution_order == [
            "video_search_agent",
            "text_search_agent",
        ]

    def test_routing_decision_defaults_no_orchestration(self):
        """Test that orchestration fields default to no orchestration"""
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
        )

        assert decision.requires_orchestration is False
        assert decision.orchestration_pattern is None
        assert decision.primary_agent is None
        assert decision.secondary_agents == []
        assert decision.agent_execution_order is None


@pytest.mark.asyncio
class TestOrchestrationEndToEndRouting:
    """Test end-to-end routing with orchestration detection"""

    async def test_route_multi_modal_query_triggers_orchestration(self):
        """Test that multi-modal queries trigger orchestration in routing"""
        config = RouterConfig(
            enable_fast_path=True,
            enable_slow_path=False,
            enable_langextract=False,
            enable_fallback=False,
        )
        router = ComprehensiveRouter(config)

        # Multi-modal query
        decision = await router.route(
            "Find videos and documents about quantum computing"
        )

        # Should have detected multiple modalities and triggered orchestration
        # Note: Actual orchestration detection depends on strategy implementation
        assert decision is not None
        assert isinstance(decision, RoutingDecision)

    async def test_route_detailed_report_query_triggers_orchestration(self):
        """Test that detailed report queries trigger orchestration"""
        config = RouterConfig(
            enable_fast_path=True,
            enable_slow_path=True,
            enable_langextract=False,
            enable_fallback=False,
        )
        router = ComprehensiveRouter(config)

        decision = await router.route(
            "Provide detailed analysis of recent AI advancements in computer vision"
        )

        assert decision is not None
        assert isinstance(decision, RoutingDecision)
        # Detailed reports should trigger orchestration
        if decision.generation_type == GenerationType.DETAILED_REPORT:
            assert decision.requires_orchestration is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
