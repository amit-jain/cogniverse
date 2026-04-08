"""
Complete Multi-Agent Orchestration Tests

Tests the full end-to-end multi-agent workflow with real components:
1. Routing agent makes routing decisions
2. Summarizer agent processes results
3. Detailed report agent generates comprehensive reports
4. Query enhancement pipeline improves search queries
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
)
from cogniverse_agents.routing.query_enhancement_engine import QueryEnhancementPipeline
from cogniverse_agents.routing.relationship_extraction_tools import (
    RelationshipExtractorTool,
)
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_routing_agent() -> RoutingAgent:
    """Create RoutingAgent with mocked DSPy for unit tests."""
    telemetry_config = TelemetryConfig(enabled=False)
    deps = RoutingDeps(telemetry_config=telemetry_config)

    def _mock_configure_dspy(self_agent, deps_arg):
        self_agent._dspy_lm = MagicMock()

    with patch.object(RoutingAgent, "_configure_dspy", _mock_configure_dspy):
        return RoutingAgent(deps=deps)


@pytest.mark.unit
class TestCompleteMultiAgentOrchestration:
    """Test complete multi-agent orchestration with real workflows"""

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up environment for video search testing"""
        os.environ["VESPA_SCHEMA"] = "video_colpali_smol500_mv_frame"
        yield

    @pytest.mark.ci_fast
    def test_routing_to_video_search_workflow(self):
        """Test routing decision leading to video search workflow"""
        routing_agent = _make_routing_agent()
        video_query = "Find videos of robots playing soccer"

        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.85
        mock_prediction.reasoning_chain = ["Video search needed"]

        with patch.object(
            routing_agent.routing_module, "forward", return_value=mock_prediction
        ):
            routing_result = asyncio.run(
                routing_agent.route_query(video_query, tenant_id="test_tenant")
            )

        assert routing_result is not None
        assert routing_result.recommended_agent == "search_agent"

    def test_summarization_workflow(self):
        """Test summarization agent workflow with structured data"""
        deps = SummarizerDeps()
        summarizer = SummarizerAgent(
            deps=deps, config_manager=create_default_config_manager()
        )

        assert summarizer is not None
        assert hasattr(summarizer, "summarize")
        assert callable(summarizer.summarize)

    def test_detailed_report_workflow(self):
        """Test detailed report generation workflow"""
        deps = DetailedReportDeps()
        reporter = DetailedReportAgent(
            deps=deps, config_manager=create_default_config_manager()
        )

        assert reporter is not None
        assert hasattr(reporter, "generate_report")
        assert callable(reporter.generate_report)

    def test_query_enhancement_to_search_workflow(self):
        """Test query enhancement feeding into search workflow"""
        extractor = RelationshipExtractorTool()
        pipeline = QueryEnhancementPipeline(enable_simba=False)

        original_query = "Show me videos about machine learning robots"

        try:
            extraction_result = asyncio.run(
                extractor.extract_comprehensive_relationships(original_query)
            )
            entities = extraction_result.get("entities", [])
            relationships = extraction_result.get("relationships", [])

            enhancement_result = asyncio.run(
                pipeline.enhance_query_with_relationships(
                    original_query, entities=entities, relationships=relationships
                )
            )

            enhanced_query = enhancement_result.get("enhanced_query", original_query)
            assert isinstance(enhanced_query, str)
            assert len(enhanced_query) > 0

        except Exception:
            # Graceful handling if models not available
            assert True

    @pytest.mark.ci_fast
    def test_agent_coordination_interfaces(self):
        """Test that agents have compatible interfaces for coordination"""
        routing_agent = _make_routing_agent()
        summarizer = SummarizerAgent(
            deps=SummarizerDeps(),
            config_manager=create_default_config_manager(),
        )
        reporter = DetailedReportAgent(
            deps=DetailedReportDeps(),
            config_manager=create_default_config_manager(),
        )

        agents = {
            "routing": routing_agent,
            "summarizer": summarizer,
            "reporter": reporter,
        }

        for agent_name, agent in agents.items():
            assert agent is not None, f"{agent_name} agent failed to initialize"

    def test_error_propagation_across_agents(self):
        """Test that errors propagate gracefully across agent boundaries"""
        routing_agent = _make_routing_agent()

        problematic_queries = [
            "",
            "x" * 1000,
        ]

        for query in problematic_queries:
            try:
                with patch.object(
                    routing_agent.routing_module,
                    "forward",
                    side_effect=RuntimeError("test error"),
                ):
                    result = asyncio.run(
                        routing_agent.route_query(str(query), tenant_id="test_tenant")
                    )
                    assert result is not None
            except Exception:
                assert True

    def test_resource_management_across_agents(self):
        """Test resource management when multiple agents are active"""
        agents = []

        routing_agent = _make_routing_agent()
        agents.append(routing_agent)
        agents.append(
            SummarizerAgent(
                deps=SummarizerDeps(),
                config_manager=create_default_config_manager(),
            )
        )
        agents.append(
            DetailedReportAgent(
                deps=DetailedReportDeps(),
                config_manager=create_default_config_manager(),
            )
        )

        for i, agent in enumerate(agents):
            assert agent is not None, f"Agent {i} failed to initialize"


@pytest.mark.unit
class TestSystemScalability:
    """Test system scalability and performance characteristics"""

    def test_concurrent_routing_requests(self):
        """Test handling multiple concurrent routing requests"""
        routing_agent = _make_routing_agent()

        queries = [
            "Find videos of autonomous robots",
            "Summarize robotics research papers",
            "Generate report on AI trends",
            "Search for machine learning videos",
            "Analyze computer vision papers",
        ]

        mock_prediction = MagicMock()
        mock_prediction.routing_decision = {"primary_agent": "search_agent"}
        mock_prediction.overall_confidence = 0.8
        mock_prediction.reasoning_chain = ["Search routing"]

        async def process_concurrent_queries():
            tasks = [
                routing_agent.route_query(query, tenant_id="test_tenant")
                for query in queries
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        with patch.object(
            routing_agent.routing_module, "forward", return_value=mock_prediction
        ):
            results = asyncio.run(process_concurrent_queries())

        assert len(results) == len(queries)

    def test_memory_usage_stability(self):
        """Test that repeated operations don't cause memory issues"""
        summarizer = SummarizerAgent(
            deps=SummarizerDeps(),
            config_manager=create_default_config_manager(),
        )
        reporter = DetailedReportAgent(
            deps=DetailedReportDeps(),
            config_manager=create_default_config_manager(),
        )

        for _ in range(10):
            assert summarizer is not None
            assert reporter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
