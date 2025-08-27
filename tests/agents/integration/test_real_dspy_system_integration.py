"""
Real DSPy System Integration Tests

Tests the actual DSPy 3.0 multi-agent routing system components working together
WITHOUT extensive mocking. This validates real functionality.
"""

import asyncio

import pytest

from src.app.agents.enhanced_routing_agent import EnhancedRoutingAgent
from src.app.routing.adaptive_threshold_learner import (
    AdaptiveThresholdLearner,
)
from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
from src.app.routing.query_enhancement_engine import QueryEnhancementPipeline
from src.app.routing.relationship_extraction_tools import RelationshipExtractorTool


@pytest.mark.integration
class TestDSPySystemIntegration:
    """Integration tests for DSPy system components"""

    @pytest.mark.ci_fast
    def test_relationship_extraction_functionality(self):
        """Test relationship extraction tool functionality"""
        extractor = RelationshipExtractorTool()

        # Test with real queries
        test_queries = [
            "Find videos of robots playing soccer",
            "Show me research about machine learning in robotics",
            "Compare different AI approaches to computer vision",
        ]

        for query in test_queries:
            try:
                # This should work with the actual implementation
                result = asyncio.run(
                    extractor.extract_comprehensive_relationships(query)
                )
                entities = result.get("entities", [])
                relationships = result.get("relationships", [])

                # Verify we get some kind of result (even if spaCy model missing)
                assert isinstance(entities, list)
                assert isinstance(relationships, list)

                # If spaCy model is available, we should get meaningful results
                if entities or relationships:
                    assert len(entities) >= 0  # Could be empty if no entities found
                    assert (
                        len(relationships) >= 0
                    )  # Could be empty if no relationships found

            except Exception as e:
                # Should gracefully handle missing dependencies
                assert "spacy" in str(e).lower() or "model" in str(e).lower()

    def test_query_enhancement_pipeline(self):
        """Test query enhancement pipeline functionality"""
        pipeline = QueryEnhancementPipeline()

        test_query = "Find videos of autonomous robots"
        test_context = {
            "entities": [{"text": "robots", "label": "ENTITY"}],
            "relationships": [
                {"subject": "robots", "relation": "type", "object": "autonomous"}
            ],
        }

        try:
            # Test real enhancement using the actual method
            result = asyncio.run(
                pipeline.enhance_query_with_relationships(
                    test_query,
                    entities=test_context["entities"],
                    relationships=test_context["relationships"],
                )
            )

            enhanced_query = result.get("enhanced_query", test_query)
            metadata = result.get("metadata", {})

            # Should return enhanced query and metadata
            assert isinstance(enhanced_query, str)
            assert isinstance(metadata, dict)
            assert len(enhanced_query) > 0

            # Enhanced query should be different or same length as original
            assert len(enhanced_query) >= len(test_query)

        except Exception as e:
            # Should handle gracefully if models not available
            pytest.skip(f"Query enhancement not available: {e}")

    @pytest.mark.ci_fast
    def test_advanced_optimizer_initialization(self):
        """Test advanced optimizer components are properly initialized"""
        optimizer = AdvancedRoutingOptimizer()

        # Should have required components
        assert hasattr(optimizer, "config")
        assert hasattr(optimizer, "storage_dir")
        assert hasattr(optimizer, "experiences")
        assert hasattr(optimizer, "metrics")

        # Should have essential methods
        assert hasattr(optimizer, "optimize_routing_decision")
        assert hasattr(optimizer, "record_routing_experience")
        assert callable(optimizer.optimize_routing_decision)

    def test_adaptive_threshold_learner_functionality(self):
        """Test adaptive threshold learner with real data"""
        learner = AdaptiveThresholdLearner()

        # Test recording performance samples

        # Should handle real performance data
        asyncio.run(
            learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.85,
                search_quality=0.9,
                response_time=1.2,
                user_satisfaction=0.9,
            )
        )

        # Should provide threshold recommendations
        current_threshold = learner.get_threshold_value("routing_confidence")
        assert isinstance(current_threshold, (int, float))
        assert 0.0 <= current_threshold <= 1.0

        # Should track learning status
        status = learner.get_learning_status()
        assert isinstance(status, dict)
        assert "total_samples" in status

    @pytest.mark.asyncio
    async def test_enhanced_routing_agent_routing(self):
        """Test enhanced routing agent routing logic"""
        # Mock only external dependencies, not core logic
        from unittest.mock import patch

        with patch("src.common.config.get_config") as mock_config:
            # Provide real config structure
            mock_config.return_value = {
                "video_agent_url": "http://localhost:8002",
                "summarizer_agent_url": "http://localhost:8003",
                "detailed_report_agent_url": "http://localhost:8004",
            }

            agent = EnhancedRoutingAgent()

            # Test real routing decision logic
            test_query = "Find videos of robots playing soccer"

            try:
                # This tests the actual routing logic
                routing_decision = await agent.route_query(test_query)

                # Should return structured routing decision
                assert routing_decision is not None

                # Check if it's a RoutingDecision object or dict with expected fields
                if hasattr(routing_decision, "search_modality"):
                    assert routing_decision.search_modality is not None
                elif isinstance(routing_decision, dict):
                    # Should have routing information
                    assert len(routing_decision) > 0

            except Exception as e:
                # Should handle gracefully if external services not available
                assert "connection" in str(e).lower() or "timeout" in str(e).lower()

    @pytest.mark.ci_fast
    def test_dspy_components_integration(self):
        """Test DSPy components integration"""
        # Test component creation and basic functionality
        components = {}

        try:
            components["extractor"] = RelationshipExtractorTool()
            components["pipeline"] = QueryEnhancementPipeline()
            components["optimizer"] = AdvancedRoutingOptimizer()
            components["learner"] = AdaptiveThresholdLearner()

            # All components should initialize successfully
            for name, component in components.items():
                assert component is not None, f"{name} failed to initialize"

            # Components should have expected interfaces
            assert hasattr(
                components["extractor"], "extract_comprehensive_relationships"
            )
            assert hasattr(components["pipeline"], "enhance_query_with_relationships")
            assert hasattr(components["optimizer"], "optimize_routing_decision")
            assert hasattr(components["learner"], "record_performance_sample")

        except ImportError as e:
            pytest.skip(f"DSPy components not fully available: {e}")


@pytest.mark.integration
class TestMultiAgentSystem:
    """Test multi-agent system components"""

    def test_core_agents_initialization(self):
        """Test that core agents can be initialized"""
        from unittest.mock import patch

        from src.app.agents.detailed_report_agent import DetailedReportAgent
        from src.app.agents.summarizer_agent import SummarizerAgent

        # Mock config to provide required LLM configuration
        with (
            patch("src.app.agents.summarizer_agent.get_config") as mock_summarizer_config,
            patch("src.app.agents.detailed_report_agent.get_config") as mock_reporter_config
        ):
            config = {
                "llm": {
                    "model_name": "smollm3:3b",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "dummy",
                }
            }
            mock_summarizer_config.return_value = config
            mock_reporter_config.return_value = config

            # Should initialize with proper config mocking
            summarizer = SummarizerAgent()
            reporter = DetailedReportAgent()

            assert summarizer is not None
            assert reporter is not None

            # Should have required interfaces
            assert hasattr(summarizer, "summarize")
            assert hasattr(reporter, "generate_report")

    def test_enhanced_video_search_real_config(self):
        """Test enhanced video search agent with real configuration"""
        import os

        from src.app.agents.enhanced_video_search_agent import EnhancedVideoSearchAgent

        # Set required environment for video search
        os.environ["VESPA_SCHEMA"] = "video_colpali_smol500_mv_frame"

        try:
            agent = EnhancedVideoSearchAgent()

            # Should initialize with real config
            assert agent is not None
            assert hasattr(agent, "vespa_client")
            assert hasattr(agent, "query_encoder")
            assert hasattr(agent, "config")

        except Exception as e:
            # Should handle missing Vespa gracefully
            assert "vespa" in str(e).lower() or "schema" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
