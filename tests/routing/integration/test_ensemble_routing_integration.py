"""
Integration tests for ensemble routing with real local models.
Tests the complete ensemble routing flow with actual model inference.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.routing.base import GenerationType, SearchModality
from src.app.routing.router import ComprehensiveRouter


@pytest.fixture
def ensemble_test_config():
    """Load ensemble test configuration with real models."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            full_config = json.load(f)
            # Extract routing section
            if "routing" in full_config:
                base_config = full_config["routing"]
                # Add ensemble configuration
                base_config["ensemble_config"] = {
                    "enabled": True,
                    "enabled_strategies": ["gliner", "keyword"],
                    "voting_method": "weighted",
                    "strategy_weights": {"gliner": 2.0, "keyword": 1.0},
                    "min_agreement": 0.5,
                    "timeout_seconds": 30.0,
                }
                return base_config

    # Fallback config for CI/CD when config.json doesn't exist
    return {
        "ensemble_config": {
            "enabled": True,
            "enabled_strategies": ["gliner", "keyword"],
            "voting_method": "weighted",
            "strategy_weights": {"gliner": 2.0, "keyword": 1.0},
            "min_agreement": 0.5,
            "timeout_seconds": 30.0,
        },
        "tier_config": {
            "enable_fast_path": True,
            "enable_slow_path": False,  # Disable to avoid Ollama dependency in CI
            "enable_langextract": False,  # Disable to avoid Ollama dependency in CI
            "enable_fallback": True,
        },
        "gliner_config": {
            "gliner_model": "urchade/gliner_multi-v2.1",
            "gliner_threshold": 0.3,
        },
        "keyword_config": {},
    }


@pytest.fixture
def ensemble_router(ensemble_test_config):
    """Create a comprehensive router instance with ensemble configuration."""
    return ComprehensiveRouter(ensemble_test_config)


class TestEnsembleWithRealModels:
    """Test ensemble routing with real model inference."""

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_routing_simple_query(self, ensemble_router):
        """Test ensemble routing with a simple video query using real models."""
        decision = await ensemble_router.route("show me videos about cats")

        # Verify ensemble routing was used
        assert decision.routing_method == "ensemble"
        assert "voting_method" in decision.metadata
        assert decision.metadata["voting_method"] == "weighted"
        assert "strategies_used" in decision.metadata

        # Should have high confidence from ensemble voting
        assert decision.confidence_score > 0.0

        # GLiNER should contribute to the decision
        strategies_used = decision.metadata.get("strategies_used", [])
        assert len(strategies_used) > 0

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_voting_weighted_decision(self, ensemble_router):
        """Test that ensemble weighted voting produces correct decisions."""
        # Test with a video-focused query that should favor GLiNER
        video_decision = await ensemble_router.route("find videos of dogs playing")

        assert video_decision.routing_method == "ensemble"
        assert video_decision.search_modality in [
            SearchModality.VIDEO,
            SearchModality.BOTH,
        ]

        # Test with a general query
        general_decision = await ensemble_router.route(
            "search for information about machine learning"
        )

        assert general_decision.routing_method == "ensemble"
        assert general_decision.search_modality in [
            SearchModality.TEXT,
            SearchModality.BOTH,
            SearchModality.VIDEO,
        ]

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_strategy_failure_handling(self, ensemble_router):
        """Test ensemble handles individual strategy failures gracefully."""
        # Use a complex query that might cause some strategies to struggle
        decision = await ensemble_router.route(
            "extremely complex multi-layered query with unusual terminology"
        )

        # Should still get a decision even if some strategies fail
        assert decision is not None
        assert decision.routing_method == "ensemble"

        # Should have at least one successful strategy
        strategies_used = decision.metadata.get("strategies_used", [])
        assert len(strategies_used) > 0

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_confidence_aggregation(self, ensemble_router):
        """Test that ensemble properly aggregates confidence scores."""
        queries = [
            "show videos",  # Simple, high confidence expected
            "find content about data science machine learning artificial intelligence neural networks",  # Complex
            "cats",  # Very simple
        ]

        decisions = []
        for query in queries:
            decision = await ensemble_router.route(query)
            decisions.append(decision)

        # All should use ensemble routing
        for decision in decisions:
            assert decision.routing_method == "ensemble"
            assert decision.confidence_score > 0.0
            assert "voting_method" in decision.metadata

        # Confidence scores should be reasonable (not all the same)
        confidence_scores = [d.confidence_score for d in decisions]
        assert len(set(confidence_scores)) > 1 or all(
            0.3 <= score <= 1.0 for score in confidence_scores
        )

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_vs_tiered_performance(self, ensemble_test_config):
        """Compare ensemble routing performance vs tiered routing."""
        # Create both router types
        ensemble_config = ensemble_test_config.copy()

        tiered_config = ensemble_test_config.copy()
        tiered_config.pop(
            "ensemble_config", None
        )  # Remove ensemble config for tiered mode

        ensemble_router = ComprehensiveRouter(ensemble_config)
        tiered_router = ComprehensiveRouter(tiered_config)

        test_query = "show me videos about technology"

        # Route with ensemble
        ensemble_decision = await ensemble_router.route(test_query)

        # Route with tiered
        tiered_decision = await tiered_router.route(test_query)

        # Both should produce valid decisions
        assert ensemble_decision is not None
        assert tiered_decision is not None

        # Ensemble should have ensemble metadata
        assert ensemble_decision.routing_method == "ensemble"
        assert "voting_method" in ensemble_decision.metadata

        # Tiered should use single strategy
        assert tiered_decision.routing_method != "ensemble"

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_timeout_handling(self):
        """Test ensemble handles timeouts gracefully."""
        # Create config with very short timeout
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["gliner", "keyword"],
                "voting_method": "weighted",
                "timeout_seconds": 0.1,  # Very short timeout
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_fallback": True,
            },
            "gliner_config": {
                "gliner_model": "urchade/gliner_multi-v2.1",
                "gliner_threshold": 0.3,
            },
            "keyword_config": {},
        }

        router = ComprehensiveRouter(config)

        # This should timeout and fall back to tiered routing
        decision = await router.route("test query for timeout handling")

        # Should still get a decision (fallback to tiered)
        assert decision is not None
        # Might be ensemble if it completed quickly, or tiered if it timed out
        assert decision.routing_method in ["ensemble", "gliner", "keyword"]


class TestEnsembleConfigurationValidation:
    """Test ensemble configuration validation with real router initialization."""

    @pytest.mark.integration
    @pytest.mark.requires_models
    def test_invalid_ensemble_config_fails_initialization(self):
        """Test that invalid ensemble configs prevent router initialization."""
        invalid_config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": [],  # Empty list should fail
            }
        }

        with pytest.raises(
            ValueError, match="ensemble_config.enabled_strategies cannot be empty"
        ):
            ComprehensiveRouter(invalid_config)

    @pytest.mark.integration
    @pytest.mark.requires_models
    def test_valid_ensemble_config_succeeds_initialization(self):
        """Test that valid ensemble configs allow router initialization."""
        valid_config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["keyword"],
                "voting_method": "majority",
            },
            "keyword_config": {},
        }

        # Should not raise any exceptions
        router = ComprehensiveRouter(valid_config)
        assert router is not None

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_with_all_strategies(self):
        """Test ensemble routing with ALL available strategies including real LLM."""
        # Force ALL strategies to be tested - no wimpy CI fallbacks
        strategies = ["gliner", "llm", "keyword"]

        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": strategies,
                "voting_method": "weighted",
                "strategy_weights": {"gliner": 2.0, "llm": 1.5, "keyword": 1.0},
                "timeout_seconds": 60.0,
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,  # ENABLE LLM TIER
                "enable_langextract": False,
                "enable_fallback": True,
            },
            "gliner_config": {
                "gliner_model": "urchade/gliner_multi-v2.1",
                "gliner_threshold": 0.3,
            },
            "llm_config": {
                "provider": "local",
                "model": "smollm2:1.7b",
                "endpoint": "http://localhost:11434",
            },
            "keyword_config": {},
        }

        router = ComprehensiveRouter(config)
        decision = await router.route(
            "find videos about machine learning and AI tutorials"
        )

        assert decision.routing_method == "ensemble"
        assert decision.metadata["voting_method"] == "weighted"

        strategies_used = decision.metadata.get("strategies_used", [])
        print(f"🎯 Strategies that actually ran: {strategies_used}")

        # ALL THREE strategies must have run successfully
        assert "gliner" in strategies_used, "GLiNER should have participated"
        assert "llm" in strategies_used, "LLM should have participated via Ollama"
        assert "keyword" in strategies_used, "Keyword should have participated"
        assert (
            len(strategies_used) == 3
        ), f"Expected 3 strategies, got {len(strategies_used)}"

        # Should have real confidence from weighted voting
        assert decision.confidence_score > 0.0
        assert decision.search_modality is not None

        # Reasoning should show all strategy contributions
        assert "gliner" in decision.reasoning.lower()
        assert "llm" in decision.reasoning.lower()
        assert "keyword" in decision.reasoning.lower()


class TestEnsembleRealWorldScenarios:
    """Test ensemble routing with real-world query scenarios."""

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_video_search_queries(self, ensemble_router):
        """Test ensemble routing with various video search queries."""
        video_queries = [
            "show me videos about cooking",
            "find videos of people dancing",
            "videos about machine learning tutorials",
            "show cooking videos with pasta",
            "find dance performance videos",
        ]

        for query in video_queries:
            decision = await ensemble_router.route(query)

            assert decision.routing_method == "ensemble"
            assert decision.search_modality in [
                SearchModality.VIDEO,
                SearchModality.BOTH,
            ]
            assert decision.confidence_score > 0.0

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_mixed_content_queries(self, ensemble_router):
        """Test ensemble routing with queries requiring mixed content."""
        mixed_queries = [
            "compare videos and articles about climate change",
            "find both video tutorials and written guides for programming",
            "show me videos and text about data science",
        ]

        for query in mixed_queries:
            decision = await ensemble_router.route(query)

            assert decision.routing_method == "ensemble"
            # Mixed queries should often result in BOTH modality
            assert decision.search_modality in [
                SearchModality.BOTH,
                SearchModality.VIDEO,
                SearchModality.TEXT,
            ]

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_concurrent_ensemble_requests(self, ensemble_router):
        """Test ensemble routing with concurrent requests."""
        queries = [
            "videos about cats",
            "articles about dogs",
            "tutorials on python",
            "cooking recipes",
            "dance performances",
        ]

        # Route all queries concurrently
        tasks = [ensemble_router.route(query) for query in queries]
        decisions = await asyncio.gather(*tasks)

        assert len(decisions) == len(queries)

        # All should use ensemble routing
        for decision in decisions:
            assert decision is not None
            assert decision.routing_method == "ensemble"
            assert decision.confidence_score > 0.0


class TestLLMStrictParsing:
    """Test strict LLM parsing without fallbacks."""

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_llm_strict_json_validation(self):
        """Test that LLM parsing strictly validates JSON responses."""
        from src.app.routing.strategies import LLMRoutingStrategy

        config = {
            "provider": "local",
            "model": "smollm2:1.7b",
            "endpoint": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 150,
        }

        llm_strategy = LLMRoutingStrategy(config)
        query = "show me videos about cats"

        # Test valid query works
        decision = await llm_strategy.route(query, None)
        assert decision.routing_method == "llm"  # Not "llm_parsed"
        assert decision.search_modality in [
            SearchModality.VIDEO,
            SearchModality.TEXT,
            SearchModality.BOTH,
        ]
        assert decision.generation_type in [
            GenerationType.RAW_RESULTS,
            GenerationType.SUMMARY,
            GenerationType.DETAILED_REPORT,
        ]

    @pytest.mark.integration
    @pytest.mark.requires_models
    def test_llm_rejects_invalid_responses(self):
        """Test that LLM parsing rejects invalid responses without fallback."""
        from src.app.routing.strategies import LLMRoutingStrategy

        config = {
            "provider": "local",
            "model": "smollm2:1.7b",
            "endpoint": "http://localhost:11434",
        }

        llm_strategy = LLMRoutingStrategy(config)
        query = "test query"

        # Test 1: No JSON found
        with pytest.raises(ValueError, match="No JSON found in LLM response"):
            llm_strategy._parse_llm_response("video|text|both", query)

        # Test 2: Invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON in LLM response"):
            llm_strategy._parse_llm_response('{"invalid": json}', query)

        # Test 3: Missing required fields
        with pytest.raises(
            ValueError, match="Missing 'generation_type' in LLM response"
        ):
            llm_strategy._parse_llm_response('{"search_modality": "video"}', query)

        # Test 4: Invalid enum values
        with pytest.raises(ValueError, match="Invalid search_modality"):
            llm_strategy._parse_llm_response(
                '{"search_modality": "invalid", "generation_type": "raw_results"}',
                query,
            )

    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_ensemble_with_strict_llm_parsing(self):
        """Test that ensemble routing works correctly with strict LLM parsing."""
        from src.app.routing import ComprehensiveRouter

        # Force ALL strategies including real LLM
        config = {
            "ensemble_config": {
                "enabled": True,
                "enabled_strategies": ["gliner", "llm", "keyword"],
                "voting_method": "weighted",
                "strategy_weights": {"gliner": 2.0, "llm": 1.5, "keyword": 1.0},
                "timeout_seconds": 60.0,
                "min_agreement": 0.5,
            },
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,  # ENABLE LLM TIER
                "enable_langextract": False,
                "enable_fallback": True,
            },
            "gliner_config": {
                "gliner_model": "urchade/gliner_multi-v2.1",
                "gliner_threshold": 0.3,
            },
            "llm_config": {
                "provider": "local",
                "model": "smollm2:1.7b",
                "endpoint": "http://localhost:11434",
            },
        }

        router = ComprehensiveRouter(config)

        # This query should escalate to LLM tier in ensemble
        query = "compare the video analysis with the text summary"

        decision = await router.route(query)

        # Ensemble should succeed with properly parsed LLM
        assert decision.routing_method == "ensemble"
        assert decision.search_modality == SearchModality.BOTH
        # Should detect this as a detailed report request due to "compare" keyword
        assert decision.generation_type == GenerationType.DETAILED_REPORT
        assert decision.confidence_score > 0.0

        # Reasoning should include LLM contribution (not llm_parsed fallback)
        assert "llm:" in decision.reasoning.lower()
        # Should NOT contain fallback indicators
        assert "llm_parsed" not in decision.reasoning.lower()
        assert "parsed from non-json" not in decision.reasoning.lower()

    @pytest.mark.integration
    @pytest.mark.requires_models
    def test_llm_dspy_optimization_integration(self):
        """Test that LLM strategy integrates with DSPy optimization system."""
        from src.app.routing.strategies import LLMRoutingStrategy

        config = {
            "provider": "local",
            "model": "smollm2:1.7b",
            "endpoint": "http://localhost:11434",
            "enable_dspy_optimization": True,  # Enable DSPy
        }

        llm_strategy = LLMRoutingStrategy(config)

        # Test optimization status
        status = llm_strategy.get_optimization_status()
        assert status["dspy_enabled"] is True
        assert "optimized_prompts_loaded" in status
        assert "available_optimizations" in status
        assert "using_optimized_system_prompt" in status

        # Test runtime control
        llm_strategy.enable_dspy_optimization(False)
        status_disabled = llm_strategy.get_optimization_status()
        assert status_disabled["dspy_enabled"] is False

        llm_strategy.enable_dspy_optimization(True)
        status_enabled = llm_strategy.get_optimization_status()
        assert status_enabled["dspy_enabled"] is True


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-k", "test_ensemble_routing_simple_query"])
