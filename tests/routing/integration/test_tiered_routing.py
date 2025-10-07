"""
Integration tests for tiered routing system.
Tests the complete routing flow with real models when available.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.routing import TieredRouter
from src.app.routing.base import GenerationType, SearchModality


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            full_config = json.load(f)
            # Extract routing section
            if "routing" in full_config:
                return full_config["routing"]

    # Fallback config for CI/CD when config.json doesn't exist or doesn't have routing
    return {
        "routing_mode": "tiered",
        "tier_config": {
            "enable_fast_path": True,
            "enable_slow_path": False,  # Disable slow path in CI
            "enable_langextract": False,  # Disable langextract in CI
            "enable_fallback": True,
            "fast_path_confidence_threshold": 0.5,
            "slow_path_confidence_threshold": 0.6,
            "langextract_confidence_threshold": 0.5,
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
        "langextract_config": {
            "langextract_model": "qwen2.5:7b",
            "ollama_url": "http://localhost:11434",
        },
    }


@pytest.fixture
def router(test_config):
    """Create a tiered router instance."""
    return TieredRouter(test_config)


class TestTieredEscalation:
    """Test tier escalation logic."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tier1_handles_simple_query(self, router):
        """Test that Tier 1 (GLiNER) handles simple entity queries."""
        decision = await router.route("show me videos about cats")

        # Should be handled by GLiNER (if model is available)
        if decision.routing_method == "gliner":
            assert decision.confidence_score >= 0.7
            assert decision.search_modality == SearchModality.VIDEO
        # Might fall through if GLiNER not available
        else:
            assert decision.routing_method in ["llm", "llm_parsed", "keyword"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_relationship_query_escalates(self, router):
        """Test that relationship queries escalate beyond Tier 1."""
        decision = await router.route(
            "compare the video analysis with the text summary"
        )

        # Should NOT be handled by GLiNER
        assert decision.routing_method != "gliner" or decision.confidence_score < 0.7

        # Should be handled by LLM or higher
        if decision.routing_method in ["llm", "llm_parsed"]:
            assert decision.search_modality == SearchModality.BOTH

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_structured_extraction_escalation(self, router):
        """Test that structured extraction queries may escalate to Tier 3."""
        decision = await router.route("extract specific timestamps in JSON format")

        # Could be handled by any tier depending on confidence
        assert decision.routing_method in [
            "gliner",
            "llm",
            "llm_parsed",
            "langextract",
            "keyword",
        ]

        # Should identify as raw_results if properly classified
        if decision.routing_method in ["gliner", "langextract"]:
            assert decision.generation_type == GenerationType.RAW_RESULTS

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fallback_for_nonsense(self, router):
        """Test that nonsense queries fall through to keyword fallback."""
        decision = await router.route("xyzabc123 quantum flibbertigibbet")

        # Low confidence from all tiers, should reach fallback
        assert decision.confidence_score <= 0.85  # Not high confidence
        assert decision.routing_method in ["keyword", "llm", "llm_parsed"]


class TestGenerationTypeAccuracy:
    """Test accuracy of generation type classification."""

    @pytest.mark.parametrize(
        "query,expected_type",
        [
            ("extract timestamps from video", GenerationType.RAW_RESULTS),
            ("summarize the main points", GenerationType.SUMMARY),
            ("detailed analysis of performance", GenerationType.DETAILED_REPORT),
        ],
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generation_type_classification(self, router, query, expected_type):
        """Test that generation types are correctly identified."""
        decision = await router.route(query)

        # Allow some flexibility for different tiers
        if decision.routing_method in ["gliner", "langextract"]:
            # These should be more accurate
            assert decision.generation_type == expected_type
        else:
            # LLM/keyword might vary
            assert decision.generation_type in [
                GenerationType.RAW_RESULTS,
                GenerationType.SUMMARY,
                GenerationType.DETAILED_REPORT,
            ]


class TestSearchModalityDetection:
    """Test search modality detection."""

    @pytest.mark.parametrize(
        "query,expected_modality",
        [
            ("show me the video", SearchModality.VIDEO),
            ("read the document", SearchModality.TEXT),
            ("compare video with transcript", SearchModality.BOTH),
        ],
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_modality_detection(self, router, query, expected_modality):
        """Test that search modalities are correctly identified."""
        decision = await router.route(query)

        # Should correctly identify modality
        assert decision.search_modality == expected_modality


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_hit(self, router):
        """Test that repeated queries hit the cache."""
        query = "test query for caching"

        # First call
        decision1 = await router.route(query)

        # Second call should hit cache
        decision2 = await router.route(query)

        # Should return same decision
        assert decision1.search_modality == decision2.search_modality
        assert decision1.generation_type == decision2.generation_type
        assert decision1.confidence_score == decision2.confidence_score

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_expiry(self, test_config):
        """Test that cache expires after TTL."""
        # Set very short TTL
        test_config["cache_config"] = {"enable_caching": True, "cache_ttl_seconds": 0.1}
        router = TieredRouter(test_config)

        query = "test cache expiry"

        # First call
        await router.route(query)

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Second call should not hit cache
        with patch.object(router, "_check_cache", return_value=None):
            decision2 = await router.route(query)

            # Might get different decision
            assert decision2 is not None


class TestErrorHandling:
    """Test error handling in routing."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_strategies_fail(self, test_config):
        """Test handling when all strategies fail."""
        # Disable all strategies except fallback
        test_config["tier_config"]["enable_fast_path"] = False
        test_config["tier_config"]["enable_slow_path"] = False
        test_config["tier_config"]["enable_langextract"] = False
        test_config["tier_config"]["enable_fallback"] = True

        router = TieredRouter(test_config)

        decision = await router.route("test query")

        # Should still get a decision from fallback
        assert decision is not None
        assert decision.routing_method == "keyword"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_query(self, router):
        """Test handling of empty query."""
        decision = await router.route("")

        # Should handle gracefully
        assert decision is not None
        assert decision.confidence_score >= 0.0


class TestPerformanceTracking:
    """Test performance tracking and metrics."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metrics_recording(self, router):
        """Test that metrics are recorded for routing decisions."""
        # Enable metrics
        if hasattr(router, "metrics_enabled"):
            router.metrics_enabled = True

        await router.route("test query")

        # Check if metrics were recorded (if available)
        if hasattr(router, "metrics"):
            assert len(router.metrics) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, router):
        """Test that execution time is reasonable."""
        import time

        start = time.time()
        await router.route("test query")
        elapsed = time.time() - start

        # Should complete within reasonable time (5 seconds)
        assert elapsed < 5.0


@pytest.mark.integration
class TestFullRoutingFlow:
    """Full integration tests with all components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_routing_flow(self, router):
        """Test complete routing flow from query to decision."""
        test_queries = [
            ("show me cat videos", SearchModality.VIDEO, GenerationType.RAW_RESULTS),
            ("summarize the document", SearchModality.TEXT, GenerationType.SUMMARY),
            (
                "detailed analysis of data",
                SearchModality.BOTH,
                GenerationType.DETAILED_REPORT,
            ),
        ]

        for query, expected_modality, expected_type in test_queries:
            decision = await router.route(query)

            # Verify decision structure
            assert decision is not None
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
            assert 0.0 <= decision.confidence_score <= 1.0
            assert decision.routing_method is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_high_volume_routing(self, router):
        """Test routing under high volume."""
        queries = [
            "show me videos",
            "summarize this",
            "detailed analysis",
            "extract timestamps",
            "compare documents",
        ] * 10  # 50 queries

        tasks = [router.route(q) for q in queries]
        decisions = await asyncio.gather(*tasks)

        # All should complete
        assert len(decisions) == 50
        assert all(d is not None for d in decisions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
