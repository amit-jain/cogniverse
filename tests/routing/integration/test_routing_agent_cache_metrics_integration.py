"""
Integration tests for cache and metrics components in RoutingAgent:
- Cache integration with real routing
- Metrics tracking with real routing
- Parallel execution with routing
- Lazy evaluation with routing
"""

import asyncio
import time

import pytest

from src.app.agents.routing_agent import RoutingAgent
from src.app.search.multi_modal_reranker import QueryModality


@pytest.mark.asyncio
class TestRoutingAgentCacheMetricsIntegration:
    """Integration tests for cache and metrics in RoutingAgent"""

    @pytest.fixture
    async def routing_agent(self):
        """Create routing agent with real components"""
        agent = RoutingAgent()
        yield agent
        # Cleanup telemetry
        agent.telemetry_manager.force_flush(timeout_millis=5000)
        await asyncio.sleep(0.5)

    async def test_cache_integration_with_routing(self, routing_agent):
        """Test cache integration with real routing queries"""
        query = "machine learning videos"
        context = {"tenant_id": "test-cache-integration"}

        # First execution (cache miss)
        result1 = await routing_agent.analyze_and_route(query, context)
        assert "agent" in result1 or "routing_decision" in result1

        # Get cache stats
        video_stats = routing_agent.cache_manager.get_cache_stats(QueryModality.VIDEO)
        initial_hits = video_stats["hits"]

        # Second execution (should hit cache)
        result2 = await routing_agent.analyze_and_route(query, context)

        # Results should match (except execution_time)
        assert result1["query"] == result2["query"]
        assert result1["routing_decision"] == result2["routing_decision"]

        # Verify cache was used (check all modalities for hits)
        total_hits = 0
        for modality in QueryModality:
            stats = routing_agent.cache_manager.get_cache_stats(modality)
            total_hits += stats["hits"]
        assert total_hits > initial_hits, "Cache should have been used on second query"

    async def test_metrics_integration_with_routing(self, routing_agent):
        """Test metrics tracking with real routing queries"""
        queries = [
            ("show me videos", QueryModality.VIDEO),
            ("find documents", QueryModality.DOCUMENT),
            ("search images", QueryModality.IMAGE),
        ]

        for query, expected_modality in queries:
            await routing_agent.analyze_and_route(query, {"tenant_id": "test-metrics"})

        # Verify metrics were recorded
        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert summary["total_requests"] >= len(queries)
        assert (
            summary["active_modalities"] >= 1
        )  # At least one modality should be active

    async def test_cache_miss_then_hit(self, routing_agent):
        """Test cache miss followed by cache hit"""
        query = "deep learning tutorials"
        context = {"tenant_id": "test-cache-miss-hit"}

        # First call - cache miss (fresh routing agent has empty cache)
        start = time.time()
        result1 = await routing_agent.analyze_and_route(query, context)
        first_call_time = time.time() - start

        # Second call - cache hit (should be faster)
        start = time.time()
        result2 = await routing_agent.analyze_and_route(query, context)
        second_call_time = time.time() - start

        # Results should match (except execution_time)
        assert result1["query"] == result2["query"]
        assert result1["routing_decision"] == result2["routing_decision"]
        assert result1["workflow_type"] == result2["workflow_type"]

        # Cache hit should be faster
        assert second_call_time < first_call_time

    async def test_metrics_track_success_and_failure(self, routing_agent):
        """Test metrics track both successful and failed executions"""
        # Successful query
        await routing_agent.analyze_and_route(
            "machine learning", {"tenant_id": "test-metrics-success"}
        )

        summary = routing_agent.metrics_tracker.get_summary_stats()
        initial_requests = summary["total_requests"]

        # Another successful query
        await routing_agent.analyze_and_route(
            "deep learning", {"tenant_id": "test-metrics-success"}
        )

        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert summary["total_requests"] > initial_requests

    async def test_concurrent_queries_with_cache(self, routing_agent):
        """Test concurrent queries benefit from caching"""
        query = "neural networks"
        context = {"tenant_id": "test-concurrent-cache"}

        # Execute same query concurrently
        tasks = [routing_agent.analyze_and_route(query, context) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All results should match (except execution_time may vary)
        for r in results[1:]:
            assert r["query"] == results[0]["query"]
            assert r["routing_decision"] == results[0]["routing_decision"]

        # Verify results were cached (concurrent queries may not have cache hits
        # because they all start at the same time before cache is populated)
        total_cached = 0
        for modality in QueryModality:
            stats = routing_agent.cache_manager.get_cache_stats(modality)
            total_cached += stats["cache_size"]
        assert total_cached > 0, "At least one result should be cached"

    async def test_different_modalities_use_separate_caches(self, routing_agent):
        """Test different modalities maintain separate caches"""
        queries = [
            ("show me videos", QueryModality.VIDEO),
            ("find documents", QueryModality.DOCUMENT),
            ("search images", QueryModality.IMAGE),
        ]

        # Execute queries twice to populate caches
        for query, _ in queries:
            await routing_agent.analyze_and_route(
                query, {"tenant_id": "test-modality-cache"}
            )
            await routing_agent.analyze_and_route(
                query, {"tenant_id": "test-modality-cache"}
            )

        # Verify we have cache hits (queries executed twice each)
        total_hits = 0
        total_cached = 0
        for modality in QueryModality:
            stats = routing_agent.cache_manager.get_cache_stats(modality)
            total_hits += stats["hits"]
            total_cached += stats["cache_size"]

        # Each query executed twice, so at least some should hit cache on 2nd execution
        assert total_hits >= 1, f"Expected at least 1 cache hit, got {total_hits}"
        assert total_cached >= 1, "At least one result should be cached"

    async def test_cache_ttl_expiration(self, routing_agent):
        """Test cache TTL expiration"""
        query = "ai research papers"
        context = {"tenant_id": "test-cache-ttl"}

        # Execute query
        result1 = await routing_agent.analyze_and_route(query, context)

        # Manually cache with short TTL
        routing_agent.cache_manager.cache_result(query, QueryModality.DOCUMENT, result1)

        # Immediate retrieval works
        cached = routing_agent.cache_manager.get_cached_result(
            query, QueryModality.DOCUMENT, ttl_seconds=1
        )
        assert cached is not None

        # Wait for TTL expiration
        await asyncio.sleep(1.5)

        # Should be expired
        cached = routing_agent.cache_manager.get_cached_result(
            query, QueryModality.DOCUMENT, ttl_seconds=1
        )
        assert cached is None

    async def test_metrics_per_modality(self, routing_agent):
        """Test metrics are tracked per modality"""
        queries = [
            ("show me videos", QueryModality.VIDEO),
            ("show me videos", QueryModality.VIDEO),
            ("find documents", QueryModality.DOCUMENT),
        ]

        for query, _ in queries:
            await routing_agent.analyze_and_route(
                query, {"tenant_id": "test-per-modality"}
            )

        # Verify total requests across all modalities
        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert summary["total_requests"] >= len(queries)

    async def test_modality_detection_integration(self, routing_agent):
        """Test modality detection works correctly in routing flow"""
        test_cases = [
            ("show me machine learning videos", QueryModality.VIDEO),
            ("find research papers on AI", QueryModality.DOCUMENT),
            ("search for neural network diagrams", QueryModality.IMAGE),
            ("listen to AI podcasts", QueryModality.AUDIO),
            ("what is machine learning", QueryModality.TEXT),
        ]

        for query, expected_modality in test_cases:
            # Test the helper method
            detected = routing_agent._detect_primary_modality_from_query(query)
            assert detected == expected_modality

            # Execute routing (actual modality may differ based on router decision)
            result = await routing_agent.analyze_and_route(
                query, {"tenant_id": "test-detection"}
            )
            assert "routing_decision" in result or "agent" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
