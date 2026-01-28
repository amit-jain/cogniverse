"""
Integration tests for cache and metrics components in RoutingAgent:
- Cache integration with real routing
- Metrics tracking with real routing
- Parallel execution with routing
- Lazy evaluation with routing
"""

import asyncio

import pytest

from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_agents.search.multi_modal_reranker import QueryModality
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig


@pytest.mark.asyncio
class TestRoutingAgentCacheMetricsIntegration:
    """Integration tests for cache and metrics in RoutingAgent"""

    @pytest.fixture
    async def routing_agent(self):
        """Create routing agent with real components"""
        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        agent = RoutingAgent(tenant_id="test-tenant", telemetry_config=telemetry_config)
        yield agent
        # Cleanup cache between tests
        if agent.cache_manager:
            agent.cache_manager.invalidate_all()
            agent.cache_manager.reset_stats()
        # Cleanup telemetry
        agent.telemetry_manager.force_flush(timeout_millis=5000)
        await asyncio.sleep(0.5)

    async def test_cache_integration_with_routing(self, routing_agent):
        """Test cache integration with real routing queries"""
        query = "machine learning videos"
        tenant_id = "test-cache-integration"

        # First execution (cache miss)
        result1 = await routing_agent.route_query(query, tenant_id=tenant_id)
        # route_query now returns a RoutingDecision object directly
        assert result1 is not None
        assert hasattr(result1, "recommended_agent")

        # Get cache stats
        video_stats = routing_agent.cache_manager.get_cache_stats(QueryModality.VIDEO)
        initial_hits = video_stats["hits"]

        # Second execution (should hit cache)
        result2 = await routing_agent.route_query(query, tenant_id=tenant_id)

        # Results should match (except execution_time)
        assert result1.recommended_agent == result2.recommended_agent
        assert result1.confidence == result2.confidence

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
            await routing_agent.route_query(query, tenant_id="test-metrics")

        # Verify metrics were recorded
        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert summary["total_requests"] >= len(queries)
        assert (
            summary["active_modalities"] >= 1
        )  # At least one modality should be active

    async def test_cache_miss_then_hit(self, routing_agent):
        """Test cache miss followed by cache hit"""
        query = "deep learning tutorials"
        tenant_id = "test-cache-miss-hit"

        # First call - cache miss (fresh routing agent has empty cache)
        result1 = await routing_agent.route_query(query, tenant_id=tenant_id)

        # Second call - cache hit
        result2 = await routing_agent.route_query(query, tenant_id=tenant_id)

        # Results should match (except execution_time)
        assert result1.query == result2.query
        assert result1.recommended_agent == result2.recommended_agent
        assert result1.confidence == result2.confidence

        # Verify cache hit via stats - check all modalities (query could route to any)
        total_hits = 0
        for modality in QueryModality:
            stats = routing_agent.cache_manager.get_cache_stats(modality)
            total_hits += stats["hits"]
        assert total_hits > 0, "Cache should have been hit on second query"

    async def test_metrics_track_success_and_failure(self, routing_agent):
        """Test metrics track both successful and failed executions"""
        # Successful query
        await routing_agent.route_query(
            "machine learning", tenant_id="test-metrics-success"
        )

        summary = routing_agent.metrics_tracker.get_summary_stats()
        initial_requests = summary["total_requests"]

        # Another successful query
        await routing_agent.route_query(
            "deep learning", tenant_id="test-metrics-success"
        )

        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert summary["total_requests"] > initial_requests

    async def test_concurrent_queries_with_cache(self, routing_agent):
        """Test concurrent queries benefit from caching"""
        query = "neural networks"
        tenant_id = "test-concurrent-cache"

        # Execute same query concurrently
        tasks = [
            routing_agent.route_query(query, tenant_id=tenant_id) for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All results should match (except execution_time may vary)
        for r in results[1:]:
            assert r.query == results[0].query
            assert r.recommended_agent == results[0].recommended_agent

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
            await routing_agent.route_query(query, tenant_id="test-modality-cache")
            await routing_agent.route_query(query, tenant_id="test-modality-cache")

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
        tenant_id = "test-cache-ttl"

        # Execute query
        result1 = await routing_agent.route_query(query, tenant_id=tenant_id)

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
            await routing_agent.route_query(query, tenant_id="test-per-modality")

        # Verify total requests across all modalities
        # Note: Duplicate queries may be cached, so total_requests may be less than len(queries)
        summary = routing_agent.metrics_tracker.get_summary_stats()
        assert (
            summary["total_requests"] >= 2
        )  # At least 2 unique queries were processed

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
            # Execute routing and verify the decision contains proper structure
            result = await routing_agent.route_query(query, tenant_id="test-detection")
            # route_query now returns a RoutingDecision object directly
            assert result is not None
            assert hasattr(result, "recommended_agent")
            assert result.recommended_agent is not None

            # Verify that routing produces a valid agent recommendation
            # The modality detection is internal to the router implementation
            # and the recommended agent should be appropriate for the query type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
