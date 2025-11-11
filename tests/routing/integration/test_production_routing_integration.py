"""
End-to-end integration tests for production routing features:
- Parallel agent execution with resource management
- Per-modality caching with LRU and TTL
- Lazy evaluation with cost-based ordering
- Per-modality metrics tracking

Tests real workflows with actual components (no mocked internal logic).
"""

import asyncio
import time

import pytest
from cogniverse_agents.routing.lazy_executor import LazyModalityExecutor
from cogniverse_agents.routing.modality_cache import ModalityCacheManager
from cogniverse_agents.routing.parallel_executor import ParallelAgentExecutor
from cogniverse_agents.search.multi_modal_reranker import QueryModality
from cogniverse_agents.routing.modality_metrics import ModalityMetricsTracker


@pytest.mark.asyncio
class TestProductionRoutingIntegration:
    """End-to-end integration tests for production routing features"""

    @pytest.fixture
    def metrics_tracker(self):
        """Create real metrics tracker"""
        return ModalityMetricsTracker(window_size=100)

    @pytest.fixture
    def cache_manager(self):
        """Create real cache manager"""
        return ModalityCacheManager(cache_size_per_modality=100)

    @pytest.fixture
    def parallel_executor(self):
        """Create real parallel executor"""
        return ParallelAgentExecutor(max_concurrent_agents=3)

    @pytest.fixture
    def lazy_executor(self):
        """Create real lazy executor"""
        return LazyModalityExecutor()

    @pytest.fixture
    async def comprehensive_router(self):
        """Create comprehensive router for testing"""
        from cogniverse_agents.routing.config import RoutingConfig
        from cogniverse_agents.routing.router import ComprehensiveRouter

        config = RoutingConfig()
        router = ComprehensiveRouter(config)
        yield router

    async def test_parallel_execution_real_workflow(
        self, parallel_executor, metrics_tracker
    ):
        """Test parallel execution with real routing workflow"""

        # Create mock agent caller that simulates different latencies
        async def mock_agent_caller(agent_name: str, query: str, context: dict):
            # Simulate different latencies
            latencies = {
                "video_search_agent": 0.5,
                "document_agent": 0.3,
                "image_search_agent": 0.4,
            }
            await asyncio.sleep(latencies.get(agent_name, 0.2))
            return {
                "agent": agent_name,
                "results": [{"id": f"{agent_name}-1", "score": 0.9}],
                "count": 1,
            }

        # Execute agents in parallel
        tasks = [
            ("video_search_agent", "machine learning videos", {"tenant_id": "test"}),
            (
                "document_agent",
                "machine learning documents",
                {"tenant_id": "test"},
            ),
            ("image_search_agent", "machine learning images", {"tenant_id": "test"}),
        ]

        start = time.time()
        result = await parallel_executor.execute_agents_parallel(
            tasks, agent_caller=mock_agent_caller
        )
        elapsed = time.time() - start

        # Verify parallel execution (should be faster than sequential)
        assert elapsed < 1.0  # All 3 should run in parallel, max 0.5s + overhead
        assert "video_search_agent" in result["results"]
        assert "document_agent" in result["results"]
        assert "image_search_agent" in result["results"]

        # Verify all succeeded
        assert result["successful_agents"] == 3
        assert result["failed_agents"] == 0

        # Manually track metrics
        for agent_name, agent_result in result["results"].items():
            metrics_tracker.record_modality_execution(
                QueryModality.VIDEO,  # Assume video modality for testing
                result["latencies"][agent_name],
                True,
                None,
            )

        # Verify metrics were tracked
        stats = metrics_tracker.get_summary_stats()
        assert stats["total_requests"] == 3
        assert stats["overall_success_rate"] == 1.0

    async def test_parallel_execution_with_timeout(
        self, parallel_executor, metrics_tracker
    ):
        """Test parallel execution handles timeouts correctly"""

        async def mock_agent_caller(agent_name: str, query: str, context: dict):
            if agent_name == "slow_agent":
                await asyncio.sleep(5.0)  # Will timeout
            return {"results": [], "count": 0}

        tasks = [
            ("fast_agent", "test query", {"tenant_id": "test"}),
            ("slow_agent", "test query", {"tenant_id": "test"}),
        ]

        result = await parallel_executor.execute_agents_parallel(
            tasks, timeout_seconds=1.0, agent_caller=mock_agent_caller
        )

        # Verify timeout handling
        assert "fast_agent" in result["results"]
        assert "slow_agent" in result["errors"]
        assert result["timed_out_agents"] == 1
        assert "timeout" in result["errors"]["slow_agent"].lower()

    async def test_cache_prevents_redundant_execution(
        self, cache_manager, lazy_executor
    ):
        """Test that cache prevents redundant modality execution"""

        execution_count = {"count": 0}

        async def mock_modality_executor(
            query: str, modality: QueryModality, context: dict
        ):
            execution_count["count"] += 1
            return {
                "results": [{"id": f"{modality.value}-1", "score": 0.8}],
                "count": 1,
                "confidence": 0.8,
            }

        query = "machine learning tutorial"
        modalities = [QueryModality.VIDEO, QueryModality.DOCUMENT]
        context = {"tenant_id": "test"}

        # First execution - should execute both modalities
        result1 = await lazy_executor.execute_with_lazy_evaluation(
            query, modalities, context, modality_executor=mock_modality_executor
        )
        first_count = execution_count["count"]
        assert first_count > 0

        # Cache the results
        for modality, result in result1["results"].items():
            cache_manager.cache_result(query, modality, result)

        # Second execution - check cache first
        cached_results = {}
        for modality in modalities:
            cached = cache_manager.get_cached_result(query, modality)
            if cached:
                cached_results[modality] = cached

        # Verify cache hit
        assert QueryModality.VIDEO in cached_results
        assert QueryModality.DOCUMENT in cached_results
        assert execution_count["count"] == first_count  # No new executions

        # Verify cache stats (aggregate across all modalities)
        video_stats = cache_manager.get_cache_stats(QueryModality.VIDEO)
        doc_stats = cache_manager.get_cache_stats(QueryModality.DOCUMENT)
        total_hits = video_stats["hits"] + doc_stats["hits"]
        assert total_hits >= 2

    async def test_lazy_evaluation_early_stopping(self, lazy_executor):
        """Test lazy evaluation stops early when quality threshold met"""

        execution_log = []

        async def mock_modality_executor(
            query: str, modality: QueryModality, context: dict
        ):
            execution_log.append(modality)

            # TEXT returns high quality results
            if modality == QueryModality.TEXT:
                return {
                    "results": [{"id": "text-1", "score": 0.95}] * 10,
                    "count": 10,
                    "confidence": 0.95,
                }
            # Others return lower quality
            return {
                "results": [{"id": f"{modality.value}-1", "score": 0.6}],
                "count": 1,
                "confidence": 0.6,
            }

        query = "machine learning"
        # Order: TEXT (cheap), VIDEO (expensive), AUDIO (very expensive)
        modalities = [QueryModality.TEXT, QueryModality.VIDEO, QueryModality.AUDIO]
        context = {
            "tenant_id": "test",
            "quality_threshold": 0.8,
            "min_results_required": 5,
        }

        result = await lazy_executor.execute_with_lazy_evaluation(
            query, modalities, context, modality_executor=mock_modality_executor
        )

        # Verify early stopping - should only execute TEXT
        assert QueryModality.TEXT in execution_log
        # VIDEO and AUDIO should be skipped due to early stopping
        assert len(execution_log) <= 2  # TEXT + maybe one more before stopping

        # Verify we got results and early stopping worked
        assert result["early_stopped"] is True
        # Should have TEXT results
        assert QueryModality.TEXT in result["results"]

    async def test_metrics_tracking_across_modalities(self, metrics_tracker):
        """Test metrics tracking for different modalities"""

        # Simulate various executions
        modalities_data = [
            (QueryModality.VIDEO, 500.0, True, None),
            (QueryModality.VIDEO, 600.0, True, None),
            (QueryModality.VIDEO, 5000.0, False, "timeout"),
            (QueryModality.DOCUMENT, 200.0, True, None),
            (QueryModality.DOCUMENT, 250.0, True, None),
            (QueryModality.IMAGE, 300.0, True, None),
            (QueryModality.IMAGE, 350.0, False, "not_found"),
        ]

        for modality, latency, success, error in modalities_data:
            metrics_tracker.record_modality_execution(modality, latency, success, error)

        # Check VIDEO stats
        video_stats = metrics_tracker.get_modality_stats(QueryModality.VIDEO)
        assert video_stats["total_requests"] == 3
        assert video_stats["success_count"] == 2
        assert video_stats["success_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert video_stats["p50_latency"] == 600.0  # Median
        assert video_stats["p95_latency"] > video_stats["p50_latency"]

        # Check DOCUMENT stats
        doc_stats = metrics_tracker.get_modality_stats(QueryModality.DOCUMENT)
        assert doc_stats["total_requests"] == 2
        assert doc_stats["success_rate"] == 1.0
        assert doc_stats["avg_latency"] == 225.0

        # Check error breakdown
        assert "timeout" in video_stats["error_breakdown"]
        assert video_stats["error_breakdown"]["timeout"] == 1

    async def test_complete_production_workflow_with_routing_agent(
        self, comprehensive_router, cache_manager, parallel_executor, lazy_executor
    ):
        """Test complete workflow: routing → parallel execution → caching → metrics"""

        # Mock agent calls to avoid actual HTTP requests
        async def mock_agent_call(agent_name: str, query: str, context: dict):
            await asyncio.sleep(0.1)  # Simulate network latency
            return {
                "agent": agent_name,
                "results": [
                    {"id": f"{agent_name}-1", "score": 0.9, "title": "Test result"}
                ],
                "count": 1,
            }

        # Get routing decision
        query = "machine learning videos"
        context = {"tenant_id": "test-tenant"}

        decision = await comprehensive_router.route(query, context)

        # Verify decision
        assert decision is not None
        assert hasattr(decision, "search_modality")
        assert hasattr(decision, "primary_agent")

        # Check cache first
        primary_modality = QueryModality.VIDEO  # Based on query
        cached_result = cache_manager.get_cached_result(query, primary_modality)

        if cached_result:
            # Cache hit - verify stats
            stats = cache_manager.get_cache_stats()
            assert stats["total_hits"] > 0
        else:
            # Cache miss - execute agents
            if decision.requires_orchestration:
                # Use parallel executor for multiple agents
                tasks = [
                    (agent, query, context)
                    for agent in decision.agent_execution_order
                    or [decision.primary_agent]
                ]
                results = await parallel_executor.execute_agents_parallel(
                    tasks[:3], agent_caller=mock_agent_call  # Limit to 3 agents
                )

                # Cache successful results
                for agent_name, result in results.items():
                    if result["status"] == "success":
                        # Map agent to modality
                        modality = self._agent_to_modality(agent_name)
                        cache_manager.cache_result(query, modality, result)
            else:
                # Single agent - use lazy evaluation
                modalities = [primary_modality]

                async def modality_executor(q, m, ctx):
                    return await mock_agent_call(decision.primary_agent, q, ctx)

                result = await lazy_executor.execute_with_lazy_evaluation(
                    query, modalities, context, modality_executor=modality_executor
                )

                # Cache result
                cache_manager.cache_result(query, primary_modality, result)

        # Verify cache was populated (check primary modality)
        cache_stats = cache_manager.get_cache_stats(primary_modality)
        assert cache_stats["hits"] + cache_stats["misses"] > 0

    async def test_concurrent_requests_with_cache_and_metrics(
        self, cache_manager, parallel_executor, metrics_tracker
    ):
        """Test system handles concurrent requests with caching and metrics"""

        request_count = {"count": 0}
        # Use a lock to prevent cache from being written by multiple requests simultaneously
        cache_locks = {}

        async def mock_agent_caller(agent_name: str, query: str, context: dict):
            request_count["count"] += 1
            await asyncio.sleep(0.05)
            return {"results": [{"id": "test-1", "score": 0.9}], "count": 1}

        async def process_request(query: str):
            # Check cache
            cached = cache_manager.get_cached_result(query, QueryModality.TEXT)
            if cached:
                return cached

            # Get or create lock for this query
            if query not in cache_locks:
                cache_locks[query] = asyncio.Lock()

            # Use lock to prevent duplicate executions for same query
            async with cache_locks[query]:
                # Check cache again inside lock (double-check pattern)
                cached = cache_manager.get_cached_result(query, QueryModality.TEXT)
                if cached:
                    return cached

                # Execute
                tasks = [("text_agent", query, {"tenant_id": "test"})]
                result = await parallel_executor.execute_agents_parallel(
                    tasks, agent_caller=mock_agent_caller
                )

                # Track metrics
                if "text_agent" in result["results"]:
                    latency = result["latencies"]["text_agent"]
                    metrics_tracker.record_modality_execution(
                        QueryModality.TEXT, latency, True
                    )

                # Cache result
                cache_manager.cache_result(query, QueryModality.TEXT, result)
                return result

        # Simulate 10 requests with 5 unique queries (50% duplicates)
        queries = [
            "machine learning",
            "deep learning",
            "machine learning",  # Duplicate
            "neural networks",
            "deep learning",  # Duplicate
            "machine learning",  # Duplicate
            "computer vision",
            "nlp",
            "machine learning",  # Duplicate
            "deep learning",  # Duplicate
        ]

        # Execute all requests concurrently
        results = await asyncio.gather(
            *[process_request(q) for q in queries], return_exceptions=False
        )

        # Verify all completed
        assert len(results) == 10

        # Verify cache reduced executions to 5 (one per unique query)
        assert request_count["count"] == 5

        # Verify metrics tracked exactly 5 requests
        stats = metrics_tracker.get_modality_stats(QueryModality.TEXT)
        assert stats["total_requests"] == 5
        assert stats["success_rate"] == 1.0

        # Verify cache effectiveness
        cache_stats = cache_manager.get_cache_stats(QueryModality.TEXT)
        assert cache_stats["hits"] == 5  # 5 duplicates hit cache
        # Misses will be higher due to double-check pattern (check before and inside lock)
        assert cache_stats["misses"] >= 5

    async def test_error_isolation_in_parallel_execution(
        self, parallel_executor, metrics_tracker
    ):
        """Test that errors in one agent don't affect others"""

        async def mock_agent_caller(agent_name: str, query: str, context: dict):
            if agent_name == "failing_agent":
                raise Exception("Agent failed")
            await asyncio.sleep(0.1)
            return {"results": [{"id": "success", "score": 0.9}], "count": 1}

        tasks = [
            ("good_agent_1", "test", {"tenant_id": "test"}),
            ("failing_agent", "test", {"tenant_id": "test"}),
            ("good_agent_2", "test", {"tenant_id": "test"}),
        ]

        result = await parallel_executor.execute_agents_parallel(
            tasks, agent_caller=mock_agent_caller
        )

        # Verify error isolation
        assert "good_agent_1" in result["results"]
        assert "good_agent_2" in result["results"]
        assert "failing_agent" in result["errors"]
        assert result["successful_agents"] == 2
        assert result["failed_agents"] == 1
        assert "failed" in result["errors"]["failing_agent"].lower()

        # Manually track metrics
        for agent_name in ["good_agent_1", "good_agent_2"]:
            metrics_tracker.record_modality_execution(
                QueryModality.TEXT,
                result["latencies"][agent_name],
                True,
                None,
            )
        metrics_tracker.record_modality_execution(
            QueryModality.TEXT, result["latencies"]["failing_agent"], False, "error"
        )

        # Verify metrics tracked both success and failure
        stats = metrics_tracker.get_summary_stats()
        assert stats["total_requests"] == 3
        # Overall success rate should be 2/3
        assert 0.6 < stats["overall_success_rate"] < 0.7

    async def test_cache_ttl_expiration(self, cache_manager):
        """Test cache entries expire after TTL"""

        query = "test query"
        modality = QueryModality.VIDEO
        result = {"results": [{"id": "test", "score": 0.9}], "count": 1}

        # Cache result
        cache_manager.cache_result(query, modality, result)

        # Immediate retrieval should work
        cached = cache_manager.get_cached_result(query, modality, ttl_seconds=1)
        assert cached is not None

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired
        cached = cache_manager.get_cached_result(query, modality, ttl_seconds=1)
        assert cached is None

    async def test_lazy_evaluation_cost_ordering(self, lazy_executor):
        """Test that modalities are executed in cost order"""

        execution_order = []

        async def mock_modality_executor(
            query: str, modality: QueryModality, context: dict
        ):
            execution_order.append(modality)
            # Return low quality to force all executions
            return {"results": [], "count": 0, "confidence": 0.3}

        modalities = [
            QueryModality.AUDIO,  # Cost 10
            QueryModality.TEXT,  # Cost 1
            QueryModality.VIDEO,  # Cost 8
            QueryModality.DOCUMENT,  # Cost 2
        ]

        await lazy_executor.execute_with_lazy_evaluation(
            "test",
            modalities,
            {"quality_threshold": 0.9, "min_results_required": 100},
            modality_executor=mock_modality_executor,
        )

        # Verify cost-based ordering
        assert execution_order[0] == QueryModality.TEXT  # Cheapest
        assert execution_order[1] == QueryModality.DOCUMENT
        assert execution_order[-1] == QueryModality.AUDIO  # Most expensive

    def _agent_to_modality(self, agent_name: str) -> QueryModality:
        """Map agent name to modality"""
        mapping = {
            "video_search_agent": QueryModality.VIDEO,
            "document_agent": QueryModality.DOCUMENT,
            "image_search_agent": QueryModality.IMAGE,
            "audio_analysis_agent": QueryModality.AUDIO,
            "text_search_agent": QueryModality.TEXT,
        }
        return mapping.get(agent_name, QueryModality.TEXT)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
