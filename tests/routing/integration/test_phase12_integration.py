"""
Integration tests for Phase 12: Production Readiness

Tests the complete workflow with parallel execution, caching, lazy evaluation, and metrics.
"""

import asyncio
import time

import pytest

from src.app.routing.lazy_executor import LazyModalityExecutor
from src.app.routing.modality_cache import ModalityCacheManager
from src.app.routing.parallel_executor import ParallelAgentExecutor
from src.app.search.multi_modal_reranker import QueryModality
from src.app.telemetry.modality_metrics import ModalityMetricsTracker


class TestPhase12Integration:
    """Integration tests for Phase 12 components"""

    @pytest.mark.asyncio
    async def test_parallel_execution_with_metrics(self):
        """Test parallel execution tracking metrics"""
        executor = ParallelAgentExecutor(max_concurrent_agents=3)
        tracker = ModalityMetricsTracker()

        # Mock agent caller that records metrics
        async def agent_caller_with_metrics(agent_name, query, context):
            start = time.time()
            await asyncio.sleep(0.05)  # Simulate work
            latency_ms = (time.time() - start) * 1000

            # Record metrics
            modality = (
                QueryModality.VIDEO if "video" in agent_name else QueryModality.DOCUMENT
            )
            tracker.record_modality_execution(modality, latency_ms, True)

            return {"agent": agent_name, "results": ["result1", "result2"]}

        # Execute agents
        agent_tasks = [
            ("video_search", "test query", {}),
            ("video_analysis", "test query", {}),
            ("document_agent", "test query", {}),
        ]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=agent_caller_with_metrics,
        )

        # Verify execution
        assert result["successful_agents"] == 3

        # Verify metrics tracked
        video_stats = tracker.get_modality_stats(QueryModality.VIDEO)
        doc_stats = tracker.get_modality_stats(QueryModality.DOCUMENT)

        assert video_stats["total_requests"] == 2  # 2 video agents
        assert doc_stats["total_requests"] == 1  # 1 document agent
        assert video_stats["success_rate"] == 1.0
        assert doc_stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_caching_with_lazy_evaluation(self):
        """Test caching integrated with lazy evaluation"""
        cache = ModalityCacheManager(cache_size_per_modality=10)
        lazy_executor = LazyModalityExecutor(default_quality_threshold=0.8)

        execution_count = {"total": 0}

        # Mock modality executor that uses cache
        async def cached_modality_executor(query, modality, context):
            # Check cache first
            cached_result = cache.get_cached_result(query, modality, ttl_seconds=3600)
            if cached_result:
                return cached_result

            # Execute (expensive)
            execution_count["total"] += 1
            await asyncio.sleep(0.01)

            result = {
                "results": [
                    {"content": f"result_{i}", "confidence": 0.9} for i in range(10)
                ],
                "confidence": 0.9,
            }

            # Cache result
            cache.cache_result(query, modality, result)

            return result

        # First execution - should hit modality executor
        modalities = [QueryModality.TEXT, QueryModality.VIDEO]
        query = "test query"

        await lazy_executor.execute_with_lazy_evaluation(
            query=query,
            modalities=modalities,
            context={"quality_threshold": 0.8},
            modality_executor=cached_modality_executor,
        )

        first_execution_count = execution_count["total"]
        assert first_execution_count > 0

        # Second execution with same query - should use cache
        await lazy_executor.execute_with_lazy_evaluation(
            query=query,
            modalities=modalities,
            context={"quality_threshold": 0.8},
            modality_executor=cached_modality_executor,
        )

        # Execution count should not increase
        assert execution_count["total"] == first_execution_count

        # Verify cache stats
        cache_stats = cache.get_cache_stats(QueryModality.TEXT)
        assert cache_stats["hits"] > 0

    @pytest.mark.asyncio
    async def test_complete_production_workflow(self):
        """Test complete production workflow with all Phase 12 components"""
        # Initialize all components
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=5)
        cache_manager = ModalityCacheManager()
        lazy_executor = LazyModalityExecutor(default_quality_threshold=0.8)
        metrics_tracker = ModalityMetricsTracker()

        # Mock agent caller for parallel execution
        async def agent_caller(agent_name, query, context):
            modality_str = context.get("modality", "text")
            modality = QueryModality(modality_str)

            start = time.time()
            await asyncio.sleep(0.01)
            latency_ms = (time.time() - start) * 1000

            # Record metrics
            metrics_tracker.record_modality_execution(modality, latency_ms, True)

            return {
                "agent": agent_name,
                "results": [
                    {"content": f"{agent_name}_result_{i}", "score": 0.9}
                    for i in range(5)
                ],
            }

        # Mock modality executor for lazy evaluation (with caching)
        async def modality_executor(query, modality, context):
            # Check cache
            cached = cache_manager.get_cached_result(query, modality, ttl_seconds=3600)
            if cached:
                return cached

            # Execute agents for this modality
            agent_tasks = [
                (f"{modality.value}_agent_{i}", query, {"modality": modality.value})
                for i in range(2)
            ]

            parallel_result = await parallel_executor.execute_agents_parallel(
                agent_tasks,
                timeout_seconds=5.0,
                agent_caller=agent_caller,
            )

            # Aggregate results
            all_results = []
            for agent_results in parallel_result["results"].values():
                all_results.extend(agent_results.get("results", []))

            result = {
                "results": all_results,
                "confidence": 0.9,
            }

            # Cache result
            cache_manager.cache_result(query, modality, result)

            return result

        # Execute workflow with lazy evaluation
        query = "machine learning tutorials"
        modalities = [QueryModality.TEXT, QueryModality.VIDEO, QueryModality.DOCUMENT]

        result = await lazy_executor.execute_with_lazy_evaluation(
            query=query,
            modalities=modalities,
            context={"quality_threshold": 0.8, "min_results_required": 5},
            modality_executor=modality_executor,
        )

        # Verify results
        assert len(result["results"]) > 0
        assert result["total_cost"] > 0

        # Verify metrics collected
        summary = metrics_tracker.get_summary_stats()
        assert summary["total_requests"] > 0
        assert summary["overall_success_rate"] == 1.0

        # Verify cache working
        cache_stats = cache_manager.get_cache_stats()
        # At least one modality should have cached data
        assert any(stats["cache_size"] > 0 for stats in cache_stats.values())

        # Second execution should be faster due to caching
        start = time.time()
        await lazy_executor.execute_with_lazy_evaluation(
            query=query,
            modalities=modalities,
            context={"quality_threshold": 0.8},
            modality_executor=modality_executor,
        )
        cached_duration = time.time() - start

        # Cached execution should be significantly faster
        # (no actual agent calls, just cache lookups)
        assert cached_duration < 0.1  # Should be under 100ms

    @pytest.mark.asyncio
    async def test_error_handling_in_production_workflow(self):
        """Test error handling across Phase 12 components"""
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=3)
        metrics_tracker = ModalityMetricsTracker()

        # Mock agent caller that sometimes fails
        async def failing_agent_caller(agent_name, query, context):
            modality = QueryModality.VIDEO

            if "failing" in agent_name:
                # Record failure
                metrics_tracker.record_modality_execution(
                    modality, 50.0, False, "connection_error"
                )
                raise ValueError("Simulated failure")

            # Success
            metrics_tracker.record_modality_execution(modality, 100.0, True)
            return {"agent": agent_name, "results": ["result"]}

        agent_tasks = [
            ("good_agent", "test", {}),
            ("failing_agent", "test", {}),
            ("another_good_agent", "test", {}),
        ]

        result = await parallel_executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=failing_agent_caller,
        )

        # Verify mixed success/failure
        assert result["successful_agents"] == 2
        assert result["failed_agents"] == 1

        # Verify metrics tracked errors
        stats = metrics_tracker.get_modality_stats(QueryModality.VIDEO)
        assert stats["error_count"] == 1
        assert "connection_error" in stats["error_breakdown"]

    @pytest.mark.asyncio
    async def test_latency_tracking_across_modalities(self):
        """Test latency tracking for different modalities"""
        metrics_tracker = ModalityMetricsTracker()

        # Simulate different latencies per modality
        modality_latencies = {
            QueryModality.TEXT: 50.0,
            QueryModality.DOCUMENT: 100.0,
            QueryModality.VIDEO: 300.0,
            QueryModality.AUDIO: 500.0,
        }

        # Record executions
        for modality, latency in modality_latencies.items():
            for i in range(10):
                metrics_tracker.record_modality_execution(
                    modality, latency + (i * 10), True
                )

        # Get slowest modalities
        slowest = metrics_tracker.get_slowest_modalities(top_k=2)

        assert len(slowest) == 2
        assert slowest[0]["modality"] == "audio"  # Slowest
        assert slowest[1]["modality"] == "video"  # Second slowest
        assert slowest[0]["p95_latency"] > slowest[1]["p95_latency"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_workflow(self):
        """Test cache invalidation in production workflow"""
        cache = ModalityCacheManager()

        # Cache results for multiple modalities
        cache.cache_result("query1", QueryModality.VIDEO, {"videos": ["v1", "v2"]})
        cache.cache_result("query2", QueryModality.DOCUMENT, {"docs": ["d1", "d2"]})

        # Verify cached
        assert cache.get_cached_result("query1", QueryModality.VIDEO, 3600) is not None
        assert (
            cache.get_cached_result("query2", QueryModality.DOCUMENT, 3600) is not None
        )

        # Invalidate VIDEO cache
        cache.invalidate_modality(QueryModality.VIDEO)

        # Verify VIDEO invalidated, DOCUMENT still cached
        assert cache.get_cached_result("query1", QueryModality.VIDEO, 3600) is None
        assert (
            cache.get_cached_result("query2", QueryModality.DOCUMENT, 3600) is not None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
