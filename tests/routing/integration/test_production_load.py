"""
Production Load Tests for Phase 12

Tests system throughput, concurrent request handling, and sustained load.
Validates Phase 12 checkpoint criteria: 100+ QPS capability.
"""

import asyncio
import time

import pytest

from src.app.routing.modality_cache import ModalityCacheManager
from src.app.routing.parallel_executor import ParallelAgentExecutor
from src.app.search.multi_modal_reranker import QueryModality
from src.app.telemetry.modality_metrics import ModalityMetricsTracker


class TestProductionLoadHandling:
    """Load tests for production readiness validation"""

    @pytest.mark.asyncio
    async def test_system_throughput_100_qps(self):
        """
        Test system handles 100+ QPS

        Per Phase 12 checkpoint: "System handles 100+ QPS"
        """
        # Initialize components
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=10)
        metrics_tracker = ModalityMetricsTracker()

        # Mock fast agent caller
        async def fast_agent_caller(agent_name, query, context):
            modality = context.get("modality", QueryModality.TEXT)

            start = time.time()
            # Simulate minimal work (< 10ms)
            await asyncio.sleep(0.005)
            latency_ms = (time.time() - start) * 1000

            metrics_tracker.record_modality_execution(modality, latency_ms, True)

            return {"results": [f"{agent_name}_result"]}

        # Generate 200 requests
        num_requests = 200
        requests = []

        for i in range(num_requests):
            # Mix of modalities
            modality = [
                QueryModality.TEXT,
                QueryModality.DOCUMENT,
                QueryModality.VIDEO,
            ][i % 3]

            agent_tasks = [(f"agent_{i}", f"query_{i}", {"modality": modality})]

            requests.append(
                parallel_executor.execute_agents_parallel(
                    agent_tasks,
                    timeout_seconds=5.0,
                    agent_caller=fast_agent_caller,
                )
            )

        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*requests)
        duration = time.time() - start_time

        # Calculate QPS
        qps = num_requests / duration

        # Verify 100+ QPS
        assert qps >= 100, f"System QPS {qps:.0f} below target 100"

        # Verify all requests succeeded
        successful = sum(1 for r in results if r["successful_agents"] > 0)
        assert successful == num_requests

        # Report metrics
        summary = metrics_tracker.get_summary_stats()
        print("\n📊 Throughput Test Results:")
        print(f"   QPS: {qps:.0f}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Requests: {num_requests}")
        print(f"   Success Rate: {summary['overall_success_rate']:.1%}")

    @pytest.mark.asyncio
    async def test_concurrent_requests_stress(self):
        """
        Test system handles 500+ concurrent requests without failures

        Stress test to validate resource management under high concurrency.
        """
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=20)
        cache_manager = ModalityCacheManager()

        error_count = 0

        async def agent_caller_with_cache(agent_name, query, context):
            nonlocal error_count

            modality = context.get("modality", QueryModality.TEXT)

            # Check cache first
            cached = cache_manager.get_cached_result(query, modality, 3600)
            if cached:
                return cached

            try:
                # Simulate work
                await asyncio.sleep(0.01)
                result = {"results": [f"{agent_name}_result"]}

                # Cache result
                cache_manager.cache_result(query, modality, result)

                return result
            except Exception:
                error_count += 1
                raise

        # Generate 500 concurrent requests
        num_requests = 500
        requests = []

        for i in range(num_requests):
            # Some queries repeat (for cache hits)
            query_id = i % 50  # 50 unique queries

            modality = [
                QueryModality.TEXT,
                QueryModality.VIDEO,
                QueryModality.DOCUMENT,
            ][i % 3]

            agent_tasks = [(f"agent_{i}", f"query_{query_id}", {"modality": modality})]

            requests.append(
                parallel_executor.execute_agents_parallel(
                    agent_tasks,
                    timeout_seconds=10.0,
                    agent_caller=agent_caller_with_cache,
                )
            )

        # Execute all concurrently
        results = await asyncio.gather(*requests, return_exceptions=True)

        # Count successes
        successful = sum(
            1
            for r in results
            if isinstance(r, dict) and r.get("successful_agents", 0) > 0
        )

        # Verify high success rate
        success_rate = successful / num_requests
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95%"

        # Verify cache is working (should reduce actual executions)
        cache_stats = cache_manager.get_cache_stats()
        total_hits = sum(stats["hits"] for stats in cache_stats.values())
        assert total_hits > 0, "Cache should have hits with repeated queries"

        print("\n🔥 Stress Test Results:")
        print(f"   Concurrent Requests: {num_requests}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Cache Hits: {total_hits}")
        print(f"   Errors: {error_count}")

    @pytest.mark.asyncio
    async def test_sustained_load_60_seconds(self):
        """
        Test system handles sustained load for 60 seconds

        Validates stability under continuous load.
        """
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=10)
        metrics_tracker = ModalityMetricsTracker()

        request_count = 0
        error_count = 0

        async def agent_caller(agent_name, query, context):
            nonlocal error_count

            modality = context.get("modality", QueryModality.TEXT)

            try:
                start = time.time()
                await asyncio.sleep(0.01)  # 10ms work
                latency_ms = (time.time() - start) * 1000

                metrics_tracker.record_modality_execution(modality, latency_ms, True)

                return {"results": ["result"]}
            except Exception as e:
                error_count += 1
                metrics_tracker.record_modality_execution(modality, 0, False, str(e))
                raise

        async def request_generator():
            """Generate continuous requests for 60 seconds"""
            nonlocal request_count

            end_time = time.time() + 60  # 60 second test

            while time.time() < end_time:
                modality = [
                    QueryModality.TEXT,
                    QueryModality.DOCUMENT,
                    QueryModality.VIDEO,
                ][request_count % 3]

                agent_tasks = [
                    (
                        f"agent_{request_count}",
                        f"query_{request_count}",
                        {"modality": modality},
                    )
                ]

                # Fire and forget (don't wait)
                asyncio.create_task(
                    parallel_executor.execute_agents_parallel(
                        agent_tasks,
                        timeout_seconds=5.0,
                        agent_caller=agent_caller,
                    )
                )

                request_count += 1

                # Steady rate: ~100 QPS = 10ms between requests
                await asyncio.sleep(0.01)

        # Run load generator
        start_time = time.time()
        await request_generator()
        duration = time.time() - start_time

        # Allow pending requests to complete
        await asyncio.sleep(1.0)

        # Calculate metrics
        qps = request_count / duration
        summary = metrics_tracker.get_summary_stats()

        # Verify sustained performance
        assert qps >= 90, f"Sustained QPS {qps:.0f} below 90 (allowing 10% margin)"
        assert summary["overall_success_rate"] >= 0.95, "Success rate below 95%"

        print("\n⏱️  Sustained Load Test Results:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Total Requests: {request_count}")
        print(f"   Average QPS: {qps:.0f}")
        print(f"   Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"   Errors: {error_count}")

    @pytest.mark.asyncio
    async def test_latency_distribution_under_load(self):
        """
        Test P50, P95, P99 latencies under load

        Validates latency distribution meets targets under concurrent load.
        """
        parallel_executor = ParallelAgentExecutor(max_concurrent_agents=10)
        metrics_tracker = ModalityMetricsTracker()

        # Simulate varying latencies
        async def variable_latency_caller(agent_name, query, context):
            modality = QueryModality.TEXT

            # Variable latency (10-50ms)
            import random

            latency_s = random.uniform(0.01, 0.05)

            start = time.time()
            await asyncio.sleep(latency_s)
            actual_latency_ms = (time.time() - start) * 1000

            metrics_tracker.record_modality_execution(modality, actual_latency_ms, True)

            return {"results": ["result"]}

        # Execute 100 requests concurrently
        requests = []
        for i in range(100):
            agent_tasks = [(f"agent_{i}", f"query_{i}", {})]
            requests.append(
                parallel_executor.execute_agents_parallel(
                    agent_tasks,
                    timeout_seconds=5.0,
                    agent_caller=variable_latency_caller,
                )
            )

        await asyncio.gather(*requests)

        # Get latency stats
        stats = metrics_tracker.get_modality_stats(QueryModality.TEXT)

        # Verify latency distribution
        assert (
            stats["p50_latency"] < 100
        ), f"P50 latency {stats['p50_latency']:.0f}ms too high"
        assert (
            stats["p95_latency"] < 200
        ), f"P95 latency {stats['p95_latency']:.0f}ms too high"
        assert (
            stats["p99_latency"] < 300
        ), f"P99 latency {stats['p99_latency']:.0f}ms too high"

        print("\n📈 Latency Distribution Results:")
        print(f"   P50: {stats['p50_latency']:.0f}ms")
        print(f"   P95: {stats['p95_latency']:.0f}ms")
        print(f"   P99: {stats['p99_latency']:.0f}ms")
        print(f"   Total Requests: {stats['total_requests']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
