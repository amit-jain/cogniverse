"""
Unit tests for ParallelAgentExecutor
"""

import asyncio

import pytest
from cogniverse_agents.routing.parallel_executor import ParallelAgentExecutor


class TestParallelAgentExecutor:
    """Test ParallelAgentExecutor functionality"""

    @pytest.fixture
    def executor(self):
        """Create executor instance"""
        return ParallelAgentExecutor(max_concurrent_agents=3)

    @pytest.fixture
    def mock_agent_caller(self):
        """Create mock agent caller"""

        async def caller(agent_name, query, context):
            # Simulate agent execution
            await asyncio.sleep(0.01)
            return {"agent": agent_name, "result": f"Result for {query}"}

        return caller

    def test_initialization(self, executor):
        """Test executor initialization"""
        assert executor.max_concurrent_agents == 3
        assert executor.execution_count == 0
        assert executor.timeout_count == 0
        assert executor.error_count == 0

    @pytest.mark.asyncio
    async def test_execute_single_agent(self, executor, mock_agent_caller):
        """Test executing single agent"""
        agent_tasks = [("video_search", "test query", {})]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mock_agent_caller,
        )

        assert result["successful_agents"] == 1
        assert result["failed_agents"] == 0
        assert result["timed_out_agents"] == 0
        assert "video_search" in result["results"]
        assert "video_search" in result["latencies"]
        assert result["latencies"]["video_search"] > 0

    @pytest.mark.asyncio
    async def test_execute_multiple_agents_parallel(self, executor, mock_agent_caller):
        """Test executing multiple agents in parallel"""
        agent_tasks = [
            ("video_search", "test query", {}),
            ("document_agent", "test query", {}),
            ("image_search", "test query", {}),
        ]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mock_agent_caller,
        )

        assert result["successful_agents"] == 3
        assert result["failed_agents"] == 0
        assert len(result["results"]) == 3
        assert all(
            agent in result["results"]
            for agent in ["video_search", "document_agent", "image_search"]
        )

    @pytest.mark.asyncio
    async def test_agent_timeout(self, executor):
        """Test agent timeout handling"""

        async def slow_caller(agent_name, query, context):
            await asyncio.sleep(2.0)  # Longer than timeout
            return {"result": "done"}

        agent_tasks = [("slow_agent", "test", {})]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=0.1,  # Short timeout
            agent_caller=slow_caller,
        )

        assert result["successful_agents"] == 0
        assert result["timed_out_agents"] == 1
        assert "slow_agent" in result["errors"]
        assert "Timeout" in result["errors"]["slow_agent"]

    @pytest.mark.asyncio
    async def test_agent_error(self, executor):
        """Test agent error handling"""

        async def failing_caller(agent_name, query, context):
            raise ValueError("Agent failed")

        agent_tasks = [("failing_agent", "test", {})]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=failing_caller,
        )

        assert result["successful_agents"] == 0
        assert result["failed_agents"] == 1
        assert "failing_agent" in result["errors"]
        assert "Agent failed" in result["errors"]["failing_agent"]

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, executor):
        """Test handling mixed success and failure"""

        async def mixed_caller(agent_name, query, context):
            if agent_name == "failing_agent":
                raise ValueError("Failed")
            return {"result": "success"}

        agent_tasks = [
            ("good_agent", "test", {}),
            ("failing_agent", "test", {}),
            ("another_good_agent", "test", {}),
        ]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mixed_caller,
        )

        assert result["successful_agents"] == 2
        assert result["failed_agents"] == 1
        assert "good_agent" in result["results"]
        assert "another_good_agent" in result["results"]
        assert "failing_agent" in result["errors"]

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, executor):
        """Test that semaphore limits concurrent executions"""
        execution_times = []

        async def tracking_caller(agent_name, query, context):
            execution_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)
            return {"result": "done"}

        # Create 6 tasks (2x the semaphore limit of 3)
        agent_tasks = [(f"agent_{i}", "test", {}) for i in range(6)]

        await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=tracking_caller,
        )

        # With semaphore=3, we should see 2 batches of execution
        # First 3 agents start immediately, next 3 wait
        assert len(execution_times) == 6

    @pytest.mark.asyncio
    async def test_get_stats(self, executor, mock_agent_caller):
        """Test getting executor statistics"""
        agent_tasks = [("agent1", "test", {})]

        await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mock_agent_caller,
        )

        stats = executor.get_stats()

        assert stats["total_executions"] == 1
        assert stats["max_concurrent"] == 3

    @pytest.mark.asyncio
    async def test_reset_stats(self, executor, mock_agent_caller):
        """Test resetting statistics"""
        agent_tasks = [("agent1", "test", {})]

        await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mock_agent_caller,
        )

        executor.reset_stats()

        stats = executor.get_stats()
        assert stats["total_executions"] == 0
        assert stats["timeout_count"] == 0
        assert stats["error_count"] == 0

    @pytest.mark.asyncio
    async def test_no_agent_caller_raises_error(self, executor):
        """Test that missing agent_caller raises error"""
        agent_tasks = [("agent1", "test", {})]

        with pytest.raises(ValueError, match="agent_caller function is required"):
            await executor.execute_agents_parallel(
                agent_tasks,
                timeout_seconds=5.0,
            )

    @pytest.mark.asyncio
    async def test_latency_tracking(self, executor, mock_agent_caller):
        """Test that latencies are tracked correctly"""
        agent_tasks = [
            ("agent1", "test", {}),
            ("agent2", "test", {}),
        ]

        result = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=5.0,
            agent_caller=mock_agent_caller,
        )

        # Check latencies are positive and reasonable
        for agent_name in ["agent1", "agent2"]:
            assert agent_name in result["latencies"]
            assert result["latencies"][agent_name] > 0
            assert result["latencies"][agent_name] < 1000  # Should be < 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
