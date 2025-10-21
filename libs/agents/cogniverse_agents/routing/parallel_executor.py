"""
Parallel Agent Execution

Executes agents in parallel with resource management, timeouts, and error isolation.
Part of Phase 12: Production Readiness.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ParallelAgentExecutor:
    """
    Execute agents in parallel with resource management

    Features:
    - Concurrent execution with semaphore limits
    - Per-agent timeouts
    - Error isolation (one agent failure doesn't crash others)
    - Result aggregation
    - Latency tracking

    Example:
        executor = ParallelAgentExecutor(max_concurrent_agents=5)

        agent_tasks = [
            ("video_search", "machine learning tutorials", {}),
            ("document_agent", "machine learning tutorials", {}),
        ]

        results = await executor.execute_agents_parallel(
            agent_tasks,
            timeout_seconds=10.0
        )
    """

    def __init__(self, max_concurrent_agents: int = 5):
        """
        Initialize parallel executor

        Args:
            max_concurrent_agents: Maximum agents to run concurrently
        """
        self.max_concurrent_agents = max_concurrent_agents
        self.semaphore = asyncio.Semaphore(max_concurrent_agents)
        self.execution_count = 0
        self.timeout_count = 0
        self.error_count = 0

        logger.info(
            f"🚀 Initialized ParallelAgentExecutor "
            f"(max_concurrent: {max_concurrent_agents})"
        )

    async def execute_agents_parallel(
        self,
        agent_tasks: List[Tuple[str, str, Dict]],
        timeout_seconds: float = 30.0,
        agent_caller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute agents in parallel

        Args:
            agent_tasks: List of (agent_name, query, context) tuples
            timeout_seconds: Timeout per agent in seconds
            agent_caller: Function to call agents (async callable)

        Returns:
            {
                "results": {agent_name: result},
                "errors": {agent_name: error},
                "latencies": {agent_name: latency_ms},
                "total_duration_ms": float,
                "successful_agents": int,
                "failed_agents": int,
                "timed_out_agents": int,
            }
        """
        if not agent_caller:
            raise ValueError("agent_caller function is required")

        start_time = time.time()

        # Execute with timeout wrapper
        async def execute_with_timeout(
            agent_name: str, query: str, context: Dict
        ) -> Tuple[str, Dict[str, Any]]:
            """Execute single agent with timeout and error handling"""
            agent_start = time.time()

            async with self.semaphore:
                try:
                    result = await asyncio.wait_for(
                        agent_caller(agent_name, query, context),
                        timeout=timeout_seconds,
                    )

                    latency_ms = (time.time() - agent_start) * 1000

                    logger.info(
                        f"✅ Agent {agent_name} completed in {latency_ms:.0f}ms"
                    )

                    return agent_name, {
                        "status": "success",
                        "result": result,
                        "latency_ms": latency_ms,
                    }

                except asyncio.TimeoutError:
                    latency_ms = (time.time() - agent_start) * 1000
                    self.timeout_count += 1

                    logger.warning(
                        f"⏱️ Agent {agent_name} timed out after {timeout_seconds}s"
                    )

                    return agent_name, {
                        "status": "timeout",
                        "error": f"Timeout after {timeout_seconds}s",
                        "latency_ms": latency_ms,
                    }

                except Exception as e:
                    latency_ms = (time.time() - agent_start) * 1000
                    self.error_count += 1

                    logger.error(f"❌ Agent {agent_name} failed: {e}")

                    return agent_name, {
                        "status": "error",
                        "error": str(e),
                        "latency_ms": latency_ms,
                    }

        # Create tasks for all agents
        tasks = [
            execute_with_timeout(agent_name, query, context)
            for agent_name, query, context in agent_tasks
        ]

        # Execute in parallel
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        results = {}
        errors = {}
        latencies = {}
        successful = 0
        failed = 0
        timed_out = 0

        for task_result in task_results:
            if isinstance(task_result, Exception):
                # Gather itself raised an exception (shouldn't happen with return_exceptions=True)
                logger.error(f"Unexpected exception in gather: {task_result}")
                failed += 1
                continue

            agent_name, agent_result = task_result
            status = agent_result["status"]
            latencies[agent_name] = agent_result["latency_ms"]

            if status == "success":
                results[agent_name] = agent_result["result"]
                successful += 1
            elif status == "timeout":
                errors[agent_name] = agent_result["error"]
                timed_out += 1
            else:  # error
                errors[agent_name] = agent_result["error"]
                failed += 1

        total_duration_ms = (time.time() - start_time) * 1000
        self.execution_count += len(agent_tasks)

        logger.info(
            f"🏁 Parallel execution complete: {successful} success, "
            f"{failed} failed, {timed_out} timeout in {total_duration_ms:.0f}ms"
        )

        return {
            "results": results,
            "errors": errors,
            "latencies": latencies,
            "total_duration_ms": total_duration_ms,
            "successful_agents": successful,
            "failed_agents": failed,
            "timed_out_agents": timed_out,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics

        Returns:
            Execution statistics
        """
        return {
            "total_executions": self.execution_count,
            "timeout_count": self.timeout_count,
            "error_count": self.error_count,
            "max_concurrent": self.max_concurrent_agents,
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.execution_count = 0
        self.timeout_count = 0
        self.error_count = 0
        logger.info("📊 Reset executor statistics")
