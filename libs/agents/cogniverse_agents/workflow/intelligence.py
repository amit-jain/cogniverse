"""
Workflow Intelligence - Template loader, profile provider, and execution recorder

Loads workflow templates and agent performance profiles from artifacts at startup.
Provides template matching for the orchestrator. Records workflow executions to
in-memory history (used by OrchestrationEvaluator in batch jobs). Does NOT run
DSPy optimization inline.
"""

import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.workflow_types import (
    WorkflowPlan,
    WorkflowTask,
)
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider


class OptimizationStrategy(Enum):
    """Workflow optimization strategies"""

    PERFORMANCE_BASED = "performance_based"
    SUCCESS_RATE_BASED = "success_rate_based"
    LATENCY_OPTIMIZED = "latency_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


@dataclass
class WorkflowExecution:
    """Historical workflow execution record"""

    workflow_id: str
    query: str
    query_type: str
    execution_time: float
    success: bool
    agent_sequence: List[str]
    task_count: int
    parallel_efficiency: float
    confidence_score: float
    user_satisfaction: Optional[float] = None
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Agent performance metrics"""

    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    average_confidence: float = 0.0
    error_rate: float = 0.0
    preferred_query_types: List[str] = field(default_factory=list)
    performance_trend: str = "stable"  # improving, degrading, stable
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowTemplate:
    """Reusable workflow template"""

    template_id: str
    name: str
    description: str
    query_patterns: List[str]
    task_sequence: List[Dict[str, Any]]
    expected_execution_time: float
    success_rate: float
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


class WorkflowIntelligence:
    """
    Template loader, profile provider, and execution recorder.

    Loads workflow templates and agent performance profiles from artifacts
    at startup. Provides template matching for the orchestrator. Records
    workflow executions to in-memory history for batch evaluation. Does NOT
    run DSPy optimization inline.
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        tenant_id: str,
        max_history_size: int = 10000,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required for WorkflowIntelligence")
        self.logger = logging.getLogger(__name__)
        self.tenant_id = tenant_id
        self.max_history_size = max_history_size
        self.optimization_strategy = optimization_strategy
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        # In-memory data structures (loaded at startup, read-only at runtime)
        self.workflow_history: deque = deque(maxlen=max_history_size)
        self.agent_performance: Dict[str, AgentPerformance] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.query_type_patterns: Dict[str, List[str]] = defaultdict(list)

        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "templates_created": 0,
            "templates_used": 0,
        }

    def get_workflow_templates(self) -> List[WorkflowTemplate]:
        """Return loaded workflow templates."""
        return list(self.workflow_templates.values())

    async def load_historical_data(self) -> None:
        """Load historical data from telemetry."""
        try:
            # Load workflow executions
            demos = await self._artifact_manager.load_demonstrations("workflow")
            if demos:
                for demo in demos:
                    try:
                        data = json.loads(demo["input"])
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                        execution = WorkflowExecution(**data)
                        self.workflow_history.append(execution)
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed execution: {e}")

            # Load agent performance
            profiles = await self._artifact_manager.load_demonstrations(
                "agent_profiles"
            )
            if profiles:
                for profile in profiles:
                    try:
                        data = json.loads(profile["input"])
                        data["last_updated"] = datetime.fromisoformat(
                            data["last_updated"]
                        )
                        perf = AgentPerformance(**data)
                        self.agent_performance[perf.agent_name] = perf
                    except Exception as e:
                        self.logger.warning(f"Skipping malformed agent profile: {e}")

            # Load query patterns
            patterns_json = await self._artifact_manager.load_blob(
                "workflow", "query_patterns"
            )
            if patterns_json:
                self.query_type_patterns = defaultdict(list, json.loads(patterns_json))

            # Load workflow templates
            template_index_json = await self._artifact_manager.load_blob(
                "workflow", "template_index"
            )
            if template_index_json:
                template_ids = json.loads(template_index_json)
                for tid in template_ids:
                    tmpl_json = await self._artifact_manager.load_blob(
                        "workflow", f"template_{tid}"
                    )
                    if tmpl_json:
                        try:
                            data = json.loads(tmpl_json)
                            data["created_at"] = datetime.fromisoformat(
                                data["created_at"]
                            )
                            if data.get("last_used"):
                                data["last_used"] = datetime.fromisoformat(
                                    data["last_used"]
                                )
                            template = WorkflowTemplate(**data)
                            self.workflow_templates[tid] = template
                        except Exception as e:
                            self.logger.warning(
                                f"Skipping malformed template {tid}: {e}"
                            )

            self.logger.info(
                f"Loaded {len(self.workflow_history)} executions, "
                f"{len(self.agent_performance)} agent profiles, "
                f"{len(self.workflow_templates)} templates"
            )

        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")

    async def record_workflow_execution(self, workflow_plan: WorkflowPlan) -> None:
        """No-op — workflow executions are recorded via telemetry spans.

        Batch optimization jobs rebuild in-memory history from spans via
        ``load_historical_data``; the per-request hot path does not write to
        ``workflow_history`` to avoid unbounded in-pod growth and two sources
        of truth.
        """
        self.logger.debug(
            "Workflow %s completed (recorded via telemetry spans)",
            workflow_plan.workflow_id,
        )

    async def optimize_workflow_plan(
        self,
        query: str,
        initial_plan: WorkflowPlan,
        optimization_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowPlan:
        """Optimize workflow plan using templates (no inline DSPy)."""
        self.optimization_stats["total_optimizations"] += 1

        try:
            template_match = self._find_matching_template(query)
            if template_match:
                optimized_plan = self._apply_template(initial_plan, template_match)
                self.optimization_stats["templates_used"] += 1
                self.logger.info(f"Applied template '{template_match.name}'")
                return optimized_plan

            final_plan = self._apply_optimization_strategy(initial_plan)
            self.optimization_stats["successful_optimizations"] += 1
            return final_plan

        except Exception as e:
            self.logger.error(f"Workflow optimization failed: {e}")
            return initial_plan

    def _find_matching_template(self, query: str) -> Optional[WorkflowTemplate]:
        """Find matching workflow template for query"""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0

        for template in self.workflow_templates.values():
            # Calculate similarity score with template patterns
            score = 0.0
            for pattern in template.query_patterns:
                pattern_words = set(pattern.lower().split())
                query_words = set(query_lower.split())

                # Simple Jaccard similarity
                intersection = len(pattern_words & query_words)
                union = len(pattern_words | query_words)

                if union > 0:
                    pattern_score = intersection / union
                    score = max(score, pattern_score)

            # Consider template success rate and usage
            weighted_score = score * (0.7 + 0.3 * template.success_rate)

            if (
                weighted_score > best_score and weighted_score > 0.6
            ):  # Minimum similarity threshold
                best_score = weighted_score
                best_match = template

        return best_match

    def _apply_template(
        self, initial_plan: WorkflowPlan, template: WorkflowTemplate
    ) -> WorkflowPlan:
        """Apply workflow template to create optimized plan"""
        # Create tasks based on template
        template_tasks = []

        for i, task_spec in enumerate(template.task_sequence):
            task_id = f"template_task_{i}"
            agent_name = task_spec.get("agent", "search_agent")
            task_type = task_spec.get("task", "process")

            # Generate appropriate query based on task type
            if task_type == "search":
                task_query = initial_plan.original_query
            elif task_type == "summarize":
                task_query = f"Summarize results for: {initial_plan.original_query}"
            elif task_type == "analyze":
                task_query = f"Analyze results for: {initial_plan.original_query}"
            else:
                task_query = initial_plan.original_query

            task = WorkflowTask(
                task_id=task_id,
                agent_name=agent_name,
                query=task_query,
                dependencies=set(task_spec.get("dependencies", [])),
            )
            template_tasks.append(task)

        # Create templated plan
        templated_plan = WorkflowPlan(
            workflow_id=f"template_{initial_plan.workflow_id}",
            original_query=initial_plan.original_query,
            tasks=template_tasks,
            metadata={
                **initial_plan.metadata,
                "template_applied": template.template_id,
                "template_name": template.name,
                "expected_execution_time": template.expected_execution_time,
                "expected_success_rate": template.success_rate,
            },
        )

        # Update template usage
        template.usage_count += 1
        template.last_used = datetime.now()

        # Recalculate execution order
        templated_plan.execution_order = self._calculate_execution_order(template_tasks)

        return templated_plan

    def _apply_optimization_strategy(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Apply specific optimization strategy to the plan"""
        if self.optimization_strategy == OptimizationStrategy.LATENCY_OPTIMIZED:
            return self._optimize_for_latency(plan)
        elif self.optimization_strategy == OptimizationStrategy.SUCCESS_RATE_BASED:
            return self._optimize_for_success_rate(plan)
        elif self.optimization_strategy == OptimizationStrategy.PERFORMANCE_BASED:
            return self._optimize_for_performance(plan)
        else:
            # Balanced strategy - no additional changes
            return plan

    def _optimize_for_latency(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize plan for minimum latency"""
        # Sort agents by average execution time (fastest first)
        agent_speeds = {
            name: perf.average_execution_time
            for name, perf in self.agent_performance.items()
        }

        # Reorder tasks to prioritize faster agents when possible
        for task in plan.tasks:
            if task.agent_name in agent_speeds:
                # Could add logic to prefer faster agents for similar capabilities
                pass

        plan.metadata["latency_optimized"] = True
        return plan

    def _optimize_for_success_rate(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize plan for maximum success rate"""
        # Prefer agents with higher success rates
        for task in plan.tasks:
            agent_name = task.agent_name
            if agent_name in self.agent_performance:
                perf = self.agent_performance[agent_name]
                success_rate = perf.successful_executions / max(
                    perf.total_executions, 1
                )
                if success_rate < 0.7:  # Consider alternative if success rate is low
                    # Could implement agent substitution logic here
                    pass

        plan.metadata["success_rate_optimized"] = True
        return plan

    def _optimize_for_performance(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Optimize plan for overall performance"""
        # Balance between speed, success rate, and confidence
        for task in plan.tasks:
            agent_name = task.agent_name
            if agent_name in self.agent_performance:
                perf = self.agent_performance[agent_name]

                # Calculate composite performance score
                success_rate = perf.successful_executions / max(
                    perf.total_executions, 1
                )
                time_factor = 1.0 / (
                    1.0 + perf.average_execution_time
                )  # Inverse of time
                confidence_factor = perf.average_confidence

                composite_score = (
                    success_rate * 0.4 + time_factor * 0.3 + confidence_factor * 0.3
                )

                # Could use this score for agent selection optimization
                task.metadata = task.metadata or {}
                task.metadata["performance_score"] = composite_score

        plan.metadata["performance_optimized"] = True
        return plan

    def _classify_query_type(self, query: str) -> str:
        """Classify query into type for pattern recognition"""
        query_lower = query.lower()

        # Video search queries
        if any(
            word in query_lower
            for word in ["video", "show", "watch", "visual", "footage"]
        ):
            return "video_search"

        # Summarization queries
        elif any(
            word in query_lower
            for word in ["summarize", "summary", "brief", "overview"]
        ):
            return "summarization"

        # Analysis queries
        elif any(
            word in query_lower
            for word in ["analyze", "analysis", "examine", "investigate"]
        ):
            return "analysis"

        # Report generation queries
        elif any(
            word in query_lower
            for word in ["report", "detailed", "comprehensive", "document"]
        ):
            return "report_generation"

        # Comparison queries
        elif any(
            word in query_lower
            for word in ["compare", "comparison", "versus", "vs", "difference"]
        ):
            return "comparison"

        # Multi-step queries
        elif any(
            word in query_lower
            for word in ["then", "and", "followed by", "after", "next"]
        ):
            return "multi_step"

        else:
            return "general"

    def _calculate_parallel_efficiency(self, workflow_plan: WorkflowPlan) -> float:
        """Calculate parallel execution efficiency."""
        if not workflow_plan.execution_order or not workflow_plan.tasks:
            return 0.0

        max_task_time = (
            max(
                (task.end_time - task.start_time).total_seconds()
                for task in workflow_plan.tasks
                if task.end_time and task.start_time
            )
            if workflow_plan.tasks
            else 0.0
        )

        actual_time = (
            (workflow_plan.end_time - workflow_plan.start_time).total_seconds()
            if workflow_plan.end_time and workflow_plan.start_time
            else 0.0
        )

        return max_task_time / actual_time if actual_time > 0 else 0.0

    def _calculate_execution_order(self, tasks: List[WorkflowTask]) -> List[List[str]]:
        """Calculate execution order considering dependencies (simplified version)"""
        # This is a simplified version - full implementation would be in the orchestrator
        if not tasks:
            return []

        # Simple sequential execution for now
        return [[task.task_id] for task in tasks]

    def get_intelligence_statistics(self) -> Dict[str, Any]:
        """Get workflow intelligence performance statistics"""
        stats: Dict[str, Any] = self.optimization_stats.copy()

        # Add historical data statistics
        stats.update(
            {
                "workflow_history_size": len(self.workflow_history),
                "tracked_agents": len(self.agent_performance),
                "available_templates": len(self.workflow_templates),
                "success_rate": (
                    len([exec for exec in self.workflow_history if exec.success])
                    / len(self.workflow_history)
                    if self.workflow_history
                    else 0.0
                ),
                "average_execution_time": (
                    statistics.mean(
                        [exec.execution_time for exec in self.workflow_history]
                    )
                    if self.workflow_history
                    else 0.0
                ),
                "query_types": len(
                    set(exec.query_type for exec in self.workflow_history)
                ),
                "persistence_enabled": True,
                "optimization_strategy": self.optimization_strategy.value,
            }
        )

        return stats

    def get_agent_performance_report(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed agent performance report"""
        return {
            agent_name: {
                "total_executions": perf.total_executions,
                "success_rate": perf.successful_executions
                / max(perf.total_executions, 1),
                "average_execution_time": perf.average_execution_time,
                "average_confidence": perf.average_confidence,
                "error_rate": perf.error_rate,
                "preferred_query_types": perf.preferred_query_types,
                "performance_trend": perf.performance_trend,
                "last_updated": perf.last_updated.isoformat(),
            }
            for agent_name, perf in self.agent_performance.items()
        }

    def get_workflow_templates_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of available workflow templates"""
        return {
            template.template_id: {
                "name": template.name,
                "description": template.description,
                "query_patterns": template.query_patterns,
                "expected_execution_time": template.expected_execution_time,
                "success_rate": template.success_rate,
                "usage_count": template.usage_count,
                "created_at": template.created_at.isoformat(),
                "last_used": (
                    template.last_used.isoformat() if template.last_used else None
                ),
            }
            for template in self.workflow_templates.values()
        }

    async def record_execution(self, workflow_execution: WorkflowExecution) -> None:
        """Record workflow execution directly (called by OrchestrationEvaluator in batch jobs)."""
        self.workflow_history.append(workflow_execution)
        self.logger.debug(
            "Recorded workflow execution: %s", workflow_execution.workflow_id
        )

    async def record_ground_truth_execution(
        self, workflow_execution: WorkflowExecution
    ) -> None:
        """Record ground truth execution (no-op, spans are the record)."""
        self.logger.info(
            "Ground truth workflow %s recorded via telemetry spans, not inline",
            workflow_execution.workflow_id,
        )

    async def optimize_from_ground_truth(self) -> Dict[str, Any]:
        """No-op — DSPy optimization is now handled by batch Argo jobs."""
        self.logger.info(
            "optimize_from_ground_truth is a no-op; use Argo batch jobs instead"
        )
        return {"status": "skipped", "reason": "use_argo_batch_jobs"}

    def get_successful_workflows(
        self, min_quality: float = 0.7, limit: int = 100
    ) -> List[WorkflowExecution]:
        """Return successful high-quality workflows for downstream optimization.

        Args:
            min_quality: Minimum user_satisfaction score (0.0-1.0)
            limit: Maximum number of workflows to return

        Returns:
            List of successful, high-quality workflow executions
        """
        successful_workflows = [
            w
            for w in self.workflow_history
            if w.success
            and w.user_satisfaction is not None
            and w.user_satisfaction >= min_quality
        ]

        # Sort by quality (user_satisfaction) descending
        successful_workflows.sort(
            key=lambda w: w.user_satisfaction or 0.0, reverse=True
        )

        return successful_workflows[:limit]

    async def generate_synthetic_training_data(
        self,
        count: int = 100,
        backend: Optional[Any] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Any] = None,
    ) -> int:
        """
        Generate synthetic training data using libs/synthetic system

        Args:
            count: Number of synthetic examples to generate
            backend: Optional Backend instance for content sampling
            backend_config: Backend configuration with profiles
            generator_config: Optional SyntheticGeneratorConfig for DSPy modules

        Returns:
            Number of examples added to workflow history
        """
        from cogniverse_synthetic import (
            SyntheticDataRequest,
            SyntheticDataService,
        )

        self.logger.info(f"Generating {count} synthetic workflow examples...")

        service = SyntheticDataService(
            backend=backend,
            backend_config=backend_config,
            generator_config=generator_config,
        )
        request = SyntheticDataRequest(
            optimizer="workflow", count=count, tenant_id=self.tenant_id
        )
        response = await service.generate(request)

        initial_count = len(self.workflow_history)
        for example_data in response.data:
            execution = WorkflowExecution(
                workflow_id=example_data["workflow_id"],
                query=example_data["query"],
                query_type=example_data["query_type"],
                execution_time=example_data["execution_time"],
                success=example_data["success"],
                agent_sequence=example_data["agent_sequence"],
                task_count=example_data["task_count"],
                parallel_efficiency=example_data["parallel_efficiency"],
                confidence_score=example_data["confidence_score"],
                user_satisfaction=example_data.get("user_satisfaction"),
                error_details=example_data.get("error_details"),
                metadata=example_data.get("metadata", {}),
            )
            self.workflow_history.append(execution)

        added_count = len(self.workflow_history) - initial_count
        self.logger.info(
            f"Added {added_count} synthetic examples to workflow history "
            f"(total: {len(self.workflow_history)})"
        )
        return added_count


def create_workflow_intelligence(
    telemetry_provider: TelemetryProvider,
    tenant_id: str,
    max_history_size: int = 10000,
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
) -> WorkflowIntelligence:
    """Factory function to create workflow intelligence system"""
    return WorkflowIntelligence(
        telemetry_provider=telemetry_provider,
        tenant_id=tenant_id,
        max_history_size=max_history_size,
        optimization_strategy=optimization_strategy,
    )
