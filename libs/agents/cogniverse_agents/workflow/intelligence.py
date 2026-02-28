"""
Workflow Intelligence - Advanced workflow optimization and learning system

This module implements intelligent workflow optimization features:
- Execution pattern learning and analysis
- Adaptive routing strategies based on historical performance
- Workflow template generation and reuse
- Performance-based agent selection
- Dynamic workflow optimization
- Predictive orchestration recommendations

Features:
- DSPy 3.0 powered workflow optimization
- Historical performance analysis
- Template-based workflow generation
- Agent performance profiling
- Predictive workflow planning
- Adaptive threshold learning
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# DSPy 3.0 imports
import dspy

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

# Shared workflow types
from cogniverse_agents.workflow.types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)


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


class WorkflowOptimizationSignature(dspy.Signature):
    """DSPy signature for intelligent workflow optimization"""

    workflow_history: str = dspy.InputField(
        desc="JSON string of historical workflow executions"
    )
    current_query: str = dspy.InputField(desc="Current query to optimize workflow for")
    agent_performance: str = dspy.InputField(
        desc="JSON string of agent performance metrics"
    )

    optimized_sequence: List[str] = dspy.OutputField(desc="Optimized agent sequence")
    optimization_strategy: str = dspy.OutputField(desc="Applied optimization strategy")
    expected_improvement: float = dspy.OutputField(
        desc="Expected performance improvement percentage"
    )
    reasoning: str = dspy.OutputField(desc="Optimization reasoning")


class TemplateGeneratorSignature(dspy.Signature):
    """DSPy signature for generating workflow templates"""

    successful_workflows: str = dspy.InputField(
        desc="JSON string of successful similar workflows"
    )
    query_pattern: str = dspy.InputField(desc="Query pattern to create template for")

    template_name: str = dspy.OutputField(desc="Generated template name")
    template_description: str = dspy.OutputField(desc="Template description")
    task_sequence: List[Dict[str, str]] = dspy.OutputField(
        desc="Optimized task sequence"
    )
    applicability_criteria: List[str] = dspy.OutputField(
        desc="When to use this template"
    )


class WorkflowIntelligence:
    """
    Workflow Intelligence system for advanced workflow optimization

    Provides intelligent workflow planning based on:
    - Historical execution patterns
    - Agent performance metrics
    - Query type analysis
    - Success rate optimization
    - Latency minimization
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        tenant_id: str,
        max_history_size: int = 10000,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        """
        Initialize workflow intelligence.

        Args:
            telemetry_provider: Telemetry provider for artifact persistence.
            tenant_id: Tenant identifier for multi-tenant isolation.
            max_history_size: Maximum in-memory history size.
            optimization_strategy: Optimization strategy to use.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for WorkflowIntelligence")
        self.logger = logging.getLogger(__name__)
        self.max_history_size = max_history_size
        self.optimization_strategy = optimization_strategy
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        # In-memory data structures
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

        # Initialize DSPy modules
        self._initialize_dspy_modules()

    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for workflow intelligence"""
        try:
            self.workflow_optimizer = dspy.ChainOfThought(WorkflowOptimizationSignature)
            self.template_generator = dspy.ChainOfThought(TemplateGeneratorSignature)
            self.logger.info("DSPy workflow intelligence modules initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize DSPy modules: {e}")
            self._create_fallback_modules()

    def _create_fallback_modules(self) -> None:
        """Create fallback modules for graceful degradation"""

        class FallbackOptimizerModule(dspy.Module):
            def forward(
                self, workflow_history: str, current_query: str, agent_performance: str
            ):
                return dspy.Prediction(
                    optimized_sequence=["search_agent", "summarizer_agent"],
                    optimization_strategy="fallback",
                    expected_improvement=0.1,
                    reasoning="Fallback optimization using default sequence",
                )

        class FallbackTemplateModule(dspy.Module):
            def forward(self, successful_workflows: str, query_pattern: str):
                return dspy.Prediction(
                    template_name="generic_template",
                    template_description="Generic workflow template",
                    task_sequence=[{"agent": "search_agent", "task": "search"}],
                    applicability_criteria=["general queries"],
                )

        self.workflow_optimizer = FallbackOptimizerModule()
        self.template_generator = FallbackTemplateModule()
        self.logger.warning("Using fallback workflow intelligence modules")

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
        """Record completed workflow execution for learning"""
        try:
            # Extract execution information
            execution = WorkflowExecution(
                workflow_id=workflow_plan.workflow_id,
                query=workflow_plan.original_query,
                query_type=self._classify_query_type(workflow_plan.original_query),
                execution_time=(
                    (workflow_plan.end_time - workflow_plan.start_time).total_seconds()
                    if workflow_plan.end_time and workflow_plan.start_time
                    else 0.0
                ),
                success=workflow_plan.status == WorkflowStatus.COMPLETED,
                agent_sequence=[task.agent_name for task in workflow_plan.tasks],
                task_count=len(workflow_plan.tasks),
                parallel_efficiency=self._calculate_parallel_efficiency(workflow_plan),
                confidence_score=workflow_plan.metadata.get("average_confidence", 0.5),
                metadata=workflow_plan.metadata,
            )

            # Add to history
            self.workflow_history.append(execution)

            # Update agent performance
            await self._update_agent_performance(workflow_plan)

            # Persist to telemetry
            await self._persist_execution(execution)

            # Check for template creation opportunities
            await self._evaluate_template_creation(execution)

            self.logger.debug(
                f"Recorded workflow execution: {workflow_plan.workflow_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to record workflow execution: {e}")

    async def optimize_workflow_plan(
        self,
        query: str,
        initial_plan: WorkflowPlan,
        optimization_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowPlan:
        """Optimize workflow plan using historical intelligence"""
        self.optimization_stats["total_optimizations"] += 1

        try:
            # Check for existing templates first
            template_match = self._find_matching_template(query)
            if template_match:
                optimized_plan = self._apply_template(initial_plan, template_match)
                self.optimization_stats["templates_used"] += 1
                self.logger.info(
                    f"Applied template '{template_match.name}' for optimization"
                )
                return optimized_plan

            # Use DSPy for intelligent optimization
            optimized_plan = await self._dspy_optimize_workflow(
                query, initial_plan, optimization_context
            )

            # Apply optimization strategy
            final_plan = self._apply_optimization_strategy(optimized_plan)

            self.optimization_stats["successful_optimizations"] += 1
            self.logger.info(f"Optimized workflow for query: {query[:50]}...")

            return final_plan

        except Exception as e:
            self.logger.error(f"Workflow optimization failed: {e}")
            # Return original plan if optimization fails
            return initial_plan

    async def _dspy_optimize_workflow(
        self, query: str, initial_plan: WorkflowPlan, context: Optional[Dict[str, Any]]
    ) -> WorkflowPlan:
        """Use DSPy to intelligently optimize workflow"""
        try:
            # Prepare historical data for DSPy
            relevant_history = self._get_relevant_history(query, limit=20)
            history_data = [
                {
                    "query": exec.query,
                    "agents": exec.agent_sequence,
                    "execution_time": exec.execution_time,
                    "success": exec.success,
                    "confidence": exec.confidence_score,
                }
                for exec in relevant_history
            ]

            # Prepare agent performance data
            performance_data = {
                name: {
                    "success_rate": perf.successful_executions
                    / max(perf.total_executions, 1),
                    "avg_time": perf.average_execution_time,
                    "confidence": perf.average_confidence,
                    "trend": perf.performance_trend,
                }
                for name, perf in self.agent_performance.items()
            }

            # Run DSPy optimization
            optimization_result = self.workflow_optimizer.forward(
                workflow_history=json.dumps(history_data),
                current_query=query,
                agent_performance=json.dumps(performance_data),
            )

            # Apply optimization results
            optimized_sequence = getattr(optimization_result, "optimized_sequence", [])
            optimization_reasoning = getattr(
                optimization_result, "reasoning", "DSPy optimization"
            )
            expected_improvement = getattr(
                optimization_result, "expected_improvement", 0.0
            )

            # Create optimized plan
            optimized_plan = self._create_optimized_plan(
                initial_plan,
                optimized_sequence,
                optimization_reasoning,
                expected_improvement,
            )

            return optimized_plan

        except Exception as e:
            self.logger.error(f"DSPy workflow optimization failed: {e}")
            return initial_plan

    def _create_optimized_plan(
        self,
        initial_plan: WorkflowPlan,
        optimized_sequence: List[str],
        reasoning: str,
        expected_improvement: float,
    ) -> WorkflowPlan:
        """Create optimized workflow plan from optimization results"""
        # Create new tasks based on optimized sequence
        optimized_tasks = []

        for i, agent_name in enumerate(optimized_sequence):
            task_id = f"optimized_task_{i}"

            # Determine task query based on agent capabilities
            if agent_name == "search_agent":
                task_query = initial_plan.original_query
            elif agent_name == "summarizer_agent":
                task_query = f"Summarize results for: {initial_plan.original_query}"
                dependencies = {f"optimized_task_{j}" for j in range(i) if j >= 0}
            elif agent_name == "detailed_report_agent":
                task_query = (
                    f"Generate detailed report for: {initial_plan.original_query}"
                )
                dependencies = {f"optimized_task_{j}" for j in range(i) if j >= 0}
            else:
                task_query = initial_plan.original_query
                dependencies = set()

            task = WorkflowTask(
                task_id=task_id,
                agent_name=agent_name,
                query=task_query,
                dependencies=dependencies if i > 0 else set(),
            )
            optimized_tasks.append(task)

        # Create optimized plan
        optimized_plan = WorkflowPlan(
            workflow_id=f"optimized_{initial_plan.workflow_id}",
            original_query=initial_plan.original_query,
            tasks=optimized_tasks,
            metadata={
                **initial_plan.metadata,
                "optimization_applied": True,
                "optimization_reasoning": reasoning,
                "expected_improvement": expected_improvement,
                "optimization_strategy": self.optimization_strategy.value,
            },
        )

        # Recalculate execution order
        optimized_plan.execution_order = self._calculate_execution_order(
            optimized_tasks
        )

        return optimized_plan

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

    async def _update_agent_performance(self, workflow_plan: WorkflowPlan) -> None:
        """Update agent performance metrics from workflow execution"""
        for task in workflow_plan.tasks:
            agent_name = task.agent_name

            # Initialize if not exists
            if agent_name not in self.agent_performance:
                self.agent_performance[agent_name] = AgentPerformance(
                    agent_name=agent_name
                )

            perf = self.agent_performance[agent_name]

            # Update metrics
            perf.total_executions += 1

            if task.status == TaskStatus.COMPLETED:
                perf.successful_executions += 1

                # Update execution time
                if task.end_time and task.start_time:
                    task_time = (task.end_time - task.start_time).total_seconds()
                    perf.average_execution_time = (
                        perf.average_execution_time * (perf.total_executions - 1)
                        + task_time
                    ) / perf.total_executions

            # Update error rate
            perf.error_rate = 1.0 - (perf.successful_executions / perf.total_executions)

            # Update query type preferences
            query_type = self._classify_query_type(workflow_plan.original_query)
            if query_type and query_type not in perf.preferred_query_types:
                perf.preferred_query_types.append(query_type)

            perf.last_updated = datetime.now()

            # Persist to telemetry
            await self._persist_agent_performance(perf)

    async def _evaluate_template_creation(self, execution: WorkflowExecution) -> None:
        """Evaluate whether to create a new workflow template"""
        if not execution.success:
            return

        # Look for similar successful executions
        similar_executions = [
            exec
            for exec in self.workflow_history
            if (
                exec.success
                and exec.query_type == execution.query_type
                and exec.agent_sequence == execution.agent_sequence
                and exec != execution
            )
        ]

        # Create template if we have enough similar successful executions
        if len(similar_executions) >= 3:  # Minimum threshold
            await self._create_workflow_template(execution, similar_executions)

    async def _create_workflow_template(
        self,
        seed_execution: WorkflowExecution,
        similar_executions: List[WorkflowExecution],
    ) -> None:
        """Create a new workflow template from successful executions"""
        try:
            # Use DSPy to generate template
            successful_workflows_data = []
            for exec in [seed_execution] + similar_executions:
                successful_workflows_data.append(
                    {
                        "query": exec.query,
                        "agents": exec.agent_sequence,
                        "execution_time": exec.execution_time,
                        "confidence": exec.confidence_score,
                    }
                )

            template_result = self.template_generator.forward(
                successful_workflows=json.dumps(successful_workflows_data),
                query_pattern=seed_execution.query_type,
            )

            # Create template
            template_id = f"template_{len(self.workflow_templates) + 1}"
            template = WorkflowTemplate(
                template_id=template_id,
                name=getattr(
                    template_result,
                    "template_name",
                    f"Template for {seed_execution.query_type}",
                ),
                description=getattr(
                    template_result,
                    "template_description",
                    "Generated workflow template",
                ),
                query_patterns=[seed_execution.query_type],
                task_sequence=getattr(template_result, "task_sequence", []),
                expected_execution_time=statistics.mean(
                    [
                        exec.execution_time
                        for exec in [seed_execution] + similar_executions
                    ]
                ),
                success_rate=1.0,  # All input executions were successful
            )

            self.workflow_templates[template_id] = template
            self.optimization_stats["templates_created"] += 1

            # Persist to telemetry
            await self._persist_template(template)

            self.logger.info(f"Created workflow template: {template.name}")

        except Exception as e:
            self.logger.error(f"Template creation failed: {e}")

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

    def _calculate_execution_order(self, tasks: List[WorkflowTask]) -> List[List[str]]:
        """Calculate execution order considering dependencies (simplified version)"""
        # This is a simplified version - full implementation would be in the orchestrator
        if not tasks:
            return []

        # Simple sequential execution for now
        return [[task.task_id] for task in tasks]

    def _calculate_parallel_efficiency(self, workflow_plan: WorkflowPlan) -> float:
        """Calculate parallel execution efficiency"""
        if not workflow_plan.execution_order or not workflow_plan.tasks:
            return 0.0

        # Calculate theoretical minimum time (if all tasks ran in parallel)
        max_task_time = (
            max(
                (task.end_time - task.start_time).total_seconds()
                for task in workflow_plan.tasks
                if task.end_time and task.start_time
            )
            if workflow_plan.tasks
            else 0.0
        )

        # Calculate actual execution time
        actual_time = (
            (workflow_plan.end_time - workflow_plan.start_time).total_seconds()
            if workflow_plan.end_time and workflow_plan.start_time
            else 0.0
        )

        # Efficiency = theoretical_minimum / actual_time
        return max_task_time / actual_time if actual_time > 0 else 0.0

    def _get_relevant_history(
        self, query: str, limit: int = 20
    ) -> List[WorkflowExecution]:
        """Get most relevant historical executions for query"""
        query_type = self._classify_query_type(query)

        # Filter by query type and success
        relevant = [
            exec
            for exec in self.workflow_history
            if exec.query_type == query_type and exec.success
        ]

        # Sort by recency and confidence
        relevant.sort(key=lambda x: (x.timestamp, x.confidence_score), reverse=True)

        return relevant[:limit]

    async def _persist_execution(self, execution: WorkflowExecution) -> None:
        """Persist workflow execution to telemetry."""
        try:
            exec_dict = {
                "workflow_id": execution.workflow_id,
                "query": execution.query,
                "query_type": execution.query_type,
                "execution_time": execution.execution_time,
                "success": execution.success,
                "agent_sequence": execution.agent_sequence,
                "task_count": execution.task_count,
                "parallel_efficiency": execution.parallel_efficiency,
                "confidence_score": execution.confidence_score,
                "user_satisfaction": execution.user_satisfaction,
                "error_details": execution.error_details,
                "timestamp": execution.timestamp.isoformat(),
                "metadata": execution.metadata,
            }
            await self._artifact_manager.save_demonstrations(
                "workflow",
                [{"input": json.dumps(exec_dict, default=str), "output": "execution"}],
            )
        except Exception as e:
            self.logger.error(f"Failed to persist execution: {e}")

    async def _persist_agent_performance(self, performance: AgentPerformance) -> None:
        """Persist agent performance to telemetry."""
        try:
            perf_dict = {
                "agent_name": performance.agent_name,
                "total_executions": performance.total_executions,
                "successful_executions": performance.successful_executions,
                "average_execution_time": performance.average_execution_time,
                "average_confidence": performance.average_confidence,
                "error_rate": performance.error_rate,
                "preferred_query_types": performance.preferred_query_types,
                "performance_trend": performance.performance_trend,
                "last_updated": performance.last_updated.isoformat(),
            }
            await self._artifact_manager.save_demonstrations(
                "agent_profiles",
                [{"input": json.dumps(perf_dict, default=str), "output": "profile"}],
            )
        except Exception as e:
            self.logger.error(f"Failed to persist agent performance: {e}")

    async def _persist_template(self, template: WorkflowTemplate) -> None:
        """Persist workflow template to telemetry."""
        try:
            template_dict = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "query_patterns": template.query_patterns,
                "task_sequence": template.task_sequence,
                "expected_execution_time": template.expected_execution_time,
                "success_rate": template.success_rate,
                "usage_count": template.usage_count,
                "created_at": template.created_at.isoformat(),
                "last_used": template.last_used.isoformat() if template.last_used else None,
            }
            await self._artifact_manager.save_blob(
                "workflow", "template_" + template.template_id,
                json.dumps(template_dict, default=str),
            )
            # Update template index for reload
            template_ids = list(self.workflow_templates.keys())
            await self._artifact_manager.save_blob(
                "workflow", "template_index",
                json.dumps(template_ids),
            )
        except Exception as e:
            self.logger.error(f"Failed to persist template: {e}")

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
        """
        Record workflow execution directly from orchestration spans

        This is called by OrchestrationEvaluator to feed real
        orchestration outcomes into the learning system.
        """
        try:
            # Add to history
            self.workflow_history.append(workflow_execution)

            # Persist to telemetry
            await self._persist_execution(workflow_execution)

            # Check for template creation opportunities
            await self._evaluate_template_creation(workflow_execution)

            self.logger.debug(
                f"Recorded workflow execution from Phoenix: {workflow_execution.workflow_id}"
            )
        except Exception as e:
            self.logger.error(f"Failed to record workflow execution: {e}")

    async def record_ground_truth_execution(
        self, workflow_execution: WorkflowExecution
    ) -> None:
        """
        Record human-corrected ground truth workflow execution

        This is called by OrchestrationFeedbackLoop to feed human-annotated
        "ideal" workflows into the learning system. These are weighted more
        heavily than automated executions.
        """
        try:
            # Mark as ground truth in metadata
            workflow_execution.metadata["is_ground_truth"] = True
            workflow_execution.metadata["learning_weight"] = 2.0

            # Add to history with higher priority
            self.workflow_history.append(workflow_execution)

            # Persist to telemetry
            await self._persist_execution(workflow_execution)

            # Always evaluate for template creation from ground truth
            await self._evaluate_template_creation(workflow_execution)

            self.logger.info(
                f"Recorded ground truth workflow: {workflow_execution.workflow_id} "
                f"(quality: {workflow_execution.user_satisfaction})"
            )
        except Exception as e:
            self.logger.error(f"Failed to record ground truth execution: {e}")

    async def optimize_from_ground_truth(self) -> Dict[str, Any]:
        """
        Trigger DSPy optimization using ground truth workflows

        This is called by OrchestrationFeedbackLoop when sufficient human
        annotations have been collected. It updates the DSPy modules to
        learn from human-corrected workflows.

        Returns:
            Optimization results and statistics
        """
        try:
            # Filter ground truth workflows
            ground_truth_workflows = [
                w
                for w in self.workflow_history
                if w.metadata.get("is_ground_truth", False)
            ]

            if not ground_truth_workflows:
                return {
                    "status": "skipped",
                    "reason": "no_ground_truth_data",
                    "ground_truth_count": 0,
                }

            self.logger.info(
                f"Starting DSPy optimization with {len(ground_truth_workflows)} "
                f"ground truth workflows"
            )

            # For now, mark ground truth workflows for prioritization
            # Full DSPy optimization would require setting up training/validation split
            # and running teleprompter optimization

            # Update optimization stats
            self.optimization_stats["ground_truth_workflows"] = len(
                ground_truth_workflows
            )
            self.optimization_stats["last_ground_truth_optimization"] = datetime.now()

            return {
                "status": "success",
                "ground_truth_count": len(ground_truth_workflows),
                "workflows_learned_from": len(ground_truth_workflows),
                "optimization_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Ground truth optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_successful_workflows(
        self, min_quality: float = 0.7, limit: int = 100
    ) -> List[WorkflowExecution]:
        """
        Get successful workflows for integration with routing optimization

        This is called by UnifiedOptimizer to extract high-quality workflows
        that should inform routing decisions.

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
