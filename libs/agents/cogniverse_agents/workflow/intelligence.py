"""
Workflow Intelligence - Template loader, profile provider, and execution recorder

Loads workflow templates and agent performance profiles from artifacts at startup.
Provides template matching for the orchestrator. Records workflow executions to
in-memory history (used by OrchestrationEvaluator in batch jobs). Does NOT run
DSPy optimization inline.
"""

import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from cogniverse_agents.workflow_types import (
    WorkflowPlan,
    WorkflowTask,
)
from cogniverse_core.registries import WorkflowStoreRegistry
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowTemplate,
)


class OptimizationStrategy(Enum):
    """Workflow optimization strategies"""

    PERFORMANCE_BASED = "performance_based"
    SUCCESS_RATE_BASED = "success_rate_based"
    LATENCY_OPTIMIZED = "latency_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


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
        tenant_id: str,
        max_history_size: int = 10000,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required for WorkflowIntelligence")
        self.logger = logging.getLogger(__name__)
        self.tenant_id = tenant_id
        self.optimization_strategy = optimization_strategy
        # Persistence goes through the WorkflowStore abstraction, resolved via
        # the registry (same path the backend/adapter registries use). The
        # telemetry impl rides ArtifactManager → Phoenix and resolves the
        # per-tenant provider itself, so no provider is threaded through here.
        self._store = WorkflowStoreRegistry.get(name="telemetry")

        # In-memory data structures. Loaded at startup; read-only on the
        # serving path (batch jobs append learned query patterns via
        # record_execution before persisting the corpus).
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
        """Load and publish one complete workflow-learning generation."""
        try:
            state = await self._store.load_learning_state(self.tenant_id)
            executions = [
                execution
                for execution in state.executions
                if self._validate_outcome_metadata(execution)
            ]
            staged_state = {
                "workflow_history": deque(
                    executions,
                    maxlen=self.workflow_history.maxlen,
                ),
                "agent_performance": {
                    profile.agent_name: profile for profile in state.profiles
                },
                "query_type_patterns": defaultdict(
                    list,
                    {
                        query_type: list(patterns)
                        for query_type, patterns in state.patterns.items()
                    },
                ),
                "workflow_templates": {
                    template.template_id: template for template in state.templates
                },
            }

            self.__dict__.update(staged_state)

            self.logger.info(
                f"Loaded {len(self.workflow_history)} executions, "
                f"{len(self.agent_performance)} agent profiles, "
                f"{len(self.workflow_templates)} templates"
            )

        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            raise

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
            # Calculate similarity score with template patterns, extended by
            # the learned queries of the template's own dominant type — a
            # phrasing that succeeded before matches even when no built-in
            # pattern does.
            candidate_patterns = list(template.query_patterns)
            candidate_patterns += self.query_type_patterns.get(
                self._template_query_type(template), []
            )
            score = 0.0
            for pattern in candidate_patterns:
                pattern_words = set(pattern.lower().split())
                query_words = set(query_lower.split())

                # Simple Jaccard similarity
                intersection = len(pattern_words & query_words)
                union = len(pattern_words | query_words)

                if union > 0:
                    pattern_score = intersection / union
                    score = max(score, pattern_score)

            weighted_score = (
                score
                if template.success_rate is None
                else score * (0.7 + 0.3 * template.success_rate)
            )

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
        template.last_used = datetime.now(timezone.utc)

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

                task.metadata["performance_score"] = composite_score

        plan.metadata["performance_optimized"] = True
        return plan

    def _template_query_type(self, template: WorkflowTemplate) -> str:
        """Dominant query type of a template, by classifying its built-in
        patterns (majority vote, first winner on ties). Links learned
        queries to templates without adding a schema field."""
        votes: Dict[str, int] = {}
        for pattern in template.query_patterns:
            pattern_type = self._classify_query_type(pattern)
            votes[pattern_type] = votes.get(pattern_type, 0) + 1
        if not votes:
            return "general"
        return max(votes, key=votes.get)

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
        """Group tasks into dependency-ordered phases.

        Each phase contains every task whose intra-plan dependencies are
        satisfied by earlier phases, so independent tasks run in parallel
        (layered topological sort / Kahn's algorithm by layers). Dependencies
        on task ids not in this plan are ignored. A dependency cycle can't be
        layered, so any tasks left when nothing is ready are emitted together
        in a final phase rather than looping forever.
        """
        if not tasks:
            return []

        task_ids = {t.task_id for t in tasks}
        remaining = {
            t.task_id: {d for d in t.dependencies if d in task_ids} for t in tasks
        }

        phases: List[List[str]] = []
        done: set = set()
        while remaining:
            ready = [tid for tid, deps in remaining.items() if deps <= done]
            if not ready:
                # Unsatisfiable (cycle / dangling dep) — emit the rest at once.
                ready = list(remaining)
            phases.append(sorted(ready))
            done.update(ready)
            for tid in ready:
                remaining.pop(tid, None)
        return phases

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

    _REQUIRED_FIELD_SEMANTICS = {
        False: {
            "execution_time": "unobserved_zero_sentinel",
            "success": "unobserved_false_sentinel",
            "parallel_efficiency": "unobserved_zero_sentinel",
            "confidence_score": "unobserved_zero_sentinel",
        },
        True: {
            "execution_time": "observed_duration_seconds",
            "success": "observed_execution_outcome",
            "parallel_efficiency": "observed_parallel_efficiency",
            "confidence_score": "observed_confidence_score",
        },
    }
    _OBSERVED_REQUIRED_FIELD_SEMANTICS = {
        "execution_time": {"observed_duration_seconds"},
        "success": {"observed_execution_outcome"},
        "parallel_efficiency": {
            "observed_parallel_efficiency",
            "unobserved_zero_sentinel",
        },
        "confidence_score": {
            "observed_confidence_score",
            "unobserved_zero_sentinel",
        },
    }
    _UNOBSERVED_SENTINELS = {
        "execution_time": 0.0,
        "success": False,
        "parallel_efficiency": 0.0,
        "confidence_score": 0.0,
    }

    def _validate_outcome_metadata(self, workflow_execution: WorkflowExecution) -> bool:
        outcome_metadata = workflow_execution.metadata.get("_outcome_metadata")
        if outcome_metadata is None:
            raise ValueError("metadata must contain _outcome_metadata")
        if not isinstance(outcome_metadata, dict):
            raise ValueError("metadata._outcome_metadata must be a dict")

        observed = outcome_metadata.get("observed")
        if type(observed) is not bool:
            raise ValueError("metadata._outcome_metadata.observed must be a bool")

        semantics = outcome_metadata.get("required_field_semantics")
        if not isinstance(semantics, dict):
            raise ValueError(
                "metadata._outcome_metadata.required_field_semantics must be a dict"
            )
        if observed:
            semantics_valid = set(semantics) == set(
                self._OBSERVED_REQUIRED_FIELD_SEMANTICS
            ) and all(
                semantics[field] in allowed
                for field, allowed in self._OBSERVED_REQUIRED_FIELD_SEMANTICS.items()
            )
        else:
            semantics_valid = semantics == self._REQUIRED_FIELD_SEMANTICS[False]
        if not semantics_valid:
            observed_label = str(observed).lower()
            raise ValueError(
                "metadata._outcome_metadata.required_field_semantics must exactly "
                f"match the observed={observed_label} contract"
            )

        for field, semantic in semantics.items():
            if semantic.startswith("unobserved_") and (
                getattr(workflow_execution, field) != self._UNOBSERVED_SENTINELS[field]
            ):
                raise ValueError(
                    f"unobserved workflow execution field {field} must equal "
                    "its declared sentinel"
                )
        return observed

    async def record_execution(self, workflow_execution: WorkflowExecution) -> None:
        """Record workflow execution directly (called by OrchestrationEvaluator in batch jobs).

        Successful executions also feed the learned query-pattern corpus:
        the query is classified and kept (deduplicated, bounded) under its
        type, so template matching recognises phrasings that actually
        succeeded for this tenant, not only each template's built-in
        patterns. The batch save persists the corpus; the serving path
        loads it read-only at startup.
        """
        if not self._validate_outcome_metadata(workflow_execution):
            self.logger.debug(
                "Ignored unobserved workflow plan: %s",
                workflow_execution.workflow_id,
            )
            return

        self.workflow_history.append(workflow_execution)
        if workflow_execution.success and workflow_execution.query.strip():
            self._learn_query_pattern(workflow_execution.query)
        self.logger.debug(
            "Recorded workflow execution: %s", workflow_execution.workflow_id
        )

    def derive_learning_artifacts(
        self,
        executions: List[WorkflowExecution],
    ) -> tuple[List[AgentPerformance], List[WorkflowTemplate]]:
        """Derive serving profiles and replayable templates from executions."""
        now = datetime.now(timezone.utc)
        by_agent: Dict[
            str,
            List[tuple[WorkflowExecution, Dict[str, Any]]],
        ] = defaultdict(list)
        by_workflow: Dict[tuple[str, str, tuple[str, ...]], List[WorkflowExecution]] = (
            defaultdict(list)
        )

        for execution in executions:
            if len(execution.agent_sequence) != len(set(execution.agent_sequence)):
                raise ValueError(
                    f"workflow {execution.workflow_id!r} has duplicate agent names"
                )
            for observation in self._agent_profile_observations(execution):
                by_agent[observation["agent_name"]].append((execution, observation))
            pattern = execution.metadata.get("orchestration_pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    f"workflow {execution.workflow_id!r} has no orchestration pattern"
                )
            by_workflow[
                (execution.query_type, pattern, tuple(execution.agent_sequence))
            ].append(execution)

        profiles = []
        for agent_name, samples in sorted(by_agent.items()):
            confidence_samples = [
                observation["confidence"]
                for _, observation in samples
                if "confidence" in observation
            ]
            if not confidence_samples:
                continue
            successful = [
                (execution, observation)
                for execution, observation in samples
                if observation["success"]
            ]
            query_counts: Dict[str, int] = defaultdict(int)
            for execution, _ in successful:
                query_counts[execution.query_type] += 1
            preferred_query_types = sorted(
                query_counts,
                key=lambda query_type: (-query_counts[query_type], query_type),
            )
            profiles.append(
                AgentPerformance(
                    agent_name=agent_name,
                    total_executions=len(samples),
                    successful_executions=len(successful),
                    average_execution_time=float(
                        statistics.fmean(
                            observation["execution_time"] for _, observation in samples
                        )
                    ),
                    average_confidence=float(statistics.fmean(confidence_samples)),
                    error_rate=float((len(samples) - len(successful)) / len(samples)),
                    preferred_query_types=preferred_query_types,
                    performance_trend="stable",
                    last_updated=now,
                )
            )

        templates = []
        for (query_type, pattern, agent_sequence), samples in sorted(
            by_workflow.items()
        ):
            successful = [sample for sample in samples if sample.success]
            if not successful or not agent_sequence:
                continue
            query_patterns = []
            seen_queries = set()
            for sample in successful:
                normalized = sample.query.strip()
                folded = normalized.casefold()
                if folded not in seen_queries:
                    seen_queries.add(folded)
                    query_patterns.append(normalized)
            signature = json.dumps(
                [query_type, pattern, list(agent_sequence)],
                separators=(",", ":"),
            )
            template_id = (
                f"workflow-{hashlib.sha256(signature.encode()).hexdigest()[:16]}"
            )
            task_sequence = []
            for index, agent_name in enumerate(agent_sequence):
                dependencies = (
                    []
                    if pattern == "parallel" or index == 0
                    else [f"template_task_{index - 1}"]
                )
                task_sequence.append(
                    {
                        "agent": agent_name,
                        "task": "process",
                        "dependencies": dependencies,
                    }
                )
            templates.append(
                WorkflowTemplate(
                    template_id=template_id,
                    name=f"{query_type} {pattern} workflow",
                    description="Workflow learned from checked orchestration outcomes",
                    query_patterns=query_patterns,
                    task_sequence=task_sequence,
                    expected_execution_time=float(
                        statistics.fmean(sample.execution_time for sample in samples)
                    ),
                    success_rate=float(len(successful) / len(samples)),
                    usage_count=0,
                    created_at=now,
                )
            )

        self.agent_performance = {profile.agent_name: profile for profile in profiles}
        self.workflow_templates = {
            template.template_id: template for template in templates
        }
        self.optimization_stats["templates_created"] = len(templates)
        return profiles, templates

    @staticmethod
    def _agent_profile_observations(
        execution: WorkflowExecution,
    ) -> List[Dict[str, Any]]:
        raw_observations = execution.metadata.get("agent_observations")
        if raw_observations is None:
            return []
        if not isinstance(raw_observations, list):
            raise ValueError(
                f"workflow {execution.workflow_id!r} agent_observations must be a list"
            )
        allowed_agents = set(execution.agent_sequence)
        observations = []
        for raw in raw_observations:
            if not isinstance(raw, dict):
                raise ValueError(
                    f"workflow {execution.workflow_id!r} agent observation must be "
                    "a dict"
                )
            required = {"agent_name", "execution_time", "success"}
            allowed = required | {"confidence"}
            if set(raw) - allowed or not required <= set(raw):
                raise ValueError(
                    f"workflow {execution.workflow_id!r} agent observation must "
                    "contain agent_name, execution_time, success, and optional "
                    "confidence"
                )
            agent_name = raw["agent_name"]
            if not isinstance(agent_name, str) or agent_name not in allowed_agents:
                raise ValueError(
                    f"workflow {execution.workflow_id!r} agent observation "
                    "references an agent outside agent_sequence"
                )
            execution_time = raw["execution_time"]
            if (
                type(execution_time) is not float
                or not math.isfinite(execution_time)
                or execution_time < 0.0
            ):
                raise ValueError(
                    f"workflow {execution.workflow_id!r} agent observation "
                    "execution_time must be a non-negative finite float"
                )
            if type(raw["success"]) is not bool:
                raise ValueError(
                    f"workflow {execution.workflow_id!r} agent observation success "
                    "must be a bool"
                )
            if "confidence" in raw:
                confidence = raw["confidence"]
                if (
                    type(confidence) is not float
                    or not math.isfinite(confidence)
                    or not 0.0 <= confidence <= 1.0
                ):
                    raise ValueError(
                        f"workflow {execution.workflow_id!r} agent observation "
                        "confidence must be a finite float between 0 and 1"
                    )
            observations.append(dict(raw))
        return observations

    _MAX_LEARNED_PATTERNS_PER_TYPE = 50

    def _learn_query_pattern(self, query: str) -> None:
        query_type = self._classify_query_type(query)
        patterns = self.query_type_patterns[query_type]
        normalized = query.strip()
        if any(existing.lower() == normalized.lower() for existing in patterns):
            return
        patterns.append(normalized)
        overflow = len(patterns) - self._MAX_LEARNED_PATTERNS_PER_TYPE
        if overflow > 0:
            del patterns[:overflow]

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
        agents_config: Dict[str, Any],
        count: int = 100,
        backend: Optional[Any] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Any] = None,
    ) -> int:
        """
        Generate synthetic training data using libs/synthetic system

        Args:
            agents_config: Explicit agents section from the active configuration
            count: Number of synthetic examples to generate
            backend: Optional Backend instance for content sampling
            backend_config: Backend configuration with profiles
            generator_config: Optional SyntheticGeneratorConfig for DSPy modules

        Returns:
            Number of distinct generated plans persisted as serving templates.
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
            agents_config=agents_config,
        )
        request = SyntheticDataRequest(
            optimizer="workflow", count=count, tenant_id=self.tenant_id
        )
        response = await service.generate(request)

        response_data = response.data
        response_count = getattr(response, "count", None)
        if response_count != count or len(response_data) != count:
            raise RuntimeError(
                f"Synthetic workflow response must contain exactly {count} plans: "
                f"count={response_count} rows={len(response_data)}"
            )

        templates = []
        seen_identities = set()
        for example_data in response_data:
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
                timestamp=example_data.get(
                    "timestamp",
                    datetime.now(timezone.utc),
                ),
                metadata=example_data.get("metadata", {}),
            )
            if self._validate_outcome_metadata(execution):
                raise ValueError(
                    "Synthetic workflow plans must not claim observed outcomes"
                )
            query = execution.query.strip()
            if not query:
                raise ValueError("Synthetic workflow plan query must be non-empty")
            if (
                not execution.agent_sequence
                or execution.task_count != len(execution.agent_sequence)
                or len(execution.agent_sequence) != len(set(execution.agent_sequence))
            ):
                raise ValueError(
                    "Synthetic workflow plan must contain one unique agent per task"
                )
            identity_payload = json.dumps(
                [execution.query_type, query, execution.agent_sequence],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if identity_payload in seen_identities:
                continue
            seen_identities.add(identity_payload)
            template_id = (
                "synthetic-workflow-"
                f"{hashlib.sha256(identity_payload.encode()).hexdigest()[:16]}"
            )
            task_sequence = [
                {
                    "agent": agent_name,
                    "task": "process",
                    "dependencies": (
                        [] if index == 0 else [f"template_task_{index - 1}"]
                    ),
                }
                for index, agent_name in enumerate(execution.agent_sequence)
            ]
            templates.append(
                WorkflowTemplate(
                    template_id=template_id,
                    name=f"{execution.query_type.lower()} generated workflow",
                    description="Workflow plan generated from tenant content",
                    query_patterns=[query],
                    task_sequence=task_sequence,
                    expected_execution_time=None,
                    success_rate=None,
                    created_at=execution.timestamp,
                )
            )

        if len(templates) != count:
            raise RuntimeError(
                "Synthetic workflow response produced "
                f"{len(templates)} unique grounded plans; expected {count}"
            )

        await self._persist_generated_templates(templates)
        self.logger.info(
            "Persisted %d generated workflow plans for tenant %s",
            len(templates),
            self.tenant_id,
        )
        return len(templates)

    async def _persist_generated_templates(
        self,
        templates: List[WorkflowTemplate],
    ) -> None:
        stored_ids = await self._store.save_generated_templates(
            self.tenant_id,
            templates,
        )
        expected_ids = [template.template_id for template in templates]
        if stored_ids != expected_ids:
            raise RuntimeError(
                "Workflow store returned the wrong generated template identities: "
                f"expected={expected_ids} actual={stored_ids}"
            )
        self.workflow_templates.update(
            {template.template_id: template for template in templates}
        )


def create_workflow_intelligence(
    tenant_id: str,
    max_history_size: int = 10000,
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
) -> WorkflowIntelligence:
    """Factory function to create workflow intelligence system"""
    return WorkflowIntelligence(
        tenant_id=tenant_id,
        max_history_size=max_history_size,
        optimization_strategy=optimization_strategy,
    )
