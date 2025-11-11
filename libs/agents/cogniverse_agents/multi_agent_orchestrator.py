"""
Multi-Agent Orchestrator for Complex Query Processing

This orchestrator coordinates multiple agents to handle complex queries that require
multiple processing steps, such as:
- Search + Summarization workflows
- Multi-step analysis pipelines
- Parallel processing with result aggregation
- Sequential agent chains with dependency management

Features:
- DSPy 3.0 powered workflow intelligence
- A2A protocol for agent communication
- Dependency resolution and execution planning
- Result aggregation and synthesis
- Error handling and recovery strategies
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.config import TelemetryConfig

# DSPy 3.0 imports
import dspy

# Enhanced routing imports
from cogniverse_agents.routing_agent import (
    RoutingAgent,
)

# A2A protocol imports
from cogniverse_agents.tools.a2a_utils import A2AClient

# Workflow intelligence (import after types to avoid circular dependency)
from cogniverse_agents.workflow_intelligence import (
    OptimizationStrategy,
    create_workflow_intelligence,
)

# Shared workflow types
from cogniverse_agents.workflow_types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)


class FusionStrategy(Enum):
    """Strategies for combining results from multiple agents across modalities"""

    SCORE_BASED = "score"  # Weight by confidence scores
    TEMPORAL = "temporal"  # Time-aligned fusion for temporal queries
    SEMANTIC = "semantic"  # Semantic similarity-based fusion
    HIERARCHICAL = "hierarchical"  # Structured combination with hierarchy
    SIMPLE = "simple"  # Basic concatenation (legacy)


class WorkflowPlannerSignature(dspy.Signature):
    """DSPy signature for workflow planning"""

    query: str = dspy.InputField(
        desc="Complex user query requiring multi-agent processing"
    )
    available_agents: str = dspy.InputField(
        desc="Available agents and their capabilities"
    )

    workflow_tasks: List[Dict[str, str]] = dspy.OutputField(
        desc="List of tasks with agent assignments and dependencies"
    )
    execution_strategy: str = dspy.OutputField(
        desc="Sequential, parallel, or hybrid execution strategy"
    )
    expected_outcome: str = dspy.OutputField(desc="Expected final result structure")
    reasoning: str = dspy.OutputField(desc="Workflow planning reasoning")


class ResultAggregatorSignature(dspy.Signature):
    """DSPy signature for cross-modal fusion of multi-agent results"""

    original_query: str = dspy.InputField(desc="Original user query")
    task_results: str = dspy.InputField(
        desc="JSON string of individual task results with modalities"
    )
    fusion_strategy: str = dspy.InputField(
        desc="Fusion strategy: score, temporal, semantic, hierarchical"
    )
    agent_modalities: str = dspy.InputField(
        desc="Modalities of each agent result (video, image, audio, document, text)"
    )

    aggregated_result: str = dspy.OutputField(desc="Synthesized final result")
    confidence_score: float = dspy.OutputField(desc="Confidence in aggregated result")
    fusion_quality: str = dspy.OutputField(
        desc="Fusion quality metrics: coverage, consistency, coherence"
    )
    cross_modal_consistency: str = dspy.OutputField(
        desc="Consistency analysis across modalities"
    )


class MultiAgentOrchestrator:
    """
    Multi-Agent Orchestrator for complex query processing

    Coordinates multiple agents to handle complex workflows that require:
    - Sequential processing chains
    - Parallel execution with result aggregation
    - Dependency management
    - Error recovery and fallback strategies
    """

    def __init__(
        self,
        tenant_id: str,
        telemetry_config: "TelemetryConfig",
        routing_agent: Optional[RoutingAgent] = None,
        available_agents: Optional[Dict[str, Dict[str, Any]]] = None,
        max_parallel_tasks: int = 3,
        workflow_timeout_minutes: int = 15,
        enable_workflow_intelligence: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        """
        Initialize Multi-Agent Orchestrator

        Args:
            tenant_id: Tenant identifier (REQUIRED)
            telemetry_config: Telemetry configuration (REQUIRED)
            routing_agent: Optional pre-configured routing agent
            available_agents: Dictionary of available agents
            max_parallel_tasks: Maximum parallel task execution
            workflow_timeout_minutes: Workflow timeout in minutes
            enable_workflow_intelligence: Enable workflow intelligence
            optimization_strategy: Optimization strategy

        Raises:
            ValueError: If tenant_id is empty
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")

        self.tenant_id = tenant_id
        self.logger = logging.getLogger(__name__)

        # Initialize routing agent
        self.routing_agent = routing_agent or RoutingAgent(
            tenant_id=tenant_id,
            telemetry_config=telemetry_config
        )

        # Configure available agents and their capabilities
        self.available_agents = available_agents or self._get_default_agents()

        # Orchestration settings
        self.max_parallel_tasks = max_parallel_tasks
        self.workflow_timeout = timedelta(minutes=workflow_timeout_minutes)

        # Initialize DSPy modules
        self._initialize_dspy_modules()

        # A2A client for agent communication
        self.a2a_client = A2AClient()

        # Initialize workflow intelligence
        self.enable_workflow_intelligence = enable_workflow_intelligence
        if enable_workflow_intelligence:
            self.workflow_intelligence = create_workflow_intelligence(
                optimization_strategy=optimization_strategy
            )
            self.logger.info("Workflow intelligence enabled")
        else:
            self.workflow_intelligence = None

        # Active workflows tracking
        self.active_workflows: Dict[str, WorkflowPlan] = {}

        # Statistics
        self.orchestration_stats = {
            "total_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "total_tasks_executed": 0,
            "agent_utilization": {},
        }

    def _get_default_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get default agent configuration"""
        return {
            "video_search_agent": {
                "capabilities": [
                    "video_content_search",
                    "visual_query_analysis",
                    "multimodal_retrieval",
                    "temporal_video_analysis",
                ],
                "endpoint": "http://localhost:8002",
                "timeout_seconds": 120,
                "parallel_capacity": 2,
            },
            "summarizer_agent": {
                "capabilities": [
                    "content_summarization",
                    "key_point_extraction",
                    "document_synthesis",
                    "report_generation",
                ],
                "endpoint": "http://localhost:8003",
                "timeout_seconds": 60,
                "parallel_capacity": 3,
            },
            "detailed_report_agent": {
                "capabilities": [
                    "comprehensive_analysis",
                    "detailed_reporting",
                    "data_correlation",
                    "in_depth_investigation",
                ],
                "endpoint": "http://localhost:8004",
                "timeout_seconds": 180,
                "parallel_capacity": 1,
            },
        }

    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for workflow planning and result aggregation"""
        try:
            self.workflow_planner = dspy.ChainOfThought(WorkflowPlannerSignature)
            self.result_aggregator = dspy.ChainOfThought(ResultAggregatorSignature)
            self.logger.info("DSPy orchestration modules initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize DSPy modules: {e}")
            # Create fallback modules
            self._create_fallback_modules()

    def _create_fallback_modules(self) -> None:
        """Create fallback modules for graceful degradation"""

        class FallbackPlannerModule(dspy.Module):
            def forward(self, query: str, available_agents: str):
                # Simple fallback: always route to video search first, then summarizer
                return dspy.Prediction(
                    workflow_tasks=[
                        {
                            "task_id": "search",
                            "agent": "video_search_agent",
                            "query": query,
                            "dependencies": [],
                        },
                        {
                            "task_id": "summarize",
                            "agent": "summarizer_agent",
                            "query": f"Summarize results for: {query}",
                            "dependencies": ["search"],
                        },
                    ],
                    execution_strategy="sequential",
                    expected_outcome="Search results followed by summary",
                    reasoning="Fallback workflow: search then summarize",
                )

        class FallbackAggregatorModule(dspy.Module):
            def forward(self, original_query: str, task_results: str):
                return dspy.Prediction(
                    aggregated_result=f"Combined results for query: {original_query}",
                    confidence_score=0.6,
                    synthesis_strategy="basic_concatenation",
                )

        self.workflow_planner = FallbackPlannerModule()
        self.result_aggregator = FallbackAggregatorModule()
        self.logger.warning("Using fallback orchestration modules")

    async def process_complex_query(
        self,
        query: str,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a complex query using multi-agent orchestration

        Args:
            query: Complex user query
            context: Additional context
            user_id: User identifier
            preferences: User preferences for processing

        Returns:
            Orchestrated result from multiple agents
        """
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        self.orchestration_stats["total_workflows"] += 1

        try:
            # Step 1: Plan the workflow
            workflow_plan = await self._plan_workflow(
                workflow_id, query, context, user_id, preferences
            )

            # Step 2: Execute the workflow
            await self._execute_workflow(workflow_plan)

            # Step 3: Aggregate and synthesize results
            final_result = await self._aggregate_results(workflow_plan)

            # Update statistics
            self._update_orchestration_stats(workflow_plan, success=True)

            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": final_result,
                "execution_summary": {
                    "total_tasks": len(workflow_plan.tasks),
                    "completed_tasks": len(
                        [
                            t
                            for t in workflow_plan.tasks
                            if t.status == TaskStatus.COMPLETED
                        ]
                    ),
                    "execution_time": (
                        workflow_plan.end_time - workflow_plan.start_time
                    ).total_seconds(),
                    "agents_used": list(set(t.agent_name for t in workflow_plan.tasks)),
                },
                "metadata": workflow_plan.metadata,
            }

        except Exception as e:
            self.logger.error(f"Orchestration failed for query '{query}': {e}")
            self.orchestration_stats["failed_workflows"] += 1

            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "fallback_result": await self._generate_fallback_result(query, context),
            }

    async def _plan_workflow(
        self,
        workflow_id: str,
        query: str,
        context: Optional[str],
        user_id: Optional[str],
        preferences: Optional[Dict[str, Any]],
    ) -> WorkflowPlan:
        """Plan workflow using DSPy-powered intelligent planning"""
        self.logger.info(f"Planning workflow {workflow_id} for query: {query}")

        try:
            # Prepare available agents information
            agents_info = []
            for agent_name, agent_config in self.available_agents.items():
                capabilities = ", ".join(agent_config["capabilities"])
                agents_info.append(f"{agent_name}: {capabilities}")
            available_agents_str = " | ".join(agents_info)

            # Use DSPy to plan the workflow
            planning_result = self.workflow_planner.forward(
                query=query, available_agents=available_agents_str
            )

            # Create workflow plan
            workflow_plan = WorkflowPlan(
                workflow_id=workflow_id,
                original_query=query,
                status=WorkflowStatus.PENDING,
                metadata={
                    "context": context,
                    "user_id": user_id,
                    "preferences": preferences or {},
                    "planning_reasoning": getattr(planning_result, "reasoning", ""),
                    "execution_strategy": getattr(
                        planning_result, "execution_strategy", "sequential"
                    ),
                },
            )

            # Parse workflow tasks from DSPy result
            tasks_data = getattr(planning_result, "workflow_tasks", [])
            tasks = []

            for i, task_data in enumerate(tasks_data):
                if isinstance(task_data, dict):
                    task_id = task_data.get("task_id", f"task_{i}")
                    agent_name = task_data.get("agent", "video_search_agent")
                    task_query = task_data.get("query", query)
                    dependencies = set(task_data.get("dependencies", []))
                else:
                    # Fallback parsing
                    task_id = f"task_{i}"
                    agent_name = "video_search_agent"
                    task_query = query
                    dependencies = set()

                task = WorkflowTask(
                    task_id=task_id,
                    agent_name=agent_name,
                    query=task_query,
                    dependencies=dependencies,
                    timeout_seconds=self.available_agents.get(agent_name, {}).get(
                        "timeout_seconds", 120
                    ),
                )
                tasks.append(task)

            workflow_plan.tasks = tasks
            workflow_plan.execution_order = self._calculate_execution_order(tasks)

            # Apply workflow intelligence optimization if available
            if self.workflow_intelligence:
                try:
                    optimized_plan = (
                        await self.workflow_intelligence.optimize_workflow_plan(
                            query,
                            workflow_plan,
                            {"user_id": user_id, "preferences": preferences},
                        )
                    )
                    workflow_plan = optimized_plan
                    self.logger.info("Applied workflow intelligence optimization")
                except Exception as e:
                    self.logger.warning(
                        f"Workflow optimization failed, using original plan: {e}"
                    )

            self.logger.info(
                f"Workflow planned: {len(workflow_plan.tasks)} tasks, "
                f"{len(workflow_plan.execution_order)} execution phases"
            )

            return workflow_plan

        except Exception as e:
            self.logger.error(f"Workflow planning failed: {e}")
            # Return simple fallback plan
            return self._create_fallback_workflow_plan(
                workflow_id, query, context, user_id
            )

    def _calculate_execution_order(self, tasks: List[WorkflowTask]) -> List[List[str]]:
        """Calculate optimal execution order considering dependencies"""
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        remaining_tasks = set(task.task_id for task in tasks)
        execution_order = []

        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_map[task_id]
                if task.dependencies.issubset(
                    set(
                        task.task_id
                        for task in tasks
                        if task.status == TaskStatus.COMPLETED
                    )
                    | set(task_id for phase in execution_order for task_id in phase)
                ):
                    ready_tasks.append(task_id)

            if not ready_tasks:
                # Handle circular dependencies by taking any remaining task
                ready_tasks = [next(iter(remaining_tasks))]
                self.logger.warning(
                    f"Potential circular dependency detected, forcing execution of {ready_tasks[0]}"
                )

            # Limit parallel execution
            parallel_batch = ready_tasks[: self.max_parallel_tasks]
            execution_order.append(parallel_batch)
            remaining_tasks -= set(parallel_batch)

        return execution_order

    async def _execute_workflow(self, workflow_plan: WorkflowPlan) -> bool:
        """Execute workflow according to the plan"""
        workflow_plan.status = WorkflowStatus.RUNNING
        workflow_plan.start_time = datetime.now()
        self.active_workflows[workflow_plan.workflow_id] = workflow_plan

        try:
            self.logger.info(f"Executing workflow {workflow_plan.workflow_id}")

            # Execute tasks in planned order
            for phase_num, task_ids in enumerate(workflow_plan.execution_order):
                self.logger.debug(f"Executing phase {phase_num + 1}: {task_ids}")

                # Run tasks in parallel within this phase
                phase_tasks = [
                    task for task in workflow_plan.tasks if task.task_id in task_ids
                ]
                phase_results = await asyncio.gather(
                    *[self._execute_task(task, workflow_plan) for task in phase_tasks],
                    return_exceptions=True,
                )

                # Check for failures in this phase
                failed_tasks = []
                for task, result in zip(phase_tasks, phase_results):
                    if isinstance(result, Exception):
                        task.status = TaskStatus.FAILED
                        task.error = str(result)
                        failed_tasks.append(task.task_id)
                        self.logger.error(f"Task {task.task_id} failed: {result}")

                # Decide whether to continue or abort
                if failed_tasks and not self._can_continue_with_failures(
                    workflow_plan, failed_tasks
                ):
                    workflow_plan.status = WorkflowStatus.FAILED
                    self.logger.error(
                        f"Workflow aborted due to critical task failures: {failed_tasks}"
                    )
                    return False

            # Mark as completed if we got here
            completed_tasks = [
                t for t in workflow_plan.tasks if t.status == TaskStatus.COMPLETED
            ]
            if completed_tasks:
                workflow_plan.status = WorkflowStatus.COMPLETED
            else:
                workflow_plan.status = WorkflowStatus.FAILED

            workflow_plan.end_time = datetime.now()
            self.logger.info(
                f"Workflow {workflow_plan.workflow_id} {workflow_plan.status.value}: "
                f"{len(completed_tasks)}/{len(workflow_plan.tasks)} tasks completed"
            )

            return workflow_plan.status == WorkflowStatus.COMPLETED

        except Exception as e:
            workflow_plan.status = WorkflowStatus.FAILED
            workflow_plan.end_time = datetime.now()
            self.logger.error(f"Workflow execution failed: {e}")
            return False

        finally:
            # Record workflow execution for intelligence learning
            if self.workflow_intelligence and workflow_plan.end_time:
                try:
                    await self.workflow_intelligence.record_workflow_execution(
                        workflow_plan
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to record workflow execution for learning: {e}"
                    )

            # Clean up active workflow
            self.active_workflows.pop(workflow_plan.workflow_id, None)

    async def _execute_task(
        self, task: WorkflowTask, workflow_plan: WorkflowPlan
    ) -> Dict[str, Any]:
        """Execute individual task by communicating with the assigned agent"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        self.orchestration_stats["total_tasks_executed"] += 1

        # Update agent utilization stats
        agent_name = task.agent_name
        if agent_name not in self.orchestration_stats["agent_utilization"]:
            self.orchestration_stats["agent_utilization"][agent_name] = 0
        self.orchestration_stats["agent_utilization"][agent_name] += 1

        try:
            self.logger.debug(f"Executing task {task.task_id} on {agent_name}")

            # Get agent endpoint
            agent_config = self.available_agents.get(agent_name, {})
            agent_endpoint = agent_config.get("endpoint", "http://localhost:8000")

            # Prepare task context with dependency results
            task_context = await self._prepare_task_context(task, workflow_plan)

            # Execute via A2A protocol
            result = await self.a2a_client.send_message(
                agent_endpoint,
                {
                    "query": task.query,
                    "context": task_context,
                    "task_id": task.task_id,
                    "workflow_id": workflow_plan.workflow_id,
                    "parameters": task.parameters,
                },
                timeout=task.timeout_seconds,
            )

            # Process successful result
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = datetime.now()

            self.logger.debug(f"Task {task.task_id} completed successfully")
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()

            self.logger.error(f"Task {task.task_id} failed: {e}")

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.READY
                self.logger.info(
                    f"Retrying task {task.task_id} (attempt {task.retry_count + 1})"
                )
                await asyncio.sleep(2**task.retry_count)  # Exponential backoff
                return await self._execute_task(task, workflow_plan)

            raise e

    async def _prepare_task_context(
        self, task: WorkflowTask, workflow_plan: WorkflowPlan
    ) -> str:
        """Prepare context for task execution including dependency results"""
        context_parts = []

        # Add original context
        if workflow_plan.metadata.get("context"):
            context_parts.append(
                f"Original context: {workflow_plan.metadata['context']}"
            )

        # Add results from dependency tasks
        for dep_task_id in task.dependencies:
            dep_task = next(
                (t for t in workflow_plan.tasks if t.task_id == dep_task_id), None
            )
            if dep_task and dep_task.result:
                # Summarize dependency result
                dep_summary = str(dep_task.result).replace("\n", " ")[:200]
                context_parts.append(f"Result from {dep_task_id}: {dep_summary}...")

        return " | ".join(context_parts)

    def _can_continue_with_failures(
        self, workflow_plan: WorkflowPlan, failed_task_ids: List[str]
    ) -> bool:
        """Determine if workflow can continue despite task failures"""
        # Simple heuristic: continue if less than 50% of tasks failed
        total_tasks = len(workflow_plan.tasks)
        failed_count = len(
            [t for t in workflow_plan.tasks if t.status == TaskStatus.FAILED]
        )

        return failed_count / total_tasks < 0.5

    async def _aggregate_results(self, workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """Cross-modal fusion of results from completed tasks"""
        completed_tasks = [
            t
            for t in workflow_plan.tasks
            if t.status == TaskStatus.COMPLETED and t.result
        ]

        if not completed_tasks:
            return {"error": "No completed tasks to aggregate"}

        try:
            # Prepare task results with modality detection
            task_results = {}
            agent_modalities = {}

            for task in completed_tasks:
                # Detect modality from agent name
                modality = self._detect_agent_modality(task.agent_name)
                agent_modalities[task.task_id] = modality

                task_results[task.task_id] = {
                    "agent": task.agent_name,
                    "modality": modality,
                    "query": task.query,
                    "result": task.result,
                    "execution_time": (task.end_time - task.start_time).total_seconds(),
                    "confidence": (
                        task.result.get("confidence", 0.5)
                        if isinstance(task.result, dict)
                        else 0.5
                    ),
                }

            # Select fusion strategy based on query and modalities
            fusion_strategy = self._select_fusion_strategy(
                workflow_plan.original_query, agent_modalities
            )

            # Apply modality-aware fusion
            if fusion_strategy == FusionStrategy.SCORE_BASED:
                fused_result = self._fuse_by_score(task_results)
            elif fusion_strategy == FusionStrategy.TEMPORAL:
                fused_result = self._fuse_by_temporal_alignment(task_results)
            elif fusion_strategy == FusionStrategy.SEMANTIC:
                fused_result = await self._fuse_by_semantic_similarity(
                    task_results, workflow_plan.original_query
                )
            elif fusion_strategy == FusionStrategy.HIERARCHICAL:
                fused_result = self._fuse_hierarchically(task_results, agent_modalities)
            else:
                # Simple fallback
                fused_result = self._fuse_simple(task_results)

            # Calculate cross-modal consistency
            consistency_metrics = self._check_cross_modal_consistency(task_results)

            # Calculate fusion quality metrics
            fusion_quality = self._calculate_fusion_quality(
                task_results, fused_result, consistency_metrics
            )

            final_result = {
                "aggregated_content": fused_result["content"],
                "confidence": fused_result["confidence"],
                "fusion_strategy": fusion_strategy.value,
                "fusion_quality": fusion_quality,
                "cross_modal_consistency": consistency_metrics,
                "modality_coverage": list(set(agent_modalities.values())),
                "individual_results": task_results,
                "workflow_metadata": {
                    "total_tasks": len(workflow_plan.tasks),
                    "completed_tasks": len(completed_tasks),
                    "execution_time": (
                        workflow_plan.end_time - workflow_plan.start_time
                    ).total_seconds(),
                    "agents_used": list(set(t.agent_name for t in completed_tasks)),
                },
            }

            workflow_plan.final_result = final_result
            return final_result

        except Exception as e:
            self.logger.error(f"Result aggregation failed: {e}")
            # Fallback aggregation
            return self._create_fallback_aggregation(completed_tasks, workflow_plan)

    def _detect_agent_modality(self, agent_name: str) -> str:
        """Detect modality from agent name"""
        agent_name_lower = agent_name.lower()

        if "video" in agent_name_lower:
            return "video"
        elif "image" in agent_name_lower:
            return "image"
        elif "audio" in agent_name_lower:
            return "audio"
        elif "document" in agent_name_lower:
            return "document"
        elif "text" in agent_name_lower:
            return "text"
        else:
            return "text"  # Default

    def _select_fusion_strategy(
        self, query: str, agent_modalities: Dict[str, str]
    ) -> FusionStrategy:
        """Select fusion strategy based on query and modalities"""
        modalities = set(agent_modalities.values())
        query_lower = query.lower()

        # Temporal fusion for time-related queries with multiple modalities
        temporal_keywords = [
            "timeline",
            "sequence",
            "chronological",
            "when",
            "time",
            "duration",
        ]
        if (
            any(keyword in query_lower for keyword in temporal_keywords)
            and len(modalities) > 1
        ):
            return FusionStrategy.TEMPORAL

        # Hierarchical fusion for structured comparison queries
        hierarchical_keywords = [
            "compare",
            "contrast",
            "difference",
            "similarity",
            "versus",
            "vs",
        ]
        if any(keyword in query_lower for keyword in hierarchical_keywords):
            return FusionStrategy.HIERARCHICAL

        # Semantic fusion for meaning-focused queries across modalities
        semantic_keywords = ["explain", "describe", "understand", "mean", "concept"]
        if (
            any(keyword in query_lower for keyword in semantic_keywords)
            and len(modalities) > 1
        ):
            return FusionStrategy.SEMANTIC

        # Score-based fusion for multi-modality queries
        if len(modalities) > 1:
            return FusionStrategy.SCORE_BASED

        # Simple fusion for single modality
        return FusionStrategy.SIMPLE

    def _fuse_by_score(self, task_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Score-based fusion: weight results by confidence scores"""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        total_confidence = sum(tr["confidence"] for tr in task_results.values())

        if total_confidence == 0:
            # Equal weights if no confidence scores
            weights = {task_id: 1.0 / len(task_results) for task_id in task_results}
        else:
            # Normalize confidence scores to weights
            weights = {
                task_id: tr["confidence"] / total_confidence
                for task_id, tr in task_results.items()
            }

        # Build weighted content
        content_parts = []
        for task_id, task_result in sorted(
            task_results.items(), key=lambda x: weights[x[0]], reverse=True
        ):
            weight = weights[task_id]
            modality = task_result["modality"]
            result_str = str(task_result["result"])

            content_parts.append(
                f"[{modality.upper()} - confidence: {weight:.2f}]\n{result_str}"
            )

        aggregated_confidence = sum(
            tr["confidence"] * weights[task_id] for task_id, tr in task_results.items()
        )

        return {
            "content": "\n\n".join(content_parts),
            "confidence": aggregated_confidence,
        }

    def _fuse_by_temporal_alignment(
        self, task_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Temporal fusion: time-aligned combination of results"""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        # Sort by execution time (chronological order)
        sorted_tasks = sorted(
            task_results.items(), key=lambda x: x[1]["execution_time"]
        )

        # Build chronologically ordered content
        content_parts = []
        total_confidence = 0.0

        for task_id, task_result in sorted_tasks:
            modality = task_result["modality"]
            result_str = str(task_result["result"])
            confidence = task_result["confidence"]
            total_confidence += confidence

            content_parts.append(
                f"[{modality.upper()} @ {task_result['execution_time']:.2f}s]\n{result_str}"
            )

        avg_confidence = total_confidence / len(sorted_tasks)

        return {
            "content": "\n\n".join(content_parts),
            "confidence": avg_confidence,
        }

    async def _fuse_by_semantic_similarity(
        self, task_results: Dict[str, Dict], query: str
    ) -> Dict[str, Any]:
        """Semantic fusion: combine based on semantic relevance to query"""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        # For now, use simple heuristic based on query keywords
        # In production, would use embedding similarity
        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        # Calculate relevance score for each result
        relevance_scores = {}
        for task_id, task_result in task_results.items():
            result_str = str(task_result["result"]).lower()
            result_keywords = set(result_str.split())

            # Simple keyword overlap
            overlap = len(query_keywords & result_keywords)
            relevance_scores[task_id] = overlap / max(len(query_keywords), 1)

        # Normalize relevance scores
        total_relevance = sum(relevance_scores.values())
        if total_relevance == 0:
            relevance_scores = {
                task_id: 1.0 / len(task_results) for task_id in task_results
            }
        else:
            relevance_scores = {
                task_id: score / total_relevance
                for task_id, score in relevance_scores.items()
            }

        # Build content ordered by relevance
        content_parts = []
        total_confidence = 0.0

        for task_id, task_result in sorted(
            task_results.items(), key=lambda x: relevance_scores[x[0]], reverse=True
        ):
            modality = task_result["modality"]
            result_str = str(task_result["result"])
            confidence = task_result["confidence"]
            relevance = relevance_scores[task_id]

            total_confidence += confidence * relevance

            content_parts.append(
                f"[{modality.upper()} - relevance: {relevance:.2f}]\n{result_str}"
            )

        return {
            "content": "\n\n".join(content_parts),
            "confidence": total_confidence,
        }

    def _fuse_hierarchically(
        self, task_results: Dict[str, Dict], agent_modalities: Dict[str, str]
    ) -> Dict[str, Any]:
        """Hierarchical fusion: structured combination by modality groups"""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        # Group results by modality
        modality_groups = {}
        for task_id, task_result in task_results.items():
            modality = task_result["modality"]
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append((task_id, task_result))

        # Build hierarchical structure
        content_parts = []
        total_confidence = 0.0
        modality_count = 0

        # Define modality ordering preference (can be customized)
        modality_order = ["video", "image", "audio", "document", "text"]

        for modality in modality_order:
            if modality not in modality_groups:
                continue

            modality_count += 1
            modality_tasks = modality_groups[modality]

            # Header for modality group
            content_parts.append(
                f"## {modality.upper()} RESULTS ({len(modality_tasks)} sources)"
            )

            # Add each result in the group
            modality_confidence = 0.0
            for task_id, task_result in modality_tasks:
                result_str = str(task_result["result"])
                confidence = task_result["confidence"]
                modality_confidence += confidence

                content_parts.append(f"- {result_str}")

            # Average confidence for this modality
            avg_modality_confidence = modality_confidence / len(modality_tasks)
            total_confidence += avg_modality_confidence
            content_parts.append(f"  (Confidence: {avg_modality_confidence:.2f})")
            content_parts.append("")  # Blank line

        avg_confidence = (
            total_confidence / modality_count if modality_count > 0 else 0.0
        )

        return {
            "content": "\n".join(content_parts),
            "confidence": avg_confidence,
        }

    def _fuse_simple(self, task_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Simple fusion: basic concatenation"""
        if not task_results:
            return {"content": "", "confidence": 0.0}

        content_parts = []
        total_confidence = 0.0

        for task_id, task_result in task_results.items():
            result_str = str(task_result["result"])
            confidence = task_result["confidence"]
            total_confidence += confidence

            content_parts.append(result_str)

        avg_confidence = total_confidence / len(task_results)

        return {
            "content": "\n\n".join(content_parts),
            "confidence": avg_confidence,
        }

    def _check_cross_modal_consistency(
        self, task_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Check consistency across different modalities"""
        if len(task_results) <= 1:
            return {
                "consistency_score": 1.0,
                "conflicts": [],
                "agreements": [],
                "note": "Single modality, no cross-modal comparison needed",
            }

        # Group by modality
        modality_results = {}
        for task_id, task_result in task_results.items():
            modality = task_result["modality"]
            if modality not in modality_results:
                modality_results[modality] = []
            modality_results[modality].append(task_result)

        # Simple keyword-based consistency check
        conflicts = []
        agreements = []

        # Extract all result texts
        all_texts = [str(tr["result"]).lower() for tr in task_results.values()]

        # Check for common keywords (agreement indicators)
        common_keywords = set()
        for text in all_texts:
            keywords = set(text.split())
            if not common_keywords:
                common_keywords = keywords
            else:
                common_keywords &= keywords

        # Calculate consistency score based on keyword overlap
        total_keywords = sum(len(text.split()) for text in all_texts)
        consistency_score = (len(common_keywords) * len(all_texts)) / max(
            total_keywords, 1
        )

        # Identify potential conflicts (different confidence levels for similar content)
        confidences = [tr["confidence"] for tr in task_results.values()]
        confidence_variance = sum(
            (c - sum(confidences) / len(confidences)) ** 2 for c in confidences
        ) / len(confidences)

        if confidence_variance > 0.1:
            conflicts.append(
                f"High confidence variance across modalities: {confidence_variance:.3f}"
            )

        if common_keywords:
            agreements.append(
                f"Common concepts across modalities: {', '.join(list(common_keywords)[:5])}"
            )

        return {
            "consistency_score": min(consistency_score, 1.0),
            "confidence_variance": confidence_variance,
            "conflicts": conflicts,
            "agreements": agreements,
            "modality_count": len(modality_results),
        }

    def _calculate_fusion_quality(
        self,
        task_results: Dict[str, Dict],
        fused_result: Dict[str, Any],
        consistency_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate fusion quality metrics: coverage, consistency, coherence"""
        # Coverage: how many modalities contributed
        modalities = set(tr["modality"] for tr in task_results.values())
        coverage_score = len(modalities) / 5.0  # Normalize by max modalities (5)

        # Consistency: from cross-modal consistency check
        consistency_score = consistency_metrics.get("consistency_score", 0.0)

        # Coherence: based on confidence and variance
        coherence_score = fused_result["confidence"]

        # Redundancy: measure of duplicate information
        result_texts = [str(tr["result"]) for tr in task_results.values()]
        unique_words = set()
        total_words = 0
        for text in result_texts:
            words = text.lower().split()
            total_words += len(words)
            unique_words.update(words)

        redundancy_score = 1.0 - (len(unique_words) / max(total_words, 1))

        # Complementarity: how well results complement each other
        complementarity_score = 1.0 - redundancy_score

        # Overall quality score
        overall_quality = (
            coverage_score * 0.3
            + consistency_score * 0.3
            + coherence_score * 0.2
            + complementarity_score * 0.2
        )

        return {
            "overall_quality": overall_quality,
            "coverage": coverage_score,
            "consistency": consistency_score,
            "coherence": coherence_score,
            "redundancy": redundancy_score,
            "complementarity": complementarity_score,
            "modality_count": len(modalities),
            "modalities": list(modalities),
        }

    def _create_fallback_workflow_plan(
        self,
        workflow_id: str,
        query: str,
        context: Optional[str],
        user_id: Optional[str],
    ) -> WorkflowPlan:
        """Create simple fallback workflow plan"""
        # Simple sequential workflow: search -> summarize
        tasks = [
            WorkflowTask(
                task_id="search",
                agent_name="video_search_agent",
                query=query,
                dependencies=set(),
            ),
            WorkflowTask(
                task_id="summarize",
                agent_name="summarizer_agent",
                query=f"Summarize the search results for: {query}",
                dependencies={"search"},
            ),
        ]

        return WorkflowPlan(
            workflow_id=workflow_id,
            original_query=query,
            tasks=tasks,
            execution_order=[["search"], ["summarize"]],
            metadata={"context": context, "user_id": user_id, "fallback_plan": True},
        )

    def _create_fallback_aggregation(
        self, completed_tasks: List[WorkflowTask], workflow_plan: WorkflowPlan
    ) -> Dict[str, Any]:
        """Create simple fallback result aggregation"""
        results = []
        for task in completed_tasks:
            if task.result:
                results.append(
                    f"Results from {task.agent_name}: {str(task.result)[:200]}..."
                )

        return {
            "aggregated_content": "\n\n".join(results),
            "confidence": 0.4,
            "synthesis_strategy": "simple_concatenation",
            "individual_results": {
                task.task_id: task.result for task in completed_tasks
            },
            "workflow_metadata": {
                "fallback_aggregation": True,
                "completed_tasks": len(completed_tasks),
            },
        }

    async def _generate_fallback_result(
        self, query: str, context: Optional[str]
    ) -> Dict[str, Any]:
        """Generate fallback result when orchestration fails"""
        try:
            # Try to at least get a single agent result
            routing_decision = await self.routing_agent.route_query(query, context)

            return {
                "fallback_agent": routing_decision.recommended_agent,
                "confidence": routing_decision.confidence,
                "enhanced_query": routing_decision.enhanced_query,
                "message": "Orchestration failed, providing single-agent fallback recommendation",
            }

        except Exception as e:
            return {
                "error": "Complete orchestration failure",
                "message": f"Both orchestration and fallback routing failed: {e}",
                "suggested_action": "Try simplifying the query or contacting support",
            }

    def _update_orchestration_stats(
        self, workflow_plan: WorkflowPlan, success: bool
    ) -> None:
        """Update orchestration statistics"""
        if success:
            self.orchestration_stats["completed_workflows"] += 1
        else:
            self.orchestration_stats["failed_workflows"] += 1

        if workflow_plan.start_time and workflow_plan.end_time:
            execution_time = (
                workflow_plan.end_time - workflow_plan.start_time
            ).total_seconds()

            # Update average execution time
            total_completed = self.orchestration_stats["completed_workflows"]
            current_avg = self.orchestration_stats["average_execution_time"]
            self.orchestration_stats["average_execution_time"] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
                if total_completed > 0
                else execution_time
            )

    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration performance statistics"""
        stats = self.orchestration_stats.copy()

        total_workflows = stats["total_workflows"]
        if total_workflows > 0:
            stats["completion_rate"] = stats["completed_workflows"] / total_workflows
            stats["failure_rate"] = stats["failed_workflows"] / total_workflows
        else:
            stats["completion_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Active workflows info
        stats["active_workflows"] = len(self.active_workflows)
        stats["active_workflow_ids"] = list(self.active_workflows.keys())

        # Add workflow intelligence stats if available
        if self.workflow_intelligence:
            stats["workflow_intelligence_stats"] = (
                self.workflow_intelligence.get_intelligence_statistics()
            )
            stats["agent_performance_report"] = (
                self.workflow_intelligence.get_agent_performance_report()
            )
            stats["workflow_templates"] = (
                self.workflow_intelligence.get_workflow_templates_summary()
            )

        return stats

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False

        workflow.status = WorkflowStatus.CANCELLED
        workflow.end_time = datetime.now()

        # Mark running tasks as cancelled
        for task in workflow.tasks:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.SKIPPED
                task.end_time = datetime.now()

        self.logger.info(f"Workflow {workflow_id} cancelled")
        return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "progress": {
                "total_tasks": len(workflow.tasks),
                "completed_tasks": len(
                    [t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]
                ),
                "running_tasks": len(
                    [t for t in workflow.tasks if t.status == TaskStatus.RUNNING]
                ),
                "failed_tasks": len(
                    [t for t in workflow.tasks if t.status == TaskStatus.FAILED]
                ),
            },
            "execution_time": (
                (datetime.now() - workflow.start_time).total_seconds()
                if workflow.start_time
                else 0
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "agent": task.agent_name,
                    "status": task.status.value,
                    "error": task.error,
                }
                for task in workflow.tasks
            ],
        }


def create_multi_agent_orchestrator(
    routing_agent: Optional[RoutingAgent] = None,
    available_agents: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MultiAgentOrchestrator:
    """Factory function to create Multi-Agent Orchestrator"""
    return MultiAgentOrchestrator(
        routing_agent=routing_agent, available_agents=available_agents
    )


# Example usage
if __name__ == "__main__":

    async def main():
        # Create orchestrator with enhanced routing
        orchestrator = create_multi_agent_orchestrator()

        # Test complex query that requires multiple agents
        complex_query = (
            "Find videos of robots playing soccer, summarize the key techniques used, "
            "and generate a detailed report comparing the approaches across different teams"
        )

        print(f"Processing complex query: {complex_query}")

        result = await orchestrator.process_complex_query(
            query=complex_query,
            context="Research project on robotic athletics",
            user_id="test_user",
        )

        print("\nOrchestration Result:")
        print(f"Status: {result['status']}")
        if result["status"] == "completed":
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Agents Used: {result['execution_summary']['agents_used']}")
            print(
                f"Execution Time: {result['execution_summary']['execution_time']:.2f}s"
            )
            print(f"Tasks Completed: {result['execution_summary']['completed_tasks']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Print statistics
        print("\nOrchestration Statistics:")
        stats = orchestrator.get_orchestration_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif not isinstance(value, (list, dict)):
                print(f"  {key}: {value}")

    asyncio.run(main())
