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
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

# DSPy 3.0 imports
import dspy

# A2A protocol imports
from src.app.agents.a2a_client import A2AClient

# Enhanced routing imports
from src.app.agents.enhanced_routing_agent import (
    EnhancedRoutingAgent,
    RoutingDecision,
    EnhancedRoutingConfig
)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class TaskStatus(Enum):
    """Individual task status"""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    agent_name: str
    query: str
    dependencies: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timeout_seconds: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class WorkflowPlan:
    """Complete workflow execution plan"""
    workflow_id: str
    original_query: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    execution_order: List[List[str]] = field(default_factory=list)  # Parallel execution groups
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    final_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowPlannerSignature(dspy.Signature):
    """DSPy signature for workflow planning"""
    query: str = dspy.InputField(desc="Complex user query requiring multi-agent processing")
    available_agents: str = dspy.InputField(desc="Available agents and their capabilities")
    
    workflow_tasks: List[Dict[str, str]] = dspy.OutputField(desc="List of tasks with agent assignments and dependencies")
    execution_strategy: str = dspy.OutputField(desc="Sequential, parallel, or hybrid execution strategy")
    expected_outcome: str = dspy.OutputField(desc="Expected final result structure")
    reasoning: str = dspy.OutputField(desc="Workflow planning reasoning")


class ResultAggregatorSignature(dspy.Signature):
    """DSPy signature for aggregating multi-agent results"""
    original_query: str = dspy.InputField(desc="Original user query")
    task_results: str = dspy.InputField(desc="JSON string of individual task results")
    
    aggregated_result: str = dspy.OutputField(desc="Synthesized final result")
    confidence_score: float = dspy.OutputField(desc="Confidence in aggregated result")
    synthesis_strategy: str = dspy.OutputField(desc="How results were combined")


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
        routing_agent: Optional[EnhancedRoutingAgent] = None,
        available_agents: Optional[Dict[str, Dict[str, Any]]] = None,
        max_parallel_tasks: int = 3,
        workflow_timeout_minutes: int = 15
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize routing agent
        self.routing_agent = routing_agent or EnhancedRoutingAgent()
        
        # Configure available agents and their capabilities
        self.available_agents = available_agents or self._get_default_agents()
        
        # Orchestration settings
        self.max_parallel_tasks = max_parallel_tasks
        self.workflow_timeout = timedelta(minutes=workflow_timeout_minutes)
        
        # Initialize DSPy modules
        self._initialize_dspy_modules()
        
        # A2A client for agent communication
        self.a2a_client = A2AClient()
        
        # Active workflows tracking
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        
        # Statistics
        self.orchestration_stats = {
            "total_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0,
            "total_tasks_executed": 0,
            "agent_utilization": {}
        }

    def _get_default_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get default agent configuration"""
        return {
            "video_search_agent": {
                "capabilities": [
                    "video_content_search",
                    "visual_query_analysis",
                    "multimodal_retrieval",
                    "temporal_video_analysis"
                ],
                "endpoint": "http://localhost:8002",
                "timeout_seconds": 120,
                "parallel_capacity": 2
            },
            "summarizer_agent": {
                "capabilities": [
                    "content_summarization",
                    "key_point_extraction", 
                    "document_synthesis",
                    "report_generation"
                ],
                "endpoint": "http://localhost:8003",
                "timeout_seconds": 60,
                "parallel_capacity": 3
            },
            "detailed_report_agent": {
                "capabilities": [
                    "comprehensive_analysis",
                    "detailed_reporting",
                    "data_correlation",
                    "in_depth_investigation"
                ],
                "endpoint": "http://localhost:8004",
                "timeout_seconds": 180,
                "parallel_capacity": 1
            }
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
                        {"task_id": "search", "agent": "video_search_agent", "query": query, "dependencies": []},
                        {"task_id": "summarize", "agent": "summarizer_agent", "query": f"Summarize results for: {query}", "dependencies": ["search"]}
                    ],
                    execution_strategy="sequential",
                    expected_outcome="Search results followed by summary",
                    reasoning="Fallback workflow: search then summarize"
                )
        
        class FallbackAggregatorModule(dspy.Module):
            def forward(self, original_query: str, task_results: str):
                return dspy.Prediction(
                    aggregated_result=f"Combined results for query: {original_query}",
                    confidence_score=0.6,
                    synthesis_strategy="basic_concatenation"
                )
        
        self.workflow_planner = FallbackPlannerModule()
        self.result_aggregator = FallbackAggregatorModule()
        self.logger.warning("Using fallback orchestration modules")

    async def process_complex_query(
        self,
        query: str,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
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
            execution_result = await self._execute_workflow(workflow_plan)
            
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
                    "completed_tasks": len([t for t in workflow_plan.tasks if t.status == TaskStatus.COMPLETED]),
                    "execution_time": (workflow_plan.end_time - workflow_plan.start_time).total_seconds(),
                    "agents_used": list(set(t.agent_name for t in workflow_plan.tasks))
                },
                "metadata": workflow_plan.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Orchestration failed for query '{query}': {e}")
            self.orchestration_stats["failed_workflows"] += 1
            
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "fallback_result": await self._generate_fallback_result(query, context)
            }

    async def _plan_workflow(
        self,
        workflow_id: str,
        query: str,
        context: Optional[str],
        user_id: Optional[str],
        preferences: Optional[Dict[str, Any]]
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
                query=query,
                available_agents=available_agents_str
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
                    "execution_strategy": getattr(planning_result, "execution_strategy", "sequential")
                }
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
                    timeout_seconds=self.available_agents.get(agent_name, {}).get("timeout_seconds", 120)
                )
                tasks.append(task)
            
            workflow_plan.tasks = tasks
            workflow_plan.execution_order = self._calculate_execution_order(tasks)
            
            self.logger.info(
                f"Workflow planned: {len(tasks)} tasks, "
                f"{len(workflow_plan.execution_order)} execution phases"
            )
            
            return workflow_plan
            
        except Exception as e:
            self.logger.error(f"Workflow planning failed: {e}")
            # Return simple fallback plan
            return self._create_fallback_workflow_plan(workflow_id, query, context, user_id)

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
                if task.dependencies.issubset(set(task.task_id for task in tasks if task.status == TaskStatus.COMPLETED) | 
                                           set(task_id for phase in execution_order for task_id in phase)):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Handle circular dependencies by taking any remaining task
                ready_tasks = [next(iter(remaining_tasks))]
                self.logger.warning(f"Potential circular dependency detected, forcing execution of {ready_tasks[0]}")
            
            # Limit parallel execution
            parallel_batch = ready_tasks[:self.max_parallel_tasks]
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
                phase_tasks = [task for task in workflow_plan.tasks if task.task_id in task_ids]
                phase_results = await asyncio.gather(
                    *[self._execute_task(task, workflow_plan) for task in phase_tasks],
                    return_exceptions=True
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
                if failed_tasks and not self._can_continue_with_failures(workflow_plan, failed_tasks):
                    workflow_plan.status = WorkflowStatus.FAILED
                    self.logger.error(f"Workflow aborted due to critical task failures: {failed_tasks}")
                    return False
            
            # Mark as completed if we got here
            completed_tasks = [t for t in workflow_plan.tasks if t.status == TaskStatus.COMPLETED]
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
            # Clean up active workflow
            self.active_workflows.pop(workflow_plan.workflow_id, None)

    async def _execute_task(self, task: WorkflowTask, workflow_plan: WorkflowPlan) -> Dict[str, Any]:
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
            agent_endpoint = agent_config.get("endpoint", f"http://localhost:8000")
            
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
                    "parameters": task.parameters
                },
                timeout=task.timeout_seconds
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
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self._execute_task(task, workflow_plan)
            
            raise e

    async def _prepare_task_context(self, task: WorkflowTask, workflow_plan: WorkflowPlan) -> str:
        """Prepare context for task execution including dependency results"""
        context_parts = []
        
        # Add original context
        if workflow_plan.metadata.get("context"):
            context_parts.append(f"Original context: {workflow_plan.metadata['context']}")
        
        # Add results from dependency tasks
        for dep_task_id in task.dependencies:
            dep_task = next((t for t in workflow_plan.tasks if t.task_id == dep_task_id), None)
            if dep_task and dep_task.result:
                # Summarize dependency result
                dep_summary = str(dep_task.result).replace('\n', ' ')[:200]
                context_parts.append(f"Result from {dep_task_id}: {dep_summary}...")
        
        return " | ".join(context_parts)

    def _can_continue_with_failures(
        self, 
        workflow_plan: WorkflowPlan, 
        failed_task_ids: List[str]
    ) -> bool:
        """Determine if workflow can continue despite task failures"""
        # Simple heuristic: continue if less than 50% of tasks failed
        total_tasks = len(workflow_plan.tasks)
        failed_count = len([t for t in workflow_plan.tasks if t.status == TaskStatus.FAILED])
        
        return failed_count / total_tasks < 0.5

    async def _aggregate_results(self, workflow_plan: WorkflowPlan) -> Dict[str, Any]:
        """Aggregate results from completed tasks"""
        completed_tasks = [t for t in workflow_plan.tasks if t.status == TaskStatus.COMPLETED and t.result]
        
        if not completed_tasks:
            return {"error": "No completed tasks to aggregate"}
        
        try:
            # Prepare task results for DSPy aggregation
            task_results = {}
            for task in completed_tasks:
                task_results[task.task_id] = {
                    "agent": task.agent_name,
                    "query": task.query,
                    "result": task.result,
                    "execution_time": (task.end_time - task.start_time).total_seconds()
                }
            
            # Use DSPy to intelligently aggregate results
            aggregation_result = self.result_aggregator.forward(
                original_query=workflow_plan.original_query,
                task_results=str(task_results)
            )
            
            final_result = {
                "aggregated_content": getattr(aggregation_result, "aggregated_result", ""),
                "confidence": getattr(aggregation_result, "confidence_score", 0.5),
                "synthesis_strategy": getattr(aggregation_result, "synthesis_strategy", "concatenation"),
                "individual_results": task_results,
                "workflow_metadata": {
                    "total_tasks": len(workflow_plan.tasks),
                    "completed_tasks": len(completed_tasks),
                    "execution_time": (workflow_plan.end_time - workflow_plan.start_time).total_seconds(),
                    "agents_used": list(set(t.agent_name for t in completed_tasks))
                }
            }
            
            workflow_plan.final_result = final_result
            return final_result
            
        except Exception as e:
            self.logger.error(f"Result aggregation failed: {e}")
            # Fallback aggregation
            return self._create_fallback_aggregation(completed_tasks, workflow_plan)

    def _create_fallback_workflow_plan(
        self,
        workflow_id: str,
        query: str,
        context: Optional[str],
        user_id: Optional[str]
    ) -> WorkflowPlan:
        """Create simple fallback workflow plan"""
        # Simple sequential workflow: search -> summarize
        tasks = [
            WorkflowTask(
                task_id="search",
                agent_name="video_search_agent",
                query=query,
                dependencies=set()
            ),
            WorkflowTask(
                task_id="summarize",
                agent_name="summarizer_agent", 
                query=f"Summarize the search results for: {query}",
                dependencies={"search"}
            )
        ]
        
        return WorkflowPlan(
            workflow_id=workflow_id,
            original_query=query,
            tasks=tasks,
            execution_order=[["search"], ["summarize"]],
            metadata={
                "context": context,
                "user_id": user_id,
                "fallback_plan": True
            }
        )

    def _create_fallback_aggregation(
        self,
        completed_tasks: List[WorkflowTask],
        workflow_plan: WorkflowPlan
    ) -> Dict[str, Any]:
        """Create simple fallback result aggregation"""
        results = []
        for task in completed_tasks:
            if task.result:
                results.append(f"Results from {task.agent_name}: {str(task.result)[:200]}...")
        
        return {
            "aggregated_content": "\n\n".join(results),
            "confidence": 0.4,
            "synthesis_strategy": "simple_concatenation",
            "individual_results": {task.task_id: task.result for task in completed_tasks},
            "workflow_metadata": {
                "fallback_aggregation": True,
                "completed_tasks": len(completed_tasks)
            }
        }

    async def _generate_fallback_result(
        self,
        query: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Generate fallback result when orchestration fails"""
        try:
            # Try to at least get a single agent result
            routing_decision = await self.routing_agent.route_query(query, context)
            
            return {
                "fallback_agent": routing_decision.recommended_agent,
                "confidence": routing_decision.confidence,
                "enhanced_query": routing_decision.enhanced_query,
                "message": "Orchestration failed, providing single-agent fallback recommendation"
            }
            
        except Exception as e:
            return {
                "error": "Complete orchestration failure",
                "message": f"Both orchestration and fallback routing failed: {e}",
                "suggested_action": "Try simplifying the query or contacting support"
            }

    def _update_orchestration_stats(self, workflow_plan: WorkflowPlan, success: bool) -> None:
        """Update orchestration statistics"""
        if success:
            self.orchestration_stats["completed_workflows"] += 1
        else:
            self.orchestration_stats["failed_workflows"] += 1
        
        if workflow_plan.start_time and workflow_plan.end_time:
            execution_time = (workflow_plan.end_time - workflow_plan.start_time).total_seconds()
            
            # Update average execution time
            total_completed = self.orchestration_stats["completed_workflows"]
            current_avg = self.orchestration_stats["average_execution_time"]
            self.orchestration_stats["average_execution_time"] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
                if total_completed > 0 else execution_time
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
                "completed_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]),
                "running_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.RUNNING]),
                "failed_tasks": len([t for t in workflow.tasks if t.status == TaskStatus.FAILED])
            },
            "execution_time": (
                (datetime.now() - workflow.start_time).total_seconds()
                if workflow.start_time else 0
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "agent": task.agent_name,
                    "status": task.status.value,
                    "error": task.error
                }
                for task in workflow.tasks
            ]
        }


def create_multi_agent_orchestrator(
    routing_agent: Optional[EnhancedRoutingAgent] = None,
    available_agents: Optional[Dict[str, Dict[str, Any]]] = None
) -> MultiAgentOrchestrator:
    """Factory function to create Multi-Agent Orchestrator"""
    return MultiAgentOrchestrator(
        routing_agent=routing_agent,
        available_agents=available_agents
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
            user_id="test_user"
        )
        
        print("\nOrchestration Result:")
        print(f"Status: {result['status']}")
        if result["status"] == "completed":
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Agents Used: {result['execution_summary']['agents_used']}")
            print(f"Execution Time: {result['execution_summary']['execution_time']:.2f}s")
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