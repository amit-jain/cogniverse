"""
OrchestratorAgent - Autonomous A2A agent for coordinating multi-agent query processing.

Implements two-phase orchestration:
1. Planning Phase: Analyze query and create execution plan
2. Action Phase: Execute plan by coordinating specialized agents
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import dspy
from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Available agent types for orchestration"""

    ENTITY_EXTRACTION = "entity_extraction"
    PROFILE_SELECTION = "profile_selection"
    QUERY_ENHANCEMENT = "query_enhancement"
    SEARCH = "search"
    SUMMARIZER = "summarizer"
    DETAILED_REPORT = "detailed_report"


class AgentStep(BaseModel):
    """Single step in orchestration plan"""

    agent_type: AgentType
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input for this agent"
    )
    depends_on: List[int] = Field(
        default_factory=list, description="Indices of steps this depends on"
    )
    reasoning: str = Field(description="Why this step is needed")


class OrchestrationPlan(BaseModel):
    """Plan for query processing workflow"""

    query: str
    steps: List[AgentStep] = Field(description="Ordered sequence of agent invocations")
    parallel_groups: List[List[int]] = Field(
        default_factory=list,
        description="Groups of step indices that can run in parallel",
    )
    reasoning: str = Field(description="Overall plan reasoning")


class OrchestrationResult(BaseModel):
    """Result of orchestrated query processing"""

    query: str
    plan: OrchestrationPlan
    agent_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results from each agent"
    )
    final_output: Dict[str, Any] = Field(description="Aggregated final output")
    execution_summary: str = Field(description="Summary of execution")


class OrchestrationSignature(dspy.Signature):
    """Create execution plan for query processing"""

    query: str = dspy.InputField(desc="User query to process")
    available_agents: str = dspy.InputField(
        desc="Comma-separated list of available agents"
    )

    agent_sequence: str = dspy.OutputField(
        desc="Comma-separated sequence of agents to invoke (e.g., 'entity_extraction,profile_selection,search')"
    )
    parallel_steps: str = dspy.OutputField(
        desc="Indices of steps that can run in parallel (e.g., '0,1|2,3' means 0&1 parallel, then 2&3 parallel)"
    )
    reasoning: str = dspy.OutputField(desc="Explanation of orchestration plan")


class OrchestrationModule(dspy.Module):
    """DSPy module for orchestration planning"""

    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(OrchestrationSignature)

    def forward(self, query: str, available_agents: str) -> dspy.Prediction:
        """Create orchestration plan using LLM reasoning"""
        try:
            return self.planner(query=query, available_agents=available_agents)
        except Exception as e:
            logger.warning(f"Orchestration planning failed: {e}, using fallback")
            return self._fallback_plan(query, available_agents)

    def _fallback_plan(self, query: str, available_agents: str) -> dspy.Prediction:
        """Fallback orchestration using simple heuristics"""
        # Default pipeline: enhancement -> entity extraction -> profile selection -> search
        agent_sequence = "query_enhancement,entity_extraction,profile_selection,search"

        # Enhancement and entity extraction can run in parallel (step 0,1)
        # Then profile selection and search run sequentially
        parallel_steps = "0,1"

        reasoning = "Fallback plan: enhance query and extract entities in parallel, then select profile and search"

        return dspy.Prediction(
            agent_sequence=agent_sequence,
            parallel_steps=parallel_steps,
            reasoning=reasoning,
        )


class OrchestratorAgent(DSPyA2AAgentBase):
    """
    Autonomous A2A agent for multi-agent orchestration.

    Implements two-phase orchestration:
    1. Planning Phase: Analyze query and create execution plan
    2. Action Phase: Execute plan by coordinating agents

    Capabilities:
    - LLM-based query analysis for optimal agent selection
    - Parallel agent execution where possible
    - Result aggregation and synthesis
    - Adaptive workflow based on query characteristics
    """

    def __init__(
        self,
        tenant_id: str = "default",
        agent_registry: Optional[Dict[AgentType, Any]] = None,
        port: int = 8013,
    ):
        """
        Initialize OrchestratorAgent

        Args:
            tenant_id: Tenant identifier
            agent_registry: Registry of available agents
            port: Port for A2A server
        """
        self.tenant_id = tenant_id
        self.agent_registry = agent_registry or {}

        # Initialize DSPy module
        orchestration_module = OrchestrationModule()

        # Initialize base class
        super().__init__(
            agent_name="orchestrator_agent",
            agent_description="Autonomous orchestration with planning and action phases",
            dspy_module=orchestration_module,
            capabilities=[
                "orchestration",
                "planning",
                "multi_agent_coordination",
                "parallel_execution",
                "result_aggregation",
            ],
            port=port,
            version="1.0.0",
        )

        logger.info(
            f"OrchestratorAgent initialized for tenant: {tenant_id}, "
            f"agents: {len(self.agent_registry)}"
        )

    async def _process(self, dspy_input: Dict[str, Any]) -> OrchestrationResult:
        """
        Process orchestration request with planning and action phases

        Args:
            dspy_input: Input with 'query' field

        Returns:
            OrchestrationResult with plan, agent results, and final output
        """
        query = dspy_input.get("query", "")

        if not query:
            return OrchestrationResult(
                query="",
                plan=OrchestrationPlan(
                    query="", steps=[], reasoning="Empty query, no orchestration needed"
                ),
                agent_results={},
                final_output={"status": "error", "message": "Empty query"},
                execution_summary="No execution performed",
            )

        # Phase 1: Planning
        plan = await self._create_plan(query)

        # Phase 2: Action
        agent_results = await self._execute_plan(plan)

        # Aggregate results
        final_output = self._aggregate_results(query, agent_results)

        # Generate summary
        execution_summary = self._generate_summary(plan, agent_results)

        return OrchestrationResult(
            query=query,
            plan=plan,
            agent_results=agent_results,
            final_output=final_output,
            execution_summary=execution_summary,
        )

    async def _create_plan(self, query: str) -> OrchestrationPlan:
        """
        Planning Phase: Create execution plan using LLM reasoning

        Args:
            query: User query to analyze

        Returns:
            OrchestrationPlan with agent sequence and parallelization
        """
        # Get available agents
        available_agents = ", ".join([a.value for a in AgentType])

        # Use DSPy to create plan
        result = self.dspy_module.forward(
            query=query, available_agents=available_agents
        )

        # Parse agent sequence
        agent_sequence = [
            a.strip() for a in result.agent_sequence.split(",") if a.strip()
        ]

        # Parse parallel groups
        parallel_groups = []
        if result.parallel_steps:
            for group in result.parallel_steps.split("|"):
                indices = [int(i.strip()) for i in group.split(",") if i.strip()]
                if indices:
                    parallel_groups.append(indices)

        # Create agent steps
        steps = []
        for i, agent_name in enumerate(agent_sequence):
            try:
                agent_type = AgentType(agent_name)
                step = AgentStep(
                    agent_type=agent_type,
                    input_data={"query": query},
                    depends_on=self._calculate_dependencies(i, parallel_groups),
                    reasoning=f"Step {i+1}: {agent_type.value} processing",
                )
                steps.append(step)
            except ValueError:
                logger.warning(f"Unknown agent type: {agent_name}, skipping")

        return OrchestrationPlan(
            query=query,
            steps=steps,
            parallel_groups=parallel_groups,
            reasoning=result.reasoning,
        )

    def _calculate_dependencies(
        self, step_index: int, parallel_groups: List[List[int]]
    ) -> List[int]:
        """Calculate which steps this step depends on"""
        depends_on = []

        # Find which parallel group this step belongs to
        current_group = None
        for group in parallel_groups:
            if step_index in group:
                current_group = group
                break

        if current_group:
            # Depends on all steps before the current group
            for group in parallel_groups:
                if group == current_group:
                    break
                depends_on.extend(group)
        else:
            # Sequential step - check if previous step is in a parallel group
            if step_index > 0:
                prev_step = step_index - 1

                # Check if previous step is in a parallel group
                prev_in_group = None
                for group in parallel_groups:
                    if prev_step in group:
                        prev_in_group = group
                        break

                if prev_in_group:
                    # Previous step is in a parallel group, depend on entire group
                    depends_on.extend(prev_in_group)
                else:
                    # Previous step is sequential, just depend on it
                    depends_on.append(prev_step)

        return depends_on

    async def _execute_plan(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """
        Action Phase: Execute orchestration plan with parallel execution support

        Args:
            plan: OrchestrationPlan to execute

        Returns:
            Dictionary of agent results
        """
        import asyncio

        agent_results = {}
        executed = [False] * len(plan.steps)

        # Execute steps respecting dependencies and parallelism
        while not all(executed):
            # Find steps ready to execute (all dependencies met)
            ready_steps = []
            for i, step in enumerate(plan.steps):
                if executed[i]:
                    continue
                # Check if all dependencies are satisfied
                deps_met = all(executed[dep_idx] for dep_idx in step.depends_on)
                if deps_met:
                    ready_steps.append((i, step))

            if not ready_steps:
                # No steps ready but not all executed - circular dependency or error
                logger.error("No steps ready to execute but execution incomplete")
                break

            # Execute all ready steps in parallel using asyncio.gather
            logger.info(
                f"Executing {len(ready_steps)} steps in parallel: {[s[1].agent_type.value for s in ready_steps]}"
            )

            async def execute_step(step_index: int, step: AgentStep):
                """Execute a single step"""
                agent = self.agent_registry.get(step.agent_type)
                if not agent:
                    logger.warning(
                        f"Agent {step.agent_type.value} not found in registry"
                    )
                    return step.agent_type.value, {
                        "status": "error",
                        "message": f"Agent {step.agent_type.value} not available",
                    }

                # Prepare input (merge query with previous results if needed)
                agent_input = step.input_data.copy()
                for dep_idx in step.depends_on:
                    if dep_idx < len(plan.steps):
                        dep_agent = plan.steps[dep_idx].agent_type.value
                        if dep_agent in agent_results:
                            agent_input[f"{dep_agent}_result"] = agent_results[
                                dep_agent
                            ]

                # Execute agent
                try:
                    result = await agent._process(agent_input)
                    return step.agent_type.value, result
                except Exception as e:
                    logger.error(f"Agent {step.agent_type.value} execution failed: {e}")
                    return step.agent_type.value, {
                        "status": "error",
                        "message": str(e),
                    }

            # Execute all ready steps concurrently
            results = await asyncio.gather(
                *[execute_step(idx, step) for idx, step in ready_steps]
            )

            # Store results and mark as executed
            for (step_idx, _), (agent_name, result) in zip(ready_steps, results):
                agent_results[agent_name] = result
                executed[step_idx] = True

        return agent_results

    def _aggregate_results(
        self, query: str, agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from all agents into final output"""
        final_output = {"query": query, "status": "success", "results": {}}

        # Collect results from each agent
        for agent_type, result in agent_results.items():
            if isinstance(result, BaseModel):
                final_output["results"][agent_type] = result.model_dump()
            else:
                final_output["results"][agent_type] = result

        return final_output

    def _generate_summary(
        self, plan: OrchestrationPlan, agent_results: Dict[str, Any]
    ) -> str:
        """Generate execution summary"""
        executed_steps = len(agent_results)  # All executed, including errors
        successful_steps = sum(
            1
            for result in agent_results.values()
            if not (isinstance(result, dict) and result.get("status") == "error")
        )
        total_steps = len(plan.steps)

        return (
            f"Executed {executed_steps}/{total_steps} steps "
            f"({successful_steps} successful). "
            f"Plan: {plan.reasoning}"
        )

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert OrchestrationResult to A2A format"""
        if isinstance(dspy_output, OrchestrationResult):
            return {
                "status": "success",
                "agent": self.agent_name,
                "query": dspy_output.query,
                "plan": {
                    "steps": [
                        {
                            "agent_type": step.agent_type.value,
                            "reasoning": step.reasoning,
                            "depends_on": step.depends_on,
                        }
                        for step in dspy_output.plan.steps
                    ],
                    "parallel_groups": dspy_output.plan.parallel_groups,
                    "reasoning": dspy_output.plan.reasoning,
                },
                "agent_results": dspy_output.agent_results,
                "final_output": dspy_output.final_output,
                "execution_summary": dspy_output.execution_summary,
            }
        else:
            return {
                "status": "success",
                "agent": self.agent_name,
                "output": str(dspy_output),
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define agent skills for A2A protocol"""
        return [
            {
                "name": "orchestrate",
                "description": "Orchestrate multi-agent query processing with planning and execution",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "query": "string",
                    "plan": {
                        "steps": "array of agent steps",
                        "parallel_groups": "array of parallel step groups",
                        "reasoning": "string",
                    },
                    "agent_results": "dictionary of agent results",
                    "final_output": "aggregated final output",
                    "execution_summary": "string",
                },
                "examples": [
                    {
                        "input": {"query": "Show me videos about machine learning"},
                        "output": {
                            "query": "Show me videos about machine learning",
                            "plan": {
                                "steps": [
                                    {
                                        "agent_type": "query_enhancement",
                                        "reasoning": "Enhance query with ML synonyms and context",
                                    },
                                    {
                                        "agent_type": "entity_extraction",
                                        "reasoning": "Extract ML-related entities",
                                    },
                                    {
                                        "agent_type": "profile_selection",
                                        "reasoning": "Select video-based profile",
                                    },
                                    {
                                        "agent_type": "search",
                                        "reasoning": "Execute search with selected profile",
                                    },
                                ],
                                "parallel_groups": [[0, 1]],
                                "reasoning": "Enhance and extract entities in parallel, then select profile and search",
                            },
                            "execution_summary": "Executed 4/4 steps successfully",
                        },
                    }
                ],
            }
        ]


# FastAPI app for standalone deployment
from fastapi import FastAPI

app = FastAPI(
    title="OrchestratorAgent",
    description="Autonomous orchestration agent with planning and action phases",
    version="1.0.0",
)

# Global agent instance
orchestrator_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global orchestrator_agent

    import os

    tenant_id = os.getenv("TENANT_ID", "default")
    orchestrator_agent = OrchestratorAgent(tenant_id=tenant_id)
    logger.info("OrchestratorAgent started")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not orchestrator_agent:
        return {"status": "initializing"}
    return orchestrator_agent.app.routes[2].endpoint()


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not orchestrator_agent:
        return {"error": "Agent not initialized"}
    return orchestrator_agent.app.routes[0].endpoint()


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not orchestrator_agent:
        return {"error": "Agent not initialized"}
    return await orchestrator_agent.app.routes[1].endpoint(task)


if __name__ == "__main__":

    agent = OrchestratorAgent(tenant_id="default", port=8013)
    logger.info("Starting OrchestratorAgent on port 8013...")
    agent.run()
