"""
OrchestratorAgent - Autonomous A2A agent for coordinating multi-agent query processing.

Implements two-phase orchestration:
1. Planning Phase: Analyze query and create execution plan
2. Action Phase: Execute plan by coordinating specialized agents via A2A HTTP
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import dspy
from pydantic import BaseModel, Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

if TYPE_CHECKING:
    from cogniverse_agents.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class OrchestratorInput(AgentInput):
    """Type-safe input for orchestration"""

    query: str = Field(..., description="Query to orchestrate")
    tenant_id: str = Field(
        default="default", description="Tenant identifier (per-request)"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session identifier (per-request)"
    )


class OrchestratorOutput(AgentOutput):
    """Type-safe output from orchestration"""

    query: str = Field(..., description="Original query")
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Orchestration plan steps"
    )
    parallel_groups: List[List[int]] = Field(
        default_factory=list, description="Parallel execution groups"
    )
    plan_reasoning: str = Field("", description="Plan reasoning")
    agent_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results from each agent"
    )
    final_output: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated final output"
    )
    execution_summary: str = Field("", description="Summary of execution")


class OrchestratorDeps(AgentDeps):
    """Dependencies for orchestrator agent (tenant-agnostic at startup)."""

    pass


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


class OrchestratorAgent(
    MemoryAwareMixin, A2AAgent[OrchestratorInput, OrchestratorOutput, OrchestratorDeps]
):
    """
    Type-safe autonomous A2A agent for multi-agent orchestration.

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
        deps: OrchestratorDeps,
        registry: "AgentRegistry",
        port: int = 8013,
    ):
        """
        Initialize OrchestratorAgent with real AgentRegistry.

        Args:
            deps: Typed dependencies (tenant-agnostic)
            registry: AgentRegistry for dynamic agent discovery
            port: Port for A2A server

        Raises:
            TypeError: If deps is not OrchestratorDeps
            ValueError: If registry is not provided
        """
        self.registry = registry

        # Initialize DSPy module
        orchestration_module = OrchestrationModule()

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="orchestrator_agent",
            agent_description="Type-safe orchestration with planning and action phases",
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

        # Initialize base class
        super().__init__(deps=deps, config=config, dspy_module=orchestration_module)

        # Track which tenants have memory initialized
        self._memory_initialized_tenants: set = set()

        logger.info(
            f"OrchestratorAgent initialized with {len(self.registry.agents)} registered agents"
        )

    def _ensure_memory_for_tenant(self, tenant_id: str) -> None:
        """Lazily initialize memory for a tenant (first request only)."""
        if tenant_id in self._memory_initialized_tenants:
            return

        try:
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )

            config_manager = create_default_config_manager()
            config = get_config(tenant_id="default", config_manager=config_manager)

            inference = config.get("inference", {})
            backend_section = config.get("backend", {})
            backend_url = backend_section.get("url", "http://localhost")
            backend_port = backend_section.get("port", 8080)
            provider = inference.get("provider", "ollama")
            llm_model = inference.get("model", "gemma3:4b")
            embedding_model = inference.get("embedding_model", "nomic-embed-text")
            llm_base_url = inference.get("local_endpoint", "http://localhost:11434")
            if provider == "modal":
                llm_base_url = inference.get("modal_endpoint", llm_base_url)

            from pathlib import Path

            from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

            schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

            self.initialize_memory(
                agent_name="orchestrator_agent",
                tenant_id=tenant_id,
                backend_host=backend_url,
                backend_port=backend_port,
                llm_model=llm_model,
                embedding_model=embedding_model,
                llm_base_url=llm_base_url,
                config_manager=config_manager,
                schema_loader=schema_loader,
                provider=provider,
            )
            self._memory_initialized_tenants.add(tenant_id)
        except Exception as e:
            logger.warning(f"Memory initialization failed for tenant {tenant_id}: {e}")

    # ==========================================================================
    # Type-safe process method (required by AgentBase)
    # ==========================================================================

    async def _process_impl(
        self, input: Union[OrchestratorInput, Dict[str, Any]]
    ) -> OrchestratorOutput:
        """
        Process orchestration request with typed input/output.

        Args:
            input: Typed input with query, tenant_id, session_id (or dict)

        Returns:
            OrchestratorOutput with plan, agent results, and final output
        """
        # Coerce dict to typed input (A2A sends JSON dicts)
        if isinstance(input, dict):
            input = OrchestratorInput(**input)

        query = input.query
        tenant_id = input.tenant_id
        session_id = input.session_id

        if not query:
            return OrchestratorOutput(
                query="",
                plan_steps=[],
                parallel_groups=[],
                plan_reasoning="Empty query, no orchestration needed",
                agent_results={},
                final_output={"status": "error", "message": "Empty query"},
                execution_summary="No execution performed",
            )

        # Lazy memory initialization for this tenant
        self._ensure_memory_for_tenant(tenant_id)

        # Get relevant context from memory (cross-session)
        memory_context = self.get_relevant_context(query)
        if memory_context:
            logger.info(f"Retrieved memory context for query: {query[:50]}...")

        # Phase 1: Planning
        plan = await self._create_plan(query)

        # Phase 2: Action — execute via A2A HTTP, passing tenant_id/session_id
        agent_results = await self._execute_plan(
            plan, tenant_id=tenant_id, session_id=session_id
        )

        # Aggregate results
        final_output = self._aggregate_results(query, agent_results)

        # Generate summary
        execution_summary = self._generate_summary(plan, agent_results)

        # Remember this interaction for future context
        self.remember_success(query, execution_summary)

        return OrchestratorOutput(
            query=query,
            plan_steps=[
                {
                    "agent_type": step.agent_type.value,
                    "reasoning": step.reasoning,
                    "depends_on": step.depends_on,
                }
                for step in plan.steps
            ],
            parallel_groups=plan.parallel_groups,
            plan_reasoning=plan.reasoning,
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
                    reasoning=f"Step {i + 1}: {agent_type.value} processing",
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
            # Parallel step - check what comes before this group
            min_index_in_group = min(current_group)
            if min_index_in_group > 0:
                prev_step = min_index_in_group - 1

                # Check if previous step is in a parallel group
                prev_in_group = None
                for group in parallel_groups:
                    if prev_step in group:
                        prev_in_group = group
                        break

                if prev_in_group:
                    # Previous parallel group - depend on entire group
                    depends_on.extend(prev_in_group)
                else:
                    # Sequential step before this group - depend on it
                    depends_on.append(prev_step)
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

    async def _execute_plan(
        self,
        plan: OrchestrationPlan,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Action Phase: Execute orchestration plan via A2A HTTP calls.

        Discovers agents from the real AgentRegistry, calls them via HTTP,
        and passes tenant_id/session_id through to each agent.

        Args:
            plan: OrchestrationPlan to execute
            tenant_id: Tenant identifier (per-request)
            session_id: Session identifier (per-request)

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
                deps_met = all(executed[dep_idx] for dep_idx in step.depends_on)
                if deps_met:
                    ready_steps.append((i, step))

            if not ready_steps:
                logger.error("No steps ready to execute but execution incomplete")
                break

            logger.info(
                f"Executing {len(ready_steps)} steps in parallel: "
                f"{[s[1].agent_type.value for s in ready_steps]}"
            )

            async def execute_step(step_index: int, step: AgentStep):
                """Execute a single step via A2A HTTP."""
                agent_name = step.agent_type.value

                # Discover agent from registry by name
                agent_endpoint = self.registry.get_agent(agent_name)
                if not agent_endpoint:
                    # Try finding by capability (agent_type.value matches capability name)
                    candidates = self.registry.find_agents_by_capability(agent_name)
                    if candidates:
                        agent_endpoint = candidates[0]

                if not agent_endpoint:
                    logger.warning(f"Agent '{agent_name}' not found in registry")
                    return agent_name, {
                        "status": "error",
                        "message": f"Agent '{agent_name}' not available in registry",
                    }

                # Prepare input (merge query with previous results if needed)
                agent_input = step.input_data.copy()
                agent_input["tenant_id"] = tenant_id
                if session_id:
                    agent_input["session_id"] = session_id

                for dep_idx in step.depends_on:
                    if dep_idx < len(plan.steps):
                        dep_agent = plan.steps[dep_idx].agent_type.value
                        if dep_agent in agent_results:
                            agent_input[f"{dep_agent}_result"] = agent_results[
                                dep_agent
                            ]

                # Call agent via A2A HTTP
                try:
                    query = agent_input.pop("query", "")
                    result = await self.a2a_client.send_task(
                        agent_endpoint.url,
                        query=query,
                        **agent_input,
                    )
                    return agent_name, result
                except Exception as e:
                    logger.error(
                        f"Agent {agent_name} at {agent_endpoint.url} failed: {e}"
                    )
                    return agent_name, {
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

    def _dspy_to_a2a_output(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Convert OrchestrationResult to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "query": result.query,
            "plan": {
                "steps": [
                    {
                        "agent_type": step.agent_type.value,
                        "reasoning": step.reasoning,
                        "depends_on": step.depends_on,
                    }
                    for step in result.plan.steps
                ],
                "parallel_groups": result.plan.parallel_groups,
                "reasoning": result.plan.reasoning,
            },
            "agent_results": result.agent_results,
            "final_output": result.final_output,
            "execution_summary": result.execution_summary,
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "orchestrate",
                "description": "Orchestrate multi-agent query processing with planning and execution",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "plan": "object",
                    "agent_results": "object",
                    "final_output": "object",
                    "execution_summary": "string",
                },
                "examples": [
                    {
                        "input": {"query": "Show me machine learning videos"},
                        "output": {
                            "plan": {
                                "steps": [
                                    {"agent_type": "query_enhancement"},
                                    {"agent_type": "search"},
                                ]
                            },
                            "execution_summary": "Executed 2/2 steps (2 successful)",
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
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global orchestrator_agent

    from cogniverse_agents.agent_registry import AgentRegistry
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()

    # AgentRegistry reads agents from config.json > agents section
    registry = AgentRegistry(config_manager=config_manager)

    deps = OrchestratorDeps()
    orchestrator_agent = OrchestratorAgent(deps=deps, registry=registry)
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
    from cogniverse_agents.agent_registry import AgentRegistry
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    registry = AgentRegistry(config_manager=config_manager)
    deps = OrchestratorDeps()
    agent = OrchestratorAgent(deps=deps, registry=registry, port=8013)
    logger.info("Starting OrchestratorAgent on port 8013...")
    agent.start()
