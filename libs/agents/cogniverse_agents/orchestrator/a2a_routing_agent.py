"""
A2A wrapper for routing agent that provides standardized A2A communication.
Handles message formatting, routing coordination, and response aggregation.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.config.manager import ConfigManager

import httpx
from cogniverse_core.common.a2a_utils import (
    DataPart,
    Task,
    TextPart,
    create_data_message,
    create_task,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.config.utils import get_config

from cogniverse_agents.routing_agent import RoutingAgent

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result from A2A routing execution"""

    task_id: str
    routing_decision: Dict[str, Any]
    agent_responses: Dict[str, Any]
    final_result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


class A2ARoutingAgent:
    """
    A2A wrapper for the routing agent that handles multi-agent coordination.
    Provides standardized A2A communication and response aggregation.
    """

    def __init__(self, tenant_id: str = "default", routing_agent: Optional[RoutingAgent] = None, config_manager: "ConfigManager" = None):
        """
        Initialize A2A routing agent.

        Args:
            tenant_id: Tenant identifier for config scoping
            routing_agent: Optional routing agent instance (will create if not provided)
            config_manager: ConfigManager instance (required for dependency injection)

        Raises:
            ValueError: If config_manager is not provided
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for A2ARoutingAgent. "
                "Pass ConfigManager() explicitly."
            )


        self.tenant_id = tenant_id
        self.config_manager = config_manager
        self.config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        self.routing_agent = routing_agent or RoutingAgent(tenant_id=tenant_id, config_manager=config_manager)
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize agent registry
        from cogniverse_agents.agent_registry import AgentRegistry

        self.agent_registry = AgentRegistry(tenant_id=tenant_id, config_manager=config_manager)

        logger.info("A2ARoutingAgent initialized successfully")

    async def process_task(self, task: Task) -> RoutingResult:
        """
        Process A2A task through complete routing workflow.

        Args:
            task: A2A task to process

        Returns:
            RoutingResult with complete execution information
        """
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(f"Processing A2A task: {task.id}")

            # Step 1: Extract query from task
            query, context = self._extract_query_and_context(task)

            # Step 2: Get routing decision
            routing_analysis = await self.routing_agent.analyze_and_route(
                query, context
            )

            # Step 3: Execute agent workflow
            agent_responses = await self._execute_agent_workflow(
                routing_analysis, task.id
            )

            # Step 4: Aggregate final result
            final_result = self._aggregate_results(routing_analysis, agent_responses)

            execution_time = asyncio.get_event_loop().time() - start_time

            return RoutingResult(
                task_id=task.id,
                routing_decision=routing_analysis,
                agent_responses=agent_responses,
                final_result=final_result,
                execution_time=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Task processing failed: {e}")

            return RoutingResult(
                task_id=task.id,
                routing_decision={},
                agent_responses={},
                final_result=None,
                execution_time=execution_time,
                success=False,
                error=str(e),
            )

    def _extract_query_and_context(self, task: Task) -> tuple[str, Dict[str, Any]]:
        """
        Extract query and context from A2A task.

        Args:
            task: A2A task

        Returns:
            Tuple of (query, context)
        """
        if not task.messages:
            raise ValueError("Task has no messages")

        # Use the last message
        last_message = task.messages[-1]
        query = None
        context = {"task_id": task.id}

        for part in last_message.parts:
            if isinstance(part, DataPart):
                data = part.data
                if isinstance(data, dict):
                    query = data.get("query")
                    # Add all other data as context
                    context.update({k: v for k, v in data.items() if k != "query"})
            elif isinstance(part, TextPart):
                query = part.text

        if not query:
            raise ValueError("No query found in task messages")

        return query, context

    async def _execute_agent_workflow(
        self, routing_analysis: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        """
        Execute the agent workflow based on routing analysis.

        Args:
            routing_analysis: Analysis from routing agent
            task_id: Task identifier

        Returns:
            Dictionary of agent responses
        """
        agent_responses = {}
        # Handle both RoutingDecision object and dict for backward compatibility
        if hasattr(routing_analysis, 'metadata'):
            execution_plan = routing_analysis.metadata.get("execution_plan", [])
        else:
            execution_plan = routing_analysis.get("execution_plan", [])

        for step in execution_plan:
            agent_name = step["agent"]
            step["action"]
            parameters = step["parameters"]

            try:
                # Get agent endpoint
                agent_endpoint = self.agent_registry.get_agent(agent_name)
                if not agent_endpoint:
                    logger.warning(f"Agent {agent_name} not found in registry")
                    continue

                # Create A2A task for agent
                agent_task = self._create_agent_task(parameters, task_id, step["step"])

                # Send to agent
                response = await self._send_to_agent(agent_endpoint, agent_task)
                agent_responses[agent_name] = response

                logger.info(f"Agent {agent_name} completed successfully")

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                agent_responses[agent_name] = {"error": str(e)}

        return agent_responses

    def _create_agent_task(
        self, parameters: Dict[str, Any], task_id: str, step: int
    ) -> Task:
        """
        Create A2A task for downstream agent.

        Args:
            parameters: Parameters for the agent
            task_id: Original task ID
            step: Step number

        Returns:
            A2A Task for the agent
        """
        # Create data message with parameters
        message = create_data_message(parameters)

        # Create task with sub-task ID
        subtask_id = f"{task_id}_step_{step}"
        return create_task([message], subtask_id)

    async def _send_to_agent(
        self, agent_endpoint: AgentEndpoint, task: Task
    ) -> Dict[str, Any]:
        """
        Send A2A task to downstream agent.

        Args:
            agent_endpoint: Agent endpoint configuration
            task: A2A task to send

        Returns:
            Agent response
        """
        url = f"{agent_endpoint.url}{agent_endpoint.process_endpoint}"

        try:
            response = await self.http_client.post(
                url, json=task.dict(), timeout=agent_endpoint.timeout
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            raise Exception(f"Agent {agent_endpoint.name} timed out")
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"Agent {agent_endpoint.name} returned {e.response.status_code}"
            )
        except Exception as e:
            raise Exception(f"Failed to communicate with {agent_endpoint.name}: {e}")

    def _aggregate_results(
        self, routing_analysis: Dict[str, Any], agent_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from all agents based on workflow type.

        Args:
            routing_analysis: Original routing analysis
            agent_responses: Responses from all agents

        Returns:
            Aggregated final result
        """
        # Handle both RoutingDecision object and dict for backward compatibility
        if hasattr(routing_analysis, 'metadata'):
            workflow_type = routing_analysis.metadata.get("workflow_type", "raw_results")
        else:
            workflow_type = routing_analysis.get("workflow_type", "raw_results")

        # Collect search results
        search_results = []
        for agent_name, response in agent_responses.items():
            if agent_name in ["video_search", "text_search"] and "results" in response:
                search_results.extend(response["results"])

        # Base result structure
        # Handle both RoutingDecision object and dict for backward compatibility
        if hasattr(routing_analysis, 'query'):
            query = routing_analysis.query
            routing_decision = routing_analysis.recommended_agent
        else:
            query = routing_analysis.get("query")
            routing_decision = routing_analysis.get("routing_decision")

        final_result = {
            "workflow_type": workflow_type,
            "query": query,
            "routing_decision": routing_decision,
            "search_results": search_results,
            "agent_responses": agent_responses,
        }

        # Add workflow-specific processing
        if workflow_type == "summary":
            summarizer_response = agent_responses.get("summarizer")
            if summarizer_response:
                final_result["summary"] = summarizer_response.get("summary")
            else:
                # Fallback: create simple summary from search results
                final_result["summary"] = self._create_fallback_summary(search_results)

        elif workflow_type == "detailed_report":
            report_response = agent_responses.get("detailed_report")
            if report_response:
                final_result["detailed_report"] = report_response.get("report")
                final_result["reasoning"] = report_response.get("reasoning")
            else:
                # Fallback: create simple report from search results
                final_result["detailed_report"] = self._create_fallback_report(
                    search_results
                )

        return final_result

    def _create_fallback_summary(self, search_results: List[Dict[str, Any]]) -> str:
        """Create fallback summary when summarizer agent is not available"""
        if not search_results:
            return "No results found."

        num_results = len(search_results)
        sources = set()
        for result in search_results[:5]:  # Top 5 results
            if "source_id" in result:
                sources.add(result["source_id"])

        return (
            f"Found {num_results} results from {len(sources)} sources. "
            + f"Top results include content from: {', '.join(list(sources)[:3])}."
        )

    def _create_fallback_report(self, search_results: List[Dict[str, Any]]) -> str:
        """Create fallback detailed report when report agent is not available"""
        if not search_results:
            return "No results available for detailed analysis."

        report_parts = [f"Detailed Analysis of {len(search_results)} Results:", ""]

        for i, result in enumerate(search_results[:3]):  # Top 3 results
            score = result.get("score", 0)
            source = result.get("source_id", "Unknown")
            content_type = result.get("content_type", "unknown")

            report_parts.append(f"{i+1}. {source} ({content_type})")
            report_parts.append(f"   Relevance Score: {score:.3f}")

            if "metadata" in result:
                metadata = result["metadata"]
                if "start_time" in metadata:
                    report_parts.append(
                        f"   Time Range: {metadata.get('start_time')} - {metadata.get('end_time', 'end')}"
                    )

            report_parts.append("")

        return "\n".join(report_parts)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on A2A routing agent and downstream agents.

        Returns:
            Health status information
        """
        health_status = {
            "status": "healthy",
            "routing_agent": "healthy",
            "downstream_agents": {},
        }

        # Check downstream agents
        for agent_name in self.agent_registry.list_agents():
            try:
                agent_endpoint = self.agent_registry.get_agent(agent_name)
                if agent_endpoint:
                    health_url = f"{agent_endpoint.url}{agent_endpoint.health_endpoint}"
                    response = await self.http_client.get(health_url, timeout=5.0)

                    if response.status_code == 200:
                        health_status["downstream_agents"][agent_name] = "healthy"
                    else:
                        health_status["downstream_agents"][
                            agent_name
                        ] = f"unhealthy (status: {response.status_code})"
                        health_status["status"] = "degraded"
                else:
                    health_status["downstream_agents"][agent_name] = "not_registered"

            except Exception as e:
                health_status["downstream_agents"][
                    agent_name
                ] = f"unreachable ({str(e)})"
                health_status["status"] = "degraded"

        return health_status

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
