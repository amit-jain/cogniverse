"""Enhanced Result Aggregator with Relationship Context and Multi-Agent Integration."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cogniverse_agents.results.enhancement_engine import (
    EnhancedResult,
    EnhancementContext,
    ResultEnhancementEngine,
)
from cogniverse_agents.routing_agent import RoutingOutput

logger = logging.getLogger(__name__)


@dataclass
class AggregationRequest:
    """Request for result aggregation and enhancement"""

    routing_decision: RoutingOutput
    search_results: List[Dict[str, Any]]
    agents_to_invoke: Optional[List[str]] = None
    include_summaries: bool = True
    include_detailed_report: bool = True
    max_results_to_process: int = 50
    enhancement_config: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """Result from a specific agent"""

    agent_name: str
    result_data: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class AggregatedResult:
    """Aggregated and enhanced result from multiple agents"""

    routing_decision: RoutingOutput
    enhanced_search_results: List[EnhancedResult]
    agent_results: Dict[str, AgentResult]
    summaries: Optional[Dict[str, Any]] = None
    detailed_report: Optional[Dict[str, Any]] = None
    enhancement_statistics: Dict[str, Any] = None
    aggregation_metadata: Dict[str, Any] = None
    total_processing_time: float = 0.0


class ResultAggregator:
    """Aggregates and enhances results from multiple agents with relationship context"""

    def __init__(self, **kwargs):
        """Initialize result aggregator"""
        logger.info("Initializing ResultAggregator...")

        # Initialize enhancement engine
        self.enhancement_engine = ResultEnhancementEngine(
            **kwargs.get("enhancement_config", {})
        )

        # Configuration
        self.max_concurrent_agents = kwargs.get("max_concurrent_agents", 3)
        self.agent_timeout = kwargs.get("agent_timeout", 30.0)
        self.enable_fallbacks = kwargs.get("enable_fallbacks", True)

        # Agent endpoints (would typically be configured externally)
        self.agent_endpoints = {
            "summarizer": "http://localhost:8002",
            "detailed_report": "http://localhost:8003",
            "enhanced_video_search": "http://localhost:8001",
        }

        logger.info("ResultAggregator initialization complete")

    async def aggregate_and_enhance(
        self, request: AggregationRequest
    ) -> AggregatedResult:
        """
        Aggregate and enhance results from multiple agents

        Args:
            request: Aggregation request with routing decision and search results

        Returns:
            Aggregated result with enhanced search results and agent outputs
        """
        logger.info(
            f"Aggregating results for query: '{request.routing_decision.query}'"
        )
        start_time = asyncio.get_event_loop().time()

        try:
            # Phase 1: Enhance search results with relationship context
            enhanced_results = await self._enhance_search_results(request)

            # Phase 2: Invoke agents in parallel
            agent_results = await self._invoke_agents_parallel(
                request, enhanced_results
            )

            # Phase 3: Process agent results
            summaries = None
            detailed_report = None

            if request.include_summaries and "summarizer" in agent_results:
                summaries = agent_results["summarizer"].result_data

            if request.include_detailed_report and "detailed_report" in agent_results:
                detailed_report = agent_results["detailed_report"].result_data

            # Phase 4: Generate enhancement statistics
            enhancement_stats = self.enhancement_engine.get_enhancement_statistics(
                enhanced_results
            )

            # Phase 5: Create aggregation metadata
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            aggregation_metadata = {
                "query": request.routing_decision.query,
                "enhanced_query": request.routing_decision.enhanced_query,
                "routing_confidence": request.routing_decision.confidence,
                "agents_invoked": list(agent_results.keys()),
                "successful_agents": [
                    name for name, result in agent_results.items() if result.success
                ],
                "failed_agents": [
                    name for name, result in agent_results.items() if not result.success
                ],
                "original_results_count": len(request.search_results),
                "enhanced_results_count": len(enhanced_results),
                "entities_identified": len(request.routing_decision.entities),
                "relationships_identified": len(request.routing_decision.relationships),
                "processing_time": total_time,
            }

            # Create aggregated result
            aggregated_result = AggregatedResult(
                routing_decision=request.routing_decision,
                enhanced_search_results=enhanced_results,
                agent_results=agent_results,
                summaries=summaries,
                detailed_report=detailed_report,
                enhancement_statistics=enhancement_stats,
                aggregation_metadata=aggregation_metadata,
                total_processing_time=total_time,
            )

            logger.info(f"Result aggregation complete in {total_time:.2f}s")
            return aggregated_result

        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            # Return minimal result with error information
            return self._create_error_result(request, str(e), start_time)

    async def _enhance_search_results(
        self, request: AggregationRequest
    ) -> List[EnhancedResult]:
        """Enhance search results with relationship context"""
        logger.info("Enhancing search results with relationship context...")

        # Create enhancement context
        context = EnhancementContext(
            entities=request.routing_decision.entities,
            relationships=request.routing_decision.relationships,
            query=request.routing_decision.query,
            enhanced_query=request.routing_decision.enhanced_query,
            routing_confidence=request.routing_decision.confidence,
            enhancement_metadata=request.enhancement_config or {},
        )

        # Limit results if needed
        results_to_enhance = request.search_results[: request.max_results_to_process]

        # Enhance results
        enhanced_results = self.enhancement_engine.enhance_results(
            results_to_enhance, context
        )

        logger.info(f"Enhanced {len(enhanced_results)} search results")
        return enhanced_results

    async def _invoke_agents_parallel(
        self, request: AggregationRequest, enhanced_results: List[EnhancedResult]
    ) -> Dict[str, AgentResult]:
        """Invoke multiple agents in parallel"""
        logger.info("Invoking agents in parallel...")

        # Determine which agents to invoke
        agents_to_invoke = request.agents_to_invoke or []
        if request.include_summaries and "summarizer" not in agents_to_invoke:
            agents_to_invoke.append("summarizer")
        if (
            request.include_detailed_report
            and "detailed_report" not in agents_to_invoke
        ):
            agents_to_invoke.append("detailed_report")

        # Create agent tasks
        agent_tasks = []
        for agent_name in agents_to_invoke:
            if agent_name in self.agent_endpoints:
                task = asyncio.create_task(
                    self._invoke_single_agent(agent_name, request, enhanced_results),
                    name=f"agent_{agent_name}",
                )
                agent_tasks.append(task)

        # Execute agents with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)

        async def execute_with_semaphore(task):
            async with semaphore:
                return await task

        # Wait for all agents to complete (with timeout)
        try:
            agent_results_list = await asyncio.wait_for(
                asyncio.gather(
                    *[execute_with_semaphore(task) for task in agent_tasks],
                    return_exceptions=True,
                ),
                timeout=self.agent_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent invocation timeout - some agents may not have completed"
            )
            agent_results_list = []
            for task in agent_tasks:
                if not task.done():
                    task.cancel()
                try:
                    result = task.result() if task.done() else None
                    if result:
                        agent_results_list.append(result)
                except Exception as e:
                    logger.warning(f"Failed to get result from cancelled task: {e}")

        # Process agent results
        agent_results = {}
        for result in agent_results_list:
            if isinstance(result, AgentResult):
                agent_results[result.agent_name] = result
            elif isinstance(result, Exception):
                logger.error(f"Agent execution exception: {result}")

        logger.info(f"Completed {len(agent_results)} agent invocations")
        return agent_results

    async def _invoke_single_agent(
        self,
        agent_name: str,
        request: AggregationRequest,
        enhanced_results: List[EnhancedResult],
    ) -> AgentResult:
        """Invoke a single agent with enhanced results"""
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(f"Invoking {agent_name} agent...")

            # Prepare agent request data
            agent_request_data = self._prepare_agent_request_data(
                agent_name, request, enhanced_results
            )

            # Simulate agent invocation (in real implementation, would use HTTP client)
            result_data = await self._simulate_agent_invocation(
                agent_name, agent_request_data
            )

            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            logger.info(f"Agent {agent_name} completed in {processing_time:.2f}s")

            return AgentResult(
                agent_name=agent_name,
                result_data=result_data,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            logger.error(f"Agent {agent_name} failed after {processing_time:.2f}s: {e}")

            return AgentResult(
                agent_name=agent_name,
                result_data={},
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )

    def _prepare_agent_request_data(
        self,
        agent_name: str,
        request: AggregationRequest,
        enhanced_results: List[EnhancedResult],
    ) -> Dict[str, Any]:
        """Prepare request data for specific agent"""

        # Convert enhanced results back to dict format for agents
        search_results = [result.original_result for result in enhanced_results]

        base_data = {
            "routing_decision": {
                "query": request.routing_decision.query,
                "enhanced_query": request.routing_decision.enhanced_query,
                "entities": request.routing_decision.entities,
                "relationships": request.routing_decision.relationships,
                "confidence": request.routing_decision.confidence,
                "recommended_agent": request.routing_decision.recommended_agent,
                "metadata": request.routing_decision.metadata,
            },
            "search_results": search_results,
            "enhancement_applied": True,
            "enhanced_result_metadata": [
                result.enhancement_metadata for result in enhanced_results
            ],
        }

        # Agent-specific customization
        if agent_name == "summarizer":
            base_data.update(
                {
                    "summary_type": "comprehensive",
                    "focus_on_relationships": True,
                    "include_entity_analysis": True,
                }
            )
        elif agent_name == "detailed_report":
            base_data.update(
                {
                    "report_type": "comprehensive",
                    "include_visual_analysis": True,
                    "include_technical_details": True,
                    "include_recommendations": True,
                    "focus_on_relationships": True,
                }
            )

        return base_data

    async def _simulate_agent_invocation(
        self, agent_name: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate agent invocation (placeholder for real HTTP calls)"""

        # Simulate processing delay
        await asyncio.sleep(0.1 + (hash(agent_name) % 3) * 0.1)  # 0.1-0.4 seconds

        # Return mock results based on agent type
        if agent_name == "summarizer":
            return {
                "summary": f"Enhanced summary for query '{request_data['routing_decision']['query']}' with relationship context",
                "key_entities": [
                    e["text"] for e in request_data["routing_decision"]["entities"][:3]
                ],
                "key_relationships": [
                    f"{r['subject']} {r['relation']} {r['object']}"
                    for r in request_data["routing_decision"]["relationships"][:2]
                ],
                "enhancement_applied": True,
                "confidence": 0.85,
            }
        elif agent_name == "detailed_report":
            return {
                "executive_summary": f"Detailed analysis of '{request_data['routing_decision']['query']}' with relationship-enhanced results",
                "findings_count": len(request_data["search_results"]),
                "entity_analysis": {
                    "entities_identified": len(
                        request_data["routing_decision"]["entities"]
                    ),
                    "relationships_identified": len(
                        request_data["routing_decision"]["relationships"]
                    ),
                },
                "recommendations": [
                    "Leverage relationship context for improved accuracy",
                    "Consider entity-driven result filtering",
                ],
                "enhancement_applied": True,
            }
        else:
            return {
                "result": f"Processed by {agent_name}",
                "enhancement_applied": True,
                "processing_metadata": {"agent": agent_name},
            }

    def _create_error_result(
        self, request: AggregationRequest, error_message: str, start_time: float
    ) -> AggregatedResult:
        """Create error result when aggregation fails"""
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        return AggregatedResult(
            routing_decision=request.routing_decision,
            enhanced_search_results=[],
            agent_results={},
            summaries=None,
            detailed_report=None,
            enhancement_statistics={"error": True, "error_message": error_message},
            aggregation_metadata={
                "error": True,
                "error_message": error_message,
                "processing_time": total_time,
            },
            total_processing_time=total_time,
        )

    def get_aggregation_summary(self, result: AggregatedResult) -> Dict[str, Any]:
        """Get summary of aggregation results"""
        return {
            "query": result.routing_decision.query,
            "enhanced_query": result.routing_decision.enhanced_query,
            "routing_confidence": result.routing_decision.confidence,
            "search_results_processed": len(result.enhanced_search_results),
            "agents_invoked": len(result.agent_results),
            "successful_agents": sum(
                1 for r in result.agent_results.values() if r.success
            ),
            "enhancement_rate": (
                result.enhancement_statistics.get("enhancement_rate", 0)
                if result.enhancement_statistics
                else 0
            ),
            "total_processing_time": result.total_processing_time,
            "has_summaries": result.summaries is not None,
            "has_detailed_report": result.detailed_report is not None,
        }
