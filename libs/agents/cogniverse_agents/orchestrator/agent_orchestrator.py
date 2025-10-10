"""Enhanced Agent Orchestrator - Unified service for relationship-aware multi-agent processing."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from cogniverse_agents.result_aggregator import (
    AggregatedResult,
    AggregationRequest,
    ResultAggregator,
)
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDecision
from cogniverse_vespa.vespa.vespa_search_client import VespaVideoSearchClient
from cogniverse_core.config.utils import get_config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Agent Orchestrator",
    description="Unified orchestration service for relationship-aware multi-agent processing",
    version="1.0.0",
)


@dataclass
class ProcessingRequest:
    """Request for complete processing pipeline"""

    query: str
    tenant_id: str  # Tenant identifier (REQUIRED - no default)
    profiles: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    top_k: int = 20
    include_summaries: bool = True
    include_detailed_report: bool = True
    enable_relationship_extraction: bool = True
    enable_query_enhancement: bool = True
    max_results_to_process: int = 50
    agent_config: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Complete processing result"""

    original_query: str
    routing_decision: RoutingDecision
    aggregated_result: AggregatedResult
    processing_summary: Dict[str, Any]
    total_processing_time: float


class AgentOrchestrator:
    """Orchestrates the complete multi-agent processing pipeline"""

    def __init__(self, tenant_id: str, **kwargs):
        """
        Initialize agent orchestrator

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        logger.info(f"Initializing AgentOrchestrator for tenant: {tenant_id}...")

        self.tenant_id = tenant_id
        self.config = get_config()

        # Initialize components
        self.routing_agent = None
        self.result_aggregator = None
        self.vespa_client = None

        # Configuration
        self.default_profiles = kwargs.get(
            "default_profiles", ["video_colpali_smol500_mv_frame"]
        )
        self.default_strategies = kwargs.get("default_strategies", ["binary_binary"])
        self.enable_caching = kwargs.get("enable_caching", True)
        self.cache_ttl = kwargs.get("cache_ttl", 300)  # 5 minutes

        # Initialize components
        self._initialize_components(**kwargs)

        logger.info("AgentOrchestrator initialization complete")

    def _initialize_components(self, **kwargs):
        """Initialize orchestrator components"""
        try:
            # Initialize routing agent
            self.routing_agent = RoutingAgent(
                tenant_id=self.tenant_id,
                enable_telemetry=kwargs.get("enable_telemetry", True),
                **kwargs.get("routing_agent_config", {}),
            )
            logger.info(f"Routing agent initialized for tenant: {self.tenant_id}")

            # Initialize result aggregator
            self.result_aggregator = ResultAggregator(
                **kwargs.get("aggregator_config", {})
            )
            logger.info("Result aggregator initialized")

            # Initialize Vespa client
            self.vespa_client = VespaVideoSearchClient()
            logger.info("Vespa client initialized")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def process_complete_pipeline(
        self, request: ProcessingRequest
    ) -> ProcessingResult:
        """
        Process complete enhanced multi-agent pipeline

        Args:
            request: Processing request with query and configuration

        Returns:
            Complete processing result with routing, search, and agent outputs
        """
        logger.info(f"Starting complete pipeline for query: '{request.query}'")
        start_time = asyncio.get_event_loop().time()

        try:
            # Phase 1: Enhanced Routing
            routing_decision = await self._perform_enhanced_routing(request)

            # Phase 2: Enhanced Search
            search_results = await self._perform_enhanced_search(
                request, routing_decision
            )

            # Phase 3: Result Aggregation and Agent Processing
            aggregated_result = await self._aggregate_and_process_results(
                request, routing_decision, search_results
            )

            # Phase 4: Generate processing summary
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            processing_summary = self._generate_processing_summary(
                request, routing_decision, aggregated_result, total_time
            )

            # Create final result
            final_result = ProcessingResult(
                original_query=request.query,
                routing_decision=routing_decision,
                aggregated_result=aggregated_result,
                processing_summary=processing_summary,
                total_processing_time=total_time,
            )

            logger.info(f"Complete pipeline processed in {total_time:.2f}s")
            return final_result

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            # Return error result
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            return self._create_error_processing_result(request, str(e), total_time)

    async def _perform_enhanced_routing(
        self, request: ProcessingRequest
    ) -> RoutingDecision:
        """Perform enhanced routing with relationship extraction"""
        logger.info("Performing enhanced routing...")

        try:
            # Use routing agent to analyze query and extract relationships
            routing_decision = await self.routing_agent.analyze_and_route_with_relationships(
                query=request.query,
                enable_relationship_extraction=request.enable_relationship_extraction,
                enable_query_enhancement=request.enable_query_enhancement,
            )

            logger.info(
                f"Routing complete: {routing_decision.recommended_agent} "
                f"(confidence: {routing_decision.confidence:.2f})"
            )

            return routing_decision

        except Exception as e:
            logger.error(f"Enhanced routing failed: {e}")
            # Fallback to basic routing decision
            return RoutingDecision(
                query=request.query,
                enhanced_query=request.query,
                recommended_agent="enhanced_video_search",
                confidence=0.5,
                entities=[],
                relationships=[],
                metadata={"error": str(e), "fallback": True},
            )

    async def _perform_enhanced_search(
        self, request: ProcessingRequest, routing_decision: RoutingDecision
    ) -> List[Dict[str, Any]]:
        """Perform enhanced search with relationship context"""
        logger.info("Performing enhanced search...")

        try:
            # Determine search parameters
            profiles = request.profiles or self.default_profiles
            strategies = request.strategies or self.default_strategies

            # Use enhanced query if available
            search_query = routing_decision.enhanced_query or routing_decision.query

            # Perform search
            search_results = []
            for profile in profiles:
                for strategy in strategies:
                    try:
                        results = await self.vespa_client.query(
                            query_text=search_query,
                            profile=profile,
                            strategy=strategy,
                            top_k=request.top_k
                            // len(profiles),  # Distribute across profiles
                        )
                        search_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Search failed for {profile}/{strategy}: {e}")

            # Remove duplicates and limit results
            unique_results = self._deduplicate_results(search_results)
            limited_results = unique_results[: request.max_results_to_process]

            logger.info(f"Enhanced search returned {len(limited_results)} results")
            return limited_results

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []

    async def _aggregate_and_process_results(
        self,
        request: ProcessingRequest,
        routing_decision: RoutingDecision,
        search_results: List[Dict[str, Any]],
    ) -> AggregatedResult:
        """Aggregate results and process with agents"""
        logger.info("Aggregating and processing results...")

        try:
            # Create aggregation request
            aggregation_request = AggregationRequest(
                routing_decision=routing_decision,
                search_results=search_results,
                agents_to_invoke=None,  # Let aggregator decide based on includes
                include_summaries=request.include_summaries,
                include_detailed_report=request.include_detailed_report,
                max_results_to_process=request.max_results_to_process,
                enhancement_config=request.agent_config,
            )

            # Perform aggregation and enhancement
            aggregated_result = await self.result_aggregator.aggregate_and_enhance(
                aggregation_request
            )

            logger.info("Result aggregation and processing complete")
            return aggregated_result

        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            # Return minimal aggregated result
            from cogniverse_agents.result_aggregator import AggregatedResult

            return AggregatedResult(
                routing_decision=routing_decision,
                enhanced_search_results=[],
                agent_results={},
                summaries=None,
                detailed_report=None,
                enhancement_statistics={"error": True, "error_message": str(e)},
                aggregation_metadata={"error": True, "error_message": str(e)},
                total_processing_time=0.0,
            )

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on ID or content"""
        seen_ids = set()
        unique_results = []

        for result in results:
            # Use ID if available, otherwise use title/content hash
            result_id = result.get("id") or result.get("video_id")
            if not result_id:
                # Create hash from title and first 100 chars of content/description
                content = (
                    result.get("title", "")
                    + result.get("description", "")[:100]
                    + result.get("content", "")[:100]
                )
                result_id = str(hash(content))

            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results

    def _generate_processing_summary(
        self,
        request: ProcessingRequest,
        routing_decision: RoutingDecision,
        aggregated_result: AggregatedResult,
        total_time: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive processing summary"""

        # Get aggregation summary
        aggregation_summary = self.result_aggregator.get_aggregation_summary(
            aggregated_result
        )

        return {
            "query_analysis": {
                "original_query": request.query,
                "enhanced_query": routing_decision.enhanced_query,
                "query_enhanced": routing_decision.enhanced_query
                != routing_decision.query,
                "routing_confidence": routing_decision.confidence,
                "recommended_agent": routing_decision.recommended_agent,
            },
            "relationship_extraction": {
                "entities_identified": len(routing_decision.entities),
                "relationships_identified": len(routing_decision.relationships),
                "extraction_enabled": request.enable_relationship_extraction,
                "enhancement_enabled": request.enable_query_enhancement,
            },
            "search_performance": {
                "profiles_used": request.profiles or self.default_profiles,
                "strategies_used": request.strategies or self.default_strategies,
                "results_found": aggregation_summary.get("search_results_processed", 0),
                "enhancement_rate": aggregation_summary.get("enhancement_rate", 0),
            },
            "agent_processing": {
                "agents_invoked": aggregation_summary.get("agents_invoked", 0),
                "successful_agents": aggregation_summary.get("successful_agents", 0),
                "summaries_generated": aggregation_summary.get("has_summaries", False),
                "detailed_report_generated": aggregation_summary.get(
                    "has_detailed_report", False
                ),
            },
            "performance_metrics": {
                "total_processing_time": total_time,
                "routing_enhanced": True,
                "results_enhanced": True,
                "multi_agent_processing": True,
            },
            "configuration": {
                "top_k": request.top_k,
                "max_results_processed": request.max_results_to_process,
                "relationship_extraction_enabled": request.enable_relationship_extraction,
                "query_enhancement_enabled": request.enable_query_enhancement,
            },
        }

    def _create_error_processing_result(
        self, request: ProcessingRequest, error_message: str, total_time: float
    ) -> ProcessingResult:
        """Create error processing result"""

        # Create minimal routing decision
        error_routing_decision = RoutingDecision(
            query=request.query,
            enhanced_query=request.query,
            recommended_agent="error",
            confidence=0.0,
            entities=[],
            relationships=[],
            metadata={"error": True, "error_message": error_message},
        )

        # Create minimal aggregated result
        from cogniverse_agents.result_aggregator import AggregatedResult

        error_aggregated_result = AggregatedResult(
            routing_decision=error_routing_decision,
            enhanced_search_results=[],
            agent_results={},
            summaries=None,
            detailed_report=None,
            enhancement_statistics={"error": True, "error_message": error_message},
            aggregation_metadata={"error": True, "error_message": error_message},
            total_processing_time=total_time,
        )

        return ProcessingResult(
            original_query=request.query,
            routing_decision=error_routing_decision,
            aggregated_result=error_aggregated_result,
            processing_summary={
                "error": True,
                "error_message": error_message,
                "total_processing_time": total_time,
            },
            total_processing_time=total_time,
        )


# Per-tenant orchestrator instances cache
orchestrators: Dict[str, AgentOrchestrator] = {}


def get_orchestrator(tenant_id: str) -> AgentOrchestrator:
    """
    Get or create orchestrator instance for tenant

    Args:
        tenant_id: Tenant identifier

    Returns:
        AgentOrchestrator instance for the tenant
    """
    if tenant_id not in orchestrators:
        logger.info(f"Creating new orchestrator for tenant: {tenant_id}")
        orchestrators[tenant_id] = AgentOrchestrator(tenant_id=tenant_id)
    return orchestrators[tenant_id]


@app.on_event("startup")
async def startup_event():
    """Startup event - orchestrators created on-demand per tenant"""
    logger.info("Enhanced agent orchestrator service started - ready for multi-tenant requests")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent_orchestrator",
        "capabilities": [
            "enhanced_routing",
            "relationship_extraction",
            "query_enhancement",
            "multi_agent_processing",
            "result_aggregation",
        ],
    }


@app.post("/process")
async def process_pipeline(request: ProcessingRequest):
    """
    Process complete enhanced multi-agent pipeline

    The tenant_id can be provided in the request body.
    If not provided, defaults to "default_tenant".
    """
    try:
        # Get orchestrator for the tenant
        orchestrator = get_orchestrator(request.tenant_id)
        result = await orchestrator.process_complete_pipeline(request)
        return result

    except Exception as e:
        logger.error(f"Pipeline processing error for tenant {request.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/routing-only")
async def process_routing_only(
    query: str,
    tenant_id: str,
    enable_relationship_extraction: bool = True,
    enable_query_enhancement: bool = True,
):
    """
    Process only the routing phase

    Args:
        query: Query to route
        tenant_id: Tenant identifier (REQUIRED - no default)
        enable_relationship_extraction: Enable relationship extraction
        enable_query_enhancement: Enable query enhancement
    """
    try:
        # Get orchestrator for the tenant
        orchestrator = get_orchestrator(tenant_id)
        routing_decision = (
            await orchestrator.routing_agent.analyze_and_route_with_relationships(
                query=query,
                enable_relationship_extraction=enable_relationship_extraction,
                enable_query_enhancement=enable_query_enhancement,
            )
        )
        return routing_decision

    except Exception as e:
        logger.error(f"Routing processing error for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_configuration(tenant_id: str):
    """
    Get current orchestrator configuration for a tenant

    Args:
        tenant_id: Tenant identifier (REQUIRED - no default)
    """
    try:
        orchestrator = get_orchestrator(tenant_id)
        return {
            "tenant_id": tenant_id,
            "default_profiles": orchestrator.default_profiles,
            "default_strategies": orchestrator.default_strategies,
            "enable_caching": orchestrator.enable_caching,
            "cache_ttl": orchestrator.cache_ttl,
            "components_initialized": {
                "routing_agent": orchestrator.routing_agent is not None,
                "result_aggregator": orchestrator.result_aggregator is not None,
                "vespa_client": orchestrator.vespa_client is not None,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get config for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
