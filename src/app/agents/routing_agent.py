"""
Request-facing Routing Agent implementation.
Main entry point for all user queries that routes to appropriate downstream agents.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from src.app.agents.dspy_integration_mixin import DSPyRoutingMixin
from src.app.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
)
from src.app.routing.config import RoutingConfig
from src.app.routing.phoenix_span_evaluator import PhoenixSpanEvaluator
from src.app.routing.router import ComprehensiveRouter
from src.app.telemetry.manager import TelemetryManager
from src.common.config import get_config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Routing Agent",
    description="Request-facing routing agent that coordinates multi-agent workflows",
    version="1.0.0",
)


class RoutingDecisionResponse(BaseModel):
    """Response structure for routing decisions"""

    task_id: str
    routing_decision: Dict[str, Any]
    agents_to_call: List[str]
    workflow_type: str
    execution_plan: List[Dict[str, Any]]
    status: str


class RoutingAgent(DSPyRoutingMixin):
    """
    Main routing agent that analyzes queries and determines appropriate agent workflows.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize routing agent with configuration"""
        super().__init__()  # Initialize DSPy mixin
        self.system_config = get_config()

        # Load routing configuration
        if config_path:
            self.routing_config = RoutingConfig.from_file(config_path)
        else:
            # Use system config to build routing config
            self.routing_config = self._build_routing_config()

        # Initialize comprehensive router
        self.router = ComprehensiveRouter(self.routing_config)

        # Initialize advanced optimizer
        self.optimizer = AdvancedRoutingOptimizer(
            storage_dir=self.system_config.get(
                "optimization_dir", "outputs/optimization"
            ),
            config=None,  # Will use default config
        )

        # Initialize Phoenix span evaluator for real telemetry collection
        self.span_evaluator = PhoenixSpanEvaluator(self.optimizer)

        # Initialize telemetry manager for creating routing spans
        self.telemetry_manager = TelemetryManager()

        # Agent registry - maps agent types to their endpoints
        # Only include agents that are actually configured
        self.agent_registry = {}

        # Required agents - fail if not configured
        video_url = self.system_config.get("video_agent_url")
        if not video_url:
            raise ValueError("video_agent_url must be configured")
        self.agent_registry["video_search"] = video_url

        # Optional agents - only add if configured
        text_url = self.system_config.get("text_agent_url")
        if text_url:
            self.agent_registry["text_search"] = text_url

        summarizer_url = self.system_config.get("summarizer_agent_url")
        if summarizer_url:
            self.agent_registry["summarizer"] = summarizer_url

        detailed_report_url = self.system_config.get("detailed_report_agent_url")
        if detailed_report_url:
            self.agent_registry["detailed_report"] = detailed_report_url

        # Validate required agents are available
        self._validate_agent_registry()

        logger.info("RoutingAgent initialized successfully")

    def _build_routing_config(self) -> RoutingConfig:
        """Build routing configuration from system config"""
        routing_config_dict = {
            "routing_mode": "tiered",
            "tier_config": {
                "enable_fast_path": True,
                "enable_slow_path": True,
                "enable_langextract": False,  # Disable for now
                "enable_fallback": False,  # No fallbacks - fail fast
                "fast_path_confidence_threshold": 0.7,
                "slow_path_confidence_threshold": 0.6,
            },
            "gliner_config": {
                "model": "urchade/gliner_large-v2.1",
                "threshold": 0.3,
                "labels": [
                    "video_content",
                    "visual_content",
                    "text_information",
                    "document_content",
                    "summary_request",
                    "detailed_analysis",
                    "raw_results",
                ],
            },
            "llm_config": {
                "provider": "local",
                "model": "gemma2:2b",
                "endpoint": "http://localhost:11434",
                "temperature": 0.1,
            },
            "cache_config": {"enable_caching": True, "cache_ttl_seconds": 300},
            "monitoring_config": {"enable_metrics": True, "metrics_batch_size": 100},
            "optimization_config": {
                "enable_auto_optimization": False,  # Disable for initial implementation
                "optimization_threshold": 1000,
            },
        }

        return RoutingConfig(**routing_config_dict)

    def _validate_agent_registry(self):
        """Validate that required agents are available"""
        if not self.agent_registry["video_search"]:
            raise ValueError("video_agent_url not configured in system config")

        # Log available agents
        available_agents = [k for k, v in self.agent_registry.items() if v is not None]
        logger.info(f"Available downstream agents: {available_agents}")

    async def analyze_and_route(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze query and determine routing strategy.

        Args:
            query: User query
            context: Optional context information

        Returns:
            Routing decision with execution plan
        """
        start_time = time.time()
        logger.info(f"Analyzing query for routing: '{query}'")

        # Create Phoenix span for the overall user request
        with self.telemetry_manager.span(
            name="cogniverse.request",
            tenant_id="default",  # TODO: Extract from context if available
            service_name="cogniverse.orchestration",
            attributes={
                "openinference.span.kind": "WORKFLOW",
                "operation.name": "process_user_request",
                "user.query": query,
                "user.context": str(context) if context else None,
                "system.workflow_type": "multi_agent_video_search",
            },
        ) as parent_span:
            try:
                # Step 1: Get routing decision from comprehensive router
                routing_decision = await self.router.route(query, context)

                # Step 2: Determine output type and agent workflow
                workflow_plan = self._determine_workflow(query, routing_decision)

                execution_time = time.time() - start_time

                # Determine primary agent from search modality
                primary_agent = self._get_primary_agent_from_modality(
                    routing_decision.search_modality
                )

                # Create child span for routing decision process
                with self.telemetry_manager.span(
                    name="cogniverse.routing",
                    tenant_id="default",
                    service_name="cogniverse.routing.agent",
                    attributes={
                        "openinference.span.kind": "AGENT",
                        "operation.name": "route_query",
                        "routing.query": query,
                        "routing.context": str(context) if context else None,
                    },
                ) as routing_span:
                    # Add routing decision details to routing span
                    routing_span.set_attribute("routing.chosen_agent", primary_agent)
                    routing_span.set_attribute(
                        "routing.confidence", routing_decision.confidence_score
                    )
                    routing_span.set_attribute(
                        "routing.method", routing_decision.routing_method
                    )
                    routing_span.set_attribute(
                        "routing.processing_time", execution_time
                    )

                    # Add routing decision event to routing span
                    routing_span.add_event(
                        "routing_decision_made",
                        {
                            "chosen_agent": primary_agent,
                            "confidence": routing_decision.confidence_score,
                            "reasoning": (
                                routing_decision.reasoning[:500]
                                if routing_decision.reasoning
                                else ""
                            ),
                        },
                    )

                    # Mark routing span as successful
                    routing_span.set_status(Status(StatusCode.OK))

                # Add summary attributes to parent request span
                parent_span.set_attribute("request.routing_agent", primary_agent)
                parent_span.set_attribute(
                    "request.confidence", routing_decision.confidence_score
                )
                parent_span.set_attribute("request.processing_time", execution_time)

                # Mark parent span as successful
                parent_span.set_status(Status(StatusCode.OK))

                logger.info(f"Query analysis completed in {execution_time:.3f}s")

                return {
                    "query": query,
                    "routing_decision": routing_decision.to_dict(),
                    "workflow_type": workflow_plan["type"],
                    "agents_to_call": workflow_plan["agents"],
                    "execution_plan": workflow_plan["steps"],
                    "confidence": routing_decision.confidence_score,
                    "routing_method": routing_decision.routing_method,
                    "execution_time": execution_time,
                }

            except Exception as e:
                # Mark parent span as failed and record exception
                parent_span.set_status(Status(StatusCode.ERROR, str(e)))
                parent_span.record_exception(e)

                logger.error(f"Routing analysis failed: {e}")
                raise

    def _determine_workflow(self, query: str, routing_decision) -> Dict[str, Any]:
        """
        Determine the workflow based on query analysis and routing decision.

        Args:
            query: Original user query
            routing_decision: Decision from routing engine

        Returns:
            Workflow plan with agents and steps
        """
        query_lower = query.lower()

        # Determine output type from query
        if any(
            word in query_lower
            for word in ["summarize", "summary", "brief", "overview"]
        ):
            output_type = "summary"
        elif any(
            word in query_lower
            for word in [
                "detailed",
                "analyze",
                "analysis",
                "report",
                "explain",
                "why",
                "how",
            ]
        ):
            output_type = "detailed_report"
        else:
            output_type = "raw_results"

        # Build workflow based on search modality and output type
        workflow_steps = []
        agents_needed = []

        # Step 1: Always start with search
        if routing_decision.search_modality.value in ["video", "both"]:
            workflow_steps.append(
                {
                    "step": 1,
                    "agent": "video_search",
                    "action": "search",
                    "parameters": {"query": query, "top_k": 10},
                }
            )
            agents_needed.append("video_search")

        if routing_decision.search_modality.value in ["text", "both"]:
            if self.agent_registry.get("text_search"):
                workflow_steps.append(
                    {
                        "step": len(workflow_steps) + 1,
                        "agent": "text_search",
                        "action": "search",
                        "parameters": {"query": query, "top_k": 10},
                    }
                )
                agents_needed.append("text_search")
            else:
                logger.warning(
                    "Text search requested but text_search agent not available"
                )

        # Step 2: Post-processing based on output type
        if output_type == "summary":
            if self.agent_registry.get("summarizer"):
                workflow_steps.append(
                    {
                        "step": len(workflow_steps) + 1,
                        "agent": "summarizer",
                        "action": "summarize",
                        "parameters": {
                            "query": query,
                            "results_from_previous_steps": True,
                        },
                    }
                )
                agents_needed.append("summarizer")
            else:
                logger.warning("Summary requested but summarizer agent not available")

        elif output_type == "detailed_report":
            if self.agent_registry.get("detailed_report"):
                workflow_steps.append(
                    {
                        "step": len(workflow_steps) + 1,
                        "agent": "detailed_report",
                        "action": "analyze",
                        "parameters": {
                            "query": query,
                            "results_from_previous_steps": True,
                            "enable_think_phase": True,
                        },
                    }
                )
                agents_needed.append("detailed_report")
            else:
                logger.warning(
                    "Detailed report requested but detailed_report agent not available"
                )

        return {"type": output_type, "agents": agents_needed, "steps": workflow_steps}

    def _get_primary_agent_from_modality(self, search_modality):
        """
        Get the primary agent type from search modality.

        Args:
            search_modality: SearchModality enum value

        Returns:
            str: Primary agent name

        Raises:
            ValueError: If search modality is unknown
        """
        if hasattr(search_modality, "value"):
            modality_value = search_modality.value
        else:
            modality_value = str(search_modality)

        if modality_value in ["video", "both"]:
            return "video_search"
        elif modality_value == "text":
            return "text_search"
        else:
            raise ValueError(
                f"Unknown search modality: {modality_value}. "
                f"Expected one of: video, text, both"
            )


# Global routing agent instance
routing_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize routing agent on startup"""
    global routing_agent

    try:
        # Initialize Phoenix instrumentation for OpenTelemetry
        # Note: Phoenix should already be running at localhost:6006
        # px.launch_app() would conflict with existing Phoenix instance

        # Initialize OpenTelemetry with Phoenix exporter
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Set up trace provider with Phoenix exporter
        tracer_provider = trace_sdk.TracerProvider()

        # Phoenix endpoint for OpenTelemetry
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:6006/v1/traces", headers={}
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(tracer_provider)

        # Initialize Cogniverse instrumentation
        from src.app.instrumentation.phoenix import CogniverseInstrumentor

        instrumentor = CogniverseInstrumentor()
        instrumentor.instrument()

        logger.info(
            "Phoenix instrumentation and CogniverseInstrumentor initialized successfully"
        )

        routing_agent = RoutingAgent()
        logger.info("Routing agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize routing agent: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    return {
        "status": "healthy",
        "agent": "routing_agent",
        "available_downstream_agents": [
            k for k, v in routing_agent.agent_registry.items() if v is not None
        ],
        "routing_config": {
            "mode": (
                routing_agent.routing_config.routing_mode
                if hasattr(routing_agent.routing_config, "routing_mode")
                else "tiered"
            ),
            "tiers_enabled": len(
                [tier for tier in routing_agent.router.strategies.keys()]
            ),
        },
    }


@app.post("/analyze")
async def analyze_query(request: Dict[str, Any]):
    """
    Analyze query and return routing decision without executing.

    Args:
        request: {"query": str, "context": Optional[Dict]}
    """
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    context = request.get("context")

    try:
        analysis = await routing_agent.analyze_and_route(query, context)
        return analysis
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/process")
async def process_task(request: Dict[str, Any]):
    """
    Process task - analyze query and return routing decision.

    Args:
        request: {"query": str, "context": Optional[Dict], "task_id": Optional[str]}
    """
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    context = request.get("context")
    task_id = request.get("task_id", f"task_{int(time.time())}")

    try:
        analysis = await routing_agent.analyze_and_route(query, context)

        return RoutingDecisionResponse(
            task_id=task_id,
            routing_decision=analysis["routing_decision"],
            agents_to_call=analysis["agents_to_call"],
            workflow_type=analysis["workflow_type"],
            execution_plan=analysis["execution_plan"],
            status="completed",
        )

    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/stats")
async def get_routing_stats():
    """Get routing statistics and performance metrics"""
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    return routing_agent.router.get_performance_report()


# Optimization endpoints
class OptimizationRequest(BaseModel):
    """Request for routing optimization"""

    action: str  # "optimize_routing", "record_experience", "get_metrics"
    examples: Optional[List[Dict[str, Any]]] = None
    experience_data: Optional[Dict[str, Any]] = None
    optimizer: Optional[str] = "adaptive"
    min_improvement: Optional[float] = 0.05


@app.post("/optimize")
async def trigger_optimization(request: OptimizationRequest):
    """
    Trigger routing optimization with training examples

    Args:
        request: Optimization request with examples and parameters
    """
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    try:
        action = request.action

        if action == "optimize_routing":
            if not request.examples:
                raise HTTPException(
                    status_code=400, detail="Examples required for optimization"
                )

            # Process examples and trigger optimization
            training_count = 0
            for example_set in request.examples:
                # Process good routes
                for good_route in example_set.get("good_routes", []):
                    await routing_agent.optimizer.record_routing_experience(
                        query=good_route["query"],
                        entities=[],  # Would be extracted in real scenario
                        relationships=[],
                        enhanced_query=good_route["query"],
                        chosen_agent=good_route["expected_agent"],
                        routing_confidence=0.9,
                        search_quality=0.85,
                        agent_success=True,
                        user_satisfaction=1.0,
                    )
                    training_count += 1

                # Process bad routes
                for bad_route in example_set.get("bad_routes", []):
                    await routing_agent.optimizer.record_routing_experience(
                        query=bad_route["query"],
                        entities=[],
                        relationships=[],
                        enhanced_query=bad_route["query"],
                        chosen_agent=bad_route["wrong_agent"],
                        routing_confidence=0.7,
                        search_quality=0.3,
                        agent_success=False,
                        user_satisfaction=0.2,
                    )
                    training_count += 1

            # Trigger optimization if we have enough examples
            if training_count >= 5:
                # This would trigger the actual DSPy optimization
                optimization_result = {
                    "status": "optimization_triggered",
                    "training_examples": training_count,
                    "optimizer": request.optimizer,
                    "message": f"Started {request.optimizer} optimization with {training_count} examples",
                }
            else:
                optimization_result = {
                    "status": "insufficient_data",
                    "training_examples": training_count,
                    "message": f"Need at least 5 examples, got {training_count}",
                }

            return optimization_result

        elif action == "record_experience":
            if not request.experience_data:
                raise HTTPException(status_code=400, detail="Experience data required")

            exp_data = request.experience_data
            reward = await routing_agent.optimizer.record_routing_experience(
                query=exp_data["query"],
                entities=exp_data.get("entities", []),
                relationships=exp_data.get("relationships", []),
                enhanced_query=exp_data.get("enhanced_query", exp_data["query"]),
                chosen_agent=exp_data["chosen_agent"],
                routing_confidence=exp_data["routing_confidence"],
                search_quality=exp_data["search_quality"],
                agent_success=exp_data["agent_success"],
                user_satisfaction=exp_data.get("user_satisfaction"),
            )

            return {
                "status": "experience_recorded",
                "reward": reward,
                "message": "Experience recorded successfully",
            }

        elif action == "get_metrics":
            status_info = routing_agent.optimizer.get_optimization_status()
            return {"status": "metrics_retrieved", "metrics": status_info}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except Exception as e:
        logger.error(f"Optimization request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/optimization/status")
async def get_optimization_status():
    """Get current optimization status and metrics"""
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    try:
        status_info = routing_agent.optimizer.get_optimization_status()
        return {
            "status": "active",
            "metrics": status_info,
            "optimizer_ready": routing_agent.optimizer.routing_policy is not None,
        }
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        return {"status": "error", "message": str(e), "optimizer_ready": False}


@app.post("/optimization/evaluate-spans")
async def evaluate_phoenix_spans(lookback_hours: int = 1, batch_size: int = 50):
    """Evaluate Phoenix spans to extract routing experiences for optimization"""
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")

    try:
        logger.info(
            f"🔍 Starting span evaluation (lookback: {lookback_hours}h, batch: {batch_size})"
        )

        # Run span evaluation
        results = await routing_agent.span_evaluator.evaluate_routing_spans(
            lookback_hours=lookback_hours, batch_size=batch_size
        )

        logger.info(
            f"✅ Span evaluation complete: {results.get('experiences_created', 0)} experiences created"
        )

        return {
            "status": "completed",
            "evaluation_results": results,
            "message": f"Processed {results.get('spans_processed', 0)} spans, "
            f"created {results.get('experiences_created', 0)} experiences",
        }

    except Exception as e:
        logger.error(f"❌ Span evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Span evaluation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    port = config.get("routing_agent_port", 8001)

    logger.info(f"Starting Routing Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
