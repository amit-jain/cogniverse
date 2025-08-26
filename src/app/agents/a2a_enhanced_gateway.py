"""
A2A Enhanced Gateway - Integration layer for DSPy 3.0 routing system

This gateway provides seamless integration between the enhanced DSPy 3.0 routing system
and the existing A2A protocol infrastructure, maintaining backward compatibility while
adding advanced routing, orchestration, and relationship-aware processing.

Features:
- Backward-compatible A2A protocol support
- Enhanced routing with relationship extraction and query enhancement
- Multi-agent orchestration for complex queries
- Intelligent fallback to original routing system
- Performance monitoring and telemetry
- Graceful degradation strategies
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Enhanced routing imports
from src.app.agents.enhanced_routing_agent import (
    EnhancedRoutingAgent,
    EnhancedRoutingConfig,
)
from src.app.agents.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
)

# Original A2A imports (for fallback compatibility)

# Existing agent imports (for fallback routing)
try:
    from src.app.agents.routing_agent import RoutingAgent as OriginalRoutingAgent
except ImportError:
    OriginalRoutingAgent = None


class A2AQueryRequest(BaseModel):
    """Standard A2A query request format"""

    query: str = Field(..., description="User query")
    context: Optional[str] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    routing_options: Optional[Dict[str, Any]] = Field(
        None, description="Routing configuration"
    )


class A2AQueryResponse(BaseModel):
    """Enhanced A2A query response format"""

    # Standard A2A fields
    agent: str = Field(..., description="Recommended agent")
    confidence: float = Field(..., description="Routing confidence")
    reasoning: str = Field(..., description="Routing reasoning")

    # Enhanced fields
    enhanced_query: Optional[str] = Field(
        None, description="Enhanced query with relationship context"
    )
    fallback_agents: List[str] = Field(
        default_factory=list, description="Fallback agent recommendations"
    )

    # Orchestration fields
    needs_orchestration: bool = Field(
        False, description="Whether orchestration is recommended"
    )
    workflow_id: Optional[str] = Field(
        None, description="Workflow identifier if orchestrated"
    )

    # Analysis fields
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted relationships"
    )

    # Metadata
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    routing_method: str = Field(
        ..., description="Routing method used (enhanced/fallback)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class OrchestrationRequest(BaseModel):
    """Request for multi-agent orchestration"""

    query: str = Field(..., description="Complex user query")
    context: Optional[str] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    force_orchestration: bool = Field(
        False, description="Force orchestration even for simple queries"
    )


class OrchestrationResponse(BaseModel):
    """Response from multi-agent orchestration"""

    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Workflow status")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Final orchestrated result"
    )
    execution_summary: Optional[Dict[str, Any]] = Field(
        None, description="Execution summary"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow metadata"
    )


class A2AEnhancedGateway:
    """
    A2A Enhanced Gateway - Integration layer for DSPy 3.0 routing system

    Provides seamless integration between enhanced routing/orchestration and
    existing A2A protocol infrastructure with backward compatibility.
    """

    def __init__(
        self,
        enhanced_routing_config: Optional[EnhancedRoutingConfig] = None,
        enable_orchestration: bool = True,
        enable_fallback: bool = True,
        port: int = 8000,
    ):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.enable_orchestration = enable_orchestration
        self.enable_fallback = enable_fallback
        self.port = port

        # Initialize enhanced routing system
        self._initialize_enhanced_system(enhanced_routing_config)

        # Initialize fallback system if enabled
        if enable_fallback:
            self._initialize_fallback_system()

        # Initialize FastAPI app with A2A endpoints
        self.app = self._create_fastapi_app()

        # Statistics tracking
        self.gateway_stats = {
            "total_requests": 0,
            "enhanced_routing_requests": 0,
            "orchestration_requests": 0,
            "fallback_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
        }

    def _initialize_enhanced_system(
        self, config: Optional[EnhancedRoutingConfig]
    ) -> None:
        """Initialize enhanced routing and orchestration system"""
        try:
            # Initialize enhanced routing agent
            self.enhanced_router = EnhancedRoutingAgent(
                config=config or EnhancedRoutingConfig(),
                port=self.port + 1,  # Use different port to avoid conflicts
                enable_telemetry=True,
            )

            # Initialize multi-agent orchestrator
            if self.enable_orchestration:
                self.orchestrator = MultiAgentOrchestrator(
                    routing_agent=self.enhanced_router
                )

            self.enhanced_system_available = True
            self.logger.info("Enhanced routing system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced system: {e}")
            self.enhanced_system_available = False
            if not self.enable_fallback:
                raise RuntimeError("Enhanced system failed and fallback disabled")

    def _initialize_fallback_system(self) -> None:
        """Initialize fallback routing system for backward compatibility"""
        try:
            # Initialize original routing agent if available
            if OriginalRoutingAgent:
                self.fallback_router = OriginalRoutingAgent()
                self.fallback_system_available = True
                self.logger.info("Fallback routing system initialized")
            else:
                self.fallback_system_available = False
                self.logger.warning("Original routing agent not available for fallback")

        except Exception as e:
            self.logger.error(f"Failed to initialize fallback system: {e}")
            self.fallback_system_available = False

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with A2A protocol endpoints"""
        app = FastAPI(
            title="A2A Enhanced Gateway",
            description="Enhanced A2A routing with DSPy 3.0, relationship extraction, and orchestration",
            version="1.0.0",
        )

        # Standard A2A routing endpoint (backward compatible)
        @app.post("/route", response_model=A2AQueryResponse)
        async def route_query(request: A2AQueryRequest) -> A2AQueryResponse:
            return await self._handle_routing_request(request)

        # Enhanced orchestration endpoint
        @app.post("/orchestrate", response_model=OrchestrationResponse)
        async def orchestrate_query(
            request: OrchestrationRequest,
        ) -> OrchestrationResponse:
            return await self._handle_orchestration_request(request)

        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "enhanced_system": self.enhanced_system_available,
                "fallback_system": self.fallback_system_available,
                "orchestration_enabled": self.enable_orchestration,
                "timestamp": datetime.now().isoformat(),
            }

        # Statistics endpoint
        @app.get("/stats")
        async def get_statistics():
            gateway_stats = self.gateway_stats.copy()

            # Add enhanced routing stats if available
            if self.enhanced_system_available:
                gateway_stats["enhanced_routing_stats"] = (
                    self.enhanced_router.get_routing_statistics()
                )

            # Add orchestration stats if available
            if self.enable_orchestration and hasattr(self, "orchestrator"):
                gateway_stats["orchestration_stats"] = (
                    self.orchestrator.get_orchestration_statistics()
                )

            return gateway_stats

        # Workflow status endpoint
        @app.get("/workflow/{workflow_id}")
        async def get_workflow_status(workflow_id: str):
            if not self.enable_orchestration or not hasattr(self, "orchestrator"):
                raise HTTPException(status_code=404, detail="Orchestration not enabled")

            status = self.orchestrator.get_workflow_status(workflow_id)
            if not status:
                raise HTTPException(status_code=404, detail="Workflow not found")

            return status

        # Cancel workflow endpoint
        @app.post("/workflow/{workflow_id}/cancel")
        async def cancel_workflow(workflow_id: str):
            if not self.enable_orchestration or not hasattr(self, "orchestrator"):
                raise HTTPException(status_code=404, detail="Orchestration not enabled")

            success = await self.orchestrator.cancel_workflow(workflow_id)
            if not success:
                raise HTTPException(
                    status_code=404, detail="Workflow not found or already completed"
                )

            return {"status": "cancelled", "workflow_id": workflow_id}

        return app

    async def _handle_routing_request(
        self, request: A2AQueryRequest
    ) -> A2AQueryResponse:
        """Handle standard A2A routing request with enhancement"""
        start_time = datetime.now()
        self.gateway_stats["total_requests"] += 1

        try:
            # Try enhanced routing first
            if self.enhanced_system_available:
                return await self._enhanced_routing(request, start_time)

            # Fallback to original routing system
            elif self.fallback_system_available:
                return await self._fallback_routing(request, start_time)

            else:
                # Ultimate fallback
                return self._create_emergency_response(
                    request, start_time, "No routing systems available"
                )

        except Exception as e:
            self.logger.error(f"Routing request failed: {e}")
            self.gateway_stats["failed_requests"] += 1

            # Try fallback if enhanced failed
            if self.enhanced_system_available and self.fallback_system_available:
                try:
                    return await self._fallback_routing(request, start_time)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback routing also failed: {fallback_error}")

            return self._create_emergency_response(request, start_time, str(e))

    async def _enhanced_routing(
        self, request: A2AQueryRequest, start_time: datetime
    ) -> A2AQueryResponse:
        """Process request using enhanced DSPy 3.0 routing system"""
        self.gateway_stats["enhanced_routing_requests"] += 1

        # Extract routing options
        routing_options = request.routing_options or {}
        require_orchestration = routing_options.get("require_orchestration")

        # Perform enhanced routing
        routing_decision = await self.enhanced_router.route_query(
            query=request.query,
            context=request.context,
            user_id=request.user_id,
            require_orchestration=require_orchestration,
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Update average response time
        self._update_response_time_stats(processing_time)

        # Build enhanced response
        response = A2AQueryResponse(
            agent=routing_decision.recommended_agent,
            confidence=routing_decision.confidence,
            reasoning=routing_decision.reasoning,
            enhanced_query=routing_decision.enhanced_query,
            fallback_agents=routing_decision.fallback_agents,
            needs_orchestration=routing_decision.routing_metadata.get(
                "needs_orchestration", False
            ),
            entities=routing_decision.extracted_entities,
            relationships=routing_decision.extracted_relationships,
            processing_time_ms=processing_time,
            routing_method="enhanced_dspy",
            metadata={
                **routing_decision.routing_metadata,
                "gateway_version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
            },
        )

        self.logger.info(
            f"Enhanced routing: {request.query[:50]}... → {response.agent} "
            f"(confidence: {response.confidence:.3f}, time: {processing_time:.1f}ms)"
        )

        return response

    async def _fallback_routing(
        self, request: A2AQueryRequest, start_time: datetime
    ) -> A2AQueryResponse:
        """Process request using fallback routing system"""
        self.gateway_stats["fallback_requests"] += 1

        try:
            # Use original routing agent
            if hasattr(self.fallback_router, "analyze_and_route"):
                # Async routing method
                result = await self.fallback_router.analyze_and_route(
                    request.query, request.context
                )
            else:
                # Sync routing method (wrap in async)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.fallback_router.route, request.query, request.context
                )

            # Extract fallback result
            agent = result.get("agent", "video_search_agent")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "Fallback routing decision")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_response_time_stats(processing_time)

            response = A2AQueryResponse(
                agent=agent,
                confidence=confidence,
                reasoning=reasoning,
                enhanced_query=request.query,  # No enhancement in fallback
                fallback_agents=[],
                needs_orchestration=False,
                entities=[],
                relationships=[],
                processing_time_ms=processing_time,
                routing_method="fallback_original",
                metadata={
                    "fallback_used": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            self.logger.info(
                f"Fallback routing: {request.query[:50]}... → {response.agent} "
                f"(confidence: {response.confidence:.3f}, time: {processing_time:.1f}ms)"
            )

            return response

        except Exception as e:
            self.logger.error(f"Fallback routing failed: {e}")
            return self._create_emergency_response(request, start_time, str(e))

    def _create_emergency_response(
        self, request: A2AQueryRequest, start_time: datetime, error: str
    ) -> A2AQueryResponse:
        """Create emergency response when all routing systems fail"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Simple heuristic routing as last resort
        query_lower = request.query.lower()

        if any(
            word in query_lower
            for word in ["video", "visual", "image", "watch", "show"]
        ):
            agent = "video_search_agent"
            reasoning = "Emergency routing based on video keywords"
        elif any(
            word in query_lower
            for word in ["summarize", "summary", "brief", "overview"]
        ):
            agent = "summarizer_agent"
            reasoning = "Emergency routing based on summary keywords"
        elif any(
            word in query_lower
            for word in ["report", "analysis", "detailed", "comprehensive"]
        ):
            agent = "detailed_report_agent"
            reasoning = "Emergency routing based on analysis keywords"
        else:
            agent = "video_search_agent"  # Default fallback
            reasoning = "Emergency routing with default agent"

        return A2AQueryResponse(
            agent=agent,
            confidence=0.2,  # Low confidence for emergency routing
            reasoning=f"{reasoning} (Emergency: {error})",
            enhanced_query=request.query,
            fallback_agents=[],
            needs_orchestration=False,
            entities=[],
            relationships=[],
            processing_time_ms=processing_time,
            routing_method="emergency_fallback",
            metadata={
                "emergency_routing": True,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _handle_orchestration_request(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Handle multi-agent orchestration request"""
        if not self.enable_orchestration or not hasattr(self, "orchestrator"):
            raise HTTPException(status_code=501, detail="Orchestration not enabled")

        self.gateway_stats["orchestration_requests"] += 1

        try:
            # Process complex query with orchestration
            result = await self.orchestrator.process_complex_query(
                query=request.query,
                context=request.context,
                user_id=request.user_id,
                preferences=request.preferences,
            )

            # Build orchestration response
            response = OrchestrationResponse(
                workflow_id=result["workflow_id"],
                status=result["status"],
                result=result.get("result"),
                execution_summary=result.get("execution_summary"),
                error=result.get("error"),
                metadata=result.get("metadata", {}),
            )

            self.logger.info(
                f"Orchestration completed: {request.query[:50]}... → "
                f"{response.status} (workflow: {response.workflow_id})"
            )

            return response

        except Exception as e:
            self.logger.error(f"Orchestration request failed: {e}")
            self.gateway_stats["failed_requests"] += 1
            raise HTTPException(status_code=500, detail=str(e))

    def _update_response_time_stats(self, processing_time: float) -> None:
        """Update average response time statistics"""
        current_avg = self.gateway_stats["average_response_time"]
        total_requests = self.gateway_stats["total_requests"]

        # Calculate new average
        self.gateway_stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
            if total_requests > 0
            else processing_time
        )

    def start_server(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """Start the A2A Enhanced Gateway server"""
        import uvicorn

        server_port = port or self.port

        self.logger.info(f"Starting A2A Enhanced Gateway on {host}:{server_port}")
        self.logger.info(
            f"Enhanced routing: {'enabled' if self.enhanced_system_available else 'disabled'}"
        )
        self.logger.info(
            f"Orchestration: {'enabled' if self.enable_orchestration else 'disabled'}"
        )
        self.logger.info(
            f"Fallback routing: {'enabled' if self.enable_fallback else 'disabled'}"
        )

        uvicorn.run(
            self.app, host=host, port=server_port, log_level="info", access_log=True
        )


def create_a2a_enhanced_gateway(
    enhanced_routing_config: Optional[EnhancedRoutingConfig] = None,
    enable_orchestration: bool = True,
    enable_fallback: bool = True,
    port: int = 8000,
) -> A2AEnhancedGateway:
    """Factory function to create A2A Enhanced Gateway"""
    return A2AEnhancedGateway(
        enhanced_routing_config=enhanced_routing_config,
        enable_orchestration=enable_orchestration,
        enable_fallback=enable_fallback,
        port=port,
    )


# Example usage and configuration
if __name__ == "__main__":
    import asyncio

    async def test_gateway():
        """Test the A2A Enhanced Gateway"""
        # Create gateway with enhanced routing and orchestration
        gateway = create_a2a_enhanced_gateway(
            enhanced_routing_config=EnhancedRoutingConfig(
                model_name="smollm3:3b",
                base_url="http://localhost:11434/v1",
                enable_relationship_extraction=True,
                enable_query_enhancement=True,
            ),
            enable_orchestration=True,
            enable_fallback=True,
            port=8000,
        )

        # Test routing requests
        test_requests = [
            A2AQueryRequest(
                query="Show me videos of robots playing soccer",
                context="Sports analysis research",
            ),
            A2AQueryRequest(
                query="Find videos of autonomous vehicles, analyze the navigation techniques, and summarize key findings",
                context="AI research project",
            ),
            A2AQueryRequest(
                query="Generate a comprehensive report on renewable energy trends",
                context="Environmental analysis",
            ),
        ]

        print("Testing A2A Enhanced Gateway:")
        print("=" * 50)

        for i, request in enumerate(test_requests, 1):
            print(f"\nTest {i}: {request.query}")

            try:
                # Test routing
                response = await gateway._handle_routing_request(request)
                print(f"  Agent: {response.agent}")
                print(f"  Confidence: {response.confidence:.3f}")
                print(f"  Method: {response.routing_method}")
                print(f"  Orchestration needed: {response.needs_orchestration}")
                print(f"  Processing time: {response.processing_time_ms:.1f}ms")

                # Test orchestration for complex queries
                if response.needs_orchestration:
                    orchestration_request = OrchestrationRequest(
                        query=request.query,
                        context=request.context,
                        user_id=request.user_id,
                    )

                    orch_response = await gateway._handle_orchestration_request(
                        orchestration_request
                    )
                    print(
                        f"  Orchestration: {orch_response.status} (workflow: {orch_response.workflow_id})"
                    )

            except Exception as e:
                print(f"  Error: {e}")

        # Print statistics
        print("\nGateway Statistics:")
        stats = gateway.gateway_stats
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # Run test
    asyncio.run(test_gateway())

    # Uncomment to start server
    # gateway = create_a2a_enhanced_gateway()
    # gateway.start_server()
