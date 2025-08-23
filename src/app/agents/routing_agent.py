"""
Request-facing Routing Agent implementation.
Main entry point for all user queries that routes to appropriate downstream agents.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

from src.app.routing.router import ComprehensiveRouter, RouterConfig
from src.app.routing.config import RoutingConfig
from src.common.config import get_config
from src.tools.a2a_utils import A2AMessage, DataPart, TextPart
from src.app.agents.dspy_integration_mixin import DSPyRoutingMixin

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Routing Agent",
    description="Request-facing routing agent that coordinates multi-agent workflows",
    version="1.0.0",
)


class RoutingTask(BaseModel):
    """Task structure for A2A communication"""
    id: str
    messages: List[A2AMessage]


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
        
        # Agent registry - maps agent types to their endpoints
        self.agent_registry = {
            "video_search": self.system_config.get("video_agent_url"),
            "text_search": self.system_config.get("text_agent_url"), 
            "summarizer": None,  # Will be set when implemented
            "detailed_report": None,  # Will be set when implemented
        }
        
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
                "enable_fallback": True,
                "fast_path_confidence_threshold": 0.7,
                "slow_path_confidence_threshold": 0.6,
            },
            "gliner_config": {
                "model": "urchade/gliner_large-v2.1",
                "threshold": 0.3,
                "labels": [
                    "video_content", "visual_content", "text_information", 
                    "document_content", "summary_request", "detailed_analysis",
                    "raw_results"
                ]
            },
            "llm_config": {
                "provider": "local",
                "model": "gemma2:2b",
                "endpoint": "http://localhost:11434",
                "temperature": 0.1
            },
            "cache_config": {
                "enable_caching": True,
                "cache_ttl_seconds": 300
            },
            "monitoring_config": {
                "enable_metrics": True,
                "metrics_batch_size": 100
            },
            "optimization_config": {
                "enable_auto_optimization": False,  # Disable for initial implementation
                "optimization_threshold": 1000
            }
        }
        
        return RoutingConfig(**routing_config_dict)
    
    def _validate_agent_registry(self):
        """Validate that required agents are available"""
        if not self.agent_registry["video_search"]:
            raise ValueError("video_agent_url not configured in system config")
        
        # Log available agents
        available_agents = [k for k, v in self.agent_registry.items() if v is not None]
        logger.info(f"Available downstream agents: {available_agents}")
    
    async def analyze_and_route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        
        # Step 1: Get routing decision from comprehensive router
        routing_decision = await self.router.route(query, context)
        
        # Step 2: Determine output type and agent workflow
        workflow_plan = self._determine_workflow(query, routing_decision)
        
        execution_time = time.time() - start_time
        logger.info(f"Query analysis completed in {execution_time:.3f}s")
        
        return {
            "query": query,
            "routing_decision": routing_decision.to_dict(),
            "workflow_type": workflow_plan["type"],
            "agents_to_call": workflow_plan["agents"],
            "execution_plan": workflow_plan["steps"],
            "confidence": routing_decision.confidence_score,
            "routing_method": routing_decision.routing_method,
            "execution_time": execution_time
        }
    
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
        if any(word in query_lower for word in ["summarize", "summary", "brief", "overview"]):
            output_type = "summary"
        elif any(word in query_lower for word in ["detailed", "analyze", "analysis", "report", "explain", "why", "how"]):
            output_type = "detailed_report"
        else:
            output_type = "raw_results"
        
        # Build workflow based on search modality and output type
        workflow_steps = []
        agents_needed = []
        
        # Step 1: Always start with search
        if routing_decision.search_modality.value in ["video", "both"]:
            workflow_steps.append({
                "step": 1,
                "agent": "video_search",
                "action": "search",
                "parameters": {
                    "query": query,
                    "top_k": 10
                }
            })
            agents_needed.append("video_search")
        
        if routing_decision.search_modality.value in ["text", "both"]:
            if self.agent_registry["text_search"]:
                workflow_steps.append({
                    "step": len(workflow_steps) + 1,
                    "agent": "text_search", 
                    "action": "search",
                    "parameters": {
                        "query": query,
                        "top_k": 10
                    }
                })
                agents_needed.append("text_search")
            else:
                logger.warning("Text search requested but text_search agent not available")
        
        # Step 2: Post-processing based on output type
        if output_type == "summary":
            if self.agent_registry["summarizer"]:
                workflow_steps.append({
                    "step": len(workflow_steps) + 1,
                    "agent": "summarizer",
                    "action": "summarize",
                    "parameters": {
                        "query": query,
                        "results_from_previous_steps": True
                    }
                })
                agents_needed.append("summarizer")
            else:
                logger.warning("Summary requested but summarizer agent not available")
                
        elif output_type == "detailed_report":
            if self.agent_registry["detailed_report"]:
                workflow_steps.append({
                    "step": len(workflow_steps) + 1,
                    "agent": "detailed_report",
                    "action": "analyze", 
                    "parameters": {
                        "query": query,
                        "results_from_previous_steps": True,
                        "enable_think_phase": True
                    }
                })
                agents_needed.append("detailed_report")
            else:
                logger.warning("Detailed report requested but detailed_report agent not available")
        
        return {
            "type": output_type,
            "agents": agents_needed,
            "steps": workflow_steps
        }


# Global routing agent instance
routing_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize routing agent on startup"""
    global routing_agent
    
    try:
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
        "available_downstream_agents": [k for k, v in routing_agent.agent_registry.items() if v is not None],
        "routing_config": {
            "mode": routing_agent.routing_config.routing_mode if hasattr(routing_agent.routing_config, 'routing_mode') else "tiered",
            "tiers_enabled": len([tier for tier in routing_agent.router.strategies.keys()])
        }
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
async def process_task(task: RoutingTask):
    """
    Process A2A task - analyze query and return routing decision.
    
    Args:
        task: A2A task with messages
    """
    if not routing_agent:
        raise HTTPException(status_code=503, detail="Routing agent not initialized")
    
    if not task.messages:
        raise HTTPException(status_code=400, detail="No messages in task")
    
    # Extract query from last message
    last_message = task.messages[-1]
    query = None
    context = {}
    
    for part in last_message.parts:
        if isinstance(part, DataPart):
            data = part.data
            if isinstance(data, dict):
                query = data.get("query")
                context.update({k: v for k, v in data.items() if k != "query"})
        elif isinstance(part, TextPart):
            query = part.text
    
    if not query:
        raise HTTPException(status_code=400, detail="No query found in message")
    
    try:
        analysis = await routing_agent.analyze_and_route(query, context if context else None)
        
        return RoutingDecisionResponse(
            task_id=task.id,
            routing_decision=analysis["routing_decision"],
            agents_to_call=analysis["agents_to_call"],
            workflow_type=analysis["workflow_type"],
            execution_plan=analysis["execution_plan"],
            status="completed"
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


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    port = config.get("routing_agent_port", 8001)
    
    logger.info(f"Starting Routing Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)