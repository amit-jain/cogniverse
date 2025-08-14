# src/agents/query_analysis_tool_v2.py
"""
Enhanced QueryAnalysisTool using the comprehensive routing system.
This replaces the original QueryAnalysisTool with the new tiered architecture.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from google.adk.tools import BaseTool

# Import the new comprehensive routing system
from src.routing import (
    ComprehensiveRouter,
    TieredRouter,
    RoutingConfig,
    AutoTuningOptimizer,
    load_config
)
from src.routing.base import SearchModality, GenerationType

logger = logging.getLogger(__name__)


class QueryAnalysisToolV2(BaseTool):
    """
    Enhanced Query Analysis Tool using the comprehensive routing system.
    Implements the tiered architecture from COMPREHENSIVE_ROUTING.md.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced query analysis tool.
        
        Args:
            config_path: Optional path to routing configuration file
        """
        super().__init__(
            name="QueryAnalyzerV2",
            description="Advanced query analyzer with tiered routing and auto-optimization"
        )
        
        # Load routing configuration
        self.routing_config = load_config(config_path)
        
        # Initialize the tiered router
        self.router = TieredRouter(self.routing_config)
        
        # Initialize optimizer if enabled
        self.optimizer = None
        if self.routing_config.optimization_config.get("enable_auto_optimization", False):
            # Create optimizers for each tier
            self.optimizers = {}
            for tier, strategy in self.router.strategies.items():
                self.optimizers[tier] = AutoTuningOptimizer(
                    strategy=strategy,
                    config=self.routing_config.optimization_config
                )
            logger.info("Auto-optimization enabled for routing strategies")
        
        # Track usage statistics
        self.total_queries = 0
        self.start_time = datetime.now()
    
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze query using the comprehensive routing system.
        
        Args:
            query: The user query to analyze
            context: Optional context (conversation history, etc.)
        
        Returns:
            Analysis results with routing decision
        """
        import time
        start_time = time.time()
        
        self.total_queries += 1
        logger.info(f"\n⏱️ [QueryAnalyzerV2] Starting analysis #{self.total_queries} for: '{query}'")
        
        try:
            # Route the query through the tiered system
            routing_decision = await self.router.route(query, context)
            
            # Convert routing decision to the expected format
            analysis = self._convert_routing_decision(query, routing_decision)
            
            # Track performance for optimization
            if self.optimizers:
                await self._track_for_optimization(query, routing_decision)
            
            # Add performance metrics
            execution_time = time.time() - start_time
            analysis["execution_time_ms"] = execution_time * 1000
            analysis["routing_tier"] = routing_decision.metadata.get("tier", "unknown")
            
            # Log routing decision
            self._log_routing_decision(analysis, execution_time)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(query, str(e))
    
    def _convert_routing_decision(self, query: str, decision) -> Dict[str, Any]:
        """
        Convert RoutingDecision to the expected analysis format.
        
        Args:
            query: Original query
            decision: RoutingDecision from the router
        
        Returns:
            Analysis dictionary in the expected format
        """
        analysis = {
            "original_query": query,
            "needs_video_search": decision.search_modality in [SearchModality.VIDEO, SearchModality.BOTH],
            "needs_text_search": decision.search_modality in [SearchModality.TEXT, SearchModality.BOTH],
            "temporal_info": decision.temporal_info or {},
            "cleaned_query": query.lower().strip(),
            "routing_method": decision.routing_method,
            "confidence_score": decision.confidence_score,
            "generation_type": decision.generation_type.value,
            "search_modality": decision.search_modality.value
        }
        
        # Add entities if detected
        if decision.entities_detected:
            analysis["entities"] = decision.entities_detected
        
        # Add reasoning if available
        if decision.reasoning:
            analysis["reasoning"] = decision.reasoning
        
        return analysis
    
    async def _track_for_optimization(self, query: str, decision):
        """
        Track routing decision for optimization.
        
        Args:
            query: The query
            decision: The routing decision
        """
        # Track in the appropriate optimizer based on tier
        tier = decision.metadata.get("tier")
        if tier and tier in self.optimizers:
            optimizer = self.optimizers[tier]
            optimizer.track_performance(
                query=query,
                predicted=decision,
                actual=None,  # Would be provided by user feedback
                user_feedback=None
            )
    
    def _log_routing_decision(self, analysis: Dict[str, Any], execution_time: float):
        """
        Log the routing decision for monitoring.
        
        Args:
            analysis: The analysis results
            execution_time: Time taken for analysis
        """
        logger.info(f"⏱️ [QueryAnalyzerV2] ANALYSIS COMPLETE in {execution_time:.3f}s")
        logger.info(f"   ├─ Routing Method: {analysis['routing_method']}")
        logger.info(f"   ├─ Routing Tier: {analysis.get('routing_tier', 'unknown')}")
        logger.info(f"   ├─ Search Modality: {analysis['search_modality']}")
        logger.info(f"   ├─ Generation Type: {analysis['generation_type']}")
        logger.info(f"   ├─ Confidence: {analysis['confidence_score']:.2f}")
        logger.info(f"   ├─ Video Search: {analysis['needs_video_search']}")
        logger.info(f"   └─ Text Search: {analysis['needs_text_search']}")
        
        if analysis.get("temporal_info"):
            logger.info(f"   └─ Temporal: {analysis['temporal_info']}")
    
    def _create_fallback_analysis(self, query: str, error: str) -> Dict[str, Any]:
        """
        Create a fallback analysis when routing fails.
        
        Args:
            query: The original query
            error: Error message
        
        Returns:
            Fallback analysis dictionary
        """
        return {
            "original_query": query,
            "needs_video_search": True,
            "needs_text_search": True,
            "temporal_info": {},
            "cleaned_query": query.lower().strip(),
            "routing_method": "fallback_error",
            "confidence_score": 0.0,
            "generation_type": "raw_results",
            "search_modality": "both",
            "error": error
        }
    
    async def provide_feedback(self, query: str, predicted: Dict[str, Any], 
                              actual: Dict[str, Any], user_satisfaction: float):
        """
        Provide feedback for routing optimization.
        
        Args:
            query: The original query
            predicted: The predicted routing (from execute())
            actual: The actual/correct routing
            user_satisfaction: User satisfaction score (0-1)
        """
        if not self.optimizers:
            return
        
        # Convert dictionaries back to RoutingDecision objects
        from src.routing.base import RoutingDecision
        
        predicted_decision = RoutingDecision(
            search_modality=SearchModality(predicted["search_modality"]),
            generation_type=GenerationType(predicted["generation_type"]),
            confidence_score=predicted["confidence_score"],
            routing_method=predicted["routing_method"]
        )
        
        actual_decision = RoutingDecision(
            search_modality=SearchModality(actual["search_modality"]),
            generation_type=GenerationType(actual["generation_type"]),
            confidence_score=1.0,
            routing_method="ground_truth"
        )
        
        # Track feedback in the appropriate optimizer
        tier = predicted.get("routing_tier")
        if tier and tier in self.optimizers:
            optimizer = self.optimizers[tier]
            optimizer.track_performance(
                query=query,
                predicted=predicted_decision,
                actual=actual_decision,
                user_feedback={"satisfaction": user_satisfaction}
            )
    
    async def trigger_optimization(self):
        """
        Manually trigger optimization of routing strategies.
        """
        if not self.optimizers:
            logger.warning("Optimization is not enabled")
            return
        
        logger.info("Triggering manual optimization of all routing strategies...")
        
        # Run optimization for each tier
        tasks = []
        for tier, optimizer in self.optimizers.items():
            logger.info(f"Optimizing {tier.value} tier...")
            tasks.append(optimizer.optimize())
        
        await asyncio.gather(*tasks)
        
        logger.info("Optimization complete for all tiers")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage and performance statistics.
        
        Returns:
            Dictionary with statistics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        stats = {
            "total_queries": self.total_queries,
            "uptime_seconds": uptime,
            "queries_per_minute": (self.total_queries / uptime * 60) if uptime > 0 else 0,
            "tier_statistics": self.router.get_tier_statistics(),
            "performance_report": self.router.get_performance_report()
        }
        
        return stats
    
    def export_metrics(self, filepath: str):
        """
        Export routing metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        self.router.export_metrics(filepath)
        logger.info(f"Metrics exported to {filepath}")


def create_query_analyzer(config_path: Optional[str] = None) -> QueryAnalysisToolV2:
    """
    Factory function to create a query analyzer.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Configured QueryAnalysisToolV2 instance
    """
    return QueryAnalysisToolV2(config_path)


# Example usage integration with the composing agent
async def integrate_with_composing_agent():
    """
    Example of how to integrate the new QueryAnalysisTool with the composing agent.
    """
    from src.agents.composing_agents_main import ComposingAgent
    
    # Create the enhanced query analyzer
    query_analyzer = create_query_analyzer()
    
    # Replace the old QueryAnalysisTool in the composing agent
    # This would be done in the composing agent initialization
    
    # Example query
    query = "Show me videos from last week about machine learning and create a detailed report"
    
    # Analyze the query
    analysis = await query_analyzer.execute(query)
    
    print("Query Analysis:")
    print(f"  Search Modality: {analysis['search_modality']}")
    print(f"  Generation Type: {analysis['generation_type']}")
    print(f"  Confidence: {analysis['confidence_score']:.2%}")
    print(f"  Routing Tier: {analysis.get('routing_tier', 'unknown')}")
    
    # Provide feedback for learning
    actual_routing = {
        "search_modality": "video",
        "generation_type": "detailed_report"
    }
    
    await query_analyzer.provide_feedback(
        query=query,
        predicted=analysis,
        actual=actual_routing,
        user_satisfaction=0.9
    )
    
    # Trigger optimization after collecting enough feedback
    if query_analyzer.total_queries % 100 == 0:
        await query_analyzer.trigger_optimization()
    
    # Export metrics periodically
    if query_analyzer.total_queries % 1000 == 0:
        query_analyzer.export_metrics(f"routing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return analysis