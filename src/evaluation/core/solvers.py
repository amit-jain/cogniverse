"""
Solvers for different evaluation modes.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from inspect_ai.solver import solver
from inspect_ai.tool import use_tools
import phoenix as px

from .tools import video_search_tool

logger = logging.getLogger(__name__)


@solver
def retrieval_solver(
    profiles: List[str],
    strategies: List[str],
    config: Optional[Dict[str, Any]] = None
):
    """
    Solver for running new experiments with actual searches.
    
    Args:
        profiles: Video processing profiles to test
        strategies: Ranking strategies to test
        config: Additional configuration
    """
    async def solve(state, generate):
        """Execute searches for all profile/strategy combinations"""
        query = state.input.get("query", "")
        if not query:
            logger.error("No query found in state.input")
            return state
        
        logger.info(f"Running retrieval for query: {query[:50]}...")
        
        # Store results for each configuration
        state.outputs = {}
        
        for profile in profiles:
            for strategy in strategies:
                config_key = f"{profile}_{strategy}"
                
                try:
                    # Use search tool to execute actual search
                    results = await use_tools(
                        [video_search_tool()],
                        query=query,
                        profile=profile,
                        strategy=strategy,
                        top_k=config.get("top_k", 10) if config else 10
                    )
                    
                    state.outputs[config_key] = {
                        "results": results,
                        "profile": profile,
                        "strategy": strategy,
                        "success": True
                    }
                    
                    logger.debug(f"Retrieved {len(results)} results for {config_key}")
                    
                except Exception as e:
                    logger.error(f"Search failed for {config_key}: {e}")
                    state.outputs[config_key] = {
                        "results": [],
                        "profile": profile,
                        "strategy": strategy,
                        "success": False,
                        "error": str(e)
                    }
        
        return state
    
    return solve


@solver  
def trace_loader_solver(
    trace_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Solver for batch evaluation of existing traces.
    
    Args:
        trace_ids: Specific trace IDs to evaluate
        config: Additional configuration including time range
    """
    async def solve(state, generate):
        """Load results from existing traces"""
        query = state.input.get("query", "")
        
        client = px.Client()
        
        # Build filter for traces
        if trace_ids:
            # Use specific trace IDs
            filter_condition = f"trace_id in {trace_ids}"
            logger.info(f"Loading specific traces: {trace_ids[:5]}...")
        else:
            # Get recent traces if no IDs specified
            start_time = datetime.now() - timedelta(
                hours=config.get("hours_back", 1) if config else 1
            )
            filter_condition = None
            logger.info(f"Loading traces from last {config.get('hours_back', 1) if config else 1} hours")
        
        try:
            # Use Phoenix's working method to get traces
            df = client.get_spans_dataframe(
                filter_condition=filter_condition,
                start_time=start_time if not trace_ids else None,
                root_spans_only=True,  # Only get root spans (full traces)
                limit=config.get("max_traces", 100) if config else 100
            )
            
            if df.empty:
                logger.warning("No traces found matching criteria")
                state.outputs = {"traces": [], "success": False}
                return state
            
            # Find trace matching this query
            matching_trace = None
            for _, row in df.iterrows():
                trace_query = row.get('attributes.input.value', '')
                if trace_query == query:
                    matching_trace = row
                    break
            
            if matching_trace is not None:
                # Extract results from trace
                results = matching_trace.get('attributes.output.value', [])
                profile = matching_trace.get('attributes.metadata.profile', 'unknown')
                strategy = matching_trace.get('attributes.metadata.strategy', 'unknown')
                
                state.outputs = {
                    f"{profile}_{strategy}": {
                        "results": results,
                        "profile": profile,
                        "strategy": strategy,
                        "success": True,
                        "trace_id": matching_trace.get('trace_id')
                    }
                }
                
                logger.info(f"Loaded results from trace {matching_trace.get('trace_id')}")
            else:
                logger.warning(f"No trace found matching query: {query[:50]}...")
                state.outputs = {"traces": [], "success": False}
                
        except Exception as e:
            logger.error(f"Failed to load traces: {e}")
            state.outputs = {"traces": [], "success": False, "error": str(e)}
        
        return state
    
    return solve


@solver
def live_trace_solver(config: Optional[Dict[str, Any]] = None):
    """
    Solver for real-time evaluation of incoming traces.
    
    Args:
        config: Configuration for live monitoring
    """
    async def solve(state, generate):
        """Evaluate traces as they arrive in real-time"""
        # This would connect to a Phoenix trace stream
        # For now, we'll implement a simplified version
        
        logger.info("Live trace evaluation not yet implemented")
        
        # Placeholder implementation
        state.outputs = {
            "live": {
                "results": [],
                "success": False,
                "message": "Live evaluation coming soon"
            }
        }
        
        return state
    
    return solve