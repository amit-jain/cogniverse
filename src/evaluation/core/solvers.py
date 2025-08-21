"""
Solvers for evaluation modes.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio
import numpy as np

from inspect_ai.solver import Solver, solver
from inspect_ai.model import ModelOutput
import phoenix as px
import pandas as pd

logger = logging.getLogger(__name__)


@solver
def create_retrieval_solver(
    profiles: List[str],
    strategies: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Solver:
    """
    Solver that runs new searches via SearchService.
    
    Args:
        profiles: Video processing profiles to test
        strategies: Ranking strategies to test
        config: Additional configuration
        
    Returns:
        Solver that runs actual searches
    """
    config = config or {}
    
    async def solve(state, generate):
        """Execute searches for all profile/strategy combinations."""
        # Extract query from state.input (can be string or dict)
        if isinstance(state.input, dict):
            query = state.input.get("query", "")
        else:
            query = state.input
        
        if not query:
            logger.error("No query found in state.input")
            return state
        
        query_str = str(query)  # Ensure it's a string
        logger.info(f"Running retrieval for query: {query_str[:50]}...")
        
        # Import here to avoid circular dependencies
        from src.app.search.service import SearchService
        from src.common.config import get_config
        
        main_config = get_config()
        
        # Store results for each configuration
        all_results = {}
        
        for profile in profiles:
            for strategy in strategies:
                config_key = f"{profile}_{strategy}"
                
                try:
                    # Create search service with specified profile
                    search_service = SearchService(main_config, profile)
                    
                    logger.info(f"Searching with {config_key}: {query_str[:30]}...")
                    
                    # Run the actual search
                    search_results = search_service.search(
                        query=query_str,
                        top_k=config.get("top_k", 10),
                        ranking_strategy=strategy
                    )
                    
                    # Convert results to standard format
                    formatted_results = []
                    for i, result in enumerate(search_results):
                        result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                        
                        # Extract video_id
                        video_id = result_dict.get('source_id', '')
                        if not video_id and 'document_id' in result_dict:
                            doc_id = result_dict['document_id']
                            if '_frame_' in doc_id:
                                video_id = doc_id.split('_frame_')[0]
                            else:
                                video_id = doc_id
                        
                        formatted_results.append({
                            "video_id": video_id,
                            "score": float(result_dict.get('score', 1.0 / (i + 1))),
                            "rank": i + 1,
                            "content": result_dict.get('content', '')
                        })
                    
                    all_results[config_key] = {
                        "results": formatted_results,
                        "profile": profile,
                        "strategy": strategy,
                        "success": True,
                        "count": len(formatted_results)
                    }
                    
                    logger.info(f"Retrieved {len(formatted_results)} results for {config_key}")
                    
                except Exception as e:
                    logger.error(f"Search failed for {config_key}: {e}")
                    all_results[config_key] = {
                        "results": [],
                        "profile": profile,
                        "strategy": strategy,
                        "success": False,
                        "error": str(e)
                    }
        
        # Pack all results into structured output for scorers
        from .solver_output import pack_solver_output
        
        # Get Phoenix trace ID if available
        phoenix_trace_id = None
        if hasattr(state, 'trace_id'):
            # Only use if it's a string (not a Mock or other object)
            trace_id = getattr(state, 'trace_id', None)
            if isinstance(trace_id, str):
                phoenix_trace_id = trace_id
        
        # Create model output with packed results
        packed_output = pack_solver_output(
            query=query_str,
            search_results=all_results,
            phoenix_trace_id=phoenix_trace_id,
            metadata={
                "profiles": profiles,
                "strategies": strategies,
                "config": config
            }
        )
        
        state.output = ModelOutput(
            completion=packed_output,
            stop_reason="completed"
        )
        
        # Also keep in metadata for backward compatibility
        state.metadata["search_results"] = all_results
        
        return state
    
    return solve


@solver
def create_batch_solver(
    trace_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Solver:
    """
    Solver that loads existing traces from Phoenix and extracts ground truth.
    
    Args:
        trace_ids: Specific trace IDs to load (None for recent)
        config: Additional configuration including ground_truth_strategy
        
    Returns:
        Solver that loads actual traces with ground truth
    """
    config = config or {}
    
    async def solve(state, generate):
        """Load and evaluate existing traces with ground truth extraction."""
        phoenix_client = px.Client()
        
        # Get ground truth strategy
        from .ground_truth import get_ground_truth_strategy
        ground_truth_strategy = get_ground_truth_strategy(config)
        
        # Get backend if needed for ground truth extraction
        backend = None
        if config.get("use_backend_for_ground_truth", False):
            try:
                from src.app.search.service import SearchService
                from src.common.config import get_config
                main_config = get_config()
                search_service = SearchService(main_config)
                backend = search_service.backend
            except Exception as e:
                logger.warning(f"Could not initialize backend for ground truth: {e}")
        
        # Get traces
        if trace_ids:
            logger.info(f"Loading {len(trace_ids)} specific traces")
            # Phoenix doesn't have get_traces_by_ids, use spans instead
            df = phoenix_client.get_spans_dataframe(
                filter_condition=f"trace_id IN {trace_ids}"
            )
        else:
            # Get recent traces
            hours_back = config.get("hours_back", 24)
            limit = config.get("limit", 100)
            
            logger.info(f"Loading traces from last {hours_back} hours (limit: {limit})")
            
            start_time = datetime.now() - timedelta(hours=hours_back)
            df = phoenix_client.get_spans_dataframe(
                start_time=start_time,
                limit=limit
            )
        
        if df.empty:
            logger.warning("No traces found")
            state.output = ModelOutput(
                completion="No traces found",
                stop_reason="no_data"
            )
            return state
        
        # Extract trace data and ground truth
        traces = []
        for _, row in df.iterrows():
            trace_data = {
                "trace_id": row.get("trace_id"),
                "query": row.get("attributes.input.value", ""),
                "results": row.get("attributes.output.value", []),
                "profile": row.get("attributes.metadata.profile", "unknown"),
                "strategy": row.get("attributes.metadata.strategy", "unknown"),
                "timestamp": row.get("timestamp"),
                "duration_ms": row.get("duration_ms", 0),
                "metadata": row.get("attributes.metadata", {})
            }
            
            # Extract ground truth for this trace
            ground_truth_result = await ground_truth_strategy.extract_ground_truth(
                trace_data, backend
            )
            
            # Use expected_items as the generic field, falling back to expected_videos for compatibility
            trace_data["ground_truth"] = ground_truth_result.get("expected_items", 
                                                                 ground_truth_result.get("expected_videos", []))
            trace_data["ground_truth_confidence"] = ground_truth_result["confidence"]
            trace_data["ground_truth_source"] = ground_truth_result["source"]
            
            traces.append(trace_data)
        
        logger.info(f"Loaded {len(traces)} traces with ground truth extraction")
        
        # Apply reranking if configured
        reranking_strategy = config.get("reranking_strategy")
        if reranking_strategy and reranking_strategy != "none":
            from .reranking import apply_reranking_to_traces
            traces = await apply_reranking_to_traces(
                traces, 
                reranking_strategy,
                config.get("reranking_config", {})
            )
            logger.info(f"Applied {reranking_strategy} reranking strategy")
        
        # Store traces in state for evaluation
        state.output = ModelOutput(
            completion=f"Loaded {len(traces)} traces with ground truth",
            stop_reason="completed"
        )
        state.metadata["loaded_traces"] = traces
        state.metadata["ground_truth_stats"] = {
            "total_traces": len(traces),
            "traces_with_ground_truth": sum(1 for t in traces if t.get("ground_truth")),
            "average_confidence": np.mean([t.get("ground_truth_confidence", 0) for t in traces])
        }
        if reranking_strategy:
            state.metadata["reranking_strategy"] = reranking_strategy
        
        return state
    
    return solve


@solver
def create_live_solver(
    config: Optional[Dict[str, Any]] = None
) -> Solver:
    """
    Solver for live/continuous evaluation.
    
    Args:
        config: Additional configuration
        
    Returns:
        Solver that monitors live traces
    """
    config = config or {}
    
    async def solve(state, generate):
        """Monitor and evaluate live traces."""
        phoenix_client = px.Client()
        
        poll_interval = config.get("poll_interval", 10)
        max_iterations = config.get("max_iterations", 10)
        
        logger.info(f"Starting live monitoring (interval: {poll_interval}s)")
        
        all_traces = []
        last_check = datetime.now()
        
        for iteration in range(max_iterations):
            # Get new traces since last check
            df = phoenix_client.get_spans_dataframe(
                start_time=last_check,
                limit=100
            )
            
            if not df.empty:
                logger.info(f"Found {len(df)} new traces")
                
                for _, row in df.iterrows():
                    trace_data = {
                        "trace_id": row.get("trace_id"),
                        "query": row.get("attributes.input.value", ""),
                        "results": row.get("attributes.output.value", []),
                        "timestamp": row.get("timestamp")
                    }
                    all_traces.append(trace_data)
            
            last_check = datetime.now()
            
            # Wait before next poll
            if iteration < max_iterations - 1:
                await asyncio.sleep(poll_interval)
        
        logger.info(f"Live monitoring complete. Collected {len(all_traces)} traces")
        
        state.output = ModelOutput(
            completion=f"Monitored {len(all_traces)} live traces",
            stop_reason="completed"
        )
        state.metadata["live_traces"] = all_traces
        
        return state
    
    return solve