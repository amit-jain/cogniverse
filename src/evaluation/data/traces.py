"""
Trace management for batch evaluation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

import pandas as pd
from .storage import PhoenixStorage

logger = logging.getLogger(__name__)


class TraceManager:
    """
    Manages trace fetching and processing for batch evaluation.
    """
    
    def __init__(self, storage: Optional[PhoenixStorage] = None):
        """
        Initialize trace manager.
        
        Args:
            storage: Phoenix storage instance
        """
        self.storage = storage or PhoenixStorage()
    
    def get_recent_traces(
        self,
        hours_back: int = 1,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get recent traces from Phoenix.
        
        Args:
            hours_back: Number of hours to look back
            limit: Maximum number of traces
            
        Returns:
            DataFrame with trace data
        """
        start_time = datetime.now() - timedelta(hours=hours_back)
        
        logger.info(f"Fetching traces from last {hours_back} hours")
        
        df = self.storage.get_traces_for_evaluation(
            start_time=start_time,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(df)} traces")
        return df
    
    def get_traces_by_ids(
        self,
        trace_ids: List[str]
    ) -> pd.DataFrame:
        """
        Get specific traces by ID.
        
        Args:
            trace_ids: List of trace IDs
            
        Returns:
            DataFrame with trace data
        """
        logger.info(f"Fetching {len(trace_ids)} specific traces")
        
        # Phoenix doesn't support batch ID fetching well
        # We might need to fetch them individually
        all_traces = []
        
        for trace_id in trace_ids:
            df = self.storage.get_traces_for_evaluation(
                trace_ids=[trace_id],
                limit=1
            )
            if not df.empty:
                all_traces.append(df)
        
        if all_traces:
            result_df = pd.concat(all_traces, ignore_index=True)
            logger.info(f"Retrieved {len(result_df)} traces")
            return result_df
        else:
            logger.warning("No traces found")
            return pd.DataFrame()
    
    def extract_trace_data(
        self,
        trace_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant data from trace DataFrame.
        
        Args:
            trace_df: DataFrame with trace data
            
        Returns:
            List of trace data dictionaries
        """
        trace_data = []
        
        for _, row in trace_df.iterrows():
            try:
                # Extract key fields from trace
                data = {
                    "trace_id": row.get("trace_id", ""),
                    "query": row.get("attributes.input.value", ""),
                    "results": row.get("attributes.output.value", []),
                    "profile": row.get("attributes.metadata.profile", "unknown"),
                    "strategy": row.get("attributes.metadata.strategy", "unknown"),
                    "timestamp": row.get("timestamp", ""),
                    "duration_ms": row.get("duration_ms", 0)
                }
                
                # Parse results if they're in string format
                if isinstance(data["results"], str):
                    try:
                        import json
                        data["results"] = json.loads(data["results"])
                    except:
                        data["results"] = []
                
                trace_data.append(data)
                
            except Exception as e:
                logger.error(f"Failed to extract data from trace: {e}")
                continue
        
        logger.info(f"Extracted data from {len(trace_data)} traces")
        return trace_data
    
    def filter_traces_by_query(
        self,
        traces: List[Dict[str, Any]],
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find trace matching a specific query.
        
        Args:
            traces: List of trace data
            query: Query to match
            
        Returns:
            Matching trace data or None
        """
        for trace in traces:
            if trace.get("query") == query:
                return trace
        return None
    
    def group_traces_by_config(
        self,
        traces: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group traces by profile and strategy.
        
        Args:
            traces: List of trace data
            
        Returns:
            Dictionary grouped by configuration
        """
        grouped = {}
        
        for trace in traces:
            profile = trace.get("profile", "unknown")
            strategy = trace.get("strategy", "unknown")
            config_key = f"{profile}_{strategy}"
            
            if config_key not in grouped:
                grouped[config_key] = []
            
            grouped[config_key].append(trace)
        
        logger.info(f"Grouped traces into {len(grouped)} configurations")
        return grouped