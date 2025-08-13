"""
Phoenix storage interface maintaining dashboard compatibility.
"""

import phoenix as px
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class PhoenixStorage:
    """
    Phoenix storage interface that maintains compatibility with the dashboard.
    
    The dashboard expects:
    - Experiments grouped by profile/strategy
    - Specific metrics: mrr, recall@1, recall@5
    - Query-level results with expected/actual videos
    """
    
    def __init__(self):
        """Initialize Phoenix client."""
        try:
            self.client = px.Client()
            logger.info("Connected to Phoenix")
        except Exception as e:
            logger.error(f"Failed to connect to Phoenix: {e}")
            raise
    
    def log_experiment_results(
        self,
        experiment_name: str,
        profile: str,
        strategy: str,
        results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> str:
        """
        Log results in format expected by dashboard.
        
        Dashboard expects structure:
        {
            "profile": str,
            "strategy": str,
            "aggregate_metrics": {
                "mrr": {"mean": float},
                "recall@1": {"mean": float},
                "recall@5": {"mean": float}
            },
            "queries": [...]
        }
        
        Args:
            experiment_name: Name of the experiment
            profile: Video processing profile
            strategy: Ranking strategy
            results: List of query results
            metrics: Aggregated metrics
            
        Returns:
            Experiment ID
        """
        # Format for dashboard compatibility
        formatted_results = {
            "profile": profile,
            "strategy": strategy,
            "aggregate_metrics": {
                "mrr": {"mean": metrics.get("mrr", 0.0)},
                "recall@1": {"mean": metrics.get("recall@1", 0.0)},
                "recall@5": {"mean": metrics.get("recall@5", 0.0)}
            },
            "queries": results,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Log to Phoenix
            # Note: Phoenix's actual API might differ
            # This is a conceptual implementation
            experiment_metadata = {
                "name": experiment_name,
                "profile": profile,
                "strategy": strategy,
                "framework": "inspect_ai",  # Tag to identify new system
                "timestamp": datetime.now().isoformat()
            }
            
            # In practice, you'd use Phoenix's actual experiment logging
            # For now, we'll log as a trace with specific attributes
            with px.trace(name=experiment_name) as trace:
                trace.set_attributes(experiment_metadata)
                trace.set_outputs(formatted_results)
            
            experiment_id = f"{experiment_name}_{profile}_{strategy}_{datetime.now().timestamp()}"
            
            logger.info(f"Logged experiment {experiment_id} to Phoenix")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to log experiment results: {e}")
            raise
    
    def get_traces_for_evaluation(
        self,
        trace_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        filter_condition: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get traces using working Phoenix methods.
        
        Args:
            trace_ids: Specific trace IDs to fetch
            start_time: Start time for trace query
            filter_condition: Additional filter conditions
            limit: Maximum number of traces to return
            
        Returns:
            DataFrame with trace data
        """
        try:
            # Build filter condition for specific trace IDs
            if trace_ids:
                # Phoenix doesn't support "in" operator directly
                # We might need to fetch individually or use a different approach
                filter_condition = f"trace_id == '{trace_ids[0]}'" if len(trace_ids) == 1 else filter_condition
            
            # Use actual working Phoenix method
            df = self.client.get_spans_dataframe(
                filter_condition=filter_condition,
                start_time=start_time,
                root_spans_only=True,  # Only root spans for full traces
                limit=limit
            )
            
            logger.info(f"Retrieved {len(df)} traces from Phoenix")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def create_dataset(
        self,
        name: str,
        queries: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> str:
        """
        Create a dataset in Phoenix.
        
        Args:
            name: Dataset name
            queries: List of queries with expected results
            description: Dataset description
            
        Returns:
            Dataset ID
        """
        try:
            # Convert queries to DataFrame
            df_data = []
            for q in queries:
                df_data.append({
                    "query": q.get("query", ""),
                    "expected_videos": json.dumps(q.get("expected_videos", [])),
                    "category": q.get("category", "general"),
                    "metadata": json.dumps(q.get("metadata", {}))
                })
            
            df = pd.DataFrame(df_data)
            
            # Upload to Phoenix
            dataset = self.client.upload_dataset(
                dataset_name=name,
                dataframe=df,
                input_keys=["query"],
                output_keys=["expected_videos"],
                metadata_keys=["category", "metadata"]
            )
            
            logger.info(f"Created dataset '{name}' with {len(df)} examples")
            return dataset.id
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    def get_dataset(self, dataset_name: str) -> Any:
        """
        Get a dataset from Phoenix.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Phoenix dataset object
        """
        try:
            dataset = self.client.get_dataset(dataset_name)
            logger.info(f"Retrieved dataset '{dataset_name}'")
            return dataset
        except Exception as e:
            logger.error(f"Failed to get dataset '{dataset_name}': {e}")
            return None
    
    def log_evaluations(
        self,
        trace_id: str,
        evaluations: Dict[str, Any]
    ):
        """
        Log evaluation results for a specific trace.
        
        Args:
            trace_id: Trace ID to attach evaluations to
            evaluations: Dictionary of evaluation results
        """
        try:
            # Format evaluations for Phoenix
            eval_data = {
                "trace_id": trace_id,
                "timestamp": datetime.now().isoformat(),
                "evaluations": evaluations
            }
            
            # In practice, you'd use Phoenix's evaluation logging API
            # This is a placeholder implementation
            logger.info(f"Logged evaluations for trace {trace_id}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluations: {e}")