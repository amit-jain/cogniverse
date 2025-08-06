"""
Phoenix Experiments for Systematic Evaluation

This module provides experiment management for running systematic evaluations
with Phoenix, including A/B testing, hyperparameter sweeps, and regression testing.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import hashlib

import pandas as pd
import numpy as np
import phoenix as px
from phoenix.evals import (
    RelevanceEvaluator,  # Changed from RetrievalEvaluator
    run_evals,
    default_templates
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    profiles: List[str]
    strategies: List[str]
    dataset_name: str
    metrics: List[str] = field(default_factory=lambda: ["mrr", "ndcg", "precision", "recall"])
    num_iterations: int = 1
    batch_size: int = 10
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


class CogniverseEvaluator:
    """Custom evaluator for Cogniverse using Phoenix Evals"""
    
    def __init__(self, profile: str, strategy: str):
        self.profile = profile
        self.strategy = strategy
        self.search_service = None
        self._init_search_service()
    
    def _init_search_service(self):
        """Initialize search service for the profile"""
        try:
            from src.search.search_service import SearchService
            from src.tools.config import get_config
            
            config = get_config()
            self.search_service = SearchService(config, self.profile)
            logger.info(f"Initialized evaluator for {self.profile}/{self.strategy}")
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
    
    async def evaluate(self, query: str, expected_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            query: Search query
            expected_output: Expected results with ground truth
            
        Returns:
            Evaluation results with metrics
        """
        if not self.search_service:
            return {"error": "Search service not initialized"}
        
        try:
            # Execute search
            start_time = time.time()
            search_results = self.search_service.search(
                query=query,
                top_k=10,
                ranking_strategy=self.strategy
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract results
            result_videos = []
            for result in search_results:
                result_dict = result.to_dict()
                video_id = result_dict.get('source_id', 
                                          result_dict['document_id'].split('_')[0])
                result_videos.append(video_id)
            
            # Get expected videos
            expected_videos = expected_output.get("expected_videos", [])
            
            # Calculate metrics
            metrics = self._calculate_metrics(result_videos, expected_videos)
            
            return {
                "status": "success",
                "results": result_videos,
                "metrics": metrics,
                "latency_ms": latency_ms,
                "profile": self.profile,
                "strategy": self.strategy
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "profile": self.profile,
                "strategy": self.strategy
            }
    
    def _calculate_metrics(self, results: List[str], expected: List[str]) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        metrics = {}
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, video in enumerate(results):
            if video in expected:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr
        
        # Precision at different k values
        for k in [1, 5, 10]:
            if k <= len(results):
                relevant_at_k = sum(1 for v in results[:k] if v in expected)
                metrics[f"precision_at_{k}"] = relevant_at_k / k
            else:
                metrics[f"precision_at_{k}"] = 0.0
        
        # Recall at different k values
        if expected:
            for k in [1, 5, 10]:
                if k <= len(results):
                    relevant_at_k = sum(1 for v in results[:k] if v in expected)
                    metrics[f"recall_at_{k}"] = relevant_at_k / len(expected)
                else:
                    metrics[f"recall_at_{k}"] = 0.0
        else:
            for k in [1, 5, 10]:
                metrics[f"recall_at_{k}"] = 0.0
        
        # NDCG@10
        relevances = [1 if vid in expected else 0 for vid in results[:10]]
        dcg = relevances[0] if relevances else 0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)
        
        ideal_relevances = [1] * min(len(expected), 10) + [0] * max(0, 10 - len(expected))
        idcg = ideal_relevances[0] if ideal_relevances else 0
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 2)
        
        metrics["ndcg_at_10"] = dcg / idcg if idcg > 0 else 0
        
        return metrics


class ExperimentOrchestrator:
    """Orchestrate experiments with Phoenix"""
    
    def __init__(self):
        self.client = px.Client()
        self.experiments = {}
        self.evaluators = {}
    
    async def run_experiment(self, config: ExperimentConfig) -> str:
        """
        Run a complete experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # Create experiment in Phoenix
        experiment_id = hashlib.md5(
            f"{config.name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        logger.info(f"Starting experiment '{config.name}' (ID: {experiment_id})")
        
        # Load dataset
        dataset = self.client.get_dataset(name=config.dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{config.dataset_name}' not found")
        
        # Initialize evaluators for each configuration
        for profile in config.profiles:
            for strategy in config.strategies:
                key = f"{profile}_{strategy}"
                self.evaluators[key] = CogniverseEvaluator(profile, strategy)
        
        # Run experiment iterations
        all_results = []
        for iteration in range(config.num_iterations):
            logger.info(f"Running iteration {iteration + 1}/{config.num_iterations}")
            
            iteration_results = await self._run_iteration(
                config, dataset, iteration, experiment_id
            )
            all_results.extend(iteration_results)
        
        # Aggregate results
        experiment_summary = self._aggregate_results(all_results, config)
        
        # Store experiment
        self.experiments[experiment_id] = {
            "config": config,
            "results": all_results,
            "summary": experiment_summary,
            "created_at": datetime.now()
        }
        
        # Log experiment summary to Phoenix
        self._log_experiment_to_phoenix(experiment_id, experiment_summary)
        
        logger.info(f"Experiment '{config.name}' completed (ID: {experiment_id})")
        return experiment_id
    
    async def _run_iteration(
        self,
        config: ExperimentConfig,
        dataset: pd.DataFrame,
        iteration: int,
        experiment_id: str
    ) -> List[Dict[str, Any]]:
        """Run a single iteration of the experiment"""
        results = []
        
        # Process dataset in batches
        # Get examples from dataset
        if hasattr(dataset, 'examples'):
            # Phoenix Dataset object
            example_items = list(dataset.examples.items())
        else:
            # Assume it's a DataFrame or similar
            if hasattr(dataset, 'as_dataframe'):
                dataset_df = dataset.as_dataframe()
            elif hasattr(dataset, 'to_pandas'):
                dataset_df = dataset.to_pandas()
            elif isinstance(dataset, pd.DataFrame):
                dataset_df = dataset
            else:
                dataset_df = pd.DataFrame(dataset)
            
            # Convert DataFrame to example items format
            example_items = []
            for idx, row in dataset_df.iterrows():
                example_id = row.get("id", f"example_{idx}")
                example_items.append((example_id, row))
        
        for batch_start in range(0, len(example_items), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(example_items))
            batch = example_items[batch_start:batch_end]
            
            # Run evaluations in parallel for this batch
            tasks = []
            for example_id, example in batch:
                # Handle Phoenix Example object
                if hasattr(example, 'input') and hasattr(example, 'output'):
                    # Phoenix Example object
                    input_data = example.input
                    output_data = example.output
                    
                    # Extract query and expected output
                    query = input_data.get("query", "")
                    
                    # Parse expected videos from string representation
                    import ast
                    expected_videos_str = output_data.get("expected_videos", "[]")
                    try:
                        expected_videos = ast.literal_eval(expected_videos_str) if isinstance(expected_videos_str, str) else expected_videos_str
                    except:
                        expected_videos = []
                    
                    expected_output = {
                        "expected_videos": expected_videos,
                        "relevance_scores": {}
                    }
                else:
                    # DataFrame row
                    if isinstance(example.get("input"), dict) and "input" in example["input"]:
                        # Nested structure
                        query = example["input"]["input"]["query"]
                        expected_output = example["output"]["expected_output"]
                    else:
                        # Direct structure
                        query = example["input"]["query"]
                        expected_output = example["expected_output"]
                
                for profile in config.profiles:
                    for strategy in config.strategies:
                        key = f"{profile}_{strategy}"
                        evaluator = self.evaluators[key]
                        
                        # Create evaluation task
                        task = self._create_evaluation_task(
                            evaluator, query, expected_output,
                            experiment_id, iteration, example_id
                        )
                        tasks.append(task)
            
            # Execute batch in parallel with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=config.timeout_seconds
                )
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Evaluation failed: {result}")
                        results.append({"error": str(result)})
                    else:
                        results.append(result)
                        
            except asyncio.TimeoutError:
                logger.error(f"Batch evaluation timed out after {config.timeout_seconds}s")
        
        return results
    
    async def _create_evaluation_task(
        self,
        evaluator: CogniverseEvaluator,
        query: str,
        expected_output: Dict[str, Any],
        experiment_id: str,
        iteration: int,
        example_id: str
    ) -> Dict[str, Any]:
        """Create an evaluation task with tracing"""
        
        # Create Phoenix trace
        from phoenix.trace import using_project
        from opentelemetry import trace as otel_trace
        
        tracer = otel_trace.get_tracer(__name__)
        
        with tracer.start_as_current_span(
            name="experiment_evaluation",
            kind=otel_trace.SpanKind.CLIENT
        ) as span:
            span.set_attributes({
                "experiment_id": experiment_id,
                "iteration": iteration,
                "example_id": example_id,
                "profile": evaluator.profile,
                "strategy": evaluator.strategy,
                "query": query
            })
            
            # Run evaluation
            result = await evaluator.evaluate(query, expected_output)
            
            # Add span outputs
            if result["status"] == "success":
                span.set_attributes({
                    "num_results": len(result.get("results", [])),
                    "mrr": result.get("metrics", {}).get("mrr", 0.0),
                    "latency_ms": result.get("latency_ms", 0)
                })
            else:
                span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, result.get("error", "Unknown error")))
            
            # Add trace ID to result
            result["trace_id"] = format(span.get_span_context().trace_id, '032x')
            result["experiment_id"] = experiment_id
            result["iteration"] = iteration
            result["example_id"] = example_id
            result["query"] = query
            result["expected_output"] = expected_output
            
            return result
    
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Aggregate experiment results"""
        summary = {
            "total_evaluations": len(results),
            "successful_evaluations": sum(1 for r in results if r.get("status") == "success"),
            "failed_evaluations": sum(1 for r in results if r.get("status") == "error"),
            "configurations": {}
        }
        
        # Group by configuration
        for profile in config.profiles:
            for strategy in config.strategies:
                config_key = f"{profile}_{strategy}"
                config_results = [
                    r for r in results
                    if r.get("profile") == profile and r.get("strategy") == strategy
                ]
                
                if not config_results:
                    continue
                
                # Calculate aggregate metrics
                metrics_agg = {}
                for metric in config.metrics:
                    metric_values = []
                    for r in config_results:
                        if "metrics" in r and metric in r["metrics"]:
                            metric_values.append(r["metrics"][metric])
                        elif "metrics" in r:
                            # Try with suffixes
                            for suffix in ["", "_at_1", "_at_5", "_at_10"]:
                                metric_key = f"{metric}{suffix}"
                                if metric_key in r["metrics"]:
                                    metric_values.append(r["metrics"][metric_key])
                                    break
                    
                    if metric_values:
                        metrics_agg[metric] = {
                            "mean": np.mean(metric_values),
                            "std": np.std(metric_values),
                            "min": np.min(metric_values),
                            "max": np.max(metric_values),
                            "median": np.median(metric_values)
                        }
                
                # Calculate latency stats
                latencies = [r.get("latency_ms", 0) for r in config_results if "latency_ms" in r]
                if latencies:
                    latency_stats = {
                        "mean": np.mean(latencies),
                        "p50": np.percentile(latencies, 50),
                        "p95": np.percentile(latencies, 95),
                        "p99": np.percentile(latencies, 99)
                    }
                else:
                    latency_stats = {}
                
                summary["configurations"][config_key] = {
                    "num_evaluations": len(config_results),
                    "success_rate": sum(1 for r in config_results if r.get("status") == "success") / len(config_results),
                    "metrics": metrics_agg,
                    "latency": latency_stats
                }
        
        return summary
    
    def _log_experiment_to_phoenix(self, experiment_id: str, summary: Dict[str, Any]):
        """Log experiment summary to Phoenix"""
        try:
            # Phoenix doesn't have a direct log_experiment API
            # Instead, we'll use OpenTelemetry to create experiment spans
            from opentelemetry import trace as otel_trace
            
            tracer = otel_trace.get_tracer(__name__)
            
            # Create experiment summary span
            with tracer.start_as_current_span(
                name=f"experiment_{experiment_id}_summary",
                kind=otel_trace.SpanKind.CLIENT
            ) as span:
                span.set_attributes({
                    "experiment_id": experiment_id,
                    "total_evaluations": summary.get("total_evaluations", 0),
                    "successful_evaluations": summary.get("successful_evaluations", 0),
                    "failed_evaluations": summary.get("failed_evaluations", 0),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log configuration metrics
                for config_key, config_summary in summary.get("configurations", {}).items():
                    if "metrics" in config_summary:
                        for metric_name, metric_stats in config_summary["metrics"].items():
                            if isinstance(metric_stats, dict):
                                span.set_attributes({
                                    f"{config_key}.{metric_name}.mean": metric_stats.get("mean", 0),
                                    f"{config_key}.{metric_name}.std": metric_stats.get("std", 0)
                                })
                
        except Exception as e:
            logger.error(f"Failed to log experiment to Phoenix: {e}")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for a specific experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiments[experiment_id]
    
    def compare_configurations(
        self,
        experiment_id: str,
        metric: str = "mrr"
    ) -> pd.DataFrame:
        """
        Compare configurations within an experiment
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to compare
            
        Returns:
            DataFrame with configuration comparison
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        summary = experiment["summary"]
        
        comparison_data = []
        for config_key, config_summary in summary["configurations"].items():
            profile, strategy = config_key.rsplit("_", 1)
            
            row = {
                "configuration": config_key,
                "profile": profile,
                "strategy": strategy,
                "num_evaluations": config_summary["num_evaluations"],
                "success_rate": config_summary["success_rate"]
            }
            
            # Add metric stats
            if metric in config_summary["metrics"]:
                metric_stats = config_summary["metrics"][metric]
                row.update({
                    f"{metric}_mean": metric_stats["mean"],
                    f"{metric}_std": metric_stats["std"],
                    f"{metric}_median": metric_stats["median"]
                })
            
            # Add latency stats
            if "latency" in config_summary:
                row.update({
                    "latency_p50": config_summary["latency"].get("p50", 0),
                    "latency_p95": config_summary["latency"].get("p95", 0)
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by metric mean
        if f"{metric}_mean" in df.columns:
            df = df.sort_values(by=f"{metric}_mean", ascending=False)
        
        return df


import time  # Add this import at the top