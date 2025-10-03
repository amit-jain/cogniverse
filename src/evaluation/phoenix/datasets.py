"""
Phoenix Dataset Management for Video Retrieval Evaluation

This module manages datasets in Phoenix for evaluation, enabling:
- Versioned datasets with ground truth
- Trace-to-dataset linking
- Batch evaluation against datasets
- Experiment tracking with metrics
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import phoenix as px
from phoenix.experiments import evaluate_experiment, run_experiment
from phoenix.trace import DocumentEvaluations, SpanEvaluations

logger = logging.getLogger(__name__)


class PhoenixDatasetManager:
    """Manages evaluation datasets in Phoenix"""
    
    def __init__(self):
        self.client = px.Client()
        self.datasets = {}
        self.active_dataset = None
    
    def create_dataset_from_queries(
        self,
        name: str,
        queries: List[Dict[str, Any]],
        version: str = None,
        description: str = None
    ) -> str:
        """
        Create a Phoenix dataset from video queries
        
        Args:
            name: Dataset name
            queries: List of query dictionaries with expected results
            version: Dataset version (auto-generated if not provided)
            description: Dataset description
            
        Returns:
            Dataset ID
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_name = f"{name}_v{version}"
        
        # Convert queries to Phoenix dataset format
        dataset_records = []
        for i, query_data in enumerate(queries):
            # Flatten the structure for Phoenix
            record = {
                "id": f"{dataset_name}_{i}",
                "query": query_data["query"],
                "category": query_data.get("category", "general"),
                "expected_videos": str(query_data["expected_videos"]),  # Convert list to string
                "relevance_scores": str(query_data.get("relevance_scores", {})),  # Convert dict to string
                "query_type": query_data.get("category", "general"),
                "created_at": datetime.now().isoformat(),
                "version": version
            }
            dataset_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(dataset_records)
        
        # Upload to Phoenix
        dataset = self.client.upload_dataset(
            dataset_name=dataset_name,
            dataframe=df,
            input_keys=["query", "category"],
            output_keys=["expected_videos", "relevance_scores"],
            metadata_keys=["query_type", "created_at", "version"]
        )
        
        # Store reference
        self.datasets[dataset_name] = {
            "id": dataset.id,
            "name": dataset_name,
            "version": version,
            "description": description,
            "num_examples": len(dataset_records),
            "created_at": datetime.now()
        }
        
        logger.info(f"Created Phoenix dataset '{dataset_name}' with {len(dataset_records)} examples")
        return dataset.id
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a dataset from Phoenix"""
        try:
            dataset = self.client.get_dataset(name=dataset_name)
            logger.info(f"Loaded dataset '{dataset_name}' with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            return None
    
    def set_active_dataset(self, dataset_name: str):
        """Set the active dataset for evaluation"""
        if dataset_name in self.datasets:
            self.active_dataset = dataset_name
            logger.info(f"Active dataset set to: {dataset_name}")
        else:
            # Try to load from Phoenix
            dataset = self.load_dataset(dataset_name)
            if dataset is not None:
                self.active_dataset = dataset_name
                self.datasets[dataset_name] = {
                    "name": dataset_name,
                    "num_examples": len(dataset)
                }
            else:
                raise ValueError(f"Dataset '{dataset_name}' not found")
    
    def link_trace_to_dataset(
        self,
        trace_id: str,
        dataset_example_id: str,
        dataset_name: str = None
    ):
        """
        Link a trace to a dataset example for evaluation
        
        Args:
            trace_id: Phoenix trace ID
            dataset_example_id: Example ID in the dataset
            dataset_name: Dataset name (uses active if not provided)
        """
        dataset_name = dataset_name or self.active_dataset
        if not dataset_name:
            raise ValueError("No dataset specified or active")
        
        # Create link in Phoenix
        self.client.log_dataset_example_to_trace(
            trace_id=trace_id,
            dataset_name=dataset_name,
            example_id=dataset_example_id
        )
        
        logger.debug(f"Linked trace {trace_id} to dataset example {dataset_example_id}")
    
    def add_evaluations_to_traces(
        self,
        evaluations: List[Dict[str, Any]],
        dataset_name: str = None
    ):
        """
        Add evaluation scores to traces in Phoenix
        
        Args:
            evaluations: List of evaluation dictionaries with trace_id and scores
            dataset_name: Dataset name for context
        """
        dataset_name = dataset_name or self.active_dataset
        
        # Convert evaluations to Phoenix format
        span_evals = []
        for eval_data in evaluations:
            span_eval = SpanEvaluations(
                trace_id=eval_data["trace_id"],
                span_id=eval_data.get("span_id", eval_data["trace_id"]),
                name="retrieval_evaluation",
                score=eval_data.get("overall_score", 0.0),
                label=eval_data.get("label", ""),
                explanation=eval_data.get("explanation", ""),
                metadata={
                    "mrr": eval_data.get("mrr", 0.0),
                    "ndcg": eval_data.get("ndcg", 0.0),
                    "precision_at_5": eval_data.get("precision_at_5", 0.0),
                    "recall_at_5": eval_data.get("recall_at_5", 0.0),
                    "dataset": dataset_name,
                    "profile": eval_data.get("profile", ""),
                    "strategy": eval_data.get("strategy", "")
                }
            )
            span_evals.append(span_eval)
        
        # Log evaluations to Phoenix
        self.client.log_evaluations(span_evals)
        
        logger.info(f"Added {len(span_evals)} evaluations to traces")
    
    def create_evaluation_dataset_from_traces(
        self,
        trace_filter: Dict[str, Any] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Create an evaluation dataset from collected traces
        
        Args:
            trace_filter: Filter criteria for traces
            limit: Maximum number of traces to include
            
        Returns:
            DataFrame with traces ready for evaluation
        """
        # Get traces from Phoenix
        traces = self.client.get_traces(
            filter=trace_filter,
            limit=limit
        )
        
        # Convert to evaluation format
        eval_records = []
        for trace in traces:
            # Extract query and results from trace
            inputs = trace.get("inputs", {})
            outputs = trace.get("outputs", {})
            metadata = trace.get("metadata", {})
            
            record = {
                "trace_id": trace["trace_id"],
                "query": inputs.get("query", ""),
                "results": outputs.get("results", []),
                "profile": metadata.get("profile", ""),
                "strategy": metadata.get("strategy", ""),
                "latency_ms": metadata.get("latency_ms", 0),
                "timestamp": trace.get("timestamp", "")
            }
            eval_records.append(record)
        
        df = pd.DataFrame(eval_records)
        logger.info(f"Created evaluation dataset with {len(df)} traces")
        return df


class PhoenixExperimentRunner:
    """Run and manage experiments with Phoenix"""
    
    def __init__(self, dataset_manager: PhoenixDatasetManager):
        self.dataset_manager = dataset_manager
        self.client = px.Client()
        self.experiments = {}
    
    async def run_retrieval_experiment(
        self,
        name: str,
        profiles: List[str],
        strategies: List[str],
        dataset_name: str,
        evaluators: List[Any] = None
    ) -> str:
        """
        Run a comprehensive retrieval experiment
        
        Args:
            name: Experiment name
            profiles: Video processing profiles to test
            strategies: Ranking strategies to test
            dataset_name: Dataset to evaluate against
            evaluators: List of evaluator functions
            
        Returns:
            Experiment ID
        """
        # Create experiment in Phoenix
        experiment_id = self.client.create_experiment(
            name=name,
            description=f"Testing {len(profiles)} profiles with {len(strategies)} strategies",
            metadata={
                "profiles": profiles,
                "strategies": strategies,
                "dataset": dataset_name,
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Load dataset
        dataset = self.dataset_manager.load_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Run experiment for each configuration
        from src.app.search.service import SearchService
        from src.common.config_compat import get_config  # DEPRECATED: Migrate to ConfigManager
        
        config = get_config()
        results = []
        
        for profile in profiles:
            for strategy in strategies:
                run_name = f"{profile}_{strategy}"
                
                # Create run in Phoenix
                with self.client.start_run(
                    experiment_id=experiment_id,
                    run_name=run_name
                ) as run:
                    try:
                        # Initialize search service
                        search_service = SearchService(config, profile)
                        
                        # Evaluate on dataset
                        run_results = await self._evaluate_configuration(
                            search_service,
                            dataset,
                            profile,
                            strategy,
                            dataset_name
                        )
                        
                        # Log metrics to Phoenix
                        run.log_metrics(run_results["metrics"])
                        run.log_params({
                            "profile": profile,
                            "strategy": strategy
                        })
                        
                        # Store results
                        results.append({
                            "run_name": run_name,
                            "profile": profile,
                            "strategy": strategy,
                            "results": run_results
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to run {run_name}: {e}")
                        run.log_metrics({"error": 1})
        
        # Store experiment reference
        self.experiments[experiment_id] = {
            "name": name,
            "profiles": profiles,
            "strategies": strategies,
            "dataset": dataset_name,
            "results": results,
            "created_at": datetime.now()
        }
        
        logger.info(f"Completed experiment '{name}' with {len(results)} runs")
        return experiment_id
    
    async def _evaluate_configuration(
        self,
        search_service,
        dataset: pd.DataFrame,
        profile: str,
        strategy: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single configuration against the dataset"""
        evaluations = []
        all_metrics = []
        
        for idx, row in dataset.iterrows():
            query = row["input"]["query"]
            expected_videos = row["expected_output"]["expected_videos"]
            
            # Create trace for this evaluation
            with px.trace(
                name="eval_retrieval",
                kind="RETRIEVAL"
            ) as trace:
                # Add to trace context
                trace.set_inputs({"query": query})
                trace.set_metadata({
                    "profile": profile,
                    "strategy": strategy,
                    "dataset": dataset_name,
                    "example_id": row["id"]
                })
                
                try:
                    # Execute search
                    start_time = time.time()
                    search_results = search_service.search(
                        query=query,
                        top_k=10,
                        ranking_strategy=strategy
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Convert results
                    result_videos = []
                    for result in search_results:
                        result_dict = result.to_dict()
                        video_id = result_dict.get('source_id', 
                                                  result_dict['document_id'].split('_')[0])
                        result_videos.append(video_id)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(result_videos, expected_videos)
                    metrics["latency_ms"] = latency_ms
                    all_metrics.append(metrics)
                    
                    # Set trace outputs
                    trace.set_outputs({
                        "results": result_videos[:5],
                        "num_results": len(result_videos)
                    })
                    
                    # Create evaluation
                    evaluation = {
                        "trace_id": trace.trace_id,
                        "example_id": row["id"],
                        "query": query,
                        "profile": profile,
                        "strategy": strategy,
                        "mrr": metrics["mrr"],
                        "ndcg": metrics["ndcg"],
                        "precision_at_5": metrics["precision_at_5"],
                        "recall_at_5": metrics["recall_at_5"],
                        "overall_score": metrics["mrr"],  # Use MRR as primary metric
                        "label": "success" if metrics["mrr"] > 0 else "failure",
                        "explanation": f"MRR: {metrics['mrr']:.3f}, P@5: {metrics['precision_at_5']:.3f}"
                    }
                    evaluations.append(evaluation)
                    
                    # Link trace to dataset example
                    self.dataset_manager.link_trace_to_dataset(
                        trace.trace_id,
                        row["id"],
                        dataset_name
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate query '{query}': {e}")
                    trace.set_outputs({"error": str(e)})
                    all_metrics.append({
                        "mrr": 0, "ndcg": 0, 
                        "precision_at_5": 0, "recall_at_5": 0,
                        "latency_ms": 0
                    })
        
        # Add evaluations to traces
        self.dataset_manager.add_evaluations_to_traces(evaluations)
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for metric_name in ["mrr", "ndcg", "precision_at_5", "recall_at_5", "latency_ms"]:
            values = [m[metric_name] for m in all_metrics]
            aggregate_metrics[f"mean_{metric_name}"] = np.mean(values)
            aggregate_metrics[f"std_{metric_name}"] = np.std(values)
            if metric_name == "latency_ms":
                aggregate_metrics[f"p95_{metric_name}"] = np.percentile(values, 95)
        
        return {
            "metrics": aggregate_metrics,
            "evaluations": evaluations,
            "num_evaluated": len(evaluations)
        }
    
    def _calculate_metrics(
        self,
        results: List[str],
        expected: List[str]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        metrics = {}
        
        # MRR
        mrr = 0.0
        for i, video in enumerate(results):
            if video in expected:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr
        
        # NDCG@10
        relevances = [1 if vid in expected else 0 for vid in results[:10]]
        dcg = relevances[0] if relevances else 0
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)
        
        ideal_relevances = [1] * min(len(expected), 10) + [0] * max(0, 10 - len(expected))
        idcg = ideal_relevances[0] if ideal_relevances else 0
        for i in range(1, len(ideal_relevances)):
            idcg += ideal_relevances[i] / np.log2(i + 2)
        
        metrics["ndcg"] = dcg / idcg if idcg > 0 else 0
        
        # Precision and Recall at 5
        top_5 = results[:5]
        relevant_in_top_5 = sum(1 for v in top_5 if v in expected)
        metrics["precision_at_5"] = relevant_in_top_5 / len(top_5) if top_5 else 0
        metrics["recall_at_5"] = relevant_in_top_5 / len(expected) if expected else 0
        
        return metrics
    
    def run_batch_evaluation(
        self,
        trace_ids: List[str],
        dataset_name: str,
        evaluators: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on existing traces
        
        Args:
            trace_ids: List of trace IDs to evaluate
            dataset_name: Dataset to evaluate against
            evaluators: Custom evaluator functions
            
        Returns:
            Evaluation results
        """
        # Load dataset
        dataset = self.dataset_manager.load_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Get traces from Phoenix
        traces = self.client.get_traces_by_ids(trace_ids)
        
        evaluations = []
        for trace in traces:
            # Extract query from trace
            query = trace["inputs"].get("query", "")
            
            # Find matching dataset example
            matching_example = None
            for idx, row in dataset.iterrows():
                if row["input"]["query"] == query:
                    matching_example = row
                    break
            
            if matching_example is None:
                logger.warning(f"No matching dataset example for query: {query}")
                continue
            
            # Extract results from trace
            results = trace["outputs"].get("results", [])
            expected = matching_example["expected_output"]["expected_videos"]
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, expected)
            
            # Create evaluation
            evaluation = {
                "trace_id": trace["trace_id"],
                "query": query,
                "profile": trace["metadata"].get("profile", ""),
                "strategy": trace["metadata"].get("strategy", ""),
                **metrics,
                "overall_score": metrics["mrr"],
                "label": "success" if metrics["mrr"] > 0 else "failure"
            }
            evaluations.append(evaluation)
        
        # Add evaluations to Phoenix
        self.dataset_manager.add_evaluations_to_traces(evaluations)
        
        # Calculate summary statistics
        summary = {
            "num_evaluated": len(evaluations),
            "mean_mrr": np.mean([e["mrr"] for e in evaluations]),
            "mean_ndcg": np.mean([e["ndcg"] for e in evaluations]),
            "success_rate": sum(1 for e in evaluations if e["label"] == "success") / len(evaluations)
        }
        
        logger.info(f"Batch evaluation complete: {summary}")
        return {
            "evaluations": evaluations,
            "summary": summary
        }
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Metrics to compare (default: all)
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ["mean_mrr", "mean_ndcg", "mean_precision_at_5", "mean_recall_at_5"]
        
        comparison_data = []
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                logger.warning(f"Experiment {exp_id} not found")
                continue
            
            exp_data = self.experiments[exp_id]
            for result in exp_data["results"]:
                row = {
                    "experiment": exp_data["name"],
                    "profile": result["profile"],
                    "strategy": result["strategy"]
                }
                
                # Add metrics
                for metric in metrics:
                    row[metric] = result["results"]["metrics"].get(metric, 0)
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Add ranking columns
        for metric in metrics:
            df[f"{metric}_rank"] = df[metric].rank(ascending=False)
        
        return df.sort_values(by=metrics[0], ascending=False)


import time  # Add this import at the top of the file
