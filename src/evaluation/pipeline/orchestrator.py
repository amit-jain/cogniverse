"""
Unified Evaluation Pipeline Orchestrator

This module orchestrates the complete evaluation pipeline, combining:
- Inspect AI for structured task-based evaluation
- Phoenix for tracing, datasets, and experiments
- Custom metrics and reporting
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import phoenix as px
from inspect_ai import eval_async
from inspect_ai.log import EvalLog

from ..phoenix.datasets import PhoenixDatasetManager, PhoenixExperimentRunner
from ..phoenix.experiments import ExperimentOrchestrator, ExperimentConfig
from ..phoenix.monitoring import RetrievalMonitor
from ..phoenix.instrumentation import CogniverseInstrumentor
from ..inspect_tasks.video_retrieval import (
    video_retrieval_accuracy,
    temporal_understanding,
    multimodal_alignment,
    failure_analysis
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main evaluation pipeline orchestrating Inspect and Phoenix"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize evaluation pipeline
        
        Args:
            config_path: Path to evaluation configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize Phoenix components
        self.dataset_manager = PhoenixDatasetManager()
        self.experiment_runner = PhoenixExperimentRunner(self.dataset_manager)
        self.experiment_orchestrator = ExperimentOrchestrator()
        self.monitor = RetrievalMonitor()
        
        # Initialize instrumentation
        self.instrumentor = CogniverseInstrumentor()
        
        # Track evaluation state
        self.current_evaluation = None
        self.evaluation_history = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load evaluation configuration"""
        if config_path is None:
            # Use default configuration
            return self._get_default_config()
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration"""
        return {
            "evaluation": {
                "name": "Cogniverse Video RAG Evaluation",
                "profiles": ["frame_based_colpali", "direct_video_global"],
                "strategies": ["float_float", "binary_binary", "hybrid_binary_bm25"],
                "metrics": ["mrr", "ndcg", "precision", "recall"],
                "tasks": ["video_retrieval_accuracy"]
            },
            "phoenix": {
                "instrumentation": True,
                "monitoring": True,
                "dataset_versioning": True
            },
            "inspect": {
                "parallel_tasks": True,
                "save_logs": True
            }
        }
    
    async def run_comprehensive_evaluation(
        self,
        evaluation_name: str,
        profiles: List[str] = None,
        strategies: List[str] = None,
        tasks: List[str] = None,
        dataset_path: str = None,
        use_phoenix_dataset: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation pipeline
        
        Args:
            evaluation_name: Name for this evaluation run
            profiles: Video processing profiles to test
            strategies: Ranking strategies to test
            tasks: Inspect AI tasks to run
            dataset_path: Path to evaluation dataset
            use_phoenix_dataset: Whether to use Phoenix for dataset management
            
        Returns:
            Comprehensive evaluation results
        """
        # Use defaults from config if not specified
        profiles = profiles or self.config["evaluation"]["profiles"]
        strategies = strategies or self.config["evaluation"]["strategies"]
        tasks = tasks or self.config["evaluation"]["tasks"]
        
        logger.info(f"Starting comprehensive evaluation: {evaluation_name}")
        
        # Start instrumentation and monitoring
        if self.config["phoenix"]["instrumentation"]:
            self.instrumentor.instrument()
        
        if self.config["phoenix"]["monitoring"]:
            self.monitor.start()
        
        # Initialize evaluation state
        self.current_evaluation = {
            "name": evaluation_name,
            "started_at": datetime.now(),
            "profiles": profiles,
            "strategies": strategies,
            "tasks": tasks
        }
        
        try:
            # Step 1: Prepare dataset in Phoenix
            if use_phoenix_dataset:
                dataset_id = await self._prepare_phoenix_dataset(dataset_path, evaluation_name)
                self.current_evaluation["dataset_id"] = dataset_id
            
            # Step 2: Run Phoenix experiment for systematic evaluation
            phoenix_results = await self._run_phoenix_experiment(
                evaluation_name, profiles, strategies, dataset_id if use_phoenix_dataset else None
            )
            self.current_evaluation["phoenix_results"] = phoenix_results
            
            # Step 3: Run Inspect AI tasks for structured evaluation
            inspect_results = await self._run_inspect_tasks(
                tasks, profiles, strategies, dataset_path
            )
            self.current_evaluation["inspect_results"] = inspect_results
            
            # Step 4: Collect and analyze traces
            trace_analysis = await self._analyze_traces(evaluation_name)
            self.current_evaluation["trace_analysis"] = trace_analysis
            
            # Step 5: Generate comprehensive report
            report = self._generate_comprehensive_report()
            self.current_evaluation["report"] = report
            
            # Step 6: Save evaluation results
            self._save_evaluation_results(evaluation_name)
            
            # Add to history
            self.current_evaluation["completed_at"] = datetime.now()
            self.evaluation_history.append(self.current_evaluation)
            
            logger.info(f"Evaluation '{evaluation_name}' completed successfully")
            return self.current_evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.current_evaluation["error"] = str(e)
            raise
        
        finally:
            # Stop monitoring
            if self.config["phoenix"]["monitoring"]:
                self.monitor.stop()
    
    async def _prepare_phoenix_dataset(
        self,
        dataset_path: Optional[str],
        evaluation_name: str
    ) -> str:
        """Prepare dataset in Phoenix"""
        if dataset_path:
            # Load dataset from file
            with open(dataset_path, 'r') as f:
                queries = json.load(f)
        else:
            # Use default test queries
            from tests.comprehensive_video_query_test_v2 import VISUAL_TEST_QUERIES
            queries = VISUAL_TEST_QUERIES
        
        # Create Phoenix dataset
        dataset_id = self.dataset_manager.create_dataset_from_queries(
            name=f"{evaluation_name}_dataset",
            queries=queries,
            description=f"Dataset for evaluation: {evaluation_name}"
        )
        
        # Set as active dataset
        self.dataset_manager.set_active_dataset(f"{evaluation_name}_dataset_v{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info(f"Created Phoenix dataset with {len(queries)} queries")
        return dataset_id
    
    async def _run_phoenix_experiment(
        self,
        evaluation_name: str,
        profiles: List[str],
        strategies: List[str],
        dataset_id: Optional[str]
    ) -> Dict[str, Any]:
        """Run Phoenix experiment"""
        if not dataset_id:
            logger.warning("No dataset ID provided for Phoenix experiment")
            return {}
        
        # Create experiment configuration
        exp_config = ExperimentConfig(
            name=evaluation_name,
            description=f"Systematic evaluation of {len(profiles)} profiles with {len(strategies)} strategies",
            profiles=profiles,
            strategies=strategies,
            dataset_name=self.dataset_manager.active_dataset,
            metrics=self.config["evaluation"]["metrics"],
            num_iterations=1
        )
        
        # Run experiment
        experiment_id = await self.experiment_orchestrator.run_experiment(exp_config)
        
        # Get results
        results = self.experiment_orchestrator.get_experiment_results(experiment_id)
        
        # Get configuration comparison
        comparison_df = self.experiment_orchestrator.compare_configurations(
            experiment_id, metric="mrr"
        )
        
        return {
            "experiment_id": experiment_id,
            "summary": results["summary"],
            "comparison": comparison_df.to_dict(orient="records"),
            "best_configuration": comparison_df.iloc[0].to_dict() if not comparison_df.empty else None
        }
    
    async def _run_inspect_tasks(
        self,
        tasks: List[str],
        profiles: List[str],
        strategies: List[str],
        dataset_path: Optional[str]
    ) -> Dict[str, Any]:
        """Run Inspect AI evaluation tasks"""
        inspect_results = {}
        
        for task_name in tasks:
            logger.info(f"Running Inspect task: {task_name}")
            
            try:
                # Get task function
                if task_name == "video_retrieval_accuracy":
                    task = video_retrieval_accuracy(profiles, strategies, dataset_path)
                elif task_name == "temporal_understanding":
                    task = temporal_understanding(profiles, dataset_path)
                elif task_name == "multimodal_alignment":
                    task = multimodal_alignment(profiles, dataset_path)
                elif task_name == "failure_analysis":
                    task = failure_analysis(profiles, strategies, dataset_path)
                else:
                    logger.warning(f"Unknown task: {task_name}")
                    continue
                
                # Run evaluation
                eval_log = await eval_async(
                    task,
                    model="cogniverse",  # Custom model wrapper
                    log_dir="outputs/inspect_logs" if self.config["inspect"]["save_logs"] else None
                )
                
                # Extract results
                inspect_results[task_name] = self._process_inspect_results(eval_log)
                
            except Exception as e:
                logger.error(f"Failed to run Inspect task '{task_name}': {e}")
                inspect_results[task_name] = {"error": str(e)}
        
        return inspect_results
    
    def _process_inspect_results(self, eval_log: EvalLog) -> Dict[str, Any]:
        """Process Inspect evaluation log"""
        results = {
            "status": eval_log.status,
            "samples_evaluated": len(eval_log.samples),
            "scores": {}
        }
        
        # Extract scores
        for sample in eval_log.samples:
            if hasattr(sample, 'score') and sample.score:
                # Get configuration from metadata
                metadata = sample.metadata or {}
                config_key = f"{metadata.get('profile', 'unknown')}_{metadata.get('strategy', 'unknown')}"
                
                if config_key not in results["scores"]:
                    results["scores"][config_key] = []
                
                results["scores"][config_key].append({
                    "value": sample.score.value,
                    "answer": sample.score.answer,
                    "explanation": sample.score.explanation,
                    "metadata": sample.score.metadata
                })
        
        # Calculate aggregate scores
        for config_key, scores in results["scores"].items():
            values = [s["value"] for s in scores if isinstance(s["value"], (int, float))]
            if values:
                results["scores"][config_key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "samples": scores[:5]  # Keep first 5 samples for inspection
                }
        
        return results
    
    async def _analyze_traces(self, evaluation_name: str) -> Dict[str, Any]:
        """Analyze traces collected during evaluation"""
        try:
            # Get traces from Phoenix
            traces = self.client.get_traces(
                filter={"metadata.evaluation_name": evaluation_name},
                limit=10000
            )
            
            if not traces:
                return {"num_traces": 0}
            
            # Analyze traces
            analysis = {
                "num_traces": len(traces),
                "trace_types": {},
                "latency_distribution": {},
                "error_analysis": {}
            }
            
            # Group by trace type
            for trace in traces:
                trace_type = trace.get("name", "unknown")
                if trace_type not in analysis["trace_types"]:
                    analysis["trace_types"][trace_type] = 0
                analysis["trace_types"][trace_type] += 1
            
            # Latency analysis
            latencies = []
            for trace in traces:
                if "latency_ms" in trace.get("metadata", {}):
                    latencies.append(trace["metadata"]["latency_ms"])
            
            if latencies:
                analysis["latency_distribution"] = {
                    "mean": np.mean(latencies),
                    "p50": np.percentile(latencies, 50),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99)
                }
            
            # Error analysis
            errors = [t for t in traces if t.get("status") == "ERROR"]
            if errors:
                error_types = {}
                for error in errors:
                    error_type = error.get("error_type", "unknown")
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1
                
                analysis["error_analysis"] = {
                    "total_errors": len(errors),
                    "error_rate": len(errors) / len(traces),
                    "error_types": error_types
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze traces: {e}")
            return {"error": str(e)}
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            "summary": {
                "evaluation_name": self.current_evaluation["name"],
                "started_at": self.current_evaluation["started_at"].isoformat(),
                "profiles_tested": len(self.current_evaluation["profiles"]),
                "strategies_tested": len(self.current_evaluation["strategies"]),
                "tasks_completed": len(self.current_evaluation.get("inspect_results", {}))
            }
        }
        
        # Add Phoenix experiment results
        if "phoenix_results" in self.current_evaluation:
            phoenix = self.current_evaluation["phoenix_results"]
            report["phoenix_experiment"] = {
                "experiment_id": phoenix.get("experiment_id"),
                "best_configuration": phoenix.get("best_configuration"),
                "summary": phoenix.get("summary")
            }
        
        # Add Inspect task results
        if "inspect_results" in self.current_evaluation:
            report["inspect_tasks"] = {}
            for task_name, task_results in self.current_evaluation["inspect_results"].items():
                if "error" not in task_results:
                    report["inspect_tasks"][task_name] = {
                        "samples_evaluated": task_results.get("samples_evaluated", 0),
                        "aggregate_scores": task_results.get("scores", {})
                    }
        
        # Add trace analysis
        if "trace_analysis" in self.current_evaluation:
            report["trace_analysis"] = self.current_evaluation["trace_analysis"]
        
        # Add monitoring metrics
        if self.monitor:
            report["monitoring_metrics"] = self.monitor.get_metrics_summary()
            report["active_alerts"] = self.monitor.get_active_alerts()
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check Phoenix results
        if "phoenix_results" in self.current_evaluation:
            phoenix = self.current_evaluation["phoenix_results"]
            if phoenix.get("best_configuration"):
                best = phoenix["best_configuration"]
                recommendations.append(
                    f"Best performing configuration: {best.get('profile')} with {best.get('strategy')} strategy "
                    f"(MRR: {best.get('mrr_mean', 0):.3f})"
                )
        
        # Check for performance issues
        if "trace_analysis" in self.current_evaluation:
            trace_analysis = self.current_evaluation["trace_analysis"]
            if "latency_distribution" in trace_analysis:
                p95_latency = trace_analysis["latency_distribution"].get("p95", 0)
                if p95_latency > 1000:
                    recommendations.append(
                        f"High P95 latency detected: {p95_latency:.0f}ms. Consider optimization."
                    )
            
            if "error_analysis" in trace_analysis:
                error_rate = trace_analysis["error_analysis"].get("error_rate", 0)
                if error_rate > 0.05:
                    recommendations.append(
                        f"High error rate: {error_rate:.2%}. Review error logs for issues."
                    )
        
        # Check for failed tasks
        if "inspect_results" in self.current_evaluation:
            failed_tasks = [
                task for task, results in self.current_evaluation["inspect_results"].items()
                if "error" in results
            ]
            if failed_tasks:
                recommendations.append(
                    f"Failed tasks detected: {', '.join(failed_tasks)}. Review logs for details."
                )
        
        return recommendations
    
    def _save_evaluation_results(self, evaluation_name: str):
        """Save evaluation results to file"""
        output_dir = Path("outputs/evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{evaluation_name}_{timestamp}.json"
        
        # Convert datetime objects to strings
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_file, 'w') as f:
            json.dump(
                self.current_evaluation,
                f,
                indent=2,
                default=json_serializer
            )
        
        logger.info(f"Evaluation results saved to: {output_file}")
    
    async def run_batch_evaluation_on_traces(
        self,
        trace_ids: List[str] = None,
        time_range: Tuple[datetime, datetime] = None,
        dataset_name: str = None
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on existing traces
        
        Args:
            trace_ids: Specific trace IDs to evaluate
            time_range: Time range for traces
            dataset_name: Dataset to evaluate against
            
        Returns:
            Batch evaluation results
        """
        if not dataset_name:
            raise ValueError("Dataset name required for batch evaluation")
        
        # Get traces to evaluate
        if trace_ids:
            # Use specific trace IDs
            traces_to_eval = trace_ids
        else:
            # Get traces from time range
            filter_criteria = {}
            if time_range:
                filter_criteria["timestamp"] = {
                    "$gte": time_range[0].isoformat(),
                    "$lte": time_range[1].isoformat()
                }
            
            # Create evaluation dataset from traces
            eval_df = self.dataset_manager.create_evaluation_dataset_from_traces(
                trace_filter=filter_criteria,
                limit=1000
            )
            traces_to_eval = eval_df["trace_id"].tolist()
        
        logger.info(f"Running batch evaluation on {len(traces_to_eval)} traces")
        
        # Run batch evaluation
        results = self.experiment_runner.run_batch_evaluation(
            trace_ids=traces_to_eval,
            dataset_name=dataset_name
        )
        
        return results
    
    def export_evaluation_report(
        self,
        evaluation_name: str = None,
        format: str = "html"
    ) -> str:
        """
        Export evaluation report in specified format
        
        Args:
            evaluation_name: Name of evaluation (uses current if not specified)
            format: Export format (html, pdf, markdown)
            
        Returns:
            Path to exported report
        """
        eval_data = self.current_evaluation if evaluation_name is None else None
        
        if eval_data is None:
            # Find in history
            for eval in self.evaluation_history:
                if eval["name"] == evaluation_name:
                    eval_data = eval
                    break
        
        if eval_data is None:
            raise ValueError(f"Evaluation '{evaluation_name}' not found")
        
        # Generate report based on format
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "html":
            output_file = output_dir / f"{eval_data['name']}_{timestamp}.html"
            self._generate_html_report(eval_data, output_file)
        elif format == "markdown":
            output_file = output_dir / f"{eval_data['name']}_{timestamp}.md"
            self._generate_markdown_report(eval_data, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Report exported to: {output_file}")
        return str(output_file)
    
    def _generate_html_report(self, eval_data: Dict[str, Any], output_file: Path):
        """Generate HTML report"""
        # This would be a full HTML report generator
        # For now, create a simple HTML version
        html_content = f"""
        <html>
        <head>
            <title>{eval_data['name']} - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{eval_data['name']} - Evaluation Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            
            <h2>Summary</h2>
            <pre>{json.dumps(eval_data.get('report', {}).get('summary', {}), indent=2)}</pre>
            
            <h2>Recommendations</h2>
            <ul>
            {"".join(f"<li>{rec}</li>" for rec in eval_data.get('report', {}).get('recommendations', []))}
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, eval_data: Dict[str, Any], output_file: Path):
        """Generate Markdown report"""
        md_content = f"""# {eval_data['name']} - Evaluation Report

Generated: {datetime.now().isoformat()}

## Summary
```json
{json.dumps(eval_data.get('report', {}).get('summary', {}), indent=2)}
```

## Recommendations
{chr(10).join(f"- {rec}" for rec in eval_data.get('report', {}).get('recommendations', []))}
"""
        
        with open(output_file, 'w') as f:
            f.write(md_content)


# Make sure to have client attribute for trace analysis
EvaluationPipeline.client = property(lambda self: px.Client())