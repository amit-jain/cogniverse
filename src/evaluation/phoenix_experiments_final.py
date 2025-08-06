"""
Phoenix Experiments Final Version - with proper project separation and scoring
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import os

import phoenix as px
from phoenix.experiments import run_experiment
import opentelemetry.trace as trace
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from tests.comprehensive_video_query_test_v2 import VISUAL_TEST_QUERIES
from src.tools.config import get_config
from .evaluators.sync_reference_free import create_sync_evaluators
from .evaluators.golden_dataset import create_low_scoring_golden_dataset

logger = logging.getLogger(__name__)


class PhoenixExperimentRunner:
    """
    Final experiment runner with proper project separation and scoring
    """
    
    def __init__(self, 
                 experiment_project_name: str = "experiments", 
                 enable_quality_evaluators: bool = True,
                 enable_llm_evaluators: bool = False,
                 llm_model: str = "deepseek-r1:7b",
                 llm_base_url: str = "http://localhost:11434"):
        """
        Initialize with separate project for experiments
        
        Args:
            experiment_project_name: Phoenix project name for experiments
            enable_quality_evaluators: Whether to include additional quality evaluators
            enable_llm_evaluators: Whether to include LLM-based evaluators
            llm_model: LLM model to use for evaluation
            llm_base_url: Base URL for LLM API
        """
        self.client = px.Client()
        self.config = get_config()
        self.experiment_project = experiment_project_name
        self.enable_quality_evaluators = enable_quality_evaluators
        self.enable_llm_evaluators = enable_llm_evaluators
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        
        # Save original project name to restore later
        self.original_project = os.environ.get("PHOENIX_PROJECT_NAME", "default")
        
        # Set up tracing for experiments project
        self._setup_experiment_tracing()
        
        # Initialize dataset manager
        from .dataset_manager import DatasetManager
        self.dataset_manager = DatasetManager()
        
    def _setup_experiment_tracing(self):
        """Set up OpenTelemetry tracing for experiments project"""
        from opentelemetry.trace import set_tracer_provider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        
        # This ensures SearchService and other components have a tracer provider
        # Phoenix will override the project destination when running experiments
        provider = TracerProvider()
        
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        headers = {"x-phoenix-project-name": self.experiment_project}
        
        exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers=headers
        )
        
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        # Set as global provider
        set_tracer_provider(provider)
        logger.info(f"Set up base tracer provider for experiments")
        
    def create_experiment_dataset(
        self, 
        dataset_name: Optional[str] = None,
        csv_path: Optional[str] = None,
        queries: Optional[List[Dict]] = None,
        force_new: bool = False
    ) -> Any:
        """
        Create or get Phoenix dataset for experiments
        
        Args:
            dataset_name: Name of dataset to use/create
            csv_path: Path to CSV file with queries
            queries: List of query dictionaries (if not using CSV)
            force_new: Force creation of new dataset
            
        Returns:
            Phoenix dataset
        """
        # If dataset name provided, try to use existing or load from CSV
        if dataset_name:
            # Check if CSV path provided
            if csv_path:
                dataset_id = self.dataset_manager.get_or_create_dataset(
                    name=dataset_name,
                    csv_path=csv_path,
                    force_new=force_new
                )
            else:
                # Try to get existing dataset
                dataset_info = self.dataset_manager.get_dataset_info(dataset_name)
                if dataset_info and not force_new:
                    dataset_id = dataset_info["phoenix_id"]
                else:
                    # Create from queries or default queries
                    if queries is None:
                        queries = VISUAL_TEST_QUERIES
                    
                    # Convert to dataframe
                    df_data = []
                    for q in queries:
                        df_data.append({
                            "query": q["query"] if isinstance(q, dict) else q,
                            "expected_videos": q.get("expected_videos", ""),
                            "category": q.get("category", "general") if isinstance(q, dict) else "general"
                        })
                    df = pd.DataFrame(df_data)
                    
                    dataset_id = self.dataset_manager.get_or_create_dataset(
                        name=dataset_name,
                        dataframe=df,
                        force_new=force_new
                    )
            
            # Get dataset from Phoenix
            dataset = self.client.get_dataset(id=dataset_id)
            return dataset
            
        # Legacy behavior - create temporary dataset
        if queries is None:
            queries = VISUAL_TEST_QUERIES
            
        # Add challenging queries
        all_queries = list(queries)
        
        # Optionally add golden dataset queries
        try:
            golden_dataset = create_low_scoring_golden_dataset()
            if golden_dataset:
                for query, data in golden_dataset.items():
                    all_queries.append({
                        "query": query,
                        "expected_videos": data["expected_videos"],
                        "category": "challenging"
                    })
        except Exception as e:
            logger.warning(f"Could not load golden dataset: {e}")
        
        # Create dataframe with all necessary fields
        df_data = []
        for q in all_queries:
            # For queries from VISUAL_TEST_QUERIES, add some expected videos
            expected_videos = q.get("expected_videos", [])
            if not expected_videos and isinstance(q, dict):
                # Add some default expected videos for testing
                # In real usage, these would come from your test data
                expected_videos = ["video1", "video2"]  # Placeholder
                
            df_data.append({
                "query": q["query"] if isinstance(q, dict) else q,
                "expected_videos": expected_videos,
                "category": q.get("category", "general") if isinstance(q, dict) else "general",
                # Add expected results as a field for golden evaluation
                "expected_results_json": str(expected_videos)
            })
        
        df = pd.DataFrame(df_data)
        
        # Upload to Phoenix
        dataset = self.client.upload_dataset(
            dataset_name=f"video_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataframe=df,
            input_keys=["query"],
            output_keys=["expected_videos"],
            metadata_keys=["category", "expected_results_json"]
        )
        
        logger.info(f"Created temporary dataset with {len(df)} queries")
        return dataset
    
    def create_retrieval_task(self, profile: str, ranking_strategy: str) -> Callable:
        """
        Create a retrieval task that includes proper scoring
        """
        config = self.config
        
        def retrieval_task(example):
            """Task that performs retrieval and formats results properly"""
            # Get query from example
            query = example.input.get("query", "") if hasattr(example, 'input') else example.get("query", "")
            
            # Get expected results if available
            expected_videos = []
            if hasattr(example, 'expected_videos'):
                expected_videos = example.expected_videos
            elif hasattr(example, 'metadata') and 'expected_videos' in example.metadata:
                expected_videos = example.metadata['expected_videos']
            
            # Log task execution and check Phoenix context
            logger.info(f"Executing retrieval task for query: {query}, profile: {profile}, strategy: {ranking_strategy}")
            
            # Check if Phoenix set any project context
            phoenix_project = os.environ.get("PHOENIX_PROJECT_NAME")
            if phoenix_project:
                logger.info(f"Phoenix project context: {phoenix_project}")
            
            # Create tracer for this task
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(f"retrieval_task_{profile}_{ranking_strategy}") as span:
                span.set_attributes({
                    "experiment.profile": profile,
                    "experiment.strategy": ranking_strategy,
                    "experiment.query": query,
                    "experiment.expected_count": len(expected_videos)
                })
                
                try:
                    # Import here to ensure it's in the right context
                    from src.search.search_service import SearchService
                    
                    # Log to confirm we're using the SearchService
                    logger.info(f"Creating SearchService with profile: {profile}")
                    
                    # Create search service
                    search_service = SearchService(config, profile)
                    
                    logger.info(f"SearchService created, instrumentation status: {hasattr(search_service.search, '__wrapped__')}")
                    
                    # Run search
                    search_results_raw = search_service.search(
                        query=query,
                        top_k=10,
                        ranking_strategy=ranking_strategy
                    )
                    
                    # Convert to proper format with scores
                    search_results = []
                    for i, result in enumerate(search_results_raw):
                        result_dict = result.to_dict()
                        
                        # Extract video_id properly
                        video_id = result_dict.get('source_id', '')
                        if not video_id and 'document_id' in result_dict:
                            # Extract from document_id (e.g., "video_frame_0" -> "video")
                            doc_id = result_dict['document_id']
                            if '_frame_' in doc_id:
                                video_id = doc_id.split('_frame_')[0]
                            else:
                                video_id = doc_id
                        
                        # Get score (ensure it's not 0)
                        score = result_dict.get('score', 0.0)
                        if score == 0.0:
                            # Use rank-based score if no score available
                            score = 1.0 / (i + 1)
                        
                        search_results.append({
                            "video_id": video_id,
                            "score": float(score),
                            "rank": i + 1,
                            "document_id": result_dict.get('document_id', ''),
                            # Include temporal info if available
                            "temporal_info": result_dict.get('temporal_info', {})
                        })
                    
                    # Add result attributes to span
                    span.set_attributes({
                        "experiment.num_results": len(search_results),
                        "experiment.top_video": search_results[0]["video_id"] if search_results else "",
                        "experiment.top_score": search_results[0]["score"] if search_results else 0.0
                    })
                    
                    # Calculate quick metrics if we have expected results
                    if expected_videos:
                        retrieved_videos = [r["video_id"] for r in search_results]
                        hits = [v for v in retrieved_videos[:5] if v in expected_videos]
                        precision_at_5 = len(hits) / 5 if len(retrieved_videos) >= 5 else len(hits) / len(retrieved_videos)
                        
                        span.set_attributes({
                            "experiment.precision_at_5": precision_at_5,
                            "experiment.hits_at_5": len(hits)
                        })
                    
                    return {
                        "results": search_results,
                        "profile": profile,
                        "ranking_strategy": ranking_strategy,
                        "num_results": len(search_results),
                        "query": query
                    }
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    
                    logger.error(f"Retrieval failed for {profile}/{ranking_strategy}: {e}")
                    return {
                        "results": [],
                        "profile": profile,
                        "ranking_strategy": ranking_strategy,
                        "num_results": 0,
                        "error": str(e),
                        "query": query
                    }
        
        return retrieval_task
    
    def run_experiment(
        self,
        profile: str,
        strategy: str,
        dataset: Any,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Run a single experiment for a profile+strategy combination
        
        Args:
            profile: Video processing profile
            strategy: Ranking strategy
            dataset: Phoenix dataset
            description: Human-readable description
            
        Returns:
            Experiment result
        """
        experiment_name = f"{profile}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Running experiment: {description or experiment_name}")
        
        try:
            # Create task
            task = self.create_retrieval_task(profile, strategy)
            
            # Get evaluators
            evaluators = []
            
            # Add quality evaluators if enabled
            if self.enable_quality_evaluators:
                from .evaluators.sync_reference_free import create_quality_evaluators
                evaluators = create_quality_evaluators()
                logger.info(f"Added {len(evaluators)} quality evaluators: relevance, diversity, distribution, temporal coverage")
            else:
                # Use basic evaluators only
                evaluators = create_sync_evaluators()
                logger.info("Using basic evaluators only (relevance and diversity)")
            
            # Add LLM evaluators if enabled
            if self.enable_llm_evaluators:
                from .evaluators.llm_judge import create_llm_evaluators
                llm_evaluators = create_llm_evaluators(
                    model_name=self.llm_model,
                    base_url=self.llm_base_url,
                    include_hybrid=True
                )
                evaluators.extend(llm_evaluators)
                logger.info(f"Added {len(llm_evaluators)} LLM evaluators: reference-free, reference-based, hybrid")
            
            # Add golden dataset evaluator with the actual queries from the dataset
            from .evaluators.sync_golden_dataset import SyncGoldenDatasetEvaluator
            
            # Create golden dataset from the actual dataset being used
            golden_data = {}
            try:
                # Get examples from dataset - dataset.examples is a property, not a method
                examples = dataset.examples if hasattr(dataset, 'examples') else []
                for example in examples:
                    if hasattr(example, 'output') and hasattr(example.output, 'expected_videos'):
                        golden_data[example.input['query']] = {
                            "expected_videos": example.output.expected_videos,
                            "relevance_scores": {}
                        }
            except Exception as e:
                logger.warning(f"Could not extract golden data from dataset: {e}")
                golden_data = {}
            
            golden_evaluator = SyncGoldenDatasetEvaluator(golden_data)
            evaluators.append(golden_evaluator)
            
            # Run experiment
            logger.info(f"Starting Phoenix experiment run for {experiment_name}")
            result = run_experiment(
                dataset=dataset,
                task=task,
                evaluators=evaluators,
                experiment_name=experiment_name,
                experiment_metadata={
                    "profile": profile,
                    "ranking_strategy": strategy,
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "project": self.experiment_project
                }
            )
            logger.info(f"Phoenix experiment completed: {experiment_name}")
            
            return {
                "status": "success",
                "experiment_name": experiment_name,
                "result": result,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                "status": "failed",
                "experiment_name": experiment_name,
                "error": str(e),
                "description": description
            }
    
    def generate_experiment_report(self, experiments: List[Dict]) -> pd.DataFrame:
        """
        Generate a report similar to comprehensive test visualization
        
        Args:
            experiments: List of experiment results
            
        Returns:
            DataFrame with experiment comparison
        """
        rows = []
        
        for exp in experiments:
            if exp["status"] == "success":
                # Extract metrics from experiment result
                # This would need to parse the actual Phoenix experiment result
                rows.append({
                    "Experiment": exp["description"],
                    "Profile": exp.get("profile", ""),
                    "Strategy": exp.get("strategy", ""),
                    "Status": "✅ Success",
                    "Avg Score": "View in Phoenix",
                    "Precision@5": "View in Phoenix",
                    "URL": f"http://localhost:6006/projects/{self.experiment_project}/experiments"
                })
            else:
                rows.append({
                    "Experiment": exp["description"],
                    "Profile": exp.get("profile", ""),
                    "Strategy": exp.get("strategy", ""),
                    "Status": "❌ Failed",
                    "Avg Score": "-",
                    "Precision@5": "-",
                    "URL": "-"
                })
        
        return pd.DataFrame(rows)
    
    def cleanup(self):
        """Restore original project configuration"""
        # No longer needed since we're not changing environment
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original project"""
        self.cleanup()
        return False