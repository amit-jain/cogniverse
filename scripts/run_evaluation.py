#!/usr/bin/env python3
"""
Run comprehensive evaluation of the Cogniverse video RAG system

This script provides a CLI interface to run evaluations using:
- Inspect AI for structured task-based evaluation
- Phoenix for tracing, datasets, and experiments
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.pipeline.orchestrator import EvaluationPipeline
from src.evaluation.phoenix.instrumentation import instrument_cogniverse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_full_evaluation(
    name: str,
    profiles: List[str],
    strategies: List[str],
    tasks: List[str],
    config_path: Optional[str] = None,
    dataset_path: Optional[str] = None
):
    """Run full evaluation pipeline"""
    logger.info(f"Starting evaluation: {name}")
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(config_path)
    
    # Run comprehensive evaluation
    results = await pipeline.run_comprehensive_evaluation(
        evaluation_name=name,
        profiles=profiles,
        strategies=strategies,
        tasks=tasks,
        dataset_path=dataset_path,
        use_phoenix_dataset=True
    )
    
    # Export reports
    html_report = pipeline.export_evaluation_report(format="html")
    md_report = pipeline.export_evaluation_report(format="markdown")
    
    logger.info(f"Evaluation complete!")
    logger.info(f"HTML report: {html_report}")
    logger.info(f"Markdown report: {md_report}")
    
    # Print summary
    if "report" in results:
        report = results["report"]
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if "summary" in report:
            summary = report["summary"]
            print(f"Name: {summary.get('evaluation_name', 'N/A')}")
            print(f"Profiles tested: {summary.get('profiles_tested', 0)}")
            print(f"Strategies tested: {summary.get('strategies_tested', 0)}")
            print(f"Tasks completed: {summary.get('tasks_completed', 0)}")
        
        if "phoenix_experiment" in report:
            phoenix = report["phoenix_experiment"]
            if phoenix.get("best_configuration"):
                best = phoenix["best_configuration"]
                print(f"\nBest Configuration:")
                print(f"  Profile: {best.get('profile', 'N/A')}")
                print(f"  Strategy: {best.get('strategy', 'N/A')}")
                print(f"  MRR: {best.get('mrr_mean', 0):.3f}")
        
        if "recommendations" in report:
            print(f"\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
    
    return results


async def run_batch_evaluation(
    dataset_name: str,
    trace_ids: Optional[List[str]] = None,
    config_path: Optional[str] = None
):
    """Run batch evaluation on existing traces"""
    logger.info(f"Starting batch evaluation with dataset: {dataset_name}")
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(config_path)
    
    # Run batch evaluation
    results = await pipeline.run_batch_evaluation_on_traces(
        trace_ids=trace_ids,
        dataset_name=dataset_name
    )
    
    logger.info(f"Batch evaluation complete!")
    
    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Traces evaluated: {summary.get('num_evaluated', 0)}")
        print(f"Mean MRR: {summary.get('mean_mrr', 0):.3f}")
        print(f"Mean NDCG: {summary.get('mean_ndcg', 0):.3f}")
        print(f"Success rate: {summary.get('success_rate', 0):.2%}")
    
    return results


async def run_quick_test(config_path: Optional[str] = None):
    """Run a quick test evaluation with minimal configuration"""
    logger.info("Running quick test evaluation")
    
    # Use minimal configuration for quick test
    profiles = ["frame_based_colpali"]
    strategies = ["binary_binary"]
    tasks = ["video_retrieval_accuracy"]
    
    results = await run_full_evaluation(
        name="quick_test",
        profiles=profiles,
        strategies=strategies,
        tasks=tasks,
        config_path=config_path
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation of Cogniverse video RAG system"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Evaluation commands")
    
    # Full evaluation command
    full_parser = subparsers.add_parser("full", help="Run full evaluation")
    full_parser.add_argument("--name", required=True, help="Evaluation name")
    full_parser.add_argument(
        "--profiles",
        nargs="+",
        default=["frame_based_colpali", "direct_video_global"],
        help="Video processing profiles to test"
    )
    full_parser.add_argument(
        "--strategies",
        nargs="+",
        default=["float_float", "binary_binary", "hybrid_binary_bm25"],
        help="Ranking strategies to test"
    )
    full_parser.add_argument(
        "--tasks",
        nargs="+",
        default=["video_retrieval_accuracy"],
        help="Inspect AI tasks to run"
    )
    full_parser.add_argument(
        "--config",
        help="Path to evaluation configuration file"
    )
    full_parser.add_argument(
        "--dataset",
        help="Path to evaluation dataset"
    )
    
    # Batch evaluation command
    batch_parser = subparsers.add_parser("batch", help="Run batch evaluation on traces")
    batch_parser.add_argument("--dataset", required=True, help="Dataset name in Phoenix")
    batch_parser.add_argument(
        "--trace-ids",
        nargs="+",
        help="Specific trace IDs to evaluate"
    )
    batch_parser.add_argument(
        "--config",
        help="Path to evaluation configuration file"
    )
    
    # Quick test command
    test_parser = subparsers.add_parser("test", help="Run quick test evaluation")
    test_parser.add_argument(
        "--config",
        help="Path to evaluation configuration file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Enable Phoenix instrumentation
    logger.info("Enabling Phoenix instrumentation")
    instrument_cogniverse()
    
    # Run appropriate command
    try:
        if args.command == "full":
            results = asyncio.run(run_full_evaluation(
                name=args.name,
                profiles=args.profiles,
                strategies=args.strategies,
                tasks=args.tasks,
                config_path=args.config,
                dataset_path=args.dataset
            ))
        elif args.command == "batch":
            results = asyncio.run(run_batch_evaluation(
                dataset_name=args.dataset,
                trace_ids=args.trace_ids,
                config_path=args.config
            ))
        elif args.command == "test":
            results = asyncio.run(run_quick_test(
                config_path=args.config
            ))
        else:
            parser.print_help()
            sys.exit(1)
        
        logger.info("Evaluation completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()