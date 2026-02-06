#!/usr/bin/env python3
"""
Run Phoenix experiments with visualization.

This script delegates to cogniverse_evaluation.core.experiment_tracker.ExperimentTracker
which provides the full experiment running and visualization functionality.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Run experiments with visualization using ExperimentTracker."""
    parser = argparse.ArgumentParser(
        description="Run Phoenix experiments with visualization"
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="Name of dataset to use (if exists) or create (if new)",
    )
    parser.add_argument(
        "--dataset-path",
        "--csv-path",
        dest="csv_path",
        help="Path to CSV file with queries (required for new datasets)",
    )
    parser.add_argument(
        "--force-new",
        dest="force_new",
        action="store_true",
        help="Force creation of new dataset even if name exists",
    )
    parser.add_argument(
        "--all-strategies",
        dest="all_strategies",
        action="store_true",
        help="Run all available strategies (default: only common ones)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Specific profiles to test (default: all)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Specific strategies to test (default: filtered list)",
    )
    parser.add_argument(
        "--quality-evaluators",
        dest="quality_evaluators",
        action="store_true",
        default=True,
        help="Enable additional quality evaluators (default: True)",
    )
    parser.add_argument(
        "--no-quality-evaluators",
        dest="quality_evaluators",
        action="store_false",
        help="Disable additional quality evaluators",
    )
    parser.add_argument(
        "--llm-evaluators",
        dest="llm_evaluators",
        action="store_true",
        default=False,
        help="Enable LLM-based evaluators",
    )
    parser.add_argument(
        "--evaluator",
        default="visual_judge",
        help="Evaluator config to use (visual_judge, llm_judge, modal_visual_judge)",
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help="Override LLM model",
    )
    parser.add_argument(
        "--llm-base-url",
        dest="llm_base_url",
        default=None,
        help="Override LLM base URL",
    )
    # Legacy argument for backwards compatibility
    parser.add_argument(
        "--test-multiple-strategies",
        action="store_true",
        help="(Deprecated) Use --all-strategies instead",
    )

    args = parser.parse_args()

    # Handle legacy argument
    if args.test_multiple_strategies:
        args.all_strategies = True

    # Import and run ExperimentTracker
    from cogniverse_evaluation.core.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(
        experiment_project_name="experiments",
        enable_quality_evaluators=args.quality_evaluators,
        enable_llm_evaluators=args.llm_evaluators,
        evaluator_name=args.evaluator,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )

    # Get configurations
    tracker.get_experiment_configurations(
        profiles=args.profiles,
        strategies=args.strategies,
        all_strategies=args.all_strategies,
    )

    # Create or get dataset
    dataset_name = tracker.create_or_get_dataset(
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
        force_new=args.force_new,
    )

    # Run all experiments
    experiments = tracker.run_all_experiments(dataset_name)

    # Create and print visualization
    tables = tracker.create_visualization_tables()
    tracker.print_visualization(tables)

    # Save results
    tracker.save_results(tables, experiments)

    # Generate HTML report if available
    tracker.generate_html_report()

    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()
