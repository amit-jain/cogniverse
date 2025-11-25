"""
Experiment Tracker - Using the New Evaluation Framework

This module provides functionality similar to run_experiments_with_visualization.py
but uses the new Inspect AI-based evaluation framework with proper Phoenix integration.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tabulate import tabulate

from cogniverse_evaluation.core.task import evaluation_task
from cogniverse_evaluation.plugins import register_plugin
from cogniverse_evaluation.providers import get_evaluation_provider
from cogniverse_evaluation.providers.base import EvaluationProvider

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks and visualizes experiments using the new Inspect AI evaluation framework.

    This is fully compatible with run_experiments_with_visualization.py but uses:
    - Inspect AI for evaluation execution
    - Plugin system for evaluators
    - Proper Phoenix integration
    - New evaluation task framework
    """

    def __init__(
        self,
        experiment_project_name: str = "experiments",
        output_dir: Path | None = None,
        enable_quality_evaluators: bool = True,
        enable_llm_evaluators: bool = False,
        evaluator_name: str = "visual_judge",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        evaluation_provider: Optional[EvaluationProvider] = None,
        tenant_id: str = "default",
    ):
        """
        Initialize the experiment tracker with provider-agnostic evaluation framework.

        Args:
            experiment_project_name: Project name for experiments
            output_dir: Directory to save results
            enable_quality_evaluators: Enable quality metrics
            enable_llm_evaluators: Enable LLM-based evaluators
            evaluator_name: Name of the evaluator to use
            llm_model: LLM model for evaluators
            llm_base_url: Base URL for LLM API
            evaluation_provider: Optional evaluation provider (auto-detects if None)
            tenant_id: Tenant identifier for multi-tenancy
        """
        self.experiment_project_name = experiment_project_name
        self.output_dir = output_dir or Path("outputs/experiment_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_quality_evaluators = enable_quality_evaluators
        self.enable_llm_evaluators = enable_llm_evaluators
        self.evaluator_name = evaluator_name
        self.llm_model = llm_model or "deepseek-r1:7b"
        self.llm_base_url = llm_base_url

        # Initialize provider (auto-detect if not provided)
        self.provider = evaluation_provider or get_evaluation_provider(
            tenant_id=tenant_id
        )
        self.tenant_id = tenant_id

        # Track experiments
        self.experiments: list[dict] = []
        self.configurations: list[dict] = []
        self.dataset_url: str | None = None

        # Register plugins if needed
        self._register_evaluator_plugins()

    def _register_evaluator_plugins(self):
        """Register evaluator plugins based on configuration."""
        if self.enable_quality_evaluators:
            # Register quality evaluator plugins
            try:
                from cogniverse_evaluation.plugins.video_analyzer import (
                    VideoAnalyzerPlugin,
                )

                register_plugin("video_analyzer", VideoAnalyzerPlugin())
                logger.info("Registered video analyzer plugin for quality metrics")
            except Exception as e:
                logger.warning(f"Could not register video analyzer plugin: {e}")

        if self.enable_llm_evaluators:
            # Register LLM evaluator plugins
            try:
                from cogniverse_evaluation.plugins.visual_evaluator import (
                    VisualEvaluatorPlugin,
                )

                plugin = VisualEvaluatorPlugin()
                register_plugin("visual_evaluator", plugin)
                logger.info(
                    f"Registered visual evaluator plugin with {self.evaluator_name}"
                )
            except Exception as e:
                logger.warning(f"Could not register visual evaluator plugin: {e}")

    def get_experiment_configurations(
        self,
        profiles: list[str] | None = None,
        strategies: list[str] | None = None,
        all_strategies: bool = False,
    ) -> list[dict]:
        """
        Get experiment configurations from strategy registry.

        Compatible with run_experiments_with_visualization.py
        """
        from cogniverse_core.registries.registry import get_registry

        registry = get_registry()
        configurations = []

        # Get profiles to test
        profiles_to_test = profiles or registry.list_profiles()

        # Strategy descriptions for visualization
        strategy_descriptions = {
            "binary_binary": "Binary",
            "float_float": "Float",
            "float_binary": "Float-Binary",
            "phased": "Phased",
            "hybrid_binary_bm25": "Hybrid + Desc",
            "hybrid_binary_bm25_no_description": "Hybrid (No Desc)",
            "bm25_only": "Text Only",
            "hybrid_float_bm25": "Hybrid Float + Text",
            "hybrid_float_bm25_no_description": "Hybrid Float (No Desc)",
        }

        # Default common strategies
        common_strategies = [
            "binary_binary",
            "float_float",
            "float_binary",
            "phased",
            "hybrid_binary_bm25",
            "hybrid_binary_bm25_no_description",
            "bm25_only",
        ]

        for profile in profiles_to_test:
            try:
                # Get ranking strategies for this profile
                available_strategies = registry.list_ranking_strategies(profile)

                # Filter strategies
                if strategies:
                    strategies_to_use = [
                        s for s in available_strategies if s in strategies
                    ]
                elif all_strategies:
                    strategies_to_use = available_strategies
                else:
                    strategies_to_use = [
                        s for s in available_strategies if s in common_strategies
                    ]

                # Build strategy list with descriptions
                experiment_strategies = []
                for strategy in strategies_to_use:
                    description = strategy_descriptions.get(
                        strategy, strategy.replace("_", " ").title()
                    )
                    experiment_strategies.append((strategy, description))

                if experiment_strategies:
                    configurations.append(
                        {"profile": profile, "strategies": experiment_strategies}
                    )

            except Exception as e:
                logger.warning(f"Could not get strategies for profile {profile}: {e}")

        self.configurations = configurations
        return configurations

    async def run_experiment_async(
        self, profile: str, strategy: str, dataset_name: str, description: str
    ) -> dict:
        """
        Run a single experiment using the new Inspect AI framework.

        This uses the evaluation_task we created with proper Phoenix integration.
        """
        try:
            # Log experiment start
            logger.info(f"Starting experiment: {profile}_{strategy}")
            self.provider.log_experiment_event(
                event_type="experiment_start",
                data={
                    "profile": profile,
                    "strategy": strategy,
                    "description": description,
                    "dataset": dataset_name,
                },
            )

            # Create evaluation task using our new framework
            task = evaluation_task(
                mode="experiment",
                dataset_name=dataset_name,
                profiles=[profile],
                strategies=[strategy],
                config={
                    "evaluation": {
                        "enable_quality_evaluators": self.enable_quality_evaluators,
                        "enable_llm_evaluators": self.enable_llm_evaluators,
                        "evaluator_name": self.evaluator_name,
                        "llm_model": self.llm_model,
                        "llm_base_url": self.llm_base_url,
                    }
                },
            )

            # Run evaluation using Inspect AI
            from inspect_ai import eval as inspect_eval

            result = await inspect_eval(
                task,
                model="openai/gpt-4",  # This is overridden by our solvers
                log_dir=self.output_dir / "logs",
            )

            # Extract metrics from result
            metrics = {}
            if result and hasattr(result, "scores"):
                for score_name, score_value in result.scores.items():
                    metrics[score_name] = (
                        score_value.value
                        if hasattr(score_value, "value")
                        else score_value
                    )

            # Record experiment completion
            self.provider.log_experiment_event(
                event_type="experiment_complete",
                data={
                    "profile": profile,
                    "strategy": strategy,
                    "mrr": metrics.get("mrr", 0),
                    "error": False,
                },
            )

            return {
                "status": "success",
                "profile": profile,
                "strategy": strategy,
                "description": description,
                "experiment_name": f"{profile}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "metrics": metrics,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Experiment failed for {profile}/{strategy}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "profile": profile,
                "strategy": strategy,
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }

    def run_experiment(
        self, profile: str, strategy: str, dataset_name: str, description: str
    ) -> dict:
        """
        Synchronous wrapper for run_experiment_async.

        Maintains compatibility with run_experiments_with_visualization.py
        """
        return asyncio.run(
            self.run_experiment_async(profile, strategy, dataset_name, description)
        )

    def create_or_get_dataset(
        self,
        dataset_name: str | None = None,
        csv_path: str | None = None,
        force_new: bool = False,
    ) -> str:
        """
        Create or retrieve a dataset for experiments.

        Returns:
            Dataset name to use for experiments
        """
        # Generate dataset name if not provided
        final_dataset_name = dataset_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if dataset_name and not force_new:
            # Provider will check if dataset exists
            logger.info(f"Using dataset: {final_dataset_name}")

        # Create new dataset via provider
        if csv_path:
            # Load data from CSV
            df = pd.read_csv(csv_path)
            data = df.to_dict('records')
            _ = self.provider.create_dataset(
                name=final_dataset_name,
                data=data,
                description="Experiment dataset",
            )
        else:
            # Create minimal test dataset
            test_data = [
                {"query": "test query 1", "expected": "test result 1"},
                {"query": "test query 2", "expected": "test result 2"},
            ]
            _ = self.provider.create_dataset(
                name=final_dataset_name,
                data=test_data,
                description="Test dataset",
            )

        # Get provider-specific dataset URL
        self.dataset_url = self.provider.get_dataset_url(final_dataset_name)
        logger.info(f"Dataset ready: {final_dataset_name}")
        return final_dataset_name

    def run_all_experiments(self, dataset_name: str) -> list[dict]:
        """
        Run all configured experiments using the new evaluation framework.

        Fully compatible with run_experiments_with_visualization.py output format.
        """
        if not self.configurations:
            raise ValueError(
                "No configurations set. Call get_experiment_configurations first."
            )

        all_experiments = []
        total_experiments = sum(
            len(config["strategies"]) for config in self.configurations
        )
        experiment_count = 0

        print(f"\n{'='*80}")
        print("PHOENIX EXPERIMENTS WITH VISUALIZATION")
        print(f"{'='*80}")
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment Project: {self.experiment_project_name}")
        print(f"Dataset: {dataset_name}")

        # Show evaluator status (same as original)
        if self.enable_quality_evaluators:
            print(
                "Quality Evaluators: ✅ ENABLED (relevance, diversity, distribution, temporal coverage)"
            )
        else:
            print("Quality Evaluators: ❌ DISABLED")

        if self.enable_llm_evaluators:
            print(f"LLM Evaluators: ✅ ENABLED (model: {self.llm_model})")
        else:
            print("LLM Evaluators: ❌ DISABLED")

        # Run experiments
        for config in self.configurations:
            profile = config["profile"]
            print(f"\n{'='*60}")
            print(f"Profile: {profile}")
            print(f"{'='*60}")

            for strategy, description in config["strategies"]:
                experiment_count += 1
                full_description = (
                    f"{profile.replace('_', ' ').title()} - {description}"
                )

                print(f"\n[{experiment_count}/{total_experiments}] {full_description}")
                print(f"  Strategy: {strategy}")

                # Run experiment using new framework
                result = self.run_experiment(
                    profile=profile,
                    strategy=strategy,
                    dataset_name=dataset_name,
                    description=full_description,
                )

                all_experiments.append(result)

                # Print status (same format as original)
                if result["status"] == "success":
                    print("  ✅ Success")
                    if result.get("metrics"):
                        for metric, value in result["metrics"].items():
                            print(f"     {metric}: {value:.3f}")
                else:
                    error = result.get("error", "Unknown error")
                    if "Text encoder not available" in error:
                        print("  ⚠️  Skipped: Encoder not available")
                    else:
                        print(f"  ❌ Failed: {error[:50]}...")

        self.experiments = all_experiments
        return all_experiments

    def create_visualization_tables(
        self,
        experiments: list[dict] | None = None,
        include_quality_metrics: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Create visualization tables - exact same format as run_experiments_with_visualization.py
        """
        experiments = experiments or self.experiments

        # 1. Summary table by profile
        profile_summary = []
        profiles = {}

        for exp in experiments:
            profile = exp.get("profile", "unknown")
            if profile not in profiles:
                profiles[profile] = {"total": 0, "success": 0, "failed": 0}

            profiles[profile]["total"] += 1
            if exp["status"] == "success":
                profiles[profile]["success"] += 1
            else:
                profiles[profile]["failed"] += 1

        for profile, stats in profiles.items():
            profile_summary.append(
                {
                    "Profile": profile,
                    "Total": stats["total"],
                    "Success": stats["success"],
                    "Failed": stats["failed"],
                    "Success Rate": (
                        f"{(stats['success']/stats['total']*100):.1f}%"
                        if stats["total"] > 0
                        else "0%"
                    ),
                }
            )

        # 2. Detailed experiment results
        detailed_results = []
        for exp in experiments:
            row = {
                "Profile": exp.get("profile", ""),
                "Strategy": exp.get("strategy", ""),
                "Description": exp.get("description", ""),
                "Status": "✅" if exp["status"] == "success" else "❌",
                "Experiment Name": exp.get("experiment_name", ""),
            }

            # Add quality metrics if available (from new framework)
            if include_quality_metrics and exp["status"] == "success":
                metrics = exp.get("metrics", {})
                if metrics:
                    # Add key metrics
                    if "relevance" in metrics:
                        row["Relevance"] = f"{metrics['relevance']:.3f}"
                    if "diversity" in metrics:
                        row["Diversity"] = f"{metrics['diversity']:.3f}"
                    if "mrr" in metrics:
                        row["MRR"] = f"{metrics['mrr']:.3f}"
                    if "recall" in metrics:
                        row["Recall@10"] = f"{metrics['recall']:.3f}"

            detailed_results.append(row)

        # 3. Strategy comparison (same as original)
        strategy_comparison = []
        for config in self.configurations:
            profile = config["profile"]
            for strategy, desc in config["strategies"]:
                matching = [
                    e
                    for e in experiments
                    if e.get("profile") == profile and e.get("strategy") == strategy
                ]

                if matching:
                    exp = matching[0]
                    status = "✅ Success" if exp["status"] == "success" else "❌ Failed"
                else:
                    status = "⏭️ Skipped"

                strategy_comparison.append(
                    {
                        "Profile": profile,
                        "Strategy": strategy,
                        "Description": desc,
                        "Status": status,
                    }
                )

        return {
            "profile_summary": pd.DataFrame(profile_summary),
            "detailed_results": pd.DataFrame(detailed_results),
            "strategy_comparison": pd.DataFrame(strategy_comparison),
        }

    def print_visualization(
        self,
        tables: dict[str, pd.DataFrame] | None = None,
        dataset_url: str | None = None,
    ):
        """
        Print visualization - exact same format as run_experiments_with_visualization.py
        """
        if tables is None:
            tables = self.create_visualization_tables()

        dataset_url = dataset_url or self.dataset_url

        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS VISUALIZATION")
        print("=" * 80)

        # 1. Profile Summary
        print("\n📊 PROFILE SUMMARY")
        print("-" * 60)
        print(
            tabulate(
                tables["profile_summary"],
                headers="keys",
                tablefmt="grid",
                showindex=False,
            )
        )

        # 2. Strategy Comparison
        print("\n🔍 STRATEGY COMPARISON BY PROFILE")
        print("-" * 60)

        strategy_df = tables["strategy_comparison"]
        for profile in strategy_df["Profile"].unique():
            profile_strategies = strategy_df[strategy_df["Profile"] == profile]
            print(f"\n{profile}:")
            print(
                tabulate(
                    profile_strategies[["Strategy", "Description", "Status"]],
                    headers="keys",
                    tablefmt="simple",
                    showindex=False,
                )
            )

        # 3. Detailed Results (same as original)
        print("\n📋 DETAILED EXPERIMENT RESULTS (First 10)")
        print("-" * 60)
        print(
            tabulate(
                tables["detailed_results"].head(10),
                headers="keys",
                tablefmt="grid",
                showindex=False,
            )
        )

        # 4. Summary Statistics
        successful = len([e for e in self.experiments if e["status"] == "success"])
        failed = len([e for e in self.experiments if e["status"] == "failed"])
        total = len(self.experiments)

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal Experiments Attempted: {total}")
        if total > 0:
            print(f"Successful: {successful} ({successful/total*100:.1f}%)")
            print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        # 5. Evaluation Provider UI Links
        print("\n" + "=" * 80)
        print("VIEW IN EVALUATION UI")
        print("=" * 80)
        if dataset_url:
            print(f"\n🔗 Dataset: {dataset_url}")

        # Get provider-specific experiment URL if available
        try:
            experiment_url = self.provider.get_experiment_url(self.experiment_project_name)
            print(f"🔗 Experiments Project: {experiment_url}")
        except (AttributeError, NotImplementedError):
            # Provider doesn't support experiment URLs
            logger.debug("Provider does not support experiment URLs")

        print("\nℹ️  Notes:")
        print("  - Experiments are tracked in separate project")
        print("  - Each experiment has its own traces with detailed spans")
        print("  - Use evaluation UI to compare experiments side-by-side")
        print("  - Evaluation scores are attached to each experiment")

    def save_results(
        self,
        tables: dict[str, pd.DataFrame] | None = None,
        experiments: list[dict] | None = None,
    ) -> tuple[Path, Path]:
        """
        Save experiment results - same format as original.
        """
        tables = tables or self.create_visualization_tables()
        experiments = experiments or self.experiments

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary as CSV (same as original)
        csv_path = self.output_dir / f"experiment_summary_{timestamp}.csv"
        tables["strategy_comparison"].to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")

        # Save detailed results as JSON (same format as original)
        json_path = self.output_dir / f"experiment_details_{timestamp}.json"

        successful = len([e for e in experiments if e["status"] == "success"])
        failed = len([e for e in experiments if e["status"] == "failed"])

        # Convert results to serializable format
        serializable_experiments = []
        for exp in experiments:
            exp_copy = exp.copy()
            # Remove non-serializable result object
            if "result" in exp_copy:
                exp_copy["result"] = str(exp_copy["result"])
            serializable_experiments.append(exp_copy)

        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "dataset_url": self.dataset_url,
                    "experiments": serializable_experiments,
                    "summary": {
                        "total": len(experiments),
                        "successful": successful,
                        "failed": failed,
                    },
                },
                f,
                indent=2,
                default=str,
            )

        print(f"💾 Detailed results saved to: {json_path}")

        return csv_path, json_path

    def generate_html_report(self) -> Path | None:
        """
        Generate integrated HTML report - same as original.
        """
        try:
            from scripts.generate_integrated_evaluation_report import (
                generate_integrated_report,
            )

            print("\n📊 Generating integrated HTML report...")
            html_report = generate_integrated_report(
                experiment_results_dir=self.output_dir.parent
            )
            print(f"📄 HTML report saved to: {html_report}")
            return html_report

        except Exception as e:
            logger.warning(f"Could not generate integrated report: {e}")
            print(f"⚠️  Could not generate integrated report: {e}")
            print("   Run quantitative tests first for integrated view")
            return None


def main(args=None):
    """
    Main function that mimics run_experiments_with_visualization.py
    but uses the new evaluation framework.
    """
    # Create tracker with new framework
    tracker = ExperimentTracker(
        experiment_project_name="experiments",
        enable_quality_evaluators=args.quality_evaluators if args else True,
        enable_llm_evaluators=args.llm_evaluators if args else False,
        evaluator_name=args.evaluator if args else "visual_judge",
        llm_model=args.llm_model if args else None,
        llm_base_url=args.llm_base_url if args else None,
    )

    # Get configurations (same as original)
    _ = tracker.get_experiment_configurations(
        profiles=args.profiles if args else None,
        strategies=args.strategies if args else None,
        all_strategies=args.all_strategies if args else False,
    )

    # Create or get dataset (same as original)
    dataset_name = tracker.create_or_get_dataset(
        dataset_name=args.dataset_name if args else None,
        csv_path=args.csv_path if args else None,
        force_new=args.force_new if args else False,
    )

    # Run all experiments using new framework
    experiments = tracker.run_all_experiments(dataset_name)

    # Create visualization tables (same as original)
    tables = tracker.create_visualization_tables()

    # Print visualization (same as original)
    tracker.print_visualization(tables)

    # Save results (same as original)
    tracker.save_results(tables, experiments)

    # Generate HTML report if available
    tracker.generate_html_report()

    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    # This can be called exactly like run_experiments_with_visualization.py
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experiments with new evaluation framework"
    )
    parser.add_argument("--dataset-name", help="Name of dataset")
    parser.add_argument("--csv-path", help="Path to CSV file")
    parser.add_argument("--profiles", nargs="+", help="Profiles to test")
    parser.add_argument("--strategies", nargs="+", help="Strategies to test")
    parser.add_argument(
        "--all-strategies", action="store_true", help="Test all strategies"
    )
    parser.add_argument(
        "--quality-evaluators", action="store_true", help="Enable quality evaluators"
    )
    parser.add_argument(
        "--llm-evaluators", action="store_true", help="Enable LLM evaluators"
    )
    parser.add_argument("--evaluator", default="visual_judge", help="Evaluator name")
    parser.add_argument("--llm-model", help="LLM model")
    parser.add_argument("--llm-base-url", help="LLM base URL")
    parser.add_argument("--force-new", action="store_true", help="Force new dataset")

    args = parser.parse_args()
    main(args)
