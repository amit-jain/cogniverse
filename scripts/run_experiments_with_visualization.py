#!/usr/bin/env python3
"""
Run Phoenix experiments with visualization similar to comprehensive tests
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.phoenix_experiments_final import PhoenixExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_experiment_configurations(args=None):
    """
    Get experiment configurations from strategy registry.
    This ensures we use the same profiles/strategies as defined in the system.
    """
    from src.core.strategy_registry import get_registry
    
    registry = get_registry()
    configurations = []
    
    # Get profiles to test
    if args and args.profiles:
        profiles = args.profiles
    else:
        profiles = registry.list_profiles()
    
    # Common strategy descriptions
    strategy_descriptions = {
        "binary_binary": "Binary",
        "float_float": "Float", 
        "float_binary": "Float-Binary",
        "phased": "Phased",
        "hybrid_binary_bm25": "Hybrid + Desc",
        "hybrid_binary_bm25_no_description": "Hybrid (No Desc)",
        "bm25_only": "Text Only",
        "hybrid_float_bm25": "Hybrid Float + Text",
        "hybrid_float_bm25_no_description": "Hybrid Float (No Desc)"
    }
    
    # Default common strategies if not running all
    common_strategies = [
        "binary_binary", "float_float", "float_binary", 
        "phased", "hybrid_binary_bm25", "hybrid_binary_bm25_no_description",
        "bm25_only"
    ]
    
    for profile in profiles:
        try:
            # Get ranking strategies for this profile
            all_strategies = registry.list_ranking_strategies(profile)
            
            # Filter strategies based on arguments
            if args and args.strategies:
                # Use specific strategies requested
                strategies_to_use = [s for s in all_strategies if s in args.strategies]
            elif args and args.all_strategies:
                # Use all available strategies
                strategies_to_use = all_strategies
            else:
                # Use only common strategies
                strategies_to_use = [s for s in all_strategies if s in common_strategies]
            
            # Build strategy list with descriptions
            experiment_strategies = []
            for strategy in strategies_to_use:
                description = strategy_descriptions.get(strategy, strategy.replace("_", " ").title())
                experiment_strategies.append((strategy, description))
            
            if experiment_strategies:
                configurations.append({
                    "profile": profile,
                    "strategies": experiment_strategies
                })
                
        except Exception as e:
            logger.warning(f"Could not get strategies for profile {profile}: {e}")
    
    return configurations

# Note: EXPERIMENT_CONFIGURATIONS will be set in main() with args


def create_visualization_tables(experiments: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Create visualization tables similar to comprehensive test output
    
    Returns multiple DataFrames for different views
    """
    # 1. Summary table by profile
    profile_summary = []
    profiles = {}
    
    for exp in experiments:
        if exp["status"] == "success":
            profile = exp.get("profile", "unknown")
            if profile not in profiles:
                profiles[profile] = {"total": 0, "success": 0, "failed": 0}
            profiles[profile]["total"] += 1
            profiles[profile]["success"] += 1
        else:
            profile = exp.get("profile", "unknown")
            if profile not in profiles:
                profiles[profile] = {"total": 0, "success": 0, "failed": 0}
            profiles[profile]["total"] += 1
            profiles[profile]["failed"] += 1
    
    for profile, stats in profiles.items():
        profile_summary.append({
            "Profile": profile,
            "Total": stats["total"],
            "Success": stats["success"],
            "Failed": stats["failed"],
            "Success Rate": f"{(stats['success']/stats['total']*100):.1f}%"
        })
    
    # 2. Detailed experiment results
    detailed_results = []
    for exp in experiments:
        detailed_results.append({
            "Profile": exp.get("profile", ""),
            "Strategy": exp.get("strategy", ""),
            "Description": exp.get("description", ""),
            "Status": "✅" if exp["status"] == "success" else "❌",
            "Experiment Name": exp.get("experiment_name", "")
        })
    
    # 3. Strategy comparison (grouped by profile)
    strategy_comparison = []
    for config in EXPERIMENT_CONFIGURATIONS:
        profile = config["profile"]
        for strategy, desc in config["strategies"]:
            # Find matching experiment
            matching = [e for e in experiments 
                       if e.get("profile") == profile and e.get("strategy") == strategy]
            
            if matching:
                exp = matching[0]
                strategy_comparison.append({
                    "Profile": profile,
                    "Strategy": strategy,
                    "Description": desc,
                    "Status": "✅ Success" if exp["status"] == "success" else "❌ Failed"
                })
            else:
                strategy_comparison.append({
                    "Profile": profile,
                    "Strategy": strategy,
                    "Description": desc,
                    "Status": "⏭️ Skipped"
                })
    
    return {
        "profile_summary": pd.DataFrame(profile_summary),
        "detailed_results": pd.DataFrame(detailed_results),
        "strategy_comparison": pd.DataFrame(strategy_comparison)
    }


def main(args=None):
    """Run experiments with visualization"""
    
    # Get experiment configurations with args
    global EXPERIMENT_CONFIGURATIONS
    EXPERIMENT_CONFIGURATIONS = get_experiment_configurations(args)
    
    print("=" * 80)
    print("PHOENIX EXPERIMENTS WITH VISUALIZATION")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment Project: experiments (separate from default traces)\n")
    
    # Use context manager to ensure project is restored after experiments
    with PhoenixExperimentRunner(experiment_project_name="experiments") as runner:
        # List datasets if requested
        if args and args.list_datasets:
            print("\nAvailable datasets:")
            datasets = runner.dataset_manager.list_datasets()
            if datasets:
                for ds in datasets:
                    print(f"  - {ds['name']} (ID: {ds['phoenix_id']}, Examples: {ds['num_examples']})")
                    if ds['description']:
                        print(f"    Description: {ds['description']}")
            else:
                print("  No datasets registered yet")
            return
        
        # Create or get dataset
        print("Preparing experiment dataset...")
        dataset = runner.create_experiment_dataset(
            dataset_name=args.dataset_name if args else None,
            csv_path=args.csv_path if args else None,
            force_new=args.force_new if args else False
        )
        dataset_url = f"http://localhost:6006/datasets/{getattr(dataset, 'id', 'latest')}"
        print(f"✅ Dataset ready: {dataset_url}\n")
        
        # Run all experiments
        all_experiments = []
        total_experiments = sum(len(config["strategies"]) for config in EXPERIMENT_CONFIGURATIONS)
        experiment_count = 0
        
        for config in EXPERIMENT_CONFIGURATIONS:
            profile = config["profile"]
            print(f"\n{'='*60}")
            print(f"Profile: {profile}")
            print(f"{'='*60}")
            
            for strategy, description in config["strategies"]:
                experiment_count += 1
                full_description = f"{profile.replace('_', ' ').title()} - {description}"
                
                print(f"\n[{experiment_count}/{total_experiments}] {full_description}")
                print(f"  Strategy: {strategy}")
                
                # Run experiment
                result = runner.run_experiment(
                    profile=profile,
                    strategy=strategy,
                    dataset=dataset,
                    description=full_description
                )
                
                # Add profile and strategy to result
                result["profile"] = profile
                result["strategy"] = strategy
                result["description"] = full_description
                
                all_experiments.append(result)
                
                if result["status"] == "success":
                    print(f"  ✅ Success")
                else:
                    error = result.get("error", "Unknown error")
                    if "Text encoder not available" in error:
                        print(f"  ⚠️  Skipped: Encoder not available")
                    else:
                        print(f"  ❌ Failed: {error[:50]}...")
    
    # Generate visualization tables
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS VISUALIZATION")
    print("="*80)
    
    tables = create_visualization_tables(all_experiments)
    
    # 1. Profile Summary
    print("\n📊 PROFILE SUMMARY")
    print("-" * 60)
    print(tabulate(tables["profile_summary"], headers='keys', tablefmt='grid', showindex=False))
    
    # 2. Strategy Comparison
    print("\n🔍 STRATEGY COMPARISON BY PROFILE")
    print("-" * 60)
    
    # Group by profile for better visualization
    strategy_df = tables["strategy_comparison"]
    for profile in strategy_df["Profile"].unique():
        profile_strategies = strategy_df[strategy_df["Profile"] == profile]
        print(f"\n{profile}:")
        print(tabulate(profile_strategies[["Strategy", "Description", "Status"]], 
                      headers='keys', tablefmt='simple', showindex=False))
    
    # 3. Detailed Results (first 10)
    print("\n📋 DETAILED EXPERIMENT RESULTS (First 10)")
    print("-" * 60)
    print(tabulate(tables["detailed_results"].head(10), 
                  headers='keys', tablefmt='grid', showindex=False))
    
    # 4. Summary Statistics
    successful = len([e for e in all_experiments if e["status"] == "success"])
    failed = len([e for e in all_experiments if e["status"] == "failed"])
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal Experiments Attempted: {len(all_experiments)}")
    print(f"Successful: {successful} ({successful/len(all_experiments)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(all_experiments)*100:.1f}%)")
    
    # 5. Phoenix UI Links
    print("\n" + "="*80)
    print("VIEW IN PHOENIX UI")
    print("="*80)
    print(f"\n🔗 Dataset: {dataset_url}")
    print(f"🔗 Experiments Project: http://localhost:6006/projects/experiments")
    print(f"🔗 Default Project (spans): http://localhost:6006/projects/default")
    
    print("\nℹ️  Notes:")
    print("  - Experiments are in separate 'experiments' project")
    print("  - Each experiment has its own traces with detailed spans")
    print("  - Use Phoenix UI to compare experiments side-by-side")
    print("  - Evaluation scores are attached to each experiment")
    
    # Save results to file
    output_dir = Path(__file__).parent.parent / "outputs" / "experiment_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary as CSV
    summary_path = output_dir / f"experiment_summary_{timestamp}.csv"
    tables["strategy_comparison"].to_csv(summary_path, index=False)
    print(f"\n💾 Results saved to: {summary_path}")
    
    # Save detailed results as JSON
    import json
    json_path = output_dir / f"experiment_details_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "dataset_url": dataset_url,
            "experiments": all_experiments,
            "summary": {
                "total": len(all_experiments),
                "successful": successful,
                "failed": failed
            }
        }, f, indent=2)
    print(f"💾 Detailed results saved to: {json_path}")
    
    # Generate integrated HTML report if quantitative results exist
    try:
        from scripts.generate_integrated_evaluation_report import generate_integrated_report
        
        print("\n📊 Generating integrated HTML report...")
        html_report = generate_integrated_report(
            experiment_results_dir=output_dir.parent
        )
        print(f"📄 HTML report saved to: {html_report}")
    except Exception as e:
        print(f"⚠️  Could not generate integrated report: {e}")
        print("   Run quantitative tests first for integrated view")
    
    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phoenix experiments")
    parser.add_argument("--dataset-name", 
                       help="Name of dataset to use (if exists) or create (if new)")
    parser.add_argument("--csv-path", 
                       help="Path to CSV file with queries (required for new datasets)")
    parser.add_argument("--force-new", action="store_true", 
                       help="Force creation of new dataset even if name exists")
    parser.add_argument("--list-datasets", action="store_true", 
                       help="List available datasets and exit")
    parser.add_argument("--all-strategies", action="store_true",
                       help="Run all available strategies (default: only common ones)")
    parser.add_argument("--profiles", nargs="+",
                       help="Specific profiles to test (default: all)")
    parser.add_argument("--strategies", nargs="+",
                       help="Specific strategies to test (default: filtered list)")
    args = parser.parse_args()
    
    # Pass args to main
    main(args)