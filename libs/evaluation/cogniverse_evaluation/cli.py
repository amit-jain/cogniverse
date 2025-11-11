#!/usr/bin/env python3
"""
Unified CLI for evaluation framework.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from inspect_ai import eval as inspect_eval

from cogniverse_evaluation.core import evaluation_task
from cogniverse_evaluation.data import DatasetManager, TraceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """Cogniverse Evaluation Framework CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["experiment", "batch", "live"]),
    required=True,
    help="Evaluation mode",
)
@click.option("--dataset", required=True, help="Phoenix dataset name")
@click.option(
    "--profiles",
    "-p",
    multiple=True,
    help="Video processing profiles (for experiment mode)",
)
@click.option(
    "--strategies", "-s", multiple=True, help="Ranking strategies (for experiment mode)"
)
@click.option(
    "--trace-ids", "-t", multiple=True, help="Specific trace IDs (for batch mode)"
)
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def evaluate(mode, dataset, profiles, strategies, trace_ids, config, output, verbose):
    """
    Run evaluation in specified mode.

    Examples:
        # Run new experiment
        cogniverse-eval evaluate --mode experiment --dataset test_dataset \\
            -p frame_based_colpali -s binary_binary

        # Evaluate existing traces
        cogniverse-eval evaluate --mode batch --dataset test_dataset \\
            -t trace_id_1 -t trace_id_2

        # Live evaluation
        cogniverse-eval evaluate --mode live --dataset test_dataset
    """
    # Validate mode-specific requirements
    if mode == "experiment":
        if not profiles or not strategies:
            raise click.UsageError(
                "--profiles and --strategies required for experiment mode"
            )

    if mode == "batch" and not trace_ids:
        click.echo("Warning: No trace IDs provided, will evaluate recent traces")

    # Load configuration if provided
    eval_config = {}
    if config:
        with open(config) as f:
            if config.endswith(".json"):
                eval_config = json.load(f)
            else:
                import yaml

                eval_config = yaml.safe_load(f)

    click.echo(f"Starting {mode} evaluation with dataset '{dataset}'")

    try:
        # Create evaluation task
        task = evaluation_task(
            mode=mode,
            dataset_name=dataset,
            profiles=list(profiles) if profiles else None,
            strategies=list(strategies) if strategies else None,
            trace_ids=list(trace_ids) if trace_ids else None,
            config=eval_config,
        )

        # Run evaluation
        click.echo("Running evaluation...")
        results = inspect_eval(task)

        # Process results
        click.echo("\n" + "=" * 60)
        click.echo("EVALUATION RESULTS")
        click.echo("=" * 60)

        if results and hasattr(results, "samples"):
            for i, sample in enumerate(results.samples):
                click.echo(f"\nSample {i+1}:")
                click.echo(f"  Query: {sample.input.get('query', 'N/A')[:50]}...")

                if hasattr(sample, "scores"):
                    for scorer_name, score in sample.scores.items():
                        if score.value is not None:
                            status = "✓" if score.value > 0.5 else "✗"
                            click.echo(f"  {status} {scorer_name}: {score.value:.3f}")
                        else:
                            click.echo(f"  - {scorer_name}: N/A")

        # Save results if output specified
        if output:
            output_data = {
                "mode": mode,
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "results": [],
            }

            if results and hasattr(results, "samples"):
                for sample in results.samples:
                    sample_data = {"input": sample.input, "scores": {}}
                    if hasattr(sample, "scores"):
                        for scorer_name, score in sample.scores.items():
                            sample_data["scores"][scorer_name] = {
                                "value": score.value,
                                "explanation": score.explanation,
                            }
                    output_data["results"].append(sample_data)

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)

            click.echo(f"\nResults saved to {output}")

        click.echo("\n✓ Evaluation complete")

    except Exception as e:
        click.echo(f"✗ Evaluation failed: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--name", required=True, help="Dataset name")
@click.option("--csv", type=click.Path(exists=True), help="CSV file with queries")
@click.option(
    "--queries-json", type=click.Path(exists=True), help="JSON file with queries"
)
@click.option("--description", help="Dataset description")
def create_dataset(name, csv, queries_json, description):
    """
    Create a new dataset in Phoenix.

    Examples:
        # From CSV
        cogniverse-eval create-dataset --name my_dataset --csv queries.csv

        # From JSON
        cogniverse-eval create-dataset --name my_dataset --queries-json queries.json
    """
    if not csv and not queries_json:
        raise click.UsageError("Either --csv or --queries-json must be provided")

    dataset_manager = DatasetManager()

    try:
        if csv:
            click.echo(f"Creating dataset from CSV: {csv}")
            dataset_id = dataset_manager.create_from_csv(
                csv_path=csv, dataset_name=name, description=description
            )
        else:
            click.echo(f"Creating dataset from JSON: {queries_json}")
            with open(queries_json) as f:
                queries = json.load(f)
            dataset_id = dataset_manager.create_from_queries(
                queries=queries, dataset_name=name, description=description
            )

        click.echo(f"✓ Dataset '{name}' created with ID: {dataset_id}")

    except Exception as e:
        click.echo(f"✗ Failed to create dataset: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--hours", type=int, default=1, help="Hours to look back")
@click.option("--limit", type=int, default=100, help="Maximum number of traces")
def list_traces(hours, limit):
    """
    List recent traces from Phoenix.

    Example:
        cogniverse-eval list-traces --hours 2 --limit 50
    """
    trace_manager = TraceManager()

    try:
        click.echo(f"Fetching traces from last {hours} hours...")
        df = trace_manager.get_recent_traces(hours_back=hours, limit=limit)

        if df.empty:
            click.echo("No traces found")
            return

        traces = trace_manager.extract_trace_data(df)

        click.echo(f"\nFound {len(traces)} traces:")
        click.echo("-" * 80)

        for trace in traces[:10]:  # Show first 10
            click.echo(f"ID: {trace['trace_id'][:8]}...")
            click.echo(f"  Query: {trace['query'][:50]}...")
            click.echo(f"  Profile: {trace['profile']}, Strategy: {trace['strategy']}")
            click.echo(
                f"  Results: {len(trace.get('results', []))}, Duration: {trace['duration_ms']}ms"
            )
            click.echo("-" * 80)

        if len(traces) > 10:
            click.echo(f"... and {len(traces) - 10} more traces")

    except Exception as e:
        click.echo(f"✗ Failed to fetch traces: {e}", err=True)
        sys.exit(1)


@cli.command()
def test():
    """
    Run a quick test evaluation.
    """
    click.echo("Running test evaluation...")

    # Create test dataset
    dataset_manager = DatasetManager()

    try:
        _ = dataset_manager.create_test_dataset()
        dataset_name = f"test_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        click.echo(f"Created test dataset: {dataset_name}")

        # Run experiment mode test
        click.echo("\nTesting experiment mode...")
        task = evaluation_task(
            mode="experiment",
            dataset_name=dataset_name,
            profiles=["frame_based_colpali"],
            strategies=["binary_binary"],
            config={"use_ragas": False, "use_custom": True},
        )

        results = inspect_eval(task)

        if results:
            click.echo("✓ Experiment mode test passed")
        else:
            click.echo("✗ Experiment mode test failed")

        click.echo("\n✓ All tests complete")

    except Exception as e:
        click.echo(f"✗ Test failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
