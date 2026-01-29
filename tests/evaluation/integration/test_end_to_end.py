"""
End-to-end integration tests for evaluation framework.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from inspect_ai import eval as inspect_eval

from cogniverse_evaluation.cli import cli
from cogniverse_evaluation.core.task import evaluation_task


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_experiment_mode_e2e(self, mock_evaluator_provider, mock_search_service):
        """Test complete experiment mode workflow."""
        # Patch SearchService at the module level where it's imported
        with patch(
            "cogniverse_agents.search.service.SearchService",
            return_value=mock_search_service,
        ):
            # Patch get_config where it's imported
            with patch(
                "cogniverse_core.config.utils.get_config",
                return_value={"vespa_url": "http://localhost", "vespa_port": 8080},
            ):
                # Create evaluation task
                task = evaluation_task(
                    mode="experiment",
                    dataset_name="test_dataset",
                    profiles=["test_profile"],
                    strategies=["test_strategy"],
                    config={
                        "use_ragas": False,
                        "use_custom": True,
                        "custom_metrics": ["diversity", "result_count"],
                    },
                )

                # Run evaluation
                results = inspect_eval(task, model="mockllm/model")

                # Verify results structure - it returns a list of EvalLog
                assert results is not None
                assert isinstance(results, list)
                assert len(results) > 0

                # Get the first EvalLog
                eval_log = results[0]

                # Validate evaluation completed successfully
                assert (
                    eval_log.status == "success"
                ), f"Evaluation failed with status: {eval_log.status}"

                # Validate scores were generated
                assert eval_log.results is not None, "No results in eval log"
                assert eval_log.results.scores is not None, "No scores in results"
                assert len(eval_log.results.scores) > 0, "No scores generated"

                # Validate scorer names match configured scorers
                scorer_names = {score.name for score in eval_log.results.scores}
                expected_scorers = {
                    "diversity_scorer",
                    "result_count_scorer",
                }  # From config (custom_metrics)
                assert expected_scorers.issubset(
                    scorer_names
                ), f"Expected scorers {expected_scorers}, got {scorer_names}"

                # Validate score values are valid (0.0 to 1.0)
                for score in eval_log.results.scores:
                    # EvalScore has metrics dict, not direct value
                    assert (
                        score.metrics is not None
                    ), f"Scorer {score.name} has no metrics"
                    # Check mean metric (configured with @scorer(metrics=[mean()]))
                    assert (
                        "mean" in score.metrics
                    ), f"Scorer {score.name} missing 'mean' metric"
                    mean_value = score.metrics["mean"].value
                    assert (
                        mean_value is not None
                    ), f"Scorer {score.name} mean metric has no value"
                    assert (
                        0.0 <= mean_value <= 1.0
                    ), f"Scorer {score.name} mean {mean_value} not in [0, 1]"

                # Validate samples were processed
                assert eval_log.samples is not None, "No samples in eval log"
                assert len(eval_log.samples) > 0, "No samples processed"

                # Validate each sample has scores
                for sample in eval_log.samples:
                    assert (
                        sample.scores is not None
                    ), f"Sample {sample.id} has no scores"
                    assert (
                        len(sample.scores) > 0
                    ), f"Sample {sample.id} has empty scores"

    @pytest.mark.integration
    def test_batch_mode_e2e(self, mock_evaluator_provider):
        """Test complete batch mode workflow."""
        # Create evaluation task for batch mode
        task = evaluation_task(
            mode="batch",
            dataset_name="test_dataset",
            trace_ids=["trace1"],
            config={"use_ragas": True, "ragas_metrics": ["context_relevancy"]},
        )

        # Run evaluation
        results = inspect_eval(task, model="mockllm/model")

        # Verify results - it returns a list of EvalLog
        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.integration
    def test_cli_evaluate_command(self, mock_evaluator_provider, mock_search_service):
        """Test CLI evaluate command."""
        from click.testing import CliRunner

        with patch(
            "cogniverse_agents.search.service.SearchService",
            return_value=mock_search_service,
        ):
            runner = CliRunner()

            # Test experiment mode via CLI
            result = runner.invoke(
                cli,
                [
                    "evaluate",
                    "--mode",
                    "experiment",
                    "--dataset",
                    "test_dataset",
                    "-p",
                    "test_profile",
                    "-s",
                    "test_strategy",
                ],
            )

            # Check command executed successfully
            assert result.exit_code == 0
            assert "Starting experiment evaluation" in result.output
            assert (
                "Evaluation complete" in result.output
                or "Running evaluation" in result.output
            )

    @pytest.mark.integration
    def test_cli_list_traces_command(self, mock_evaluator_provider):
        """Test CLI list-traces command."""
        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(cli, ["list-traces", "--hours", "2", "--limit", "10"])

        assert result.exit_code == 0
        assert "Fetching traces" in result.output

    @pytest.mark.integration
    def test_evaluation_with_output_file(
        self, mock_evaluator_provider, mock_search_service
    ):
        """Test evaluation with output file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")

            from click.testing import CliRunner

            with patch(
                "cogniverse_agents.search.service.SearchService",
                return_value=mock_search_service,
            ):
                runner = CliRunner()

                result = runner.invoke(
                    cli,
                    [
                        "evaluate",
                        "--mode",
                        "experiment",
                        "--dataset",
                        "test_dataset",
                        "-p",
                        "test_profile",
                        "-s",
                        "test_strategy",
                        "--output",
                        output_file,
                    ],
                )

            # Check file was created
            if result.exit_code == 0:
                assert os.path.exists(output_file)

                # Verify JSON structure
                with open(output_file) as f:
                    data = json.load(f)
                    assert data["mode"] == "experiment"
                    assert data["dataset"] == "test_dataset"
                    assert "timestamp" in data

    @pytest.mark.integration
    def test_evaluation_with_config_file(
        self, mock_evaluator_provider, mock_search_service
    ):
        """Test evaluation with configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.json")

            # Create config file
            config = {
                "use_ragas": True,
                "ragas_metrics": ["context_relevancy"],
                "use_custom": True,
                "custom_metrics": ["diversity"],
                "top_k": 5,
            }

            with open(config_file, "w") as f:
                json.dump(config, f)

            from click.testing import CliRunner

            with patch(
                "cogniverse_agents.search.service.SearchService",
                return_value=mock_search_service,
            ):
                runner = CliRunner()

                result = runner.invoke(
                    cli,
                    [
                        "evaluate",
                        "--mode",
                        "experiment",
                        "--dataset",
                        "test_dataset",
                        "-p",
                        "test_profile",
                        "-s",
                        "test_strategy",
                        "--config",
                        config_file,
                    ],
                )

                assert result.exit_code == 0

    @pytest.mark.integration
    def test_multiple_profiles_strategies(
        self, mock_evaluator_provider, mock_search_service
    ):
        """Test evaluation with multiple profiles and strategies."""
        with patch(
            "cogniverse_agents.search.service.SearchService",
            return_value=mock_search_service,
        ):
            task = evaluation_task(
                mode="experiment",
                dataset_name="test_dataset",
                profiles=["profile1", "profile2"],
                strategies=["strategy1", "strategy2", "strategy3"],
                config={"use_custom": True},
            )

            # This should create 2*3=6 configurations
            results = inspect_eval(task, model="mockllm/model")

            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0

    @pytest.mark.integration
    def test_error_handling_invalid_dataset(
        self, mock_evaluator_provider, mock_phoenix_client
    ):
        """Test error handling with invalid dataset."""
        mock_phoenix_client.get_dataset.return_value = None

        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            evaluation_task(
                mode="experiment",
                dataset_name="nonexistent",
                profiles=["p1"],
                strategies=["s1"],
            )
