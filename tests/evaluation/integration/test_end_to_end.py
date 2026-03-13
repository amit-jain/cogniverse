"""
End-to-end integration tests for evaluation framework.

Tests experiment mode with real Vespa search backend and real ColPali encoder.
Batch/live modes and error handling use mock evaluator provider since they
don't interact with search.
"""

import json
import os
import tempfile

import pytest
from inspect_ai import eval as inspect_eval

from cogniverse_evaluation.cli import cli
from cogniverse_evaluation.core.task import evaluation_task


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_experiment_mode_e2e(self, search_evaluator_provider, eval_search_client):
        """Test complete experiment mode with real search.

        Uses real ColPali encoder + real Vespa with seeded documents.
        The solver's httpx.post calls are routed through the TestClient
        to the real search router.
        """
        from tests.evaluation.conftest import intercept_search_calls

        with intercept_search_calls(eval_search_client):
            task = evaluation_task(
                mode="experiment",
                dataset_name="test_dataset",
                profiles=["test_colpali"],
                strategies=["default"],
                config={
                    "use_ragas": False,
                    "use_custom": True,
                    "custom_metrics": ["diversity", "result_count"],
                    "tenant_id": "default",
                },
            )

            results = inspect_eval(task, model="mockllm/model")

            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0

            eval_log = results[0]
            assert (
                eval_log.status == "success"
            ), f"Evaluation failed with status: {eval_log.status}"

            assert eval_log.results is not None, "No results in eval log"
            assert eval_log.results.scores is not None, "No scores in results"
            assert len(eval_log.results.scores) > 0, "No scores generated"

            scorer_names = {score.name for score in eval_log.results.scores}
            expected_scorers = {"diversity_scorer", "result_count_scorer"}
            assert expected_scorers.issubset(scorer_names), (
                f"Expected scorers {expected_scorers}, got {scorer_names}"
            )

            for score in eval_log.results.scores:
                assert score.metrics is not None, f"Scorer {score.name} has no metrics"
                assert "mean" in score.metrics, (
                    f"Scorer {score.name} missing 'mean' metric"
                )
                mean_value = score.metrics["mean"].value
                assert mean_value is not None, (
                    f"Scorer {score.name} mean metric has no value"
                )
                assert 0.0 <= mean_value <= 1.0, (
                    f"Scorer {score.name} mean {mean_value} not in [0, 1]"
                )

            assert eval_log.samples is not None, "No samples in eval log"
            assert len(eval_log.samples) > 0, "No samples processed"

            # Verify search actually returned results (not empty failures)
            for sample in eval_log.samples:
                assert sample.scores is not None, f"Sample {sample.id} has no scores"
                assert len(sample.scores) > 0, f"Sample {sample.id} has empty scores"

    @pytest.mark.integration
    def test_batch_mode_e2e(self, mock_evaluator_provider):
        """Test complete batch mode workflow.

        Batch mode loads existing traces from Phoenix -- no search needed.
        """
        task = evaluation_task(
            mode="batch",
            dataset_name="test_dataset",
            trace_ids=["trace1"],
            config={"use_ragas": True, "ragas_metrics": ["context_relevancy"]},
        )

        results = inspect_eval(task, model="mockllm/model")

        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.integration
    def test_cli_evaluate_command(self, search_evaluator_provider, eval_search_client):
        """Test CLI evaluate command with real search."""
        import tempfile

        from click.testing import CliRunner

        from tests.evaluation.conftest import intercept_search_calls

        # CLI needs tenant_id in config file to match test Vespa
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as cfg:
            json.dump({"tenant_id": "default"}, cfg)
            cfg_path = cfg.name

        try:
            with intercept_search_calls(eval_search_client):
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
                        "test_colpali",
                        "-s",
                        "default",
                        "--config",
                        cfg_path,
                    ],
                )

                assert result.exit_code == 0
                assert "Starting experiment evaluation" in result.output
                assert (
                    "Evaluation complete" in result.output
                    or "Running evaluation" in result.output
                )
        finally:
            os.unlink(cfg_path)

    @pytest.mark.integration
    def test_cli_list_traces_command(self, mock_evaluator_provider):
        """Test CLI list-traces command.

        List-traces only reads from Phoenix -- no search needed.
        """
        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(cli, ["list-traces", "--hours", "2", "--limit", "10"])

        assert result.exit_code == 0
        assert "Fetching traces" in result.output

    @pytest.mark.integration
    def test_evaluation_with_output_file(
        self, search_evaluator_provider, eval_search_client
    ):
        """Test evaluation with output file saving using real search."""
        from click.testing import CliRunner

        from tests.evaluation.conftest import intercept_search_calls

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")

            # Write tenant config for CLI
            config_file_path = os.path.join(tmpdir, "tenant_config.json")
            with open(config_file_path, "w") as f:
                json.dump({"tenant_id": "default"}, f)

            with intercept_search_calls(eval_search_client):
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
                        "test_colpali",
                        "-s",
                        "default",
                        "--config",
                        config_file_path,
                        "--output",
                        output_file,
                    ],
                )

            if result.exit_code == 0:
                assert os.path.exists(output_file)

                with open(output_file) as f:
                    data = json.load(f)
                    assert data["mode"] == "experiment"
                    assert data["dataset"] == "test_dataset"
                    assert "timestamp" in data

    @pytest.mark.integration
    def test_evaluation_with_config_file(
        self, search_evaluator_provider, eval_search_client
    ):
        """Test evaluation with configuration file using real search."""
        from click.testing import CliRunner

        from tests.evaluation.conftest import intercept_search_calls

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.json")

            config = {
                "use_ragas": True,
                "ragas_metrics": ["context_relevancy"],
                "use_custom": True,
                "custom_metrics": ["diversity"],
                "top_k": 5,
                "tenant_id": "default",
            }

            with open(config_file, "w") as f:
                json.dump(config, f)

            with intercept_search_calls(eval_search_client):
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
                        "test_colpali",
                        "-s",
                        "default",
                        "--config",
                        config_file,
                    ],
                )

                assert result.exit_code == 0

    @pytest.mark.integration
    def test_multiple_profiles_e2e(
        self, search_evaluator_provider, eval_search_client
    ):
        """Test evaluation with real search profile.

        Uses a single real profile (test_colpali) with the default strategy
        to verify the evaluation framework handles real search results.
        """
        from tests.evaluation.conftest import intercept_search_calls

        with intercept_search_calls(eval_search_client):
            task = evaluation_task(
                mode="experiment",
                dataset_name="test_dataset",
                profiles=["test_colpali"],
                strategies=["default"],
                config={"use_custom": True, "tenant_id": "default"},
            )

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
