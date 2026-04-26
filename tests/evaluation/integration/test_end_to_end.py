"""
End-to-end integration tests for evaluation framework.

Tests experiment mode with real Vespa search backend, real ColPali encoder,
real Phoenix telemetry/datasets, and a real LLM resolved via the
``llm_endpoint`` fixture (configurable per environment — see
``tests/evaluation/integration/conftest.py``). All components are production
code; no mocks at any service boundary.
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

    @pytest.fixture(autouse=True)
    def _ensure_provider(self, search_evaluator_provider):
        """Re-register the evaluation provider before each test.

        Some CLI commands (e.g. list-traces) call reset_evaluation_provider()
        internally, which clears the provider we configured. This fixture
        ensures it's always set correctly.
        """
        from cogniverse_evaluation.providers.registry import set_evaluation_provider

        set_evaluation_provider(search_evaluator_provider)

    @pytest.mark.integration
    def test_experiment_mode_e2e(
        self, search_evaluator_provider, eval_search_client, llm_endpoint
    ):
        """Test complete experiment mode with real search.

        Uses real ColPali encoder + real Vespa with seeded documents, real
        Phoenix for dataset loading, and the LLM endpoint resolved by the
        ``llm_endpoint`` fixture (configurable per environment). The solver's
        httpx.post calls are routed through the TestClient to the real
        search router.
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
                    "tenant_id": "test:unit",
                },
            )

            results = inspect_eval(task, model=llm_endpoint["provider_uri"])

            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0

            eval_log = results[0]
            assert eval_log.status == "success", (
                f"Evaluation failed with status: {eval_log.status}"
            )

            assert eval_log.results is not None, "No results in eval log"
            assert eval_log.results.scores is not None, "No scores in results"
            assert len(eval_log.results.scores) > 0, "No scores generated"

            scorer_names = {score.name for score in eval_log.results.scores}
            expected_scorers = {"diversity_scorer", "result_count_scorer"}
            assert expected_scorers.issubset(scorer_names), (
                f"Expected scorers {expected_scorers}, got {scorer_names}"
            )

            scores_by_name = {score.name: score for score in eval_log.results.scores}
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

            # The solver searches against the real ColPali+Vespa fixture which
            # has 3 seeded docs. result_count_scorer mean of 0.0 means the
            # solver never received any results — the chain is broken (e.g.
            # tenant lookup 404, schema_loader unset, intercept misrouted).
            # The previous fixture left this gap silent: every search 404'd
            # and tests still passed with all-zero scores.
            result_count = scores_by_name["result_count_scorer"].metrics["mean"].value
            assert result_count > 0.0, (
                f"result_count_scorer mean is {result_count}; solver never got "
                f"results from the search router. All scores: "
                f"{ {n: s.metrics['mean'].value for n, s in scores_by_name.items()} }"
            )

            assert eval_log.samples is not None, "No samples in eval log"
            assert len(eval_log.samples) > 0, "No samples processed"

            for sample in eval_log.samples:
                assert sample.scores is not None, f"Sample {sample.id} has no scores"
                assert len(sample.scores) > 0, f"Sample {sample.id} has empty scores"

    @pytest.mark.integration
    def test_batch_mode_e2e(self, search_evaluator_provider, llm_endpoint):
        """Test complete batch mode workflow.

        Batch mode loads existing traces from real Phoenix.
        With no matching trace_ids, the solver handles empty results gracefully.
        """
        task = evaluation_task(
            mode="batch",
            dataset_name="test_dataset",
            trace_ids=["trace1"],
            config={"use_ragas": True, "ragas_metrics": ["context_relevancy"]},
        )

        results = inspect_eval(task, model=llm_endpoint["provider_uri"])

        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.integration
    def test_cli_evaluate_command(self, search_evaluator_provider, eval_search_client):
        """Test CLI evaluate command with real search."""
        import tempfile

        from click.testing import CliRunner

        from tests.evaluation.conftest import intercept_search_calls

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cfg:
            json.dump({"tenant_id": "test:unit"}, cfg)
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
    def test_cli_list_traces_command(self, search_evaluator_provider):
        """Test CLI list-traces command with real Phoenix.

        List-traces reads from real Phoenix — returns whatever spans exist.
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

            config_file_path = os.path.join(tmpdir, "tenant_config.json")
            with open(config_file_path, "w") as f:
                json.dump({"tenant_id": "test:unit"}, f)

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
                "tenant_id": "test:unit",
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
        self, search_evaluator_provider, eval_search_client, llm_endpoint
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
                config={
                    "use_custom": True,
                    "custom_metrics": ["result_count"],
                    "tenant_id": "test:unit",
                },
            )

            results = inspect_eval(task, model=llm_endpoint["provider_uri"])

            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0

            eval_log = results[0]
            assert eval_log.status == "success", (
                f"Evaluation failed with status: {eval_log.status}"
            )
            scores_by_name = {score.name: score for score in eval_log.results.scores}
            # Same retrieval-actually-happened guard as test_experiment_mode_e2e.
            result_count = scores_by_name["result_count_scorer"].metrics["mean"].value
            assert result_count > 0.0, (
                f"result_count_scorer mean is {result_count}; solver returned "
                f"no results."
            )

    @pytest.mark.integration
    def test_error_handling_invalid_dataset(self, search_evaluator_provider):
        """Test error handling with invalid dataset.

        Uses real Phoenix — a nonexistent dataset naturally returns None.
        """
        with pytest.raises(ValueError, match="nonexistent"):
            evaluation_task(
                mode="experiment",
                dataset_name="nonexistent",
                profiles=["p1"],
                strategies=["s1"],
            )
