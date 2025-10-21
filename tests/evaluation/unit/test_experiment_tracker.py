"""
Unit tests for experiment tracker module.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pandas as pd
import pytest
from cogniverse_core.evaluation.core.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Test experiment tracker functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with (
            patch("cogniverse_core.evaluation.core.experiment_tracker.DatasetManager") as mock_dm,
            patch("cogniverse_core.evaluation.core.experiment_tracker.RetrievalMonitor") as mock_pm,
            patch("cogniverse_core.evaluation.core.experiment_tracker.register_plugin") as mock_reg,
        ):

            yield {
                "dataset_manager": mock_dm,
                "phoenix_monitor": mock_pm,
                "register_plugin": mock_reg,
            }

    @pytest.fixture
    def tracker(self, mock_dependencies):
        """Create experiment tracker with mocked dependencies."""
        return ExperimentTracker(
            experiment_project_name="test_project",
            enable_quality_evaluators=True,
            enable_llm_evaluators=False,
        )

    @pytest.mark.unit
    def test_init_default_params(self, mock_dependencies):
        """Test initialization with default parameters."""
        tracker = ExperimentTracker()

        assert tracker.experiment_project_name == "experiments"
        assert tracker.enable_quality_evaluators is True
        assert tracker.enable_llm_evaluators is False
        assert tracker.evaluator_name == "visual_judge"
        assert tracker.llm_model == "deepseek-r1:7b"
        assert tracker.experiments == []
        assert tracker.configurations == []

    @pytest.mark.unit
    def test_init_custom_params(self, mock_dependencies):
        """Test initialization with custom parameters."""
        output_dir = Path("/tmp/test_output")

        tracker = ExperimentTracker(
            experiment_project_name="custom_project",
            output_dir=output_dir,
            enable_quality_evaluators=False,
            enable_llm_evaluators=True,
            evaluator_name="custom_evaluator",
            llm_model="gpt-4",
            llm_base_url="http://custom.api",
        )

        assert tracker.experiment_project_name == "custom_project"
        assert tracker.output_dir == output_dir
        assert tracker.enable_quality_evaluators is False
        assert tracker.enable_llm_evaluators is True
        assert tracker.evaluator_name == "custom_evaluator"
        assert tracker.llm_model == "gpt-4"
        assert tracker.llm_base_url == "http://custom.api"

    @pytest.mark.unit
    def test_register_evaluator_plugins_quality(self, mock_dependencies):
        """Test registering quality evaluator plugins."""
        # Mock the entire module since VideoAnalyzerPlugin may not exist
        with patch("cogniverse_core.evaluation.plugins.video_analyzer") as mock_module:
            mock_plugin = Mock()
            mock_module.VideoAnalyzerPlugin = Mock(return_value=mock_plugin)

            ExperimentTracker(enable_quality_evaluators=True)

            # Should not raise exception even if plugin doesn't exist

    @pytest.mark.unit
    def test_register_evaluator_plugins_llm(self, mock_dependencies):
        """Test registering LLM evaluator plugins."""
        with patch("cogniverse_core.evaluation.plugins.visual_evaluator.VisualEvaluatorPlugin"):
            ExperimentTracker(enable_llm_evaluators=True)

            mock_dependencies["register_plugin"].assert_called()

    @pytest.mark.unit
    def test_register_evaluator_plugins_import_error(self, mock_dependencies):
        """Test handling import errors when registering plugins."""
        # Mock the import to fail
        with patch(
            "cogniverse_core.evaluation.core.experiment_tracker.register_plugin"
        ) as mock_reg:
            mock_reg.side_effect = ImportError("Module not found")

            # Should not raise exception
            tracker = ExperimentTracker(enable_quality_evaluators=True)
            assert tracker is not None

    @pytest.mark.unit
    def test_get_experiment_configurations_default(self, tracker):
        """Test getting experiment configurations with defaults."""
        with patch("cogniverse_core.registries.registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_profiles.return_value = ["profile1", "profile2"]
            mock_registry.list_ranking_strategies.return_value = [
                "binary_binary",
                "float_float",
                "custom_strategy",
            ]
            mock_get_registry.return_value = mock_registry

            configs = tracker.get_experiment_configurations()

            assert len(configs) == 2
            assert configs[0]["profile"] == "profile1"
            assert configs[1]["profile"] == "profile2"

            # Should filter to common strategies only
            strategies = [strategy for strategy, desc in configs[0]["strategies"]]
            assert "binary_binary" in strategies
            assert "float_float" in strategies
            assert "custom_strategy" not in strategies

    @pytest.mark.unit
    def test_get_experiment_configurations_specific_profiles(self, tracker):
        """Test getting configurations for specific profiles."""
        with patch("cogniverse_core.registries.registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_ranking_strategies.return_value = [
                "binary_binary",
                "float_float",
            ]
            mock_get_registry.return_value = mock_registry

            configs = tracker.get_experiment_configurations(profiles=["profile1"])

            assert len(configs) == 1
            assert configs[0]["profile"] == "profile1"

    @pytest.mark.unit
    def test_get_experiment_configurations_specific_strategies(self, tracker):
        """Test getting configurations for specific strategies."""
        with patch("cogniverse_core.registries.registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_profiles.return_value = ["profile1"]
            mock_registry.list_ranking_strategies.return_value = [
                "binary_binary",
                "float_float",
                "phased",
            ]
            mock_get_registry.return_value = mock_registry

            configs = tracker.get_experiment_configurations(
                strategies=["binary_binary"]
            )

            assert len(configs) == 1
            strategies = [strategy for strategy, desc in configs[0]["strategies"]]
            assert strategies == ["binary_binary"]

    @pytest.mark.unit
    def test_get_experiment_configurations_all_strategies(self, tracker):
        """Test getting configurations with all strategies."""
        with patch("cogniverse_core.registries.registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_profiles.return_value = ["profile1"]
            mock_registry.list_ranking_strategies.return_value = [
                "binary_binary",
                "custom_strategy",
            ]
            mock_get_registry.return_value = mock_registry

            configs = tracker.get_experiment_configurations(all_strategies=True)

            assert len(configs) == 1
            strategies = [strategy for strategy, desc in configs[0]["strategies"]]
            assert "binary_binary" in strategies
            assert "custom_strategy" in strategies

    @pytest.mark.unit
    def test_get_experiment_configurations_registry_error(self, tracker):
        """Test handling registry errors gracefully."""
        with patch("cogniverse_core.registries.registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_profiles.return_value = ["profile1"]
            mock_registry.list_ranking_strategies.side_effect = Exception(
                "Registry error"
            )
            mock_get_registry.return_value = mock_registry

            configs = tracker.get_experiment_configurations()

            # Should handle error and return empty list
            assert configs == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_experiment_async_success(self, tracker):
        """Test successful experiment execution."""
        mock_result = Mock()
        mock_result.scores = {"mrr": Mock(value=0.85), "recall": Mock(value=0.75)}

        with (
            patch("cogniverse_core.evaluation.core.experiment_tracker.evaluation_task"),
            patch("inspect_ai.eval", new_callable=AsyncMock) as mock_eval,
        ):

            mock_eval.return_value = mock_result

            result = await tracker.run_experiment_async(
                profile="test_profile",
                strategy="test_strategy",
                dataset_name="test_dataset",
                description="Test Description",
            )

            assert result["status"] == "success"
            assert result["profile"] == "test_profile"
            assert result["strategy"] == "test_strategy"
            assert result["description"] == "Test Description"
            assert "metrics" in result
            assert result["metrics"]["mrr"] == 0.85
            assert result["metrics"]["recall"] == 0.75
            assert "timestamp" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_experiment_async_failure(self, tracker):
        """Test experiment execution failure."""
        with (
            patch("cogniverse_core.evaluation.core.experiment_tracker.evaluation_task"),
            patch("inspect_ai.eval", side_effect=Exception("Evaluation failed")),
        ):

            result = await tracker.run_experiment_async(
                profile="test_profile",
                strategy="test_strategy",
                dataset_name="test_dataset",
                description="Test Description",
            )

            assert result["status"] == "failed"
            assert result["error"] == "Evaluation failed"
            assert result["profile"] == "test_profile"
            assert result["strategy"] == "test_strategy"

    @pytest.mark.unit
    def test_run_experiment_sync(self, tracker):
        """Test synchronous experiment wrapper."""
        mock_result = {"status": "success", "metrics": {}}

        with patch.object(
            tracker, "run_experiment_async", return_value=mock_result
        ) as mock_async:
            result = tracker.run_experiment("profile", "strategy", "dataset", "desc")

            assert result == mock_result
            mock_async.assert_called_once_with("profile", "strategy", "dataset", "desc")

    @pytest.mark.unit
    def test_create_or_get_dataset_existing(self, tracker):
        """Test getting existing dataset."""
        mock_dataset = Mock()
        tracker.dataset_manager.get_dataset.return_value = mock_dataset

        result = tracker.create_or_get_dataset(dataset_name="existing_dataset")

        assert result == "existing_dataset"
        assert tracker.dataset_url == "http://localhost:6006/datasets/existing_dataset"

    @pytest.mark.unit
    def test_create_or_get_dataset_new_from_csv(self, tracker):
        """Test creating new dataset from CSV."""
        tracker.dataset_manager.get_dataset.return_value = None
        tracker.dataset_manager.create_from_csv.return_value = "new_dataset_id"

        result = tracker.create_or_get_dataset(
            dataset_name="new_dataset", csv_path="/path/to/file.csv", force_new=True
        )

        assert result == "new_dataset_id"
        assert tracker.dataset_url == "http://localhost:6006/datasets/new_dataset_id"
        tracker.dataset_manager.create_from_csv.assert_called_once()

    @pytest.mark.unit
    def test_create_or_get_dataset_test_dataset(self, tracker):
        """Test creating test dataset."""
        tracker.dataset_manager.get_dataset.return_value = None
        tracker.dataset_manager.create_test_dataset.return_value = "test_dataset_id"

        result = tracker.create_or_get_dataset()

        assert result == "test_dataset_id"
        tracker.dataset_manager.create_test_dataset.assert_called_once()

    @pytest.mark.unit
    def test_run_all_experiments_no_configurations(self, tracker):
        """Test running experiments with no configurations."""
        with pytest.raises(ValueError, match="No configurations set"):
            tracker.run_all_experiments("test_dataset")

    @pytest.mark.unit
    def test_run_all_experiments_success(self, tracker):
        """Test running all experiments successfully."""
        # Set up configurations
        tracker.configurations = [
            {
                "profile": "profile1",
                "strategies": [
                    ("strategy1", "Strategy 1"),
                    ("strategy2", "Strategy 2"),
                ],
            }
        ]

        mock_result = {"status": "success", "metrics": {"mrr": 0.8}}

        with patch.object(tracker, "run_experiment", return_value=mock_result):
            results = tracker.run_all_experiments("test_dataset")

            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)
            assert tracker.experiments == results

    @pytest.mark.unit
    def test_create_visualization_tables_empty(self, tracker):
        """Test creating visualization tables with no experiments."""
        tables = tracker.create_visualization_tables([])

        assert "profile_summary" in tables
        assert "detailed_results" in tables
        assert "strategy_comparison" in tables
        assert len(tables["profile_summary"]) == 0
        assert len(tables["detailed_results"]) == 0

    @pytest.mark.unit
    def test_create_visualization_tables_with_data(self, tracker):
        """Test creating visualization tables with experiment data."""
        experiments = [
            {
                "profile": "profile1",
                "strategy": "strategy1",
                "description": "Test 1",
                "status": "success",
                "experiment_name": "exp1",
                "metrics": {"mrr": 0.85, "diversity": 0.7},
            },
            {
                "profile": "profile1",
                "strategy": "strategy2",
                "description": "Test 2",
                "status": "failed",
                "experiment_name": "exp2",
            },
        ]

        tracker.configurations = [
            {
                "profile": "profile1",
                "strategies": [
                    ("strategy1", "Strategy 1"),
                    ("strategy2", "Strategy 2"),
                ],
            }
        ]

        tables = tracker.create_visualization_tables(experiments)

        # Check profile summary
        profile_summary = tables["profile_summary"]
        assert len(profile_summary) == 1
        assert profile_summary.iloc[0]["Profile"] == "profile1"
        assert profile_summary.iloc[0]["Total"] == 2
        assert profile_summary.iloc[0]["Success"] == 1
        assert profile_summary.iloc[0]["Failed"] == 1

        # Check detailed results
        detailed_results = tables["detailed_results"]
        assert len(detailed_results) == 2
        assert detailed_results.iloc[0]["Status"] == "✅"
        assert detailed_results.iloc[1]["Status"] == "❌"

        # Check strategy comparison
        strategy_comparison = tables["strategy_comparison"]
        assert len(strategy_comparison) == 2
        assert strategy_comparison.iloc[0]["Status"] == "✅ Success"
        assert strategy_comparison.iloc[1]["Status"] == "❌ Failed"

    @pytest.mark.unit
    def test_print_visualization(self, tracker):
        """Test printing visualization output."""
        mock_tables = {
            "profile_summary": pd.DataFrame(
                [
                    {
                        "Profile": "test",
                        "Total": 1,
                        "Success": 1,
                        "Failed": 0,
                        "Success Rate": "100%",
                    }
                ]
            ),
            "detailed_results": pd.DataFrame(
                [{"Profile": "test", "Strategy": "test", "Status": "✅"}]
            ),
            "strategy_comparison": pd.DataFrame(
                [
                    {
                        "Profile": "test",
                        "Strategy": "test",
                        "Description": "Test",
                        "Status": "✅ Success",
                    }
                ]
            ),
        }

        tracker.experiments = [{"status": "success"}]
        tracker.dataset_url = "http://localhost:6006/datasets/test"

        # Should not raise exception
        tracker.print_visualization(mock_tables)

    @pytest.mark.unit
    def test_save_results(self, tracker):
        """Test saving experiment results."""
        experiments = [
            {
                "status": "success",
                "profile": "test",
                "result": Mock(),  # Non-serializable object
            }
        ]

        mock_tables = {
            "strategy_comparison": pd.DataFrame(
                [{"Profile": "test", "Strategy": "test"}]
            )
        }

        tracker.output_dir = Path("/tmp/test")
        tracker.dataset_url = "http://test.com"

        with (
            patch("builtins.open", mock_open()),
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
            patch("json.dump") as mock_json_dump,
        ):

            csv_path, json_path = tracker.save_results(mock_tables, experiments)

            assert csv_path.name.startswith("experiment_summary_")
            assert json_path.name.startswith("experiment_details_")
            mock_to_csv.assert_called_once()
            mock_json_dump.assert_called_once()

    @pytest.mark.unit
    def test_generate_html_report_success(self, tracker):
        """Test generating HTML report successfully."""
        with patch(
            "scripts.generate_integrated_evaluation_report.generate_integrated_report"
        ) as mock_gen:
            mock_gen.return_value = Path("/tmp/report.html")

            result = tracker.generate_html_report()

            assert result == Path("/tmp/report.html")
            mock_gen.assert_called_once()

    @pytest.mark.unit
    def test_generate_html_report_failure(self, tracker):
        """Test handling HTML report generation failure."""
        with patch(
            "scripts.generate_integrated_evaluation_report.generate_integrated_report",
            side_effect=ImportError("Module not found"),
        ):
            result = tracker.generate_html_report()

            assert result is None

    @pytest.mark.unit
    def test_main_function_with_args(self, mock_dependencies):
        """Test main function with arguments."""
        from cogniverse_core.evaluation.core.experiment_tracker import main

        mock_args = Mock()
        mock_args.quality_evaluators = True
        mock_args.llm_evaluators = False
        mock_args.evaluator = "test_evaluator"
        mock_args.llm_model = "test_model"
        mock_args.llm_base_url = "http://test.com"
        mock_args.profiles = ["profile1"]
        mock_args.strategies = ["strategy1"]
        mock_args.all_strategies = False
        mock_args.dataset_name = "test_dataset"
        mock_args.csv_path = None
        mock_args.force_new = False

        with patch(
            "cogniverse_core.evaluation.core.experiment_tracker.ExperimentTracker"
        ) as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.get_experiment_configurations.return_value = []
            mock_tracker.create_or_get_dataset.return_value = "test_dataset"
            mock_tracker.run_all_experiments.return_value = []
            mock_tracker.create_visualization_tables.return_value = {}

            # Should not raise exception
            main(mock_args)

            mock_tracker_class.assert_called_once()
            mock_tracker.get_experiment_configurations.assert_called_once()
            mock_tracker.run_all_experiments.assert_called_once()

    @pytest.mark.unit
    def test_main_function_no_args(self, mock_dependencies):
        """Test main function with no arguments."""
        from cogniverse_core.evaluation.core.experiment_tracker import main

        with patch(
            "cogniverse_core.evaluation.core.experiment_tracker.ExperimentTracker"
        ) as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.get_experiment_configurations.return_value = []
            mock_tracker.create_or_get_dataset.return_value = "test_dataset"
            mock_tracker.run_all_experiments.return_value = []
            mock_tracker.create_visualization_tables.return_value = {}

            # Should not raise exception
            main(None)

            mock_tracker_class.assert_called_once()
