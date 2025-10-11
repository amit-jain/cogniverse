"""
Unit tests for evaluation task orchestrator.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from cogniverse_core.evaluation.core.task import evaluation_task


class TestEvaluationTask:
    """Test evaluation task orchestrator."""

    @pytest.mark.unit
    def test_evaluation_task_experiment_mode(self, mock_phoenix_client):
        """Test creating evaluation task in experiment mode."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.core.task.MemoryDataset") as mock_dataset_class:
                with patch(
                    "cogniverse_core.evaluation.core.solvers.create_retrieval_solver"
                ) as mock_solver:
                    with patch(
                        "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        # Create mock dataset with examples and __len__
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []  # Empty list of examples
                        mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                        mock_dataset_class.return_value = mock_dataset
                        mock_solver.return_value = Mock()
                        mock_scorers.return_value = [Mock(), Mock()]

                        task = evaluation_task(
                            mode="experiment",
                            dataset_name="test_dataset",
                            profiles=["profile1", "profile2"],
                            strategies=["strategy1", "strategy2"],
                        )

                    assert task is not None
                    assert task.dataset is not None
                    assert task.solver is not None
                    assert len(task.scorer) == 2
                    assert task.metadata["mode"] == "experiment"

                    # Verify retrieval solver was called with correct params
                    mock_solver.assert_called_once()
                    args = mock_solver.call_args[0]
                    assert args[0] == ["profile1", "profile2"]
                    assert args[1] == ["strategy1", "strategy2"]

    @pytest.mark.unit
    def test_evaluation_task_batch_mode(self, mock_phoenix_client):
        """Test creating evaluation task in batch mode."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.core.task.MemoryDataset") as mock_dataset_class:
                with patch(
                    "cogniverse_core.evaluation.core.solvers.create_batch_solver"
                ) as mock_solver:
                    with patch(
                        "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        # Create mock dataset with examples and __len__
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []  # Empty list of examples
                        mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                        mock_dataset_class.return_value = mock_dataset
                        mock_solver.return_value = Mock()
                        mock_scorers.return_value = [Mock()]

                        task = evaluation_task(
                            mode="batch",
                            dataset_name="test_dataset",
                            trace_ids=["trace1", "trace2"],
                        )

                    assert task is not None
                    assert task.metadata["mode"] == "batch"

                    # Verify trace loader solver was called
                    mock_solver.assert_called_once()
                    args = mock_solver.call_args[0]
                    assert args[0] == ["trace1", "trace2"]

    @pytest.mark.unit
    def test_evaluation_task_live_mode(self, mock_phoenix_client):
        """Test creating evaluation task in live mode."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.core.task.MemoryDataset") as mock_dataset_class:
                with patch(
                    "cogniverse_core.evaluation.core.solvers.create_live_solver"
                ) as mock_solver:
                    with patch(
                        "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        # Create mock dataset with examples and __len__
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []  # Empty list of examples
                        mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                        mock_dataset_class.return_value = mock_dataset
                        mock_solver.return_value = Mock()
                        mock_scorers.return_value = []

                        task = evaluation_task(mode="live", dataset_name="test_dataset")

                    assert task is not None
                    assert task.metadata["mode"] == "live"
                    mock_solver.assert_called_once()

    @pytest.mark.unit
    def test_invalid_mode_raises_error(self, mock_phoenix_client):
        """Test that invalid mode raises ValueError."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.core.task.MemoryDataset") as mock_dataset_class:
                mock_dataset_class.return_value = Mock()
                with pytest.raises(ValueError, match="Unknown mode"):
                    evaluation_task(mode="invalid", dataset_name="test_dataset")

    @pytest.mark.unit
    def test_experiment_mode_missing_params(self, mock_phoenix_client):
        """Test experiment mode without required parameters."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with pytest.raises(ValueError, match="profiles and strategies required"):
                evaluation_task(mode="experiment", dataset_name="test_dataset")

    @pytest.mark.unit
    def test_dataset_not_found(self):
        """Test handling of missing dataset."""
        mock_client = MagicMock()
        mock_client.get_dataset.return_value = None

        with patch("cogniverse_core.evaluation.core.task.px.Client", return_value=mock_client):
            with pytest.raises(ValueError, match="Dataset 'missing_dataset' not found"):
                evaluation_task(mode="batch", dataset_name="missing_dataset")

    @pytest.mark.unit
    def test_config_passed_to_scorers(self, mock_phoenix_client):
        """Test that config is passed to get_configured_scorers."""
        config = {"use_ragas": True, "custom_metrics": ["diversity"]}

        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.core.task.MemoryDataset") as mock_dataset_class:
                with patch("cogniverse_core.evaluation.core.solvers.create_retrieval_solver"):
                    with patch(
                        "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        # Create mock dataset with examples and __len__
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []  # Empty list of examples
                        mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                        mock_dataset_class.return_value = mock_dataset
                        mock_scorers.return_value = []

                        evaluation_task(
                            mode="experiment",
                            dataset_name="test_dataset",
                            profiles=["p1"],
                            strategies=["s1"],
                            config=config,
                        )

                    mock_scorers.assert_called_once_with(config)

    @pytest.mark.unit
    def test_plugin_auto_registration_with_config(self, mock_phoenix_client):
        """Test that plugins are auto-registered from config."""
        config = {"evaluation": {"plugins": ["video"]}}

        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.plugins.auto_register_plugins") as mock_register:
                with patch(
                    "cogniverse_core.evaluation.core.task.MemoryDataset"
                ) as mock_dataset_class:
                    with patch("cogniverse_core.evaluation.core.solvers.create_batch_solver"):
                        with patch(
                            "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                        ):
                            # Create mock dataset with examples and __len__
                            mock_dataset = MagicMock()
                            mock_dataset.examples = []  # Empty list of examples
                            mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                            mock_dataset_class.return_value = mock_dataset

                            evaluation_task(
                                mode="batch", dataset_name="test_dataset", config=config
                            )

                            mock_register.assert_called_once_with(config)

    @pytest.mark.unit
    def test_video_plugin_auto_registration(self, mock_phoenix_client):
        """Test that video plugin is auto-registered for video datasets."""
        with patch(
            "cogniverse_core.evaluation.core.task.px.Client", return_value=mock_phoenix_client
        ):
            with patch("cogniverse_core.evaluation.plugins.register_video_plugin") as mock_register:
                with patch(
                    "cogniverse_core.evaluation.core.task.MemoryDataset"
                ) as mock_dataset_class:
                    with patch("cogniverse_core.evaluation.core.solvers.create_retrieval_solver"):
                        with patch(
                            "cogniverse_core.evaluation.core.inspect_scorers.get_configured_scorers"
                        ):
                            # Create mock dataset with examples and __len__
                            mock_dataset = MagicMock()
                            mock_dataset.examples = []  # Empty list of examples
                            mock_dataset.__len__ = Mock(return_value=1)  # Has 1 sample
                            mock_dataset_class.return_value = mock_dataset

                            evaluation_task(
                                mode="experiment",
                                dataset_name="video_test_dataset",
                                profiles=["frame_based"],
                                strategies=["tfidf"],
                            )

                            mock_register.assert_called_once()
