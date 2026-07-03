"""
Unit tests for evaluation task orchestrator.

Mode/dataset/solver/scorer assembly is verified end to end against real
Phoenix + Inspect AI in
``tests/evaluation/integration/test_task_real.py``. The cases here cover
the pieces that resolve before any boundary is touched: input validation
and plugin auto-registration branch selection.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from cogniverse_evaluation.core.task import evaluation_task


class TestEvaluationTask:
    """Test evaluation task orchestrator."""

    @pytest.mark.unit
    def test_experiment_mode_missing_params(self, mock_evaluator_provider):
        """Test experiment mode without required parameters."""
        with pytest.raises(ValueError, match="profiles and strategies required"):
            evaluation_task(mode="experiment", dataset_name="test_dataset")

    @pytest.mark.unit
    def test_plugin_auto_registration_with_config(self, mock_evaluator_provider):
        """Test that plugins are auto-registered from config."""
        config = {"evaluation": {"plugins": ["video"]}}

        with patch(
            "cogniverse_evaluation.plugins.auto_register_plugins"
        ) as mock_register:
            with patch(
                "cogniverse_evaluation.core.task.MemoryDataset"
            ) as mock_dataset_class:
                with patch("cogniverse_evaluation.core.solvers.create_batch_solver"):
                    with patch(
                        "cogniverse_evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []
                        mock_dataset.__len__ = Mock(return_value=1)
                        mock_dataset_class.return_value = mock_dataset
                        mock_scorers.return_value = []

                        evaluation_task(
                            mode="batch", dataset_name="test_dataset", config=config
                        )

                        mock_register.assert_called_once_with(config)

    @pytest.mark.unit
    def test_video_plugin_auto_registration(self, mock_evaluator_provider):
        """Test that video plugin is auto-registered for video datasets."""
        with patch(
            "cogniverse_evaluation.plugins.register_video_plugin"
        ) as mock_register:
            with patch(
                "cogniverse_evaluation.core.task.MemoryDataset"
            ) as mock_dataset_class:
                with patch(
                    "cogniverse_evaluation.core.solvers.create_retrieval_solver"
                ):
                    with patch(
                        "cogniverse_evaluation.core.inspect_scorers.get_configured_scorers"
                    ) as mock_scorers:
                        mock_dataset = MagicMock()
                        mock_dataset.examples = []
                        mock_dataset.__len__ = Mock(return_value=1)
                        mock_dataset_class.return_value = mock_dataset
                        mock_scorers.return_value = []

                        evaluation_task(
                            mode="experiment",
                            dataset_name="video_test_dataset",
                            profiles=["frame_based"],
                            strategies=["tfidf"],
                        )

                        mock_register.assert_called_once()


class TestCustomPluginRegistration:
    """auto_register_plugins resolves custom plugins under the real package.

    Regression guard: the import path was `src.evaluation.plugins.*`, a
    namespace that does not exist post-workspace-migration, so every custom
    plugin silently ImportError'd. It must resolve to
    `cogniverse_evaluation.plugins.*`.
    """

    def test_custom_plugin_module_register_is_invoked(self):
        import cogniverse_evaluation.plugins.visual_evaluator as ve
        from cogniverse_evaluation.plugins import auto_register_plugins

        with patch.object(ve, "register") as mock_register:
            auto_register_plugins({"evaluation": {"plugins": ["visual_evaluator"]}})

        mock_register.assert_called_once()


@pytest.mark.unit
class TestDatasetFetchOncePerSweep:
    """evaluation_task must reuse the downloaded dataset frame across task
    builds — an experiment sweep constructs one task per profile x strategy
    and previously re-downloaded the identical dataset each time."""

    def test_second_task_build_reuses_cached_dataset(self, mock_evaluator_provider):
        import pandas as pd

        from cogniverse_evaluation.core import task as task_mod

        task_mod._DATASET_FRAMES.clear()

        df = pd.DataFrame(
            [{"input": {"query": "q1", "expected_videos": "v1"}, "output": {}}]
        )
        phoenix_dataset = MagicMock()
        phoenix_dataset.to_dataframe.return_value = df

        with (
            patch("phoenix.client.Client") as mock_client_cls,
            patch("cogniverse_evaluation.core.solvers.create_batch_solver"),
            patch(
                "cogniverse_evaluation.core.inspect_scorers.get_configured_scorers",
                return_value=[],
            ),
        ):
            mock_client_cls.return_value.datasets.get_dataset.return_value = (
                phoenix_dataset
            )

            task_mod.evaluation_task(mode="batch", dataset_name="golden_v1")
            task_mod.evaluation_task(mode="batch", dataset_name="golden_v1")

            assert mock_client_cls.call_count == 1, (
                "second task build must reuse the cached dataset frame "
                "instead of re-downloading it"
            )

        task_mod._DATASET_FRAMES.clear()
