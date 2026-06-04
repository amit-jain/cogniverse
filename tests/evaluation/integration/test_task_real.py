"""Real-boundary coverage for ``evaluation_task`` against Phoenix + Inspect AI.

``evaluation_task`` loads a Phoenix dataset, converts each row into an Inspect
AI ``Sample``, and assembles a real ``Task`` (dataset + solver + scorers). These
tests drive that full path through a real Phoenix container and the real
Inspect AI library so the DataFrame→Sample conversion and Task assembly are
verified end to end, not against self-confirming mocks.
"""

from __future__ import annotations

import pytest

from cogniverse_evaluation.core.task import evaluation_task

pytestmark = pytest.mark.integration


def _samples_by_input(task):
    return {sample.input: sample for sample in task.dataset}


class TestEvaluationTaskRealPhoenix:
    def test_experiment_mode_builds_samples_solver_scorers(
        self, search_evaluator_provider
    ):
        task = evaluation_task(
            mode="experiment",
            dataset_name="test_dataset",
            profiles=["colpali_prof"],
            strategies=["float_float"],
        )

        assert len(task.dataset) == 2
        samples = _samples_by_input(task)
        assert set(samples) == {"sunset landscape mountains", "ocean waves coastal"}

        sunset = samples["sunset landscape mountains"]
        assert sunset.target == ["sunset_vid"]
        assert sunset.metadata["query_type"] == "visual"

        ocean = samples["ocean waves coastal"]
        assert ocean.target == ["ocean_vid"]
        assert ocean.metadata["query_type"] == "visual"

        # Real solver factory returns a callable solver (not None / not a mock).
        assert task.solver is not None
        assert callable(task.solver)

        # Default config enables relevance + diversity + result_count +
        # precision + recall.
        assert isinstance(task.scorer, list)
        assert len(task.scorer) == 5

        assert task.metadata["mode"] == "experiment"
        assert task.metadata["dataset_name"] == "test_dataset"
        assert task.metadata["profiles"] == ["colpali_prof"]
        assert task.metadata["strategies"] == ["float_float"]

    def test_batch_mode_builds_task_with_batch_solver(self, search_evaluator_provider):
        task = evaluation_task(
            mode="batch",
            dataset_name="test_dataset",
            trace_ids=["trace_a", "trace_b"],
        )

        assert len(task.dataset) == 2
        assert task.solver is not None
        assert callable(task.solver)
        assert task.metadata["mode"] == "batch"
        assert task.metadata["dataset_name"] == "test_dataset"

    def test_config_controls_scorer_set(self, search_evaluator_provider):
        task = evaluation_task(
            mode="experiment",
            dataset_name="test_dataset",
            profiles=["colpali_prof"],
            strategies=["float_float"],
            config={
                "use_diversity": False,
                "use_result_count": False,
                "use_precision_recall": False,
            },
        )

        # Only relevance survives the config switches.
        assert len(task.scorer) == 1

    def test_live_mode_builds_task_with_live_solver(self, search_evaluator_provider):
        task = evaluation_task(mode="live", dataset_name="test_dataset")

        assert len(task.dataset) == 2
        assert task.solver is not None
        assert callable(task.solver)
        assert task.metadata["mode"] == "live"

    def test_unknown_mode_raises_after_real_load(self, search_evaluator_provider):
        with pytest.raises(ValueError, match="Unknown mode"):
            evaluation_task(mode="invalid", dataset_name="test_dataset")

    def test_missing_dataset_raises(self, search_evaluator_provider):
        with pytest.raises(Exception) as exc_info:
            evaluation_task(mode="batch", dataset_name="no_such_dataset_xyz")
        # Real Phoenix reports the missing dataset by name rather than
        # silently yielding an empty task.
        assert "no_such_dataset_xyz" in str(exc_info.value)
