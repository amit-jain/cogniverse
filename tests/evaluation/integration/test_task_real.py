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

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


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


class TestDatasetManagerWriterReadByTask:
    """A dataset written through the real DatasetManager (expected_videos in
    the OUTPUT key slot) must yield non-empty Sample targets through
    evaluation_task. The conftest test_dataset masks this by seeding the
    opposite key classification."""

    def test_manager_created_dataset_has_non_empty_targets(
        self, search_evaluator_provider, phoenix_container
    ):
        import uuid

        from cogniverse_evaluation.core import task as task_mod
        from cogniverse_evaluation.data.datasets import DatasetManager
        from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

        name = f"dm-task-{uuid.uuid4().hex[:8]}"
        store = PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])
        DatasetManager(tenant_id="acme:t", dataset_store=store).create_from_queries(
            [
                {
                    "query": "red kite over the field",
                    "expected_videos": ["kite1", "kite2"],
                    "category": "visual",
                }
            ],
            name,
        )
        # Bypass any process-cached frame from a prior name reuse.
        task_mod._DATASET_FRAMES.clear()

        task = evaluation_task(
            mode="experiment",
            dataset_name=name,
            profiles=["colpali_prof"],
            strategies=["float_float"],
        )

        samples = _samples_by_input(task)
        assert "red kite over the field" in samples
        sample = samples["red kite over the field"]
        assert sample.target == ["kite1", "kite2"], (
            f"expected_videos from the output slot were dropped: {sample.target}"
        )
        assert sample.metadata["query_type"] == "visual"


class TestBatchSolverRealPhoenix:
    """The batch solver against real Phoenix spans.

    Emits real search spans, then loads them back through the real provider —
    verifying the tenant-derived project, the ``context.trace_id`` filter, and
    the trace-dict extraction (query, parsed results, duration) against the
    frame shape Phoenix actually returns.
    """

    @pytest.mark.asyncio
    async def test_batch_solver_loads_requested_trace_from_real_spans(
        self, search_evaluator_provider
    ):
        import asyncio
        import time
        import uuid
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace

        from cogniverse_evaluation.core.solvers import create_batch_solver
        from cogniverse_foundation.telemetry.context import (
            add_search_results_to_span,
            search_span,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        suffix = uuid.uuid4().hex[:8]
        tenant_id = f"bsolver{suffix}:main"
        project_name = f"cogniverse-{tenant_id}"
        manager = get_telemetry_manager()

        def _emit(query: str) -> str:
            results = [
                SimpleNamespace(
                    document=SimpleNamespace(
                        id="vid_pos",
                        metadata={"source_id": "vid_pos"},
                        content_type=None,
                    ),
                    score=0.93,
                ),
                SimpleNamespace(
                    document=SimpleNamespace(
                        id="vid_neg",
                        metadata={"source_id": "vid_neg"},
                        content_type=None,
                    ),
                    score=0.40,
                ),
            ]
            with search_span(tenant_id=tenant_id, query=query, top_k=5) as span:
                add_search_results_to_span(span, results)
                trace_id_hex = format(span.get_span_context().trace_id, "032x")
            return trace_id_hex

        wanted_trace = _emit("kite surfing on a windy beach")
        _emit("decoy query that must be filtered out")
        manager.force_flush(timeout_millis=10000)

        # Wait until both spans are indexed so the trace-id filter is proven
        # to exclude the decoy rather than racing its ingestion.
        provider = search_evaluator_provider
        deadline = time.time() + 60
        while time.time() < deadline:
            df = await provider.telemetry.traces.get_spans(
                project=project_name,
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                limit=100,
            )
            if (
                df is not None
                and not df.empty
                and "name" in df.columns
                and int((df["name"] == "search_service.search").sum()) >= 2
            ):
                break
            await asyncio.sleep(2.0)
        else:
            pytest.fail("emitted spans were not indexed within 60s")

        assert "context.trace_id" in df.columns
        wanted_rows = df[
            (df["context.trace_id"] == wanted_trace)
            & (df["name"] == "search_service.search")
        ]
        assert len(wanted_rows) == 1
        wanted_row = wanted_rows.iloc[0]

        solver = create_batch_solver(
            trace_ids=[wanted_trace], config={"tenant_id": tenant_id}
        )
        state = SimpleNamespace(outputs={}, metadata={})
        result = await solver(state, None)

        loaded = result.metadata["loaded_traces"]
        assert [t["trace_id"] for t in loaded] == [wanted_trace]
        trace = loaded[0]
        assert trace["query"] == "kite surfing on a windy beach"
        assert (
            trace["duration_ms"]
            == (wanted_row["end_time"] - wanted_row["start_time"]).total_seconds()
            * 1000.0
        )
        assert trace["timestamp"] == wanted_row["start_time"]
        # No backend configured: the schema-aware strategy reports that
        # explicitly instead of fabricating ground truth.
        assert trace["ground_truth"] == []
        assert trace["ground_truth_source"] == "no_backend"
        stats = result.metadata["ground_truth_stats"]
        assert stats["total_traces"] == 1


_MEMO_PROBE_DATASET = "dataset-frame-memo-probe"


class TestDatasetFrameMemoIsScopedToOneTest:
    """``evaluation_task`` memoises the fetched Phoenix frame per
    ``(endpoint, dataset_name)`` so one experiment sweep fetches a dataset once
    across its profile x strategy tasks.

    These two tests run in order against the same module-scoped Phoenix. The
    first populates the memo; the second appends a row to the same dataset and
    must see it. If the memo survives the test that created it, the second
    reads the first's frame and the appended row is invisible.
    """

    def test_first_task_populates_the_memo(
        self, search_evaluator_provider, phoenix_container
    ):
        from cogniverse_evaluation.core import task as task_mod
        from cogniverse_evaluation.data.datasets import DatasetManager
        from cogniverse_evaluation.providers import get_evaluation_provider
        from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

        assert task_mod._DATASET_FRAMES == {}

        store = PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])
        DatasetManager(tenant_id="acme:t", dataset_store=store).create_from_queries(
            [
                {
                    "query": "alpha marker query",
                    "expected_videos": ["alpha_vid"],
                    "category": "visual",
                }
            ],
            _MEMO_PROBE_DATASET,
        )

        task = evaluation_task(
            mode="experiment",
            dataset_name=_MEMO_PROBE_DATASET,
            profiles=["colpali_prof"],
            strategies=["float_float"],
        )

        samples = _samples_by_input(task)
        assert sorted(samples) == ["alpha marker query"]
        assert samples["alpha marker query"].target == ["alpha_vid"]

        endpoint = get_evaluation_provider().http_endpoint
        assert list(task_mod._DATASET_FRAMES) == [(endpoint, _MEMO_PROBE_DATASET)]

    def test_second_task_reads_live_phoenix_not_the_previous_frame(
        self, search_evaluator_provider, phoenix_container
    ):
        from cogniverse_evaluation.core import task as task_mod
        from cogniverse_evaluation.data.datasets import DatasetManager
        from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

        memo_at_entry = dict(task_mod._DATASET_FRAMES)

        store = PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])
        DatasetManager(tenant_id="acme:t", dataset_store=store).create_from_queries(
            [
                {
                    "query": "beta marker query",
                    "expected_videos": ["beta_vid"],
                    "category": "visual",
                }
            ],
            _MEMO_PROBE_DATASET,
        )

        task = evaluation_task(
            mode="experiment",
            dataset_name=_MEMO_PROBE_DATASET,
            profiles=["colpali_prof"],
            strategies=["float_float"],
        )

        samples = _samples_by_input(task)
        assert sorted(samples) == ["alpha marker query", "beta marker query"]
        assert samples["beta marker query"].target == ["beta_vid"]
        assert samples["alpha marker query"].target == ["alpha_vid"]
        assert memo_at_entry == {}
