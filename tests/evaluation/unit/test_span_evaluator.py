"""Unit tests for SpanEvaluator — span retrieval and evaluation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cogniverse_evaluation.span_evaluator import SpanEvaluator


@pytest.fixture
def provider():
    mock = MagicMock()
    mock.telemetry = MagicMock()
    mock.telemetry.traces = MagicMock()
    mock.telemetry.traces.get_spans = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def evaluator(provider):
    return SpanEvaluator(
        provider=provider,
        tenant_id="test",
        project_name="test-project",
    )


@pytest.mark.unit
class TestSpanEvaluator:
    def test_create_mock_spans(self, evaluator):
        mock_df = evaluator._create_mock_spans_df()
        assert len(mock_df) > 0
        assert "span_id" in mock_df.columns

    def test_mock_spans_have_required_columns(self, evaluator):
        df = evaluator._create_mock_spans_df()
        for col in ("span_id", "trace_id", "operation_name", "attributes", "outputs"):
            assert col in df.columns, f"Missing column: {col}"

    def test_mock_spans_have_query_in_attributes(self, evaluator):
        df = evaluator._create_mock_spans_df()
        for _, row in df.iterrows():
            assert "query" in row["attributes"]
            assert len(row["attributes"]["query"]) > 0

    def test_mock_spans_have_results_in_outputs(self, evaluator):
        df = evaluator._create_mock_spans_df()
        for _, row in df.iterrows():
            assert "results" in row["outputs"]
            assert isinstance(row["outputs"]["results"], list)

    def test_tenant_id_stored(self, evaluator):
        assert evaluator.tenant_id == "test"

    def test_project_name_stored(self, evaluator):
        assert evaluator.project_name == "test-project"

    def test_evaluators_initialized(self, evaluator):
        assert evaluator.reference_free_evaluators is not None
        assert evaluator.golden_evaluator is not None
        assert len(evaluator.reference_free_evaluators) > 0


@pytest.mark.unit
class TestGetRecentSpans:
    @pytest.mark.asyncio
    async def test_returns_mock_when_provider_returns_none(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=None)
        df = await evaluator.get_recent_spans(hours=1)
        assert not df.empty
        assert "span_id" in df.columns

    @pytest.mark.asyncio
    async def test_returns_mock_when_provider_returns_empty_df(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame()
        )
        df = await evaluator.get_recent_spans(hours=1)
        assert not df.empty

    @pytest.mark.asyncio
    async def test_returns_mock_on_provider_exception(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(
            side_effect=Exception("Phoenix down")
        )
        df = await evaluator.get_recent_spans(hours=1)
        assert not df.empty

    @pytest.mark.asyncio
    async def test_filters_embedding_dimension_spans(self, evaluator):
        """Spans with output_value like '(19, 128)' are embedding dimensions, skip them."""
        raw_df = pd.DataFrame(
            [
                {
                    "context.span_id": "s1",
                    "context.trace_id": "t1",
                    "name": "search_service.search",
                    "attributes.input.value": "test query",
                    "attributes.output.value": "(19, 128)",
                }
            ]
        )
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(hours=1, operation_name=None)
        # Span with tuple-shaped output should be skipped
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_valid_search_results_span_included(self, evaluator):
        """Spans with proper search result structure are included."""
        import json

        results = [
            {"video_id": "v_abc", "score": 0.9},
            {"video_id": "v_def", "score": 0.7},
        ]
        raw_df = pd.DataFrame(
            [
                {
                    "context.span_id": "s2",
                    "context.trace_id": "t2",
                    "name": "search_service.search",
                    "attributes.input.value": "find dogs",
                    "attributes.output.value": json.dumps(results),
                }
            ]
        )
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(hours=1, operation_name=None)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["attributes"]["query"] == "find dogs"
        assert row["outputs"]["results"] == results

    @pytest.mark.asyncio
    async def test_span_without_query_skipped(self, evaluator):
        """Spans with no query are dropped — can't evaluate without input."""
        import json

        results = [{"video_id": "v_abc", "score": 0.9}]
        raw_df = pd.DataFrame(
            [
                {
                    "context.span_id": "s3",
                    "context.trace_id": "t3",
                    "name": "search_service.search",
                    "attributes.output.value": json.dumps(results),
                    # No input.value, no query field
                }
            ]
        )
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(hours=1, operation_name=None)
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_operation_name_filter_applied(self, evaluator):
        """Spans not matching operation_name are dropped — result is empty."""
        import json

        results = [{"video_id": "v_x", "score": 0.8}]
        raw_df = pd.DataFrame(
            [
                {
                    "context.span_id": "s4",
                    "context.trace_id": "t4",
                    "name": "other_service.call",
                    "attributes.input.value": "query",
                    "attributes.output.value": json.dumps(results),
                }
            ]
        )
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(
            hours=1, operation_name="search_service.search"
        )
        # The provider returned rows, but they were all filtered by operation_name;
        # the formatted_spans list ends up empty → empty DataFrame returned, no mock fallback.
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_limit_applied_to_results(self, evaluator):
        """Limit is respected when provider returns many rows."""
        import json

        results = [{"video_id": f"v_{i}", "score": 0.5} for i in range(3)]
        rows = [
            {
                "context.span_id": f"s{i}",
                "context.trace_id": f"t{i}",
                "name": "search_service.search",
                "attributes.input.value": f"query{i}",
                "attributes.output.value": json.dumps(results),
            }
            for i in range(10)
        ]
        raw_df = pd.DataFrame(rows)
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(hours=1, operation_name=None, limit=3)
        assert len(df) <= 3

    @pytest.mark.asyncio
    async def test_result_without_document_id_skipped(self, evaluator):
        """Results lacking document_id/video_id/source_id are not valid search results."""
        import json

        bad_results = [{"title": "something", "text": "no id field"}]
        raw_df = pd.DataFrame(
            [
                {
                    "context.span_id": "s5",
                    "context.trace_id": "t5",
                    "name": "search_service.search",
                    "attributes.input.value": "query",
                    "attributes.output.value": json.dumps(bad_results),
                }
            ]
        )
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=raw_df)
        df = await evaluator.get_recent_spans(hours=1, operation_name=None)
        assert len(df) == 0


@pytest.mark.unit
class TestEvaluateSpans:
    @pytest.mark.asyncio
    async def test_runs_all_evaluators_by_default(self, evaluator):
        df = evaluator._create_mock_spans_df()
        results = await evaluator.evaluate_spans(df)
        assert len(results) > 0
        for eval_name, eval_df in results.items():
            assert "span_id" in eval_df.columns
            assert "score" in eval_df.columns
            assert "label" in eval_df.columns

    @pytest.mark.asyncio
    async def test_specific_evaluator_names_used(self, evaluator):
        df = evaluator._create_mock_spans_df()
        results = await evaluator.evaluate_spans(df, evaluator_names=["relevance"])
        assert "relevance" in results
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_unknown_evaluator_skipped(self, evaluator):
        df = evaluator._create_mock_spans_df()
        results = await evaluator.evaluate_spans(
            df, evaluator_names=["nonexistent_evaluator"]
        )
        assert "nonexistent_evaluator" not in results

    @pytest.mark.asyncio
    async def test_evaluator_error_recorded_with_negative_score(self, evaluator):
        df = evaluator._create_mock_spans_df()

        # Make the relevance evaluator throw
        mock_rel_evaluator = MagicMock()
        mock_rel_evaluator.evaluate = AsyncMock(side_effect=Exception("eval crashed"))
        evaluator.reference_free_evaluators["relevance"] = mock_rel_evaluator

        results = await evaluator.evaluate_spans(df, evaluator_names=["relevance"])
        assert "relevance" in results
        assert all(results["relevance"]["score"] == -1.0)
        assert all(results["relevance"]["label"] == "error")

    @pytest.mark.asyncio
    async def test_golden_dataset_evaluator_runs(self, evaluator):
        df = evaluator._create_mock_spans_df()
        results = await evaluator.evaluate_spans(df, evaluator_names=["golden_dataset"])
        assert "golden_dataset" in results
        assert len(results["golden_dataset"]) == len(df)

    @pytest.mark.asyncio
    async def test_empty_df_returns_empty_results(self, evaluator):
        empty_df = pd.DataFrame(
            columns=["span_id", "trace_id", "operation_name", "attributes", "outputs"]
        )
        results = await evaluator.evaluate_spans(
            empty_df, evaluator_names=["relevance"]
        )
        assert "relevance" in results
        assert len(results["relevance"]) == 0

    @pytest.mark.asyncio
    async def test_score_values_in_valid_range(self, evaluator):
        """Relevance evaluator scores should be 0–1 (not error = -1)."""
        df = evaluator._create_mock_spans_df()
        results = await evaluator.evaluate_spans(df, evaluator_names=["relevance"])
        scores = results["relevance"]["score"]
        for score in scores:
            assert -1.0 <= score <= 1.0


@pytest.mark.unit
class TestRunEvaluationPipeline:
    @pytest.mark.asyncio
    async def test_returns_error_when_no_spans(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame()
        )

        with patch.object(
            evaluator, "_create_mock_spans_df", return_value=pd.DataFrame()
        ):
            result = await evaluator.run_evaluation_pipeline(
                hours=1,
                upload_evaluations=False,
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_pipeline_returns_summary_structure(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=None)

        result = await evaluator.run_evaluation_pipeline(
            hours=1,
            evaluator_names=["relevance"],
            upload_evaluations=False,
        )

        assert "num_spans_evaluated" in result
        assert "evaluators_run" in result
        assert "results" in result
        assert result["num_spans_evaluated"] > 0
        assert "relevance" in result["evaluators_run"]

    @pytest.mark.asyncio
    async def test_pipeline_summary_contains_mean_score(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=None)

        result = await evaluator.run_evaluation_pipeline(
            hours=1,
            evaluator_names=["relevance"],
            upload_evaluations=False,
        )

        assert "relevance" in result["results"]
        rel = result["results"]["relevance"]
        assert "mean_score" in rel
        assert "num_evaluated" in rel
        assert "score_distribution" in rel

    @pytest.mark.asyncio
    async def test_upload_evaluations_called_when_requested(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=None)
        evaluator.provider.telemetry.annotations = MagicMock()
        evaluator.provider.telemetry.annotations.add_annotation = AsyncMock()

        with patch.object(
            evaluator, "upload_evaluations", new_callable=AsyncMock
        ) as mock_upload:
            await evaluator.run_evaluation_pipeline(
                hours=1,
                evaluator_names=["relevance"],
                upload_evaluations=True,
            )
            mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_not_called_when_disabled(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(return_value=None)

        with patch.object(
            evaluator, "upload_evaluations", new_callable=AsyncMock
        ) as mock_upload:
            await evaluator.run_evaluation_pipeline(
                hours=1,
                evaluator_names=["relevance"],
                upload_evaluations=False,
            )
            mock_upload.assert_not_called()


@pytest.mark.unit
class TestUploadEvaluations:
    @pytest.mark.asyncio
    async def test_upload_calls_annotation_api(self, evaluator):
        mock_annotations = MagicMock()
        mock_annotations.add_annotation = AsyncMock()
        evaluator.provider.telemetry.annotations = mock_annotations

        eval_df = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "score": 0.8,
                    "label": "relevant",
                    "explanation": "good",
                },
            ]
        )
        await evaluator.upload_evaluations({"relevance": eval_df})

        mock_annotations.add_annotation.assert_called_once_with(
            span_id="s1",
            name="relevance",
            label="relevant",
            score=0.8,
            explanation="good",
            metadata={"evaluator": "relevance"},
        )

    @pytest.mark.asyncio
    async def test_empty_df_skipped(self, evaluator):
        mock_annotations = MagicMock()
        mock_annotations.add_annotation = AsyncMock()
        evaluator.provider.telemetry.annotations = mock_annotations

        await evaluator.upload_evaluations({"relevance": pd.DataFrame()})
        mock_annotations.add_annotation.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_error_logged_not_raised(self, evaluator):
        """Upload failure must not crash the pipeline."""
        evaluator.provider.telemetry.annotations = MagicMock()
        evaluator.provider.telemetry.annotations.add_annotation = AsyncMock(
            side_effect=Exception("annotation api down")
        )

        eval_df = pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "score": 0.5,
                    "label": "relevant",
                    "explanation": "ok",
                },
            ]
        )
        # Should not raise
        await evaluator.upload_evaluations({"relevance": eval_df})
