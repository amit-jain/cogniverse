"""Unit tests for SpanEvaluator — span retrieval and evaluation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from cogniverse_evaluation.span_evaluator import SpanEvaluator


@pytest.fixture
def evaluator():
    mock_provider = MagicMock()
    mock_provider.telemetry = MagicMock()
    mock_provider.telemetry.traces = MagicMock()
    mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

    ev = SpanEvaluator(
        provider=mock_provider,
        tenant_id="test",
        project_name="test-project",
    )
    return ev


@pytest.mark.unit
class TestSpanEvaluator:
    @pytest.mark.asyncio
    async def test_get_recent_spans_returns_dataframe(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame([
                {
                    "name": "search_service.search",
                    "attributes.input.value": "test query",
                    "attributes.output.value": '[{"video_id": "v1", "score": 0.9}]',
                    "context.span_id": "span1",
                    "context.trace_id": "trace1",
                }
            ])
        )

        result = await evaluator.get_recent_spans(hours=1)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_get_recent_spans_empty(self, evaluator):
        evaluator.provider.telemetry.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame()
        )

        result = await evaluator.get_recent_spans(hours=1)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_evaluate_spans_returns_dict(self, evaluator):
        spans_df = pd.DataFrame([
            {
                "span_id": "s1",
                "trace_id": "t1",
                "operation_name": "search_service.search",
                "attributes": {"query": "test"},
                "outputs": {"results": [{"video_id": "v1"}]},
            }
        ])

        results = await evaluator.evaluate_spans(spans_df)
        assert isinstance(results, dict)

    def test_create_mock_spans(self, evaluator):
        mock_df = evaluator._create_mock_spans_df()
        assert len(mock_df) > 0
        assert "span_id" in mock_df.columns
