"""Unit tests for SpanEvaluator — span retrieval and evaluation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_evaluation.span_evaluator import SpanEvaluator


@pytest.fixture
def evaluator():
    mock_provider = MagicMock()
    mock_provider.telemetry = MagicMock()
    mock_provider.telemetry.traces = MagicMock()
    mock_provider.telemetry.traces.get_spans = AsyncMock(return_value=None)

    ev = SpanEvaluator(
        provider=mock_provider,
        tenant_id="test",
        project_name="test-project",
    )
    return ev


@pytest.mark.unit
class TestSpanEvaluator:
    def test_create_mock_spans(self, evaluator):
        mock_df = evaluator._create_mock_spans_df()
        assert len(mock_df) > 0
        assert "span_id" in mock_df.columns
