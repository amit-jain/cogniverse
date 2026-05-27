"""Unit test for PhoenixEvaluationProvider.log_session_evaluation wiring.

The annotation store's ``add_annotation`` requires a ``project`` argument.
The prior code omitted it, so the call raised TypeError inside a fire-and-forget
task whose exception was swallowed — the dashboard reported "Evaluation saved"
while nothing persisted. This pins that ``project`` (resolved from the
provider's configured project name) is passed through.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
    PhoenixEvaluationProvider,
)


@pytest.mark.unit
def test_log_session_evaluation_passes_project_to_annotation_store():
    provider = PhoenixEvaluationProvider()
    provider._initialized = True
    provider._project_name = "cogniverse-search"
    annotations = MagicMock()
    annotations.add_annotation = AsyncMock(return_value="ann-1")
    provider._telemetry_provider = MagicMock(annotations=annotations)

    # Called from a sync context (no running loop), so log_session_evaluation
    # awaits the annotation write before returning.
    provider.log_session_evaluation(
        session_id="span-123",
        evaluation_name="dashboard_annotation",
        session_score=0.8,
        session_outcome="good",
    )

    annotations.add_annotation.assert_awaited_once()
    kwargs = annotations.add_annotation.await_args.kwargs
    assert kwargs["project"] == "cogniverse-search"
    assert kwargs["span_id"] == "span-123"
    assert kwargs["score"] == 0.8
    assert kwargs["label"] == "good"
