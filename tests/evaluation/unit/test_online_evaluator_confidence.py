"""OnlineEvaluator coerces a label-shaped routing confidence span attribute."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from cogniverse_evaluation.online_evaluator import OnlineEvaluator

pytestmark = pytest.mark.unit


def test_label_confidence_is_coerced_not_crashed():
    ev = OnlineEvaluator(provider=Mock(), project_name="p")
    span = {"attributes.routing": {"confidence": "high"}, "status_code": "OK"}

    out = ev._eval_confidence_calibration(span, "span-1")

    # "high" -> 0.9; success path keeps calibration == confidence.
    assert out.score == pytest.approx(0.9)
    assert out.label == "well_calibrated"


def test_missing_confidence_uses_default():
    ev = OnlineEvaluator(provider=Mock(), project_name="p")
    span = {"attributes.routing": {}, "status_code": "OK"}

    out = ev._eval_confidence_calibration(span, "span-2")

    assert out.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_persist_failure_raises_after_attempting_all():
    """An annotation-store outage must surface, not read as persisted success
    — and every result is still attempted before raising (no early abort
    hiding which scores made it)."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock

    from cogniverse_evaluation.online_evaluator import OnlineEvalResult

    provider = Mock()
    attempted = []

    async def flaky_add(**kwargs):
        attempted.append(kwargs["span_id"])
        if kwargs["span_id"] == "s2":
            raise ConnectionError("annotation store down")

    provider.annotations.add_annotation = AsyncMock(side_effect=flaky_add)
    ev = OnlineEvaluator(provider=provider, project_name="p")

    results = [
        OnlineEvalResult(
            span_id=f"s{i}",
            evaluator_name="routing_outcome",
            score=0.5,
            label="ok",
            explanation="x",
            timestamp=datetime.now(timezone.utc),
        )
        for i in (1, 2, 3)
    ]

    with pytest.raises(RuntimeError, match="1/3"):
        await ev._persist_results(results)

    assert attempted == ["s1", "s2", "s3"]
