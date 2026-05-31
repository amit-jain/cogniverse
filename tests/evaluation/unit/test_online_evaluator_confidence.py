"""OnlineEvaluator coerces a label-shaped routing confidence span attribute."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from cogniverse_evaluation.online_evaluator import OnlineEvaluator


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
