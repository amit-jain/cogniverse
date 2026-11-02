"""VLMInterface coerces an LM-shaped relevance_score without crashing."""

from __future__ import annotations

from unittest.mock import Mock

import dspy
import pytest

from cogniverse_core.common.vlm_interface import VLMInterface


class _StubResult:
    descriptions = "frame one, frame two"
    themes = ""
    key_objects = ""
    insights = ""
    relevance_score = "high"


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_label_relevance_score_is_coerced(monkeypatch):
    vlm = object.__new__(VLMInterface)
    vlm._dspy_lm = Mock()

    # The real LM commonly returns a label like "high" rather than a float;
    # a bare float() crashed here before parse_confidence.
    class _StubPredict:
        def __init__(self, signature):
            pass

        async def acall(self, **kwargs):
            return _StubResult()

    monkeypatch.setattr(dspy, "Predict", _StubPredict)

    out = await vlm.analyze_visual_content([], "find safety gear")

    assert out["relevance_score"] == pytest.approx(0.9)
    assert out["descriptions"] == ["frame one", "frame two"]
    assert out["themes"] == []
