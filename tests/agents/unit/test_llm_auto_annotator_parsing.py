"""``LLMAutoAnnotator._parse_llm_response`` must survive the confidence shapes
real LMs emit. A naked ``float(data["confidence"])`` crashed on ``"high"`` /
``"85%"``; the broad ``except`` then returned the annotation as a fake parse
failure (label INSUFFICIENT_INFO, confidence 0.0, requires_human_review True),
discarding a valid high-confidence annotation.
"""

from __future__ import annotations

import json

from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    LLMAutoAnnotator,
)


def _annotator() -> LLMAutoAnnotator:
    # _parse_llm_response reads no instance attributes; skip the LM-config
    # __init__.
    return object.__new__(LLMAutoAnnotator)


def _payload(confidence: object) -> str:
    return json.dumps(
        {
            "label": "correct_routing",
            "confidence": confidence,
            "reasoning": "agent matched the query intent",
            "requires_human_review": False,
        }
    )


def test_label_band_string_confidence_parses_not_crashes() -> None:
    out = _annotator()._parse_llm_response(_payload("high"), "span-1")
    assert out.label is AnnotationLabel.CORRECT_ROUTING
    assert out.confidence == 0.9
    assert out.requires_human_review is False
    assert "Error parsing" not in out.reasoning


def test_percent_string_confidence_scales_to_fraction() -> None:
    out = _annotator()._parse_llm_response(_payload("85%"), "span-2")
    assert out.confidence == 0.85
    assert out.label is AnnotationLabel.CORRECT_ROUTING


def test_numeric_confidence_preserved() -> None:
    out = _annotator()._parse_llm_response(_payload(0.72), "span-3")
    assert out.confidence == 0.72


def test_missing_confidence_uses_default() -> None:
    payload = json.dumps({"label": "ambiguous", "reasoning": "unclear"})
    out = _annotator()._parse_llm_response(payload, "span-4")
    assert out.confidence == 0.5
    assert out.label is AnnotationLabel.AMBIGUOUS


def test_uninterpretable_confidence_falls_back_to_default() -> None:
    out = _annotator()._parse_llm_response(_payload("banana"), "span-5")
    assert out.confidence == 0.5
    # A garbage confidence must NOT collapse the annotation into a parse
    # failure — the label the LM gave still stands.
    assert out.label is AnnotationLabel.CORRECT_ROUTING


def test_malformed_json_still_returns_parse_failure() -> None:
    out = _annotator()._parse_llm_response("not json at all", "span-6")
    assert out.label is AnnotationLabel.INSUFFICIENT_INFO
    assert out.confidence == 0.0
    assert out.requires_human_review is True
