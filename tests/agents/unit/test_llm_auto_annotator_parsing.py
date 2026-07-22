"""``LLMAutoAnnotator._parse_llm_response`` must survive the confidence shapes
real LMs emit. A naked ``float(data["confidence"])`` crashed on ``"high"`` /
``"85%"``; the broad ``except`` then returned the annotation as a fake parse
failure (label INSUFFICIENT_INFO, confidence 0.0, requires_human_review True),
discarding a valid high-confidence annotation.

The batch tests pin ``max_annotations_per_batch``: a set cap slices the request
list before any LM call so a large identify-run cannot fan out an unbounded
number of LM completions, and the drop is logged rather than silent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
)
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    AutoAnnotation,
    LLMAutoAnnotator,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


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


def _request(index: int) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=f"span-{index}",
        timestamp=datetime.now(timezone.utc),
        query=f"query {index}",
        chosen_agent="video_search",
        routing_confidence=0.5,
        outcome=RoutingOutcome.SUCCESS,
        priority=AnnotationPriority.MEDIUM,
        reason="low confidence",
        context={},
    )


def _capped_annotator(cap: int | None) -> tuple[LLMAutoAnnotator, list[str]]:
    """Annotator whose ``annotate`` records span ids instead of calling an LM."""
    ann = object.__new__(LLMAutoAnnotator)
    ann.max_annotations_per_batch = cap
    called: list[str] = []

    def _recorder(request: AnnotationRequest) -> AutoAnnotation:
        called.append(request.span_id)
        return AutoAnnotation(
            span_id=request.span_id,
            label=AnnotationLabel.CORRECT,
            confidence=0.9,
            reasoning="recorded",
            suggested_correct_agent=None,
            requires_human_review=False,
        )

    ann.annotate = _recorder
    return ann, called


def test_batch_cap_truncates_and_logs(caplog) -> None:
    ann, called = _capped_annotator(3)
    requests = [_request(i) for i in range(5)]

    with caplog.at_level(logging.INFO):
        out = ann.batch_annotate(requests)

    assert len(out) == 3
    assert called == ["span-0", "span-1", "span-2"]
    assert [a.span_id for a in out] == ["span-0", "span-1", "span-2"]
    cap_logs = [r.message for r in caplog.records if "capping" in r.message.lower()]
    assert len(cap_logs) == 1
    assert "5" in cap_logs[0] and "3" in cap_logs[0]


def test_batch_cap_none_processes_all(caplog) -> None:
    ann, called = _capped_annotator(None)
    requests = [_request(i) for i in range(5)]

    with caplog.at_level(logging.INFO):
        out = ann.batch_annotate(requests)

    assert len(out) == 5
    assert called == [f"span-{i}" for i in range(5)]
    assert not [r for r in caplog.records if "capping" in r.message.lower()]


def test_batch_cap_above_count_processes_all(caplog) -> None:
    ann, called = _capped_annotator(10)
    requests = [_request(i) for i in range(5)]

    with caplog.at_level(logging.INFO):
        out = ann.batch_annotate(requests)

    assert len(out) == 5
    assert called == [f"span-{i}" for i in range(5)]
    assert not [r for r in caplog.records if "capping" in r.message.lower()]
