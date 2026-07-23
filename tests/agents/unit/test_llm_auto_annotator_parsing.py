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
    # PUTs run through a thread pool, so call order is not deterministic — the
    # SET of processed spans is the first 3, and the RESULT stays in request
    # order.
    assert set(called) == {"span-0", "span-1", "span-2"}
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
    assert set(called) == {f"span-{i}" for i in range(5)}
    assert [a.span_id for a in out] == [f"span-{i}" for i in range(5)]
    assert not [r for r in caplog.records if "capping" in r.message.lower()]


def test_batch_cap_above_count_processes_all(caplog) -> None:
    ann, called = _capped_annotator(10)
    requests = [_request(i) for i in range(5)]

    with caplog.at_level(logging.INFO):
        out = ann.batch_annotate(requests)

    assert len(out) == 5
    assert set(called) == {f"span-{i}" for i in range(5)}
    assert [a.span_id for a in out] == [f"span-{i}" for i in range(5)]
    assert not [r for r in caplog.records if "capping" in r.message.lower()]


def test_batch_annotate_preserves_request_order_under_parallelism() -> None:
    """The pool completes spans out of order; the result list must still be in
    request order (a downstream persist zips it against the request list)."""
    import threading
    import time

    ann = object.__new__(LLMAutoAnnotator)
    ann.max_annotations_per_batch = None
    lock = threading.Lock()
    completion_order: list[int] = []

    def _slow(request: AnnotationRequest) -> AutoAnnotation:
        idx = int(request.span_id.rsplit("-", 1)[1])
        time.sleep(0.02 * (8 - idx))  # earlier spans finish last
        with lock:
            completion_order.append(idx)
        return AutoAnnotation(
            span_id=request.span_id,
            label=AnnotationLabel.CORRECT,
            confidence=0.9,
            reasoning="ok",
            suggested_correct_agent=None,
            requires_human_review=False,
        )

    ann.annotate = _slow
    requests = [_request(i) for i in range(8)]
    out = ann.batch_annotate(requests)

    assert completion_order[0] > completion_order[-1]  # genuinely out of order
    assert [a.span_id for a in out] == [f"span-{i}" for i in range(8)]


def test_batch_annotate_raises_on_lm_outage_not_fabricated_annotations() -> None:
    """When the LM endpoint is unreachable, batch_annotate RAISES rather than
    returning a full batch of fabricated INSUFFICIENT_INFO verdicts that the
    optimization loop would consume as real signal."""
    ann = object.__new__(LLMAutoAnnotator)
    ann.max_annotations_per_batch = None

    def _down(request: AnnotationRequest) -> AutoAnnotation:
        raise ConnectionError("LM endpoint refused connection")

    ann.annotate = _down
    with pytest.raises(ConnectionError, match="refused connection"):
        ann.batch_annotate([_request(i) for i in range(4)])


def test_annotate_propagates_transport_error(monkeypatch) -> None:
    """A completion() transport failure propagates out of annotate() — it is NOT
    laundered into a confidence-0.0 annotation the caller reads as a real
    verdict."""
    ann = object.__new__(LLMAutoAnnotator)
    ann.model = "openai/x"
    ann.api_base = "http://127.0.0.1:9"
    ann.api_key = "k"

    def _boom(**_kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(
        "cogniverse_agents.routing.llm_auto_annotator.completion", _boom
    )
    with pytest.raises(ConnectionError, match="connection refused"):
        ann.annotate(_request(0))


def test_annotate_degrades_only_malformed_content_not_outage(monkeypatch) -> None:
    """A well-formed completion whose body is not our JSON schema degrades to a
    per-span review-needed annotation (content issue), while the surrounding
    outage path still raises — the two failure modes stay distinct."""
    ann = object.__new__(LLMAutoAnnotator)
    ann.model = "openai/x"
    ann.api_base = None
    ann.api_key = None

    class _Msg:
        content = "the model rambled instead of returning json"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    monkeypatch.setattr(
        "cogniverse_agents.routing.llm_auto_annotator.completion",
        lambda **_: _Resp(),
    )
    out = ann.annotate(_request(0))
    assert out.label is AnnotationLabel.INSUFFICIENT_INFO
    assert out.requires_human_review is True
