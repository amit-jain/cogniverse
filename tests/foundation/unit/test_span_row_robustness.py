"""Search-result span rows survive hostile inputs.

``search_result_row`` coerced ``float(score)`` unguarded — a non-numeric score
raised ValueError and a NaN passed through into ``output.value``, producing
invalid JSON (``"score": NaN``) that strict consumers (Phoenix UI, JS
``JSON.parse``) reject. ``add_search_results_to_span``'s top-3 event crashed on
dict rows and non-enum content types while its sibling ``search_result_row``
explicitly accepts dicts — an internal contract asymmetry.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from cogniverse_foundation.telemetry.context import (
    add_search_results_to_span,
    serialize_search_results,
)
from cogniverse_foundation.telemetry.span_contract import search_result_row

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingSpan:
    def __init__(self):
        self.attrs = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def add_event(self, name, payload):
        self.events.append((name, payload))


def test_non_numeric_score_coerces_to_zero_instead_of_raising():
    row = search_result_row({"id": "d1", "score": "high", "content": "x"})
    assert row["score"] == 0.0
    assert row["document_id"] == "d1"


def test_nan_score_never_reaches_the_json_payload():
    payload = serialize_search_results(
        [{"id": "d1", "score": float("nan"), "content": "x"}]
    )
    rows = json.loads(payload)  # strict json — NaN would fail to parse
    assert rows[0]["score"] == 0.0
    assert math.isfinite(rows[0]["score"])


def test_inf_score_never_reaches_the_json_payload():
    row = search_result_row({"id": "d1", "score": float("inf")})
    assert row["score"] == 0.0


def test_top3_event_accepts_dict_rows():
    span = _RecordingSpan()
    add_search_results_to_span(
        span,
        [
            {"id": "d1", "score": 0.9, "source_id": "vid1", "content": "a"},
            {"id": "d2", "score": 0.5, "content": "b"},
        ],
    )
    assert span.attrs["num_results"] == 2
    assert span.attrs["top_score"] == 0.9
    (name, payload) = span.events[0]
    assert name == "search_results"
    assert "d1" in payload["top_3"]


def test_top3_event_accepts_string_content_type():
    doc = SimpleNamespace(id="d1", metadata={"source_id": "vid1"}, content_type="video")
    span = _RecordingSpan()
    add_search_results_to_span(span, [SimpleNamespace(document=doc, score=0.7)])
    (_, payload) = span.events[0]
    assert "video" in payload["top_3"]
