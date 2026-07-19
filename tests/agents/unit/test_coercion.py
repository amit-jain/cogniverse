"""coerce_float maps every untrusted shape to a finite float or the default.

The KG-reading agents (temporal_reasoning, citation_tracing,
contradiction_reconciliation, knowledge_summarization, audit_explanation) route
edge-field values (ts_start / ts_end / confidence) through this helper — matching
the kg_traversal and orchestrator siblings — so a non-numeric value from a future
graph read path degrades to the default instead of raising mid-parse.
"""

from __future__ import annotations

import pytest

from cogniverse_agents._coercion import coerce_bool, coerce_float, coerce_int

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1.5, 1.5),
        (2, 2.0),
        ("3.25", 3.25),
        (0, 0.0),
        ("0.0", 0.0),
    ],
)
def test_valid_inputs_pass_through(raw, expected):
    assert coerce_float(raw) == expected


@pytest.mark.parametrize("raw", [None, "high", "", "85%", [], {}, object()])
def test_non_numeric_falls_back_to_default(raw):
    assert coerce_float(raw) == 0.0


def test_non_finite_falls_back_to_default():
    assert coerce_float(float("inf")) == 0.0
    assert coerce_float(float("nan")) == 0.0
    assert coerce_float("nan") == 0.0


def test_custom_default_is_honored():
    assert coerce_float(None, default=-1.0) == -1.0
    assert coerce_float("bad", default=9.9) == 9.9


@pytest.mark.parametrize(
    "raw,expected",
    [(5, 5), ("7", 7), (3.7, 3), ("3.7", 3), (0, 0), (True, 1)],
)
def test_coerce_int_valid_inputs(raw, expected):
    assert coerce_int(raw, 10) == expected


@pytest.mark.parametrize("raw", [None, "", "many", "5x", [], {}, float("nan")])
def test_coerce_int_bad_inputs_fall_back(raw):
    # The exact a2a-metadata shapes that crashed int(metadata["top_k"]).
    assert coerce_int(raw, 10) == 10


@pytest.mark.parametrize("raw", [True, "true", "True", "1", "yes", "on", 1, 2.0])
def test_coerce_bool_truthy(raw):
    assert coerce_bool(raw) is True


@pytest.mark.parametrize("raw", [False, "false", "False", "0", "no", "off", "", 0])
def test_coerce_bool_falsy(raw):
    # Notably the string "false" — plain bool("false") is True.
    assert coerce_bool(raw) is False


def test_coerce_bool_unknown_string_uses_default():
    assert coerce_bool("maybe") is False
    assert coerce_bool("maybe", default=True) is True
    assert coerce_bool(None, default=True) is True
