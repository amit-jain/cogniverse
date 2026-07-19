"""coerce_float maps every untrusted shape to a finite float or the default.

The KG-reading agents (temporal_reasoning, citation_tracing,
contradiction_reconciliation, knowledge_summarization, audit_explanation) route
edge-field values (ts_start / ts_end / confidence) through this helper — matching
the kg_traversal and orchestrator siblings — so a non-numeric value from a future
graph read path degrades to the default instead of raising mid-parse.
"""

from __future__ import annotations

import pytest

from cogniverse_agents._coercion import coerce_float

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
