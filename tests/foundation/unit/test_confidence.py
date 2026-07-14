"""parse_confidence — robust LM-output coercion (canonical foundation home)."""

from __future__ import annotations

import pytest

from cogniverse_foundation.confidence import parse_confidence


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0, 0.0),
        (1, 1.0),
        ("0.5", 0.5),
        ("1", 1.0),
        ("85%", 0.85),
        ("100%", 1.0),
        ("85", 0.85),
        ("high", 0.9),
        ("medium", 0.5),
        ("low", 0.1),
        ("HIGH", 0.9),
        (True, 1.0),
        (False, 0.0),
        (1.5, 1.0),
        (-0.2, 0.0),
    ],
)
def test_parse_confidence_shapes(raw, expected):
    assert parse_confidence(raw) == pytest.approx(expected)


def test_parse_confidence_falls_back_on_unparseable():
    assert parse_confidence("definitely", default=0.3) == pytest.approx(0.3)
    assert parse_confidence(None, default=0.7) == pytest.approx(0.7)
    assert parse_confidence("", default=0.42) == pytest.approx(0.42)


def test_agents_reexport_is_same_function():
    from cogniverse_agents._confidence import parse_confidence as agents_pc

    assert agents_pc is parse_confidence


@pytest.mark.parametrize(
    "raw",
    ["nan", "NaN", "+nan", "-nan", float("nan")],
    ids=["nan_str", "NaN_str", "plus_nan", "minus_nan", "nan_float"],
)
def test_nan_inputs_fall_back_to_default(raw):
    """NaN survives the [0,1] clamp (nan<0 and nan>1 are both False) and
    would propagate into eval scores and routing comparisons — it must map
    to the default like any other unparseable input."""
    assert parse_confidence(raw, default=0.42) == pytest.approx(0.42)
    assert parse_confidence(raw) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "raw,expected",
    [("inf", 1.0), ("-inf", 0.0), (float("inf"), 1.0), (float("-inf"), 0.0)],
    ids=["inf_str", "minus_inf_str", "inf_float", "minus_inf_float"],
)
def test_inf_inputs_clamp_to_bounds(raw, expected):
    assert parse_confidence(raw) == pytest.approx(expected)
