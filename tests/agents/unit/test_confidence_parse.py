"""parse_confidence — robust LM-output coercion."""

from __future__ import annotations

import pytest

from cogniverse_agents._confidence import parse_confidence


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Plain numeric inputs.
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0, 0.0),
        (1, 1.0),
        # Strings the LM commonly returns.
        ("0.0", 0.0),
        ("0.5", 0.5),
        ("1", 1.0),
        # Percent-string inputs.
        ("85%", 0.85),
        ("100%", 1.0),
        ("0%", 0.0),
        # Bare-integer-in-string inputs above 1 — treated as percent.
        ("85", 0.85),
        ("50", 0.5),
        ("7", 0.07),
        # Sentence-embedded numeric forms real LMs return.
        ("0.9 (very confident)", 0.9),
        ("confidence: 0.9", 0.9),
        ("0.9/1.0", 0.9),
        # Whitespace tolerant.
        ("  0.7  ", 0.7),
        # Label strings.
        ("high", 0.9),
        ("HIGH", 0.9),
        ("medium", 0.5),
        ("low", 0.1),
        # Bool — defined contract: True=1.0, False=0.0.
        (True, 1.0),
        (False, 0.0),
        # Clamping.
        (1.5, 1.0),
        (-0.3, 0.0),
        ("200%", 1.0),
    ],
)
def test_recognised_inputs_map_to_unit_interval(raw, expected) -> None:
    assert parse_confidence(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "unknown",
        "n/a",
        object(),
        "no number here",
        "7 (very confident)",
    ],
)
def test_unrecognised_inputs_fall_back_to_default(raw) -> None:
    assert parse_confidence(raw, default=0.42) == 0.42


def test_returns_float_type_for_every_branch() -> None:
    for raw in [0.5, "0.5", "high", "85%", None, "garbage"]:
        out = parse_confidence(raw, default=0.5)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0
