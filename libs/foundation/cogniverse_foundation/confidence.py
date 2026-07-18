"""Robust LM-output confidence parsing.

DSPy modules backed by real LMs return confidence/relevance as floats,
percent strings (``"85%"``), label strings (``"high"`` / ``"medium"`` /
``"low"``), sentence-embedded numbers (``"0.9 (very confident)"``,
``"confidence: 0.9"``, ``"0.9/1.0"``), or empty strings. A naked
``float(result.confidence)`` crashes the caller on any non-numeric shape.
:func:`parse_confidence` maps every shape to a value in ``[0.0, 1.0]`` and
falls back to ``default`` on any input it cannot interpret.

Lives in foundation so every layer (core, agents, evaluation, finetuning)
can share one implementation.
"""

from __future__ import annotations

import math
import re

_LABEL_BANDS = {"high": 0.9, "medium": 0.5, "low": 0.1}

# Lookarounds keep digits embedded in larger tokens — hex reprs ("0x7f.."),
# versions ("1.2.3"), identifiers — from reading as confidence values.
_RATIO_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)(?!\w|\.\d)"
)
_FLOAT_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")


def _extract_unit_interval(s: str) -> float | None:
    """Bounded numeric extraction from an LM sentence.

    An ``x/y`` ratio counts only when it lands in ``[0, 1]``; otherwise the
    first float token in ``[0, 1]`` wins. Out-of-range numbers never match,
    so "7 (very confident)" stays uninterpretable rather than guessing.
    """
    m = _RATIO_RE.search(s)
    if m:
        denominator = float(m.group(2))
        if denominator != 0.0:
            ratio = float(m.group(1)) / denominator
            if 0.0 <= ratio <= 1.0:
                return ratio
    for token in _FLOAT_RE.finditer(s):
        v = float(token.group())
        if 0.0 <= v <= 1.0:
            return v
    return None


def parse_confidence(raw: object, default: float = 0.0) -> float:
    """Map an LM confidence output to a clamped float in ``[0.0, 1.0]``."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        # Numeric input is taken at face value, then clamped — a caller
        # passing 1.5 means "saturate at the top", not "1.5%".
        v = float(raw)
    else:
        s = str(raw).strip().lower()
        had_percent = s.endswith("%")
        s = s.rstrip("%").strip()
        if not s:
            return default
        try:
            v = float(s)
        except ValueError:
            if s in _LABEL_BANDS:
                v = _LABEL_BANDS[s]
            else:
                extracted = _extract_unit_interval(s)
                v = extracted if extracted is not None else default
        # Strings the LM writes as "85" or "85%" mean 0.85 — only convert
        # when the input is a string AND the parsed value exceeds 1.
        if had_percent or v > 1.0:
            v = v / 100.0
    if math.isnan(v):
        # NaN survives the range clamp (nan<0 and nan>1 are both False)
        # and would propagate into eval scores and routing comparisons.
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
