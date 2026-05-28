"""Robust LM-output confidence parsing.

DSPy modules backed by real LMs return confidence as floats, percent
strings (``"85%"``), label strings (``"high"`` / ``"medium"`` / ``"low"``),
or empty strings. A naked ``float(result.confidence)`` crashes the route
on any of the non-numeric shapes; the FastAPI handler then surfaces it as
a 500. Callers should use :func:`parse_confidence`, which maps every
shape to a value in ``[0.0, 1.0]`` and falls back to ``default`` on any
input it cannot interpret.
"""

from __future__ import annotations

_LABEL_BANDS = {"high": 0.9, "medium": 0.5, "low": 0.1}


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
            v = _LABEL_BANDS.get(s, default)
        # Strings the LM writes as "85" or "85%" mean 0.85 — only convert
        # when the input is a string AND the parsed value exceeds 1.
        if had_percent or v > 1.0:
            v = v / 100.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
