"""Defensive numeric coercion for untrusted agent/KG payload fields.

Sub-agent A2A results and KG mention/edge blobs carry ``ts_start`` /
``ts_end`` / ``score`` as arbitrary JSON — a real agent may emit ``"00:12"``,
``"high"``, ``""`` or ``null``. A naked ``float()`` raises ``ValueError`` /
``TypeError`` and crashes the evidence walk or the traversal time-window
filter. :func:`coerce_float` maps every shape to a finite float and falls
back to ``default`` on anything non-numeric or non-finite.

Unlike :func:`cogniverse_foundation.confidence.parse_confidence` it does NOT
clamp to ``[0, 1]`` — timestamps and ranking scores are unbounded.
"""

from __future__ import annotations

import math


def coerce_float(raw: object, default: float = 0.0) -> float:
    """Coerce untrusted input to a finite float, else ``default``."""
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def coerce_int(raw: object, default: int) -> int:
    """Coerce untrusted input to an int, else ``default``.

    Accepts int / float / numeric string; ``None``, empty, and non-numeric
    fall back. Floats truncate toward zero (``int(3.7) == 3``).
    """
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return int(value)


def coerce_bool(raw: object, default: bool = False) -> bool:
    """Coerce untrusted input to a bool, else ``default``.

    A real bool passes through. A string is truthy only for the affirmative
    tokens (``"true"``/``"1"``/``"yes"``/``"on"``), so the string ``"false"``
    is False (plain ``bool("false")`` is True). Numbers use their truthiness.
    """
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("true", "1", "yes", "on"):
            return True
        if token in ("false", "0", "no", "off", ""):
            return False
        return default
    if isinstance(raw, (int, float)):
        return bool(raw)
    return default
