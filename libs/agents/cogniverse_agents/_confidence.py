"""Robust LM-output confidence parsing.

Canonical implementation lives in :mod:`cogniverse_foundation.confidence`;
re-exported here so the agents package keeps its established import path.
"""

from __future__ import annotations

from cogniverse_foundation.confidence import _LABEL_BANDS, parse_confidence

__all__ = ["_LABEL_BANDS", "parse_confidence"]
