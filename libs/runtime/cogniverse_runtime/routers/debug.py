"""Debug endpoints for runtime diagnostics (gated behind ``COGNIVERSE_DEBUG_MEM``).

``POST /debug/memsnap`` takes a tracemalloc snapshot, diffs it against the
previous one held in memory, and returns the top allocation sites whose
retained size grew the most. The endpoint stays dark until the env var
flips so it has zero startup cost in production.
"""

from __future__ import annotations

import logging
import os
import tracemalloc
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

_prev_snapshot: Optional[tracemalloc.Snapshot] = None


def _debug_enabled() -> bool:
    return os.environ.get("COGNIVERSE_DEBUG_MEM", "").lower() in ("1", "true", "yes")


def _format_frame(stat: tracemalloc.Statistic) -> Dict[str, Any]:
    frame = stat.traceback[0] if stat.traceback else None
    return {
        "file": frame.filename if frame else "<unknown>",
        "line": frame.lineno if frame else 0,
        "size_bytes": stat.size,
        "count": stat.count,
    }


@router.post("/memsnap")
def memsnap(top_n: int = 25, mode: str = "lineno") -> Dict[str, Any]:
    """Take a tracemalloc snapshot and diff against the previous one.

    Args:
        top_n: How many allocation sites to return.
        mode: Snapshot grouping — ``lineno`` (per file:line) or ``filename``.

    Returns:
        ``{
            "started": <bool — True iff this call started tracing>,
            "total_mb": <current traced size>,
            "top_current": [ {file, line, size_bytes, count}, ... ],
            "top_growth":  [ {file, line, size_bytes, count}, ... ]   # diff vs previous
        }``
    """
    global _prev_snapshot

    if not _debug_enabled():
        raise HTTPException(
            status_code=403,
            detail="set COGNIVERSE_DEBUG_MEM=1 to enable memsnap",
        )

    started = False
    if not tracemalloc.is_tracing():
        # 25 is deep enough to distinguish call sites without taxing memory.
        tracemalloc.start(25)
        started = True

    snap = tracemalloc.take_snapshot()
    total = sum(stat.size for stat in snap.statistics(mode))

    top_current = [_format_frame(s) for s in snap.statistics(mode)[:top_n]]

    top_growth: List[Dict[str, Any]] = []
    if _prev_snapshot is not None:
        diff = snap.compare_to(_prev_snapshot, mode)
        # Sort by size growth descending; drop sites that shrank.
        diff = [d for d in diff if d.size_diff > 0]
        diff.sort(key=lambda d: d.size_diff, reverse=True)
        for d in diff[:top_n]:
            frame = d.traceback[0] if d.traceback else None
            top_growth.append(
                {
                    "file": frame.filename if frame else "<unknown>",
                    "line": frame.lineno if frame else 0,
                    "size_bytes": d.size,
                    "size_diff_bytes": d.size_diff,
                    "count": d.count,
                    "count_diff": d.count_diff,
                }
            )

    _prev_snapshot = snap

    return {
        "started": started,
        "total_mb": round(total / 1024 / 1024, 2),
        "top_current": top_current,
        "top_growth": top_growth,
    }


@router.post("/memreset")
def memreset() -> Dict[str, Any]:
    """Stop tracing and drop the previous snapshot reference."""
    global _prev_snapshot

    if not _debug_enabled():
        raise HTTPException(
            status_code=403,
            detail="set COGNIVERSE_DEBUG_MEM=1 to enable memreset",
        )

    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    _prev_snapshot = None

    return {"was_tracing": was_tracing}
