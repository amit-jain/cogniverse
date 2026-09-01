"""Which Streamlit tab a label refers to, and in what order to try them.

Kept out of ``conftest.py`` so it is importable (and testable) without
executing the e2e session fixtures, which provision a cluster.

The dashboard nests tabs, so one label can name two of them: a top-level
``🔬 Synthetic Data & Optimization`` and, inside it, a ``🔬 Synthetic Data``
sub-tab. A substring fallback reaches the parent, and by the time a caller
asks for the sub-tab the parent is already selected -- so clicking it is a
no-op that reports success and leaves the caller reading the parent's panel.
An exact match therefore suppresses the substring fallback entirely:
activating a tab other than the one named is never the right answer.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_LEADING_EMOJI = re.compile(r"^[\U0001f300-\U0001faff☀-➿️‍]+\s*")


def normalize_label(text: str | None) -> str:
    """A tab's comparable text: leading emoji stripped, trimmed, lowercased."""

    return _LEADING_EMOJI.sub("", text or "").strip().lower()


def tab_candidates(
    tabs: Sequence[tuple[str, bool]] | Iterable[tuple[str, bool]],
    target: str,
) -> list[int]:
    """Indices of the tabs to try for ``target``, best first.

    ``tabs`` is ``(label, is_visible)`` in DOM order. Visible tabs come before
    hidden ones because a hidden tab needs a forced click, which is a weaker
    signal that the caller reached what it asked for.
    """

    wanted = normalize_label(target)
    if not wanted:
        raise ValueError("tab label must be non-empty after normalization")

    entries = [
        (i, normalize_label(label), visible) for i, (label, visible) in enumerate(tabs)
    ]

    exact = [(i, visible) for i, label, visible in entries if label == wanted]
    if exact:
        return [i for i, visible in exact if visible] + [
            i for i, visible in exact if not visible
        ]

    partial = [(i, visible) for i, label, visible in entries if wanted in label]
    return [i for i, visible in partial if visible] + [
        i for i, visible in partial if not visible
    ]
