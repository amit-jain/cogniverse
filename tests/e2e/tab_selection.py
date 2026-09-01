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


_SCOPES = {"top": False, "sub": True}


def tab_candidates_in_scope(
    tabs: Sequence[tuple[str, bool, bool]] | Iterable[tuple[str, bool, bool]],
    target: str,
    scope: str,
) -> list[int]:
    """Indices of the tabs to try for ``target`` within one tab strip.

    ``tabs`` is ``(label, is_visible, is_nested)`` in DOM order, where
    ``is_nested`` is true for a tab rendered inside a ``[role="tabpanel"]``
    -- that is, a sub-tab of some other tab.

    Scoping is what disambiguates a label that names two tabs. A page-wide
    search for "Synthetic Data" finds an exact match on the nested sub-tab and
    only a substring match on the top-level "Synthetic Data & Optimization"
    parent, so a caller asking for the parent is sent into a panel that is not
    open and every click is swallowed. Which strip a tab lives in is the fact
    that separates them; preferring an exact match is a proxy for it that
    happens to point the wrong way here.

    Within a scope the ordering rule is unchanged: exact before substring,
    visible before hidden.
    """

    if scope not in _SCOPES:
        raise ValueError(f"tab scope must be one of {sorted(_SCOPES)}, got {scope!r}")
    want_nested = _SCOPES[scope]

    entries = list(tabs)
    in_scope = [
        (dom_index, label, visible)
        for dom_index, (label, visible, nested) in enumerate(entries)
        if nested is want_nested
    ]
    ordered = tab_candidates(
        [(label, visible) for _, label, visible in in_scope], target
    )
    return [in_scope[i][0] for i in ordered]
