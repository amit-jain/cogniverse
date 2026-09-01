"""The order in which tab candidates are tried, and which are eligible.

Nested Streamlit tabs make one label ambiguous: the dashboard renders a
top-level ``🔬 Synthetic Data & Optimization`` tab and, inside it, a
``🔬 Synthetic Data`` sub-tab. Asking for "Synthetic Data" must reach the
sub-tab. Falling back to the substring match reaches the parent instead --
and because the parent is already selected by then, the click is a no-op
that reports success, leaving the test reading the wrong panel.
"""

from __future__ import annotations

import pytest

from tests.e2e.tab_selection import tab_candidates

# The real labels, read off the running dashboard.
NESTED = [
    ("🔬 Synthetic Data & Optimization", True),
    ("📊 Overview", True),
    ("🔬 Synthetic Data", True),
    ("🎯 Module Optimization", True),
]


def test_an_exact_match_excludes_the_substring_parent():
    """Asking for the sub-tab must never offer the parent as a candidate."""
    assert tab_candidates(NESTED, "Synthetic Data") == [2]


def test_substring_is_used_only_when_nothing_matches_exactly():
    """The parent is reachable by its own full label, and by nothing shorter."""
    assert tab_candidates(NESTED, "Synthetic Data & Optimization") == [0]
    assert tab_candidates(NESTED, "Module Optimization") == [3]


def test_no_exact_match_falls_back_to_substring_in_dom_order():
    labels = [("📊 Overview", True), ("🎯 Module Optimization", True)]
    assert tab_candidates(labels, "Optimization") == [1]


def test_visible_tabs_are_tried_before_hidden_ones():
    labels = [
        ("🔬 Synthetic Data", False),
        ("📊 Overview", True),
        ("🔬 Synthetic Data", True),
    ]
    assert tab_candidates(labels, "Synthetic Data") == [2, 0]


def test_emoji_and_case_are_ignored_when_matching():
    assert tab_candidates([("🧠 Memory", True)], "memory") == [0]
    assert tab_candidates([("🧠 MEMORY", True)], "Memory") == [0]


def test_a_label_that_names_no_tab_yields_no_candidates():
    assert tab_candidates(NESTED, "Nonexistent") == []


@pytest.mark.parametrize("target", ["", "   "])
def test_a_blank_target_is_rejected_rather_than_matching_everything(target):
    with pytest.raises(ValueError, match="non-empty"):
        tab_candidates(NESTED, target)
