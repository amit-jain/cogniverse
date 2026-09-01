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

from tests.e2e.tab_selection import tab_candidates, tab_candidates_in_scope

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


# --- scope: a top-level tab and a nested sub-tab can share a label ----------
# The dashboard nests a "🔬 Synthetic Data" sub-tab inside a
# "🔬 Synthetic Data & Optimization" parent. Searching the whole page for
# "Synthetic Data" finds an exact match on the *sub*-tab, so a caller asking
# for the top tab is sent into an unopened panel and every click is swallowed.
# Which strip a tab lives in is the fact that disambiguates them; a
# label-shape heuristic is only a proxy for it.

NESTED_PAIR = [
    ("🔬 Synthetic Data & Optimization", True, False),  # 0: top-level strip
    ("📊 Overview", False, True),  # 1: inside the parent's panel
    ("🔬 Synthetic Data", False, True),  # 2: inside the parent's panel
]


def test_a_top_level_search_reaches_the_parent_not_the_nested_exact_match():
    assert tab_candidates_in_scope(NESTED_PAIR, "Synthetic Data", "top") == [0]


def test_a_sub_level_search_reaches_the_nested_tab_not_the_parent():
    assert tab_candidates_in_scope(NESTED_PAIR, "Synthetic Data", "sub") == [2]


def test_a_top_level_search_never_offers_a_nested_tab_even_with_no_top_match():
    assert tab_candidates_in_scope(NESTED_PAIR, "Overview", "top") == []


def test_a_sub_level_search_never_offers_a_top_level_tab():
    tabs = [("🔧 Optimization", True, False), ("🎯 Module Optimization", True, True)]
    assert tab_candidates_in_scope(tabs, "Optimization", "sub") == [1]


def test_scope_preserves_the_visible_before_hidden_ordering():
    tabs = [
        ("📊 Overview", False, True),
        ("📊 Overview", True, True),
    ]
    assert tab_candidates_in_scope(tabs, "Overview", "sub") == [1, 0]


@pytest.mark.parametrize("scope", ["", "TOP", "toplevel", None])
def test_an_unknown_scope_is_rejected_rather_than_silently_matching_everything(scope):
    with pytest.raises(ValueError):
        tab_candidates_in_scope(NESTED_PAIR, "Synthetic Data", scope)
