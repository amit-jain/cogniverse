"""Widget locators in the dashboard e2e suite must be scoped to the open panel.

Streamlit renders *every* tab body into the DOM at once -- 54 panels at the
time of writing -- so a locator rooted at ``page`` also matches widgets
belonging to tabs the test never opened.  A page-wide ``count() > 0`` is then
satisfied by some other tab's content and the assertion passes whether or not
the tab under test rendered anything at all.

The rule is structural rather than a list of known offenders: a locator that
targets a widget must be rooted at a panel, and the handful of elements that
genuinely live outside every panel are named by *what they are*, so the
exemption travels with the selector instead of a line number.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

E2E_DASHBOARD_TEST = (
    Path(__file__).resolve().parents[2] / "e2e" / "test_dashboard_e2e.py"
)

# Selector fragments for elements that are not inside any tab panel.  Keyed on
# the thing the selector targets, so an exemption cannot drift onto an
# unrelated call the way a line-number exemption would.
PANEL_EXEMPT_FRAGMENTS = (
    '[role="tabpanel"]',  # panel discovery itself
    '[data-testid="stSidebar"]',  # sidebar renders outside every panel
    'button[role="tab"]',  # the tab strip, not a tab body
    '[data-testid="stStatusWidget"]',  # Streamlit's own running indicator
    '[data-testid="stAppViewContainer"]',  # wraps the whole app, panels included
)

# Literal arguments whose page-wide reach is the point, each with the reason
# the call site documents. Keyed on the argument rather than a line number, so
# the exemption travels with the code instead of drifting onto a neighbour.
PAGE_WIDE_BY_DESIGN = {
    # Streamlit renders an open selectbox's options into a popover attached to
    # the document body, not inside the tab panel, so a panel-scoped read
    # cannot see them.
    "body",
    # These wait for the Optimization panel's own heading in order to *find*
    # that panel; scoping them to it would be circular.
    "\U0001f680 Optimization Controls",
    "\U0001f680 Run Optimization",
}

# Locator-producing methods on the Playwright page/locator API.
LOCATOR_METHODS = frozenset(
    {
        "locator",
        "get_by_role",
        "get_by_text",
        "get_by_label",
        "get_by_placeholder",
        "get_by_test_id",
    }
)


# Roles whose elements are rendered outside every tab panel, so a locator for
# them cannot be scoped to one.
PANEL_EXEMPT_ROLES = frozenset({"tab"})


def _literals_of(call: ast.Call) -> list[str]:
    """Every string literal a locator call was given.

    ``get_by_role("heading", name="X")`` carries its identifying text in a
    keyword, so reading only the first positional argument would compare the
    exemption against the role and never match.
    """
    out: list[str] = []
    for node in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _role_of(call: ast.Call) -> str:
    """The role argument of a ``get_by_role`` call, or "" for other calls."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "get_by_role":
        return ""
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        return value if isinstance(value, str) else ""
    return ""


def _filter_predicate_calls(tree: ast.AST) -> set[int]:
    """Locators passed as ``.filter(has=...)`` matchers rather than as scopes.

    ``panel = page.locator(X).filter(has=page.get_by_role(...))`` roots the
    inner call at ``page`` by necessity: it is the thing being matched, not the
    search scope, so it cannot narrow anything and is not an offender.
    """
    predicates: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "filter":
            continue
        for kw in node.keywords:
            if kw.arg in {"has", "has_not"}:
                for inner in ast.walk(kw.value):
                    if isinstance(inner, ast.Call):
                        predicates.add(inner.lineno)
    return predicates


def _unscoped_locator_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Locator calls rooted at ``page`` that target something inside a panel."""
    offenders: list[tuple[int, str]] = []
    predicates = _filter_predicate_calls(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in LOCATOR_METHODS:
            continue
        # Only a receiver that is literally ``page`` is unscoped; anything else
        # (a panel variable, active_tab_panel(page), a chained .filter) is fine.
        if not (isinstance(func.value, ast.Name) and func.value.id == "page"):
            continue
        if node.lineno in predicates:
            continue
        if _role_of(node) in PANEL_EXEMPT_ROLES:
            continue
        literals = _literals_of(node)
        if any(frag in lit for lit in literals for frag in PANEL_EXEMPT_FRAGMENTS):
            continue
        if any(lit in PAGE_WIDE_BY_DESIGN for lit in literals):
            continue
        offenders.append((node.lineno, " ".join(literals)))
    return sorted(offenders)


def _page_wide_body_reads(tree: ast.AST) -> list[int]:
    """``page.inner_text("body")`` reads the whole DOM, every panel included."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "inner_text":
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "page"):
            continue
        if any(lit in PAGE_WIDE_BY_DESIGN for lit in _literals_of(node)):
            continue
        hits.append(node.lineno)
    return sorted(hits)


@pytest.fixture(scope="module")
def dashboard_tree() -> ast.AST:
    return ast.parse(E2E_DASHBOARD_TEST.read_text())


def test_widget_locators_are_scoped_to_the_open_panel(dashboard_tree):
    offenders = _unscoped_locator_calls(dashboard_tree)
    rendered = [f"{E2E_DASHBOARD_TEST.name}:{ln}: {sel!r}" for ln, sel in offenders]
    assert offenders == [], (
        "Locator rooted at `page` targets widgets that live inside a tab panel. "
        "Streamlit keeps every tab body in the DOM, so this also matches tabs "
        "the test never opened and the assertion can pass on another tab's "
        "content. Scope it to active_tab_panel(page) or the specific panel:\n"
        + "\n".join(rendered)
    )


def test_no_page_wide_body_text_reads(dashboard_tree):
    hits = _page_wide_body_reads(dashboard_tree)
    assert hits == [], (
        "page.inner_text('body') returns the text of every rendered tab panel, "
        "so a substring assertion over it passes on another tab's copy. Read "
        "active_tab_panel(page).inner_text() instead. Lines: "
        f"{hits}"
    )


def test_exempt_fragments_are_each_load_bearing(dashboard_tree):
    """Every exemption must still be justified by a real call in the suite.

    An exemption nobody uses is dead permission: it widens the guard without
    protecting anything, and hides that the element moved inside a panel.
    """
    source = E2E_DASHBOARD_TEST.read_text()
    unused = [
        frag
        for frag in (*PANEL_EXEMPT_FRAGMENTS, *PAGE_WIDE_BY_DESIGN, *PANEL_EXEMPT_ROLES)
        if frag not in source
    ]
    assert unused == [], (
        "These panel exemptions match no call in the suite and should be "
        f"removed rather than left as standing permission: {unused}"
    )


# The tests above assert "no offenders remain", which cannot protect the
# detector: once the suite is clean, gutting `_unscoped_locator_calls` leaves
# them green. These drive it over synthetic source so the detector itself has
# to keep working.

SYNTHETIC = """
def t(page):
    a = page.locator('[data-testid="stMetric"]')              # OFFENDER
    b = active_tab_panel(page).locator('[data-testid="stMetric"]')  # scoped
    c = page.locator('[role="tabpanel"]:visible')             # exempt fragment
    d = page.get_by_role("tab", name="Analytics")             # exempt role
    e = page.locator('[role="tabpanel"]').filter(
        has=page.get_by_role("heading", name="X")             # matcher, not scope
    )
    f = page.get_by_role("heading", name="\\U0001f680 Run Optimization")  # by design
    g = page.inner_text("main")                               # BODY-READ
"""


def _synthetic_line(marker: str) -> int:
    """The 1-based line of ``marker`` in SYNTHETIC.

    Derived rather than written out: a hand-counted line number is a value
    nobody measured, and it silently rots the moment the fixture is edited.
    """
    for i, line in enumerate(SYNTHETIC.splitlines(), 1):
        if marker in line:
            return i
    raise AssertionError(f"marker {marker!r} missing from SYNTHETIC")


def test_detector_finds_a_page_rooted_widget_locator():
    offenders = _unscoped_locator_calls(ast.parse(SYNTHETIC))
    assert [(ln, sel) for ln, sel in offenders] == [
        (_synthetic_line("OFFENDER"), '[data-testid="stMetric"]')
    ], offenders


def test_detector_ignores_scoped_and_exempt_locators():
    """Each exemption is checked by construction, not by absence of a hit.

    A detector that simply found nothing would also satisfy the "no offenders"
    tests, so the synthetic source deliberately contains one real offender
    among the exempt forms: finding exactly it proves both halves.
    """
    lines = {ln for ln, _ in _unscoped_locator_calls(ast.parse(SYNTHETIC))}
    assert lines == {_synthetic_line("OFFENDER")}, (
        "expected only the page-rooted widget locator; a scoped "
        f"call, an exempt fragment/role, a filter matcher or a by-design "
        f"page-wide read was misreported: {sorted(lines)}"
    )


def test_detector_finds_page_wide_body_reads():
    assert _page_wide_body_reads(ast.parse(SYNTHETIC)) == [_synthetic_line("BODY-READ")]
