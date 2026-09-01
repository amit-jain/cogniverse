"""A button must not be disabled by a text widget's same-run value.

``st.text_area`` and ``st.text_input`` commit their value on blur, which is the
same interaction that delivers a button click. A ``disabled=`` computed from
the value those widgets returned earlier in the run is therefore always one
render stale: the button is disabled on exactly the render where the user
clicks it, and the click is swallowed. The feature reads as dead while every
import and unit test stays green.

Emptiness belongs in the handler (``if clicked and not value: warn``), where
the committed value is available.
"""

from __future__ import annotations

import ast
from pathlib import Path

DASHBOARD_ROOT = (
    Path(__file__).resolve().parents[3] / "libs/dashboard/cogniverse_dashboard"
)

_DEFERRED_COMMIT_WIDGETS = {"text_area", "text_input"}


def _streamlit_call_name(node: ast.AST) -> str | None:
    """``st.text_area(...)`` -> ``"text_area"``, anything else -> None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in {"st", "streamlit"}:
            return func.attr
    return None


def _deferred_commit_names(tree: ast.AST) -> set[str]:
    """Names bound to the return of a deferred-commit text widget."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if _streamlit_call_name(node.value) not in _DEFERRED_COMMIT_WIDGETS:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _buttons_disabled_by(tree: ast.AST, names: set[str]) -> list[tuple[int, str]]:
    """(lineno, offending name) for each st.button disabled by one of `names`."""
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if _streamlit_call_name(node) != "button":
            continue
        for keyword in node.keywords:
            if keyword.arg != "disabled":
                continue
            for referenced in ast.walk(keyword.value):
                if isinstance(referenced, ast.Name) and referenced.id in names:
                    offenders.append((node.lineno, referenced.id))
    return offenders


def test_no_button_is_disabled_by_a_deferred_commit_widget_value():
    sources = sorted(DASHBOARD_ROOT.rglob("*.py"))
    assert sources, f"no dashboard sources found under {DASHBOARD_ROOT}"

    offenders: dict[str, list[tuple[int, str]]] = {}
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _buttons_disabled_by(tree, _deferred_commit_names(tree))
        if hits:
            offenders[str(path.relative_to(DASHBOARD_ROOT))] = hits

    assert offenders == {}, (
        "st.button(disabled=...) computed from a text_area/text_input value "
        "swallows the user's first click, because the widget commits on the "
        f"same interaction that delivers it: {offenders}"
    )
