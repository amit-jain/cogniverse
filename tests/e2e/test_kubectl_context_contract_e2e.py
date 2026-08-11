"""Contracts for the shared kubectl context used by e2e tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tests.e2e.conftest as e2e_conftest

E2E_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session", autouse=True)
def e2e_stack():
    """Suppress the shared e2e cluster fixture for this unit-level contract."""
    yield


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        if any(isinstance(part, ast.FormattedValue) for part in node.values):
            return None
        pieces: list[str] = []
        for part in node.values:
            value = _constant_string(part)
            if value is None:
                return None
            pieces.append(value)
        return "".join(pieces)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _annotate_parents(node: ast.AST, parent: ast.AST | None = None) -> None:
    setattr(node, "_parent", parent)
    for child in ast.iter_child_nodes(node):
        _annotate_parents(child, node)


def _has_ancestor(node: ast.AST, ancestor_type: type[ast.AST]) -> bool:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ancestor_type):
            return True
        parent = getattr(parent, "_parent", None)
    return False


def _kubectl_context_literals() -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(E2E_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        _annotate_parents(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target_names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
                if not any(
                    "KUBECTL" in name and "CONTEXT" in name for name in target_names
                ):
                    continue
                literal = _constant_string(node.value)
                if literal is not None and literal.startswith("k3d-"):
                    violations.append(
                        (path, node.lineno, "hardcodes a kubectl context literal")
                    )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                if "KUBECTL" not in target_name or "CONTEXT" not in target_name:
                    continue
                literal = (
                    _constant_string(node.value) if node.value is not None else None
                )
                if literal is not None and literal.startswith("k3d-"):
                    violations.append(
                        (path, node.lineno, "hardcodes a kubectl context literal")
                    )
            elif isinstance(node, (ast.List, ast.Tuple)) and _has_ancestor(
                node, ast.Call
            ):
                literal_strings = [
                    literal
                    for literal in (_constant_string(element) for element in node.elts)
                    if literal is not None
                ]
                if not any(literal.startswith("k3d-") for literal in literal_strings):
                    continue
                if not any(
                    literal == "kubectl" or literal.startswith("--context")
                    for literal in literal_strings
                ):
                    continue
                violations.append(
                    (path, node.lineno, "inlines a kubectl context literal")
                )

    return violations


def test_e2e_files_do_not_hardcode_kubectl_context_literals():
    violations = _kubectl_context_literals()
    assert not violations, "\n".join(
        f"{path.relative_to(E2E_ROOT.parent)}:{line}: {message}"
        for path, line, message in violations
    )


def test_shared_kubectl_context_matches_provisioned_cluster():
    assert e2e_conftest.KUBECTL_CONTEXT == f"k3d-{e2e_conftest.E2E_CLUSTER_NAME}"
