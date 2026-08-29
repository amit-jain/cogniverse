"""Contracts for the shared kubectl context used by e2e tests."""

from __future__ import annotations

import ast
from pathlib import Path

import tests.e2e.conftest as e2e_conftest

E2E_ROOT = Path(__file__).resolve().parents[2] / "e2e"
KUBECTL_CONTEXT_EXEMPTIONS: dict[tuple[str, int], str] = {
    (
        "e2e/test_annotation_feedback_e2e.py",
        62,
    ): "kubectl version --client only checks the client binary",
    (
        "e2e/test_coding_cli_e2e.py",
        65,
    ): "kubectl config current-context inspects the active local selection",
}


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


def _kubectl_context_violations() -> list[tuple[Path, int, str]]:
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
                if path.name != "conftest.py":
                    violations.append(
                        (
                            path,
                            node.lineno,
                            "defines its own kubectl context; import it from "
                            "tests.e2e.conftest",
                        )
                    )
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
            elif isinstance(node, (ast.List, ast.Tuple)):
                literal_strings = [
                    literal
                    for literal in (_constant_string(element) for element in node.elts)
                    if literal is not None
                ]
                if "kubectl" not in literal_strings:
                    continue
                rel_path = str(path.relative_to(E2E_ROOT.parent))
                exemption = KUBECTL_CONTEXT_EXEMPTIONS.get((rel_path, node.lineno))
                if exemption is not None:
                    continue
                if any(literal.startswith("k3d-") for literal in literal_strings):
                    violations.append(
                        (path, node.lineno, "hardcodes a kubectl context literal")
                    )
                    continue
                if "--context" not in literal_strings:
                    violations.append(
                        (path, node.lineno, "omits --context on a kubectl invocation")
                    )

    return violations


def test_e2e_kubectl_invocations_use_shared_context():
    violations = _kubectl_context_violations()
    assert not violations, "\n".join(
        f"{path.relative_to(E2E_ROOT.parent)}:{line}: {message}"
        for path, line, message in violations
    )


def test_shared_kubectl_context_matches_provisioned_cluster():
    assert e2e_conftest.KUBECTL_CONTEXT == f"k3d-{e2e_conftest.E2E_CLUSTER_NAME}"
