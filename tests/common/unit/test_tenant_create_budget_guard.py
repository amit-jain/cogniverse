"""Every tenant-creating call in tests/e2e runs under the one measured budget.

A tenant create recompiles the Vespa application package and waits for
convergence, so its cost is a property of the cluster, not of the test that
happens to issue it. A per-file literal drifts the moment the measurement
moves; the guard walks every ``httpx.Client(...)`` context that posts to
``/admin/tenants`` and requires the budget in force for that post to be the
shared constant. httpx precedence applies: a ``timeout`` keyword on the
``post`` itself wins over the client's.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.e2e.conftest import TENANT_DEPLOY_TIMEOUT_S

_E2E_DIR = Path(__file__).resolve().parents[2] / "e2e"
_BUDGET_NAME = "TENANT_DEPLOY_TIMEOUT_S"
_CREATE_PATH = "/admin/tenants"


def _tenant_create_posts(node: ast.AST) -> list[ast.Call]:
    posts: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not (isinstance(func, ast.Attribute) and func.attr == "post"):
            continue
        first = child.args[0] if child.args else None
        if isinstance(first, ast.Constant) and first.value == _CREATE_PATH:
            posts.append(child)
    return posts


def _timeout_expr(call: ast.Call) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == "timeout":
            return keyword.value
    return None


def _is_client_call(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Attribute) and func.attr == "Client") or (
        isinstance(func, ast.Name) and func.id == "Client"
    )


def tenant_create_posts_off_budget(source: str, name: str) -> list[str]:
    """``<name>:<line> timeout=<what>`` for every tenant-create post whose
    budget in force is not the shared constant."""
    offenders: list[str] = []
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not (isinstance(call, ast.Call) and _is_client_call(call)):
                continue
            client_timeout = _timeout_expr(call)
            for post in _tenant_create_posts(node):
                timeout = _timeout_expr(post)
                if timeout is None:
                    timeout = client_timeout
                if isinstance(timeout, ast.Name) and timeout.id == _BUDGET_NAME:
                    continue
                shown = ast.unparse(timeout) if timeout is not None else "<default>"
                offenders.append(f"{name}:{post.lineno} timeout={shown}")
    return offenders


def test_shared_budget_is_the_measured_value() -> None:
    assert TENANT_DEPLOY_TIMEOUT_S == 180.0


def test_every_e2e_tenant_create_runs_under_the_shared_budget() -> None:
    offenders = [
        offender
        for path in sorted(_E2E_DIR.glob("*.py"))
        for offender in tenant_create_posts_off_budget(path.read_text(), path.name)
    ]
    assert offenders == []


def _synthetic(client_timeout: str | None, post_timeout: str | None) -> str:
    client_kw = f", timeout={client_timeout}" if client_timeout else ""
    post_kw = f", timeout={post_timeout}" if post_timeout else ""
    return (
        "def make():\n"
        f"    with httpx.Client(base_url=RUNTIME{client_kw}) as client:\n"
        f'        client.post("/admin/tenants", json={{"tenant_id": "t"}}{post_kw})\n'
    )


@pytest.mark.parametrize(
    ("client_timeout", "post_timeout", "expected"),
    [
        ("30.0", None, ["synthetic.py:3 timeout=30.0"]),
        (None, None, ["synthetic.py:3 timeout=<default>"]),
        ("TENANT_DEPLOY_TIMEOUT_S", None, []),
        ("1800.0", "TENANT_DEPLOY_TIMEOUT_S", []),
        ("1800.0", "30.0", ["synthetic.py:3 timeout=30.0"]),
        ("TENANT_DEPLOY_TIMEOUT_S", "30.0", ["synthetic.py:3 timeout=30.0"]),
    ],
)
def test_detector_reports_the_budget_in_force(
    client_timeout: str | None, post_timeout: str | None, expected: list[str]
) -> None:
    source = _synthetic(client_timeout, post_timeout)
    assert tenant_create_posts_off_budget(source, "synthetic.py") == expected


def test_detector_ignores_clients_that_do_not_create_tenants() -> None:
    source = (
        "def make():\n"
        "    with httpx.Client(base_url=RUNTIME, timeout=5.0) as client:\n"
        '        client.get("/health")\n'
    )
    assert tenant_create_posts_off_budget(source, "synthetic.py") == []
