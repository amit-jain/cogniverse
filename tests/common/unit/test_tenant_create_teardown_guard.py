"""Every tenant the e2e suite creates is owned by the scope that created it.

A tenant is created either explicitly (``POST /admin/tenants``) or by the
first request that touches a minted id, so ownership is bound at the mint
(``unique_id``) and at the one explicit create helper
(``register_tenant_and_wait``), both in ``tests/e2e/conftest.py``. This guard
walks ``tests/e2e`` and fails any other create site: a direct post to
``/admin/tenants`` (literal or f-string URL) outside the two allowed conftest
functions, or an org/tenant id built from ``uuid4`` in a scope that then
creates an org or tenant with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_E2E_DIR = Path(__file__).resolve().parents[2] / "e2e"
_CREATE_PATH = "/admin/tenants"
_ORG_PATH = "/admin/organizations"
_CREATE_HELPER = "register_tenant_and_wait"
_ALLOWED_POST_SITES = {
    ("conftest.py", "register_tenant_and_wait"),
    ("conftest.py", "_bootstrap_tenant_and_schemas"),
}


def _posts_to(call: ast.Call, path: str) -> bool:
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "post" and call.args):
        return False
    first = call.args[0]
    if isinstance(first, ast.Constant):
        return first.value == path
    if isinstance(first, ast.JoinedStr) and first.values:
        last = first.values[-1]
        return isinstance(last, ast.Constant) and str(last.value).endswith(path)
    return False


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
        )
        for child in ast.walk(node)
    )


def _scopes(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """(label, node) for the module top level and every function body."""
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    top = ast.Module(
        body=[
            s
            for s in tree.body
            if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    return [("<module>", top), *((f.name, f) for f in functions)]


def tenant_create_posts(source: str, name: str) -> list[tuple[str, str]]:
    """(file, function) for every ``POST /admin/tenants`` in ``source``."""
    tree = ast.parse(source, filename=name)
    sites: list[tuple[str, str]] = []
    for label, scope in _scopes(tree):
        for child in ast.walk(scope):
            if isinstance(child, ast.Call) and _posts_to(child, _CREATE_PATH):
                sites.append((name, label))
    return sites


def uuid_minted_create_scopes(source: str, name: str) -> list[str]:
    """``<file>:<scope>`` for every scope that both calls ``uuid4`` and creates
    an org or tenant; test ids are minted by ``unique_id`` so their owner can
    delete them."""
    tree = ast.parse(source, filename=name)
    offenders: list[str] = []
    for label, scope in _scopes(tree):
        creates = _calls(scope, _CREATE_HELPER) or any(
            isinstance(c, ast.Call)
            and (_posts_to(c, _ORG_PATH) or _posts_to(c, _CREATE_PATH))
            for c in ast.walk(scope)
        )
        if creates and _calls(scope, "uuid4"):
            offenders.append(f"{name}:{label}")
    return offenders


def _e2e_sources() -> list[tuple[str, str]]:
    return [(p.name, p.read_text()) for p in sorted(_E2E_DIR.glob("*.py"))]


def test_the_only_tenant_create_posts_are_the_conftest_helper_and_the_seeded_tenant():
    sites = [s for name, src in _e2e_sources() for s in tenant_create_posts(src, name)]
    assert sorted(sites) == sorted(_ALLOWED_POST_SITES)


def test_no_e2e_scope_creates_an_org_or_tenant_from_a_uuid4_id():
    offenders = [
        o
        for name, src in _e2e_sources()
        if name != "conftest.py"
        for o in uuid_minted_create_scopes(src, name)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'def make():\n    client.post("/admin/tenants", json={"tenant_id": "t"})\n',
            [("synthetic.py", "make")],
        ),
        (
            'def make():\n    httpx.post(f"{RUNTIME}/admin/tenants", json={"tenant_id": "t"})\n',
            [("synthetic.py", "make")],
        ),
        (
            'created = httpx.post(f"{RUNTIME}/admin/tenants", json={"tenant_id": "t"})\n',
            [("synthetic.py", "<module>")],
        ),
        (
            'def make():\n    register_tenant_and_wait(unique_id("opt"))\n',
            [],
        ),
        (
            'def make():\n    client.post("/admin/tenants/x/knowledge/summarize", json={})\n'
            '    client.get("/admin/tenants")\n',
            [],
        ),
    ],
)
def test_detector_reports_every_direct_tenant_create_post(source, expected):
    assert tenant_create_posts(source, "synthetic.py") == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'def seed():\n    org = f"opt_gw_{uuid.uuid4().hex[:8]}"\n'
            '    register_tenant_and_wait(f"{org}:t1")\n',
            ["synthetic.py:seed"],
        ),
        (
            'def seed():\n    org = f"cron_{uuid4().hex[:8]}"\n'
            '    client.post(f"{RUNTIME}/admin/organizations", json={"org_id": org})\n',
            ["synthetic.py:seed"],
        ),
        (
            "RUN = uuid4().hex[:8]\n"
            'httpx.post(f"{RUNTIME}/admin/tenants", json={"tenant_id": f"e2e_{RUN}"})\n',
            ["synthetic.py:<module>"],
        ),
        (
            'def seed():\n    org = unique_id("opt_gw")\n'
            '    register_tenant_and_wait(f"{org}:t1")\n',
            [],
        ),
        (
            'def probe():\n    return f"warmup-{uuid.uuid4().hex[:8]}"\n',
            [],
        ),
    ],
)
def test_detector_reports_uuid_minted_ids_that_create_orgs_or_tenants(source, expected):
    assert uuid_minted_create_scopes(source, "synthetic.py") == expected
