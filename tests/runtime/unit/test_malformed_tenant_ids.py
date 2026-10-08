"""A malformed tenant ID in a request is the caller's error: 400, not 500.

The tenant routes are driven over ASGITransport with IDs ``canonical_tenant_id``
refuses (more than one ``:``, an empty part); each answers 400
``invalid_tenant_id`` before it reads any store. A source scan pins that no
route handler hands a tenant it was given straight to ``canonical_tenant_id``,
whose ``ValueError`` would surface as a 500.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.routers import (
    approvals,
    routing_decisions,
    telemetry_metrics,
    tenant,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

RUNTIME = Path(__file__).resolve().parents[3] / "libs/runtime/cogniverse_runtime"


@pytest.fixture(scope="module")
def app():
    app = FastAPI()
    app.include_router(tm.router, prefix="/admin")
    for router in (approvals, routing_decisions, telemetry_metrics):
        app.include_router(router.router, prefix="/admin/tenant")
    app.include_router(tenant.router, prefix="/admin/tenant")
    return app


def _refusal(tenant_id: str) -> dict:
    return {
        "detail": {
            "error": "invalid_tenant_id",
            "message": f"Tenant ID '{tenant_id}' is malformed: use '<org>:<tenant>' "
            "or '<tenant>', with no empty part.",
            "failure": "ValueError",
            "tenant_id": tenant_id,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", ["a:b:c", "acme:", ":prod"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/tenants/{t}"),
        ("DELETE", "/admin/tenants/{t}"),
        ("GET", "/admin/tenants/{t}/tier"),
        ("GET", "/admin/tenant/{t}/telemetry/traces"),
        ("GET", "/admin/tenant/{t}/telemetry/phoenix"),
        ("GET", "/admin/tenant/{t}/telemetry/root-causes"),
        ("GET", "/admin/tenant/{t}/routing-decisions"),
        ("GET", "/admin/tenant/{t}/approvals"),
        ("GET", "/admin/tenant/{t}/memories/stats?agent_name=search_agent"),
    ],
)
async def test_a_malformed_tenant_id_answers_400_with_the_reason(
    app, method, path, tenant_id
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
    ) as client:
        response = await client.request(method, path.format(t=tenant_id))
    assert (response.status_code, response.json()) == (400, _refusal(tenant_id))


def unguarded_tenant_canonicalizations(source: str) -> list[str]:
    """Route handlers in ``source`` passing a tenant they were given (a
    parameter, or a field of one) to ``canonical_tenant_id`` outside a
    ``try`` that handles ``ValueError``; as ``handler:line``."""
    found = []
    for handler in ast.walk(ast.parse(source)):
        if not isinstance(handler, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and isinstance(d.func.value, ast.Name)
            and d.func.value.id == "router"
            for d in handler.decorator_list
        ):
            continue
        params = {a.arg for a in handler.args.args + handler.args.kwonlyargs}
        guarded = {
            id(node)
            for block in ast.walk(handler)
            if isinstance(block, ast.Try)
            and any(
                h.type is not None and "ValueError" in ast.unparse(h.type)
                for h in block.handlers
            )
            for statement in block.body
            for node in ast.walk(statement)
        }
        for call in ast.walk(handler):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "canonical_tenant_id"
                and call.args
                and id(call) not in guarded
            ):
                continue
            names = {n.id for n in ast.walk(call.args[0]) if isinstance(n, ast.Name)}
            if names & params:
                found.append(f"{handler.name}:{call.lineno}")
    return found


def test_the_scan_finds_a_handler_canonicalizing_its_tenant_unguarded():
    source = """
@router.get("/{tenant_id}/a")
async def raw(tenant_id: str):
    return canonical_tenant_id(tenant_id)

@router.post("/b")
async def from_body(request: Body):
    return canonical_tenant_id(request.tenant_id)

@router.get("/{tenant_id}/c")
async def checked(tenant_id: str):
    return canonical_tenant_or_400(tenant_id)

@router.get("/{tenant_id}/d")
async def caught(tenant_id: str):
    try:
        return canonical_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400) from exc

@router.get("/e")
async def stored():
    return canonical_tenant_id(read_row()["tenant_id"])

def helper(tenant_id):
    return canonical_tenant_id(tenant_id)
"""
    assert unguarded_tenant_canonicalizations(source) == ["raw:4", "from_body:8"]


def test_no_route_handler_canonicalizes_a_tenant_it_was_given_unguarded():
    files = sorted((RUNTIME / "routers").glob("*.py")) + [
        RUNTIME / "admin" / "tenant_manager.py"
    ]
    assert {
        path.name: unguarded_tenant_canonicalizations(path.read_text())
        for path in files
    } == {path.name: [] for path in files}
