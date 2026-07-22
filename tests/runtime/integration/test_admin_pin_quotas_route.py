"""HTTP-level coverage for the GET /admin/tenants/{tenant_id}/pin_quotas body.

This read route had no in-process test at all — its only exerciser was a
skip-gated full-cluster e2e, so the load-blob -> json.loads -> PinQuotasResponse
serialization path never ran in an ordinary pytest run. These drive the mounted
FastAPI app via ASGITransport with a faithful ArtifactManager double (async
``load_blob`` returning the JSON blob the real store persists), asserting the
exact wire body for both a persisted-blob tenant and an unset tenant (defaults).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


class _StubArtifactManager:
    """Faithful double for ArtifactManager: async ``load_blob(kind, key)``
    returning the persisted raw string, or None when the blob is unset."""

    def __init__(self, blobs):
        self._blobs = blobs
        self.received = []

    async def load_blob(self, kind, key):
        self.received.append((kind, key))
        return self._blobs.get((kind, key))


def _build_app(blobs, monkeypatch):
    stub = _StubArtifactManager(blobs)
    admin_router._reset_admin_overrides_for_tests()
    monkeypatch.setattr(admin_router, "_build_artifact_manager", lambda key: stub)
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    return app, stub


async def _get(app, path):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_pin_quotas_returns_persisted_blob(monkeypatch):
    persisted = {"user": 7, "tenant_admin": 20, "org_admin": -1}
    app, stub = _build_app(
        {("config", "pin_quotas"): json.dumps(persisted)}, monkeypatch
    )
    try:
        resp = await _get(app, "/admin/tenants/acme:acme/pin_quotas")
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "tenant_id": "acme:acme",
        "quotas": {"user": 7, "tenant_admin": 20, "org_admin": -1},
    }
    # The blob was read from the durable store under the canonical tenant key.
    assert stub.received == [("config", "pin_quotas")]


@pytest.mark.asyncio
async def test_pin_quotas_unset_returns_defaults(monkeypatch):
    from cogniverse_core.memory.pinning import PinQuotas

    d = PinQuotas()
    expected = {
        "user": d.user,
        "tenant_admin": d.tenant_admin,
        "org_admin": -1 if d.org_admin is None else d.org_admin,
    }
    app, stub = _build_app({}, monkeypatch)
    try:
        resp = await _get(app, "/admin/tenants/acme:acme/pin_quotas")
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"tenant_id": "acme:acme", "quotas": expected}
    assert stub.received == [("config", "pin_quotas")]
