"""Unit coverage for the /admin/tenants/{tenant_id}/pin_quotas route body.

Drives the mounted FastAPI app in-process via ASGITransport to pin the route's
request/response serialization and validation: the exact wire body shape, the
org_admin unlimited-sentinel rejection, and canonical-tenant routing of the
artifact key. The ArtifactManager is a double here because these assert route
LOGIC, not store behavior — the real Phoenix save->cold-read->load round-trip
and cross-replica enforcement live in
tests/runtime/integration/test_pin_quota_enforcement_reads_blob.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


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
    stub.built_for = []
    admin_router._reset_admin_overrides_for_tests()

    def _factory(key):
        stub.built_for.append(key)
        return stub

    monkeypatch.setattr(admin_router, "_build_artifact_manager", _factory)
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
    # The CANONICAL tenant must reach the per-tenant store — a factory that
    # dropped or mis-derived the tenant would read another tenant's quotas.
    assert stub.built_for == ["acme:acme"]


async def _put(app, path, body):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(path, json=body)


@pytest.mark.asyncio
async def test_org_admin_quota_rejects_sub_sentinel_negatives(monkeypatch):
    """-1 means unlimited; any other negative used to persist as a literal
    limit that usage comparisons always exceed — every org_admin pin for
    the tenant was silently rejected until the value was corrected."""
    from unittest.mock import AsyncMock

    app, stub = _build_app({}, monkeypatch)
    stub.save_blob = AsyncMock()
    try:
        bad = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"org_admin": -5})
        ok = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"org_admin": -1})
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert bad.status_code == 400
    assert "org_admin" in bad.json()["detail"]
    assert ok.status_code == 200
    assert ok.json()["quotas"]["org_admin"] == -1
