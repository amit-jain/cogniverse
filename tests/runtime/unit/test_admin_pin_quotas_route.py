"""Unit coverage for the /admin/tenants/{tenant_id}/pin_quotas route body.

Drives the mounted FastAPI app in-process via ASGITransport to pin the route's
request/response serialization and validation: the exact wire body shape, the
org_admin unlimited-sentinel rejection, and canonical-tenant routing of the
stored record. The config store is the in-memory one here because these assert
route LOGIC — the real Vespa round-trip and cross-process enforcement live in
tests/runtime/integration/test_pin_quota_enforcement_reads_blob.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingStore(InMemoryConfigStore):
    """In-memory config store that records which record each read named."""

    def __init__(self):
        super().__init__()
        self.reads = []

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        self.reads.append((tenant_id, scope, service, config_key))
        return super().get_config(tenant_id, scope, service, config_key, version)


class _OutageStore(InMemoryConfigStore):
    """Config store that is down."""

    def get_config(self, *args, **kwargs):
        raise ConnectionError("config store down")


@pytest.fixture
def wired():
    previous = admin_router._config_manager

    def wire(store):
        admin_router.set_config_manager(ConfigManager(store=store))
        app = FastAPI()
        app.include_router(admin_router.router, prefix="/admin")
        return app

    yield wire
    admin_router.set_config_manager(previous)


def _seed(store, tenant_id, quotas):
    store.set_config(
        tenant_id, ConfigScope.SYSTEM, "admin_overrides", "pin_quotas", quotas
    )


async def _get(app, path):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.get(path)


async def _put(app, path, body):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(path, json=body)


RECORD = ("acme:acme", ConfigScope.SYSTEM, "admin_overrides", "pin_quotas")


@pytest.mark.asyncio
async def test_pin_quotas_returns_the_stored_record(wired):
    store = _RecordingStore()
    persisted = {"user": 7, "tenant_admin": 20, "org_admin": -1}
    _seed(store, "acme:acme", persisted)
    app = wired(store)

    resp = await _get(app, "/admin/tenants/acme:acme/pin_quotas")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "tenant_id": "acme:acme",
        "quotas": {"user": 7, "tenant_admin": 20, "org_admin": -1},
    }
    # The record was read from the store under the canonical tenant key.
    assert store.reads == [RECORD]


@pytest.mark.asyncio
async def test_pin_quotas_unset_returns_defaults(wired):
    from cogniverse_core.memory.pinning import PinQuotas

    d = PinQuotas()
    expected = {
        "user": d.user,
        "tenant_admin": d.tenant_admin,
        "org_admin": -1 if d.org_admin is None else d.org_admin,
    }
    store = _RecordingStore()
    app = wired(store)

    resp = await _get(app, "/admin/tenants/acme/pin_quotas")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"tenant_id": "acme", "quotas": expected}
    # The CANONICAL tenant must reach the store — a read under the bare id
    # would look up a record no PUT writes.
    assert store.reads == [RECORD]


@pytest.mark.asyncio
async def test_put_merges_the_named_fields_onto_the_stored_record(wired):
    store = _RecordingStore()
    _seed(store, "acme:acme", {"user": 1, "tenant_admin": 2, "org_admin": -1})
    app = wired(store)

    resp = await _put(app, "/admin/tenants/acme/pin_quotas", {"tenant_admin": 9})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "tenant_id": "acme",
        "quotas": {"user": 1, "tenant_admin": 9, "org_admin": -1},
    }
    stored = store.get_config(*RECORD)
    assert (stored.version, stored.config_value) == (
        2,
        {"user": 1, "tenant_admin": 9, "org_admin": -1},
    )


@pytest.mark.asyncio
async def test_org_admin_quota_rejects_sub_sentinel_negatives(wired):
    """-1 means unlimited; any other negative used to persist as a literal
    limit that usage comparisons always exceed — every org_admin pin for
    the tenant was silently rejected until the value was corrected."""
    store = _RecordingStore()
    app = wired(store)

    bad = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"org_admin": -5})
    assert store.get_config(*RECORD) is None
    ok = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"org_admin": -1})

    assert bad.status_code == 400
    assert "org_admin" in bad.json()["detail"]
    assert ok.status_code == 200
    assert ok.json()["quotas"]["org_admin"] == -1
    assert store.get_config(*RECORD).config_value["org_admin"] == -1


@pytest.mark.asyncio
async def test_pin_quotas_get_maps_store_outage_to_503(wired):
    """A config-store outage is a dependency failure — 503 with a descriptive
    detail, never an opaque 500."""
    app = wired(_OutageStore())
    response = await _get(app, "/admin/tenants/acme:acme/pin_quotas")
    assert response.status_code == 503
    assert response.json() == {
        "detail": "pin-quota store unavailable: config store down"
    }


@pytest.mark.asyncio
async def test_pin_quotas_put_maps_store_outage_to_503(wired):
    app = wired(_OutageStore())
    response = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"user": 5})
    assert response.status_code == 503
    assert response.json() == {
        "detail": "pin-quota store unavailable: config store down"
    }


@pytest.mark.asyncio
async def test_pin_quotas_put_still_validates_before_store(wired):
    """Request validation fires before the store is touched — a bad request
    is a 400 even when the store is down."""
    app = wired(_OutageStore())
    response = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"user": -2})
    assert response.status_code == 400
    assert response.json() == {"detail": "user quota must be >= 0"}


@pytest.mark.asyncio
async def test_a_put_losing_every_compare_and_set_answers_409_and_writes_nothing(
    wired,
):
    class _ContendedStore(InMemoryConfigStore):
        def compare_and_set_config(self, *args, **kwargs):
            return None

    store = _ContendedStore()
    _seed(store, "acme:acme", {"user": 1, "tenant_admin": 2, "org_admin": -1})
    app = wired(store)

    response = await _put(app, "/admin/tenants/acme:acme/pin_quotas", {"user": 5})

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "pin-quota update conflicted: config acme:acme:system:admin_overrides:"
            "pin_quotas changed under every one of 10 compare-and-set attempts; "
            "nothing was written"
        )
    }
    stored = store.get_config(*RECORD)
    assert (stored.version, stored.config_value) == (
        1,
        {"user": 1, "tenant_admin": 2, "org_admin": -1},
    )
