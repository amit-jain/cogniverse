"""Admin config-blob PUTs are write-behind through the BlobWriteQueue.

Applied inline, a pin-quota PUT paid load->delete->create round-trips
against Phoenix — 0.06s idle, 35s measured under span-ingestion load. The
route now accepts the write, serves it read-your-write from the moment of
acceptance, applies it in the background, and surfaces a write the queue
could not persist as a 503 on every read until a newer PUT supersedes it.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.blob_write_queue import BlobWriteQueue
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:acme"


class _GatedStore:
    """ArtifactManager double whose save can be gated or forced to fail."""

    def __init__(self, blobs=None):
        self.blobs = dict(blobs or {})
        self.saved: list[tuple[str, str, str]] = []
        self.save_gate = asyncio.Event()
        self.save_gate.set()
        self.fail_saves = False

    async def load_blob(self, kind, key):
        return self.blobs.get((kind, key))

    async def save_blob(self, kind, key, content):
        await self.save_gate.wait()
        if self.fail_saves:
            raise ConnectionError("phoenix write path down")
        self.saved.append((kind, key, content))
        self.blobs[(kind, key)] = content


@pytest.fixture
def gated_app(monkeypatch):
    store = _GatedStore()
    admin_router._reset_admin_overrides_for_tests()
    monkeypatch.setattr(admin_router, "_build_artifact_manager", lambda key: store)
    # Deterministic fast failure for the fault-path tests.
    monkeypatch.setattr(
        admin_router,
        "_blob_write_queue",
        BlobWriteQueue(admin_router._apply_blob_write, max_attempts=1, backoff_s=0),
    )
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    yield app, store
    admin_router._reset_admin_overrides_for_tests()


async def _request(app, method, path, body=None):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await getattr(client, method)(path, **({"json": body} if body else {}))


@pytest.mark.asyncio
async def test_put_is_accepted_before_the_store_write_runs(gated_app):
    app, store = gated_app
    store.save_gate.clear()

    response = await _request(
        app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pending_write"] is True
    assert body["quotas"]["user"] == 5
    assert store.saved == []

    store.save_gate.set()
    await admin_router._blob_write_queue.flush()
    assert store.saved == [
        (
            "config",
            "pin_quotas",
            json.dumps({"user": 5, "tenant_admin": 500, "org_admin": -1}),
        )
    ]

    settled = await _request(app, "get", f"/admin/tenants/{TENANT}/pin_quotas")
    assert settled.json()["pending_write"] is False


@pytest.mark.asyncio
async def test_get_serves_the_accepted_write_before_it_lands(gated_app):
    app, store = gated_app
    store.save_gate.clear()

    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})
    response = await _request(app, "get", f"/admin/tenants/{TENANT}/pin_quotas")

    assert response.json() == {
        "tenant_id": TENANT,
        "quotas": {"user": 5, "tenant_admin": 500, "org_admin": -1},
        "pending_write": True,
    }
    store.save_gate.set()
    await admin_router._blob_write_queue.flush()


@pytest.mark.asyncio
async def test_get_overlays_the_pending_write_when_the_cache_expired(
    gated_app, monkeypatch
):
    """Under load the cache TTL can lapse before the write lands; the reader
    must see the accepted value, never the stale durable blob."""
    app, store = gated_app
    store.blobs[("config", "pin_quotas")] = json.dumps(
        {"user": 3, "tenant_admin": 10, "org_admin": -1}
    )
    store.save_gate.clear()

    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})
    monkeypatch.setattr(admin_router, "_PIN_QUOTA_CACHE_TTL_S", 0.0)
    response = await _request(app, "get", f"/admin/tenants/{TENANT}/pin_quotas")

    assert response.json()["quotas"] == {
        "user": 5,
        "tenant_admin": 10,
        "org_admin": -1,
    }
    assert response.json()["pending_write"] is True
    store.save_gate.set()
    await admin_router._blob_write_queue.flush()


@pytest.mark.asyncio
async def test_failed_write_surfaces_on_read_as_503(gated_app):
    app, store = gated_app
    store.fail_saves = True

    accepted = await _request(
        app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5}
    )
    assert accepted.status_code == 200
    await admin_router._blob_write_queue.flush()

    response = await _request(app, "get", f"/admin/tenants/{TENANT}/pin_quotas")
    assert response.status_code == 503
    assert "phoenix write path down" in response.json()["detail"]


@pytest.mark.asyncio
async def test_put_after_failed_write_merges_from_the_accepted_state(gated_app):
    """The recovery PUT builds on what the admin last accepted, not on the
    stale durable blob, and supersedes the failure."""
    app, store = gated_app
    store.blobs[("config", "pin_quotas")] = json.dumps(
        {"user": 3, "tenant_admin": 10, "org_admin": -1}
    )
    store.fail_saves = True
    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})
    await admin_router._blob_write_queue.flush()

    store.fail_saves = False
    response = await _request(
        app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"tenant_admin": 9}
    )
    assert response.json()["quotas"] == {
        "user": 5,
        "tenant_admin": 9,
        "org_admin": -1,
    }
    await admin_router._blob_write_queue.flush()
    assert store.blobs[("config", "pin_quotas")] == json.dumps(
        {"user": 5, "tenant_admin": 9, "org_admin": -1}
    )

    settled = await _request(app, "get", f"/admin/tenants/{TENANT}/pin_quotas")
    assert settled.status_code == 200
    assert settled.json()["pending_write"] is False


@pytest.mark.asyncio
async def test_signature_variant_put_is_accepted_then_applied(gated_app):
    app, store = gated_app
    store.save_gate.clear()

    response = await _request(
        app,
        "put",
        f"/admin/tenants/{TENANT}/signature_variants/routing",
        {"variant_id": "v2"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "tenant_id": TENANT,
        "selections": {"routing": "v2"},
        "pending_write": True,
    }
    assert store.saved == []

    store.save_gate.set()
    await admin_router._blob_write_queue.flush()
    assert store.saved == [
        ("config", "signature_variants", json.dumps({"routing": "v2"}))
    ]
    assert await admin_router.load_signature_variants(TENANT) == {"routing": "v2"}


@pytest.mark.asyncio
async def test_failed_variant_write_surfaces_to_the_dispatcher_loader(gated_app):
    """load_signature_variants feeds the dispatcher; a lost selection must
    raise there too, never silently fall back to the stale blob."""
    app, store = gated_app
    store.fail_saves = True

    await _request(
        app,
        "put",
        f"/admin/tenants/{TENANT}/signature_variants/routing",
        {"variant_id": "v2"},
    )
    await admin_router._blob_write_queue.flush()

    with pytest.raises(Exception, match="phoenix write path down"):
        await admin_router.load_signature_variants(TENANT)


@pytest.mark.asyncio
async def test_shutdown_drain_applies_pending_writes(gated_app):
    app, store = gated_app
    store.save_gate.clear()
    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})
    store.save_gate.set()

    assert await admin_router.drain_blob_writes(timeout_s=5.0) is True
    assert store.saved == [
        (
            "config",
            "pin_quotas",
            json.dumps({"user": 5, "tenant_admin": 500, "org_admin": -1}),
        )
    ]


@pytest.mark.asyncio
async def test_shutdown_drain_reports_writes_it_could_not_land(gated_app):
    app, store = gated_app
    store.save_gate.clear()
    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})

    assert await admin_router.drain_blob_writes(timeout_s=0.05) is False
    store.save_gate.set()
    await admin_router._blob_write_queue.flush()


@pytest.mark.asyncio
async def test_shutdown_drain_reports_terminal_failures(gated_app):
    app, store = gated_app
    store.fail_saves = True
    await _request(app, "put", f"/admin/tenants/{TENANT}/pin_quotas", {"user": 5})

    assert await admin_router.drain_blob_writes(timeout_s=5.0) is False


def test_lifespan_shutdown_drains_the_queue():
    """The shutdown path must call the drain; without it a write accepted
    moments before SIGTERM is silently lost."""
    import inspect

    from cogniverse_runtime import main as runtime_main

    source = inspect.getsource(runtime_main.lifespan)
    shutdown = source.split("yield", 1)[1]
    assert "drain_blob_writes(" in shutdown
