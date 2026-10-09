"""Harness keys minted with a ttl stop authenticating at ``expires_at``.

The store runs over the in-memory immutable config store with an injected
clock, so expiry is crossed by moving the clock, never by sleeping. The route
tests drive ``/admin/harness/keys`` over ASGI with the store's real clock.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.harness_keys import (
    MAX_TTL_SECONDS,
    HarnessKeyNotFoundError,
    HarnessKeyStore,
)
from cogniverse_runtime.routers import admin, openai_compat
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

T0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def store():
    value = InMemoryConfigStore()
    value.initialize()
    return value


@pytest.fixture
def clock():
    return Clock()


def test_a_ttl_is_stored_as_an_exact_expiry(store, clock):
    record = HarnessKeyStore(store, now=clock).create("acme", "web", ttl_seconds=60)
    stored = store.get_immutable_config(
        SYSTEM_TENANT_ID, ConfigScope.SYSTEM, "harness_keys", record["key_hash"]
    )
    assert stored.config_value == {
        "tenant_id": "acme:acme",
        "name": "web",
        "created_at": "2026-10-09T12:00:00+00:00",
        "expires_at": "2026-10-09T12:01:00+00:00",
        "revoked": False,
    }
    assert {k: v for k, v in record.items() if k != "key"} == {
        "key_hash": hashlib.sha256(record["key"].encode()).hexdigest(),
        "key_prefix": record["key_hash"][:12],
        "tenant_id": "acme:acme",
        "name": "web",
        "created_at": "2026-10-09T12:00:00+00:00",
        "expires_at": "2026-10-09T12:01:00+00:00",
        "revoked": False,
    }


def test_resolve_accepts_before_expiry_and_refuses_at_and_after_it(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    record = keys.create("acme", "web", ttl_seconds=60)
    clock.now = T0 + timedelta(seconds=59, microseconds=999_999)
    assert keys.resolve(record["key"]) == "acme:acme"
    for moment in (T0 + timedelta(seconds=60), T0 + timedelta(days=3)):
        clock.now = moment
        with pytest.raises(HarnessKeyNotFoundError, match="Harness key not found"):
            keys.resolve(record["key"])


def test_list_reports_an_expired_key_as_revoked(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    record = keys.create("acme", "web", ttl_seconds=60)
    public = {k: v for k, v in record.items() if k != "key"}
    clock.now = T0 + timedelta(seconds=30)
    assert keys.list("acme") == {"keys": [public], "continuation": None}
    clock.now = T0 + timedelta(seconds=60)
    assert keys.list("acme") == {
        "keys": [{**public, "revoked": True}],
        "continuation": None,
    }
    # Expiry writes no revocation; the key simply stops authenticating.
    assert (
        store.get_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            "harness_key_revocations",
            record["key_hash"],
        )
        is None
    )


def test_a_key_without_a_ttl_never_expires(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    record = keys.create("acme", "pi", ttl_seconds=None)
    assert record["expires_at"] is None
    clock.now = T0 + timedelta(days=3650)
    assert keys.resolve(record["key"]) == "acme:acme"
    assert keys.list("acme")["keys"][0]["revoked"] is False


def test_a_record_written_without_an_expiry_field_never_expires(store, clock):
    plaintext = "cgv-written-before-expiry"
    digest = hashlib.sha256(plaintext.encode()).hexdigest()
    store.put_immutable_config(
        SYSTEM_TENANT_ID,
        ConfigScope.SYSTEM,
        "harness_keys",
        digest,
        {
            "tenant_id": "acme:acme",
            "name": "legacy",
            "created_at": "2026-01-01T00:00:00+00:00",
            "revoked": False,
        },
    )
    keys = HarnessKeyStore(store, now=clock)
    clock.now = T0 + timedelta(days=3650)
    assert keys.resolve(plaintext) == "acme:acme"
    assert keys.list("acme") == {
        "keys": [
            {
                "key_hash": digest,
                "key_prefix": digest[:12],
                "tenant_id": "acme:acme",
                "name": "legacy",
                "created_at": "2026-01-01T00:00:00+00:00",
                "expires_at": None,
                "revoked": False,
            }
        ],
        "continuation": None,
    }


def test_revoke_tenant_skips_expired_keys(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    expired = keys.create("acme", "web-old", ttl_seconds=10)
    live = keys.create("acme", "web-new", ttl_seconds=3600)
    lasting = keys.create("acme", "pi")
    clock.now = T0 + timedelta(seconds=10)
    assert keys.revoke_tenant("acme") == 2
    revocations = {
        record["key_hash"]: store.get_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            "harness_key_revocations",
            record["key_hash"],
        )
        for record in (expired, live, lasting)
    }
    assert [
        entry.config_value if entry else None for entry in revocations.values()
    ] == [None, {"revoked": True}, {"revoked": True}]
    assert [r["revoked"] for r in keys.list("acme")["keys"]] == [True, True, True]


def test_an_expired_key_is_refused_like_a_revoked_one_on_the_bearer_path(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    record = keys.create("acme", "web", ttl_seconds=60)
    openai_compat.set_key_resolver(keys.resolve)
    try:
        assert openai_compat.resolve_tenant(record["key"]) == "acme:acme"
        clock.now = T0 + timedelta(seconds=60)
        assert openai_compat.resolve_tenant(record["key"]) is None
    finally:
        openai_compat.set_key_resolver(None)


def test_the_store_refuses_a_ttl_outside_its_bounds(store, clock):
    keys = HarnessKeyStore(store, now=clock)
    for ttl in (0, -5, MAX_TTL_SECONDS + 1):
        with pytest.raises(
            ValueError,
            match=f"ttl_seconds must be from 1 to {MAX_TTL_SECONDS}, got {ttl}",
        ):
            keys.create("acme", "web", ttl_seconds=ttl)
    assert keys.list("acme") == {"keys": [], "continuation": None}


def _client(store):
    app = FastAPI()
    manager = ConfigManager(store=store)
    app.dependency_overrides[admin.get_config_manager_dependency] = lambda: manager
    app.include_router(admin.router, prefix="/admin")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )


@pytest.mark.asyncio
async def test_the_route_mints_a_key_that_expires_after_its_ttl(store):
    async with _client(store) as client:
        minted = await client.post(
            "/admin/harness/keys",
            json={"tenant_id": "acme", "name": "web", "ttl_seconds": 90},
        )
        lasting = await client.post(
            "/admin/harness/keys", json={"tenant_id": "acme", "name": "pi"}
        )
    assert (minted.status_code, lasting.status_code) == (200, 200)
    record = minted.json()
    assert datetime.fromisoformat(record["expires_at"]) - datetime.fromisoformat(
        record["created_at"]
    ) == timedelta(seconds=90)
    assert lasting.json()["expires_at"] is None


@pytest.mark.asyncio
async def test_the_route_refuses_a_ttl_outside_its_bounds(store):
    async with _client(store) as client:
        responses = [
            await client.post(
                "/admin/harness/keys",
                json={"tenant_id": "acme", "name": "web", "ttl_seconds": ttl},
            )
            for ttl in (0, -1, MAX_TTL_SECONDS + 1, 1.5, "soon")
        ]
        accepted = await client.post(
            "/admin/harness/keys",
            json={"tenant_id": "acme", "name": "web", "ttl_seconds": MAX_TTL_SECONDS},
        )
    assert [r.status_code for r in responses] == [422, 422, 422, 422, 422]
    assert [r.json()["detail"][0]["loc"] for r in responses] == [
        ["body", "ttl_seconds"]
    ] * 5
    assert accepted.status_code == 200
    assert MAX_TTL_SECONDS == 7 * 24 * 3600
    # Only the accepted request wrote a key.
    entries, _ = store.list_immutable_configs(
        SYSTEM_TENANT_ID, ConfigScope.SYSTEM, "harness_keys"
    )
    assert [entry.config_key for entry in entries] == [accepted.json()["key_hash"]]
