"""Harness credentials round-trip through HTTP and test-owned Vespa."""

import asyncio
import hashlib
import importlib
import os
import subprocess
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin
from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]


@pytest.fixture(scope="module")
def key_vespa():
    from cogniverse_vespa.metadata_schemas import (
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
    from tests.conftest import _shared_vespa_application_package
    from tests.utils.vllm_sidecar import OWNER_LABEL

    manager = VespaDockerManager()
    info = manager.start_container(f"harness-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
        owner = subprocess.check_output(
            [
                "docker",
                "inspect",
                "-f",
                '{{ index .Config.Labels "' + OWNER_LABEL + '" }}',
                info["container_name"],
            ],
            text=True,
        ).strip()
        assert owner == str(os.getpid())
        package = _shared_vespa_application_package(
            [
                create_config_metadata_schema(),
                create_organization_metadata_schema(),
                create_tenant_metadata_schema(),
            ]
        )
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=info["config_port"]
        )._deploy_package(package)
        manager.wait_for_application_ready(info)
        yield info
    finally:
        subprocess.run(
            ["docker", "unpause", info["container_name"]], capture_output=True
        )
        manager.stop_container(info)


@pytest.fixture
def store(key_vespa):
    value = VespaConfigStore(backend_port=key_vespa["http_port"])
    yield value
    value.close()


def keys(store):
    return importlib.import_module("cogniverse_runtime.harness_keys").HarnessKeyStore(
        store
    )


def client_for(store):
    app = FastAPI()
    cm = ConfigManager(store=store)
    app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm
    app.include_router(admin.router, prefix="/admin")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )


@pytest.mark.asyncio
async def test_round_trip_canonical_hash_and_tombstone(store):
    tenant = f"acme{uuid.uuid4().hex[:8]}"
    async with client_for(store) as client:
        created = await client.post(
            "/admin/harness/keys", json={"tenant_id": tenant, "name": "editor"}
        )
        assert created.status_code == 200, created.text
        record = created.json()
        digest = hashlib.sha256(record["key"].encode()).hexdigest()
        assert record["key_hash"] == digest
        assert record["key_prefix"] == digest[:12]
        assert (
            await asyncio.to_thread(keys(store).resolve, record["key"])
            == f"{tenant}:{tenant}"
        )
        public = {k: v for k, v in record.items() if k != "key"}
        assert set(public) == {
            "tenant_id",
            "name",
            "created_at",
            "revoked",
            "key_hash",
            "key_prefix",
        }
        for tid in (tenant, f"{tenant}:{tenant}"):
            listed = await client.get("/admin/harness/keys", params={"tenant_id": tid})
            assert listed.status_code == 200
            assert listed.json() == {"keys": [public], "continuation": None}
            assert record["key"] not in listed.text
        removed = await client.delete(f"/admin/harness/keys/{digest}")
        assert removed.status_code == 200
        assert removed.json() == {"revoked": True, "key_hash": digest}
        with pytest.raises(
            importlib.import_module(
                "cogniverse_runtime.harness_keys"
            ).HarnessKeyNotFoundError
        ):
            await asyncio.to_thread(keys(store).resolve, record["key"])
        listed = await client.get("/admin/harness/keys", params={"tenant_id": tenant})
        assert listed.json() == {
            "keys": [{**public, "revoked": True}],
            "continuation": None,
        }
        tombstone = await asyncio.to_thread(
            store.get_immutable_config,
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            "harness_key_revocations",
            digest,
        )
        assert tombstone.config_value == {"revoked": True}


@pytest.mark.asyncio
async def test_dead_port_routes_and_resolution_report_cause():
    store = VespaConfigStore(backend_url="http://127.0.0.1", backend_port=29071)
    try:
        async with client_for(store) as client:
            responses = await asyncio.gather(
                client.post(
                    "/admin/harness/keys", json={"tenant_id": "acme", "name": "dead"}
                ),
                client.get("/admin/harness/keys", params={"tenant_id": "acme"}),
                client.delete("/admin/harness/keys/" + "a" * 64),
            )
        assert [r.status_code for r in responses] == [503, 503, 503]
        for response in responses:
            assert "127.0.0.1" in response.json()["detail"]
            assert "29071" in response.json()["detail"]
        with pytest.raises(ConfigStoreUnavailableError, match="29071"):
            await asyncio.to_thread(keys(store).resolve, "unknown")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_validate_before_io():
    store = VespaConfigStore(backend_port=29071)
    try:
        async with client_for(store) as client:
            responses = await asyncio.gather(
                client.post(
                    "/admin/harness/keys", json={"tenant_id": "", "name": "editor"}
                ),
                client.post(
                    "/admin/harness/keys", json={"tenant_id": "acme", "name": " "}
                ),
                client.delete("/admin/harness/keys/not-a-hash"),
                client.get(
                    "/admin/harness/keys", params={"tenant_id": "acme", "page_size": 0}
                ),
            )
        assert [r.status_code for r in responses] == [422, 422, 422, 422]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_concurrent_revokes_keep_loop_responsive(store, monkeypatch):
    records = [
        await asyncio.to_thread(keys(store).create, "loop", str(i)) for i in range(20)
    ]
    original = store.vespa_app.feed_data_point

    def delayed(*args, **kwargs):
        time.sleep(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(store.vespa_app, "feed_data_point", delayed)
    gaps = []
    stopped = asyncio.Event()

    async def ticker():
        previous = time.monotonic()
        while not stopped.is_set():
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    ticking = asyncio.create_task(ticker())
    try:
        async with client_for(store) as client:
            responses = await asyncio.gather(
                *(
                    client.delete(f"/admin/harness/keys/{r['key_hash']}")
                    for r in records
                )
            )
    finally:
        stopped.set()
        await ticking
    assert [r.status_code for r in responses] == [200] * 20
    print(f"LOOP_MAX_GAP_MS={max(gaps) * 1000:.3f}")
    assert max(gaps) < 0.05
    for record in records:
        with pytest.raises(
            importlib.import_module(
                "cogniverse_runtime.harness_keys"
            ).HarnessKeyNotFoundError
        ):
            await asyncio.to_thread(keys(store).resolve, record["key"])


@pytest.mark.parametrize("operation", ["create", "revoke"])
def test_confirmation_failure_leaves_complete_record(store, monkeypatch, operation):
    module = importlib.import_module("cogniverse_runtime.harness_keys")
    record = keys(store).create("torn", operation)
    plaintext = "cgv-torn-create"
    digest = hashlib.sha256(plaintext.encode()).hexdigest()
    if operation == "create":
        monkeypatch.setattr(module, "generate_key", lambda: (plaintext, digest))
    else:
        digest = record["key_hash"]
    original = store.get_immutable_config

    def fail_confirmation(*args, **kwargs):
        raise ConfigStoreUnavailableError("confirmation disconnected")

    monkeypatch.setattr(store, "get_immutable_config", fail_confirmation)
    with pytest.raises(ConfigStoreUnavailableError, match="confirmation disconnected"):
        if operation == "create":
            keys(store).create("torn", "create")
        else:
            keys(store).revoke(digest)
    monkeypatch.setattr(store, "get_immutable_config", original)
    service = "harness_keys" if operation == "create" else "harness_key_revocations"
    entry = original(SYSTEM_TENANT_ID, ConfigScope.SYSTEM, service, digest)
    if operation == "create":
        assert {k: v for k, v in entry.config_value.items() if k != "created_at"} == {
            "tenant_id": "torn:torn",
            "name": "create",
            "revoked": False,
        }
        assert keys(store).resolve(plaintext) == "torn:torn"
    else:
        assert entry.config_value == {"revoked": True}
        with pytest.raises(module.HarnessKeyNotFoundError):
            keys(store).resolve(record["key"])


def test_revoke_tenant_and_pagination(store):
    tenant = f"bulk{uuid.uuid4().hex[:8]}"
    own = [keys(store).create(tenant, name) for name in ("one", "two")]
    other = keys(store).create(tenant + "other", "peer")
    assert keys(store).revoke_tenant(tenant) == 2
    assert keys(store).resolve(other["key"]) == f"{tenant}other:{tenant}other"
    for record in own:
        with pytest.raises(
            importlib.import_module(
                "cogniverse_runtime.harness_keys"
            ).HarnessKeyNotFoundError
        ):
            keys(store).resolve(record["key"])
    records = []
    continuation = None
    while True:
        page = keys(store).list(tenant, page_size=1, continuation=continuation)
        assert len(page["keys"]) <= 1
        records.extend(page["keys"])
        continuation = page["continuation"]
        if continuation is None:
            break
    assert {r["key_hash"] for r in records} == {r["key_hash"] for r in own}
    assert [r["revoked"] for r in records] == [True, True]


@pytest.mark.asyncio
async def test_pause_mid_request_reports_503(store, key_vespa, monkeypatch):
    original = store.vespa_app.feed_data_point
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()

    def pause_then_write(*args, **kwargs):
        subprocess.run(
            ["docker", "pause", key_vespa["container_name"]],
            check=True,
            capture_output=True,
        )
        loop.call_soon_threadsafe(entered.set)
        return original(*args, **kwargs)

    monkeypatch.setattr(store.vespa_app, "feed_data_point", pause_then_write)
    try:
        async with client_for(store) as client:
            create = asyncio.create_task(
                client.post(
                    "/admin/harness/keys",
                    json={"tenant_id": "paused", "name": "editor"},
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            monkeypatch.setattr(store.vespa_app, "feed_data_point", original)
            results = await asyncio.gather(
                create,
                client.get("/admin/harness/keys", params={"tenant_id": "paused"}),
                client.delete("/admin/harness/keys/" + "b" * 64),
                asyncio.to_thread(keys(store).resolve, "paused-key"),
                return_exceptions=True,
            )
        assert [r.status_code for r in results[:3]] == [503, 503, 503]
        for response in results[:3]:
            assert str(key_vespa["http_port"]) in response.json()["detail"]
        assert type(results[3]) is ConfigStoreUnavailableError
        assert str(key_vespa["http_port"]) in str(results[3])
    finally:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "unpause", key_vespa["container_name"]],
            capture_output=True,
        )


@pytest.mark.asyncio
async def test_tenant_delete_revokes_before_metadata_removal(
    store, key_vespa, monkeypatch
):
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_runtime.admin import tenant_manager as tm
    from cogniverse_vespa.backend import VespaBackend

    cm = ConfigManager(store=store)
    backend = VespaBackend(
        backend_config=BackendConfig(
            tenant_id="system",
            backend_type="vespa",
            url="http://localhost",
            port=key_vespa["http_port"],
        ),
        schema_loader=FilesystemSchemaLoader("configs/schemas"),
        config_manager=cm,
    )
    from cogniverse_core.registries.schema_registry import SchemaRegistry

    backend.schema_registry = SchemaRegistry(
        config_manager=cm,
        backend=backend,
        schema_loader=backend._schema_loader_instance,
    )
    backend.initialize({"tenant_id": SYSTEM_TENANT_ID})
    backend.schema_manager.backend_port = key_vespa["config_port"]
    monkeypatch.setattr(tm, "backend", backend)
    monkeypatch.setattr(tm, "_config_manager", cm)
    app = FastAPI()
    app.include_router(tm.router, prefix="/admin")
    tenant = "delete" + uuid.uuid4().hex[:8]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    ) as client:
        response = await client.post(
            "/admin/tenants", json={"tenant_id": tenant, "created_by": "test"}
        )
        assert response.status_code == 200, response.text
        record = await asyncio.to_thread(keys(store).create, tenant, "editor")
        original = store.put_immutable_config

        def fail(*args, **kwargs):
            raise ConfigStoreUnavailableError("revocation disconnected")

        monkeypatch.setattr(store, "put_immutable_config", fail)
        failed = await client.delete(f"/admin/tenants/{tenant}")
        assert failed.status_code == 503
        assert failed.json() == {"detail": "revocation disconnected"}
        retained = await client.get(f"/admin/tenants/{tenant}")
        assert retained.status_code == 200
        assert retained.json()["tenant_full_id"] == f"{tenant}:{tenant}"
        monkeypatch.setattr(store, "put_immutable_config", original)
        deleted = await client.delete(f"/admin/tenants/{tenant}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["tenant_full_id"] == f"{tenant}:{tenant}"
        with pytest.raises(
            importlib.import_module(
                "cogniverse_runtime.harness_keys"
            ).HarnessKeyNotFoundError
        ):
            await asyncio.to_thread(keys(store).resolve, record["key"])


def test_revoke_uses_one_write_and_one_confirmation(store, monkeypatch):
    """Revocation is one data-plane write followed by one confirming document
    read; the read goes through the store's bounded reader, the write through
    pyvespa."""
    from cogniverse_vespa.config import config_store as store_module

    record = keys(store).create("bounded", "editor")
    calls = []
    for method in ("feed_data_point", "get_data", "query"):
        original = getattr(store.vespa_app, method)

        def tracked(*args, _method=method, _original=original, **kwargs):
            calls.append(_method)
            return _original(*args, **kwargs)

        monkeypatch.setattr(store.vespa_app, method, tracked)
    real_read = store_module._config_store_read_json

    def tracked_read(*args, **kwargs):
        calls.append(f"read:{kwargs.get('operation', 'visit')}")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(store_module, "_config_store_read_json", tracked_read)
    assert keys(store).revoke(record["key_hash"]) is True
    assert calls == ["feed_data_point", "read:document"]


def test_immutable_collision_preserves_original(store):
    key = uuid.uuid4().hex
    original = store.put_immutable_config(
        SYSTEM_TENANT_ID, ConfigScope.SYSTEM, "immutable_test", key, {"owner": "one"}
    )
    with pytest.raises(ValueError, match="immutable config.*different value"):
        store.put_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            "immutable_test",
            key,
            {"owner": "two"},
        )
    loaded = store.get_immutable_config(
        SYSTEM_TENANT_ID, ConfigScope.SYSTEM, "immutable_test", key
    )
    assert loaded == original


def test_store_canonicalizes_raw_create_and_both_listing_forms(store):
    tenant = "raw" + uuid.uuid4().hex[:8]
    record = keys(store).create(tenant, "editor")
    public = {k: v for k, v in record.items() if k != "key"}
    assert public["tenant_id"] == f"{tenant}:{tenant}"
    assert keys(store).resolve(record["key"]) == f"{tenant}:{tenant}"
    assert keys(store).list(tenant) == {"keys": [public], "continuation": None}
    assert keys(store).list(f"{tenant}:{tenant}") == {
        "keys": [public],
        "continuation": None,
    }
