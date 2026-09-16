"""Signature-variant selections must persist and reach every replica.

A PUT of a tenant's per-agent variant wrote only the process dict of the
replica that served it — lost on restart and invisible to the dispatcher on
every other replica, while the route returned 200 as if applied. These drive
the REAL admin route against a REAL Phoenix container: a PUT persists the blob,
a cold replica reads it back, and the dispatcher's resolver returns the exact
persisted variant.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "sigvar-persist:sigvar-persist"


@pytest.fixture
def real_admin(telemetry_manager_with_phoenix, monkeypatch):
    provider = telemetry_manager_with_phoenix.get_provider(tenant_id=TENANT)
    monkeypatch.setattr(
        admin_router,
        "_build_artifact_manager",
        lambda key: ArtifactManager(provider, tenant_id=key),
    )
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()


async def _put(app, tenant, agent, variant):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(
            f"/admin/tenants/{tenant}/signature_variants/{agent}",
            json={"variant_id": variant},
        )


@pytest.mark.asyncio
async def test_variant_persists_and_resolves_on_cold_replica(real_admin):
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")

    resp = await _put(app, TENANT, "search_agent", "search_v2")
    assert resp.status_code == 200
    assert resp.json()["selections"]["search_agent"] == "search_v2"

    # Land the write-behind persist, then cold replica: cache cleared. It
    # must resolve the persisted variant from the durable blob, not fall
    # back to the default.
    await admin_router._blob_write_queue.flush()
    admin_router._reset_admin_overrides_for_tests()
    loaded = await admin_router.load_signature_variants(TENANT)
    assert loaded == {"search_agent": "search_v2"}

    # The dispatcher's resolver (the real consumer) reads the warmed cache.
    assert (
        AgentDispatcher._resolve_signature_variant(TENANT, "search_agent")
        == "search_v2"
    )
    # An agent the tenant never selected still resolves to the default.
    assert (
        AgentDispatcher._resolve_signature_variant(TENANT, "summarizer_agent")
        == "default"
    )


@pytest.mark.asyncio
async def test_second_agent_selection_merges_not_replaces(real_admin):
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")

    await _put(app, TENANT, "search_agent", "search_v2")
    resp = await _put(app, TENANT, "summarizer_agent", "sum_v3")
    assert resp.status_code == 200

    await admin_router._blob_write_queue.flush()
    admin_router._reset_admin_overrides_for_tests()
    loaded = await admin_router.load_signature_variants(TENANT)
    assert loaded == {"search_agent": "search_v2", "summarizer_agent": "sum_v3"}


@pytest.fixture
def replicas(phoenix_container):
    """Independent router module state against one real Phoenix server."""
    import importlib.util
    import sys
    import uuid

    modules = []
    for index in range(2):
        name = f"cogniverse_runtime.routers.state_replica_{uuid.uuid4().hex}_{index}"
        spec = importlib.util.spec_from_file_location(name, admin_router.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.set_phoenix_endpoints(
            phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
        )
        modules.append(module)
    yield modules
    for module in modules:
        module._reset_admin_overrides_for_tests()
        sys.modules.pop(module.__name__)


def _replica_client(module):
    app = FastAPI()
    app.include_router(module.router, prefix="/admin")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://replica")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pin_quotas", "signature_variants"])
async def test_warm_replica_partial_put_preserves_other_replica_fields(replicas, kind):
    import json
    import uuid

    tenant = f"prodfixstate:t{uuid.uuid4().hex}"
    first, second = replicas
    manager = first._build_artifact_manager(tenant)
    initial = (
        {"user": 1, "tenant_admin": 2, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "initial"}
    )
    await manager.save_blob("config", kind, json.dumps(initial))
    path = f"/admin/tenants/{tenant}/{kind}"
    async with _replica_client(first) as a, _replica_client(second) as b:
        warm = await b.get(path)
        assert warm.status_code == 200
        field = "quotas" if kind == "pin_quotas" else "selections"
        assert warm.json()[field] == initial
        first_path = path if kind == "pin_quotas" else path + "/search_agent"
        second_path = path if kind == "pin_quotas" else path + "/summarizer_agent"
        response = await a.put(
            first_path,
            json={"user": 7} if kind == "pin_quotas" else {"variant_id": "search-v2"},
        )
        assert response.status_code == 200
        await first._blob_write_queue.flush()
        response = await b.put(
            second_path,
            json={"tenant_admin": 9}
            if kind == "pin_quotas"
            else {"variant_id": "summary-v3"},
        )
        assert response.status_code == 200
        expected = (
            {"user": 7, "tenant_admin": 9, "org_admin": -1}
            if kind == "pin_quotas"
            else {"search_agent": "search-v2", "summarizer_agent": "summary-v3"}
        )
        assert response.json() == {
            "tenant_id": tenant,
            field: expected,
            "pending_write": True,
        }
        await second._blob_write_queue.flush()
        assert json.loads(await manager.load_blob("config", kind)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pin_quotas", "signature_variants"])
async def test_sequential_puts_keep_a_peer_field_whose_persist_has_not_landed(
    replicas, kind
):
    """The second PUT lands after the first replica's persist is still queued.

    Both PUTs are strictly sequential — the first replica answered 200 before
    the second was issued — but the first replica's write-behind persist has
    not reached the store, so the second replica's merge base cannot contain
    it. The field each PUT changed is replayed when that PUT is persisted, so
    neither erases the other.
    """
    import asyncio
    import json
    import uuid

    from cogniverse_runtime.blob_write_queue import BlobWriteQueue

    tenant = f"prodfixstate:t{uuid.uuid4().hex}"
    first, second = replicas
    manager = first._build_artifact_manager(tenant)
    initial = (
        {"user": 1, "tenant_admin": 2, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "initial"}
    )
    await manager.save_blob("config", kind, json.dumps(initial))
    path = f"/admin/tenants/{tenant}/{kind}"
    field = "quotas" if kind == "pin_quotas" else "selections"

    persist = asyncio.Event()
    applier = first._apply_blob_write

    async def gated_apply(*args):
        await persist.wait()
        await applier(*args)

    first._blob_write_queue = BlobWriteQueue(gated_apply)

    async with _replica_client(first) as a, _replica_client(second) as b:
        assert (await b.get(path)).json()[field] == initial
        first_path = path if kind == "pin_quotas" else path + "/search_agent"
        second_path = path if kind == "pin_quotas" else path + "/summarizer_agent"

        accepted = await a.put(
            first_path,
            json={"user": 7} if kind == "pin_quotas" else {"variant_id": "search-v2"},
        )
        assert accepted.status_code == 200
        assert accepted.json() == {
            "tenant_id": tenant,
            field: (
                {"user": 7, "tenant_admin": 2, "org_admin": -1}
                if kind == "pin_quotas"
                else {"search_agent": "search-v2"}
            ),
            "pending_write": True,
        }
        # Nothing of the first PUT has reached the store yet.
        assert json.loads(await manager.load_blob("config", kind)) == initial

        peer = await b.put(
            second_path,
            json={"tenant_admin": 9}
            if kind == "pin_quotas"
            else {"variant_id": "summary-v3"},
        )
        assert peer.status_code == 200
        await second._blob_write_queue.flush()

        persist.set()
        await first._blob_write_queue.flush()

    expected = (
        {"user": 7, "tenant_admin": 9, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "search-v2", "summarizer_agent": "summary-v3"}
    )
    assert json.loads(await manager.load_blob("config", kind)) == expected
    assert first._blob_write_queue.status() == {"pending": 0, "failed": []}
    assert second._blob_write_queue.status() == {"pending": 0, "failed": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pin_quotas", "signature_variants"])
async def test_partial_put_serializes_fresh_read_and_pending_overlay(
    replicas, phoenix_container, kind
):
    import asyncio
    import json
    import threading
    import uuid

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    tenant = f"prodfixstate:t{uuid.uuid4().hex}"
    first, second = replicas
    manager = first._build_artifact_manager(tenant)
    initial = (
        {"user": 1, "tenant_admin": 2, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "initial"}
    )
    await manager.save_blob("config", kind, json.dumps(initial))
    path = f"/admin/tenants/{tenant}/{kind}"
    async with _replica_client(second) as client:
        assert (await client.get(path)).status_code == 200
        remote = dict(initial)
        remote["user" if kind == "pin_quotas" else "search_agent"] = (
            7 if kind == "pin_quotas" else "search-v2"
        )
        await manager.save_blob("config", kind, json.dumps(remote))
        entered, release = threading.Event(), threading.Event()
        with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:

            def intercept(method, path, body):
                if method == "GET" and not entered.is_set():
                    entered.set()
                    if not release.wait(15):
                        return 504, {"error": "read barrier expired"}
                return None

            proxy.intercept = intercept
            second.set_phoenix_endpoints(proxy.url, phoenix_container["grpc_endpoint"])
            first_path = path if kind == "pin_quotas" else path + "/summarizer_agent"
            second_path = (
                path if kind == "pin_quotas" else path + "/detailed_report_agent"
            )
            request = asyncio.create_task(
                client.put(
                    first_path,
                    json={"tenant_admin": 9}
                    if kind == "pin_quotas"
                    else {"variant_id": "summary-v3"},
                )
            )
            sibling = None
            try:
                assert await asyncio.to_thread(entered.wait, 3) is True
                sibling = asyncio.create_task(
                    client.put(
                        second_path,
                        json={"org_admin": 12}
                        if kind == "pin_quotas"
                        else {"variant_id": "report-v4"},
                    )
                )
                await asyncio.sleep(0)
                assert sibling.done() is False
                release.set()
                responses = await asyncio.gather(request, sibling)
                assert [response.status_code for response in responses] == [200, 200]
                await second._blob_write_queue.flush()
            finally:
                release.set()
                await asyncio.gather(
                    request, *([sibling] if sibling else []), return_exceptions=True
                )
                await second._blob_write_queue.flush()
    expected = (
        {"user": 7, "tenant_admin": 9, "org_admin": 12}
        if kind == "pin_quotas"
        else {
            "search_agent": "search-v2",
            "summarizer_agent": "summary-v3",
            "detailed_report_agent": "report-v4",
        }
    )
    assert json.loads(await manager.load_blob("config", kind)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pin_quotas", "signature_variants"])
async def test_warm_partial_put_read_failure_returns_503_without_mutation(
    replicas, phoenix_container, kind
):
    import json
    import uuid

    from tests.utils.http_fault_proxy import InterceptFaultProxy

    tenant = f"prodfixstate:t{uuid.uuid4().hex}"
    first, second = replicas
    manager = first._build_artifact_manager(tenant)
    initial = (
        {"user": 1, "tenant_admin": 2, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "initial"}
    )
    await manager.save_blob("config", kind, json.dumps(initial))
    path = f"/admin/tenants/{tenant}/{kind}"
    async with _replica_client(second) as client:
        assert (await client.get(path)).status_code == 200
        with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
            proxy.intercept = lambda method, path, body: (
                503,
                {"error": "store offline"},
            )
            second.set_phoenix_endpoints(proxy.url, phoenix_container["grpc_endpoint"])
            response = await client.put(
                path if kind == "pin_quotas" else path + "/summarizer_agent",
                json={"user": 7}
                if kind == "pin_quotas"
                else {"variant_id": "summary-v3"},
            )
            try:
                assert response.status_code == 503
                prefix = "pin-quota" if kind == "pin_quotas" else "signature-variant"
                assert response.json()["detail"].startswith(
                    f"{prefix} store unavailable: "
                )
                assert second._blob_write_queue.status() == {"pending": 0, "failed": []}
            finally:
                proxy.intercept = None
                await second._blob_write_queue.flush()
    assert json.loads(await manager.load_blob("config", kind)) == initial
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        proxy.intercept = lambda method, path, body: (
            (503, {"error": "publication refused"})
            if method == "POST" and path.startswith("/v1/datasets/upload")
            else None
        )
        second.set_phoenix_endpoints(proxy.url, phoenix_container["grpc_endpoint"])
        async with _replica_client(second) as client:
            accepted = await client.put(
                path if kind == "pin_quotas" else path + "/search_agent",
                json={"user": 7}
                if kind == "pin_quotas"
                else {"variant_id": "search-v2"},
            )
            assert accepted.status_code == 200
            assert accepted.json()["pending_write"] is True
            await second._blob_write_queue.flush()
            assert second._blob_write_queue.status() == {
                "pending": 0,
                "failed": [(tenant, "config", kind)],
            }
            failed_read = await client.get(path)
            assert failed_read.status_code == 503
            proxy.intercept = None
            recovered = await client.put(
                path if kind == "pin_quotas" else path + "/summarizer_agent",
                json={"tenant_admin": 9}
                if kind == "pin_quotas"
                else {"variant_id": "summary-v3"},
            )
            assert recovered.status_code == 200
            await second._blob_write_queue.flush()
    expected = (
        {"user": 7, "tenant_admin": 9, "org_admin": -1}
        if kind == "pin_quotas"
        else {"search_agent": "search-v2", "summarizer_agent": "summary-v3"}
    )
    assert json.loads(await manager.load_blob("config", kind)) == expected
    assert second._blob_write_queue.status() == {"pending": 0, "failed": []}
