"""Deleting a tenant takes its memory schema and the memories in it.

The memory path deploys ``agent_memories_<tenant>`` and
``provenance_<tenant>`` on first use (cogniverse_core.memory.manager
``_build_and_store_memory``), so a tenant that ever used memory owns two
schemas the delete has to drop. Every schema left behind is carried by
every later application package, and its documents stay readable under a
tenant id that no longer resolves.

The ordering under test is deletion marker first, then data and schema,
tenant record last: from the marker on, every process refuses the tenant's
memory writes and schema deploys, and each later step that fails leaves the
tenant record present, so the delete is retryable and the residue is
observable through the tenant registry. Every worker process, each with its
own cluster-events subscription, releases what it holds for the tenant
before anything is dropped.
"""

from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing
import os
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import requests
from fastapi import HTTPException
from requests.exceptions import ConnectionError
from vespa.application import Vespa
from vespa.exceptions import VespaError

from cogniverse_core.common.tenant_utils import TenantDeletedError, tenant_is_deleted
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.admin.models import CreateTenantRequest
from cogniverse_runtime.cluster_events import ClusterEvents

pytestmark = pytest.mark.integration

MEMORY_BASE_SCHEMAS = ["agent_memories", "provenance"]
EMBEDDING_DIMS = 768

# The config server drops a schema on activate; the query container's
# source-ref config follows. Measured at 5.05s on the shared test Vespa with
# the k3d cluster resident. The ceiling is 60s -- headroom for a loaded box,
# still far under a budget that would pass while the schema never converged.
SOURCE_REF_CONVERGENCE_CEILING_S = 60.0

# Two memories fed verbatim; the delete has to take these, not just the
# schema definition.
MEMORIES = (
    {
        "id": "mem-orphan-1",
        "text": "The operator prefers dry-run before any schema removal.",
        "agent_id": "search_agent",
        "session_id": "sess-a",
    },
    {
        "id": "mem-orphan-2",
        "text": "Vespa redeploys cost one slot per tenant schema.",
        "agent_id": "summarizer_agent",
        "session_id": "sess-b",
    },
)


@pytest.fixture
async def cluster_events(workflow_state_redis_url):
    """This process as one runtime worker on its own cluster-events channel."""
    events = ClusterEvents(
        workflow_state_redis_url,
        f"test-worker-{os.getpid()}",
        {"tenant_deleted": tm.release_deleted_tenant},
        channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
    )
    await events.start()
    yield events
    await events.close()


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader, cluster_events):
    """tenant_manager wired to the test Vespa and this process's cluster-events
    channel, module seams restored after."""
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    previous_cluster_events = tm._cluster_events
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    tm.set_cluster_events(cluster_events)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    tm.set_cluster_events(previous_cluster_events)
    BackendRegistry.get_instance().clear_instances()


def _unique_tenant() -> str:
    """A tenant id this test owns outright on the shared Vespa."""
    return f"memdel{uuid.uuid4().hex[:8]}:t1"


def _schema_names(tenant_id: str) -> list[str]:
    suffix = tenant_id.replace(":", "_")
    return [f"{base}_{suffix}" for base in MEMORY_BASE_SCHEMAS]


def _feed_memories(base_url: str, schema: str) -> None:
    statuses: list[tuple[str, int]] = []
    documents = [
        {
            "id": memory["id"],
            "fields": {
                "id": memory["id"],
                "text": memory["text"],
                "agent_id": memory["agent_id"],
                "session_id": memory["session_id"],
                "user_id": schema,
                "subject_key": "operator",
                "metadata_": "{}",
                "created_at": 1757000000,
                "embedding": [0.0125] * EMBEDDING_DIMS,
            },
        }
        for memory in MEMORIES
    ]
    Vespa(url=base_url).feed_iterable(
        documents,
        schema=schema,
        namespace=schema,
        callback=lambda response, doc_id: statuses.append(
            (doc_id, response.status_code)
        ),
    )
    assert sorted(statuses) == [("mem-orphan-1", 200), ("mem-orphan-2", 200)]


def _assert_memories_readable(base_url: str, schema: str) -> None:
    """Both memories come back by id and through a query on the schema."""
    app = Vespa(url=base_url)
    for memory in MEMORIES:
        stored = app.get_data(schema=schema, namespace=schema, data_id=memory["id"])
        assert stored.status_code == 200
        assert stored.json["id"] == f"id:{schema}:{schema}::{memory['id']}"
        assert stored.json["fields"]["text"] == memory["text"]
        assert stored.json["fields"]["agent_id"] == memory["agent_id"]

    found = app.query(body={"yql": f"select * from {schema} where true", "hits": 10})
    assert found.status_code == 200
    assert found.json["root"]["fields"]["totalCount"] == len(MEMORIES)
    assert sorted(
        (hit["fields"]["id"], hit["fields"]["text"])
        for hit in found.json["root"]["children"]
    ) == sorted((memory["id"], memory["text"]) for memory in MEMORIES)


def _provenance_fields(tenant_id: str) -> dict:
    return {
        "id": "provenance-memory-1",
        "memory_id": MEMORIES[0]["id"],
        "tenant_id": tenant_id,
        "written_by": "search_agent",
        "written_at": 1757000000,
        "derivation_kind": "observation",
        "confidence": 0.75,
        "derived_from_ids": ["source-memory"],
        "derived_from_other": '{"source":"operator"}',
        "trace_id": "trace-memory-1",
    }


def _assert_provenance_readable(base_url: str, tenant_id: str) -> None:
    schema = _schema_names(tenant_id)[1]
    stored = Vespa(url=base_url).get_data(
        schema=schema, namespace=schema, data_id="provenance-memory-1"
    )
    assert stored.status_code == 200
    assert stored.json["fields"] == _provenance_fields(tenant_id)


def _wait_until_source_unresolvable(
    base_url: str, schema: str, ceiling_s: float = SOURCE_REF_CONVERGENCE_CEILING_S
) -> tuple[float, list]:
    """Wait until a query on ``schema`` stops resolving, returning how long it
    took and the exact error list Vespa answered with.

    The config server drops the schema from the application synchronously on
    activate; the query container's source-ref config follows. Waiting on THIS
    schema's own resolve failure, not on "some error", is what makes the wait
    mean anything.
    """
    app = Vespa(url=base_url)
    deadline = time.monotonic() + ceiling_s
    started = time.monotonic()
    while True:
        try:
            app.query(body={"yql": f"select * from {schema} where true", "hits": 1})
        except VespaError as exc:
            errors = exc.args[0]
            if errors and errors[0].get("code") == 4:
                return time.monotonic() - started, errors
            raise
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{schema} still resolved as a query source {ceiling_s}s after "
                f"it was dropped from the deployed application"
            )
        time.sleep(1.0)


async def _create_tenant_with_memory(tenant_id: str, base_url: str) -> str:
    """Create the tenant through the real route and fill its memory schema."""
    created = await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id,
            created_by="memory-orphan-test",
            base_schemas=list(MEMORY_BASE_SCHEMAS),
        )
    )
    assert created.tenant_full_id == tenant_id
    assert sorted(created.schemas_deployed) == sorted(MEMORY_BASE_SCHEMAS)

    memory_schema, provenance_schema = _schema_names(tenant_id)
    deployed = set(tm.get_backend().schema_manager.list_deployed_document_types())
    assert {memory_schema, provenance_schema} <= deployed

    _feed_memories(base_url, memory_schema)
    _assert_memories_readable(base_url, memory_schema)
    responses = []
    Vespa(url=base_url).feed_iterable(
        [{"id": "provenance-memory-1", "fields": _provenance_fields(tenant_id)}],
        schema=provenance_schema,
        namespace=provenance_schema,
        callback=lambda response, doc_id: responses.append(
            (doc_id, response.status_code)
        ),
    )
    assert responses == [("provenance-memory-1", 200)]
    _assert_provenance_readable(base_url, tenant_id)
    return memory_schema


@pytest.mark.asyncio
async def test_delete_tenant_removes_memory_schema_and_its_documents(
    wired_tenant_manager, vespa_instance
):
    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)

    await _create_tenant_with_memory(tenant_id, base_url)
    peer_id = f"peer_{tenant_id}"
    peer_schema = await _create_tenant_with_memory(peer_id, base_url)

    result = await tm.delete_tenant_internal(tenant_id)

    assert result["status"] == "deleted"
    assert result["tenant_full_id"] == tenant_id
    assert sorted(result["deleted_schemas"]) == sorted(
        [memory_schema, provenance_schema]
    )
    assert result["schemas_deleted"] == 2
    _assert_memories_readable(base_url, peer_schema)
    _assert_provenance_readable(base_url, peer_id)
    assert (await tm.get_tenant_internal(peer_id)).tenant_full_id == peer_id

    # The schema is gone from the deployed application, not merely unregistered.
    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert deployed & {memory_schema, provenance_schema} == set()

    # The same query that returned both memories before the delete stops
    # resolving the source once the query container picks up the new
    # application config: error code 4, and the enumeration of valid source
    # refs no longer names this schema. The document type is gone from the
    # content cluster, so its documents are gone with it.
    elapsed, errors = _wait_until_source_unresolvable(base_url, memory_schema)
    assert len(errors) == 1
    assert errors[0]["code"] == 4
    assert errors[0]["summary"] == "Invalid query parameter"
    assert errors[0]["message"].startswith(
        f"Could not resolve source ref '{memory_schema}'. Valid source refs are "
    )
    valid_refs = {
        ref.strip().rstrip(".")
        for ref in errors[0]["message"].split("Valid source refs are ", 1)[1].split(",")
    }
    assert f"cogniverse_content.{memory_schema}" not in valid_refs
    assert f"cogniverse_content.{provenance_schema}" not in valid_refs
    assert "cogniverse_content.tenant_metadata" in valid_refs
    print(f"MEASURED_SOURCE_REF_CONVERGENCE_S {elapsed:.2f}")

    # No residue: the tenant record is gone and nothing reports an orphan.
    assert await tm.get_tenant_internal(tenant_id) is None
    diff = tm._list_orphan_schemas()
    assert [
        n
        for n in diff["tenant_orphan_schemas"]
        if n.endswith(tenant_id.replace(":", "_"))
    ] == []
    assert [
        n for n in diff["orphan_schemas"] if n.endswith(tenant_id.replace(":", "_"))
    ] == []


@pytest.mark.asyncio
async def test_schema_removal_failure_retains_tenant_record_and_documents(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    """Data and schema go first, so a failure there leaves the tenant record
    present and the memories intact, and the retry completes. This is what
    the ordering buys instead of a ``deleting`` state."""
    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)

    await _create_tenant_with_memory(tenant_id, base_url)

    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    real_deploy = VespaSchemaManager._deploy_package
    requests = []
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        unavailable_port = unavailable.getsockname()[1]

        def deploy_to_unavailable_port(manager, package, **kwargs):
            isolated_manager = copy.copy(manager)
            isolated_manager.backend_port = unavailable_port
            return real_deploy(isolated_manager, package, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(
                VespaSchemaManager, "_deploy_package", deploy_to_unavailable_port
            )
            with pytest.raises(ConnectionError) as failure:
                await tm.delete_tenant_internal(tenant_id)
            requests.append((failure.value.request.method, failure.value.request.url))
        assert requests == [
            (
                "POST",
                f"http://localhost:{unavailable_port}/application/v2/tenant/default/session",
            )
        ]

    # Tenant record survives, so the delete is observably incomplete and
    # retryable through the same route; the tenant stays marked deleted, so
    # its writes are refused meanwhile.
    retained = await tm.get_tenant_internal(tenant_id)
    assert retained.tenant_full_id == tenant_id
    assert retained.status == "active"
    assert tenant_is_deleted(tm._config_manager.store, tenant_id) is True

    # Schemas and memories are untouched.
    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert {memory_schema, provenance_schema} <= deployed
    _assert_memories_readable(base_url, memory_schema)
    _assert_provenance_readable(base_url, tenant_id)

    result = await tm.delete_tenant_internal(tenant_id)
    assert sorted(result["deleted_schemas"]) == sorted(
        [memory_schema, provenance_schema]
    )
    assert await tm.get_tenant_internal(tenant_id) is None
    deployed_after = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert deployed_after & {memory_schema, provenance_schema} == set()


@pytest.mark.asyncio
async def test_concurrent_deletes_of_one_tenant_both_complete(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    """Two requests deleting the same tenant interleave on one loop: exactly
    one drops the schemas, and the loser reports an empty drop rather than a
    500 for a delete whose effect is already durable."""
    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)

    await _create_tenant_with_memory(tenant_id, base_url)

    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    barrier = threading.Barrier(2, timeout=60)
    calls = []
    real_delete = VespaSchemaManager.delete_tenant_schemas

    def concurrent_delete(manager, requested_tenant):
        calls.append(requested_tenant)
        barrier.wait()
        return real_delete(manager, requested_tenant)

    monkeypatch.setattr(VespaSchemaManager, "delete_tenant_schemas", concurrent_delete)

    outcomes = await asyncio.gather(
        tm.delete_tenant_internal(tenant_id),
        tm.delete_tenant_internal(tenant_id),
        return_exceptions=True,
    )

    assert calls == [tenant_id, tenant_id]
    assert [type(outcome) for outcome in outcomes] == [dict, dict], outcomes
    assert [outcome["status"] for outcome in outcomes] == ["deleted", "deleted"]
    dropped = sorted(
        (sorted(outcome["deleted_schemas"]) for outcome in outcomes),
        key=len,
    )
    assert dropped == [[], sorted([memory_schema, provenance_schema])]

    assert await tm.get_tenant_internal(tenant_id) is None
    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert deployed & {memory_schema, provenance_schema} == set()


@pytest.mark.asyncio
async def test_delete_without_a_tenant_record_or_schemas_is_a_404(
    wired_tenant_manager,
):
    tenant_id = _unique_tenant()
    with pytest.raises(HTTPException) as exc:
        await tm.delete_tenant_internal(tenant_id)
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Tenant {tenant_id} not found"
    # Nothing existed, so the id is not left marked deleted.
    assert tenant_is_deleted(tm._config_manager.store, tenant_id) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("bulk", [False, True])
async def test_concurrent_deploy_cannot_restore_a_schema_awaiting_unregistration(
    wired_tenant_manager, vespa_instance, monkeypatch, bulk
):
    from cogniverse_core.registries.schema_registry import SchemaRegistry

    tenant = _unique_tenant()
    await _create_tenant_with_memory(tenant, vespa_instance["base_url"])
    backend = tm.get_backend()
    registry = backend.schema_manager._schema_registry
    peer_registry = SchemaRegistry(tm._config_manager, backend, tm._schema_loader)
    peer = _unique_tenant()
    peer_schema = f"agent_memories_{peer.replace(':', '_')}"
    unregister = registry.unregister_schema
    read = peer_registry._get_all_schemas
    waiting = threading.Event()
    release_delete = threading.Event()
    snapshot_taken = threading.Event()
    release_snapshot = threading.Event()
    order = []
    remaining = set(MEMORY_BASE_SCHEMAS)

    def delayed_unregister(tid, base):
        if tid == tenant and base == "agent_memories":
            waiting.set()
            assert release_delete.wait(60) is True
        unregister(tid, base)
        if tid == tenant:
            remaining.remove(base)
            if remaining == set():
                order.append("unregistered")

    def capture_snapshot():
        rows = read()
        order.append("snapshot")
        snapshot_taken.set()
        assert release_snapshot.wait(60) is True
        return rows

    monkeypatch.setattr(registry, "unregister_schema", delayed_unregister)
    monkeypatch.setattr(peer_registry, "_get_all_schemas", capture_snapshot)
    if bulk:
        monkeypatch.setattr(
            backend.schema_manager,
            "delete_tenant_schemas",
            lambda tid: backend.schema_manager.delete_tenant_schemas_bulk([tid]),
        )
    deletion = asyncio.create_task(tm.delete_tenant_internal(tenant))
    deployment = None
    try:
        assert await asyncio.to_thread(waiting.wait, 60) is True
        deployment = asyncio.create_task(
            asyncio.to_thread(peer_registry.deploy_schema, peer, "agent_memories")
        )
        await asyncio.to_thread(snapshot_taken.wait, 1)
        release_delete.set()
        release_snapshot.set()
        result = await deletion
        assert await deployment == peer_schema
        assert sorted(result["deleted_schemas"]) == sorted(_schema_names(tenant))
        assert order == ["unregistered", "snapshot"]
        assert set(backend.schema_manager.list_deployed_document_types()) & {
            *_schema_names(tenant),
            peer_schema,
        } == {peer_schema}
        assert registry.get_tenant_schemas(tenant) == []
        assert await tm.get_tenant_internal(tenant) is None
    finally:
        release_delete.set()
        release_snapshot.set()
        await asyncio.gather(
            deletion, *([deployment] if deployment else []), return_exceptions=True
        )


def _seed_organization(org_id: str) -> None:
    assert (
        tm.get_backend().create_metadata_document(
            schema="organization_metadata",
            doc_id=org_id,
            fields={
                "org_id": org_id,
                "org_name": "Deletion contract",
                "created_at": 1757000000000,
                "created_by": "tenant-delete-test",
                "status": "active",
            },
        )
        is True
    )


@pytest.mark.asyncio
async def test_organization_child_failure_retains_parent_and_retry_deletes_remaining(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    failed_id = _unique_tenant()
    org_id = failed_id.split(":")[0]
    sibling_id = f"{org_id}:sibling"
    await _create_tenant_with_memory(failed_id, vespa_instance["base_url"])
    assert (
        tm.get_backend().create_metadata_document(
            schema="tenant_metadata",
            doc_id=sibling_id,
            fields={
                "tenant_full_id": sibling_id,
                "org_id": org_id,
                "tenant_name": "sibling",
                "created_at": 1757000000000,
                "created_by": "tenant-delete-test",
                "status": "active",
                "schemas_deployed": [],
            },
        )
        is True
    )
    real_deploy = VespaSchemaManager._deploy_package
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))

        def refuse_schema_drop(manager, package, **kwargs):
            isolated_manager = copy.copy(manager)
            isolated_manager.backend_port = unavailable.getsockname()[1]
            return real_deploy(isolated_manager, package, **kwargs)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tm.app), base_url="http://tenant-test"
        ) as client:
            with monkeypatch.context() as patch:
                patch.setattr(VespaSchemaManager, "_deploy_package", refuse_schema_drop)
                response = await client.delete(f"/admin/organizations/{org_id}")
            assert response.status_code == 503
            assert response.json() == {
                "detail": {
                    "message": f"Organization {org_id} deletion incomplete; retry the delete",
                    "org_id": org_id,
                    "deleted_tenant_ids": [sibling_id],
                    "failed_tenant_ids": [failed_id],
                }
            }
            assert (await tm.get_organization_internal(org_id)).org_id == org_id
            assert (await tm.get_tenant_internal(failed_id)).tenant_full_id == failed_id
            assert await tm.get_tenant_internal(sibling_id) is None
            _assert_memories_readable(
                vespa_instance["base_url"], _schema_names(failed_id)[0]
            )
            retried = await client.delete(f"/admin/organizations/{org_id}")
            assert retried.status_code == 200
            assert retried.json() == {
                "status": "deleted",
                "org_id": org_id,
                "tenants_deleted": 1,
                "deleted_tenant_ids": [failed_id],
            }
    assert await tm.get_tenant_internal(failed_id) is None
    assert await tm.get_organization_internal(org_id) is None


@pytest.mark.asyncio
async def test_overlapping_organization_deletes_retain_parent_on_child_failure(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    tenant_id = _unique_tenant()
    org_id = tenant_id.split(":")[0]
    await _create_tenant_with_memory(tenant_id, vespa_instance["base_url"])
    real_delete = VespaSchemaManager.delete_tenant_schemas
    real_deploy = VespaSchemaManager._deploy_package
    barrier = threading.Barrier(2, timeout=60)
    calls = []
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))

        def interleave(manager, requested_tenant):
            calls.append(requested_tenant)
            barrier.wait()
            return real_delete(manager, requested_tenant)

        def refuse_schema_drop(manager, package, **kwargs):
            isolated_manager = copy.copy(manager)
            isolated_manager.backend_port = unavailable.getsockname()[1]
            return real_deploy(isolated_manager, package, **kwargs)

        monkeypatch.setattr(VespaSchemaManager, "delete_tenant_schemas", interleave)
        monkeypatch.setattr(VespaSchemaManager, "_deploy_package", refuse_schema_drop)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tm.app), base_url="http://tenant-test"
        ) as client:
            responses = await asyncio.gather(
                client.delete(f"/admin/organizations/{org_id}"),
                client.delete(f"/admin/organizations/{org_id}"),
            )
    assert calls == [tenant_id, tenant_id]
    assert [response.status_code for response in responses] == [503, 503]
    assert [response.json() for response in responses] == [
        {
            "detail": {
                "message": f"Organization {org_id} deletion incomplete; retry the delete",
                "org_id": org_id,
                "deleted_tenant_ids": [],
                "failed_tenant_ids": [tenant_id],
            }
        }
    ] * 2
    assert (await tm.get_organization_internal(org_id)).org_id == org_id
    assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id
    _assert_memories_readable(vespa_instance["base_url"], _schema_names(tenant_id)[0])


@pytest.mark.asyncio
async def test_organization_delete_refusal_retains_parent_until_confirmed(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    org_id = f"orgdel{uuid.uuid4().hex[:8]}"
    _seed_organization(org_id)
    refused = []

    class Handler(BaseHTTPRequestHandler):
        def do_DELETE(self):
            refused.append(self.path)
            body = json.dumps({"message": "injected delete refusal"}).encode()
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            response = requests.get(vespa_instance["base_url"] + self.path, timeout=15)
            self.send_response(response.status_code)
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        def do_POST(self):
            response = requests.post(
                vespa_instance["base_url"] + self.path,
                data=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            self.send_response(response.status_code)
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = tm.get_backend()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tm.app), base_url="http://tenant-test"
        ) as client:
            with monkeypatch.context() as patch:
                patch.setattr(
                    backend,
                    "_metadata_app",
                    Vespa(url=f"http://127.0.0.1:{server.server_port}"),
                )
                response = await client.delete(f"/admin/organizations/{org_id}")
            assert response.status_code == 502
            assert response.json() == {
                "detail": f"organization_metadata delete for {org_id} did not confirm; retry the delete"
            }
            assert refused == [
                f"/document/v1/organization_metadata/organization_metadata/docid/{org_id}"
            ]
            assert (await tm.get_organization_internal(org_id)).org_id == org_id
            retried = await client.delete(f"/admin/organizations/{org_id}")
            assert retried.status_code == 200
            assert retried.json() == {
                "status": "deleted",
                "org_id": org_id,
                "tenants_deleted": 0,
                "deleted_tenant_ids": [],
            }
            assert await tm.get_organization_internal(org_id) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _deployed_for(tenant_id: str) -> list[str]:
    """The tenant's schemas Vespa serves right now."""
    suffix = "_" + tenant_id.replace(":", "_")
    return sorted(
        name
        for name in tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
        if name.endswith(suffix)
    )


def _deleted_message(tenant_id: str) -> str:
    return (
        f"Tenant '{tenant_id}' has been deleted; its schemas and memories are "
        "not written until the tenant is created again"
    )


@pytest.mark.asyncio
async def test_a_deleted_tenants_schemas_are_never_deployed_again_until_it_is_recreated(
    wired_tenant_manager, vespa_instance, cluster_events
):
    tenant_id = _unique_tenant()
    await _create_tenant_with_memory(tenant_id, vespa_instance["base_url"])

    result = await tm.delete_tenant_internal(tenant_id)

    assert result["workers_released"] == [cluster_events.worker_id]
    assert sorted(result["deleted_schemas"]) == sorted(_schema_names(tenant_id))
    store = tm._config_manager.store
    assert tenant_is_deleted(store, tenant_id) is True
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config_manager=tm._config_manager,
        schema_loader=tm._schema_loader,
    )
    # A write's first feed builds the ingestion client, which deploys the
    # schema it feeds when missing: refused for a deleted tenant.
    with pytest.raises(TenantDeletedError) as caught:
        backend.prepare_ingestion("agent_memories")
    assert str(caught.value) == _deleted_message(tenant_id)
    assert _deployed_for(tenant_id) == []

    recreated = await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id,
            created_by="memory-orphan-test",
            base_schemas=["provenance"],
        )
    )
    try:
        assert recreated.schemas_deployed == ["provenance"]
        assert tenant_is_deleted(store, tenant_id) is False
        assert _deployed_for(tenant_id) == [_schema_names(tenant_id)[1]]
    finally:
        await tm.delete_tenant_internal(tenant_id)


class _WarmMemory:
    """Stands in for a warm tenant's Mem0 client: a deleted tenant's write is
    refused before Mem0 is reached, so any call here fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"Mem0 reached for a deleted tenant: {name}")


def _peer_worker(
    redis_url, channel, vespa_port, tenant_id, ready, release, report
) -> None:
    """Another runtime worker process: a warm manager for the tenant, one of
    its memory writes running and one queued behind it."""
    import asyncio
    import threading

    from cogniverse_agents.background_memory_writes import (
        get_background_memory_writer,
    )
    from cogniverse_core.common.tenant_utils import TenantDeletedError
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.cluster_events import ClusterEvents
    from cogniverse_vespa.config.config_store import VespaConfigStore

    async def run():
        store = VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_port
        )
        manager = Mem0MemoryManager(tenant_id)
        manager._provenance_lease_store = store
        manager.memory = _WarmMemory()
        events = ClusterEvents(
            redis_url,
            "worker-b",
            {"tenant_deleted": tenant_manager.release_deleted_tenant},
            channel=channel,
        )
        await events.start()
        writer = get_background_memory_writer()
        running = threading.Event()
        outcomes = {"queued_ran": False}

        def running_write():
            running.set()
            release.wait(300)
            try:
                manager.add_memory("written after the delete", tenant_id, "agent")
                outcomes["running"] = "written"
            except TenantDeletedError as exc:
                outcomes["running"] = f"refused: {exc}"

        def queued_write():
            outcomes["queued_ran"] = True

        writer.submit(running_write, tenant_id=tenant_id, agent_name="running")
        writer.submit(
            lambda: release.wait(300), tenant_id="other:other", agent_name="x"
        )
        writer.submit(queued_write, tenant_id=tenant_id, agent_name="queued")
        assert running.wait(30)
        ready.set()
        await asyncio.to_thread(release.wait, 300)
        outcomes["drained"] = await writer.drain(60)
        outcomes["manager_held"] = (
            Mem0MemoryManager._instances.get(tenant_id) is not None
        )
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            "vespa",
            tenant_id=tenant_id,
            config_manager=ConfigManager(store=store),
            schema_loader=FilesystemSchemaLoader("configs/schemas"),
        )
        try:
            backend.prepare_ingestion("agent_memories")
            outcomes["new_write"] = "deployed"
        except TenantDeletedError as exc:
            outcomes["new_write"] = f"refused: {exc}"
        await events.close()
        report.put(outcomes)

    asyncio.run(run())


@pytest.mark.asyncio
async def test_a_delete_on_one_process_refuses_another_processs_queued_and_new_writes(
    wired_tenant_manager, vespa_instance, cluster_events, workflow_state_redis_url
):
    """Worker B holds the tenant warm, with a memory write running and one
    queued. The delete served by this process reaches B before anything is
    dropped: the queued write never runs, the running one is refused when it
    writes, and B's first-feed deploy for the tenant is refused — nothing of
    the tenant comes back."""
    tenant_id = _unique_tenant()
    await _create_tenant_with_memory(tenant_id, vespa_instance["base_url"])
    context = multiprocessing.get_context("spawn")
    ready, release, report = context.Event(), context.Event(), context.Queue()
    peer = context.Process(
        target=_peer_worker,
        args=(
            workflow_state_redis_url,
            cluster_events.channel,
            vespa_instance["http_port"],
            tenant_id,
            ready,
            release,
            report,
        ),
    )
    peer.start()
    try:
        assert await asyncio.to_thread(ready.wait, 120) is True
        result = await tm.delete_tenant_internal(tenant_id)
        release.set()
        outcomes = await asyncio.to_thread(report.get, True, 300)
        await asyncio.to_thread(peer.join, 60)
    finally:
        release.set()
        if peer.is_alive():
            peer.kill()

    assert result["workers_released"] == sorted([cluster_events.worker_id, "worker-b"])
    assert outcomes == {
        "queued_ran": False,
        "running": f"refused: {_deleted_message(tenant_id)}",
        "drained": True,
        "manager_held": False,
        "new_write": f"refused: {_deleted_message(tenant_id)}",
    }
    assert peer.exitcode == 0
    assert _deployed_for(tenant_id) == []
    assert await tm.get_tenant_internal(tenant_id) is None


@pytest.mark.asyncio
async def test_a_delete_whose_marker_cannot_be_written_changes_nothing(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    from cogniverse_foundation.caching import TenantLRUCache, register_tenant_cache
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    tenant_id = _unique_tenant()
    await _create_tenant_with_memory(tenant_id, vespa_instance["base_url"])
    held = register_tenant_cache(TenantLRUCache(capacity=4))
    held.set(tenant_id, "gateway-agent")
    store = tm._config_manager.store

    def unavailable(*args, **kwargs):
        raise ConfigStoreUnavailableError("config store did not answer")

    with monkeypatch.context() as patch:
        patch.setattr(store, "put_immutable_config", unavailable)
        with pytest.raises(HTTPException) as caught:
            await tm.delete_tenant_internal(tenant_id)

    assert caught.value.status_code == 503
    assert caught.value.detail == (
        f"tenant {tenant_id} not deleted: deletion marker store unavailable: "
        "config store did not answer"
    )
    # Nothing was released or dropped: no worker was told.
    assert held.get(tenant_id) == "gateway-agent"
    assert tenant_is_deleted(store, tenant_id) is False
    assert _deployed_for(tenant_id) == sorted(_schema_names(tenant_id))
    assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id

    await tm.delete_tenant_internal(tenant_id)
    assert _deployed_for(tenant_id) == []


@pytest.mark.asyncio
async def test_a_delete_no_worker_could_confirm_keeps_the_tenant_and_refuses_its_writes(
    wired_tenant_manager, vespa_instance, owned_redis
):
    """With the channel down the delete reports 503, not success: the tenant
    is marked, so every process already refuses its writes, its schemas and
    record stay for the retry, and the retry completes once the channel is
    back."""
    tenant_id = _unique_tenant()
    await _create_tenant_with_memory(tenant_id, vespa_instance["base_url"])
    events = ClusterEvents(
        owned_redis["url"],
        "worker-a",
        {"tenant_deleted": tm.release_deleted_tenant},
        channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        redis_timeout_s=2,
    )
    await events.start()
    tm.set_cluster_events(events)
    subprocess.run(
        ["docker", "pause", owned_redis["name"]],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        with pytest.raises(HTTPException) as caught:
            await tm.delete_tenant_internal(tenant_id)
    finally:
        subprocess.run(
            ["docker", "unpause", owned_redis["name"]],
            check=True,
            capture_output=True,
            timeout=30,
        )

    assert caught.value.status_code == 503
    assert caught.value.detail.startswith(
        f"tenant {tenant_id} is marked deleted and its writes are refused, but "
        "not every runtime worker released it (cluster events: cannot publish "
        "'tenant_deleted': "
    )
    assert caught.value.detail.endswith("); retry the delete")
    store = tm._config_manager.store
    assert tenant_is_deleted(store, tenant_id) is True
    assert _deployed_for(tenant_id) == sorted(_schema_names(tenant_id))
    assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config_manager=tm._config_manager,
        schema_loader=tm._schema_loader,
    )
    with pytest.raises(TenantDeletedError):
        backend.schema_registry.deploy_schema(
            tenant_id=tenant_id, base_schema_name="agent_memories", force=True
        )

    try:
        result = await tm.delete_tenant_internal(tenant_id)
    finally:
        await events.close()
    assert result["workers_released"] == ["worker-a"]
    assert sorted(result["deleted_schemas"]) == sorted(_schema_names(tenant_id))
    assert _deployed_for(tenant_id) == []
    assert await tm.get_tenant_internal(tenant_id) is None


@pytest.mark.asyncio
async def test_a_delete_landing_after_a_deploy_decided_stops_it_under_the_lease(
    wired_tenant_manager, monkeypatch
):
    """A delete on another process marks the tenant after this deploy passed
    its first check: the re-check under the deploy lease, before activation,
    refuses it, and the schema never appears."""
    from cogniverse_core.common.tenant_utils import mark_tenant_deleted

    tenant_id = _unique_tenant()
    await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id,
            created_by="memory-orphan-test",
            base_schemas=["provenance"],
        )
    )
    provenance_schema = _schema_names(tenant_id)[1]
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config_manager=tm._config_manager,
        schema_loader=tm._schema_loader,
    )
    registry = backend.schema_registry
    confirm = registry.confirm_decided_revisions
    confirmed = []

    def delete_lands_first(definitions):
        mark_tenant_deleted(tm._config_manager.store, tenant_id)
        confirmed.append(sorted(definition["name"] for definition in definitions))
        return confirm(definitions)

    monkeypatch.setattr(registry, "confirm_decided_revisions", delete_lands_first)
    try:
        with pytest.raises(TenantDeletedError) as caught:
            await asyncio.to_thread(
                registry.deploy_schema,
                tenant_id=tenant_id,
                base_schema_name="agent_memories",
            )
        assert str(caught.value) == _deleted_message(tenant_id)
        assert confirmed == [[_schema_names(tenant_id)[0]]]
        assert _deployed_for(tenant_id) == [provenance_schema]
    finally:
        monkeypatch.undo()
        result = await tm.delete_tenant_internal(tenant_id)
    assert result["deleted_schemas"] == [provenance_schema]
    assert _deployed_for(tenant_id) == []
