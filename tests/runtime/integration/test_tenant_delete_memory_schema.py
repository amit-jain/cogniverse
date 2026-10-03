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
import contextvars
import copy
import json
import logging
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
from cogniverse_runtime.task_events import INGESTION, WORKFLOW, TaskEventStore

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


def _task_event_store(redis, prefix: str) -> TaskEventStore:
    return TaskEventStore(
        redis, key_prefix=prefix, ingestion_stream_prefix=f"{prefix}:ingest:"
    )


@pytest.fixture
async def task_events(shared_state_redis):
    """The task event store tenant deletes cancel the tenant's tasks through,
    under keys of its own."""
    return _task_event_store(shared_state_redis, f"test:task-events:{uuid.uuid4().hex}")


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader, cluster_events, task_events):
    """tenant_manager wired to the test Vespa, this process's cluster-events
    channel and a task event store, module seams restored after."""
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    previous_cluster_events = tm._cluster_events
    previous_task_events = tm._task_events
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    tm.set_cluster_events(cluster_events)
    tm.set_task_event_store(task_events)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    tm.set_cluster_events(previous_cluster_events)
    tm.set_task_event_store(previous_task_events)
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


_OPERATION = contextvars.ContextVar("operation", default="tenant operation")


async def _labelled(label: str, operation):
    """Run ``operation`` with ``label`` as the operation the tenant lease
    order records for it."""
    _OPERATION.set(label)
    return await operation


def _arriving_together(
    monkeypatch, parties: int, order: list, first: str | None = None
) -> None:
    """Hold every create or delete at its tenant lease until ``parties`` of
    them have arrived, so all are in flight before any takes the tenant.
    ``order`` receives each one's label in the order they took it. With
    ``first``, the others start taking the lease only once the operation
    so labelled holds it."""
    barrier = threading.Barrier(parties, timeout=120)
    first_holds = threading.Event()
    real_lease = tm._tenant_operation_lease

    def arriving(store, tenant_id):
        lease = real_lease(store, tenant_id)
        acquire = lease.acquire

        def acquire_together():
            barrier.wait()
            label = _OPERATION.get()
            if first is not None and label != first:
                assert first_holds.wait(120), f"{first} never took the tenant"
            acquired = acquire()
            order.append(label)
            if label == first:
                first_holds.set()
            return acquired

        lease.acquire = acquire_together
        return lease

    monkeypatch.setattr(tm, "_tenant_operation_lease", arriving)


@pytest.mark.asyncio
async def test_concurrent_deletes_of_one_tenant_both_complete(
    wired_tenant_manager, vespa_instance, cluster_events, monkeypatch
):
    """Two requests deleting the same tenant are in flight together and take
    the tenant one after the other: exactly one drops the schemas, and the
    other, finding the tenant it was asked to delete already gone, reports an
    empty drop rather than a 500 or a 404 for a delete whose effect is
    already durable."""
    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)

    await _create_tenant_with_memory(tenant_id, base_url)

    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    order = []
    _arriving_together(monkeypatch, 2, order)
    calls = []
    real_delete = VespaSchemaManager.delete_tenant_schemas

    def recorded_delete(manager, requested_tenant):
        calls.append(requested_tenant)
        return real_delete(manager, requested_tenant)

    monkeypatch.setattr(VespaSchemaManager, "delete_tenant_schemas", recorded_delete)

    outcomes = await asyncio.gather(
        _labelled("delete-1", tm.delete_tenant_internal(tenant_id)),
        _labelled("delete-2", tm.delete_tenant_internal(tenant_id)),
        return_exceptions=True,
    )

    assert sorted(order) == ["delete-1", "delete-2"]
    assert calls == [tenant_id]
    assert [type(outcome) for outcome in outcomes] == [dict, dict], outcomes
    assert [outcome["status"] for outcome in outcomes] == ["deleted", "deleted"]
    dropped = sorted(
        (sorted(outcome["deleted_schemas"]) for outcome in outcomes),
        key=len,
    )
    assert dropped == [[], sorted([memory_schema, provenance_schema])]
    first, second = (outcomes[["delete-1", "delete-2"].index(label)] for label in order)
    assert first["workers_released"] == [cluster_events.worker_id]
    assert second == {
        "status": "deleted",
        "tenant_full_id": tenant_id,
        "schemas_deleted": 0,
        "deleted_schemas": [],
        "organization_deleted": False,
        "workers_released": [],
    }
    assert _tenant_rows(tm._config_manager.store, tenant_id) == []

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
    order = []
    _arriving_together(monkeypatch, 2, order)
    calls = []
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))

        def interleave(manager, requested_tenant):
            calls.append(requested_tenant)
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
    assert order == ["tenant operation", "tenant operation"]
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
    wired_tenant_manager, vespa_instance, monkeypatch, caplog
):
    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
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
    assert caught.value.detail == {
        "error": "tenant_delete_marker_unavailable",
        "message": f"Tenant {tenant_id} was not deleted: the deletion marker "
        "store did not answer; retry the delete.",
        "failure": "ConfigStoreUnavailableError",
        "tenant_id": tenant_id,
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == [
        "tenant_delete_marker_unavailable: ConfigStoreUnavailableError: "
        "config store did not answer"
    ]
    # Nothing was released or dropped: no worker was told.
    assert held.get(tenant_id) == "gateway-agent"
    assert tenant_is_deleted(store, tenant_id) is False
    assert _deployed_for(tenant_id) == sorted(_schema_names(tenant_id))
    assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id

    await tm.delete_tenant_internal(tenant_id)
    assert _deployed_for(tenant_id) == []


@pytest.mark.asyncio
async def test_a_delete_no_worker_could_confirm_keeps_the_tenant_and_refuses_its_writes(
    wired_tenant_manager, vespa_instance, owned_redis, caplog
):
    """With the channel down the delete reports 503, not success: the tenant
    is marked, so every process already refuses its writes, its schemas and
    record stay for the retry, and the retry completes once the channel is
    back."""
    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
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
    assert caught.value.detail == {
        "error": "tenant_delete_incomplete",
        "message": f"Tenant {tenant_id} is marked deleted and its writes are "
        "refused, but not every runtime worker released it; retry the delete.",
        "failure": "ClusterEventUnavailable",
        "tenant_id": tenant_id,
    }
    [logged] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ]
    assert logged.startswith(
        "tenant_delete_incomplete: ClusterEventUnavailable: cluster events: "
        "cannot publish 'tenant_deleted': "
    ), logged
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


def _counting_release(released: list):
    """The ``tenant_deleted`` handler every worker runs, recording each tenant
    this worker released."""

    def release(payload):
        result = tm.release_deleted_tenant(payload)
        released.append(payload["tenant_id"])
        return result

    return release


def _docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True, capture_output=True, timeout=30)


async def _await_subscribers(redis_url: str, channel: str, count: int) -> None:
    """Wait until ``count`` workers are subscribed to ``channel``."""
    from redis.asyncio import Redis

    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        deadline = time.monotonic() + 60
        while True:
            [(_, subscribed)] = await client.pubsub_numsub(channel)
            if subscribed == count:
                return
            assert time.monotonic() < deadline, f"{subscribed} of {count} subscribed"
            await asyncio.sleep(0.2)
    finally:
        await client.aclose()


def _provenance_status(base_url: str, tenant_id: str) -> int:
    """The HTTP status of reading the provenance document the tenant's first
    incarnation fed."""
    schema = _schema_names(tenant_id)[1]
    return (
        Vespa(url=base_url)
        .get_data(schema=schema, namespace=schema, data_id="provenance-memory-1")
        .status_code
    )


@pytest.mark.asyncio
async def test_a_create_finishes_an_incomplete_delete_and_answers_503_until_it_can(
    wired_tenant_manager, vespa_instance, owned_redis, caplog
):
    """A delete whose broadcast failed leaves the tenant marked, its record
    and schemas in place. A create of that tenant finishes the delete first.
    While the broadcast still fails, the create answers a typed 503 naming
    the incomplete delete and changes nothing. Once the channel is back, the
    create releases the tenant on every worker, drops its schemas and their
    documents, and creates the tenant afresh with no marker left."""
    from cogniverse_foundation.caching import TenantLRUCache, register_tenant_cache

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)
    await _create_tenant_with_memory(tenant_id, base_url)
    released = []
    events = ClusterEvents(
        owned_redis["url"],
        "worker-a",
        {"tenant_deleted": _counting_release(released)},
        channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        redis_timeout_s=2,
    )
    await events.start()
    tm.set_cluster_events(events)
    held = register_tenant_cache(TenantLRUCache(capacity=4))
    held.set(tenant_id, "gateway-agent")
    store = tm._config_manager.store
    request = CreateTenantRequest(
        tenant_id=tenant_id, created_by="recreate-test", base_schemas=["provenance"]
    )
    try:
        _docker("pause", owned_redis["name"])
        try:
            with pytest.raises(HTTPException) as incomplete:
                await tm.delete_tenant_internal(tenant_id)
            caplog.clear()
            with pytest.raises(HTTPException) as caught:
                await tm.create_tenant(request)
        finally:
            _docker("unpause", owned_redis["name"])

        assert (incomplete.value.status_code, incomplete.value.detail["error"]) == (
            503,
            "tenant_delete_incomplete",
        )
        assert caught.value.status_code == 503
        from cogniverse_core.common.tenant_utils import tenant_delete_pending

        assert caught.value.detail == {
            "error": "tenant_delete_incomplete",
            "message": f"Tenant {tenant_id} was not created: its earlier delete "
            "is incomplete and could not be finished, so it stays marked "
            "deleted; retry the create or the delete.",
            "failure": "ClusterEventUnavailable",
            "tenant_id": tenant_id,
        }
        [logged] = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ]
        assert logged.startswith(
            "tenant_delete_incomplete: ClusterEventUnavailable: cluster events: "
            "cannot publish 'tenant_deleted': "
        ), logged
        assert released == []
        assert held.get(tenant_id) == "gateway-agent"
        assert tenant_delete_pending(store, tenant_id) is True
        assert _deployed_for(tenant_id) == sorted([memory_schema, provenance_schema])
        retained = await tm.get_tenant_internal(tenant_id)
        assert retained.created_by == "memory-orphan-test"
        _assert_memories_readable(base_url, memory_schema)
        assert _provenance_status(base_url, tenant_id) == 200

        await _await_subscribers(owned_redis["url"], events.channel, 1)
        created = await tm.create_tenant(request)

        assert (
            created.tenant_full_id,
            created.created_by,
            created.schemas_deployed,
        ) == (tenant_id, "recreate-test", ["provenance"])
        assert released == [tenant_id]
        assert held.get(tenant_id) is None
        assert tenant_is_deleted(store, tenant_id) is False
        assert tenant_delete_pending(store, tenant_id) is False
        assert _deployed_for(tenant_id) == [provenance_schema]
        assert _provenance_status(base_url, tenant_id) == 404
        assert (await tm.get_tenant_internal(tenant_id)).created_by == "recreate-test"
    finally:
        await tm.delete_tenant_internal(tenant_id)
        await events.close()


@pytest.mark.asyncio
async def test_a_create_after_a_completed_delete_clears_the_marker_without_the_channel(
    wired_tenant_manager, owned_redis
):
    """A completed delete leaves only the marker, and a create clears it
    without reaching the workers again: it succeeds with the channel down."""
    tenant_id = _unique_tenant()
    provenance_schema = _schema_names(tenant_id)[1]
    request = CreateTenantRequest(
        tenant_id=tenant_id,
        created_by="memory-orphan-test",
        base_schemas=["provenance"],
    )
    await tm.create_tenant(request)
    released = []
    events = ClusterEvents(
        owned_redis["url"],
        "worker-a",
        {"tenant_deleted": _counting_release(released)},
        channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        redis_timeout_s=2,
    )
    await events.start()
    tm.set_cluster_events(events)
    store = tm._config_manager.store
    try:
        deleted = await tm.delete_tenant_internal(tenant_id)
        assert (deleted["workers_released"], released) == (["worker-a"], [tenant_id])
        assert tenant_is_deleted(store, tenant_id) is True

        _docker("pause", owned_redis["name"])
        try:
            recreated = await tm.create_tenant(
                CreateTenantRequest(
                    tenant_id=tenant_id,
                    created_by="recreate-test",
                    base_schemas=["provenance"],
                )
            )
        finally:
            _docker("unpause", owned_redis["name"])

        assert (recreated.tenant_full_id, recreated.created_by) == (
            tenant_id,
            "recreate-test",
        )
        assert released == [tenant_id]
        assert tenant_is_deleted(store, tenant_id) is False
        assert _deployed_for(tenant_id) == [provenance_schema]
    finally:
        await _await_subscribers(owned_redis["url"], events.channel, 1)
        await tm.delete_tenant_internal(tenant_id)
        await events.close()


def _releasing_peer(redis_url, channel, refuse, ready, done, report) -> None:
    """Another runtime worker process on the channel. While ``refuse`` is set
    it fails every ``tenant_deleted`` event; otherwise it releases the tenant
    as every worker does. Reports each tenant it released."""
    import asyncio

    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.cluster_events import ClusterEvents

    released = []

    def release(payload):
        if refuse.is_set():
            raise RuntimeError("worker-b could not release the tenant")
        result = tenant_manager.release_deleted_tenant(payload)
        released.append(payload["tenant_id"])
        return result

    async def run():
        events = ClusterEvents(
            redis_url, "worker-b", {"tenant_deleted": release}, channel=channel
        )
        await events.start()
        ready.set()
        await asyncio.to_thread(done.wait, 900)
        await events.close()
        report.put(released)

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["delete", "create-1"])
async def test_two_creates_and_a_delete_retry_of_an_incomplete_delete_leave_one_tenant(
    wired_tenant_manager, vespa_instance, workflow_state_redis_url, monkeypatch, first
):
    """A delete another worker process could not confirm leaves the tenant
    marked, its record and schemas in place. Two creates of the tenant and a
    retry of the delete then arrive together, and ``first`` takes the tenant
    first. Whichever does finishes the delete, so each worker releases the
    tenant exactly once; exactly one create makes the tenant and the other
    finds it; the retry never deletes the tenant made after it arrived; no
    marker is left."""
    from cogniverse_core.common.tenant_utils import tenant_delete_pending
    from cogniverse_runtime.admin.models import Tenant

    tenant_id = _unique_tenant()
    base_url = vespa_instance["base_url"]
    memory_schema, provenance_schema = _schema_names(tenant_id)
    await _create_tenant_with_memory(tenant_id, base_url)
    store = tm._config_manager.store
    channel = f"cogniverse:test-events:{uuid.uuid4().hex[:8]}"
    released_here = []
    events = ClusterEvents(
        workflow_state_redis_url,
        "worker-a",
        {"tenant_deleted": _counting_release(released_here)},
        channel=channel,
    )
    await events.start()
    tm.set_cluster_events(events)
    context = multiprocessing.get_context("spawn")
    refuse, ready, done = context.Event(), context.Event(), context.Event()
    report = context.Queue()
    refuse.set()
    peer = context.Process(
        target=_releasing_peer,
        args=(workflow_state_redis_url, channel, refuse, ready, done, report),
    )
    peer.start()
    try:
        assert await asyncio.to_thread(ready.wait, 120) is True
        with pytest.raises(HTTPException) as incomplete:
            await tm.delete_tenant_internal(tenant_id)
        assert (
            incomplete.value.status_code,
            incomplete.value.detail["error"],
            incomplete.value.detail["failure"],
        ) == (503, "tenant_delete_incomplete", "ClusterEventIncomplete")
        assert tenant_delete_pending(store, tenant_id) is True
        refuse.clear()
        released_here.clear()
        order = []
        _arriving_together(monkeypatch, 3, order, first=first)

        outcomes = await asyncio.gather(
            *(
                _labelled(
                    label,
                    tm.create_tenant(
                        CreateTenantRequest(
                            tenant_id=tenant_id,
                            created_by=label,
                            base_schemas=["provenance"],
                        )
                    ),
                )
                for label in ("create-1", "create-2")
            ),
            _labelled("delete", tm.delete_tenant_internal(tenant_id)),
            return_exceptions=True,
        )
        done.set()
        released_there = await asyncio.to_thread(report.get, True, 120)
        await asyncio.to_thread(peer.join, 60)

        assert sorted(order) == ["create-1", "create-2", "delete"]
        assert order[0] == first
        winner, loser = [label for label in order if label != "delete"]
        made, found = (
            outcomes[["create-1", "create-2"].index(label)] for label in (winner, loser)
        )
        assert type(made) is Tenant, made
        assert (made.tenant_full_id, made.created_by, made.schemas_deployed) == (
            tenant_id,
            winner,
            ["provenance"],
        )
        assert (type(found), found.status_code, found.detail) == (
            HTTPException,
            409,
            f"Tenant {tenant_id} already exists",
        )
        retried = outcomes[2]
        assert type(retried) is dict, retried
        if first == "delete":
            assert {
                **retried,
                "deleted_schemas": sorted(retried["deleted_schemas"]),
            } == {
                "status": "deleted",
                "tenant_full_id": tenant_id,
                "schemas_deleted": 2,
                "deleted_schemas": sorted([memory_schema, provenance_schema]),
                "organization_deleted": True,
                "workers_released": ["worker-a", "worker-b"],
            }
        else:
            assert retried == {
                "status": "deleted",
                "tenant_full_id": tenant_id,
                "schemas_deleted": 0,
                "deleted_schemas": [],
                "organization_deleted": False,
                "workers_released": [],
            }
        assert (released_here, released_there) == ([tenant_id], [tenant_id])
        assert peer.exitcode == 0
        assert tenant_is_deleted(store, tenant_id) is False
        assert tenant_delete_pending(store, tenant_id) is False
        assert (await tm.get_tenant_internal(tenant_id)).created_by == winner
        assert _deployed_for(tenant_id) == [provenance_schema]
        assert _provenance_status(base_url, tenant_id) == 404
    finally:
        done.set()
        refuse.clear()
        if peer.is_alive():
            peer.kill()
        monkeypatch.undo()
        await tm.delete_tenant_internal(tenant_id)
        await events.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "delete"])
async def test_a_tenant_operation_another_holds_answers_503_and_changes_nothing(
    wired_tenant_manager, cluster_events, monkeypatch, operation
):
    """Another create or delete of the tenant holds it for the whole wait:
    the request answers a typed 503 and changes nothing."""
    tenant_id = _unique_tenant()
    store = tm._config_manager.store
    if operation == "delete":
        await tm.create_tenant(
            CreateTenantRequest(
                tenant_id=tenant_id,
                created_by="memory-orphan-test",
                base_schemas=["provenance"],
            )
        )
    holder = tm._tenant_operation_lease(store, tenant_id)
    await asyncio.to_thread(holder.acquire)
    monkeypatch.setattr(tm, "TENANT_OPERATION_WAIT_S", 1.0)
    try:
        with pytest.raises(HTTPException) as caught:
            if operation == "create":
                await tm.create_tenant(
                    CreateTenantRequest(
                        tenant_id=tenant_id,
                        created_by="memory-orphan-test",
                        base_schemas=["provenance"],
                    )
                )
            else:
                await tm.delete_tenant_internal(tenant_id)
    finally:
        await asyncio.to_thread(holder.release)

    verb = {"create": "created", "delete": "deleted"}[operation]
    assert caught.value.status_code == 503
    assert caught.value.detail == {
        "error": "tenant_operation_in_progress",
        "message": f"Tenant {tenant_id} was not {verb}: another create or "
        "delete of it is still running; retry.",
        "failure": "LeaseWaitTimeout",
        "tenant_id": tenant_id,
    }
    assert tenant_is_deleted(store, tenant_id) is False
    if operation == "create":
        assert await tm.get_tenant_internal(tenant_id) is None
        assert _deployed_for(tenant_id) == []
    else:
        assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id
        assert _deployed_for(tenant_id) == [_schema_names(tenant_id)[1]]
        await tm.delete_tenant_internal(tenant_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "delete"])
async def test_a_tenant_operation_whose_lease_cannot_be_written_changes_nothing(
    wired_tenant_manager, vespa_instance, config_manager, operation, caplog
):
    """The config store refuses the write that takes the tenant for this
    create or delete: the request answers a typed 503 naming the failure's
    type, logs the cause, and changes nothing."""
    from urllib.parse import unquote

    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from tests.utils.http_fault_proxy import InterceptFaultProxy

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    tenant_id = _unique_tenant()
    provenance_schema = _schema_names(tenant_id)[1]
    request = CreateTenantRequest(
        tenant_id=tenant_id,
        created_by="memory-orphan-test",
        base_schemas=["provenance"],
    )
    if operation == "delete":
        await tm.create_tenant(request)

    def refuse_lease_writes(method, path, _body):
        if method != "GET" and "tenant_operation_lease" in unquote(path):
            return 500, {"message": "injected storage failure"}
        return None

    with InterceptFaultProxy(vespa_instance["base_url"], refuse_lease_writes) as proxy:
        proxied = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=proxy.port
            )
        )
        # A backend is bound to the store it was built with: drop the ones
        # built over the real store, so the request reads through the proxy.
        BackendRegistry.get_instance().clear_instances()
        tm.set_config_manager(proxied)
        try:
            with pytest.raises(HTTPException) as caught:
                if operation == "create":
                    await tm.create_tenant(request)
                else:
                    await tm.delete_tenant_internal(tenant_id)
        finally:
            BackendRegistry.get_instance().clear_instances()
            tm.set_config_manager(config_manager)
            proxied.store.close()

    verb = {"create": "created", "delete": "deleted"}[operation]
    assert caught.value.status_code == 503
    assert caught.value.detail == {
        "error": "tenant_operation_unavailable",
        "message": f"Tenant {tenant_id} was not {verb}: the config store did not "
        "answer; retry.",
        "failure": "VespaError",
        "tenant_id": tenant_id,
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == ["tenant_operation_unavailable: VespaError: injected storage failure"]
    assert tenant_is_deleted(config_manager.store, tenant_id) is False
    if operation == "create":
        assert await tm.get_tenant_internal(tenant_id) is None
        assert _deployed_for(tenant_id) == []
    else:
        assert (await tm.get_tenant_internal(tenant_id)).tenant_full_id == tenant_id
        assert _deployed_for(tenant_id) == [provenance_schema]
        await tm.delete_tenant_internal(tenant_id)


def _tenant_rows(store, tenant_id: str) -> list[str]:
    """Every config-store row the tenant's delete removes, as service:key:
    its own rows, its schema deployment intents and its provenance write
    lease."""
    from cogniverse_core.memory.manager import PROVENANCE_WRITE_LEASE_SERVICE
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    rows = [
        f"{entry.service}:{entry.config_key}" for entry in store.list_configs(tenant_id)
    ]
    rows += [
        f"schema_deployment_intents:{entry.config_key}"
        for entry in store.list_all_configs(
            scope=ConfigScope.SCHEMA,
            service="schema_deployment_intents",
            config_key_suffix="_" + tenant_id.replace(":", "_"),
        )
        if entry.config_value["registration"]["tenant_id"] == tenant_id
    ]
    if store.get_config(
        "__system__", ConfigScope.SCHEMA, PROVENANCE_WRITE_LEASE_SERVICE, tenant_id
    ):
        rows.append(f"{PROVENANCE_WRITE_LEASE_SERVICE}:{tenant_id}")
    return sorted(rows)


async def _create_tenant_with_state(tenant_id: str, base_url: str) -> list[str]:
    """A tenant with memory schemas and documents, a backend profile, a pin
    quota and a provenance write lease that was taken and released; returns
    its rows."""
    from cogniverse_core.memory.manager import PROVENANCE_WRITE_LEASE_SERVICE
    from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
    from cogniverse_foundation.config.unified_config import BackendProfileConfig
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    await _create_tenant_with_memory(tenant_id, base_url)
    store = tm._config_manager.store
    tm._config_manager.add_backend_profile(
        BackendProfileConfig(
            profile_name="state_profile",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
        ),
        tenant_id=tenant_id,
    )
    store.set_config(
        tenant_id, ConfigScope.SYSTEM, "admin_overrides", "pin_quotas", {"quota": 3}
    )
    lease = SchemaDeployLease(
        store,
        service=PROVENANCE_WRITE_LEASE_SERVICE,
        config_key=tenant_id,
        purpose=f"provenance writes for {tenant_id}",
    )
    await asyncio.to_thread(lease.acquire)
    await asyncio.to_thread(lease.release)
    return _tenant_rows(store, tenant_id)


def _expected_rows(tenant_id: str) -> list[str]:
    memory_schema, provenance_schema = _schema_names(tenant_id)
    return sorted(
        [
            "admin_overrides:pin_quotas",
            "backend:backend_config",
            "provenance_write_lease:" + tenant_id,
            "schema_deployment_intents:" + memory_schema,
            "schema_deployment_intents:" + provenance_schema,
            "schema_registry:schema_agent_memories",
            "schema_registry:schema_provenance",
        ]
    )


@pytest.mark.asyncio
async def test_a_delete_removes_every_row_of_the_tenant_but_its_marker(
    wired_tenant_manager, vespa_instance, cluster_events
):
    """Registry tombstones, deployment intents, the backend profile, a pin
    quota and the provenance write lease all go with the tenant. Its
    deletion marker stays, the delete recorded complete, until a create."""
    from cogniverse_core.common.tenant_utils import tenant_delete_pending

    tenant_id = _unique_tenant()
    store = tm._config_manager.store
    assert await _create_tenant_with_state(
        tenant_id, vespa_instance["base_url"]
    ) == _expected_rows(tenant_id)

    result = await tm.delete_tenant_internal(tenant_id)

    assert result["workers_released"] == [cluster_events.worker_id]
    assert sorted(result["deleted_schemas"]) == sorted(_schema_names(tenant_id))
    assert _tenant_rows(store, tenant_id) == []
    assert (
        tenant_is_deleted(store, tenant_id),
        tenant_delete_pending(store, tenant_id),
    ) == (True, False)

    recreated = await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id, created_by="recreate-test", base_schemas=["provenance"]
        )
    )
    try:
        assert recreated.schemas_deployed == ["provenance"]
        provenance_schema = _schema_names(tenant_id)[1]
        assert _tenant_rows(store, tenant_id) == sorted(
            [
                "schema_deployment_intents:" + provenance_schema,
                "schema_registry:schema_provenance",
            ]
        )
    finally:
        await tm.delete_tenant_internal(tenant_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["retry", "create"])
async def test_tenant_rows_the_delete_cannot_remove_are_removed_by_its_retry_or_a_create(
    wired_tenant_manager, vespa_instance, config_manager, cluster_events, caplog, finish
):
    """The config store refuses to delete the tenant's own rows. The delete
    still answers that the tenant is deleted, logs each row it could not
    remove at ERROR by tenant, and stays pending; its retry, or a create of
    the tenant, removes them and completes it."""
    from urllib.parse import unquote

    from cogniverse_core.common.tenant_utils import tenant_delete_pending
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from tests.utils.http_fault_proxy import InterceptFaultProxy

    tenant_id = _unique_tenant()
    store = config_manager.store
    rows = await _create_tenant_with_state(tenant_id, vespa_instance["base_url"])
    own = [
        row for row in rows if not row.startswith(("schema_deployment", "provenance"))
    ]
    caplog.set_level(logging.ERROR, logger=tm.logger.name)

    def refuse_own_row_deletes(method, path, _body):
        if method == "DELETE" and f"::{tenant_id}:" in unquote(path):
            return 500, {"message": "injected storage failure"}
        return None

    with InterceptFaultProxy(
        vespa_instance["base_url"], refuse_own_row_deletes
    ) as proxy:
        proxied = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=proxy.port
            )
        )
        BackendRegistry.get_instance().clear_instances()
        tm.set_config_manager(proxied)
        try:
            result = await tm.delete_tenant_internal(tenant_id)
        finally:
            BackendRegistry.get_instance().clear_instances()
            tm.set_config_manager(config_manager)
            proxied.store.close()

    assert result["status"] == "deleted"
    assert result["workers_released"] == [cluster_events.worker_id]
    assert sorted(result["deleted_schemas"]) == sorted(_schema_names(tenant_id))
    logged = sorted(
        record.getMessage()
        for record in caplog.records
        if record.name == tm.logger.name and record.levelno == logging.ERROR
    )
    assert [message.split(" (", 1)[0] for message in logged] == [
        f"Cannot delete {row} of deleted tenant {tenant_id}" for row in sorted(own)
    ]
    assert all(
        message.endswith(
            "; the delete stays pending and its retry, or the next create of the "
            "tenant, deletes it"
        )
        for message in logged
    ), logged
    assert _tenant_rows(store, tenant_id) == sorted(own)
    assert tenant_delete_pending(store, tenant_id) is True
    assert await tm.get_tenant_internal(tenant_id) is None

    caplog.clear()
    if finish == "retry":
        retried = await tm.delete_tenant_internal(tenant_id)
        assert retried == {
            "status": "deleted",
            "tenant_full_id": tenant_id,
            "schemas_deleted": 0,
            "deleted_schemas": [],
            "organization_deleted": False,
            "workers_released": [cluster_events.worker_id],
        }
        assert _tenant_rows(store, tenant_id) == []
        assert (
            tenant_is_deleted(store, tenant_id),
            tenant_delete_pending(store, tenant_id),
        ) == (True, False)
    else:
        created = await tm.create_tenant(
            CreateTenantRequest(
                tenant_id=tenant_id,
                created_by="recreate-test",
                base_schemas=["provenance"],
            )
        )
        try:
            assert created.created_by == "recreate-test"
            provenance_schema = _schema_names(tenant_id)[1]
            assert _tenant_rows(store, tenant_id) == sorted(
                [
                    "schema_deployment_intents:" + provenance_schema,
                    "schema_registry:schema_provenance",
                ]
            )
            assert tenant_is_deleted(store, tenant_id) is False
        finally:
            await tm.delete_tenant_internal(tenant_id)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == tm.logger.name and record.levelno == logging.ERROR
    ] == []


async def _create_tenant_with_tasks(tenant_id: str, store: TaskEventStore):
    """A tenant with a workflow running on this process, an ingestion job
    queued for it, and a workflow that already ended; returns the running
    workflow's queue."""
    from cogniverse_core.events import create_complete_event

    await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id, created_by="task-test", base_schemas=["provenance"]
        )
    )
    running = await store.open_task(WORKFLOW, f"wf-{tenant_id}", tenant_id)
    await store.register_queued(f"job-{tenant_id}", tenant_id)
    ended = await store.open_task(WORKFLOW, f"wf-ended-{tenant_id}", tenant_id)
    await ended.finish(
        create_complete_event(f"wf-ended-{tenant_id}", tenant_id, result={})
    )
    return running


async def _cancellations(store: TaskEventStore, tenant_id: str) -> dict:
    """Each of ``_create_tenant_with_tasks``'s tasks: cancelled, and why."""
    reads = {
        task: await store.read(task, count=0)
        for task in (
            f"wf-{tenant_id}",
            f"job-{tenant_id}",
            f"wf-ended-{tenant_id}",
        )
    }
    return {task: (read.cancelled, read.cancel_reason) for task, read in reads.items()}


@pytest.mark.asyncio
async def test_a_delete_cancels_the_tenants_running_and_queued_tasks(
    wired_tenant_manager, task_events, caplog
):
    """The tenant's running workflow and its queued ingestion job are
    cancelled, each with the delete as the reason, and the running one learns
    it on its process's next poll; its ended workflow and another tenant's
    task are left as they are. Nothing of the tenant stays listed active once
    its workflow ends."""
    from cogniverse_core.events import create_complete_event

    tenant_id = _unique_tenant()
    peer = _unique_tenant()
    caplog.set_level(logging.INFO, logger=tm.logger.name)
    running = await _create_tenant_with_tasks(tenant_id, task_events)
    other = await task_events.open_task(WORKFLOW, f"wf-{peer}", peer)
    # This process's poller keeps its tasks leased while the delete runs.
    task_events.start()
    try:
        result = await tm.delete_tenant_internal(tenant_id)
        await task_events.poll_once()
    finally:
        await task_events.close()

    reason = f"tenant {tenant_id} was deleted"
    assert result["status"] == "deleted"
    assert await _cancellations(task_events, tenant_id) == {
        f"wf-{tenant_id}": (True, reason),
        f"job-{tenant_id}": (True, reason),
        f"wf-ended-{tenant_id}": (False, None),
    }
    assert (
        running.cancellation_token.is_cancelled,
        running.cancellation_token.reason,
    ) == (
        True,
        reason,
    )
    assert other.cancellation_token.is_cancelled is False
    assert (await task_events.read(f"wf-{peer}", count=0)).cancelled is False
    assert (
        f"Cancelled the tasks ['job-{tenant_id}', 'wf-{tenant_id}'] of deleted "
        f"tenant {tenant_id}"
    ) in [record.getMessage() for record in caplog.records]
    assert sorted(
        (row["task_id"], row["kind"], row["is_cancelled"])
        for row in await task_events.list_active(tenant_id)
    ) == [(f"job-{tenant_id}", INGESTION, True), (f"wf-{tenant_id}", WORKFLOW, True)]
    await running.finish(create_complete_event(f"wf-{tenant_id}", tenant_id, result={}))
    assert [row["task_id"] for row in await task_events.list_active(tenant_id)] == [
        f"job-{tenant_id}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["retry", "create"])
async def test_tasks_the_delete_cannot_cancel_are_cancelled_by_its_retry_or_a_create(
    wired_tenant_manager, cluster_events, own_redis, caplog, finish
):
    """The task event store does not answer. The delete still drops the
    tenant and answers that it is deleted, logs at ERROR that its tasks were
    not cancelled, and stays pending; once the store answers, its retry, or a
    create of the tenant, cancels them and completes it."""
    from cogniverse_core.common.tenant_utils import tenant_delete_pending
    from cogniverse_runtime.shared_state import connect_shared_state_redis

    url, pause, resume = own_redis
    redis = await connect_shared_state_redis(url, timeout_seconds=1.0)
    store = _task_event_store(redis, f"test:task-events:{uuid.uuid4().hex}")
    tm.set_task_event_store(store)
    config_store = tm._config_manager.store
    tenant_id = _unique_tenant()
    caplog.set_level(logging.ERROR, logger=tm.logger.name)
    try:
        await _create_tenant_with_tasks(tenant_id, store)
        store.start()
        pause()
        try:
            result = await tm.delete_tenant_internal(tenant_id)
        finally:
            resume()
        logged = [
            record.getMessage()
            for record in caplog.records
            if record.name == tm.logger.name and record.levelno == logging.ERROR
        ]
        uncancelled = await _cancellations(store, tenant_id)
        pending = tenant_delete_pending(config_store, tenant_id)
        caplog.clear()
        if finish == "retry":
            finished = await tm.delete_tenant_internal(tenant_id)
            assert finished == {
                "status": "deleted",
                "tenant_full_id": tenant_id,
                "schemas_deleted": 0,
                "deleted_schemas": [],
                "organization_deleted": False,
                "workers_released": [cluster_events.worker_id],
            }
            assert (
                tenant_is_deleted(config_store, tenant_id),
                tenant_delete_pending(config_store, tenant_id),
            ) == (True, False)
        else:
            created = await tm.create_tenant(
                CreateTenantRequest(
                    tenant_id=tenant_id,
                    created_by="recreate-test",
                    base_schemas=["provenance"],
                )
            )
            assert created.created_by == "recreate-test"
            assert tenant_is_deleted(config_store, tenant_id) is False
        cancelled = await _cancellations(store, tenant_id)
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == tm.logger.name and record.levelno == logging.ERROR
        ] == []
    finally:
        if finish == "create":
            await tm.delete_tenant_internal(tenant_id)
        await store.close()
        await redis.aclose()

    assert (result["status"], result["deleted_schemas"]) == (
        "deleted",
        [_schema_names(tenant_id)[1]],
    )
    assert [message.split(" (", 1)[0] for message in logged] == [
        f"Cannot cancel the tasks of deleted tenant {tenant_id}"
    ]
    assert logged[0].endswith(
        "; the delete stays pending and its retry, or the next create of the "
        "tenant, cancels them"
    ), logged
    assert pending is True
    assert uncancelled == {
        f"wf-{tenant_id}": (False, None),
        f"job-{tenant_id}": (False, None),
        f"wf-ended-{tenant_id}": (False, None),
    }
    reason = f"tenant {tenant_id} was deleted"
    assert cancelled == {
        f"wf-{tenant_id}": (True, reason),
        f"job-{tenant_id}": (True, reason),
        f"wf-ended-{tenant_id}": (False, None),
    }


def _deleting_peer(
    redis_url, channel, task_prefix, vespa_port, tenant_id, report
) -> None:
    """Another runtime process serving a delete of the tenant."""
    import asyncio

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.cluster_events import ClusterEvents
    from cogniverse_runtime.shared_state import connect_shared_state_redis
    from cogniverse_vespa.config.config_store import VespaConfigStore

    async def run():
        store = VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_port
        )
        tenant_manager.set_config_manager(ConfigManager(store=store))
        tenant_manager.set_schema_loader(FilesystemSchemaLoader("configs/schemas"))
        events = ClusterEvents(
            redis_url,
            "worker-peer",
            {"tenant_deleted": tenant_manager.release_deleted_tenant},
            channel=channel,
        )
        await events.start()
        tenant_manager.set_cluster_events(events)
        redis = await connect_shared_state_redis(redis_url)
        tenant_manager.set_task_event_store(_task_event_store(redis, task_prefix))
        try:
            report.put(await tenant_manager.delete_tenant_internal(tenant_id))
        finally:
            await events.close()
            await redis.aclose()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_a_deploy_refused_by_a_delete_that_landed_after_its_decision_leaves_no_row(
    wired_tenant_manager,
    vespa_instance,
    cluster_events,
    task_events,
    workflow_state_redis_url,
):
    """A deploy of a new schema journals its intent; then another process
    serves the tenant's whole delete before the deploy takes the deployment
    lease. The delete leaves the deploy's pending schema out of its own
    package; under the lease the deploy is refused, and its retire does not
    write back the journal entry the delete removed, so once both finish
    nothing of the tenant is left but its marker."""
    from cogniverse_core.common.tenant_utils import TenantDeletedError

    tenant_id = _unique_tenant()
    store = tm._config_manager.store
    await tm.create_tenant(
        CreateTenantRequest(
            tenant_id=tenant_id,
            created_by="memory-orphan-test",
            base_schemas=["provenance"],
        )
    )
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config_manager=tm._config_manager,
        schema_loader=tm._schema_loader,
    )
    # The backend the registry activates through, which may be another
    # tenant's instance sharing the registry.
    activating = backend.schema_registry._backend
    context = multiprocessing.get_context("spawn")
    report = context.Queue()
    deleted = []
    deploy = activating.deploy_schemas

    def delete_lands_first(schema_definitions, *args, **kwargs):
        journaled = _tenant_rows(store, tenant_id)
        peer = context.Process(
            target=_deleting_peer,
            args=(
                workflow_state_redis_url,
                cluster_events.channel,
                task_events._prefix,
                vespa_instance["http_port"],
                tenant_id,
                report,
            ),
        )
        peer.start()
        try:
            deleted.append((journaled, report.get(True, 600)))
            peer.join(60)
        finally:
            if peer.is_alive():
                peer.kill()
        deleted.append(peer.exitcode)
        return deploy(schema_definitions, *args, **kwargs)

    activating.deploy_schemas = delete_lands_first
    try:
        with pytest.raises(TenantDeletedError) as caught:
            await asyncio.to_thread(
                backend.schema_registry.deploy_schema,
                tenant_id=tenant_id,
                base_schema_name="agent_memories",
            )
    finally:
        activating.deploy_schemas = deploy
    [(journaled, result), exitcode] = deleted
    memory_schema, provenance_schema = _schema_names(tenant_id)
    assert exitcode == 0
    assert journaled == sorted(
        [
            "schema_deployment_intents:" + memory_schema,
            "schema_deployment_intents:" + provenance_schema,
            "schema_registry:schema_provenance",
        ]
    )
    assert result["deleted_schemas"] == [provenance_schema]
    assert result["workers_released"] == sorted(
        ["worker-peer", cluster_events.worker_id]
    )
    assert str(caught.value) == _deleted_message(tenant_id)
    assert _tenant_rows(store, tenant_id) == []
    assert _deployed_for(tenant_id) == []
