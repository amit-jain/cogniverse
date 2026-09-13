"""Deleting a tenant takes its memory schema and the memories in it.

The memory path deploys ``agent_memories_<tenant>`` and
``provenance_<tenant>`` on first use (cogniverse_core.memory.manager
``_build_and_store_memory``), so a tenant that ever used memory owns two
schemas the delete has to drop. Every schema left behind is carried by
every later application package, and its documents stay readable under a
tenant id that no longer resolves.

The ordering under test is data-and-schema first, tenant record last:
each step that fails leaves the tenant record present, so the delete is
retryable and the residue is observable through the tenant registry.
There is no ``deleting`` state to reconcile.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import uuid

import pytest
from fastapi import HTTPException
from requests.exceptions import ConnectionError
from vespa.application import Vespa
from vespa.exceptions import VespaError

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.admin.models import CreateTenantRequest

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
def wired_tenant_manager(config_manager, schema_loader):
    """tenant_manager wired to the test Vespa, module seams restored after."""
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
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
            previous_port = manager.backend_port
            manager.backend_port = unavailable_port
            try:
                return real_deploy(manager, package, **kwargs)
            finally:
                manager.backend_port = previous_port

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
                f"http://localhost:{unavailable_port}/application/v2/tenant/default/prepareandactivate",
            )
        ]

    # Tenant record survives, so the delete is observably incomplete and
    # retryable through the same route.
    retained = await tm.get_tenant_internal(tenant_id)
    assert retained.tenant_full_id == tenant_id
    assert retained.status == "active"

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
