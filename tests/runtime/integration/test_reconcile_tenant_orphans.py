"""A tenant record can go while its schemas stay registered and deployed.

Schema auto-deploy paths create a schema without ever creating a tenant
(memory lazy-init, ingestion upload), and a tenant record removed outside
``DELETE /admin/tenants`` leaves its registry rows behind. Either way the
schema is deployed AND registered, so the registry diff that
``orphan_schemas`` is built from cannot see it: it survives every
application package and costs a schema slot in every deploy, while its
documents stay readable under a tenant id that no longer resolves.

``tenant_orphan_schemas`` is that second class, and the removal path drops
them through ``delete_tenant_schemas_bulk`` (one redeploy) with a readback.

The removal here is driven with this module's own tenant selection rather
than the route's global one: the shared Vespa carries schemas other test
modules own, and a global sweep would drop them.
"""

from __future__ import annotations

import copy
import socket
import uuid

import pytest
from fastapi import HTTPException
from requests.exceptions import ConnectionError
from vespa.application import Vespa

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.admin.models import CreateTenantRequest

pytestmark = pytest.mark.integration

MEMORY_BASE_SCHEMA = "agent_memories"
EMBEDDING_DIMS = 768

MEMORIES = (
    {"id": "orphan-mem-1", "text": "Reconciliation refuses an empty selection."},
    {"id": "orphan-mem-2", "text": "A readback follows every removal."},
)


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader):
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    BackendRegistry.get_instance().clear_instances()


def _schema_name(tenant_id: str) -> str:
    return f"{MEMORY_BASE_SCHEMA}_{tenant_id.replace(':', '_')}"


def _feed_memories(base_url: str, schema: str) -> None:
    statuses: list[tuple[str, int]] = []
    Vespa(url=base_url).feed_iterable(
        [
            {
                "id": memory["id"],
                "fields": {
                    "id": memory["id"],
                    "text": memory["text"],
                    "agent_id": "search_agent",
                    "session_id": "sess",
                    "user_id": schema,
                    "subject_key": "operator",
                    "metadata_": "{}",
                    "created_at": 1757000000,
                    "embedding": [0.0125] * EMBEDDING_DIMS,
                },
            }
            for memory in MEMORIES
        ],
        schema=schema,
        namespace=schema,
        callback=lambda response, doc_id: statuses.append(
            (doc_id, response.status_code)
        ),
    )
    assert sorted(statuses) == [("orphan-mem-1", 200), ("orphan-mem-2", 200)]


def _assert_memories_readable(base_url: str, schema: str) -> None:
    app = Vespa(url=base_url)
    for memory in MEMORIES:
        stored = app.get_data(schema=schema, namespace=schema, data_id=memory["id"])
        assert stored.status_code == 200
        assert stored.json["fields"]["text"] == memory["text"]


async def _seed_three_tenants(base_url: str) -> tuple[str, str, str]:
    """Create three tenants with a memory schema each, then remove the third's
    tenant record while leaving its schema and registry row in place."""
    prefix = f"orph{uuid.uuid4().hex[:8]}"
    live_a, live_b, ghost = (
        f"{prefix}a:t1",
        f"{prefix}b:t1",
        f"{prefix}c:t1",
    )
    for tenant_id in (live_a, live_b, ghost):
        created = await tm.create_tenant(
            CreateTenantRequest(
                tenant_id=tenant_id,
                created_by="tenant-orphan-test",
                base_schemas=[MEMORY_BASE_SCHEMA],
            )
        )
        assert created.tenant_full_id == tenant_id

    _feed_memories(base_url, _schema_name(ghost))
    _assert_memories_readable(base_url, _schema_name(ghost))

    with tm.metadata_backend() as backend:
        removed = backend.delete_metadata_document(
            schema="tenant_metadata", doc_id=ghost
        )
    assert removed is True
    assert await tm.get_tenant_internal(ghost) is None
    assert (await tm.get_tenant_internal(live_a)).tenant_full_id == live_a
    assert (await tm.get_tenant_internal(live_b)).tenant_full_id == live_b

    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert {_schema_name(t) for t in (live_a, live_b, ghost)} <= deployed
    return live_a, live_b, ghost


@pytest.mark.asyncio
async def test_deleted_tenant_record_leaves_an_orphan_the_registry_diff_misses(
    wired_tenant_manager, vespa_instance
):
    live_a, live_b, ghost = await _seed_three_tenants(vespa_instance["base_url"])
    mine = {_schema_name(t) for t in (live_a, live_b, ghost)}

    report = await tm.reconcile_orphans(dry_run=True, include_document_counts=True)

    assert report["dry_run"] is True
    assert sorted(set(report["tenant_orphan_schemas"]) & mine) == [_schema_name(ghost)]
    assert ghost in report["tenant_orphan_tenants"]
    assert live_a not in report["tenant_orphan_tenants"]
    assert live_b not in report["tenant_orphan_tenants"]
    assert report["tenant_orphans_deleted"] == []
    assert [row for row in report["orphan_details"] if row["schema"] in mine] == [
        {
            "schema": _schema_name(ghost),
            "tenant": ghost,
            "tenant_exists": False,
            "document_count": len(MEMORIES),
        }
    ]
    # The defect: the schema is registered, so the registry diff is blind
    # to it and the operator's clean-cluster report says nothing.
    assert set(report["orphan_schemas"]) & mine == set()
    assert sorted(t for t in report["orphan_tenants"] if t.startswith(ghost[:4])) == []

    # A dry run changes nothing.
    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert mine <= deployed
    _assert_memories_readable(vespa_instance["base_url"], _schema_name(ghost))


@pytest.mark.asyncio
async def test_removal_drops_only_the_orphan_and_reads_the_live_set_back(
    wired_tenant_manager, vespa_instance
):
    live_a, live_b, ghost = await _seed_three_tenants(vespa_instance["base_url"])
    mine = {_schema_name(t) for t in (live_a, live_b, ghost)}

    dropped = tm._remove_tenant_orphans([ghost], [_schema_name(ghost)])

    assert dropped == [_schema_name(ghost)]

    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert sorted(deployed & mine) == sorted(
        [_schema_name(live_a), _schema_name(live_b)]
    )

    # The registry row went with the schema, so the next deploy cannot
    # re-contribute it.
    registry = tm.get_backend().schema_manager._schema_registry
    assert registry.get_tenant_schemas(ghost) == []
    assert [info.base_schema_name for info in registry.get_tenant_schemas(live_a)] == [
        MEMORY_BASE_SCHEMA
    ]
    assert [info.base_schema_name for info in registry.get_tenant_schemas(live_b)] == [
        MEMORY_BASE_SCHEMA
    ]

    # The live tenants' schemas still answer queries after the redeploy.
    app = Vespa(url=vespa_instance["base_url"])
    for tenant_id in (live_a, live_b):
        answered = app.query(
            body={"yql": f"select * from {_schema_name(tenant_id)} where true"}
        )
        assert answered.status_code == 200
        assert answered.json["root"]["fields"]["totalCount"] == 0


def test_empty_selection_is_refused(wired_tenant_manager):
    with pytest.raises(HTTPException) as exc:
        tm._remove_tenant_orphans([], ["agent_memories_whatever_t1"])
    assert exc.value.status_code == 409
    assert exc.value.detail == (
        "No tenant-orphan schemas to remove: every schema registered in Vespa "
        "belongs to a tenant that still has a tenant_metadata record. Refusing "
        "an empty selection."
    )


@pytest.mark.asyncio
async def test_failed_removal_leaves_schema_documents_and_registry_row_intact(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    """A redeploy that raises must leave the orphan exactly as it was -- no
    half-dropped schema, no tombstoned registry row whose schema still
    exists -- and the retry must complete."""
    live_a, live_b, ghost = await _seed_three_tenants(vespa_instance["base_url"])
    mine = {_schema_name(t) for t in (live_a, live_b, ghost)}

    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    real_deploy = VespaSchemaManager._deploy_package
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        port = unavailable.getsockname()[1]

        def deploy_to_unavailable_port(manager, package, **kwargs):
            isolated_manager = copy.copy(manager)
            isolated_manager.backend_port = port
            return real_deploy(isolated_manager, package, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(
                VespaSchemaManager, "_deploy_package", deploy_to_unavailable_port
            )
            with pytest.raises(ConnectionError) as failure:
                tm._remove_tenant_orphans([ghost], [_schema_name(ghost)])
        assert failure.value.request.method == "POST"
        assert failure.value.request.url == (
            f"http://localhost:{port}/application/v2/tenant/default/session"
        )

    deployed = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert mine <= deployed
    _assert_memories_readable(vespa_instance["base_url"], _schema_name(ghost))
    registry = tm.get_backend().schema_manager._schema_registry
    assert [info.base_schema_name for info in registry.get_tenant_schemas(ghost)] == [
        MEMORY_BASE_SCHEMA
    ]

    assert tm._remove_tenant_orphans([ghost], [_schema_name(ghost)]) == [
        _schema_name(ghost)
    ]
    deployed_after = set(
        tm.get_backend().schema_manager.list_deployed_document_types(
            raise_on_failure=True
        )
    )
    assert sorted(deployed_after & mine) == sorted(
        [_schema_name(live_a), _schema_name(live_b)]
    )
    assert registry.get_tenant_schemas(ghost) == []


@pytest.mark.asyncio
async def test_reconciliation_refuses_an_unreachable_config_server(
    wired_tenant_manager, monkeypatch
):
    manager = tm.get_backend().schema_manager
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        port = unavailable.getsockname()[1]
        monkeypatch.setattr(manager, "backend_port", port)
        with pytest.raises(HTTPException) as failure:
            await tm.reconcile_orphans(dry_run=True)
    assert failure.value.status_code == 503
    assert failure.value.detail.startswith(
        "Cannot enumerate deployed schemas during reconciliation: "
    )
    assert failure.value.__cause__.request.method == "GET"
    assert failure.value.__cause__.request.url == (
        f"http://localhost:{port}/application/v2/tenant/default/application/default/"
        "environment/prod/region/default/instance/default/content/schemas/"
    )


@pytest.mark.asyncio
async def test_cli_reports_the_registry_owner_and_document_count_over_http(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    import asyncio
    import io
    import threading
    import time

    import uvicorn
    from cogniverse_cli import admin
    from rich.console import Console

    _, _, ghost = await _seed_three_tenants(vespa_instance["base_url"])
    output = io.StringIO()
    monkeypatch.setattr(admin, "console", Console(file=output, width=200))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(tm.app, log_level="error"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if time.monotonic() >= deadline:
                    raise AssertionError("Tenant HTTP server did not start")
                await asyncio.sleep(0.01)
            result = await asyncio.to_thread(
                admin.cmd_reconcile_orphans,
                f"http://127.0.0.1:{listener.getsockname()[1]}",
                confirm=False,
            )
            assert result == 0
            rows = [
                [cell.strip() for cell in line.split("│")[1:-1]]
                for line in output.getvalue().splitlines()
                if _schema_name(ghost) in line
            ]
            assert rows == [[_schema_name(ghost), ghost, "False", "2"]]
            _assert_memories_readable(vespa_instance["base_url"], _schema_name(ghost))
        finally:
            server.should_exit = True
            worker.join(10)
            assert worker.is_alive() is False


@pytest.mark.asyncio
async def test_concurrent_dry_runs_preserve_the_same_orphan_documents(
    wired_tenant_manager, vespa_instance, monkeypatch
):
    import asyncio
    import threading

    live_a, live_b, ghost = await _seed_three_tenants(vespa_instance["base_url"])
    mine = {_schema_name(t) for t in (live_a, live_b, ghost)}
    backend = tm.get_backend()
    query = backend.query_metadata_documents
    readers = []
    barrier = threading.Barrier(2, timeout=60)

    def concurrent_query(*args, **kwargs):
        if kwargs["schema"] == "tenant_metadata":
            readers.append(threading.get_ident())
            barrier.wait()
        return query(*args, **kwargs)

    monkeypatch.setattr(backend, "query_metadata_documents", concurrent_query)
    reports = await asyncio.gather(
        tm.reconcile_orphans(dry_run=True, include_document_counts=True),
        tm.reconcile_orphans(dry_run=True, include_document_counts=True),
    )
    assert len(readers) == 2
    assert len(set(readers)) == 2
    assert [
        [row for row in report["orphan_details"] if row["schema"] in mine]
        for report in reports
    ] == [
        [
            {
                "schema": _schema_name(ghost),
                "tenant": ghost,
                "tenant_exists": False,
                "document_count": 2,
            }
        ]
    ] * 2
    assert [report["tenant_orphans_deleted"] for report in reports] == [[], []]
    _assert_memories_readable(vespa_instance["base_url"], _schema_name(ghost))
