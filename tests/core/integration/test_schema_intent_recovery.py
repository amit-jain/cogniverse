"""Registration recovery at the real Vespa activation and storage boundaries."""

import ast
import asyncio
import copy
import multiprocessing
import re
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendConfig
from cogniverse_runtime.admin import tenant_manager
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


class ProcessDeath(BaseException):
    """Bypass every in-process Exception compensation handler."""


def _connect(port, config_port, store=None):
    if store is None:
        store = VespaConfigStore(backend_url="http://127.0.0.1", backend_port=port)
    manager = ConfigManager(store=store)
    loader = FilesystemSchemaLoader(Path("configs/schemas"))
    backend = VespaBackend(
        BackendConfig(
            backend_type="vespa", url="http://127.0.0.1", port=port, tenant_id="system"
        ),
        schema_loader=loader,
        config_manager=manager,
    )
    backend._initialize_backend({"config_port": config_port})
    backend.schema_registry = SchemaRegistry(manager, backend, loader)
    backend.schema_manager._schema_registry = backend.schema_registry
    return backend


@pytest.fixture
def recovery_backend(seeded_config_vespa):
    port = seeded_config_vespa["http_port"]
    config_port = seeded_config_vespa["config_port"]
    store = VespaConfigStore(backend_url="http://127.0.0.1", backend_port=port)
    backends = []

    def connect():
        backend = _connect(port, config_port, store)
        backends.append(backend)
        return backend

    yield connect, store
    for backend in backends:
        backend.close()
    store.close()


def _entry(store, tenant, service="schema_registry"):
    return store.get_config(
        tenant_id=SYSTEM_TENANT_ID
        if service == "schema_deployment_intents"
        else tenant,
        scope=ConfigScope.SCHEMA,
        service=service,
        config_key=f"wiki_pages_{tenant.replace(':', '_')}"
        if service == "schema_deployment_intents"
        else "schema_wiki_pages",
    )


def test_killed_create_recovers_exact_registration_automatically(
    recovery_backend,
    monkeypatch,
):
    import cogniverse_core.registries.schema_registry as registry_module

    connect, store = recovery_backend
    owner = connect()
    tenant = f"orphan_{uuid4().hex[:12]}:crash"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    expected = {}
    deployed_definitions = []
    write = store.compare_and_set_config

    import io
    import zipfile

    import requests

    send = requests.Session.send

    def capture_deployed_definition(session, request, **kwargs):
        if request.url.endswith("/prepareandactivate"):
            with zipfile.ZipFile(io.BytesIO(request.body)) as package:
                deployed_definitions.append(package.read(f"schemas/{schema}.sd"))
        return send(session, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", capture_deployed_definition)

    def die_before_registry_write(**kwargs):
        if kwargs["service"] == "schema_registry" and kwargs["tenant_id"] == tenant:
            expected.update(copy.deepcopy(kwargs["config_value"]))
            raise ProcessDeath("process killed after activation, before registration")
        return write(**kwargs)

    monkeypatch.setattr(tenant_manager, "get_backend", lambda: owner)
    monkeypatch.setattr(store, "compare_and_set_config", die_before_registry_write)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 0)
    try:
        with pytest.raises(
            ProcessDeath, match="killed after activation, before registration"
        ):
            asyncio.run(
                tenant_manager.create_tenant(
                    tenant_manager.CreateTenantRequest(
                        tenant_id=tenant,
                        created_by="recovery-test",
                        base_schemas=["wiki_pages"],
                    )
                )
            )
        assert set(owner.schema_manager.list_deployed_document_types()) & {schema} == {
            schema
        }
        assert _entry(store, tenant) is None
        assert (
            owner.get_metadata_document(schema="tenant_metadata", doc_id=tenant) is None
        )
        monkeypatch.setattr(store, "compare_and_set_config", write)
        survivor = connect()
        assert survivor.deploy_schemas([]) is True
        assert _entry(store, tenant).config_value == expected
        original_definition, reconstructed_definition = deployed_definitions
        assert reconstructed_definition == original_definition
        assert (
            _entry(store, tenant, "schema_deployment_intents").config_value["state"]
            == "complete"
        )
        assert set(survivor.schema_manager.list_deployed_document_types()) & {
            schema
        } == {schema}
    finally:
        monkeypatch.setattr(store, "compare_and_set_config", write)
        # Teardown restores only this test's confirmed live registration.
        if expected:
            store.set_config(
                tenant_id=tenant,
                scope=ConfigScope.SCHEMA,
                service="schema_registry",
                config_key="schema_wiki_pages",
                config_value=expected,
            )


def _feed_sentinel(backend, schema, tenant):
    from vespa.application import Vespa

    fields = {
        "doc_id": "sentinel",
        "tenant_id": tenant,
        "title": "Keep this document",
        "content": "Recovery preserves documents exactly.",
    }
    app = Vespa(url=backend._url, port=backend._port)
    responses = []
    app.feed_iterable(
        [{"id": "sentinel", "fields": fields}],
        schema=schema,
        namespace=schema,
        callback=lambda response, doc_id: responses.append(
            (doc_id, response.status_code)
        ),
    )
    assert responses == [("sentinel", 200)]
    return app, fields


def _assert_sentinel(app, schema, fields):
    stored = app.get_data(schema=schema, namespace=schema, data_id="sentinel")
    assert stored.status_code == 200
    assert stored.json["fields"] == fields


def test_live_owner_and_recovery_write_one_exact_registration(
    recovery_backend,
    monkeypatch,
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import cogniverse_core.registries.schema_registry as registry_module

    connect, store = recovery_backend
    owner, recovery = connect(), connect()
    tenant = f"orphan_{uuid4().hex[:12]}:race"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    barrier = threading.Barrier(2, timeout=120)
    activated = threading.Event()
    writes = []
    write = store.compare_and_set_config

    def gated_write(**kwargs):
        if kwargs["service"] == "schema_registry" and kwargs["tenant_id"] == tenant:
            writes.append(copy.deepcopy(kwargs["config_value"]))
            activated.set()
            barrier.wait()
        return write(**kwargs)

    monkeypatch.setattr(tenant_manager, "get_backend", lambda: owner)
    monkeypatch.setattr(store, "compare_and_set_config", gated_write)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 0)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(
            lambda: asyncio.run(
                tenant_manager.create_tenant(
                    tenant_manager.CreateTenantRequest(
                        tenant_id=tenant,
                        created_by="race-test",
                        base_schemas=["wiki_pages"],
                    )
                )
            )
        )
        try:
            assert activated.wait(120) is True
            assert _entry(store, tenant) is None
            app, fields = _feed_sentinel(owner, schema, tenant)
            recovered = recovery.schema_registry.reconcile_deployment_intents(
                set(recovery.schema_manager.list_deployed_document_types())
            )
            result = task.result(timeout=120)
            expected = writes[0]
            assert writes == [expected, expected]
            assert [vars(row) for row in recovered] == [expected]
            assert result.tenant_full_id == tenant
            assert result.schemas_deployed == ["wiki_pages"]
            registrations = [
                entry.config_value
                for entry in store.list_all_configs(
                    scope=ConfigScope.SCHEMA, service="schema_registry"
                )
                if entry.tenant_id == tenant
            ]
            assert registrations == [expected]
            assert _entry(store, tenant).version == 1
            assert (
                _entry(store, tenant, "schema_deployment_intents").config_value["state"]
                == "complete"
            )
            _assert_sentinel(app, schema, fields)
        finally:
            barrier.abort()
            monkeypatch.setattr(store, "compare_and_set_config", write)
            if writes:
                store.set_config(
                    tenant_id=tenant,
                    scope=ConfigScope.SCHEMA,
                    service="schema_registry",
                    config_key="schema_wiki_pages",
                    config_value=writes[0],
                )


def test_crash_before_activation_retires_intent_without_phantom_registration(
    recovery_backend,
    monkeypatch,
):
    import cogniverse_core.registries.schema_registry as registry_module

    connect, store = recovery_backend
    owner = connect()
    tenant = f"orphan_{uuid4().hex[:12]}:absent"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"

    def die_before_activation(_schemas):
        raise ProcessDeath("process killed before activation")

    monkeypatch.setattr(owner, "deploy_schemas", die_before_activation)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 0)
    with pytest.raises(ProcessDeath, match="process killed before activation"):
        owner.schema_registry.deploy_schema(tenant, "wiki_pages")
    assert (
        _entry(store, tenant, "schema_deployment_intents").config_value["state"]
        == "pending"
    )
    recovery = connect()
    assert (
        set(recovery.schema_manager.list_deployed_document_types()) & {schema} == set()
    )
    assert recovery.deploy_schemas([]) is True
    assert _entry(store, tenant) is None
    assert (
        _entry(store, tenant, "schema_deployment_intents").config_value["state"]
        == "absent"
    )
    assert (
        set(recovery.schema_manager.list_deployed_document_types()) & {schema} == set()
    )


def test_intent_storage_down_prevents_activation_with_context(
    recovery_backend,
    monkeypatch,
):
    import socket

    from cogniverse_core.registries.exceptions import RegistryStorageError

    connect, store = recovery_backend
    backend = connect()
    tenant = f"orphan_{uuid4().hex[:12]}:fault"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    write = store.compare_and_set_config
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        dead_store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=reserved.getsockname()[1]
        )

        def broken_intent_boundary(**kwargs):
            if kwargs["service"] == "schema_deployment_intents":
                return dead_store.compare_and_set_config(**kwargs)
            return write(**kwargs)

        monkeypatch.setattr(store, "compare_and_set_config", broken_intent_boundary)
        try:
            with pytest.raises(
                RegistryStorageError,
                match=f"Cannot persist deployment intent for '{schema}':",
            ) as failure:
                backend.schema_registry.deploy_schema(tenant, "wiki_pages")
            assert type(failure.value.__cause__).__name__ == "RuntimeError"
            assert type(failure.value.__cause__.__cause__).__name__ == "ConnectionError"
        finally:
            dead_store.close()
    assert _entry(store, tenant) is None
    assert _entry(store, tenant, "schema_deployment_intents") is None
    assert (
        set(backend.schema_manager.list_deployed_document_types()) & {schema} == set()
    )


def _create_in_process(port, config_port, tenant, connection, pause):
    from fastapi import HTTPException

    backend = _connect(port, config_port)
    tenant_manager.backend = backend
    registry = backend.schema_registry
    register = registry.register_schema
    if pause:

        def gated_register(**kwargs):
            connection.send(("activated", kwargs))
            if connection.recv() != "register":
                raise RuntimeError("creator barrier was not released")
            register(**kwargs)

        registry.register_schema = gated_register
    try:
        for attempt in range(120):
            try:
                result = asyncio.run(
                    tenant_manager.create_tenant(
                        tenant_manager.CreateTenantRequest(
                            tenant_id=tenant,
                            created_by="transient-test",
                            base_schemas=["wiki_pages"],
                        )
                    )
                )
                connection.send(("created", asdict(result)))
                return
            except HTTPException as exc:
                if "Refusing to deploy:" not in str(exc.detail):
                    raise
                connection.send(("refused", str(exc.detail)))
                if attempt == 119:
                    raise
                time.sleep(1)
    except BaseException as exc:
        connection.send(("error", repr(exc)))
        raise
    finally:
        backend.close()
        connection.close()


def test_two_process_tenant_creations_retry_transient_without_recovery(
    recovery_backend,
):
    connect, store = recovery_backend
    backend = connect()
    tenants = [f"orphan_{uuid4().hex[:12]}:{suffix}" for suffix in ("first", "second")]
    schemas = {f"wiki_pages_{tenant.replace(':', '_')}" for tenant in tenants}
    context = multiprocessing.get_context("spawn")
    parent_a, child_a = context.Pipe()
    parent_b, child_b = context.Pipe()
    first = context.Process(
        target=_create_in_process,
        args=(backend._port, backend._config_port, tenants[0], child_a, True),
    )
    second = context.Process(
        target=_create_in_process,
        args=(backend._port, backend._config_port, tenants[1], child_b, False),
    )
    expected = {}
    first.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert parent_a.poll(120) is True
            message, expected = parent_a.recv()
            if message != "refused":
                break
        assert message == "activated"
        schema_a = expected["full_schema_name"]
        app, fields = _feed_sentinel(backend, schema_a, tenants[0])
        second.start()
        assert parent_b.poll(120) is True
        message, detail = parent_b.recv()
        assert message == "refused"
        refused = re.search(
            r"cannot be reconstructed \((\[.*?\])\); proceeding would remove them and destroy their documents\.",
            detail,
        )
        assert set(ast.literal_eval(refused.group(1))) & schemas == {schema_a}
        assert _entry(store, tenants[0]) is None
        intent = _entry(store, tenants[0], "schema_deployment_intents").config_value
        assert (intent["state"], intent["attempts"]) == ("pending", 0)
        schema_b = f"wiki_pages_{tenants[1].replace(':', '_')}"
        assert _entry(store, tenants[1]) is None
        assert (
            set(backend.schema_manager.list_deployed_document_types()) & {schema_b}
            == set()
        )
        refused_intent = _entry(
            store, tenants[1], "schema_deployment_intents"
        ).config_value
        assert (refused_intent["state"], refused_intent["attempts"]) == ("absent", 0)
        parent_a.send("register")
        assert parent_a.poll(120) is True
        message, result_a = parent_a.recv()
        assert message == "created"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert parent_b.poll(120) is True
            message, result_b = parent_b.recv()
            if message != "refused":
                break
        assert message == "created"
        assert [result_a["tenant_full_id"], result_b["tenant_full_id"]] == tenants
        assert [result_a["schemas_deployed"], result_b["schemas_deployed"]] == [
            ["wiki_pages"],
            ["wiki_pages"],
        ]
        assert {
            _entry(store, tenant).config_value["full_schema_name"] for tenant in tenants
        } == schemas
        assert (
            set(backend.schema_manager.list_deployed_document_types()) & schemas
            == schemas
        )
        assert [
            _entry(store, tenant, "schema_deployment_intents").config_value["state"]
            for tenant in tenants
        ] == ["complete", "complete"]
        _assert_sentinel(app, schema_a, fields)
        first.join(10)
        second.join(10)
        assert (first.exitcode, second.exitcode) == (0, 0)
    finally:
        for process in (first, second):
            if process.is_alive():
                process.kill()
                process.join(10)
        if expected and "full_schema_name" in expected:
            backend.schema_registry.register_schema(**expected)
        parent_a.close()
        parent_b.close()


def test_recovery_preserves_document_orphan_and_unmanaged_peer(
    recovery_backend,
    monkeypatch,
):
    import json
    from datetime import datetime, timezone

    import cogniverse_core.registries.schema_registry as registry_module
    from cogniverse_core.registries.exceptions import BackendDeploymentError

    connect, store = recovery_backend
    backend = connect()
    owned_tenant, peer_tenant = [
        f"orphan_{uuid4().hex[:12]}:{suffix}" for suffix in ("documents", "peer")
    ]
    owned_name = f"wiki_pages_{owned_tenant.replace(':', '_')}"
    peer_name = f"wiki_pages_{peer_tenant.replace(':', '_')}"
    write = store.compare_and_set_config
    owned_registration = {}
    peer_definition = FilesystemSchemaLoader(Path("configs/schemas")).load_schema(
        "wiki_pages"
    )
    peer_definition["name"] = peer_name
    peer_registration = {
        "tenant_id": peer_tenant,
        "base_schema_name": "wiki_pages",
        "full_schema_name": peer_name,
        "schema_definition": json.dumps(peer_definition),
        "config": {},
        "deployment_time": datetime.now(timezone.utc).isoformat(),
    }

    def die_before_registry_write(**kwargs):
        if (
            kwargs["service"] == "schema_registry"
            and kwargs["tenant_id"] == owned_tenant
        ):
            owned_registration.update(copy.deepcopy(kwargs["config_value"]))
            raise ProcessDeath("interrupted owner with document schema")
        return write(**kwargs)

    monkeypatch.setattr(store, "compare_and_set_config", die_before_registry_write)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 0)
    try:
        with pytest.raises(
            ProcessDeath, match="interrupted owner with document schema"
        ):
            backend.schema_registry.deploy_schema(owned_tenant, "wiki_pages")
        app, owned_fields = _feed_sentinel(backend, owned_name, owned_tenant)
        monkeypatch.setattr(store, "compare_and_set_config", write)
        # The peer is deliberately deployed outside SchemaRegistry, so there
        # is no durable definition authorizing its registration by recovery.
        assert (
            backend.deploy_schemas(
                [
                    {
                        "name": peer_name,
                        "definition": peer_registration["schema_definition"],
                        "tenant_id": peer_tenant,
                        "base_schema_name": "wiki_pages",
                    }
                ]
            )
            is True
        )
        _, peer_fields = _feed_sentinel(backend, peer_name, peer_tenant)
        with pytest.raises(
            BackendDeploymentError, match="Refusing to deploy:"
        ) as refusal:
            connect().deploy_schemas([])
        refused = re.search(
            r"cannot be reconstructed \((\[.*?\])\); proceeding would remove them and destroy their documents\.",
            str(refusal.value),
        )
        assert set(ast.literal_eval(refused.group(1))) & {owned_name, peer_name} == {
            peer_name
        }
        assert _entry(store, owned_tenant).config_value == owned_registration
        assert _entry(store, peer_tenant) is None
        assert _entry(store, peer_tenant, "schema_deployment_intents") is None
        assert set(backend.schema_manager.list_deployed_document_types()) & {
            owned_name,
            peer_name,
        } == {owned_name, peer_name}
        _assert_sentinel(app, owned_name, owned_fields)
        _assert_sentinel(app, peer_name, peer_fields)
    finally:
        monkeypatch.setattr(store, "compare_and_set_config", write)
        live = set(
            backend.schema_manager.list_deployed_document_types(raise_on_failure=True)
        )
        for row in (owned_registration, peer_registration):
            if row and row["full_schema_name"] in live:
                backend.schema_registry.register_schema(**row)


def test_conditional_storage_race_has_one_winner(recovery_backend, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _, store = recovery_backend
    tenant = f"orphan_{uuid4().hex[:12]}:cas"
    coordinates = dict(
        tenant_id=tenant,
        scope=ConfigScope.SCHEMA,
        service="conditional_test",
        config_key="race",
    )
    barrier = threading.Barrier(2, timeout=10)
    first_committed = threading.Event()
    read = store.get_config
    read_count = 0
    lock = threading.Lock()

    def shared_snapshot(*args, **kwargs):
        nonlocal read_count
        entry = read(*args, **kwargs)
        with lock:
            read_count += 1
            count = read_count
        if count in (1, 2):
            barrier.wait()
            if threading.current_thread().name.endswith("_1"):
                assert first_committed.wait(10) is True
        return entry

    monkeypatch.setattr(store, "get_config", shared_snapshot)

    def write(value):
        result = store.compare_and_set_config(
            **coordinates, config_value=value, expected_version=0
        )
        if value == {"winner": "first"}:
            first_committed.set()
        return result

    with ThreadPoolExecutor(2, thread_name_prefix="conditional") as pool:
        first = pool.submit(write, {"winner": "first"})
        second = pool.submit(write, {"winner": "second"})
        winner = first.result(timeout=15)
        loser = second.result(timeout=15)
    assert winner.config_value == {"winner": "first"}
    assert winner.version == 1
    assert loser is None
    assert read(**coordinates).config_value == {"winner": "first"}
    assert [
        entry.config_value for entry in store.get_config_history(**coordinates)
    ] == [{"winner": "first"}]


def test_registration_outage_after_activation_preserves_recoverable_state(
    recovery_backend,
    monkeypatch,
):
    import socket

    import cogniverse_core.registries.schema_deployment_intents as intent_module
    import cogniverse_core.registries.schema_registry as registry_module
    from cogniverse_core.registries.exceptions import RegistryStorageError

    connect, store = recovery_backend
    owner = connect()
    tenant = f"orphan_{uuid4().hex[:12]}:registryfault"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    write = store.compare_and_set_config
    expected = {}
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        dead = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=reserved.getsockname()[1]
        )

        def down(**kwargs):
            if kwargs["service"] == "schema_registry" and kwargs["tenant_id"] == tenant:
                expected.update(copy.deepcopy(kwargs["config_value"]))
                return dead.compare_and_set_config(**kwargs)
            return write(**kwargs)

        monkeypatch.setattr(store, "compare_and_set_config", down)
        monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 90)
        try:
            with pytest.raises(
                RegistryStorageError,
                match=f"Failed to register schema '{schema}' in ConfigStore:.*Durable registration recovery is pending; the schema is preserved.",
            ) as failure:
                owner.schema_registry.deploy_schema(tenant, "wiki_pages")
            assert type(failure.value.__cause__).__name__ == "RuntimeError"
            assert type(failure.value.__cause__.__cause__).__name__ == "ConnectionError"
            assert _entry(store, tenant) is None
            assert (
                _entry(store, tenant, "schema_deployment_intents").config_value["state"]
                == "pending"
            )
            assert set(owner.schema_manager.list_deployed_document_types()) & {
                schema
            } == {schema}
            monkeypatch.setattr(store, "compare_and_set_config", write)
            monkeypatch.setattr(intent_module, "_now", lambda: time.time() + 91)
            assert connect().deploy_schemas([]) is True
            assert _entry(store, tenant).config_value == expected
            assert (
                _entry(store, tenant, "schema_deployment_intents").config_value["state"]
                == "complete"
            )
        finally:
            monkeypatch.setattr(store, "compare_and_set_config", write)
            if expected:
                owner.schema_registry.register_schema(**expected)
            dead.close()


def test_registry_tombstone_fences_paused_recovery(recovery_backend, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import cogniverse_core.registries.schema_registry as registry_module
    from cogniverse_core.registries.exceptions import RegistryStorageError

    connect, store = recovery_backend
    owner, recovery = connect(), connect()
    tenant = f"orphan_{uuid4().hex[:12]}:deleted"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    write = store.compare_and_set_config
    expected = {}

    def killed(**kwargs):
        if kwargs["service"] == "schema_registry" and kwargs["tenant_id"] == tenant:
            expected.update(copy.deepcopy(kwargs["config_value"]))
            raise ProcessDeath("killed creator")
        return write(**kwargs)

    monkeypatch.setattr(store, "compare_and_set_config", killed)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 0)
    with pytest.raises(ProcessDeath, match="killed creator"):
        owner.schema_registry.deploy_schema(tenant, "wiki_pages")
    reached = threading.Event()
    resume = threading.Event()

    def paused(**kwargs):
        if kwargs["service"] == "schema_registry" and kwargs["tenant_id"] == tenant:
            reached.set()
            assert resume.wait(30) is True
        return write(**kwargs)

    monkeypatch.setattr(store, "compare_and_set_config", paused)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(
            recovery.schema_registry.reconcile_deployment_intents,
            set(recovery.schema_manager.list_deployed_document_types()),
        )
        try:
            assert reached.wait(15) is True
            assert owner.schema_manager.delete_schema(tenant, "wiki_pages") == schema
            tombstone = _entry(store, tenant).config_value
            assert tombstone["deleted"] is True
            resume.set()
            with pytest.raises(
                RegistryStorageError,
                match=f"Registration of '{schema}' conflicted with a newer registry revision",
            ):
                task.result(timeout=15)
            assert _entry(store, tenant).config_value == tombstone
            assert (
                set(recovery.schema_manager.list_deployed_document_types()) & {schema}
                == set()
            )
            assert connect().deploy_schemas([]) is True
            assert (
                set(recovery.schema_manager.list_deployed_document_types()) & {schema}
                == set()
            )
        finally:
            resume.set()
            monkeypatch.setattr(store, "compare_and_set_config", write)


def test_stale_conditional_write_cannot_replace_pruned_history(
    recovery_backend, monkeypatch
):
    _, store = recovery_backend
    tenant = f"orphan_{uuid4().hex[:12]}:pruned"
    coordinates = dict(
        tenant_id=tenant,
        scope=ConfigScope.SCHEMA,
        service="conditional_test",
        config_key="history",
    )
    original = store.set_config(**coordinates, config_value={"generation": 1})
    assert original.version == 1
    read = store.get_config
    injected = False

    def advance_after_read(*args, **kwargs):
        nonlocal injected
        entry = read(*args, **kwargs)
        if not injected:
            injected = True
            for generation in range(2, 14):
                store.set_config(**coordinates, config_value={"generation": generation})
        return entry

    monkeypatch.setattr(store, "get_config", advance_after_read)
    assert (
        store.compare_and_set_config(
            **coordinates, config_value={"generation": "stale"}, expected_version=1
        )
        is None
    )
    assert read(**coordinates).config_value == {"generation": 13}
    assert read(**coordinates).version == 13
    assert [
        entry.version for entry in store.get_config_history(**coordinates, limit=100)
    ] == list(range(13, 3, -1))


def test_intent_retention_bounds_history_and_retired_scans_do_not_write(
    recovery_backend,
):
    from cogniverse_core.registries.schema_deployment_intents import (
        SchemaDeploymentIntents,
    )

    _, store = recovery_backend
    journal = SchemaDeploymentIntents(store)
    tenant = f"orphan_{uuid4().hex[:12]}:retention"
    schema = f"wiki_pages_{tenant.replace(':', '_')}"
    registration = {
        "tenant_id": tenant,
        "base_schema_name": "wiki_pages",
        "full_schema_name": schema,
        "schema_definition": '{"name": "' + schema + '"}',
        "config": {"retained": ["exact", "payload"]},
        "deployment_time": "2026-09-06T00:00:00+00:00",
    }
    for generation in range(8):
        record = journal.prepare(registration, grace_s=0, registry_version=generation)
        journal.complete(record)
    completed = _entry(store, tenant, "schema_deployment_intents")
    assert completed.version == 16
    assert completed.config_value["registration"] == registration
    assert completed.config_value["state"] == "complete"
    assert [
        entry.version
        for entry in store.get_config_history(
            tenant_id=SYSTEM_TENANT_ID,
            scope=ConfigScope.SCHEMA,
            service="schema_deployment_intents",
            config_key=schema,
            limit=100,
        )
    ] == list(range(16, 6, -1))
    for _ in range(12):
        assert (
            journal.reconcile(
                set(), {}, lambda row, version: pytest.fail(f"registered absent {row}")
            )
            == []
        )
    assert _entry(store, tenant, "schema_deployment_intents").version == 16
    assert [
        record
        for record in journal.records()
        if record["registration"]["full_schema_name"] == schema
    ] == [{**completed.config_value, "_revision": 16}]
