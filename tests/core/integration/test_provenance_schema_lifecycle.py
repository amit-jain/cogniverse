"""The provenance store deletes rows without deploying their schema, and
provenance schemas registered before ``primary_digest`` are redeployed once."""

import copy
import json
import time
from pathlib import Path
from uuid import uuid4

import pytest

from cogniverse_core.memory.backend_vector_store import BackendVectorStore
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    make_provenance,
)
from cogniverse_core.memory.provenance_store import (
    ProvenanceStore,
    ProvenanceWriteError,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration


class _PreDigestLoader(FilesystemSchemaLoader):
    """Ships the provenance schema as it was before the primary_digest field."""

    def load_schema(self, schema_name):
        definition = copy.deepcopy(super().load_schema(schema_name))
        if schema_name == "provenance":
            definition["document"]["fields"] = [
                field
                for field in definition["document"]["fields"]
                if field["name"] != "primary_digest"
            ]
        return definition


def _field_names(definition):
    return {field["name"] for field in json.loads(definition)["document"]["fields"]}


@pytest.fixture
def provenance_vespa(seeded_config_vespa):
    ports = seeded_config_vespa
    store = VespaConfigStore(
        backend_url="http://127.0.0.1", backend_port=ports["http_port"]
    )
    backends = []

    def connect(tenant_id, loader=None):
        loader = loader or FilesystemSchemaLoader(Path("configs/schemas"))
        manager = ConfigManager(store=store)
        backend = VespaBackend(
            BackendConfig(
                backend_type="vespa",
                url="http://127.0.0.1",
                port=ports["http_port"],
                tenant_id=tenant_id,
            ),
            schema_loader=loader,
            config_manager=manager,
        )
        backend._initialize_backend({"config_port": ports["config_port"]})
        backend.schema_registry = SchemaRegistry(manager, backend, loader)
        backend.schema_manager._schema_registry = backend.schema_registry
        backends.append(backend)
        return backend

    yield connect, store
    for backend in backends:
        backend.close()
    store.close()


@pytest.fixture
def deploys(monkeypatch):
    """Every registry deploy and backend package deploy this process makes."""
    calls = []
    registry_deploy = SchemaRegistry.deploy_schemas
    backend_deploy = VespaBackend.deploy_schemas

    def record_registry(self, tenant_id, base_schema_names, *args, **kwargs):
        calls.append(("registry", tenant_id, list(base_schema_names)))
        return registry_deploy(self, tenant_id, base_schema_names, *args, **kwargs)

    def record_backend(self, schema_definitions, *args, **kwargs):
        calls.append(("backend", sorted(s["name"] for s in schema_definitions)))
        return backend_deploy(self, schema_definitions, *args, **kwargs)

    monkeypatch.setattr(SchemaRegistry, "deploy_schemas", record_registry)
    monkeypatch.setattr(VespaBackend, "deploy_schemas", record_backend)
    return calls


def _schema_row(store, tenant):
    return store.get_config(
        tenant_id=tenant,
        scope=ConfigScope.SCHEMA,
        service="schema_registry",
        config_key="schema_provenance",
    )


def _provenance():
    return make_provenance(
        written_by="agent:schema-lifecycle",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.5,
        derived_from=[CitationRef.external("https://source.test/lifecycle")],
    )


@pytest.mark.parametrize(
    "state", ["client_cache_miss", "pre_digest_definition", "tombstoned_by_peer"]
)
def test_a_provenance_row_delete_never_deploys_the_schema(
    provenance_vespa, deploys, state
):
    connect, store = provenance_vespa
    tenant = f"provlc_{uuid4().hex[:10]}:acme"
    owner = connect(
        tenant,
        _PreDigestLoader(Path("configs/schemas"))
        if state == "pre_digest_definition"
        else None,
    )
    schema = owner.schema_registry.deploy_schema(tenant, "provenance")
    memory_id = f"mem-{uuid4().hex[:8]}"
    row_id = f"prov-{tenant}-{memory_id}"
    owner.put_document_fields(
        row_id,
        {
            "id": row_id,
            "memory_id": memory_id,
            "tenant_id": tenant,
            "written_by": "agent:schema-lifecycle",
            "written_at": 1700000000,
            "derivation_kind": "direct_ingest",
            "confidence": 0.5,
            "derived_from_ids": [],
            "derived_from_other": "[]",
            "trace_id": "",
        },
        schema_name=schema,
        namespace="content",
    )
    deleter = connect(tenant)
    store_for_deleter = ProvenanceStore(lambda: deleter, tenant_id=tenant)
    assert list(store_for_deleter.fetch([memory_id])) == [memory_id]
    assert deleter.schema_exists("provenance", tenant_id=tenant) is True
    if state == "tombstoned_by_peer":
        assert owner.schema_manager.delete_schema(tenant, "provenance") == schema
    row_before = _schema_row(store, tenant)
    deploys.clear()

    assert store_for_deleter.delete(memory_id) is True

    assert deploys == []
    assert _schema_row(store, tenant).version == row_before.version
    live = set(deleter.schema_manager.list_deployed_document_types(True))
    if state == "tombstoned_by_peer":
        assert schema not in live
        assert _schema_row(store, tenant).config_value["deleted"] is True
    else:
        assert schema in live
        assert store_for_deleter.fetch([memory_id]) == {}
    if state == "pre_digest_definition":
        definition = _schema_row(store, tenant).config_value["schema_definition"]
        assert "primary_digest" not in _field_names(definition)


def test_a_delete_from_a_document_type_vespa_lacks_is_an_absence(
    provenance_vespa, deploys
):
    connect, _store = provenance_vespa
    tenant = f"provnone_{uuid4().hex[:10]}:acme"
    backend = connect(tenant)
    schema = backend.get_tenant_schema_name(tenant, "provenance")
    assert schema not in set(backend.schema_manager.list_deployed_document_types(True))

    assert backend.delete_live_document("prov-row", "provenance") is True

    assert deploys == []
    assert backend._vespa_ingestion_clients == {}
    assert schema not in set(backend.schema_manager.list_deployed_document_types(True))


def test_the_migration_redeploys_a_pre_digest_provenance_schema_once(
    provenance_vespa, deploys
):
    connect, store = provenance_vespa
    tenant = f"provmig_{uuid4().hex[:10]}:acme"
    current = f"provmig_{uuid4().hex[:10]}:current"
    legacy = connect(tenant, _PreDigestLoader(Path("configs/schemas")))
    schema = legacy.schema_registry.deploy_schema(tenant, "provenance")
    current_schema = connect(current).schema_registry.deploy_schema(
        current, "provenance"
    )
    writer = connect(tenant)
    with pytest.raises(ProvenanceWriteError):
        ProvenanceStore(lambda: legacy, tenant_id=tenant).attach(
            "mem-before", _provenance(), primary_digest="a" * 64
        )
    current_row = _schema_row(store, current)
    deploys.clear()

    migrated = writer.schema_registry.redeploy_drifted_schemas("provenance")

    assert schema in migrated
    assert current_schema not in migrated
    assert all(name.startswith("provenance_") for name in migrated)
    assert ("registry", tenant, ["provenance"]) in deploys
    assert ("registry", current, ["provenance"]) not in deploys
    assert _schema_row(store, current).version == current_row.version
    definition = _schema_row(store, tenant).config_value["schema_definition"]
    assert "primary_digest" in _field_names(definition)
    row_id = ProvenanceStore(lambda: writer, tenant_id=tenant).attach(
        "mem-after", _provenance(), primary_digest="b" * 64
    )
    assert row_id == f"prov-{tenant}-mem-after"
    indexed = ProvenanceStore(lambda: writer, tenant_id=tenant).get("mem-after")
    assert indexed.primary_digest == "b" * 64
    deploys.clear()

    assert writer.schema_registry.redeploy_drifted_schemas("provenance") == []
    assert deploys == []


@pytest.mark.parametrize("peer_deletes", ["before_the_decision", "after_the_decision"])
def test_the_migration_does_not_redeploy_a_schema_a_peer_deleted_meanwhile(
    provenance_vespa, deploys, peer_deletes
):
    """The drifted list is read once and each redeploy takes minutes; a tenant
    whose schema a peer deletes in between is not deployed and registered
    again — whether the deletion lands before the redeploy reads the stored
    row or after, while the package is being built."""
    connect, store = provenance_vespa
    tenant = f"provgone_{uuid4().hex[:10]}:acme"
    legacy = connect(tenant, _PreDigestLoader(Path("configs/schemas")))
    schema = legacy.schema_registry.deploy_schema(tenant, "provenance")
    peer = connect(tenant)
    writer = connect(tenant)
    registry = writer.schema_registry
    deploy = registry.deploy_schemas
    activate = writer.deploy_schemas
    for_tenant = {}

    def peer_delete():
        assert peer.schema_manager.delete_schema(tenant, "provenance") == schema

    def deploy_for_tenant(tenant_id, base_schema_names, *args, **kwargs):
        if tenant_id != tenant:
            return deploy(tenant_id, base_schema_names, *args, **kwargs)
        if peer_deletes == "before_the_decision":
            peer_delete()
        before = len(deploys)
        try:
            return deploy(tenant_id, base_schema_names, *args, **kwargs)
        finally:
            for_tenant["calls"] = deploys[before:]

    def activate_after_a_peer_delete(schemas, *args, **kwargs):
        if peer_deletes == "after_the_decision" and schema in {
            definition["name"] for definition in schemas
        }:
            peer_delete()
        return activate(schemas, *args, **kwargs)

    registry.deploy_schemas = deploy_for_tenant
    writer.deploy_schemas = activate_after_a_peer_delete

    migrated = registry.redeploy_drifted_schemas("provenance")

    assert schema not in migrated
    assert for_tenant["calls"][0] == ("registry", tenant, ["provenance"])
    if peer_deletes == "before_the_decision":
        assert for_tenant["calls"] == [("registry", tenant, ["provenance"])]
    assert _schema_row(store, tenant).config_value["deleted"] is True
    assert schema not in set(peer.schema_manager.list_deployed_document_types(True))


def _memory_row(backend, schema, memory_id, tenant):
    backend.put_document_fields(
        memory_id,
        {"id": memory_id, "text": "a remembered turn", "user_id": tenant},
        schema_name=schema,
        namespace="memory_content",
    )


@pytest.mark.parametrize("state", ["client_cache_miss", "tombstoned_by_peer"])
def test_a_memory_row_delete_never_deploys_the_memory_schema(
    provenance_vespa, deploys, state
):
    """Memory deletes, namespace clears and retention remove primaries through
    the vector store, reading each row first as Mem0's delete does; neither
    the read nor the delete may deploy (or resurrect) the tenant's
    agent_memories schema on an ingestion-client cache miss."""
    connect, store = provenance_vespa
    tenant = f"memdel_{uuid4().hex[:10]}:acme"
    owner = connect(tenant)
    schema = owner.schema_registry.deploy_schema(tenant, "agent_memories")
    memory_id = f"mem-{uuid4().hex[:8]}"
    _memory_row(owner, schema, memory_id, tenant)
    assert owner.get_document_fields(
        memory_id, schema_name=schema, namespace="memory_content"
    ) == {"id": memory_id, "text": "a remembered turn", "user_id": tenant}
    deleter = connect(tenant)
    vectors = BackendVectorStore(
        collection_name=schema,
        backend_resolver=lambda: deleter,
        tenant_id=tenant,
        profile="agent_memories",
    )
    if state == "tombstoned_by_peer":
        assert owner.schema_manager.delete_schema(tenant, "agent_memories") == schema
    deploys.clear()

    record = vectors.get(memory_id)
    vectors.delete(memory_id)

    assert deploys == []
    assert deleter._vespa_ingestion_clients == {}
    live = set(deleter.schema_manager.list_deployed_document_types(True))
    if state == "tombstoned_by_peer":
        # The removal redeploy returns once activated; the content node may
        # still serve the dropped type's row until the generation reaches it.
        assert record is None or (record.id, record.payload["data"]) == (
            memory_id,
            "a remembered turn",
        )
        assert schema not in live
        row = store.get_config(
            tenant_id=tenant,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key="schema_agent_memories",
        )
        assert row.config_value["deleted"] is True
        deadline = time.monotonic() + 60
        after_removal = vectors.get(memory_id)
        while after_removal is not None and time.monotonic() < deadline:
            time.sleep(1)
            after_removal = vectors.get(memory_id)
        assert after_removal is None
        assert deploys == []
        assert deleter._vespa_ingestion_clients == {}
    else:
        assert (record.id, record.payload["data"]) == (memory_id, "a remembered turn")
        assert schema in live
        assert (
            owner.get_document_fields(
                memory_id, schema_name=schema, namespace="memory_content"
            )
            is None
        )
