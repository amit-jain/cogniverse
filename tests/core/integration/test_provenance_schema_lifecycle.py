"""The provenance store deletes rows without deploying their schema, and
provenance schemas registered before ``primary_digest`` are redeployed once."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    make_provenance,
)
from cogniverse_core.memory.provenance_store import ProvenanceStore
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
