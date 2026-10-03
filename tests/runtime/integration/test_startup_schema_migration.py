"""The runtime's startup schema migration, against real Vespa.

A release that changes a shipped schema definition reaches every tenant that
registered the older one: once startup completes, the runtime redeploys each
drifted tenant's schemas in one package under the deployment lease. A change
Vespa refuses is left unapplied, logged and recorded by tenant and schema,
and ``GET /admin/schemas/drift`` lists it with Vespa's reason. Two runtimes
starting together redeploy each tenant once, and a config server that is down
makes the migration retry until it returns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing
import queue
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote

import httpx
import numpy as np
import pytest
import requests
from fastapi import FastAPI

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.schema_registry import (
    SCHEMA_REFUSALS_SERVICE,
    SCHEMA_REGISTRY_SERVICE,
    DeployedSchemaNames,
    DriftedSchema,
    SchemaRefusal,
    drifted_schemas,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.ingestion_client import document_namespace
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = pytest.mark.integration

SCHEMAS_DIR = Path("configs/schemas")
FRAME = "video_colpali_smol500_mv_frame"
VISUAL = "document_visual"
TEXT = "document_text"
SHIPPED_PROFILES = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
]
DIM = 320

# A query token's binary MaxSim against its nearest patch as the ColQwen3
# schemas ship it, 1-2h/320, and as they shipped it before, 1/(1+h). A rank
# profile sums it over the query tokens or averages it (the fused hybrids).
SHIPPED_BINARY_MAXSIM = re.compile(
    r"1 - 2 \* reduce\(sum\(hamming\(query\(qtb\), attribute\((\w+)\)\), v\), "
    r"min, patch\) / 320"
)
OLDER_BINARY_MAXSIM = (
    r"reduce(1/(1+ sum(hamming(query(qtb), attribute(\1)), v)), max, patch)"
)
SHIPPED_FRAME_EXPRESSION = (
    "1 - 2 * reduce(sum(hamming(query(qtb), attribute(embedding_binary)), v), "
    "min, patch) / 320"
)
OLDER_FRAME_EXPRESSION = (
    "reduce(1/(1+ sum(hamming(query(qtb), attribute(embedding_binary)), v)), "
    "max, patch)"
)
CONFIG_SERVER_SCHEMAS = (
    "/application/v2/tenant/default/application/default/environment/prod/region/"
    "default/instance/default/content/schemas/"
)


def _shipped(base: str) -> dict:
    return json.loads((SCHEMAS_DIR / f"{base}_schema.json").read_text())


def _older_rank_profiles(definition: dict) -> dict:
    """The definition with its binary MaxSim as it shipped before the change."""
    text, count = SHIPPED_BINARY_MAXSIM.subn(
        OLDER_BINARY_MAXSIM, json.dumps(definition)
    )
    assert count > 0, f"{definition['name']} carries no binary MaxSim to revert"
    return json.loads(text)


def _chunk_index_as_string(definition: dict) -> dict:
    """The definition with ``chunk_index`` a string, so the shipped int is a
    field type change Vespa refuses without a validation override."""
    for field in definition["document"]["fields"]:
        if field["name"] == "chunk_index":
            field["type"] = "string"
    return definition


class _OlderRelease(FilesystemSchemaLoader):
    """Ships configs/schemas with the given schemas as an older release did."""

    def __init__(self, changes: dict):
        super().__init__(SCHEMAS_DIR)
        self._changes = changes

    def load_schema(self, schema_name):
        definition = super().load_schema(schema_name)
        change = self._changes.get(schema_name)
        return definition if change is None else change(definition)


class _OnlyShips(FilesystemSchemaLoader):
    """Ships only the named schemas of configs/schemas, unchanged."""

    def __init__(self, *names: str):
        super().__init__(SCHEMAS_DIR)
        self._names = set(names)

    def load_schema(self, schema_name):
        if schema_name not in self._names:
            from cogniverse_sdk.interfaces.schema_loader import SchemaNotFoundException

            raise SchemaNotFoundException(schema_name)
        return super().load_schema(schema_name)


class _OwnRelease(FilesystemSchemaLoader):
    """Ships configs/schemas plus schemas of the operator's own that
    configs/schemas does not ship: ``own`` maps each base to a builder of its
    definition. Only tenants that registered one of these bases can drift on
    it, so a run under this release changes no other test's tenants."""

    def __init__(self, own: dict):
        super().__init__(SCHEMAS_DIR)
        self._own = own

    def load_schema(self, schema_name):
        build = self._own.get(schema_name)
        if build is None:
            return super().load_schema(schema_name)
        return {**build(), "name": schema_name}


def _string_chunk_index_text() -> dict:
    return _chunk_index_as_string(_shipped(TEXT))


def _titled_string_chunk_index_text() -> dict:
    """The string ``chunk_index`` text schema with one more rank profile: a
    change Vespa applies to a schema registered without it."""
    definition = _string_chunk_index_text()
    definition["rank_profiles"].append(dict(OWN_PROFILE))
    return definition


def _string_page_number_visual() -> dict:
    """The visual schema with ``page_number`` a string, so the shipped int is
    a field type change Vespa refuses without a validation override."""
    definition = _shipped(VISUAL)
    for field in definition["document"]["fields"]:
        if field["name"] == "page_number":
            field["type"] = "string"
    return definition


def _own_bases(run: str) -> tuple[str, str]:
    """The text-derived and visual-derived bases of the operator's own one
    test run registers."""
    return f"own_text_{run}", f"own_visual_{run}"


def _refusing_release(run: str) -> _OwnRelease:
    """The run's own bases as the shipped text and visual schemas: a field
    type change from what the run registered, which Vespa refuses."""
    own_text, own_visual = _own_bases(run)
    return _OwnRelease(
        {own_text: lambda: _shipped(TEXT), own_visual: lambda: _shipped(VISUAL)}
    )


def _registering_release(run: str) -> _OwnRelease:
    own_text, own_visual = _own_bases(run)
    return _OwnRelease(
        {own_text: _string_chunk_index_text, own_visual: _string_page_number_visual}
    )


def _digest(definition: dict) -> str:
    return hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _recorded_refusals(manager, names) -> dict:
    """Each recorded migration refusal among the full schema ``names``:
    schema -> (tenant, SHA-256 of the definition it was refused)."""
    return {
        record.config_key: (
            record.config_value["tenant_id"],
            record.config_value["definition_sha256"],
        )
        for record in manager.store.list_configs(
            tenant_id=SYSTEM_TENANT_ID,
            scope=ConfigScope.SCHEMA,
            service=SCHEMA_REFUSALS_SERVICE,
        )
        if record.config_key in names
    }


def test_the_older_frame_definition_differs_only_in_its_binary_maxsim():
    """The fixture's older release differs from the shipped frame schema in
    the per-token binary MaxSim of the seven rank profiles that score it, and
    nothing else."""
    shipped = _shipped(FRAME)
    older = _older_rank_profiles(_shipped(FRAME))

    assert older["document"] == shipped["document"]
    changed = [
        old["name"]
        for old, new in zip(older["rank_profiles"], shipped["rank_profiles"])
        if old != new
    ]
    assert changed == [
        "default",
        "binary_binary",
        "phased",
        "hybrid_binary_bm25",
        "hybrid_bm25_binary",
        "hybrid_binary_bm25_no_description",
        "hybrid_bm25_binary_no_description",
    ]
    assert json.dumps(older).count(OLDER_FRAME_EXPRESSION) == 7
    assert SHIPPED_FRAME_EXPRESSION not in json.dumps(older)


@pytest.fixture
def migration_vespa(seeded_config_vespa):
    """``connect(tenant, loader, config_port, store_manager)``: a VespaBackend
    wired to a real SchemaRegistry over the shared Vespa, as runtime startup
    builds it."""
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_vespa.backend import VespaBackend
    from cogniverse_vespa.config.config_store import VespaConfigStore

    ports = seeded_config_vespa
    store = VespaConfigStore(
        backend_url="http://127.0.0.1", backend_port=ports["http_port"]
    )
    manager = ConfigManager(store=store)
    backends = []

    def connect(tenant_id, loader=None, config_port=None, store_manager=None):
        loader = loader or FilesystemSchemaLoader(SCHEMAS_DIR)
        store_manager = store_manager or manager
        backend = VespaBackend(
            BackendConfig(
                backend_type="vespa",
                url="http://127.0.0.1",
                port=ports["http_port"],
                tenant_id=tenant_id,
            ),
            schema_loader=loader,
            config_manager=store_manager,
        )
        backend._initialize_backend(
            {"config_port": config_port or ports["config_port"]}
        )
        backend.schema_registry = SchemaRegistry(store_manager, backend, loader)
        backend.schema_manager._schema_registry = backend.schema_registry
        backends.append(backend)
        return backend

    yield connect, manager, ports
    for backend in backends:
        backend.close()
    store.close()


def _row(manager, tenant_id: str, base: str):
    return manager.store.get_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SCHEMA,
        service=SCHEMA_REGISTRY_SERVICE,
        config_key=f"schema_{base}",
    )


def _registered_definition(manager, tenant_id: str, base: str) -> dict:
    return json.loads(_row(manager, tenant_id, base).config_value["schema_definition"])


def _named(definition: dict, name: str) -> dict:
    return {**definition, "name": name}


def _live_sd(ports, schema_name: str) -> str:
    response = requests.get(
        f"http://127.0.0.1:{ports['config_port']}{CONFIG_SERVER_SCHEMAS}"
        f"{schema_name}.sd",
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.text


def _ours(caplog, names) -> list[tuple[str, str]]:
    from cogniverse_runtime import main as runtime_main

    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == runtime_main.logger.name
        and any(name in record.getMessage() for name in names)
    ]


# Two frames scored against a two-token query. Frame A matches token 0
# bit-for-bit and sits 160 bits from token 1; frame B sits 32 bits from both.
# 1/(1+h) per token ranks A first (1 + 1/161 against 2/33); 1 - 2h/320 ranks
# B first (2 x 0.8 = 1.6 against 1 + 0 = 1.0).
QUERY = np.full((2, DIM), 0.05, dtype=np.float32)
QUERY[1, DIM // 2 :] = -0.05
FRAME_A = np.full((1, DIM), 0.05, dtype=np.float32)
FRAME_B = QUERY.copy()
FRAME_B[0, :32] = -0.05
FRAME_B[1, DIM // 2 : DIM // 2 + 32] = 0.05
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _min_hamming(query: np.ndarray, patches: np.ndarray) -> list[int]:
    def bits(vectors):
        return np.packbits((vectors > 0).astype(np.uint8), axis=1)

    xor = np.bitwise_xor(bits(query)[:, None, :], bits(patches)[None])
    return _POPCOUNT[xor].sum(-1).min(axis=1).tolist()


def _frame(doc_id: str, patches: np.ndarray, video_id: str) -> Document:
    document = Document(
        id=doc_id,
        content_type=ContentType.VIDEO,
        content_id=video_id,
        status=ProcessingStatus.COMPLETED,
    )
    document.add_embedding("embedding", patches, {"type": "float", "raw": True})
    document.add_metadata("video_id", video_id)
    document.add_metadata("video_title", f"{video_id}.mp4")
    document.add_metadata("segment_index", 0)
    document.add_metadata("start_time", 1.5)
    document.add_metadata("end_time", 2.5)
    return document


def _binary_ranking(manager, ports, tenant_id: str) -> list[tuple[str, float]]:
    from cogniverse_vespa.search_backend import VespaSearchBackend

    search = VespaSearchBackend(
        config={
            "url": "http://127.0.0.1",
            "port": ports["http_port"],
            "profiles": {FRAME: SHIPPED_PROFILES[FRAME]},
        },
        config_manager=manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=DeployedSchemaNames(manager),
        enable_connection_pool=False,
    )
    try:
        results = search.search(
            {
                "query": "",
                "type": "video",
                "profile": FRAME,
                "strategy": "binary_binary",
                "top_k": 10,
                "tenant_id": tenant_id,
                "query_embeddings": QUERY,
            }
        )
    finally:
        search.close()
    return [(r.document.metadata["video_id"], r.score) for r in results]


@pytest.mark.asyncio
async def test_a_tenant_on_the_older_rank_profile_is_ranked_by_the_shipped_one(
    migration_vespa, caplog
):
    """A tenant whose frame schema was registered before the binary MaxSim
    changed is redeployed at startup: the live rank profile becomes the
    shipped one, and the tenant's frames stay present and searchable."""
    from cogniverse_runtime import main as runtime_main

    connect, manager, ports = migration_vespa
    tenant = f"migrank_{uuid.uuid4().hex[:10]}:acme"
    older = _OlderRelease({FRAME: _older_rank_profiles})
    owner = connect(tenant, older)
    schema = owner.schema_registry.deploy_schema(tenant, FRAME)
    try:
        assert _min_hamming(QUERY, FRAME_A) == [0, 160]
        assert _min_hamming(QUERY, FRAME_B) == [32, 32]
        fed = owner.ingest_documents(
            [
                _frame("frame_a", FRAME_A, "video_a"),
                _frame("frame_b", FRAME_B, "video_b"),
            ],
            FRAME,
        )
        assert fed == {
            "success_count": 2,
            "failed_count": 0,
            "failed_documents": [],
            "total_documents": 2,
        }
        before = _binary_ranking(manager, ports, tenant)
        assert [video for video, _ in before] == ["video_a", "video_b"]
        assert [score for _, score in before] == pytest.approx(
            [1 + 1 / 161, 2 / 33], abs=1e-6
        )
        assert _live_sd(ports, schema).count(OLDER_FRAME_EXPRESSION) == 7
        registry = connect(tenant).schema_registry
        caplog.set_level("INFO", logger=runtime_main.logger.name)

        await asyncio.wait_for(
            runtime_main._migrate_drifted_schemas(lambda: registry), timeout=900
        )

        assert _registered_definition(manager, tenant, FRAME) == _named(
            _shipped(FRAME), schema
        )
        live = _live_sd(ports, schema)
        assert live.count(SHIPPED_FRAME_EXPRESSION) == 7
        assert OLDER_FRAME_EXPRESSION not in live
        after = _binary_ranking(manager, ports, tenant)
        assert [video for video, _ in after] == ["video_b", "video_a"]
        assert [score for _, score in after] == pytest.approx([1.6, 1.0], abs=1e-6)
        stored = {
            doc_id: {
                key: owner.get_document_fields(
                    doc_id, schema_name=schema, namespace=document_namespace(schema)
                )[key]
                for key in ("video_id", "video_title", "start_time", "end_time")
            }
            for doc_id in ("frame_a", "frame_b")
        }
        assert stored == {
            "frame_a": {
                "video_id": "video_a",
                "video_title": "video_a.mp4",
                "start_time": 1.5,
                "end_time": 2.5,
            },
            "frame_b": {
                "video_id": "video_b",
                "video_title": "video_b.mp4",
                "start_time": 1.5,
                "end_time": 2.5,
            },
        }
        [redeployed] = [
            record.args[0]
            for record in caplog.records
            if record.name == runtime_main.logger.name
            and record.msg == "Migration of drifted schemas redeployed %s"
        ]
        assert redeployed.count(schema) == 1
        assert [
            entry
            for entry in drifted_schemas(manager, FilesystemSchemaLoader(SCHEMAS_DIR))
            if entry.tenant_id == tenant
        ] == []
    finally:
        owner.schema_manager.delete_schema(tenant, FRAME)


async def _drift_listing(manager) -> httpx.Response:
    from cogniverse_runtime.routers import admin

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.dependency_overrides[admin.get_config_manager_dependency] = lambda: manager
    app.dependency_overrides[admin.get_schema_loader_dependency] = lambda: (
        FilesystemSchemaLoader(SCHEMAS_DIR)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/admin/schemas/drift")


@pytest.mark.asyncio
async def test_a_change_vespa_refuses_is_reported_and_left_unapplied(
    migration_vespa, caplog
):
    """One tenant carries two drifted schemas: a rank-profile change and a
    field type change Vespa refuses without a validation override. The
    refused schema keeps its live and registered definition and its
    documents, and is logged, recorded and listed with Vespa's reason by
    tenant and schema; the tenant's other schema is still redeployed."""
    from cogniverse_runtime import main as runtime_main

    connect, manager, ports = migration_vespa
    tenant = f"migrefuse_{uuid.uuid4().hex[:10]}:acme"
    older = _OlderRelease(
        {
            FRAME: _older_rank_profiles,
            TEXT: lambda definition: _chunk_index_as_string(definition),
        }
    )
    owner = connect(tenant, older)
    frame_schema, text_schema = owner.schema_registry.deploy_schemas(
        tenant, [FRAME, TEXT]
    )
    try:
        owner.put_document_fields(
            "chunk_0",
            {
                "document_id": "manual",
                "document_title": "manual.md",
                "full_text": "the refused schema keeps this row",
                "chunk_index": "0",
            },
            schema_name=text_schema,
            namespace=document_namespace(text_schema),
        )
        text_row = _row(manager, tenant, TEXT)
        registry = connect(tenant).schema_registry
        caplog.set_level("INFO", logger=runtime_main.logger.name)

        await asyncio.wait_for(
            runtime_main._migrate_drifted_schemas(lambda: registry), timeout=900
        )

        reported = _ours(caplog, [text_schema])
        assert [level for level, _ in reported] == ["ERROR"]
        message = reported[0][1]
        assert message.startswith(
            f"Migration of drifted schemas could not redeploy {text_schema} for "
            f"tenant {tenant}: Backend deployment failed for schema "
            f"'{text_schema}': Vespa refused the application package: "
            "Deployment failed with status 400: "
        ), message
        assert "Field 'chunk_index' changed: data type: 'string' -> 'int'" in message

        assert _row(manager, tenant, TEXT).version == text_row.version
        assert _registered_definition(manager, tenant, TEXT) == _named(
            _chunk_index_as_string(_shipped(TEXT)), text_schema
        )
        assert "field chunk_index type string" in _live_sd(ports, text_schema)
        assert {
            key: value
            for key, value in owner.get_document_fields(
                "chunk_0",
                schema_name=text_schema,
                namespace=document_namespace(text_schema),
            ).items()
            if key in ("document_id", "full_text", "chunk_index")
        } == {
            "document_id": "manual",
            "full_text": "the refused schema keeps this row",
            "chunk_index": "0",
        }

        assert _registered_definition(manager, tenant, FRAME) == _named(
            _shipped(FRAME), frame_schema
        )
        assert _live_sd(ports, frame_schema).count(SHIPPED_FRAME_EXPRESSION) == 7

        listed = [
            entry
            for entry in drifted_schemas(manager, FilesystemSchemaLoader(SCHEMAS_DIR))
            if entry.tenant_id == tenant
        ]
        assert len(listed) == 1
        refusal = listed[0].refusal
        assert listed == [
            DriftedSchema(
                tenant_id=tenant,
                base_schema_name=TEXT,
                schema_name=text_schema,
                refusal=SchemaRefusal(
                    tenant_id=tenant,
                    base_schema_name=TEXT,
                    schema_name=text_schema,
                    error=message.split(f"for tenant {tenant}: ", 1)[1],
                    refused_at=refusal.refused_at,
                ),
            )
        ]
        response = await _drift_listing(manager)
        assert response.status_code == 200, response.text
        assert [
            entry
            for entry in response.json()["drifted"]
            if entry["tenant_id"] == tenant
        ] == [
            {
                "tenant_id": tenant,
                "base_schema_name": TEXT,
                "schema_name": text_schema,
                "refusal": {
                    "error": refusal.error,
                    "refused_at": refusal.refused_at,
                },
            }
        ]
    finally:
        owner.schema_manager.delete_schema(tenant, TEXT)
        owner.schema_manager.delete_schema(tenant, FRAME)


OWN = "acme_notes"
OWN_PROFILE = {
    "name": "acme_title_first",
    "first_phase": "2 * bm25(document_title) + bm25(full_text)",
    "timeout": 2.0,
}


class _OperatorRelease(_OlderRelease):
    """The older frame schema, plus ``acme_notes``: a schema of the
    operator's own that configs/schemas does not ship, the text schema with
    one more rank profile."""

    def __init__(self):
        super().__init__({FRAME: _older_rank_profiles})

    def load_schema(self, schema_name):
        if schema_name != OWN:
            return super().load_schema(schema_name)
        definition = super().load_schema(TEXT)
        definition["name"] = OWN
        definition["rank_profiles"].append(dict(OWN_PROFILE))
        return definition


@pytest.mark.asyncio
async def test_a_tenant_marked_deleted_is_left_undeployed_and_the_rest_migrate(
    migration_vespa, caplog
):
    """A tenant whose delete has not completed stays marked deleted with its
    schemas still registered. The migration never redeploys them, and the
    other drifted tenants still migrate in the same run."""
    from cogniverse_core.common.tenant_utils import (
        clear_tenant_deleted,
        mark_tenant_deleted,
    )
    from cogniverse_core.registries import schema_registry as schema_registry_module

    connect, manager, ports = migration_vespa
    run = uuid.uuid4().hex[:10]
    gone, kept = f"migdel_{run}:gone", f"migdel_{run}:kept"
    older = _OlderRelease({FRAME: _older_rank_profiles})
    owners = {tenant: connect(tenant, older) for tenant in (gone, kept)}
    schemas = {
        tenant: owners[tenant].schema_registry.deploy_schema(tenant, FRAME)
        for tenant in owners
    }
    ours = set(schemas.values())
    try:
        mark_tenant_deleted(manager.store, gone)
        registry = connect(kept).schema_registry
        caplog.set_level("INFO", logger=schema_registry_module.logger.name)

        result = await asyncio.to_thread(registry.redeploy_drifted_schemas)

        assert [name for name in result.redeployed if name in ours] == [schemas[kept]]
        assert [name for name in result.deleted if name in ours] == [schemas[gone]]
        assert [r for r in result.refused if r.schema_name in ours] == []
        assert _registered_definition(manager, kept, FRAME) == _named(
            _shipped(FRAME), schemas[kept]
        )
        assert _registered_definition(manager, gone, FRAME) == _named(
            _older_rank_profiles(_shipped(FRAME)), schemas[gone]
        )
        assert _live_sd(ports, schemas[kept]).count(SHIPPED_FRAME_EXPRESSION) == 7
        assert _live_sd(ports, schemas[gone]).count(OLDER_FRAME_EXPRESSION) == 7
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == schema_registry_module.logger.name
            and gone in record.getMessage()
            and "marked deleted" in record.getMessage()
        ] == [
            f"Tenant '{gone}' is marked deleted; its drifted schemas "
            f"{[schemas[gone]]} are left undeployed"
        ]
    finally:
        clear_tenant_deleted(manager.store, gone)
        for tenant, owner in owners.items():
            owner.schema_manager.delete_schema(tenant, FRAME)


@pytest.mark.asyncio
async def test_the_migration_changes_only_the_drifted_definitions(migration_vespa):
    """A tenant carries a drifted schema registered with a config, a current
    schema and a schema of its own that this runtime does not ship; a
    neighbour carries the current definition of the drifted base. One package
    redeploys the drifted schema alone and its registration keeps its config;
    every other registration and live definition is left as it was."""
    connect, manager, ports = migration_vespa
    tenant = f"migkeep_{uuid.uuid4().hex[:10]}:acme"
    neighbour = f"migkeep_{uuid.uuid4().hex[:10]}:globex"
    owner = connect(tenant, _OperatorRelease())
    frame_schema = owner.schema_registry.deploy_schema(
        tenant, FRAME, config={"profile": "acme_frames"}
    )
    text_schema, own_schema = owner.schema_registry.deploy_schemas(tenant, [TEXT, OWN])
    neighbour_owner = connect(neighbour)
    neighbour_schema = neighbour_owner.schema_registry.deploy_schema(neighbour, FRAME)
    kept = {
        (tenant, TEXT): text_schema,
        (tenant, OWN): own_schema,
        (neighbour, FRAME): neighbour_schema,
    }
    try:
        rows = {key: _row(manager, *key).version for key in kept}
        live = {name: _live_sd(ports, name) for name in kept.values()}
        assert "rank-profile acme_title_first" in live[own_schema]
        assert live[neighbour_schema].count(SHIPPED_FRAME_EXPRESSION) == 7
        migrator = connect(tenant)
        packages = []
        deploy = migrator.deploy_schemas

        def recording(schema_definitions, *args, **kwargs):
            packages.append(
                sorted(
                    schema["name"]
                    for schema in schema_definitions
                    if not schema.get("carried", False)
                )
            )
            return deploy(schema_definitions, *args, **kwargs)

        migrator.deploy_schemas = recording

        result = await asyncio.to_thread(
            migrator.schema_registry.redeploy_drifted_schemas
        )

        ours = {frame_schema, *kept.values()}
        assert [name for name in result.redeployed if name in ours] == [frame_schema]
        assert [package for package in packages if ours.intersection(package)] == [
            [frame_schema]
        ]
        frame_row = _row(manager, tenant, FRAME).config_value
        assert json.loads(frame_row["schema_definition"]) == _named(
            _shipped(FRAME), frame_schema
        )
        assert frame_row["config"] == {"profile": "acme_frames"}
        assert _live_sd(ports, frame_schema).count(SHIPPED_FRAME_EXPRESSION) == 7
        assert {key: _row(manager, *key).version for key in kept} == rows
        assert {name: _live_sd(ports, name) for name in kept.values()} == live
    finally:
        for base in (OWN, TEXT, FRAME):
            owner.schema_manager.delete_schema(tenant, base)
        neighbour_owner.schema_manager.delete_schema(neighbour, FRAME)


def _runtime_starting(http_port, config_port, barrier, outcomes):
    """One runtime process: builds the system backend's schema registry over
    the shared Vespa, waits for its peer, then runs the startup migration.
    Reports every package it activated and the migration's log records."""
    import os

    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_runtime import main as runtime_main
    from cogniverse_vespa.backend import VespaBackend
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(backend_url="http://127.0.0.1", backend_port=http_port)
    manager = ConfigManager(store=store)
    loader = FilesystemSchemaLoader(SCHEMAS_DIR)
    backend = VespaBackend(
        BackendConfig(
            backend_type="vespa",
            url="http://127.0.0.1",
            port=http_port,
            tenant_id="system",
        ),
        schema_loader=loader,
        config_manager=manager,
    )
    backend._initialize_backend({"config_port": config_port})
    backend.schema_registry = SchemaRegistry(manager, backend, loader)
    backend.schema_manager._schema_registry = backend.schema_registry
    activations = []
    deploy = backend.deploy_schemas

    def recording(schema_definitions, *args, **kwargs):
        requested = sorted(
            schema["name"]
            for schema in schema_definitions
            if not schema.get("carried", False)
        )
        activated = deploy(schema_definitions, *args, **kwargs)
        if activated:
            activations.append(requested)
        return activated

    backend.deploy_schemas = recording
    records = []

    class Recorder(logging.Handler):
        """Keeps each record with its exception arguments as text: the
        report crosses a process boundary, and an exception whose
        constructor takes extra arguments does not unpickle."""

        def emit(self, record):
            args = tuple(
                repr(arg) if isinstance(arg, BaseException) else arg
                for arg in record.args or ()
            )
            records.append((record.levelname, record.getMessage(), record.msg, args))

    runtime_main.logger.addHandler(Recorder())
    runtime_main.logger.setLevel(logging.INFO)
    barrier.wait(timeout=300)
    asyncio.run(runtime_main._migrate_drifted_schemas(lambda: backend.schema_registry))
    outcomes.put({"pid": os.getpid(), "activations": activations, "logs": records})
    backend.close()
    store.close()


def test_two_runtimes_starting_together_redeploy_each_drifted_tenant_once(
    migration_vespa,
):
    """Two runtime processes run the startup migration at the same moment.
    Each drifted tenant is redeployed exactly once, in one package carrying
    every drifted schema of that tenant; the other process finds it current.
    A change Vespa refuses is activated by neither: each process reports it,
    and one refusal is recorded for it."""
    connect, manager, ports = migration_vespa
    both = f"migtwice_{uuid.uuid4().hex[:10]}:both"
    frame_only = f"migtwice_{uuid.uuid4().hex[:10]}:frame"
    refused = f"migtwice_{uuid.uuid4().hex[:10]}:refused"
    older = _OlderRelease({FRAME: _older_rank_profiles, VISUAL: _older_rank_profiles})
    both_owner = connect(both, older)
    frame_only_owner = connect(frame_only, older)
    refused_owner = connect(refused, _OlderRelease({TEXT: _chunk_index_as_string}))
    both_schemas = both_owner.schema_registry.deploy_schemas(both, [FRAME, VISUAL])
    [frame_only_schema] = frame_only_owner.schema_registry.deploy_schemas(
        frame_only, [FRAME]
    )
    refused_schema = refused_owner.schema_registry.deploy_schema(refused, TEXT)
    refused_row = _row(manager, refused, TEXT)
    ours = {*both_schemas, frame_only_schema}
    spawn = multiprocessing.get_context("spawn")
    barrier = spawn.Barrier(2)
    outcomes = spawn.Queue()
    runtimes = [
        spawn.Process(
            target=_runtime_starting,
            args=(ports["http_port"], ports["config_port"], barrier, outcomes),
        )
        for _ in range(2)
    ]
    try:
        for runtime in runtimes:
            runtime.start()
        reports = []
        deadline = time.monotonic() + 1800
        while len(reports) < len(runtimes) and time.monotonic() < deadline:
            try:
                reports.append(outcomes.get(timeout=5))
            except queue.Empty:
                if not any(runtime.is_alive() for runtime in runtimes):
                    break
        for runtime in runtimes:
            runtime.join(timeout=120)

        assert [runtime.exitcode for runtime in runtimes] == [0, 0]
        activated = sorted(
            [name for name in package if name in ours]
            for report in reports
            for package in report["activations"]
            if ours.intersection(package)
        )
        assert activated == sorted([sorted(both_schemas), [frame_only_schema]])
        redeployed = [
            name
            for report in reports
            for _, _, template, args in report["logs"]
            if template == "Migration of drifted schemas redeployed %s"
            for name in args[0]
            if name in ours
        ]
        assert sorted(redeployed) == sorted(ours)
        assert [
            (level, message)
            for report in reports
            for level, message, _, _ in report["logs"]
            if level in ("WARNING", "ERROR") and any(name in message for name in ours)
        ] == []
        for tenant, base, name in (
            (both, FRAME, both_schemas[0]),
            (both, VISUAL, both_schemas[1]),
            (frame_only, FRAME, frame_only_schema),
        ):
            assert _registered_definition(manager, tenant, base) == _named(
                _shipped(base), name
            )

        assert [
            package
            for report in reports
            for package in report["activations"]
            if refused_schema in package
        ] == []
        reported = [
            [
                (level, message)
                for level, message, _, _ in report["logs"]
                if refused_schema in message
            ]
            for report in reports
        ]
        assert [[level for level, _ in lines] for lines in reported] == [
            ["ERROR"],
            ["ERROR"],
        ]
        prefix = (
            f"Migration of drifted schemas could not redeploy {refused_schema} for "
            f"tenant {refused}: "
        )
        errors = set()
        for [(_, message)] in reported:
            assert message.startswith(
                f"{prefix}Backend deployment failed for schema '{refused_schema}': "
                "Vespa refused the application package: Deployment failed with "
                "status 400: "
            ), message
            assert "Field 'chunk_index' changed: data type: 'string' -> 'int'" in (
                message
            )
            errors.add(message.removeprefix(prefix))
        [listed] = [
            entry
            for entry in drifted_schemas(manager, FilesystemSchemaLoader(SCHEMAS_DIR))
            if entry.tenant_id == refused
        ]
        assert listed.schema_name == refused_schema
        assert listed.refusal.error in errors
        [record] = [
            record
            for record in manager.store.list_configs(
                tenant_id=SYSTEM_TENANT_ID,
                scope=ConfigScope.SCHEMA,
                service=SCHEMA_REFUSALS_SERVICE,
            )
            if record.config_key == refused_schema
        ]
        assert record.version == 2
        assert _row(manager, refused, TEXT).version == refused_row.version
        assert "field chunk_index type string" in _live_sd(ports, refused_schema)
    finally:
        for runtime in runtimes:
            if runtime.is_alive():
                runtime.kill()
        both_owner.schema_manager.delete_schema(both, VISUAL)
        both_owner.schema_manager.delete_schema(both, FRAME)
        frame_only_owner.schema_manager.delete_schema(frame_only, FRAME)
        refused_owner.schema_manager.delete_schema(refused, TEXT)


class _ConfigServerOutage:
    """A port in front of the config server that refuses every connection
    until ``restore`` makes it forward to the real one.

    The port is bound but not listening while down, so a connect is refused
    at once, as it is when the config server process is down.
    """

    def __init__(self, upstream_port: int):
        self._upstream = upstream_port
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []

    def restore(self) -> None:
        self._listener.listen(64)
        self._spawn(self._accept)

    def close(self) -> None:
        self._closed.set()
        self._listener.close()

    def _spawn(self, target, *args) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self._upstream))
            self._spawn(self._pump, client, upstream)
            self._spawn(self._pump, upstream, client)

    @staticmethod
    def _pump(source: socket.socket, sink: socket.socket) -> None:
        try:
            while chunk := source.recv(65536):
                sink.sendall(chunk)
        except OSError:
            pass
        finally:
            for end in (source, sink):
                try:
                    end.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


@pytest.mark.asyncio
async def test_with_the_config_server_down_the_migration_retries_until_it_returns(
    migration_vespa, monkeypatch, caplog
):
    """The config server refuses connections when the startup migration
    begins: each attempt is logged and retried, the runtime's task keeps
    running, and no tenant is changed or recorded as refused. Once the config
    server answers, the next attempt redeploys the drifted tenant."""
    from cogniverse_runtime import main as runtime_main

    connect, manager, ports = migration_vespa
    tenant = f"migdown_{uuid.uuid4().hex[:10]}:acme"
    owner = connect(tenant, _OlderRelease({FRAME: _older_rank_profiles}))
    schema = owner.schema_registry.deploy_schema(tenant, FRAME)
    row = _row(manager, tenant, FRAME)
    outage = _ConfigServerOutage(ports["config_port"])
    registry = connect(
        "system", _OnlyShips(FRAME), config_port=outage.port
    ).schema_registry
    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.5)
    caplog.set_level("INFO", logger=runtime_main.logger.name)

    def warnings():
        return [
            record
            for record in caplog.records
            if record.name == runtime_main.logger.name and record.levelname == "WARNING"
        ]

    stop = threading.Event()
    migration = asyncio.create_task(
        runtime_main._migrate_drifted_schemas(lambda: registry, stop)
    )
    try:
        async with asyncio.timeout(300):
            while len(warnings()) < 2:
                await asyncio.sleep(0.1)

        assert migration.done() is False
        for record in warnings()[:2]:
            message = record.getMessage()
            assert message.startswith(
                "Migration of drifted schemas did not complete (BackendDeploymentError: "
                f"Backend deployment failed for schema '{schema}': Cannot enumerate "
                "Vespa-deployed schemas before deploy: "
            ), message
            assert "Connection refused" in message, message
            assert message.endswith("; retrying in 0s"), message
        assert _row(manager, tenant, FRAME).version == row.version
        assert [
            entry
            for entry in drifted_schemas(manager, FilesystemSchemaLoader(SCHEMAS_DIR))
            if entry.tenant_id == tenant
        ] == [
            DriftedSchema(
                tenant_id=tenant,
                base_schema_name=FRAME,
                schema_name=schema,
                refusal=None,
            )
        ]

        outage.restore()
        await asyncio.wait_for(migration, timeout=900)

        assert migration.result() is None
        assert _registered_definition(manager, tenant, FRAME) == _named(
            _shipped(FRAME), schema
        )
        assert _live_sd(ports, schema).count(SHIPPED_FRAME_EXPRESSION) == 7
        assert [
            (record.levelname, record.getMessage())
            for record in caplog.records
            if record.name == runtime_main.logger.name
            and record.levelname in ("INFO", "ERROR")
        ] == [("INFO", f"Migration of drifted schemas redeployed ['{schema}']")]
    finally:
        stop.set()
        migration.cancel()
        await asyncio.gather(migration, return_exceptions=True)
        outage.close()
        owner.schema_manager.delete_schema(tenant, FRAME)


def _proxied_manager(port: int):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    return ConfigManager(
        store=VespaConfigStore(backend_url="http://127.0.0.1", backend_port=port)
    )


def _touches_refusals(path: str) -> bool:
    return "schema_migration_refusals" in unquote(path)


@pytest.mark.asyncio
async def test_a_refusal_that_cannot_be_recorded_is_retried_until_it_is(
    migration_vespa, monkeypatch, caplog
):
    """Vespa refuses a tenant's redeploy and the config store refuses the
    record of that refusal: the run does not complete, nothing is reported as
    refused, and the next run, once the store takes the write, is refused
    again and records it."""
    from cogniverse_runtime import main as runtime_main

    connect, manager, ports = migration_vespa
    tenant = f"migrecord_{uuid.uuid4().hex[:10]}:acme"
    owner = connect(tenant, _OlderRelease({TEXT: _chunk_index_as_string}))
    schema = owner.schema_registry.deploy_schema(tenant, TEXT)
    row = _row(manager, tenant, TEXT)
    store_down = threading.Event()
    store_down.set()

    def refuse_refusal_writes(method, path, body):
        if store_down.is_set() and method == "POST" and _touches_refusals(path):
            return 500, {"message": "injected storage failure"}
        return None

    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.5)
    caplog.set_level("INFO", logger=runtime_main.logger.name)

    def ours(level):
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == runtime_main.logger.name and record.levelname == level
        ]

    stop = threading.Event()
    migration = None
    try:
        with InterceptFaultProxy(
            f"http://127.0.0.1:{ports['http_port']}", refuse_refusal_writes
        ) as proxy:
            proxied = _proxied_manager(proxy.port)
            registry = connect(
                "system", _OnlyShips(TEXT), store_manager=proxied
            ).schema_registry
            migration = asyncio.create_task(
                runtime_main._migrate_drifted_schemas(lambda: registry, stop)
            )
            async with asyncio.timeout(600):
                while not ours("WARNING"):
                    await asyncio.sleep(0.1)

            first = ours("WARNING")[0]
            assert first.startswith(
                "Migration of drifted schemas did not complete (RegistryStorageError: "
                f"Cannot record the refused redeploy of '{schema}': "
            ), first
            assert first.endswith("; retrying in 0s"), first
            assert migration.done() is False
            assert ours("ERROR") == []
            assert [
                entry
                for entry in drifted_schemas(
                    manager, FilesystemSchemaLoader(SCHEMAS_DIR)
                )
                if entry.tenant_id == tenant
            ] == [
                DriftedSchema(
                    tenant_id=tenant,
                    base_schema_name=TEXT,
                    schema_name=schema,
                    refusal=None,
                )
            ]

            store_down.clear()
            await asyncio.wait_for(migration, timeout=900)
            proxied.store.close()

        [refused] = ours("ERROR")
        assert refused.startswith(
            f"Migration of drifted schemas could not redeploy {schema} for tenant "
            f"{tenant}: Backend deployment failed for schema '{schema}': Vespa "
            "refused the application package: Deployment failed with status 400: "
        ), refused
        [listed] = [
            entry
            for entry in drifted_schemas(manager, FilesystemSchemaLoader(SCHEMAS_DIR))
            if entry.tenant_id == tenant
        ]
        assert listed.refusal is not None
        assert (listed.schema_name, listed.refusal.error) == (
            schema,
            refused.split(f"for tenant {tenant}: ", 1)[1],
        )
        assert _row(manager, tenant, TEXT).version == row.version
        assert "field chunk_index type string" in _live_sd(ports, schema)
    finally:
        stop.set()
        if migration is not None:
            migration.cancel()
            await asyncio.gather(migration, return_exceptions=True)
        owner.schema_manager.delete_schema(tenant, TEXT)


@pytest.mark.parametrize("unreadable", ["registry", "refusals"])
@pytest.mark.asyncio
async def test_the_drift_listing_answers_503_when_its_store_cannot_be_read(
    migration_vespa, unreadable, caplog
):
    """A store that cannot be read answers a typed 503 naming the failure's
    type, never an empty listing that reads as every tenant being current.
    The cause goes to the runtime log, not the body."""
    _connect, _manager, ports = migration_vespa
    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")

    def refuse(method, path, body):
        if method == "GET" and path.startswith("/document/v1/"):
            if unreadable == "registry" and "schema_registry" in unquote(path):
                return 503, {"message": "injected outage"}
            if unreadable == "refusals" and _touches_refusals(path):
                return 503, {"message": "injected outage"}
        return None

    with InterceptFaultProxy(f"http://127.0.0.1:{ports['http_port']}", refuse) as proxy:
        proxied = _proxied_manager(proxy.port)
        try:
            response = await _drift_listing(proxied)
        finally:
            proxied.store.close()

    failure, cause = {
        "registry": (
            "SchemaRegistryInitializationError",
            "Cannot initialize SchemaRegistry: failed to read schema storage: ",
        ),
        "refusals": (
            "RegistryStorageError",
            "Cannot read schema migration refusals: ",
        ),
    }[unreadable]
    assert (response.status_code, response.json()) == (
        503,
        {
            "detail": {
                "error": "schema_drift_unavailable",
                "message": "The schema registry or the recorded migration "
                "refusals could not be read; retry.",
                "failure": failure,
            }
        },
    )
    [logged] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ]
    assert logged.startswith(f"schema_drift_unavailable: {failure}: {cause}"), logged


def _schema_registry_errors(caplog, names) -> list[str]:
    from cogniverse_core.registries import schema_registry as schema_registry_module

    return [
        record.getMessage()
        for record in caplog.records
        if record.name == schema_registry_module.logger.name
        and record.levelno >= logging.ERROR
        and any(name in record.getMessage() for name in names)
    ]


@pytest.mark.asyncio
async def test_a_refusal_is_removed_once_its_schema_migrates_or_is_dropped(
    migration_vespa, caplog
):
    """Three tenants each carry a schema whose redeploy Vespa refused, and the
    first run records the three refusals. Before the next run one tenant's
    schema is dropped, and the next release ships a change Vespa accepts for
    another's. The next run migrates that schema and removes both stale
    refusals; the third tenant's schema is refused again and its refusal
    stays."""
    connect, manager, _ports = migration_vespa
    run = uuid.uuid4().hex[:8]
    own_text, own_visual = _own_bases(run)
    migrates, dropped, refused = (
        f"migclean_{run}:{name}" for name in ("migrates", "dropped", "refused")
    )
    registered = _registering_release(run)
    owners = {
        tenant: connect(tenant, registered) for tenant in (migrates, dropped, refused)
    }
    schemas = {
        migrates: owners[migrates].schema_registry.deploy_schema(migrates, own_text),
        dropped: owners[dropped].schema_registry.deploy_schema(dropped, own_text),
        refused: owners[refused].schema_registry.deploy_schema(refused, own_visual),
    }
    ours = set(schemas.values())
    accepting = _OwnRelease(
        {
            own_text: _titled_string_chunk_index_text,
            own_visual: lambda: _shipped(VISUAL),
        }
    )
    caplog.set_level(logging.INFO)
    try:
        first = await asyncio.to_thread(
            connect(
                "system", _refusing_release(run)
            ).schema_registry.redeploy_drifted_schemas
        )
        assert sorted(
            refusal.schema_name
            for refusal in first.refused
            if refusal.schema_name in ours
        ) == sorted(ours)
        assert _recorded_refusals(manager, ours) == {
            schemas[migrates]: (
                migrates,
                _digest(_named(_shipped(TEXT), schemas[migrates])),
            ),
            schemas[dropped]: (
                dropped,
                _digest(_named(_shipped(TEXT), schemas[dropped])),
            ),
            schemas[refused]: (
                refused,
                _digest(_named(_shipped(VISUAL), schemas[refused])),
            ),
        }
        owners[dropped].schema_manager.delete_schema(dropped, own_text)
        assert schemas[dropped] in _recorded_refusals(manager, ours)

        second = await asyncio.to_thread(
            connect("system", accepting).schema_registry.redeploy_drifted_schemas
        )

        assert [name for name in second.redeployed if name in ours] == [
            schemas[migrates]
        ]
        assert [
            refusal.schema_name
            for refusal in second.refused
            if refusal.schema_name in ours
        ] == [schemas[refused]]
        assert _registered_definition(manager, migrates, own_text) == _named(
            _titled_string_chunk_index_text(), schemas[migrates]
        )
        assert _recorded_refusals(manager, ours) == {
            schemas[refused]: (
                refused,
                _digest(_named(_shipped(VISUAL), schemas[refused])),
            )
        }
        assert _schema_registry_errors(caplog, ours) == []
    finally:
        owners[migrates].schema_manager.delete_schema(migrates, own_text)
        owners[refused].schema_manager.delete_schema(refused, own_visual)


def _refusing_deletes_of_refusals(method: str, path: str, _body: bytes):
    if method == "DELETE" and _touches_refusals(path):
        return 500, {"message": "injected storage failure"}
    return None


def _assert_cleanup_failure_logged(caplog, tenant: str, schema: str) -> None:
    [logged] = _schema_registry_errors(caplog, [schema])
    assert logged.startswith(
        f"Cannot delete the recorded migration refusal of '{schema}' for tenant "
        f"'{tenant}' (RuntimeError: Failed to delete 1 of 1 versions for "
    ), logged
    assert logged.endswith("; the next migration removes it"), logged


@pytest.mark.asyncio
async def test_a_refusal_that_cannot_be_removed_is_left_for_the_next_run(
    migration_vespa, caplog
):
    """The config store refuses to delete the refusal of a schema the run has
    just migrated. The run still completes and reports the migration, the
    failure is logged at ERROR by tenant and schema, and the refusal stays
    until the next run, which removes it."""
    connect, manager, ports = migration_vespa
    run = uuid.uuid4().hex[:8]
    own_text, _ = _own_bases(run)
    tenant = f"migclean_{run}:acme"
    owner = connect(tenant, _registering_release(run))
    schema = owner.schema_registry.deploy_schema(tenant, own_text)
    accepting = _OwnRelease({own_text: _titled_string_chunk_index_text})
    recorded = {schema: (tenant, _digest(_named(_shipped(TEXT), schema)))}
    caplog.set_level(logging.INFO)
    try:
        await asyncio.to_thread(
            connect(
                "system", _refusing_release(run)
            ).schema_registry.redeploy_drifted_schemas
        )
        assert _recorded_refusals(manager, {schema}) == recorded

        with InterceptFaultProxy(
            f"http://127.0.0.1:{ports['http_port']}", _refusing_deletes_of_refusals
        ) as proxy:
            proxied = _proxied_manager(proxy.port)
            try:
                migrated = await asyncio.to_thread(
                    connect(
                        "system", accepting, store_manager=proxied
                    ).schema_registry.redeploy_drifted_schemas
                )
            finally:
                proxied.store.close()

        assert [name for name in migrated.redeployed if name == schema] == [schema]
        assert _registered_definition(manager, tenant, own_text) == _named(
            _titled_string_chunk_index_text(), schema
        )
        _assert_cleanup_failure_logged(caplog, tenant, schema)
        assert _recorded_refusals(manager, {schema}) == recorded

        caplog.clear()
        after = await asyncio.to_thread(
            connect("system", accepting).schema_registry.redeploy_drifted_schemas
        )
        assert [name for name in after.redeployed if name == schema] == []
        assert _recorded_refusals(manager, {schema}) == {}
        assert _schema_registry_errors(caplog, [schema]) == []
    finally:
        owner.schema_manager.delete_schema(tenant, own_text)


@pytest.fixture
async def wire_tenant_manager(workflow_state_redis_url, shared_state_redis):
    """``wire(config_manager, schema_loader)``: tenant_manager deleting through
    that store and loader, with this process as the one runtime worker on its
    own cluster-events channel and a task event store of its own. Returns the
    worker id; module seams restored after."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_runtime.admin import tenant_manager as tm
    from cogniverse_runtime.cluster_events import ClusterEvents
    from cogniverse_runtime.task_events import TaskEventStore

    events = ClusterEvents(
        workflow_state_redis_url,
        f"migration-worker-{uuid.uuid4().hex[:6]}",
        {"tenant_deleted": tm.release_deleted_tenant},
        channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
    )
    await events.start()
    prefix = f"test:task-events:{uuid.uuid4().hex}"
    task_events = TaskEventStore(
        shared_state_redis,
        key_prefix=prefix,
        ingestion_stream_prefix=f"{prefix}:ingest:",
    )
    previous = (
        tm._config_manager,
        tm._schema_loader,
        tm._cluster_events,
        tm._task_events,
    )

    def wire(config_manager, schema_loader):
        # A backend is bound to the store it was built with.
        BackendRegistry.get_instance().clear_instances()
        tm.set_config_manager(config_manager)
        tm.set_schema_loader(schema_loader)
        tm.set_cluster_events(events)
        tm.set_task_event_store(task_events)
        return events.worker_id

    yield wire
    tm.set_config_manager(previous[0])
    tm.set_schema_loader(previous[1])
    tm.set_cluster_events(previous[2])
    tm.set_task_event_store(previous[3])
    BackendRegistry.get_instance().clear_instances()
    await events.close()


@pytest.mark.asyncio
async def test_a_tenant_delete_whose_refusal_cleanup_fails_still_completes(
    migration_vespa, wire_tenant_manager, caplog
):
    """The config store refuses to delete a deleted tenant's recorded
    refusal: the delete still answers exactly as it does without a refusal,
    the failure is logged at ERROR by tenant and schema, and the next
    migration run removes the refusal of the dropped schema."""
    from cogniverse_runtime.admin import tenant_manager as tm

    connect, manager, ports = migration_vespa
    run = uuid.uuid4().hex[:8]
    own_text, _ = _own_bases(run)
    tenant = f"migclean_{run}:acme"
    registered = _registering_release(run)
    owner = connect(tenant, registered)
    schema = owner.schema_registry.deploy_schema(tenant, own_text)
    recorded = {schema: (tenant, _digest(_named(_shipped(TEXT), schema)))}
    caplog.set_level(logging.INFO)
    await asyncio.to_thread(
        connect(
            "system", _refusing_release(run)
        ).schema_registry.redeploy_drifted_schemas
    )
    assert _recorded_refusals(manager, {schema}) == recorded

    with InterceptFaultProxy(
        f"http://127.0.0.1:{ports['http_port']}", _refusing_deletes_of_refusals
    ) as proxy:
        proxied = _proxied_manager(proxy.port)
        worker = wire_tenant_manager(proxied, registered)
        try:
            result = await tm.delete_tenant_internal(tenant)
        finally:
            proxied.store.close()

    assert result == {
        "status": "deleted",
        "tenant_full_id": tenant,
        "schemas_deleted": 1,
        "deleted_schemas": [schema],
        "organization_deleted": False,
        "workers_released": [worker],
    }
    _assert_cleanup_failure_logged(caplog, tenant, schema)
    assert _recorded_refusals(manager, {schema}) == recorded

    caplog.clear()
    await asyncio.to_thread(connect("system").schema_registry.redeploy_drifted_schemas)
    assert _recorded_refusals(manager, {schema}) == {}
    assert _schema_registry_errors(caplog, [schema]) == []


def _runtime_migrating_while_deleted(
    http_port, config_port, run, doomed, barrier, refused, outcomes
):
    """One runtime process running the startup migration under the release
    that refuses this run's own schemas. Its refusal of the doomed tenant's
    schema is announced on ``refused`` and recorded only once the tenant's
    delete has marked it, so the record lands after that delete began.
    Reports how many such refusals it held and its migration log records."""
    import os

    from cogniverse_core.common.tenant_utils import (
        canonical_tenant_id,
        tenant_is_deleted,
    )
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_runtime import main as runtime_main
    from cogniverse_vespa.backend import VespaBackend
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(backend_url="http://127.0.0.1", backend_port=http_port)
    manager = ConfigManager(store=store)
    loader = _refusing_release(run)
    backend = VespaBackend(
        BackendConfig(
            backend_type="vespa",
            url="http://127.0.0.1",
            port=http_port,
            tenant_id="system",
        ),
        schema_loader=loader,
        config_manager=manager,
    )
    backend._initialize_backend({"config_port": config_port})
    backend.schema_registry = SchemaRegistry(manager, backend, loader)
    backend.schema_manager._schema_registry = backend.schema_registry
    held = []
    record = SchemaRegistry._record_refusal

    def recorded_once_marked(self, tenant_id, base_schema_name, shipped, exc):
        if canonical_tenant_id(tenant_id) == doomed:
            held.append(base_schema_name)
            refused.set()
            deadline = time.monotonic() + 300
            while not tenant_is_deleted(store, doomed):
                if time.monotonic() > deadline:
                    raise AssertionError(f"{doomed} was never marked deleted")
                time.sleep(0.1)
        return record(self, tenant_id, base_schema_name, shipped, exc)

    SchemaRegistry._record_refusal = recorded_once_marked
    records = []

    class Recorder(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    from cogniverse_core.registries import schema_registry as schema_registry_module

    recorder = Recorder(level=logging.ERROR)
    schema_registry_module.logger.addHandler(recorder)
    runtime_main.logger.addHandler(Recorder())
    runtime_main.logger.setLevel(logging.INFO)
    barrier.wait(timeout=300)
    asyncio.run(runtime_main._migrate_drifted_schemas(lambda: backend.schema_registry))
    outcomes.put({"pid": os.getpid(), "held": held, "logs": records})
    backend.close()
    store.close()


@pytest.mark.asyncio
async def test_two_runtimes_migrating_while_a_tenant_is_deleted_leave_no_refusal_of_it(
    migration_vespa, wire_tenant_manager, caplog
):
    """Two runtime processes run the startup migration together while one
    tenant is deleted. Vespa refuses the doomed tenant's schema, and that
    refusal is recorded only after the delete has marked the tenant. Once
    both runs and the delete complete, no refusal of the doomed tenant is
    left, nothing failed, and the other tenant's refusal stands."""
    from cogniverse_runtime.admin import tenant_manager as tm

    connect, manager, ports = migration_vespa
    run = uuid.uuid4().hex[:8]
    own_text, own_visual = _own_bases(run)
    doomed, kept = f"migdel_{run}:doomed", f"migdel_{run}:kept"
    registered = _registering_release(run)
    doomed_owner = connect(doomed, registered)
    kept_owner = connect(kept, registered)
    doomed_schema = doomed_owner.schema_registry.deploy_schema(doomed, own_text)
    kept_schema = kept_owner.schema_registry.deploy_schema(kept, own_visual)
    ours = {doomed_schema, kept_schema}
    worker = wire_tenant_manager(manager, registered)
    caplog.set_level(logging.INFO)
    spawn = multiprocessing.get_context("spawn")
    barrier = spawn.Barrier(2)
    refused = spawn.Event()
    outcomes = spawn.Queue()
    runtimes = [
        spawn.Process(
            target=_runtime_migrating_while_deleted,
            args=(
                ports["http_port"],
                ports["config_port"],
                run,
                doomed,
                barrier,
                refused,
                outcomes,
            ),
        )
        for _ in range(2)
    ]
    result = None
    try:
        for runtime in runtimes:
            runtime.start()
        assert await asyncio.to_thread(refused.wait, 900) is True

        result = await tm.delete_tenant_internal(doomed)

        reports = [await asyncio.to_thread(outcomes.get, True, 900) for _ in runtimes]
        for runtime in runtimes:
            await asyncio.to_thread(runtime.join, 120)

        assert [runtime.exitcode for runtime in runtimes] == [0, 0]
        assert result == {
            "status": "deleted",
            "tenant_full_id": doomed,
            "schemas_deleted": 1,
            "deleted_schemas": [doomed_schema],
            "organization_deleted": False,
            "workers_released": [worker],
        }
        assert sorted(report["held"] for report in reports) == [[], [own_text]]
        assert _recorded_refusals(manager, ours) == {
            kept_schema: (kept, _digest(_named(_shipped(VISUAL), kept_schema)))
        }
        prefix = "Migration of drifted schemas could not redeploy "
        assert sorted(
            message.split(" for tenant ", 1)[0].removeprefix(prefix)
            for report in reports
            for level, message in report["logs"]
            if level in ("WARNING", "ERROR") and any(name in message for name in ours)
        ) == sorted([doomed_schema, kept_schema, kept_schema])
        assert all(
            message.startswith(prefix)
            for report in reports
            for level, message in report["logs"]
            if level in ("WARNING", "ERROR") and any(name in message for name in ours)
        )
        assert _schema_registry_errors(caplog, ours) == []
    finally:
        for runtime in runtimes:
            if runtime.is_alive():
                runtime.kill()
        if result is None:
            await tm.delete_tenant_internal(doomed)
        kept_owner.schema_manager.delete_schema(kept, own_visual)
