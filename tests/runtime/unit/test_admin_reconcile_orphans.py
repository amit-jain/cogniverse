"""Unit tests for the ``/admin/reconcile-orphans`` endpoint.

The endpoint diffs Vespa-deployed schemas against the SchemaRegistry's
active set (registry-orphans) and against the tenant_metadata registry
(tenant-orphans), and either reports them (dry_run) or drops them in a
Vespa redeploy (confirm). These tests pin the route's own contract:
status codes, refusal messages, and which primitive is called with what.
The removal and its readback run against real Vespa in
tests/runtime/integration/test_reconcile_tenant_orphans.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.admin import tenant_manager
from tests.utils.memory_store import InMemoryConfigStore

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
SHIPPED_CONFIG = REPO_ROOT / "configs" / "config.json"


@pytest.fixture
def admin_client():
    """TestClient mounting the tenant_manager router with a mock backend.

    The schema loader is the real filesystem loader over the shipped
    ``configs/schemas``, so orphan attribution runs against the base
    schema set the platform actually ships.
    """
    app = FastAPI()
    app.include_router(tenant_manager.router, prefix="/admin")

    previous_loader = tenant_manager._schema_loader
    tenant_manager.set_schema_loader(FilesystemSchemaLoader(SHIPPED_SCHEMAS_DIR))
    # The system config names the application, and so its phantom type.
    previous_config_manager = tenant_manager._config_manager
    tenant_manager.set_config_manager(ConfigManager(store=InMemoryConfigStore()))

    backend = MagicMock()
    schema_manager = MagicMock()
    schema_registry = MagicMock()
    backend.schema_manager = schema_manager
    schema_manager._schema_registry = schema_registry
    schema_registry.reserved_schemas.return_value = {}
    # The tenant registry the reconciler diffs registered schemas against.
    # ``legit`` is the live tenant every test below registers a schema for,
    # so its schema is never a tenant-orphan.
    backend.query_metadata_documents.return_value = [{"tenant_full_id": "legit:legit"}]
    schema_manager._PROTECTED_SCHEMAS = frozenset(
        {
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
        }
    )
    previous_get_backend = tenant_manager.get_backend
    tenant_manager.get_backend = lambda: backend

    yield TestClient(app), backend, schema_manager, schema_registry

    tenant_manager.get_backend = previous_get_backend
    tenant_manager.set_schema_loader(previous_loader)
    tenant_manager.set_config_manager(previous_config_manager)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileOrphansDryRun:
    def test_dry_run_returns_orphan_diff_without_dropping(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()

        assert data["dry_run"] is True
        assert sorted(data["orphan_schemas"]) == [
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
        ]
        assert sorted(data["orphan_tenants"]) == ["alpha", "beta"]
        assert data["deleted"] == []
        # Crucial: the bulk delete was NOT called.
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_dry_run_with_clean_cluster_returns_empty(self, admin_client):
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
        ]
        schema_registry._get_all_schemas.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orphan_schemas"] == []
        assert data["orphan_tenants"] == []
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_unknown_base_prefix_listed_separately(self, admin_client):
        """Schemas whose base prefix is not a shipped schema are reported
        under ``unrecovered_schemas`` so the operator can review them
        rather than silently treated as no-op.
        """
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "config_metadata",
            "organization_metadata",
            "adapter_registry",
            "knowledge_graph_legit",
            "weird_custom_schema_acme",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "weird_custom_schema_acme" in data["orphan_schemas"]
        assert "weird_custom_schema_acme" in data["unrecovered_schemas"]
        assert data["orphan_tenants"] == []

    def test_attribution_matches_longest_base_first(self, admin_client, tmp_path):
        """When one shipped base is a prefix of another, the LONGER one wins.

        First-match-wins on the shorter ``document_text`` base would strip a
        ``document_text_semantic_<tid>`` orphan to the bogus tenant
        ``semantic_<tid>``, deleting a schema attributed to a tenant that
        never existed. No shipped pair is currently a prefix of another, so
        the rule is driven through a loader carrying the prefix pair.
        """
        client, _, schema_manager, schema_registry = admin_client

        for base in ("document_text", "document_text_semantic"):
            (tmp_path / f"{base}_schema.json").write_text(json.dumps({"name": base}))
        tenant_manager.set_schema_loader(FilesystemSchemaLoader(tmp_path))

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "config_metadata",
            "organization_metadata",
            "adapter_registry",
            "document_text_semantic_acme_acme",
            "document_text_beta",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orphan_tenants"] == ["acme_acme", "beta"]
        assert data["unrecovered_schemas"] == []

    def test_every_profile_schema_is_attributable(self, admin_client):
        """An orphan of any profile's ``schema_name`` must attribute to its tenant.

        Cross-checks the shipped profiles against the shipped schema files:
        a profile whose schema file is missing yields an unattributable
        orphan, which is never a deletion target and blocks every tenant
        delete once the redeploy refuses unresolved survivors.
        """
        client, _, schema_manager, schema_registry = admin_client

        profiles = json.loads(SHIPPED_CONFIG.read_text())["backend"]["profiles"]
        schema_names = sorted(
            {p["schema_name"] for p in profiles.values() if p.get("schema_name")}
        )
        assert len(schema_names) == len(profiles), (
            "expected one schema per shipped profile",
            len(schema_names),
            len(profiles),
        )

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            *[f"{name}_pt_pt" for name in schema_names],
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unrecovered_schemas"] == []
        assert data["orphan_tenants"] == ["pt_pt"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileOrphansBaseSchemaSource:
    def test_uninitialized_schema_loader_refuses_reconcile(self, admin_client):
        """Without a loader the base set is unknown, so every orphan would read
        as unrecoverable — blocking tenant deletes rather than reporting the
        real cause. Refuse loudly instead."""
        client, _, schema_manager, schema_registry = admin_client
        tenant_manager.set_schema_loader(None)

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "knowledge_graph_alpha",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        assert "SchemaLoader not initialized" in resp.json()["detail"]
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_empty_shipped_schema_set_refuses_reconcile(self, admin_client, tmp_path):
        """An empty schema directory must not read as "no known bases" and
        silently strand every orphan."""
        client, _, schema_manager, schema_registry = admin_client
        tenant_manager.set_schema_loader(FilesystemSchemaLoader(tmp_path))

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "knowledge_graph_alpha",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        assert "no shipped schemas" in resp.json()["detail"]
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_newly_shipped_base_is_attributable_without_editing_the_module(
        self, admin_client, tmp_path
    ):
        """Adding a schema file is enough to make its orphans attributable."""
        client, _, schema_manager, schema_registry = admin_client

        (tmp_path / "video_brand_new_sv_schema.json").write_text(
            json.dumps({"name": "video_brand_new_sv"})
        )
        tenant_manager.set_schema_loader(FilesystemSchemaLoader(tmp_path))

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "video_brand_new_sv_acme",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orphan_tenants"] == ["acme"]
        assert data["unrecovered_schemas"] == []


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileOrphansConfirm:
    def test_confirm_calls_bulk_delete_with_orphan_tenants(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
            "knowledge_graph_legit",
        ]
        # A realistic orphan scenario: the registry HAS active schemas; alpha
        # and beta are the ones missing from it. (An empty registry with
        # deployed schemas is the failed-load case the safety guard blocks.)
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]
        schema_manager.delete_orphan_schemas.return_value = [
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
        ]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()

        assert data["dry_run"] is False
        schema_manager.delete_orphan_schemas.assert_called_once_with(
            [
                "knowledge_graph_alpha",
                "video_colpali_smol500_mv_frame_beta",
            ]
        )
        assert sorted(data["deleted"]) == [
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
        ]

    def test_confirm_drops_an_orphan_no_shipped_base_attributes(self, admin_client):
        """The delete follows the orphans found, not only those a base prefix
        attributes to a tenant."""
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_legit",
            "weird_custom_schema_acme",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]
        schema_manager.delete_orphan_schemas.return_value = ["weird_custom_schema_acme"]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")

        assert resp.status_code == 200
        data = resp.json()
        assert (data["orphan_tenants"], data["unrecovered_schemas"]) == (
            [],
            ["weird_custom_schema_acme"],
        )
        schema_manager.delete_orphan_schemas.assert_called_once_with(
            ["weird_custom_schema_acme"]
        )
        assert data["deleted"] == ["weird_custom_schema_acme"]

    def test_confirm_removes_the_phantom_application_type_on_an_empty_registry(
        self, admin_client
    ):
        """A package deployed with no schemas carried pyvespa's default
        document type, named after the application; nothing registers it, and
        while it is live every deploy is refused. A cluster left that way
        usually has an empty registry too."""
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "cogniverse",
        ]
        schema_registry._get_all_schemas.return_value = []
        schema_manager.delete_orphan_schemas.return_value = ["cogniverse"]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")

        assert resp.status_code == 200
        data = resp.json()
        assert (data["orphan_schemas"], data["unrecovered_schemas"]) == (
            ["cogniverse"],
            ["cogniverse"],
        )
        schema_manager.delete_orphan_schemas.assert_called_once_with(["cogniverse"])
        assert data["deleted"] == ["cogniverse"]

    def test_confirm_with_no_orphans_does_not_call_bulk_delete(self, admin_client):
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
        ]
        schema_registry._get_all_schemas.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == []
        schema_manager.delete_orphan_schemas.assert_not_called()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileOrphansSafetyGuard:
    def test_empty_registry_with_deployed_schemas_refuses_reconcile(self, admin_client):
        """A cold pod whose registry failed to load from storage reads as an
        EMPTY registry — every deployed schema then looks orphaned. Reconciling
        would bulk-delete every tenant's schema, so it must refuse (503) rather
        than mass-delete on an unconfirmed registry."""
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
        ]
        # Registry loaded empty (storage read failed) while Vespa has schemas.
        schema_registry._get_all_schemas.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        # And crucially, nothing was deleted.
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_a_failed_registry_refresh_refuses_instead_of_reading_the_cache(
        self, admin_client
    ):
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
        ]
        schema_registry._get_all_schemas.side_effect = ConnectionError(
            "config store refused the read"
        )

        resp = client.post("/admin/reconcile-orphans?dry_run=false")

        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "Cannot read the schema registry during reconciliation: config store "
            "refused the read"
        )
        schema_registry._get_all_schemas.assert_called_once_with(strict=True)
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_a_failed_system_config_read_refuses_with_503(
        self, admin_client, monkeypatch
    ):
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "cogniverse",
        ]
        schema_registry._get_all_schemas.return_value = []

        def unreadable():
            raise ConnectionError("system config unreadable")

        monkeypatch.setattr(
            tenant_manager._config_manager, "get_system_config", unreadable
        )

        resp = client.post("/admin/reconcile-orphans?dry_run=false")

        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "Cannot read the system config naming the application during "
            "reconciliation: system config unreadable"
        )
        schema_manager.delete_orphan_schemas.assert_not_called()

    def test_the_phantom_type_does_not_open_the_guard_for_other_schemas(
        self, admin_client
    ):
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "cogniverse",
            "knowledge_graph_alpha",
        ]
        schema_registry._get_all_schemas.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=false")

        assert resp.status_code == 503
        assert resp.json()["detail"].startswith(
            "Schema registry is empty while Vespa has deployed schemas"
        )
        schema_manager.delete_orphan_schemas.assert_not_called()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileOrphansInFlightDeploys:
    def test_reserved_schema_is_not_an_orphan(self, admin_client):
        """A schema another process activated and has not registered yet is
        mid-deploy, not an orphan: the reconciler must leave it out of the diff
        and out of the bulk delete."""
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
            "knowledge_graph_inflight",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]
        schema_registry.reserved_schemas.return_value = {
            "knowledge_graph_inflight": {"full_schema_name": "knowledge_graph_inflight"}
        }
        schema_manager.delete_orphan_schemas.return_value = ["knowledge_graph_alpha"]

        preview = client.post("/admin/reconcile-orphans?dry_run=true").json()
        assert preview["orphan_schemas"] == ["knowledge_graph_alpha"]
        assert preview["orphan_tenants"] == ["alpha"]
        schema_registry.reserved_schemas.assert_called_with(
            set(schema_manager.list_deployed_document_types.return_value)
        )

        confirmed = client.post("/admin/reconcile-orphans?dry_run=false").json()
        assert confirmed["deleted"] == ["knowledge_graph_alpha"]
        schema_manager.delete_orphan_schemas.assert_called_once_with(
            ["knowledge_graph_alpha"]
        )

    def test_intent_journal_outage_refuses_to_reconcile(self, admin_client):
        """When the deployment-intent journal cannot be read, the reconciler
        cannot tell an orphan from a mid-deploy schema; it must refuse with the
        journal's error, never treat "no intents readable" as "no deploys in
        flight" and proceed to delete."""
        from cogniverse_core.registries.exceptions import RegistryStorageError

        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "knowledge_graph_alpha",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        legit.tenant_id = "legit"
        schema_registry._get_all_schemas.return_value = [legit]
        schema_registry.reserved_schemas.side_effect = RegistryStorageError(
            "Cannot read deployment intents: journal unavailable"
        )

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        assert resp.json() == {
            "detail": (
                "Cannot read schema deployment intents; refusing to reconcile "
                "orphans because a mid-deploy schema would be indistinguishable "
                "from an orphan: Cannot read deployment intents: journal "
                "unavailable"
            )
        }
        schema_manager.delete_orphan_schemas.assert_not_called()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReconcileTenantOrphans:
    """A schema whose registry row names a tenant with no tenant_metadata
    record is invisible to the registry diff: it is registered, so
    ``orphan_schemas`` never lists it, and every application package
    carries it forever."""

    def test_registered_schema_of_deleted_tenant_is_a_tenant_orphan(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
            "agent_memories_legit_legit",
            "agent_memories_ghost_t1",
            "provenance_ghost_t1",
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        ghost_memories = MagicMock()
        ghost_memories.full_schema_name = "agent_memories_ghost_t1"
        ghost_memories.tenant_id = "ghost:t1"
        ghost_provenance = MagicMock()
        ghost_provenance.full_schema_name = "provenance_ghost_t1"
        ghost_provenance.tenant_id = "ghost:t1"
        schema_registry._get_all_schemas.return_value = [
            live,
            ghost_memories,
            ghost_provenance,
        ]

        data = client.post("/admin/reconcile-orphans?dry_run=true").json()

        assert data["tenant_orphan_schemas"] == [
            "agent_memories_ghost_t1",
            "provenance_ghost_t1",
        ]
        assert data["tenant_orphan_tenants"] == ["ghost:t1"]
        assert data["tenant_orphans_deleted"] == []
        # The defect this class exists for: the registry diff cannot see them.
        assert data["orphan_schemas"] == []
        assert data["orphan_tenants"] == []
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

    def test_owner_comes_from_the_registry_row_not_the_schema_name(self, admin_client):
        """``agent_memories_a_b_c`` is ambiguous by name (a:b_c or a_b:c).
        The registry row names the owner, so the split is never guessed."""
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_a_b_c",
        ]
        row = MagicMock()
        row.full_schema_name = "agent_memories_a_b_c"
        row.tenant_id = "a_b:c"
        schema_registry._get_all_schemas.return_value = [row]
        backend.query_metadata_documents.return_value = [{"tenant_full_id": "a:b_c"}]

        data = client.post("/admin/reconcile-orphans?dry_run=true").json()

        assert data["tenant_orphan_tenants"] == ["a_b:c"]

    def test_registered_schema_not_deployed_is_not_a_tenant_orphan(self, admin_client):
        """Only a schema Vespa actually carries costs a slot in the package."""
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_legit_legit",
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        stale_row = MagicMock()
        stale_row.full_schema_name = "agent_memories_ghost_t1"
        stale_row.tenant_id = "ghost:t1"
        schema_registry._get_all_schemas.return_value = [live, stale_row]

        data = client.post("/admin/reconcile-orphans?dry_run=true").json()

        assert data["tenant_orphan_schemas"] == []
        assert data["tenant_orphan_tenants"] == []

    def test_confirm_drops_tenant_orphans_and_reads_back(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.side_effect = [
            [
                "tenant_metadata",
                "agent_memories_legit_legit",
                "agent_memories_ghost_t1",
            ],
            ["tenant_metadata", "agent_memories_legit_legit"],
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        ghost = MagicMock()
        ghost.full_schema_name = "agent_memories_ghost_t1"
        ghost.tenant_id = "ghost:t1"
        schema_registry._get_all_schemas.return_value = [live, ghost]
        schema_manager.delete_tenant_schemas_bulk.return_value = [
            "agent_memories_ghost_t1"
        ]

        data = client.post(
            "/admin/reconcile-orphans?dry_run=false&remove_tenant_orphans=true"
        ).json()

        assert data["tenant_orphans_deleted"] == ["agent_memories_ghost_t1"]
        schema_manager.delete_tenant_schemas_bulk.assert_called_once_with(["ghost:t1"])

    def test_readback_showing_a_survivor_refuses_to_report_success(self, admin_client):
        """The redeploy returned the name as dropped but Vespa still carries
        it: reporting success would leave the operator believing the cluster
        is clean."""
        client, backend, schema_manager, schema_registry = admin_client

        deployed = [
            "tenant_metadata",
            "agent_memories_legit_legit",
            "agent_memories_ghost_t1",
        ]
        schema_manager.list_deployed_document_types.side_effect = [
            deployed,
            deployed,
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        ghost = MagicMock()
        ghost.full_schema_name = "agent_memories_ghost_t1"
        ghost.tenant_id = "ghost:t1"
        schema_registry._get_all_schemas.return_value = [live, ghost]
        schema_manager.delete_tenant_schemas_bulk.return_value = [
            "agent_memories_ghost_t1"
        ]

        resp = client.post(
            "/admin/reconcile-orphans?dry_run=false&remove_tenant_orphans=true"
        )
        assert resp.status_code == 502
        assert resp.json() == {
            "detail": (
                "Tenant-orphan removal did not take for "
                "['agent_memories_ghost_t1']; they are still deployed after "
                "the redeploy"
            )
        }

    def test_empty_selection_is_refused(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_legit_legit",
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        schema_registry._get_all_schemas.return_value = [live]

        resp = client.post(
            "/admin/reconcile-orphans?dry_run=false&remove_tenant_orphans=true"
        )
        assert resp.status_code == 409
        assert resp.json() == {
            "detail": (
                "No tenant-orphan schemas to remove: every schema registered "
                "in Vespa belongs to a tenant that still has a tenant_metadata "
                "record. Refusing an empty selection."
            )
        }
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

    def test_dry_run_never_drops_tenant_orphans(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_ghost_t1",
        ]
        ghost = MagicMock()
        ghost.full_schema_name = "agent_memories_ghost_t1"
        ghost.tenant_id = "ghost:t1"
        schema_registry._get_all_schemas.return_value = [ghost]

        data = client.post(
            "/admin/reconcile-orphans?dry_run=true&remove_tenant_orphans=true"
        ).json()

        assert data["tenant_orphan_schemas"] == ["agent_memories_ghost_t1"]
        assert data["tenant_orphans_deleted"] == []
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()


@pytest.mark.unit
@pytest.mark.ci_fast
class TestTenantRegistryReadContract:
    """An unreadable or truncated tenant list makes live tenants look like
    tenant-orphans, and the removal path would drop them with their
    documents. Both must refuse, never degrade to a partial set."""

    def test_tenant_registry_outage_refuses_to_reconcile(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_legit_legit",
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        schema_registry._get_all_schemas.return_value = [live]
        backend.query_metadata_documents.side_effect = RuntimeError("vespa unreachable")

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        assert resp.json() == {
            "detail": (
                "Cannot read the tenant registry; refusing to reconcile "
                "orphans because every schema would read as a tenant-orphan: "
                "vespa unreachable"
            )
        }
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

    def test_saturated_tenant_page_refuses_to_reconcile(self, admin_client):
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_legit_legit",
        ]
        live = MagicMock()
        live.full_schema_name = "agent_memories_legit_legit"
        live.tenant_id = "legit:legit"
        schema_registry._get_all_schemas.return_value = [live]
        backend.query_metadata_documents.return_value = [
            {"tenant_full_id": f"org{i}:t"}
            for i in range(tenant_manager._TENANT_SWEEP_HITS)
        ]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 503
        assert resp.json() == {
            "detail": (
                "Tenant registry returned 400 rows, at or above the 400-row "
                "page limit; refusing to reconcile orphans because a "
                "truncated tenant list marks live tenants as orphans"
            )
        }
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

    def test_no_tenants_at_all_reports_every_registered_schema(self, admin_client):
        """A tenant registry that answers successfully but empty is a real
        state: schema auto-deploy paths create schemas without creating a
        tenant, so a cluster can hold only orphans, and that is the cluster
        reconciliation exists to clean. Refusing here would make it
        uncleanable. The reads that can FABRICATE orphans raise instead, and
        the removal still needs the explicit flag."""
        client, backend, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "agent_memories_ghost_a",
            "agent_memories_ghost_b",
        ]
        ghost_a = MagicMock()
        ghost_a.full_schema_name = "agent_memories_ghost_a"
        ghost_a.tenant_id = "ghost:a"
        ghost_b = MagicMock()
        ghost_b.full_schema_name = "agent_memories_ghost_b"
        ghost_b.tenant_id = "ghost:b"
        schema_registry._get_all_schemas.return_value = [ghost_a, ghost_b]
        backend.query_metadata_documents.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_orphan_schemas"] == [
            "agent_memories_ghost_a",
            "agent_memories_ghost_b",
        ]
        assert data["tenant_orphan_tenants"] == ["ghost:a", "ghost:b"]
        assert data["tenant_orphans_deleted"] == []
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()


@pytest.mark.unit
@pytest.mark.ci_fast
def test_direct_confirmation_does_not_enable_tenant_orphan_removal(admin_client):
    _, _, manager, registry = admin_client
    manager.list_deployed_document_types.return_value = [
        "tenant_metadata",
        "agent_memories_ghost_t1",
    ]
    row = MagicMock()
    row.full_schema_name = "agent_memories_ghost_t1"
    row.tenant_id = "ghost:t1"
    registry._get_all_schemas.return_value = [row]

    result = asyncio.run(tenant_manager.reconcile_orphans(dry_run=False))

    assert result["tenant_orphan_schemas"] == ["agent_memories_ghost_t1"]
    assert result["tenant_orphans_deleted"] == []
    assert manager.delete_tenant_schemas_bulk.call_args_list == []


@pytest.mark.unit
@pytest.mark.ci_fast
def test_malformed_tenant_metadata_refuses_orphan_removal(admin_client):
    client, backend, manager, registry = admin_client
    manager.list_deployed_document_types.return_value = [
        "tenant_metadata",
        "agent_memories_legit_legit",
    ]
    row = MagicMock()
    row.full_schema_name = "agent_memories_legit_legit"
    row.tenant_id = "legit:legit"
    registry._get_all_schemas.return_value = [row]
    backend.query_metadata_documents.return_value = [{}]

    response = client.post(
        "/admin/reconcile-orphans?dry_run=false&remove_tenant_orphans=true"
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Tenant registry contains a row without tenant_full_id; refusing to reconcile orphans"
    }
    assert manager.delete_tenant_schemas_bulk.call_args_list == []
