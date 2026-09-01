"""Unit tests for the ``/admin/reconcile-orphans`` endpoint.

The endpoint diffs Vespa-deployed schemas against the SchemaRegistry's
active set and either reports orphans (dry_run) or drops them all in
one Vespa redeploy (confirm). Tests mock the backend so they don't
need a live Vespa.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.admin import tenant_manager

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

    backend = MagicMock()
    schema_manager = MagicMock()
    schema_registry = MagicMock()
    backend.schema_manager = schema_manager
    schema_manager._schema_registry = schema_registry
    schema_manager._PROTECTED_SCHEMAS = frozenset(
        {
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
        }
    )
    tenant_manager.backend = backend

    yield TestClient(app), backend, schema_manager, schema_registry

    tenant_manager.backend = None
    tenant_manager.set_schema_loader(previous_loader)


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
