"""Unit tests for the ``/admin/reconcile-orphans`` endpoint.

The endpoint diffs Vespa-deployed schemas against the SchemaRegistry's
active set and either reports orphans (dry_run) or drops them all in
one Vespa redeploy (confirm). Tests mock the backend so they don't
need a live Vespa.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.admin import tenant_manager


@pytest.fixture
def admin_client():
    """TestClient mounting the tenant_manager router with a mock backend."""
    app = FastAPI()
    app.include_router(tenant_manager.router, prefix="/admin")

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
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

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
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()

    def test_unknown_base_prefix_listed_separately(self, admin_client):
        """Schemas whose base prefix isn't in ``KNOWN_BASES`` are
        reported under ``unrecovered_schemas`` so the operator can
        review them rather than silently treated as no-op.
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

    def test_every_shipped_base_schema_is_attributable(self, admin_client):
        """An orphan of ANY shipped base must be attributed to its tenant.

        The reconciler strips a KNOWN_BASES prefix to recover the orphan's
        tenant; a shipped base missing from that list makes its orphans
        unattributable — never a deletion target — and (since the redeploy
        refuses unresolved survivors) one such orphan then blocks every
        tenant-delete and reconcile. Drives the real route with one orphan per
        base schema shipped in configs/schemas and asserts none land in
        unrecovered_schemas.
        """
        from pathlib import Path

        client, _, schema_manager, schema_registry = admin_client

        repo_root = Path(__file__).resolve().parents[3]
        protected = {
            "tenant_metadata",
            "organization_metadata",
            "config_metadata",
            "adapter_registry",
        }
        shipped_bases = sorted(
            p.name.removesuffix("_schema.json")
            for p in (repo_root / "configs" / "schemas").glob("*_schema.json")
            if p.name.removesuffix("_schema.json") not in protected
        )
        assert shipped_bases  # the glob found the shipped schema set

        schema_manager.list_deployed_document_types.return_value = [
            *protected,
            *[f"{base}_pt_pt" for base in shipped_bases],
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

    def test_attribution_matches_longest_base_first(self, admin_client):
        """A ``document_text_semantic`` orphan must attribute to its real
        tenant token — first-match-wins on the shorter ``document_text`` prefix
        would strip it to the bogus tenant ``semantic_<tid>``."""
        client, _, schema_manager, schema_registry = admin_client

        schema_manager.list_deployed_document_types.return_value = [
            "tenant_metadata",
            "config_metadata",
            "organization_metadata",
            "adapter_registry",
            "document_text_semantic_acme_acme",
            "knowledge_graph_legit",
        ]
        legit = MagicMock()
        legit.full_schema_name = "knowledge_graph_legit"
        schema_registry._get_all_schemas.return_value = [legit]

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orphan_tenants"] == ["acme_acme"]
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
        schema_manager.delete_tenant_schemas_bulk.return_value = [
            "knowledge_graph_alpha",
            "video_colpali_smol500_mv_frame_beta",
        ]

        resp = client.post("/admin/reconcile-orphans?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()

        assert data["dry_run"] is False
        schema_manager.delete_tenant_schemas_bulk.assert_called_once_with(
            ["alpha", "beta"]
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
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()


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
        schema_manager.delete_tenant_schemas_bulk.assert_not_called()
