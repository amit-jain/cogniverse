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
            "weird_custom_schema_acme",
        ]
        schema_registry._get_all_schemas.return_value = []

        resp = client.post("/admin/reconcile-orphans?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "weird_custom_schema_acme" in data["orphan_schemas"]
        assert "weird_custom_schema_acme" in data["unrecovered_schemas"]
        assert data["orphan_tenants"] == []


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
        ]
        schema_registry._get_all_schemas.return_value = []
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
