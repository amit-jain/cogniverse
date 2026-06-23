"""Unit tests pinning the ``assert_tenant_exists`` guard on ingestion
and graph write endpoints.

In production with auth, the auth layer enforces tenant existence
before any write operation. For unauthenticated local dev clusters
and any pre-auth code path, the guard at the router level is what
prevents schema-only tenants — tenants whose schemas got auto-deployed
by an upload but were never registered via ``POST /admin/tenants``.
The previous behaviour (no check) accumulated orphan tenants on every
``/ingestion/upload`` and ``/graph/upsert`` with a fresh tenant id.

Tests construct minimal FastAPI apps and patch the registered
``assert_tenant_exists`` to either raise 404 (tenant missing) or
return None (tenant present), then assert the router behaves
correctly.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import graph as graph_router
from cogniverse_runtime.routers import ingestion as ingestion_router


@pytest.fixture
def ingestion_client_missing_tenant():
    """Client whose ``assert_tenant_exists`` raises 404.

    Stubs the FastAPI ConfigManager / SchemaLoader dependencies so
    ``/ingestion/start`` reaches its body (and hence the tenant check)
    instead of 500-ing on the unconfigured-dependency check.
    """
    from unittest.mock import MagicMock

    app = FastAPI()
    app.include_router(ingestion_router.router, prefix="/ingestion")
    app.dependency_overrides[ingestion_router.get_config_manager_dependency] = lambda: (
        MagicMock()
    )
    app.dependency_overrides[ingestion_router.get_schema_loader_dependency] = lambda: (
        MagicMock()
    )

    async def _missing(tenant_id: str) -> None:
        raise HTTPException(
            status_code=404, detail=f"Tenant '{tenant_id}' not registered"
        )

    # Disable the redis/minio short-circuit so the request reaches the
    # tenant check rather than 503-ing on missing infra envs.
    env = {"REDIS_URL": "redis://stub", "MINIO_ENDPOINT": "minio://stub"}
    with patch.dict(os.environ, env, clear=False):
        with patch.object(ingestion_router, "assert_tenant_exists", new=_missing):
            with TestClient(app) as client:
                yield client


@pytest.fixture
def graph_client_missing_tenant():
    """Client whose ``assert_tenant_exists`` raises 404 for graph ops."""
    app = FastAPI()
    app.include_router(graph_router.router, prefix="/graph")

    async def _missing(tenant_id: str) -> None:
        raise HTTPException(
            status_code=404, detail=f"Tenant '{tenant_id}' not registered"
        )

    # The router imports assert_tenant_exists *inside* upsert(), so patch
    # the source module rather than the router-local name.
    with patch(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists",
        new=_missing,
    ):
        with TestClient(app) as client:
            yield client


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIngestionUploadRequiresTenant:
    def test_upload_with_unregistered_tenant_returns_404(
        self, ingestion_client_missing_tenant
    ):
        """``POST /ingestion/upload`` must 404 when the tenant_id has no
        ``tenant_metadata`` document. Pre-fix this was 200 (auto-deploy)
        and produced a schema-only tenant.
        """
        client = ingestion_client_missing_tenant
        resp = client.post(
            "/ingestion/upload",
            files={"file": ("clip.mp4", b"FAKE", "video/mp4")},
            data={"tenant_id": "unregistered_xyz", "profile": "video_colpali"},
        )
        assert resp.status_code == 404, (
            f"upload to unregistered tenant must 404; got {resp.status_code}: "
            f"{resp.text}"
        )
        assert "not registered" in resp.text.lower()

    def test_start_with_unregistered_tenant_returns_404(
        self, ingestion_client_missing_tenant
    ):
        """``POST /ingestion/start`` (job-based path used by the dashboard
        and the legacy CLI) must 404 for an unregistered tenant before
        any backend work or background task creation.
        """
        client = ingestion_client_missing_tenant
        resp = client.post(
            "/ingestion/start",
            json={
                "video_dir": "/tmp/nonexistent",
                "profile": "video_colpali",
                "tenant_id": "unregistered_xyz",
            },
        )
        assert resp.status_code == 404, (
            f"unregistered tenant must 404 (not 500); got {resp.status_code}: "
            f"{resp.text}"
        )
        assert "not registered" in resp.text.lower(), (
            f"expected 'not registered' in body, got: {resp.text}"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestGraphUpsertRequiresTenant:
    def test_upsert_with_unregistered_tenant_returns_404(
        self, graph_client_missing_tenant
    ):
        """``POST /graph/upsert`` must 404 when the tenant has no
        ``tenant_metadata`` document. Pre-fix the endpoint blindly
        constructed a GraphManager which auto-deployed
        ``knowledge_graph_<tenant>`` and accumulated orphan tenants.
        """
        client = graph_client_missing_tenant
        resp = client.post(
            "/graph/upsert",
            json={
                "tenant_id": "unregistered_xyz",
                "source_doc_id": "x.py",
                "nodes": [{"name": "Foo"}],
                "edges": [],
            },
        )
        assert resp.status_code == 404, (
            f"graph upsert to unregistered tenant must 404; got "
            f"{resp.status_code}: {resp.text}"
        )
        assert "not registered" in resp.text.lower()
