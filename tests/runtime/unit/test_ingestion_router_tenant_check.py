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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
class TestStartIngestionBodyContract:
    """The dashboard's ``process_video`` action must post the body shape the
    route accepts. It previously sent ``profiles: [<name>]`` (a list) while
    ``IngestionRequest`` requires ``profile`` (a singular string), so every
    dashboard ingestion 422'd before reaching any real work."""

    def test_dashboard_body_shape_passes_validation(
        self, ingestion_client_missing_tenant
    ):
        """The singular ``profile`` body the dashboard now sends must reach
        the tenant check (404 here), not fail model validation (422)."""
        client = ingestion_client_missing_tenant
        resp = client.post(
            "/ingestion/start",
            json={
                "video_dir": "/tmp/nonexistent",
                "profile": "video_colpali_smol500_mv_frame",
                "tenant_id": "unregistered_xyz",
            },
        )
        assert resp.status_code != 422, (
            f"dashboard body must satisfy IngestionRequest; got 422: {resp.text}"
        )
        assert resp.status_code == 404

    def test_legacy_plural_profiles_body_is_rejected(
        self, ingestion_client_missing_tenant
    ):
        """The old ``profiles: [...]`` body (no singular ``profile``) is a
        422 — the regression this fix removes."""
        client = ingestion_client_missing_tenant
        resp = client.post(
            "/ingestion/start",
            json={
                "video_dir": "/tmp/nonexistent",
                "profiles": ["video_colpali_smol500_mv_frame"],
                "tenant_id": "unregistered_xyz",
            },
        )
        assert resp.status_code == 422


@pytest.fixture
def upload_client_backend_check():
    """Client whose tenant check passes and whose system config declares the
    ``vespa`` backend with redis/minio intentionally unset — so a matching
    backend falls through to the 503 infra check while a mismatched backend is
    rejected earlier by the backend validation under test."""
    app = FastAPI()
    app.include_router(ingestion_router.router, prefix="/ingestion")

    cm = MagicMock()
    cm.get_system_config.return_value = SimpleNamespace(
        search_backend="vespa", redis_url="", minio_endpoint=""
    )
    app.dependency_overrides[ingestion_router.get_config_manager_dependency] = lambda: (
        cm
    )
    app.dependency_overrides[ingestion_router.get_schema_loader_dependency] = lambda: (
        MagicMock()
    )

    async def _ok(tenant_id: str) -> None:
        return None

    with patch.object(ingestion_router, "assert_tenant_exists", new=_ok):
        with TestClient(app) as client:
            yield client


@pytest.mark.unit
@pytest.mark.ci_fast
class TestUploadBackendHonored:
    def test_upload_rejects_backend_not_served_here(self, upload_client_backend_check):
        """A ``backend`` this deployment doesn't serve is a 400 — the field was
        silently ignored, so a client believed it ingested to a backend that
        the single-backend queue worker never uses."""
        resp = upload_client_backend_check.post(
            "/ingestion/upload",
            files={"file": ("v.mp4", b"FAKE", "video/mp4")},
            data={
                "tenant_id": "acme:acme",
                "profile": "video_colpali",
                "backend": "qdrant",
            },
        )
        assert resp.status_code == 400, (
            f"mismatched backend must 400; got {resp.status_code}: {resp.text}"
        )
        assert "qdrant" in resp.text.lower() or "backend" in resp.text.lower()

    def test_upload_accepts_configured_backend(self, upload_client_backend_check):
        """The configured backend passes validation and reaches the infra
        check (503 here, since redis/minio are deliberately unset) — proof the
        guard doesn't over-reject the valid backend."""
        resp = upload_client_backend_check.post(
            "/ingestion/upload",
            files={"file": ("v.mp4", b"FAKE", "video/mp4")},
            data={
                "tenant_id": "acme:acme",
                "profile": "video_colpali",
                "backend": "vespa",
            },
        )
        assert resp.status_code == 503, (
            f"configured backend must pass validation; got {resp.status_code}: "
            f"{resp.text}"
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


@pytest.mark.unit
class TestTenantExistenceCache:
    """assert_tenant_exists caches positive lookups for a short TTL (it runs
    on every search/ingestion/graph request) but never caches absence, so a
    freshly created tenant is visible immediately and unknown tenants keep
    404ing."""

    @pytest.mark.asyncio
    async def test_positive_result_cached_negative_rechecked(self, monkeypatch):
        from unittest.mock import AsyncMock

        from cogniverse_core.common import tenant_utils

        monkeypatch.setattr(tenant_utils, "_TENANT_EXISTS_CACHE", {}, raising=True)

        lookups = AsyncMock(side_effect=[None, object(), object()])
        import cogniverse_runtime.admin.tenant_manager as tm

        monkeypatch.setattr(tm, "get_tenant_internal", lookups)

        from fastapi import HTTPException

        # Unknown tenant: 404 and NOT cached.
        with pytest.raises(HTTPException):
            await tenant_utils.assert_tenant_exists("acme:prod")
        assert tenant_utils._TENANT_EXISTS_CACHE == {}

        # Tenant now exists (created between calls): visible immediately.
        await tenant_utils.assert_tenant_exists("acme:prod")
        assert lookups.await_count == 2

        # Repeat checks within the TTL are served from the cache.
        await tenant_utils.assert_tenant_exists("acme:prod")
        await tenant_utils.assert_tenant_exists("acme:prod")
        assert lookups.await_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_after_delete_makes_next_check_404(self, monkeypatch):
        """Tenant deletion drops the cache entry, so the next check re-reads
        the store and 404s instead of serving the stale positive for up to
        the TTL (a deleted tenant's search kept returning its documents)."""
        from unittest.mock import AsyncMock

        from cogniverse_core.common import tenant_utils

        monkeypatch.setattr(tenant_utils, "_TENANT_EXISTS_CACHE", {}, raising=True)

        lookups = AsyncMock(side_effect=[object(), None])
        import cogniverse_runtime.admin.tenant_manager as tm

        monkeypatch.setattr(tm, "get_tenant_internal", lookups)

        from fastapi import HTTPException

        await tenant_utils.assert_tenant_exists("acme:prod")
        assert lookups.await_count == 1

        # Deletion invalidates; the next check must hit the store and 404.
        tenant_utils.invalidate_tenant_exists("acme:prod")
        with pytest.raises(HTTPException):
            await tenant_utils.assert_tenant_exists("acme:prod")
        assert lookups.await_count == 2


@pytest.mark.asyncio
async def test_partial_batch_failure_lands_in_job_status(monkeypatch, tmp_path):
    """The background ingestion task reads the pipeline's per-video results:
    a 2-of-3 batch must surface the failed video id + reason and the
    completed_with_errors status — not report completed with no errors."""
    from unittest.mock import MagicMock

    from cogniverse_runtime.routers import ingestion as ing

    class _StubPipeline:
        def __init__(self, **kwargs):
            pass

        async def process_videos_concurrent(self, video_files, max_concurrent):
            return {
                "job_id": "j-partial",
                "status": "completed_with_errors",
                "total_videos": 3,
                "successful": 2,
                "failed": 1,
                "cancelled": 0,
                "execution_time_seconds": 1.0,
                "results": [
                    {"video_path": "a.mp4", "status": "success"},
                    {
                        "video_path": "bad.mp4",
                        "status": "failed",
                        "error": "schema mismatch",
                        "error_type": "ContentError",
                        "error_context": {},
                    },
                    {"video_path": "c.mp4", "status": "success"},
                ],
            }

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.strategies.discover_ingestible_files",
        lambda d, ct: ["a.mp4", "bad.mp4", "c.mp4"],
    )

    ing.ingestion_jobs["j-partial"] = ing.IngestionStatus(
        job_id="j-partial", status="pending", videos_processed=0, videos_total=0
    )
    req = ing.IngestionRequest(
        video_dir=str(tmp_path),
        profile="video_colpali_smol500_mv_frame",
        tenant_id="acme:acme",
        content_type="video",
    )
    try:
        await ing.run_ingestion(
            "j-partial", req, config_manager=MagicMock(), schema_loader=MagicMock()
        )
        job = ing.ingestion_jobs["j-partial"]
        assert job.status == "completed_with_errors"
        assert job.errors == ["bad.mp4: schema mismatch"]
        assert job.videos_processed == 2
        assert job.videos_total == 3
    finally:
        ing.ingestion_jobs.pop("j-partial", None)
