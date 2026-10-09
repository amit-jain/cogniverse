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

import json
import logging
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from cogniverse_runtime.ingestion_jobs import IngestionJobStore
from cogniverse_runtime.routers import graph as graph_router
from cogniverse_runtime.routers import ingestion as ingestion_router
from cogniverse_runtime.task_events import INGESTION, TaskEventStore


def _task_events(redis) -> TaskEventStore:
    prefix = f"test:task-events:{uuid.uuid4().hex}"
    return TaskEventStore(
        redis, key_prefix=prefix, ingestion_stream_prefix=f"{prefix}:ingest:"
    )


@pytest.fixture
def job_store(shared_state_redis_url):
    """A job store and a task event store of their own on the test-owned
    Redis, injected into the router. Their client connects lazily, on the
    loop that first uses it."""
    redis = Redis.from_url(shared_state_redis_url, decode_responses=True)
    store = IngestionJobStore(
        redis, owner="test", key_prefix=f"test:ingestion-job:{uuid.uuid4().hex}"
    )
    previous = ingestion_router._job_store, ingestion_router._task_event_store
    ingestion_router.set_job_store(store)
    ingestion_router.set_task_event_store(_task_events(redis))
    try:
        yield store
    finally:
        ingestion_router._job_store, ingestion_router._task_event_store = previous


@pytest.fixture
def unreachable_job_store(dead_redis_url):
    """A job store whose Redis nothing answers, injected into the router."""
    redis = Redis.from_url(
        dead_redis_url, decode_responses=True, socket_connect_timeout=1
    )
    store = IngestionJobStore(redis, owner="test", key_prefix="test:ingestion-job")
    previous = ingestion_router._job_store
    ingestion_router.set_job_store(store)
    try:
        yield store
    finally:
        ingestion_router._job_store = previous


@pytest.fixture
def task_event_store(shared_state_redis_url):
    """A task event store of its own on the test-owned Redis, injected into
    the router."""
    store = _task_events(Redis.from_url(shared_state_redis_url, decode_responses=True))
    previous = ingestion_router._task_event_store
    ingestion_router.set_task_event_store(store)
    try:
        yield store
    finally:
        ingestion_router._task_event_store = previous


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
        """``POST /ingestion/start`` (the job-based path used by the legacy
        CLI) must 404 for an unregistered tenant before
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
    """A ``/ingestion/start`` caller must post the body shape the route
    accepts: ``IngestionRequest`` requires ``profile`` (a singular string),
    so a ``profiles: [<name>]`` list 422s before reaching any real work."""

    def test_singular_profile_body_passes_validation(
        self, ingestion_client_missing_tenant
    ):
        """A singular ``profile`` body must reach
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
            f"singular-profile body must satisfy IngestionRequest; got 422: {resp.text}"
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


@pytest.mark.unit
@pytest.mark.ci_fast
class TestStartIngestionSuccess:
    """POST /ingestion/start success branch: job registered, background task
    scheduled and run to completion (TestClient executes BackgroundTasks
    before returning), backend resolved through the registry."""

    def _build_app(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        app = FastAPI()
        app.include_router(ingestion_router.router, prefix="/ingestion")
        cm = MagicMock(name="config_manager")
        sl = MagicMock(name="schema_loader")
        app.dependency_overrides[ingestion_router.get_config_manager_dependency] = (
            lambda: cm
        )
        app.dependency_overrides[ingestion_router.get_schema_loader_dependency] = (
            lambda: sl
        )

        async def _ok(tenant_id: str) -> None:
            return None

        monkeypatch.setattr(ingestion_router, "assert_tenant_exists", _ok)

        registry = MagicMock(name="backend_registry")
        registry.get_ingestion_backend.return_value = MagicMock(name="backend")
        backend_registry_cls = MagicMock(name="BackendRegistry")
        backend_registry_cls.get_instance.return_value = registry
        monkeypatch.setattr(ingestion_router, "BackendRegistry", backend_registry_cls)

        recorded: dict = {}

        class _StubPipeline:
            def __init__(self, **kwargs):
                recorded["pipeline_init"] = kwargs

            async def process_videos_concurrent(self, video_files, max_concurrent):
                recorded["process_call"] = {
                    "video_files": video_files,
                    "max_concurrent": max_concurrent,
                }
                return {
                    "status": "completed",
                    "successful": 2,
                    "failed": 0,
                    "results": [
                        {"video_path": "a.mp4", "status": "success"},
                        {"video_path": "b.mp4", "status": "success"},
                    ],
                }

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline",
            _StubPipeline,
        )

        def _discover(video_dir, content_type):
            recorded["discover_call"] = {
                "video_dir": video_dir,
                "content_type": content_type,
            }
            return [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]

        monkeypatch.setattr(
            "cogniverse_runtime.ingestion.strategies.discover_ingestible_files",
            _discover,
        )
        return app, cm, sl, registry, recorded

    def test_start_registers_job_and_runs_background_task(
        self, monkeypatch, tmp_path, job_store
    ):
        app, cm, sl, registry, recorded = self._build_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": str(tmp_path),
                    "profile": "video_colpali_smol500_mv_frame",
                    "tenant_id": "acme:acme",
                    "content_type": "video",
                },
            )
            # TestClient ran the background task before returning, so the
            # stored job record has reached its terminal state.
            status = client.get(f"/ingestion/status/{resp.json()['job_id']}")
            events = client.portal.call(
                lambda: ingestion_router.get_task_event_store().read(
                    resp.json()["job_id"], kind=INGESTION
                )
            )
            client.portal.call(job_store._redis.aclose)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        job_id = body["job_id"]
        uuid.UUID(job_id)  # job_id is a real uuid4 string
        assert body == {
            "job_id": job_id,
            "status": "started",
            "message": "Ingestion job started successfully",
        }

        registry.get_ingestion_backend.assert_called_once_with(
            name="vespa",
            tenant_id="acme:acme",
            config_manager=cm,
            schema_loader=sl,
        )

        assert status.status_code == 200, status.text
        assert status.json() == {
            "job_id": job_id,
            "status": "completed",
            "videos_processed": 2,
            "videos_total": 2,
            "errors": [],
        }

        queue = recorded["pipeline_init"].pop("event_queue")
        assert recorded["pipeline_init"] == {
            "tenant_id": "acme:acme",
            "config_manager": cm,
            "schema_loader": sl,
            "schema_name": "video_colpali_smol500_mv_frame",
        }
        # The pipeline reports to the job's own task, which ends with the
        # recorded outcome.
        assert (queue.kind, queue.task_id, queue.tenant_id) == (
            INGESTION,
            job_id,
            "acme:acme",
        )
        assert events.closed is True
        assert [json.loads(data) for _, _, data in events.events] == [
            {
                "state": "complete",
                "ingest_id": job_id,
                "result": {
                    "status": "completed",
                    "videos_processed": 2,
                    "errors": [],
                },
            }
        ]
        assert recorded["process_call"] == {
            "video_files": [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
            "max_concurrent": 10,
        }
        assert recorded["discover_call"] == {
            "video_dir": tmp_path,
            "content_type": "video",
        }

    def test_start_without_a_job_store_is_503_and_runs_nothing(
        self, monkeypatch, tmp_path, unreachable_job_store, task_event_store, caplog
    ):
        """A job that cannot be recorded is not started: no status request
        on any process could ever find it."""
        caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
        app, _cm, _sl, _registry, recorded = self._build_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": str(tmp_path),
                    "profile": "video_colpali_smol500_mv_frame",
                    "tenant_id": "acme:acme",
                    "content_type": "video",
                },
            )
            ended = client.portal.call(
                lambda: task_event_store.read(
                    resp.json()["detail"]["job_id"], kind=INGESTION
                )
            )
            client.portal.call(unreachable_job_store._redis.aclose)
            client.portal.call(task_event_store._redis.aclose)

        assert resp.status_code == 503
        # The job's task, opened first, ends failed: nothing will run it.
        assert ended.closed is True
        assert [json.loads(data)["state"] for _, _, data in ended.events] == ["failed"]
        [logged] = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ]
        cause = (
            "ingestion_job_store_unavailable: IngestionJobStoreUnavailableError: "
            "ingestion job store unavailable: create job "
        )
        assert logged.startswith(cause), logged
        job_id = logged[len(cause) :]
        assert str(uuid.UUID(job_id)) == job_id
        assert resp.json() == {
            "detail": {
                "error": "ingestion_job_store_unavailable",
                "message": "The ingestion job store did not answer, so the job "
                "was not started; retry.",
                "failure": "IngestionJobStoreUnavailableError",
                "job_id": job_id,
            }
        }
        assert recorded == {}

    def test_start_combines_org_id_with_simple_tenant(
        self, monkeypatch, tmp_path, job_store
    ):
        """A separately-supplied ``org_id`` plus a simple ``tenant_id`` must be
        combined into the canonical ``org:tenant`` form before it reaches the
        backend registry (start_ingestion) AND the ingestion pipeline
        (run_ingestion) — otherwise the upload lands in ``tenant:tenant`` while
        the search path (which combines) reads ``org:tenant`` and never sees it.
        """
        app, cm, sl, registry, recorded = self._build_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": str(tmp_path),
                    "profile": "video_colpali_smol500_mv_frame",
                    "tenant_id": "acme",
                    "org_id": "bigcorp",
                    "content_type": "video",
                },
            )
            client.portal.call(job_store._redis.aclose)
        assert resp.status_code == 200, resp.text

        # start_ingestion resolved the backend under the combined tenant.
        registry.get_ingestion_backend.assert_called_once_with(
            name="vespa",
            tenant_id="bigcorp:acme",
            config_manager=cm,
            schema_loader=sl,
        )
        # run_ingestion built the pipeline under the same combined tenant.
        assert recorded["pipeline_init"]["tenant_id"] == "bigcorp:acme"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIngestionStatusEndpoint:
    def test_unknown_job_returns_404(self, job_store):
        app = FastAPI()
        app.include_router(ingestion_router.router, prefix="/ingestion")
        with TestClient(app) as client:
            resp = client.get("/ingestion/status/nope")
            client.portal.call(job_store._redis.aclose)
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Job 'nope' not found"}

    def test_registered_job_returns_exact_status_body(self, job_store):
        app = FastAPI()
        app.include_router(ingestion_router.router, prefix="/ingestion")
        with TestClient(app) as client:
            client.portal.call(job_store.create, "job-status-x")
            client.portal.call(
                lambda: job_store.update(
                    "job-status-x",
                    status="processing",
                    videos_processed=1,
                    videos_total=3,
                    errors=["bad.mp4: schema mismatch"],
                )
            )
            resp = client.get("/ingestion/status/job-status-x")
            client.portal.call(job_store._redis.aclose)
        assert resp.status_code == 200
        assert resp.json() == {
            "job_id": "job-status-x",
            "status": "processing",
            "videos_processed": 1,
            "videos_total": 3,
            "errors": ["bad.mp4: schema mismatch"],
        }

    def test_status_without_a_job_store_is_503(self, unreachable_job_store, caplog):
        caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
        app = FastAPI()
        app.include_router(ingestion_router.router, prefix="/ingestion")
        with TestClient(app) as client:
            resp = client.get("/ingestion/status/job-status-x")
            client.portal.call(unreachable_job_store._redis.aclose)
        assert (resp.status_code, resp.json()) == (
            503,
            {
                "detail": {
                    "error": "ingestion_job_store_unavailable",
                    "message": "The ingestion job store did not answer; retry.",
                    "failure": "IngestionJobStoreUnavailableError",
                    "job_id": "job-status-x",
                }
            },
        )
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ] == [
            "ingestion_job_store_unavailable: IngestionJobStoreUnavailableError: "
            "ingestion job store unavailable: read job job-status-x"
        ]


@pytest.mark.asyncio
async def test_partial_batch_failure_lands_in_job_status(
    monkeypatch, tmp_path, shared_state_redis
):
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

    store = IngestionJobStore(
        shared_state_redis,
        owner="test",
        key_prefix=f"test:ingestion-job:{uuid.uuid4().hex}",
    )
    await store.create("j-partial")
    events = await _task_events(shared_state_redis).open_task(
        INGESTION, "j-partial", "acme:acme"
    )
    req = ing.IngestionRequest(
        video_dir=str(tmp_path),
        profile="video_colpali_smol500_mv_frame",
        tenant_id="acme:acme",
        content_type="video",
    )
    await ing.run_ingestion(
        "j-partial",
        req,
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
        job_store=store,
        events=events,
    )
    job = ing.IngestionStatus(**await store.get("j-partial"))
    assert job.status == "completed_with_errors"
    assert job.errors == ["bad.mp4: schema mismatch"]
    assert job.videos_processed == 2
    assert job.videos_total == 3


@pytest.mark.asyncio
async def test_a_cancelled_job_ends_cancelled_after_its_outcome(
    monkeypatch, tmp_path, shared_state_redis
):
    """A cancellation recorded for a running job reaches the pipeline through
    the job's event queue; the job records ``cancelled`` and its task ends
    with that outcome and the reason."""
    from cogniverse_runtime.routers import ingestion as ing

    task_events = _task_events(shared_state_redis)
    events = await task_events.open_task(INGESTION, "j-cancel", "acme:acme")

    class _StubPipeline:
        def __init__(self, **kwargs):
            self.event_queue = kwargs["event_queue"]

        async def process_videos_concurrent(self, video_files, max_concurrent):
            # Another process records the cancellation; this process's
            # poller delivers it to the job's queue.
            await task_events.cancel(INGESTION, "j-cancel", "operator stop")
            await task_events.poll_once()
            cancelled = self.event_queue.cancellation_token.is_cancelled
            return {
                "status": "cancelled" if cancelled else "completed",
                "successful": 1,
                "results": [
                    {"video_path": "a.mp4", "status": "completed"},
                    {"video_path": "b.mp4", "status": "cancelled"},
                ],
            }

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline",
        _StubPipeline,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.strategies.discover_ingestible_files",
        lambda d, ct: ["a.mp4", "b.mp4"],
    )
    store = IngestionJobStore(
        shared_state_redis,
        owner="test",
        key_prefix=f"test:ingestion-job:{uuid.uuid4().hex}",
    )
    await store.create("j-cancel")
    request = ing.IngestionRequest(
        video_dir=str(tmp_path),
        profile="video_colpali_smol500_mv_frame",
        tenant_id="acme:acme",
        content_type="video",
    )

    await ing.run_ingestion(
        "j-cancel",
        request,
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
        job_store=store,
        events=events,
    )
    job = await store.get("j-cancel")
    ended = await task_events.read("j-cancel", kind=INGESTION)

    assert job["status"] == "cancelled"
    assert ended.closed is True
    assert [json.loads(data) for _, _, data in ended.events] == [
        {
            "state": "cancelled",
            "ingest_id": "j-cancel",
            "result": {"status": "cancelled", "videos_processed": 1, "errors": []},
            "reason": "operator stop",
        }
    ]
    assert await task_events.list_active("acme:acme") == []
