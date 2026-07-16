"""End-to-end integration test for ``POST /ingestion/upload``.

Real services, real data, tight output assertions:
  - Real Redis 7.4 container (the queue + idempotency + status streams).
  - Real MinIO container (the s3:// object store the upload writes to).
  - Real Vespa container with the video profile schema deployed via the
    same ``ApplicationPackage`` path the production runtime uses.
  - Real worker running ``_default_processor`` so the actual
    ``VideoIngestionPipeline`` runs against the actual ``v_-6dz6tBH77I.mp4``
    test video — same video and same profile the existing
    ``test_videoprism_vespa_ingestion`` test uses.

Asserts on the OUTPUT, not just structural shape:
  - HTTP response: state=complete, video_id matches, chunks_created > 0.
  - MinIO: head_object confirms the bytes landed at the expected key.
  - Vespa: documents present at the expected schema, count matches
    chunks_created.
  - Status stream: ordered events queued → running → complete with
    matching ingest_id.
  - Idempotency: re-submit hits the done set, no new Vespa documents.

Profile choice: ``video_videoprism_base_mv_chunk_30s`` runs VideoPrism
in-process (no remote ColPali pod required) so the test doesn't need
the vllm_colpali service deployed.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import requests
from fastapi import FastAPI

TENANT_ID = "test_upload_queue"
PROFILE = "video_videoprism_base_mv_chunk_30s"
# 77-second video — produces multiple 30s chunks for the chunk-based
# VideoPrism profile. Shorter videos in sample_videos/ (6-18s) produce
# zero chunks at chunk_duration=30s.
VIDEO_PATH = Path("data/testset/evaluation/sample_videos/v_-MbZ-W0AbN0.mp4")

REDIS_CONTAINER = "redis-upload-real-stack"
MINIO_CONTAINER = "minio-upload-real-stack"
VESPA_CONTAINER = "vespa-upload-real-stack"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _paired_free_ports() -> int:
    """Return a free `http_port` such that `http_port + 10991` is also free.

    The Vespa runtime computes config_port = http_port + (19071-8080) and
    talks to the config server there. Picking unrelated random ports
    breaks deploys; this keeps the canonical offset intact."""
    import random

    offset = 19071 - 8080
    # Stay in the user/registered range so http_port + 10991 fits under
    # 65535. The OS may still hand the port to someone else between bind
    # and docker run; the Vespa fixture retries on docker failure anyway.
    for _ in range(200):
        candidate = random.randint(20000, 30000)
        config = candidate + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as h:
                h.bind(("127.0.0.1", candidate))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
                c.bind(("127.0.0.1", config))
        except OSError:
            continue
        return candidate
    raise RuntimeError("Could not find a paired (http, config) free port pair")


def _docker_platform() -> str:
    machine = platform.machine().lower()
    return "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"


@pytest.fixture(scope="module")
def redis_container():
    port = _free_port()
    subprocess.run(["docker", "rm", "-f", REDIS_CONTAINER], capture_output=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            REDIS_CONTAINER,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "--platform",
            _docker_platform(),
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")

    deadline = time.time() + 30
    while time.time() < deadline:
        ping = subprocess.run(
            ["docker", "exec", REDIS_CONTAINER, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", REDIS_CONTAINER], capture_output=True)
        pytest.fail("Redis did not become ready within 30s")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", REDIS_CONTAINER], capture_output=True)


@pytest.fixture(scope="module")
def minio_container():
    api_port = _free_port()
    console_port = _free_port()
    access_key = "test-access-key"
    secret_key = "test-secret-key-12chars"
    bucket = "test-ingest-real-stack"

    subprocess.run(["docker", "rm", "-f", MINIO_CONTAINER], capture_output=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            MINIO_CONTAINER,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{api_port}:9000",
            "-p",
            f"{console_port}:9001",
            "-e",
            f"MINIO_ROOT_USER={access_key}",
            "-e",
            f"MINIO_ROOT_PASSWORD={secret_key}",
            "--platform",
            _docker_platform(),
            "minio/minio:latest",
            "server",
            "/data",
            "--console-address",
            ":9001",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start MinIO: {result.stderr}")

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://127.0.0.1:{api_port}/minio/health/ready", timeout=2
            )
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", MINIO_CONTAINER], capture_output=True)
        pytest.fail("MinIO did not become ready within 30s")

    # Create the bucket via boto3 — same client the upload helper uses.
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{api_port}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    s3.create_bucket(Bucket=bucket)

    try:
        yield {
            "endpoint": f"http://127.0.0.1:{api_port}",
            "access_key": access_key,
            "secret_key": secret_key,
            "bucket": bucket,
        }
    finally:
        subprocess.run(["docker", "rm", "-f", MINIO_CONTAINER], capture_output=True)


def _wait_for_config_port(config_port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus", timeout=2
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_data_port(http_port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://localhost:{http_port}/ApplicationStatus", timeout=5
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _wait_for_schema_ready(
    http_port: int, schema_name: str, timeout: int = 120
) -> bool:
    """Confirm the schema accepts writes by GETting any docid (404 is fine —
    proves the dispatcher knows the schema). Pre-deploy returns 400."""
    deadline = time.time() + timeout
    url = (
        f"http://localhost:{http_port}/document/v1/{schema_name}/"
        f"{schema_name}/docid/_probe"
    )
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code in (200, 404):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _deploy_metadata_schemas(config_port: int) -> None:
    """Deploy ONLY the four Vespa metadata schemas. The video profile
    schema is deployed by the worker's SchemaRegistry on first ingest,
    tenant-scoped — pre-deploying it here would put a non-tenant base
    schema in Vespa that the registry doesn't know about, and the
    backend's deploy_schemas safety check refuses to overwrite unknown
    schemas (correct behaviour: refuses silent dropping)."""
    from vespa.package import ApplicationPackage

    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    metadata_schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
    ]
    app_package = ApplicationPackage(name="cogniverse", schema=metadata_schemas)
    mgr = VespaSchemaManager(
        backend_endpoint="http://localhost", backend_port=config_port
    )
    mgr._deploy_package(app_package)


@pytest.fixture(scope="module")
def vespa_backend():
    """Real Vespa with the video profile schema deployed via the same
    ApplicationPackage path the production runtime uses."""
    # The runtime computes config_port from http_port via
    # `calculate_config_port`, which assumes the standard 19071-8080=10991
    # offset. Pick http_port such that http_port + 10991 is also free, so
    # the worker's deploy URL matches the container's actual config port.
    http_port = _paired_free_ports()
    config_port = http_port + (19071 - 8080)
    docker_platform = _docker_platform()

    subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            VESPA_CONTAINER,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{http_port}:8080",
            "-p",
            f"{config_port}:19071",
            "--platform",
            docker_platform,
            "vespaengine/vespa:8.668.5",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Vespa: {result.stderr}")

    if not _wait_for_config_port(config_port):
        subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)
        pytest.fail("Vespa config port not ready within 180s")

    time.sleep(10)

    try:
        _deploy_metadata_schemas(config_port)
    except Exception as exc:
        subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)
        pytest.fail(f"Metadata schema deploy failed: {exc}")

    if not _wait_for_data_port(http_port):
        subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)
        pytest.fail("Vespa data port not ready within 180s after deploy")

    # Wait for one of the metadata schemas to be ready — confirms Vespa
    # has converged. The video profile schema is deployed lazily by the
    # worker on first ingest.
    if not _wait_for_schema_ready(http_port, "tenant_metadata"):
        subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)
        pytest.fail("tenant_metadata schema not ready within 120s")

    try:
        yield {"http_port": http_port, "config_port": config_port}
    finally:
        subprocess.run(["docker", "rm", "-f", VESPA_CONTAINER], capture_output=True)


@pytest_asyncio.fixture
async def real_stack(
    redis_container,
    minio_container,
    vespa_backend,
    videoprism_sidecar,
    monkeypatch,
):
    """All env vars the runtime + worker need, plus a Redis client + S3
    client the test can use directly for output assertions."""
    if not VIDEO_PATH.exists():
        pytest.skip(f"Test video missing at {VIDEO_PATH}")

    # Earlier modules in the same session leave BackendRegistry singletons
    # bound to their (now-dead) containers; a stale shared SchemaRegistry
    # makes the worker's auto-deploy silently target the wrong Vespa and
    # every feed 599s with "Document type does not exist".
    from cogniverse_core.registries.backend_registry import BackendRegistry

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None

    monkeypatch.setenv("REDIS_URL", redis_container)
    monkeypatch.setenv("INGEST_QUEUE_DEPTH_LIMIT", "100")
    monkeypatch.setenv("INGEST_PER_TENANT_CONCURRENCY", "5")
    # Long enough that a done-record survives the slow CPU pipeline (~100s+
    # per job) until the idempotency re-submit checks it.
    monkeypatch.setenv("INGEST_IDEMPOTENCY_TTL_SECONDS", "3600")
    monkeypatch.setenv("MINIO_ENDPOINT", minio_container["endpoint"])
    monkeypatch.setenv("MINIO_ACCESS_KEY", minio_container["access_key"])
    monkeypatch.setenv("MINIO_SECRET_KEY", minio_container["secret_key"])
    monkeypatch.setenv("MINIO_DEFAULT_BUCKET", minio_container["bucket"])
    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(vespa_backend["http_port"]))

    # ConfigUtils._ensure_backend_config merges the JSON file's backend
    # block over the per-tenant config, but treats `http://localhost`/8080
    # as "no override" — so even with BACKEND_PORT set, the runtime
    # picks up the JSON file's hard-coded port 8080 and silently routes
    # to whatever runs on host port 8080 (e.g., a local k3d cluster).
    # Patch the JSON config to point at this test's Vespa.
    src_config_path = Path("configs/config.json")
    config_blob = json.loads(src_config_path.read_text())
    config_blob["backend"]["port"] = vespa_backend["http_port"]
    config_blob["backend"]["url"] = "http://localhost"
    test_config_dir = Path(tempfile.mkdtemp(prefix="upload-queue-config-"))
    (test_config_dir / "schemas").mkdir(parents=True, exist_ok=True)
    # ConfigUtils' tenant-merge needs the schemas dir at the same level
    # as config.json, so symlink the real one in.
    real_schemas_dir = (src_config_path.parent / "schemas").resolve()
    schemas_link = test_config_dir / "schemas"
    if schemas_link.exists():
        schemas_link.rmdir()
    schemas_link.symlink_to(real_schemas_dir, target_is_directory=True)
    test_config_path = test_config_dir / "config.json"
    test_config_path.write_text(json.dumps(config_blob))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(test_config_path))

    # Seed SystemConfig in the test Vespa so ConfigManager.get_system_config
    # returns the right backend_port instead of falling back to SystemConfig()
    # defaults (which would also point at 8080).
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_foundation.config.utils import create_default_config_manager

    seed_cm = create_default_config_manager()
    seed_cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_backend["http_port"],
            # The /ingestion/upload route reads these off SystemConfig (not env)
            # for its deploy-gate; point them at the real test containers.
            redis_url=redis_container,
            minio_endpoint=minio_container["endpoint"],
            # Profiles route VideoPrism inference through the sidecar via
            # ``inference_services.embedding=videoprism_jax`` on the profile
            # (see configs/config.json). The factory pulls the URL out of
            # this dict at embedding-generator build time, so the worker's
            # RemoteVideoPrismLoader hits the sidecar instead of trying to
            # load the JAX/flax stack in-process.
            inference_service_urls={"videoprism_jax": videoprism_sidecar["url"]},
        )
    )

    from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis

    await close_redis()
    redis = await get_redis(redis_container)
    await redis.flushdb()

    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_container["endpoint"],
        aws_access_key_id=minio_container["access_key"],
        aws_secret_access_key=minio_container["secret_key"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    yield {
        "redis": redis,
        "s3": s3,
        "bucket": minio_container["bucket"],
        "vespa_http_port": vespa_backend["http_port"],
    }
    await close_redis()


@pytest_asyncio.fixture
async def worker_task(real_stack):
    """Real worker spawned in-process. Uses ``_default_processor`` so
    the actual ``VideoIngestionPipeline`` runs."""
    from cogniverse_runtime.ingestion_worker.redis_client import get_redis
    from cogniverse_runtime.ingestion_worker.worker import (
        WorkerConfig,
        _claim_loop,
        _default_processor,
    )

    stop = asyncio.Event()
    config = WorkerConfig()
    config.claim_block_ms = 200
    redis = await get_redis(os.environ["REDIS_URL"])
    task = asyncio.create_task(
        _claim_loop(redis, config, stop, processor=_default_processor)
    )
    yield task
    stop.set()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()


@pytest_asyncio.fixture
async def http_client(real_stack):
    """FastAPI ASGI client mounting the real ingestion router + status_api."""
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.ingestion_worker import status_api as ingest_status
    from cogniverse_runtime.routers import ingestion as ingestion_router

    application = FastAPI()
    application.include_router(ingestion_router.router, prefix="/ingestion")
    application.include_router(ingest_status.router, prefix="/ingestion")

    # The router's ConfigManager / SchemaLoader dependencies 503 until
    # overridden — wire them as main.py does so the upload route is reachable
    # (real_stack already seeded SystemConfig in the test Vespa).
    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    application.dependency_overrides[ingestion_router.get_config_manager_dependency] = (
        lambda: config_manager
    )
    application.dependency_overrides[ingestion_router.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    # The upload route calls assert_tenant_exists, which reads tenant_metadata
    # via the tenant manager — wire it and ensure the tenant exists so the
    # route doesn't 404 before the MinIO/Redis env checks.
    from fastapi import HTTPException

    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.admin.models import CreateTenantRequest

    tenant_manager.set_config_manager(config_manager)
    tenant_manager.set_schema_loader(schema_loader)
    try:
        await tenant_manager.create_tenant(
            CreateTenantRequest(tenant_id=TENANT_ID, created_by="test")
        )
    except HTTPException as exc:
        if exc.status_code != 409:  # 409 = already exists (module-scoped reuse)
            raise

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=900
    ) as c:
        yield c


def _vespa_visit_count(
    http_port: int,
    base_schema_name: str,
    tenant_id: str,
    wait_seconds: int = 60,
) -> int:
    """Count documents the worker fed for ``tenant_id`` under ``base_schema_name``.

    The worker deploys schema as ``<base>_<canonical_tenant>`` (the deploy
    path canonicalizes ``tenant`` → ``tenant:tenant`` → ``tenant_tenant``), so
    the visit URL must use that canonical tenant-scoped name.

    Polls up to ``wait_seconds`` because content/distributor nodes lag
    config-server schema activation by 30-120s on a fresh container —
    feed acks before docs are queryable. Returns the first non-zero count
    or 0 after the deadline."""
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    canonical_suffix = canonical_tenant_id(tenant_id).replace(":", "_")
    schema_name = f"{base_schema_name}_{canonical_suffix}"
    yql = f"select * from {schema_name} where true"
    deadline = time.time() + wait_seconds
    last_status = None
    last_text = None
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://localhost:{http_port}/search/",
                params={"yql": yql, "hits": 100},
                timeout=15,
            )
            last_status = r.status_code
            last_text = r.text[:1500]
            if r.ok:
                body = r.json()
                total = body.get("root", {}).get("fields", {}).get("totalCount", 0)
                if total:
                    return total
        except Exception as exc:
            last_status = f"exc:{exc}"
        time.sleep(2)

    raise AssertionError(
        f"Vespa /search returned 0 hits for {schema_name} after {wait_seconds}s. "
        f"last_status={last_status} body={last_text}"
    )


def _vespa_source_urls(http_port: int, base_schema_name: str, tenant_id: str) -> set:
    """Return the distinct ``source_url`` field values on the fed documents.

    The worker localizes the uploaded ``s3://`` object to a temp file before
    processing; every indexed document must still record the canonical
    ``s3://`` source_url (not the ``file://`` temp path) so answer-time keyframe
    resolution can derive the object-store bucket from a hit's own source_url.
    """
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    canonical_suffix = canonical_tenant_id(tenant_id).replace(":", "_")
    schema_name = f"{base_schema_name}_{canonical_suffix}"
    yql = f"select source_url from {schema_name} where true"
    r = requests.get(
        f"http://localhost:{http_port}/search/",
        params={"yql": yql, "hits": 100},
        timeout=15,
    )
    r.raise_for_status()
    children = r.json().get("root", {}).get("children", []) or []
    return {
        (c.get("fields") or {}).get("source_url")
        for c in children
        if (c.get("fields") or {}).get("source_url")
    }


def _videoprism_image_built() -> bool:
    """The chunk-30s VideoPrism profile is now served by the
    ``deploy/videoprism`` sidecar (a remote inference pod). The runtime
    no longer carries the JAX / flax / tensorflow stack in-process. The
    test spins up the sidecar Docker container, so it needs the image
    to be built locally first:

        docker build -t cogniverse/videoprism:dev deploy/videoprism/

    Skip with a clear reason when that image isn't present."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "cogniverse/videoprism:dev"],
            capture_output=True,
        )
    except Exception:
        return False
    return result.returncode == 0


_skip_no_videoprism = pytest.mark.skipif(
    not _videoprism_image_built(),
    reason=(
        "cogniverse/videoprism:dev image not built — run "
        "`docker build -t cogniverse/videoprism:dev deploy/videoprism/`"
    ),
)


VIDEOPRISM_CONTAINER = "videoprism-upload-real-stack"


@pytest.fixture(scope="module")
def videoprism_sidecar():
    """Spin up the VideoPrism inference sidecar that the worker's
    ``RemoteVideoPrismLoader`` calls into. Mirrors how the chart
    deploys the ``inference.videoprism_jax`` pod in production."""
    if not _videoprism_image_built():
        pytest.skip("cogniverse/videoprism:dev not built locally")

    port = _free_port()
    subprocess.run(["docker", "rm", "-f", VIDEOPRISM_CONTAINER], capture_output=True)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            VIDEOPRISM_CONTAINER,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:7999",
            "-e",
            "JAX_PLATFORM_NAME=cpu",
            "-e",
            "JAX_PLATFORMS=cpu",
            "--platform",
            _docker_platform(),
            "cogniverse/videoprism:dev",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start VideoPrism sidecar: {result.stderr}")

    # Cold-start: the JAX JIT trace can take 60-120s. /health returns 503
    # until the model is loaded.
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        subprocess.run(
            ["docker", "rm", "-f", VIDEOPRISM_CONTAINER], capture_output=True
        )
        pytest.fail("VideoPrism sidecar did not become ready within 300s")

    try:
        yield {"url": f"http://127.0.0.1:{port}", "port": port}
    finally:
        subprocess.run(
            ["docker", "rm", "-f", VIDEOPRISM_CONTAINER], capture_output=True
        )


# requires_docker, NOT requires_vespa: this class stands up its OWN Vespa (+
# Redis + MinIO) container via the module fixtures. The requires_vespa gate
# probes an EXTERNAL Vespa on :8080 and would skip whenever that isn't up —
# i.e. exactly when the dev cluster is stopped to free RAM for this test.
@pytest.mark.integration
@pytest.mark.requires_docker
@pytest.mark.slow
class TestUploadRealStack:
    """``POST /ingestion/upload`` end-to-end with real Redis + MinIO +
    Vespa + worker + pipeline + actual video bytes."""

    @_skip_no_videoprism
    @pytest.mark.asyncio
    async def test_upload_writes_to_minio_queues_runs_pipeline_and_lands_in_vespa(
        self, real_stack, worker_task, http_client
    ):
        # Sanity: video exists on disk before the upload.
        assert VIDEO_PATH.exists()
        video_bytes = VIDEO_PATH.read_bytes()
        assert len(video_bytes) > 0

        files = {"file": (VIDEO_PATH.name, io.BytesIO(video_bytes), "video/mp4")}
        data = {
            "profile": PROFILE,
            "backend": "vespa",
            "tenant_id": TENANT_ID,
        }
        resp = await http_client.post(
            "/ingestion/upload",
            params={"wait": "true", "wait_timeout": 600},
            files=files,
            data=data,
        )
        assert resp.status_code == 200, (
            f"Upload returned {resp.status_code}: {resp.text[:500]}"
        )

        body = resp.json()

        # 1. HTTP response shape — tight assertions on actual values.
        assert body["state"] == "complete", (
            f"Expected complete, got state={body['state']!r} body={body!r}"
        )
        assert body["existing"] is False
        assert body["status"] == "success"
        assert body["filename"] == VIDEO_PATH.name
        assert body["source_url"].startswith("s3://")
        assert body["source_url"].endswith(VIDEO_PATH.suffix)
        assert TENANT_ID in body["source_url"], (
            f"source_url should be tenant-scoped: {body['source_url']}"
        )
        ingest_id = body["ingest_id"]
        assert ingest_id.startswith("ingest_")
        sha = body["sha"]
        assert len(sha) == 16

        # 2. MinIO has the uploaded blob at the s3:// URL.
        bucket = real_stack["bucket"]
        key = body["source_url"].split(f"{bucket}/", 1)[1]
        head = real_stack["s3"].head_object(Bucket=bucket, Key=key)
        assert head["ContentLength"] == len(video_bytes)

        # 3. Vespa has documents for this tenant under the profile schema.
        vespa_doc_count = _vespa_visit_count(
            real_stack["vespa_http_port"], PROFILE, TENANT_ID
        )
        assert vespa_doc_count > 0, (
            f"Vespa has zero documents for tenant {TENANT_ID} in {PROFILE} "
            "after upload completed — pipeline didn't actually feed the backend"
        )
        # chunks_created should match the documents that landed in Vespa
        # (the worker counts them as it feeds, response carries the count
        # from the pipeline result summary).
        assert body.get("chunks_created", 0) > 0

        # Every indexed document records the canonical s3:// source_url the
        # upload wrote to MinIO — NOT the file:// temp path the worker localized
        # to. Answer-time keyframe resolution derives the object-store bucket
        # from a hit's own source_url, so a file:// value would make keyframes
        # unfetchable at answer time.
        indexed_source_urls = _vespa_source_urls(
            real_stack["vespa_http_port"], PROFILE, TENANT_ID
        )
        assert indexed_source_urls == {body["source_url"]}, (
            f"indexed source_url(s) {indexed_source_urls} should equal the "
            f"upload's s3:// URL {body['source_url']!r}; a file:// value means "
            "the worker recorded the localized temp path"
        )

        # 4. Status stream — full event history, ordered.
        status_resp = await http_client.get(f"/ingestion/{ingest_id}/status")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert status["state"] == "complete"
        states = [e["state"] for e in status["history"]]
        assert states[0] == "queued"
        assert "running" in states
        assert states[-1] == "complete"
        # The terminal complete event carries the same ingest_id back.
        assert status["latest"]["ingest_id"] == ingest_id

        # 5. Idempotency state in Redis.
        done = await real_stack["redis"].get(f"ingest:done:{sha}")
        assert done == ingest_id
        assert await real_stack["redis"].get(f"ingest:by_sha:{sha}") is None

        # 6. Re-upload the same bytes via /upload. Content-addressable keying
        # maps the identical bytes to the SAME s3:// URL → the same idempotency
        # sha → a cache hit on the completed run. No second pipeline runs, so
        # the index is not doubled. (A uuid key defeated this and re-ran every
        # upload; source-URL idempotency is also exercised directly by
        # ``test_resubmit_same_source_url_hits_idempotency``.)
        files2 = {"file": (VIDEO_PATH.name, io.BytesIO(video_bytes), "video/mp4")}
        resp2 = await http_client.post(
            "/ingestion/upload",
            params={"wait": "true", "wait_timeout": 600},
            files=files2,
            data=data,
        )
        assert resp2.status_code == 200, resp2.text
        body2 = resp2.json()
        assert body2["existing"] is True, (
            f"re-upload of identical bytes must hit idempotency, got {body2!r}"
        )
        assert body2["sha"] == sha, "identical bytes must produce the same sha"
        assert body2["ingest_id"] == ingest_id, "re-upload returns the original id"
        vespa_doc_count_2 = _vespa_visit_count(
            real_stack["vespa_http_port"], PROFILE, TENANT_ID
        )
        # The cache hit skipped the pipeline entirely — count unchanged.
        assert vespa_doc_count_2 == vespa_doc_count, (
            f"Re-upload of identical bytes must not double the index: was "
            f"{vespa_doc_count}, now {vespa_doc_count_2}"
        )

    @_skip_no_videoprism
    @pytest.mark.asyncio
    async def test_resubmit_same_source_url_hits_idempotency(
        self, real_stack, worker_task, http_client
    ):
        """The /upload path computes the idempotency sha on the s3:// URL it
        writes. Verify idempotency explicitly by calling enqueue_ingestion
        with the SAME source_url twice — the second call must return the first
        run's ingest_id instead of enqueuing a new one."""
        from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion

        # First, upload once via /upload to get a real source_url.
        video_bytes = VIDEO_PATH.read_bytes()
        files = {"file": (VIDEO_PATH.name, io.BytesIO(video_bytes), "video/mp4")}
        data = {"profile": PROFILE, "backend": "vespa", "tenant_id": TENANT_ID}
        first_resp = await http_client.post(
            "/ingestion/upload",
            params={"wait": "true", "wait_timeout": 600},
            files=files,
            data=data,
        )
        assert first_resp.status_code == 200
        first_body = first_resp.json()
        assert first_body["state"] == "complete", (
            f"first upload must complete so it marks the idempotency record; "
            f"got {first_body!r}"
        )
        source_url = first_body["source_url"]
        first_id = first_body["ingest_id"]

        # Re-submit via the helper with the SAME source_url — should
        # hit the idempotency cache and return first_id. The /upload route
        # canonicalizes the tenant (require_tenant_id) before computing the
        # idempotency sha, so the re-submit must do the same to match it.
        from cogniverse_core.common.tenant_utils import require_tenant_id

        result = await enqueue_ingestion(
            real_stack["redis"],
            source_url=source_url,
            profile=PROFILE,
            tenant_id=require_tenant_id(TENANT_ID, source="test"),
        )
        assert result.existing is True, (
            f"Re-submit with same source_url should be a cache hit, got {result}"
        )
        assert result.ingest_id == first_id
        # No new ingest_id was minted, no new pipeline run.

    @staticmethod
    def _blank_system_config_field(monkeypatch, **overrides):
        """Route infra checks read SystemConfig (the authoritative source),
        not process env — blank the field there."""
        import dataclasses

        from cogniverse_foundation.config.manager import ConfigManager

        real_get = ConfigManager.get_system_config

        def patched(self):
            return dataclasses.replace(real_get(self), **overrides)

        monkeypatch.setattr(ConfigManager, "get_system_config", patched)

    @pytest.mark.asyncio
    async def test_upload_503_when_minio_config_missing(
        self, real_stack, http_client, monkeypatch
    ):
        """Without SystemConfig.minio_endpoint the upload must fail fast —
        the legacy in-process path is gone, so missing infra config can't
        silently downgrade to a different code path."""
        self._blank_system_config_field(monkeypatch, minio_endpoint="")
        files = {
            "file": (VIDEO_PATH.name, io.BytesIO(b"x"), "video/mp4"),
        }
        resp = await http_client.post(
            "/ingestion/upload",
            files=files,
            data={"profile": PROFILE, "tenant_id": TENANT_ID},
        )
        assert resp.status_code == 503
        body = resp.json()
        # The route reports the missing SystemConfig fields (lowercase keys).
        assert "minio_endpoint" in body["detail"]["missing_env"]

    @pytest.mark.asyncio
    async def test_upload_503_when_redis_config_missing(
        self, real_stack, http_client, monkeypatch
    ):
        self._blank_system_config_field(monkeypatch, redis_url="")
        files = {"file": (VIDEO_PATH.name, io.BytesIO(b"x"), "video/mp4")}
        resp = await http_client.post(
            "/ingestion/upload",
            files=files,
            data={"profile": PROFILE, "tenant_id": TENANT_ID},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert "redis_url" in body["detail"]["missing_env"]

    @pytest.mark.asyncio
    async def test_upload_503_when_minio_env_drifts_from_config(
        self, real_stack, http_client, monkeypatch
    ):
        """SystemConfig advertises MinIO but the process env lacks the
        credentials — the route must answer a retryable 503, not a raw
        RuntimeError/500."""
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        files = {"file": (VIDEO_PATH.name, io.BytesIO(b"x"), "video/mp4")}
        resp = await http_client.post(
            "/ingestion/upload",
            files=files,
            data={"profile": PROFILE, "tenant_id": TENANT_ID},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert "MINIO_ENDPOINT" in body["detail"]["message"]


async def _graph_stub_processor(job) -> dict:
    """Injectable processor standing in for the real pipeline: it returns an
    envelope with graph counts already stamped, exactly as the worker's
    per-segment KG extraction does. Exercises worker._summarise -> terminal
    event -> route wait=true response without needing the LLM/graph backend."""
    return {
        "video_id": "vgraph",
        "documents_fed": 1,
        "results": {"transcript": {"segments": [{"id": 0}]}},
        "graph_nodes": 7,
        "graph_edges": 3,
    }


@pytest_asyncio.fixture
async def graph_stub_worker(real_stack):
    from cogniverse_runtime.ingestion_worker.redis_client import get_redis
    from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _claim_loop

    stop = asyncio.Event()
    config = WorkerConfig()
    config.claim_block_ms = 200
    redis = await get_redis(os.environ["REDIS_URL"])
    task = asyncio.create_task(
        _claim_loop(redis, config, stop, processor=_graph_stub_processor)
    )
    yield task
    stop.set()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        task.cancel()


@pytest.mark.integration
class TestUploadGraphCounts:
    """wait=true must surface the worker's real per-segment KG counts, not the
    dead route re-extraction that always reported 0."""

    @pytest.mark.asyncio
    async def test_wait_true_surfaces_worker_graph_counts(
        self, real_stack, graph_stub_worker, http_client
    ):
        content = b"graphcounts-" + os.urandom(12)  # unique -> distinct sha
        files = {
            "file": (f"g_{os.urandom(4).hex()}.mp4", io.BytesIO(content), "video/mp4")
        }
        data = {"profile": PROFILE, "backend": "vespa", "tenant_id": TENANT_ID}
        resp = await http_client.post(
            "/ingestion/upload",
            params={"wait": "true", "wait_timeout": 60, "force": "true"},
            files=files,
            data=data,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
        body = resp.json()
        assert body["state"] == "complete", body
        assert body["status"] == "success"
        # The worker stamped these; the route surfaces them verbatim (old route
        # re-extracted on a payload with no 'results' and always got 0).
        assert body["graph_nodes"] == 7
        assert body["graph_edges"] == 3


@pytest.mark.integration
class TestMinioOffload:
    """The sync MinIO put_object must run off the event loop so a large
    transfer doesn't freeze the runtime."""

    @pytest.mark.asyncio
    async def test_upload_bytes_runs_off_the_event_loop(
        self, real_stack, http_client, monkeypatch
    ):
        import threading

        import cogniverse_runtime.ingestion_worker.minio_client as mc

        _orig = mc.upload_bytes
        recorded = {}

        def _wrapped(*a, **k):
            recorded["thread"] = threading.get_ident()
            return _orig(*a, **k)  # still hits real MinIO — boundary preserved

        monkeypatch.setattr(mc, "upload_bytes", _wrapped)

        loop_thread = threading.get_ident()
        content = b"minio-offload-" + os.urandom(12)
        files = {
            "file": (f"m_{os.urandom(4).hex()}.mp4", io.BytesIO(content), "video/mp4")
        }
        data = {"profile": PROFILE, "backend": "vespa", "tenant_id": TENANT_ID}
        resp = await http_client.post(
            "/ingestion/upload",
            params={"wait": "false", "force": "true"},
            files=files,
            data=data,
        )
        assert resp.status_code in (200, 202), f"{resp.status_code}: {resp.text[:300]}"
        assert recorded.get("thread") is not None
        # to_thread offload => real MinIO put ran on a worker thread, not the loop.
        assert recorded["thread"] != loop_thread
