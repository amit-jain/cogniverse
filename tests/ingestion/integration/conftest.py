"""
Ingestion integration test configuration and fixtures.

Provides module-scoped Vespa + MinIO instances for ingestion tests.
Sets up BACKEND_URL environment variable required by BootstrapConfig.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.system.minio_test_manager import MinIOTestManager
from tests.utils.markers import (
    is_docker_available,
    is_ffmpeg_available,
    is_vespa_running,
)
from tests.utils.vllm_sidecar import (
    OWNER_LABEL,
    _discover_dev_model_urls,
    _discover_e2e_model_urls,
)


def feed_document_via_prod_mapping(
    vespa_app,
    http_port: int,
    schema_name: str,
    schemas_dir: Path,
    *,
    video_id: str,
    video_title: str,
    source_url: str,
) -> str:
    """Feed a production ``Document`` into Vespa through the real ingestion
    field mapping (``VespaPyClient.process``) and return the doc id.

    Round-trip tests use this instead of a test-only document builder so they
    actually validate that the production mapping carries ``source_url`` (and
    the other fields) into Vespa — a test-only builder would prove nothing
    about what live ingestion writes.
    """
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_sdk.document import ContentType, Document
    from cogniverse_vespa.ingestion_client import VespaPyClient

    client = VespaPyClient(
        {
            "schema_name": schema_name,
            "url": "http://localhost",
            "port": http_port,
            "schema_loader": FilesystemSchemaLoader(schemas_dir),
        }
    )
    doc = Document(
        id=f"{video_id}_seg_0",
        content_type=ContentType.VIDEO,
        content_id=video_id,
    )
    doc.add_metadata("video_id", video_id)
    doc.add_metadata("video_title", video_title)
    doc.add_metadata("source_url", source_url)
    doc.add_metadata("start_time", 0.0)
    doc.add_metadata("end_time", 5.0)
    doc.add_metadata("segment_index", 0)

    # process() returns the full Vespa put envelope {schema, put, fields};
    # feed the inner fields (feed_data_point supplies the id/schema itself).
    fields = client.process(doc)["fields"]
    result = vespa_app.feed_data_point(
        schema=schema_name, data_id=doc.id, fields=fields
    )
    assert result.is_successful(), f"feed failed: {result.json}"
    return doc.id


TEST_VIDEO_RESOURCE_DIR = (
    Path(__file__).resolve().parents[2] / "system" / "resources" / "videos"
)
TEST_BUCKET = "cogniverse-ingestion-corpus"


def materialise_test_pipeline_config(http_port: int) -> str:
    """Write a temporary ``config.json`` whose backend block points at the
    test Vespa container and whose per-profile segmentation strategies
    are capped to test-friendly frame counts. Returns the path; caller
    sets ``COGNIVERSE_CONFIG`` to it.

    Two reasons this exists:

    1. The pipeline's ConfigUtils reads ``backend.url`` / ``backend.port``
       from configs/config.json and treats anything other than
       ``http://localhost`` / 8080 as an override over the
       Vespa-stored SystemConfig. Without patching, every ingestion
       path issues schema-deploy and document-feed traffic at
       localhost:8080 (the developer's k3d cluster) instead of this
       test's freshly-spawned Vespa container — schema discovery
       surfaces dozens of unrelated production schemas and the deploy
       fails with ``Refusing to deploy: Vespa has schemas [...] that
       are not in SchemaRegistry``.
    2. ``config.max_frames_per_video`` only feeds the cache lookup
       path; the segmentation strategies' ``max_frames`` /
       ``max_frames_per_segment`` knobs come straight out of this
       JSON, default to 3000+, and are what actually drives how many
       frames each video's pipeline pushes through the inference
       sidecar. Without lowering them the suite runs 110+ frames per
       profile (~22 min on CPU per profile) and starts hitting the
       sidecar's sustained-load connection ceiling. Two keyframes /
       frames-per-segment is enough for the ingestion-pipeline
       contract these tests assert.
    """
    src_config_path = Path("configs/config.json")
    config_blob = json.loads(src_config_path.read_text())
    config_blob["backend"]["url"] = "http://localhost"
    config_blob["backend"]["port"] = http_port
    for profile_cfg in config_blob.get("backend", {}).get("profiles", {}).values():
        strategies = profile_cfg.get("strategies") or {}
        seg = strategies.get("segmentation") or {}
        seg_params = seg.get("params")
        if isinstance(seg_params, dict):
            if "max_frames" in seg_params:
                seg_params["max_frames"] = 2
            if "max_frames_per_segment" in seg_params:
                seg_params["max_frames_per_segment"] = 2

    import tempfile as _tempfile

    test_config_dir = Path(_tempfile.mkdtemp(prefix="ingest-conftest-"))
    schemas_link = test_config_dir / "schemas"
    if schemas_link.exists():
        schemas_link.rmdir()
    schemas_link.symlink_to(
        (src_config_path.parent / "schemas").resolve(),
        target_is_directory=True,
    )
    test_config_path = test_config_dir / "config.json"
    test_config_path.write_text(json.dumps(config_blob))
    return str(test_config_path)


# Inference services are resolved after collection, and only for tests that
# name them. Explicit test URLs come first, followed by an exact workload in
# the cogniverse-e2e cluster, the development cluster, then an identical
# local sidecar.
_VIDEOPRISM_IMAGE = "cogniverse/videoprism:dev"
_STARTED_INFERENCE_CONTAINERS: list[str] = []
_INFERENCE_SIDECARS = {
    "vllm_colpali": {
        "kind": "vllm",
        "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
        "extra_args": [
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
        ],
    },
    "videoprism_jax": {
        "image": _VIDEOPRISM_IMAGE,
        "container_name": "videoprism-jax-ingest-tests",
        "kind": "health",
        "model_name": "videoprism_public_v1_base_hf",
        "internal_port": 7999,
        "extra_env": {"JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"},
    },
    "vllm_asr": {
        "kind": "vllm",
        "model_name": "openai/whisper-large-v3-turbo",
        "extra_args": [
            "--runner",
            "generate",
            "--max-model-len",
            "448",
        ],
    },
}


def _free_port_for_sidecar() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _container_logs(container: str) -> str:
    try:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "200", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unable to read container logs: {exc}"
    return "\n".join(part for part in (logs.stdout, logs.stderr) if part).strip()


def _health_serves_exact_model(url: str, model: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        payload = response.json()
    except (requests.RequestException, ValueError):
        return False
    return (
        response.status_code == 200
        and isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("model") == model
    )


def _remove_and_raise(
    service: str,
    spec: dict,
    container: str,
    reason: str,
) -> None:
    logs = _container_logs(container)
    try:
        cleanup = subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        cleanup_detail = "\n".join(
            part for part in (cleanup.stdout, cleanup.stderr) if part
        ).strip()
        cleanup_status = (
            "cleanup completed"
            if cleanup.returncode == 0
            else f"cleanup exited {cleanup.returncode}: {cleanup_detail}"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        cleanup_status = f"cleanup failed: {type(exc).__name__}: {exc}"
    raise RuntimeError(
        f"Failed to launch exact inference service {service!r} with model "
        f"{spec['model_name']!r}: {reason}\ncontainer logs:\n{logs}\n"
        f"{cleanup_status}"
    )


def _start_inference_sidecar(service: str, spec: dict) -> str:
    """Start one exact non-vLLM sidecar or raise with logs and cleanup."""
    port = _free_port_for_sidecar()
    container = f"{spec['container_name']}-{os.getpid()}-{port}"
    try:
        subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _remove_and_raise(service, spec, container, f"{type(exc).__name__}: {exc}")
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        # Owner label so a SIGKILLed session's sidecar gets reaped by the
        # next run's reap_dead_owner_containers().
        "--label",
        f"{OWNER_LABEL}={os.getpid()}",
        "-p",
        f"{port}:{spec['internal_port']}",
        "-e",
        f"MODEL_NAME={spec['model_name']}",
    ]
    for env_key, env_val in spec.get("extra_env", {}).items():
        cmd.extend(["-e", f"{env_key}={env_val}"])
    cmd.extend(
        [
            "-v",
            f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
            spec["image"],
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _remove_and_raise(service, spec, container, f"{type(exc).__name__}: {exc}")
    if result.returncode != 0:
        _remove_and_raise(
            service,
            spec,
            container,
            f"docker run exited {result.returncode}: {result.stderr}",
        )

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if _health_serves_exact_model(url, spec["model_name"], timeout=5):
            _STARTED_INFERENCE_CONTAINERS.append(container)
            return url

        try:
            inspect = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Status}}|{{.State.ExitCode}}",
                    container,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _remove_and_raise(
                service,
                spec,
                container,
                f"{type(exc).__name__}: {exc}",
            )
        if inspect.returncode == 0:
            status, _, exit_code = inspect.stdout.strip().partition("|")
            if status == "exited":
                _remove_and_raise(
                    service,
                    spec,
                    container,
                    f"container exited with code {exit_code}",
                )
        time.sleep(5)

    _remove_and_raise(
        service,
        spec,
        container,
        "exact health contract timed out after 1800s",
    )


def _explicit_service_url(service: str) -> str | None:
    raw_urls = os.environ.get("INFERENCE_SERVICE_URLS")
    if not raw_urls:
        return None
    try:
        urls = json.loads(raw_urls)
    except json.JSONDecodeError:
        return None
    url = urls.get(service) if isinstance(urls, dict) else None
    return url if isinstance(url, str) else None


def _resolve_health_service(service: str, spec: dict) -> str:
    candidates = []
    explicit_url = _explicit_service_url(service)
    if explicit_url:
        candidates.append(explicit_url)
    candidates.extend(_discover_e2e_model_urls(spec["model_name"]))
    candidates.extend(_discover_dev_model_urls(spec["model_name"]))
    for url in candidates:
        if _health_serves_exact_model(url, spec["model_name"]):
            return url.rstrip("/")
    return _start_inference_sidecar(service, spec)


def _resolve_inference_services(required: set[str], vllm_sidecar) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for service in sorted(required):
        try:
            spec = _INFERENCE_SIDECARS[service]
        except KeyError as exc:
            raise RuntimeError(
                f"No exact inference sidecar is defined for {service!r}"
            ) from exc
        if spec["kind"] == "vllm":
            resolved[service] = vllm_sidecar.spawn(
                spec["model_name"],
                extra_args=spec["extra_args"],
            )
        else:
            resolved[service] = _resolve_health_service(service, spec)
    return resolved


def pytest_configure(config):
    """Do not start inference services before tests request them."""


@pytest.fixture(scope="session", autouse=True)
def requested_inference_services(request, vllm_sidecar):
    """Resolve only inference services named by the collected ingestion tests."""
    required = getattr(
        request.config,
        "_cogniverse_required_inference_services",
        set(),
    )
    original_urls = os.environ.get("INFERENCE_SERVICE_URLS")
    resolved: dict[str, str] = {}
    try:
        resolved = _resolve_inference_services(required, vllm_sidecar)
        if resolved:
            os.environ["INFERENCE_SERVICE_URLS"] = json.dumps(resolved)
        yield resolved
    finally:
        active_exception = sys.exception()
        cleanup_errors: list[str] = []
        for container in tuple(_STARTED_INFERENCE_CONTAINERS):
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "-f", container],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if cleanup.returncode != 0:
                    detail = "\n".join(
                        part for part in (cleanup.stdout, cleanup.stderr) if part
                    ).strip()
                    cleanup_errors.append(
                        f"{container}: docker exited {cleanup.returncode}: {detail}"
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                cleanup_errors.append(f"{container}: {type(exc).__name__}: {exc}")
            finally:
                _STARTED_INFERENCE_CONTAINERS.remove(container)
        if original_urls is None:
            os.environ.pop("INFERENCE_SERVICE_URLS", None)
        else:
            os.environ["INFERENCE_SERVICE_URLS"] = original_urls
        if cleanup_errors:
            message = "Failed to remove exact inference containers: " + "; ".join(
                cleanup_errors
            )
            if active_exception is not None:
                active_exception.add_note(message)
            else:
                raise RuntimeError(message)


def _inference_requirement(marker) -> str | None:
    reason = marker.kwargs.get("reason")
    if not isinstance(reason, str):
        return None
    for service in _INFERENCE_SIDECARS:
        if reason.startswith(service):
            return service
    return None


_KEYWORD_INFERENCE_REQUIREMENTS = {
    "requires_colpali": "vllm_colpali",
    "requires_colqwen": "vllm_colpali",
    "requires_videoprism": "videoprism_jax",
}
_INFERENCE_DEPENDENCIES = {
    "vllm_colpali": {"vllm_asr"},
    "videoprism_jax": {"vllm_asr"},
}


def _require_inference_service(required: set[str], service: str) -> None:
    required.add(service)
    required.update(_INFERENCE_DEPENDENCIES.get(service, ()))


def pytest_collection_modifyitems(config, items):
    """Resolve named inference and apply non-inference capability markers."""
    required_inference: set[str] = set()
    ffmpeg_ok = is_ffmpeg_available()
    vespa_ok = is_vespa_running()
    docker_ok = is_docker_available()
    for item in items:
        for keyword, service in _KEYWORD_INFERENCE_REQUIREMENTS.items():
            if keyword in item.keywords:
                _require_inference_service(required_inference, service)
        for node, marker in tuple(item.iter_markers_with_node(name="skipif")):
            service = _inference_requirement(marker)
            if service is not None:
                _require_inference_service(required_inference, service)
                node.own_markers = [
                    candidate
                    for candidate in node.own_markers
                    if candidate is not marker
                ]
        if "requires_ffmpeg" in item.keywords and not ffmpeg_ok:
            item.add_marker(
                pytest.mark.skip(
                    reason="FFmpeg/ffprobe not available in this environment"
                )
            )
        if "requires_vespa" in item.keywords and not vespa_ok:
            item.add_marker(
                pytest.mark.skip(reason="Vespa not running in this environment")
            )
        if "requires_docker" in item.keywords and not docker_ok:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available in this environment")
            )
    config._cogniverse_required_inference_services = required_inference


# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401, E402


@pytest.fixture(scope="module")
def ingestion_vespa_backend(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape ingestion/integration tests
    expect, backed by the project-wide ``shared_vespa``.

    Deploys the production video schema for tenant ``test_unit`` via
    SchemaRegistry (merge-safe). Sets ``BACKEND_URL`` /
    ``BACKEND_PORT`` and patches ``COGNIVERSE_CONFIG`` so the ingestion
    pipeline's ``ConfigUtils`` resolves to the shared container — same
    behavior as the prior ``VespaTestManager`` path. Includes a
    ``manager`` field that wraps a small adapter so the few consumers
    that read ``ingestion_vespa_backend["manager"].http_port`` still
    work.
    """

    class _SharedVespaIngestionAdapter:
        def __init__(self, http_port, config_port):
            self.http_port = http_port
            self.config_port = config_port

    # Save old environment
    old_backend_url = os.environ.get("BACKEND_URL")
    old_backend_port = os.environ.get("BACKEND_PORT")
    old_cogniverse_config = os.environ.get("COGNIVERSE_CONFIG")

    http_port = shared_vespa["http_port"]
    config_port = shared_vespa["config_port"]

    # Patch COGNIVERSE_CONFIG to point at the shared container.
    os.environ["COGNIVERSE_CONFIG"] = materialise_test_pipeline_config(http_port)

    # Point create_default_config_manager() at the shared container.
    os.environ["BACKEND_URL"] = "http://localhost"
    os.environ["BACKEND_PORT"] = str(http_port)

    # Deploy the production video schema for tenant test_unit (merge-safe).
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_vespa,
        tenant_id="test:unit",
        base_schema_name="video_colpali_smol500_mv_frame",
    )

    # Seed SystemConfig with the exact inference-service URLs requested by
    # collected tests. Profiles that route embedding through a remote service
    # need this so the pipeline doesn't raise "no URL configured".
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
    )

    raw_urls = os.environ.get("INFERENCE_SERVICE_URLS", "")
    try:
        inference_service_urls = json.loads(raw_urls) if raw_urls else {}
    except json.JSONDecodeError:
        inference_service_urls = {}
    cm = create_default_config_manager()
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
            inference_service_urls=inference_service_urls,
        )
    )

    try:
        yield {
            "manager": _SharedVespaIngestionAdapter(http_port, config_port),
            "http_port": http_port,
            "config_port": config_port,
            "backend_url": f"http://localhost:{http_port}",
        }
    finally:
        # Restore environment
        if old_backend_url is not None:
            os.environ["BACKEND_URL"] = old_backend_url
        elif "BACKEND_URL" in os.environ:
            del os.environ["BACKEND_URL"]

        if old_backend_port is not None:
            os.environ["BACKEND_PORT"] = old_backend_port
        elif "BACKEND_PORT" in os.environ:
            del os.environ["BACKEND_PORT"]

        if old_cogniverse_config is not None:
            os.environ["COGNIVERSE_CONFIG"] = old_cogniverse_config
        elif "COGNIVERSE_CONFIG" in os.environ:
            del os.environ["COGNIVERSE_CONFIG"]
        # No container teardown — shared_vespa owns the lifecycle.


@pytest.fixture(scope="module")
def minio_instance():
    """Module-scoped MinIO container for ingestion integration tests."""
    manager = MinIOTestManager()
    instance = manager.start(name_prefix="minio-ingestion-test")

    saved_access = os.environ.get("AWS_ACCESS_KEY_ID")
    saved_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    saved_region = os.environ.get("AWS_DEFAULT_REGION")
    os.environ["AWS_ACCESS_KEY_ID"] = instance.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = instance.secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    try:
        yield instance
    finally:
        manager.stop()
        for key, prev in (
            ("AWS_ACCESS_KEY_ID", saved_access),
            ("AWS_SECRET_ACCESS_KEY", saved_secret),
            ("AWS_DEFAULT_REGION", saved_region),
        ):
            if prev is not None:
                os.environ[key] = prev
            elif key in os.environ:
                del os.environ[key]


@pytest.fixture(scope="module")
def populated_minio_corpus(minio_instance):
    """Upload the on-disk test videos into a fresh MinIO bucket.

    Returns a dict with the bucket name, the canonical media root URI for the
    pipeline, the s3 endpoint URL, and a list of (video_id, key) tuples for
    each uploaded object.
    """
    client = minio_instance.boto3_client()
    client.create_bucket(Bucket=TEST_BUCKET)

    uploaded: list[tuple[str, str]] = []
    for video_path in sorted(TEST_VIDEO_RESOURCE_DIR.glob("*.mp4")):
        key = f"videos/{video_path.name}"
        client.upload_file(str(video_path), TEST_BUCKET, key)
        uploaded.append((video_path.stem, key))

    if not uploaded:
        pytest.skip(
            f"No test videos under {TEST_VIDEO_RESOURCE_DIR}; nothing to upload"
        )

    return {
        "bucket": TEST_BUCKET,
        "media_root_uri": f"s3://{TEST_BUCKET}/videos",
        "endpoint_url": minio_instance.endpoint,
        "uploaded": uploaded,
    }
