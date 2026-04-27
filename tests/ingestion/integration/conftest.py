"""
Ingestion integration test configuration and fixtures.

Provides module-scoped Vespa + MinIO instances for ingestion tests.
Sets up BACKEND_URL environment variable required by BootstrapConfig.
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from tests.system.minio_test_manager import MinIOTestManager
from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports
from tests.utils.markers import (
    is_docker_available,
    is_ffmpeg_available,
    is_vespa_running,
)

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


# ---------------------------------------------------------------------------
# Inference sidecars for tests that exercise remote-inference profiles.
# Tests gate themselves with ``@pytest.mark.skipif(not _service_configured(...))``
# decorators evaluated at module-import time, so the env var has to be set
# *before* pytest imports the test module — which is what ``pytest_configure``
# is for. We spin up the colpali / colqwen FastAPI sidecars (deploy/colpali)
# once per session, populate ``INFERENCE_SERVICE_URLS``, and tear them down on
# session exit. Image must be pre-built (``docker build -t cogniverse/colpali
# :dev deploy/colpali/``); session is skipped if it isn't.
# ---------------------------------------------------------------------------

_COLPALI_IMAGE = "cogniverse/colpali:dev"
_VIDEOPRISM_IMAGE = "cogniverse/videoprism:dev"
_INFERENCE_SIDECARS = {
    "colpali_infinity": {
        "image": _COLPALI_IMAGE,
        "container_name": "colpali-infinity-ingest-tests",
        "model_name": "vidore/colsmol-500m",
        "internal_port": 7997,
        "extra_env": {"DEVICE": "cpu"},
    },
    "colqwen_infinity": {
        "image": _COLPALI_IMAGE,
        "container_name": "colqwen-infinity-ingest-tests",
        # The chart's production default is ``vidore/colqwen-omni-v0.1``,
        # but that checkpoint's tensor shapes don't match colpali-engine
        # 0.3.13's ``ColQwen2`` class — the multimodal omni variant
        # needs a loader that hasn't shipped yet (RuntimeError on
        # ``Linear.bias`` size 2048 vs 1280). The standard ``colqwen2``
        # release loads cleanly and exercises the same remote-inference
        # contract the colqwen test asserts on (just embeddings come
        # back), so the test still validates the production code path
        # end-to-end without depending on a model upstream can't load.
        "model_name": "vidore/colqwen2-v0.1",
        "internal_port": 7997,
        "extra_env": {"DEVICE": "cpu"},
    },
    "videoprism_jax": {
        "image": _VIDEOPRISM_IMAGE,
        "container_name": "videoprism-jax-ingest-tests",
        "model_name": "videoprism_public_v1_base_hf",
        "internal_port": 7999,
        "extra_env": {"JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"},
    },
}


def _free_port_for_sidecar() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=5
        )
    except Exception:
        return False
    return result.returncode == 0


def _start_inference_sidecar(service: str, spec: dict) -> str | None:
    """Boot one inference sidecar serving ``spec['model_name']`` and
    return its local URL once /health responds. Returns ``None`` if
    startup times out (caller logs and falls back to skipping)."""
    subprocess.run(["docker", "rm", "-f", spec["container_name"]], capture_output=True)
    port = _free_port_for_sidecar()
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        spec["container_name"],
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ingest-conftest] {service} sidecar docker run failed: {result.stderr}")
        return None

    url = f"http://127.0.0.1:{port}"
    # ColPali / ColIdefics3 / ColQwen all download multi-GB checkpoints on
    # first run; allow a generous wait. Subsequent runs hit the HF cache
    # mounted from $HOME/.cache/huggingface. We also short-circuit when the
    # container has already exited (e.g. checkpoint architecture mismatch
    # with the colpali_engine version) so a single broken model can't burn
    # the whole 30-min budget.
    deadline = time.time() + 1800  # 30 min cap
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return url
        except Exception:
            pass

        inspect = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Status}}|{{.State.ExitCode}}",
                spec["container_name"],
            ],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            status, _, exit_code = inspect.stdout.strip().partition("|")
            if status == "exited":
                print(
                    f"[ingest-conftest] {service} sidecar exited with code "
                    f"{exit_code} — aborting wait. Last logs:"
                )
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "20", spec["container_name"]],
                    capture_output=True,
                    text=True,
                )
                print(logs.stdout[-2000:])
                print(logs.stderr[-2000:])
                subprocess.run(
                    ["docker", "rm", "-f", spec["container_name"]],
                    capture_output=True,
                )
                return None
        time.sleep(5)

    print(f"[ingest-conftest] {service} sidecar did not become ready in 30 min")
    subprocess.run(["docker", "rm", "-f", spec["container_name"]], capture_output=True)
    return None


def pytest_configure(config):
    """Boot colpali / colqwen sidecars before any test module imports so that
    ``@requires_colpali_infinity`` / ``@requires_colqwen_infinity`` skipif
    decorators evaluated at import time see a populated
    ``INFERENCE_SERVICE_URLS`` env var.

    Honours an existing env var: if the user already exported one (e.g.
    pointing at a long-running k3d sidecar) we don't fight them. Skips
    silently when ``cogniverse/colpali:dev`` isn't built locally — the
    individual tests then fall through to their own skipif and surface a
    clear ``not configured`` reason."""
    # Honour an existing INFERENCE_SERVICE_URLS only if every URL is
    # actually reachable — otherwise a stale env var from a previous
    # pytest session points at dead sidecar ports and every test that
    # talks to the inference service spins on connection-refused.
    existing = os.environ.get("INFERENCE_SERVICE_URLS")
    if existing:
        try:
            existing_urls = json.loads(existing)
        except json.JSONDecodeError:
            existing_urls = None

        if isinstance(existing_urls, dict):
            all_alive = True
            for url in existing_urls.values():
                try:
                    r = requests.get(f"{url}/health", timeout=2)
                    if r.status_code != 200:
                        all_alive = False
                        break
                except Exception:
                    all_alive = False
                    break
            if all_alive:
                return
            print(
                "[ingest-conftest] INFERENCE_SERVICE_URLS in env points at "
                "unreachable sidecars; ignoring and respawning."
            )
            del os.environ["INFERENCE_SERVICE_URLS"]
        else:
            del os.environ["INFERENCE_SERVICE_URLS"]

    urls: dict[str, str] = {}
    started: list[str] = []
    for service, spec in _INFERENCE_SIDECARS.items():
        if not _docker_image_exists(spec["image"]):
            print(
                f"[ingest-conftest] {service} image {spec['image']} not built locally — "
                "skipping (run docker build to enable that test class)"
            )
            continue
        url = _start_inference_sidecar(service, spec)
        if url is not None:
            urls[service] = url
            started.append(spec["container_name"])

    if urls:
        os.environ["INFERENCE_SERVICE_URLS"] = json.dumps(urls)
        config._cogniverse_inference_containers = started
        print(
            f"[ingest-conftest] INFERENCE_SERVICE_URLS={os.environ['INFERENCE_SERVICE_URLS']}"
        )


def pytest_unconfigure(config):
    """Tear down any sidecars ``pytest_configure`` started."""
    started = getattr(config, "_cogniverse_inference_containers", None) or []
    for name in started:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def pytest_collection_modifyitems(items):
    """Auto-skip tests marked requires_ffmpeg / requires_vespa / requires_docker
    when the corresponding service is unavailable."""
    ffmpeg_ok = is_ffmpeg_available()
    vespa_ok = is_vespa_running()
    docker_ok = is_docker_available()
    for item in items:
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


# Generate unique ports based on this module name
INGESTION_VESPA_PORT, INGESTION_VESPA_CONFIG_PORT = generate_unique_ports(__name__)


@pytest.fixture(scope="module")
def ingestion_vespa_backend():
    """
    Module-scoped Vespa instance for ingestion integration tests.

    Sets up:
    - Vespa Docker container with unique ports
    - BACKEND_URL environment variable
    - Cleans up after module tests complete
    """
    manager = VespaTestManager(
        app_name="test-ingestion-module",
        http_port=INGESTION_VESPA_PORT,
        config_port=INGESTION_VESPA_CONFIG_PORT,
    )

    # Save old environment
    old_backend_url = os.environ.get("BACKEND_URL")
    old_backend_port = os.environ.get("BACKEND_PORT")
    old_cogniverse_config = os.environ.get("COGNIVERSE_CONFIG")

    try:
        # Start Vespa container
        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        # See ``materialise_test_pipeline_config`` for why this is
        # needed (backend port + per-profile frame-count caps). Pointing
        # COGNIVERSE_CONFIG at the patched config makes the pipeline's
        # ConfigUtils resolve to the test Vespa instead of the host's
        # k3d cluster and stops every test from running 110+ frames
        # per profile.
        os.environ["COGNIVERSE_CONFIG"] = materialise_test_pipeline_config(
            manager.http_port
        )

        # Set environment for tests
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(manager.http_port)

        # Seed SystemConfig with the colpali_infinity / videoprism_jax /
        # ... URLs ``pytest_configure`` populated. Profiles that route
        # embedding through a remote service (model_loader=colpali /
        # videoprism + inference_services.embedding=...) read the URL
        # from SystemConfig at pipeline-init time, so without this seed
        # the pipeline raises ``no URL is configured. Deployed services:
        # []`` against the freshly-spawned Vespa.
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
                backend_port=manager.http_port,
                inference_service_urls=inference_service_urls,
            )
        )

        yield {
            "manager": manager,
            "http_port": manager.http_port,
            "config_port": manager.config_port,
            "backend_url": f"http://localhost:{manager.http_port}",
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

        # Cleanup container
        manager.cleanup()


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
