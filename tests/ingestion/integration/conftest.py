"""
Ingestion integration test configuration and fixtures.

Provides module-scoped Vespa + MinIO instances for ingestion tests.
Sets up BACKEND_URL environment variable required by BootstrapConfig.
"""

import json
import os
from pathlib import Path

import pytest

from tests.fixtures import inference as _inference_plugin
from tests.system.minio_test_manager import MinIOTestManager
from tests.utils.markers import (
    is_docker_available,
    is_ffmpeg_available,
)

_INFERENCE_PLUGIN_NAME = "tests.fixtures.inference"


def _register_inference_plugin(plugin_manager):
    if not plugin_manager.hasplugin(_INFERENCE_PLUGIN_NAME):
        plugin_manager.register(_inference_plugin, _INFERENCE_PLUGIN_NAME)


def pytest_configure(config):
    _register_inference_plugin(config.pluginmanager)


def feed_document_via_prod_mapping(
    vespa_app,
    http_port: int,
    schema_name: str,
    schemas_dir: Path,
    *,
    base_schema_name: str,
    video_id: str,
    video_title: str,
    source_url: str,
) -> str:
    """Feed a production ``Document`` into Vespa through the real ingestion
    field mapping (``VespaPyClient.process``) and return the doc id.

    ``schema_name`` is the deployed (tenant-scoped) schema the document is fed
    to; ``base_schema_name`` is the base definition the client loads fields and
    strategies from — the same split ``VespaBackend`` uses for tenant schemas.

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
            "base_schema_name": base_schema_name,
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


def pytest_collection_modifyitems(config, items):
    """Apply non-inference capability markers."""
    ffmpeg_ok = is_ffmpeg_available()
    docker_ok = is_docker_available()
    for item in items:
        if "requires_ffmpeg" in item.keywords and not ffmpeg_ok:
            item.add_marker(
                pytest.mark.skip(
                    reason="FFmpeg/ffprobe not available in this environment"
                )
            )
        if "requires_docker" in item.keywords and not docker_ok:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available in this environment")
            )


@pytest.fixture(autouse=True)
def _test_owned_telemetry():
    """Keep pipeline/worker span export off the default localhost:4317.

    The real ingestion paths call ``get_telemetry_manager()``; without a
    collector the batch exporter sprays connection failures after every
    successful run. When no test-owned collector is configured
    (``TELEMETRY_OTLP_ENDPOINT``, set by ``phoenix_container``), pre-build
    the singleton disabled so spans no-op instead of exporting into the
    void. Tests that assert real span export depend on
    ``phoenix_container``, which sets the env var and resets the manager.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        yield
        return
    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None


# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import phoenix_container, shared_vespa  # noqa: F401, E402


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
        pytest.fail(
            f"No test videos under {TEST_VIDEO_RESOURCE_DIR}; nothing to upload"
        )

    return {
        "bucket": TEST_BUCKET,
        "media_root_uri": f"s3://{TEST_BUCKET}/videos",
        "endpoint_url": minio_instance.endpoint,
        "uploaded": uploaded,
    }
