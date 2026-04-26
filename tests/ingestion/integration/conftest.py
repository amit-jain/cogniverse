"""
Ingestion integration test configuration and fixtures.

Provides module-scoped Vespa + MinIO instances for ingestion tests.
Sets up BACKEND_URL environment variable required by BootstrapConfig.
"""

import os
from pathlib import Path

import pytest

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

    try:
        # Start Vespa container
        if not manager.setup_application_directory():
            pytest.skip("Failed to setup application directory")

        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        # Set environment for tests
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(manager.http_port)

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
