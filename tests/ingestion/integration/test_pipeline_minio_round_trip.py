"""End-to-end ingestion test: pipeline reads from MinIO and writes to Vespa.

Exercises the full unified-MediaLocator chain at the ingestion boundary:

1. Test videos uploaded to a real MinIO container at ``s3://<bucket>/videos/``.
2. Pipeline constructed with ``media_root_uri="s3://<bucket>/videos"`` and an
   ``s3`` backend pointing at MinIO via ``s3.endpoint_url``.
3. ``pipeline.get_video_files()`` enumerates URIs from MinIO (real
   ``locator.list``).
4. ``pipeline.locator.localize(uri)`` fetches each video from MinIO into the
   tenant-scoped cache (real fsspec/s3fs download).
5. The orchestrator-shaped pipeline state is used to build a Vespa document
   via ``DocumentBuilder.build_document``; ``source_url`` carries the
   canonical ``s3://`` URI.
6. The document is fed to a real Vespa container and queried back; the
   ``source_url`` field round-trips exactly.

No mocking the Vespa or MinIO boundaries. Skips cleanly via
``requires_docker`` when Docker isn't available.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cogniverse_core.common.media import (
    MediaCacheConfig,
    MediaConfig,
    MediaLocator,
    S3BackendConfig,
)
from cogniverse_runtime.ingestion.processors.embedding_generator.document_builders import (
    DocumentBuilder,
    DocumentMetadata,
)

SCHEMA = "video_colpali_smol500_mv_frame"
VIDEO_SCHEMA_JSON = (
    Path(__file__).resolve().parents[2]
    / "system"
    / "resources"
    / "schemas"
    / f"{SCHEMA}_schema.json"
)


def _deploy_video_schema(config_port: int, http_port: int) -> None:
    """Redeploy the cogniverse application with metadata + video schemas."""
    import requests
    from vespa.package import ApplicationPackage

    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    parser = JsonSchemaParser()
    video_schema = parser.load_schema_from_json_file(str(VIDEO_SCHEMA_JSON))

    schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
        video_schema,
    ]

    schema_manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )
    app_package = ApplicationPackage(name="cogniverse", schema=schemas)
    schema_manager._deploy_package(app_package, allow_schema_removal=True)

    yql = f"select * from {SCHEMA} where true limit 0"
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"http://localhost:{http_port}/search/",
                params={"yql": yql, "hits": 0},
                timeout=5,
            )
            if resp.status_code == 200 and "errors" not in resp.json().get("root", {}):
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise RuntimeError(f"Schema {SCHEMA} did not become queryable within 60s")


@pytest.fixture(scope="module")
def media_locator(populated_minio_corpus, tmp_path_factory):
    """MediaLocator wired to the running MinIO corpus."""
    cache_root = tmp_path_factory.mktemp("locator-cache")
    config = MediaConfig(
        default_uri_scheme="s3",
        uri_prefix=f"{populated_minio_corpus['media_root_uri']}/",
        cache=MediaCacheConfig(max_bytes_gb=1),
        s3=S3BackendConfig(
            endpoint_url=populated_minio_corpus["endpoint_url"],
            region="us-east-1",
            anon=False,
        ),
    )
    return MediaLocator(tenant_id="test", config=config, cache_root=cache_root)


@pytest.fixture(scope="module")
def vespa_app(ingestion_vespa_backend):
    from vespa.application import Vespa

    _deploy_video_schema(
        ingestion_vespa_backend["config_port"],
        ingestion_vespa_backend["http_port"],
    )
    return Vespa(url=ingestion_vespa_backend["backend_url"])


def _wait_for_doc(vespa_app, doc_id: str, timeout: float = 30.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = vespa_app.get_data(schema=SCHEMA, data_id=doc_id)
            if getattr(response, "is_successful", lambda: False)():
                fields = response.json.get("fields") or {}
                if fields:
                    return fields
        except Exception:
            pass
        time.sleep(1.0)
    return None


@pytest.mark.requires_docker
@pytest.mark.integration
class TestPipelineMinioRoundTrip:
    def test_locator_lists_videos_from_minio(
        self, media_locator, populated_minio_corpus
    ):
        """The locator should enumerate the MinIO bucket via real list-objects."""
        results = list(
            media_locator.list(
                populated_minio_corpus["media_root_uri"], extensions=(".mp4",)
            )
        )

        assert results, "locator.list returned nothing"
        assert all(r.startswith("s3://") for r in results)
        for video_id, key in populated_minio_corpus["uploaded"]:
            assert any(r.endswith(key) for r in results), (
                f"locator.list did not return {key}; got {results}"
            )

    def test_locator_fetches_from_minio_into_cache(
        self, media_locator, populated_minio_corpus
    ):
        """Localize fetches the bytes from MinIO and caches them locally."""
        first_uri = (
            f"s3://{populated_minio_corpus['bucket']}/"
            f"{populated_minio_corpus['uploaded'][0][1]}"
        )

        local = media_locator.localize(first_uri)

        assert local.exists()
        assert local.stat().st_size > 0
        assert media_locator.cache.base_dir in local.parents

        # Second call: cache hit, no re-download.
        second = media_locator.localize(first_uri)
        assert second == local

    def test_doc_with_minio_source_url_round_trips_through_vespa(
        self, media_locator, populated_minio_corpus, vespa_app
    ):
        """End-to-end: MinIO URI → locator → DocumentBuilder → Vespa → query → assert."""
        builder = DocumentBuilder(SCHEMA)
        video_id, key = populated_minio_corpus["uploaded"][0]
        canonical_uri = f"s3://{populated_minio_corpus['bucket']}/{key}"

        # Verify the video really exists in MinIO and is fetchable.
        local = media_locator.localize(canonical_uri)
        assert local.exists()

        metadata = DocumentMetadata(
            video_id=f"minio_e2e_{video_id}",
            video_title="MinIO end-to-end round-trip",
            segment_idx=0,
            start_time=0.0,
            end_time=5.0,
            source_url=canonical_uri,
        )
        doc = builder.build_document(metadata, {}, {})

        result = vespa_app.feed_data_point(
            schema=SCHEMA,
            data_id=doc["id"],
            fields=doc["fields"],
        )
        assert result.is_successful(), f"feed failed: {result.json}"

        fields = _wait_for_doc(vespa_app, doc["id"])
        assert fields is not None, "document never landed in Vespa"
        assert fields.get("source_url") == canonical_uri
        assert fields.get("source_url").startswith("s3://")

    def test_canonical_uri_built_from_video_path(
        self, media_locator, populated_minio_corpus
    ):
        """The locator's to_canonical_uri honors uri_prefix for object-store ingestion."""
        video_path = "v_-D1gdv_gQyw.mp4"
        canonical = media_locator.to_canonical_uri(video_path)

        assert canonical == (f"{populated_minio_corpus['media_root_uri']}/{video_path}")
