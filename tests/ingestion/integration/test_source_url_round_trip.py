"""End-to-end round-trip test: source_url written at ingest comes back at search.

Builds a Vespa document via :class:`DocumentBuilder` carrying a known
``source_url``, feeds it through a real Vespa instance managed by the
``ingestion_vespa_backend`` fixture (Docker container via
``VespaTestManager``), queries it back, and asserts the field round-trips
exactly. No mocking the Vespa boundary.

The fixture deploys metadata schemas only; this test redeploys with the
``video_colpali_smol500_mv_frame`` schema added so documents of that type
can be fed. Skips cleanly via ``requires_docker`` when Docker isn't there.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

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


def _wait_for_searchable(vespa_app, doc_id: str, timeout: float = 60.0) -> dict | None:
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


def _deploy_video_schema_alongside_metadata(config_port: int, http_port: int) -> None:
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

    # Wait for the new document type to be queryable. Vespa returns 200 on
    # prepareandactivate before the schema is actually active for feeds.
    yql = f"select * from {SCHEMA} where true limit 0"
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"http://localhost:{http_port}/search/",
                params={"yql": yql, "hits": 0},
                timeout=5,
            )
            if resp.status_code == 200:
                root = resp.json().get("root", {})
                if "errors" not in root:
                    return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise RuntimeError(
        f"Schema {SCHEMA} did not become queryable within 60s after deploy"
    )


@pytest.fixture(scope="module")
def vespa_app(ingestion_vespa_backend):
    from vespa.application import Vespa

    _deploy_video_schema_alongside_metadata(
        ingestion_vespa_backend["config_port"],
        ingestion_vespa_backend["http_port"],
    )
    return Vespa(url=ingestion_vespa_backend["backend_url"])


@pytest.mark.requires_docker
@pytest.mark.integration
class TestSourceUrlRoundTrip:
    def test_canonical_uri_round_trips(self, ingestion_vespa_backend, vespa_app):
        builder = DocumentBuilder(SCHEMA)
        canonical_uri = "s3://corpus/videos/roundtrip_v.mp4"

        metadata = DocumentMetadata(
            video_id="roundtrip_v",
            video_title="Round-Trip Test",
            segment_idx=0,
            start_time=0.0,
            end_time=5.0,
            source_url=canonical_uri,
        )
        doc = builder.build_document(metadata, {}, {})

        # Feed via Vespa HTTP API (pyvespa)
        result = vespa_app.feed_data_point(
            schema=SCHEMA,
            data_id=doc["id"],
            fields=doc["fields"],
        )
        assert result.is_successful(), f"feed failed: {result.json}"

        fields = _wait_for_searchable(vespa_app, doc["id"])
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri
        assert fields.get("video_id") == "roundtrip_v"

    def test_pvc_uri_round_trips(self, ingestion_vespa_backend, vespa_app):
        builder = DocumentBuilder(SCHEMA)
        canonical_uri = "pvc://media/videos/roundtrip_v_pvc.mp4"

        metadata = DocumentMetadata(
            video_id="roundtrip_v_pvc",
            video_title="PVC Round-Trip Test",
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

        fields = _wait_for_searchable(vespa_app, doc["id"])
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri

    def test_file_uri_round_trips(self, ingestion_vespa_backend, vespa_app, tmp_path):
        builder = DocumentBuilder(SCHEMA)
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")
        canonical_uri = f"file://{clip}"

        metadata = DocumentMetadata(
            video_id="roundtrip_v_file",
            video_title="File Round-Trip Test",
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

        fields = _wait_for_searchable(vespa_app, doc["id"])
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri
