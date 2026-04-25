"""End-to-end round-trip test: source_url written at ingest comes back at search.

Builds a Vespa document via :class:`DocumentBuilder` carrying a known
``source_url``, feeds it through a real Vespa instance, queries it back, and
asserts the field round-trips exactly. No mocking the Vespa boundary — this
is the canonical wiring test for the unified-MediaLocator write path.

Skips cleanly when Vespa is not running (``requires_vespa`` marker).
"""

from __future__ import annotations

import time

import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.document_builders import (
    DocumentBuilder,
    DocumentMetadata,
)

SCHEMA = "video_colpali_smol500_mv_frame"


def _wait_for_searchable(vespa_app, doc_id: str, timeout: float = 30.0) -> dict | None:
    deadline = time.time() + timeout
    yql = f"select * from {SCHEMA} where video_id contains 'roundtrip_v' limit 5"
    while time.time() < deadline:
        try:
            response = vespa_app.query(yql=yql, hits=5)
            children = response.json.get("root", {}).get("children", [])
            for hit in children:
                if hit.get("id", "").endswith(doc_id):
                    return hit.get("fields", {})
        except Exception:
            pass
        time.sleep(1.0)
    return None


@pytest.fixture
def vespa_app(ingestion_vespa_backend):
    from vespa.application import Vespa

    return Vespa(url=ingestion_vespa_backend["backend_url"])


@pytest.mark.requires_vespa
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
