"""End-to-end round-trip test: source_url written at ingest comes back at search.

Builds a production :class:`Document`, maps it to Vespa fields through the real
ingestion mapping (:meth:`VespaPyClient.process` — the same code path live
ingestion uses), feeds it through a real Vespa instance managed by the
``ingestion_vespa_backend`` fixture (the shared session Vespa container),
queries it back, and asserts ``source_url`` round-trips exactly. This exercises
the prod field mapping, so it would catch prod dropping source_url — not a
test-only document builder. No mocking the Vespa boundary.

This module deploys its own tenant-scoped copy of the
``video_colpali_smol500_mv_frame`` schema via SchemaRegistry (merge-safe, so
other schemas on the shared Vespa stay intact) and feeds documents of that
type. Skips cleanly via ``requires_docker`` when Docker isn't there.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.ingestion.integration.conftest import feed_document_via_prod_mapping
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_full_name

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
TENANT_ID = "test:source_url_rt"
SCHEMA = schema_full_name(BASE_SCHEMA, TENANT_ID)
SCHEMAS_DIR = Path("configs/schemas")


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


def _deploy_video_schema(ingestion_vespa_backend) -> None:
    """Deploy the tenant-scoped video schema via SchemaRegistry (merge-safe)."""
    import requests

    deployed = deploy_tenant_schema(
        ingestion_vespa_backend,
        tenant_id=TENANT_ID,
        base_schema_name=BASE_SCHEMA,
    )
    assert deployed == SCHEMA, f"registry deployed {deployed!r}, tests use {SCHEMA!r}"

    # Wait for the new document type to be queryable. Vespa returns 200 on
    # prepareandactivate before the schema is actually active for feeds.
    http_port = ingestion_vespa_backend["http_port"]
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

    _deploy_video_schema(ingestion_vespa_backend)
    return Vespa(url=ingestion_vespa_backend["backend_url"])


@pytest.mark.requires_docker
@pytest.mark.integration
class TestSourceUrlRoundTrip:
    def test_canonical_uri_round_trips(self, ingestion_vespa_backend, vespa_app):
        canonical_uri = "s3://corpus/videos/roundtrip_v.mp4"
        doc_id = feed_document_via_prod_mapping(
            vespa_app,
            ingestion_vespa_backend["http_port"],
            SCHEMA,
            SCHEMAS_DIR,
            base_schema_name=BASE_SCHEMA,
            video_id="roundtrip_v",
            video_title="Round-Trip Test",
            source_url=canonical_uri,
        )

        fields = _wait_for_searchable(vespa_app, doc_id)
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri
        assert fields.get("video_id") == "roundtrip_v"

    def test_pvc_uri_round_trips(self, ingestion_vespa_backend, vespa_app):
        canonical_uri = "pvc://media/videos/roundtrip_v_pvc.mp4"
        doc_id = feed_document_via_prod_mapping(
            vespa_app,
            ingestion_vespa_backend["http_port"],
            SCHEMA,
            SCHEMAS_DIR,
            base_schema_name=BASE_SCHEMA,
            video_id="roundtrip_v_pvc",
            video_title="PVC Round-Trip Test",
            source_url=canonical_uri,
        )

        fields = _wait_for_searchable(vespa_app, doc_id)
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri

    def test_file_uri_round_trips(self, ingestion_vespa_backend, vespa_app, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")
        canonical_uri = f"file://{clip}"
        doc_id = feed_document_via_prod_mapping(
            vespa_app,
            ingestion_vespa_backend["http_port"],
            SCHEMA,
            SCHEMAS_DIR,
            base_schema_name=BASE_SCHEMA,
            video_id="roundtrip_v_file",
            video_title="File Round-Trip Test",
            source_url=canonical_uri,
        )

        fields = _wait_for_searchable(vespa_app, doc_id)
        assert fields is not None, "document never became searchable"
        assert fields.get("source_url") == canonical_uri
