"""Real-Vespa round-trip for the ingestion write path (VespaPyClient.process).

A video segment Document carries its values in metadata: ``video_id`` under a
matching key, but ``segment_index`` and ``description`` under names that differ
from the schema fields (``segment_id`` / ``segment_description``). Those two
renames are declared in the schema's ``document_mapping.metadata_fields`` and
applied by process() during ``ingest_documents``. This feeds one segment onto a
live video schema and asserts the renamed values come back under the SCHEMA's
field names, that ``video_id`` is the metadata value and NOT the composite
document id (``{video}_seg_{n}``), and that ``creation_timestamp`` lands in
milliseconds.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

pytestmark = pytest.mark.integration

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
# The ingestion feed writes content documents under the "content" namespace
# (id:content:<schema>::<id>); reads must target the same namespace.
CONTENT_NS = "content"


def _segment_doc(video_id: str, segment_index: int) -> Document:
    """Build a segment Document exactly as the ingestion embedding generator
    does: composite id, values in metadata, a (patches, 128) ColPali tensor."""
    doc = Document(
        id=f"{video_id}_seg_{segment_index}",
        content_type=ContentType.VIDEO,
        content_id=video_id,
        status=ProcessingStatus.COMPLETED,
    )
    # colsmol-500m emits 320-d ColPali patches (schema: tensor(patch{}, v[320])).
    doc.add_embedding(
        "embedding",
        np.full((4, 320), 0.03, dtype=np.float32),
        {"type": "float", "raw": True},
    )
    doc.add_metadata("start_time", 5.0)
    doc.add_metadata("end_time", 35.0)
    doc.add_metadata("segment_index", segment_index)
    doc.add_metadata("total_segments", 12)
    doc.add_metadata("audio_transcript", "the lion roars")
    doc.add_metadata("video_id", video_id)
    doc.add_metadata("video_title", "My Safari")
    doc.add_metadata("source_url", "s3://vids/safari.mp4")
    doc.add_metadata("description", "a lion cub")
    return doc


@pytest.fixture(scope="module")
def video_backend(vespa_instance):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )

    tenant = f"vidmap{uuid.uuid4().hex[:6]}"
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    backend.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name=BASE_SCHEMA
    )
    schema_name = backend.get_tenant_schema_name(tenant, BASE_SCHEMA)

    # prepareandactivate returns before content nodes activate the schema; feed
    # a probe until the first segment lands.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = backend.ingest_documents([_segment_doc("__ready__", 0)], BASE_SCHEMA)
        if (
            result["success_count"] == 1
            and backend.get_document_fields(
                "__ready___seg_0", schema_name=schema_name, namespace=CONTENT_NS
            )
            is not None
        ):
            break
        time.sleep(2)
    else:
        pytest.fail(f"{schema_name} not feedable within 180s of deploy")

    return backend, schema_name


class TestVideoIngestionMappingRoundTrip:
    def test_declared_renames_land_under_schema_field_names(self, video_backend):
        backend, schema_name = video_backend
        result = backend.ingest_documents([_segment_doc("safari01", 3)], BASE_SCHEMA)
        assert result["success_count"] == 1
        assert result["failed_count"] == 0

        fields = backend.get_document_fields(
            "safari01_seg_3", schema_name=schema_name, namespace=CONTENT_NS
        )
        assert fields is not None
        # The two document_mapping.metadata_fields renames landed under the
        # schema's own names — not the metadata keys the Document carried.
        assert fields["segment_id"] == 3
        assert fields["segment_description"] == "a lion cub"
        assert "segment_index" not in fields
        assert "description" not in fields
        # Straight-through metadata (keys equal to schema fields).
        assert fields["video_title"] == "My Safari"
        assert fields["source_url"] == "s3://vids/safari.mp4"
        assert fields["audio_transcript"] == "the lion roars"
        assert fields["start_time"] == 5.0
        assert fields["end_time"] == 35.0

    def test_video_id_is_metadata_not_composite_document_id(self, video_backend):
        # process() must source video_id from metadata, never from the mapping's
        # id->video_id core map: the ingestion document id is "safari01_seg_3",
        # and writing that into video_id would corrupt every segment's video id.
        backend, schema_name = video_backend
        backend.ingest_documents([_segment_doc("safari01", 3)], BASE_SCHEMA)

        fields = backend.get_document_fields(
            "safari01_seg_3", schema_name=schema_name, namespace=CONTENT_NS
        )
        assert fields["video_id"] == "safari01"
        assert fields["video_id"] != "safari01_seg_3"

    def test_creation_timestamp_stored_in_milliseconds(self, video_backend):
        backend, schema_name = video_backend
        before_ms = int(time.time() * 1000) - 5000
        backend.ingest_documents([_segment_doc("safari02", 0)], BASE_SCHEMA)

        fields = backend.get_document_fields(
            "safari02_seg_0", schema_name=schema_name, namespace=CONTENT_NS
        )
        ts = fields["creation_timestamp"]
        assert isinstance(ts, int)
        assert ts >= before_ms, f"{ts} predates the feed — not a live ms stamp"
        assert ts > 1_000_000_000_000, f"creation_timestamp not milliseconds: {ts}"
