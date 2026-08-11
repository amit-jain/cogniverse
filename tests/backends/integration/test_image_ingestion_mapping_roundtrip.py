"""Real-Vespa round-trip for the image ingestion write path.

Images ride the keyframe pipeline, so an image segment Document carries its
identity metadata under the VIDEO names (``video_id``/``video_title``) plus a
``description``; the image schema's ``document_mapping.metadata_fields``
renames them to ``image_id``/``image_title``/``image_description``. Without
those declared renames process()'s schema-gating drops every identity and
text field and the fed document is unfindable by text search and carries no
identity for consumers reading ``image_id``. This feeds one image segment
onto a live image schema and asserts the renamed values come back under the
SCHEMA's field names, and that ``image_id`` is the bare content id, NOT the
composite document id (``{image}_seg_{n}``).
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

BASE_SCHEMA = "image_colpali_mv"
# The ingestion feed writes content documents under the "content" namespace
# (id:content:<schema>::<id>); reads must target the same namespace.
CONTENT_NS = "content"


def _image_seg_doc(image_id: str, description: str | None = None) -> Document:
    """Build an image segment Document exactly as the keyframe-path ingestion
    builder does: composite id, identity values in metadata under video names,
    a (patches, 320) ColPali tensor."""
    doc = Document(
        id=f"{image_id}_seg_0",
        content_type=ContentType.VIDEO,
        content_id=image_id,
        status=ProcessingStatus.COMPLETED,
    )
    doc.add_embedding(
        "embedding",
        np.full((4, 320), 0.03, dtype=np.float32),
        {"type": "float", "raw": True},
    )
    doc.add_metadata("start_time", 0.0)
    doc.add_metadata("end_time", 1.0)
    doc.add_metadata("segment_index", 0)
    doc.add_metadata("total_segments", 1)
    doc.add_metadata("video_id", image_id)
    doc.add_metadata("video_title", image_id)
    doc.add_metadata("source_url", "s3://imgs/kite.jpg")
    if description is not None:
        doc.add_metadata("description", description)
    return doc


@pytest.fixture(scope="module")
def image_backend(vespa_instance):
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

    tenant = f"imgmap{uuid.uuid4().hex[:6]}"
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
        result = backend.ingest_documents([_image_seg_doc("__ready__")], BASE_SCHEMA)
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


class TestImageIngestionMappingRoundTrip:
    def test_identity_and_text_fields_land_under_image_names(self, image_backend):
        backend, schema_name = image_backend
        result = backend.ingest_documents(
            [_image_seg_doc("kite01", "a red kite")], BASE_SCHEMA
        )
        assert result["success_count"] == 1
        assert result["failed_count"] == 0

        fields = backend.get_document_fields(
            "kite01_seg_0", schema_name=schema_name, namespace=CONTENT_NS
        )
        assert fields is not None

        # Identity is the bare content id, not the composite document id.
        assert fields["image_id"] == "kite01"
        assert fields["image_id"] != "kite01_seg_0"
        assert fields["image_title"] == "kite01"
        assert fields["image_description"] == "a red kite"
        assert fields["source_url"] == "s3://imgs/kite.jpg"

        # The video-named metadata must not leak through under video names.
        assert "video_id" not in fields
        assert "video_title" not in fields
        assert "description" not in fields

    def test_absent_description_feeds_without_image_description(self, image_backend):
        backend, schema_name = image_backend
        result = backend.ingest_documents([_image_seg_doc("kite02")], BASE_SCHEMA)
        assert result["success_count"] == 1

        fields = backend.get_document_fields(
            "kite02_seg_0", schema_name=schema_name, namespace=CONTENT_NS
        )
        assert fields is not None
        assert fields["image_id"] == "kite02"
        assert fields["image_title"] == "kite02"
        assert "image_description" not in fields
