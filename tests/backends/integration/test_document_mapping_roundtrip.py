"""Real-Vespa round-trip for the schema document_mapping write path.

``VespaBackend.put_document`` serializes a generic Document through the
schema's declared ``document_mapping`` block and feeds it; the stored
fields must come back under the SCHEMA's names via get_document_fields.
A schema without a mapping block must refuse with a ValueError naming the
schema — never a guessed or partial feed.
"""

import time
import uuid
from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document

TENANT = f"docmap{uuid.uuid4().hex[:6]}"
BASE_SCHEMA = "document_text"


@pytest.fixture(scope="module")
def mapped_backend(vespa_instance):
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

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=TENANT,
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
        tenant_id=TENANT, base_schema_name=BASE_SCHEMA
    )
    schema_name = backend.get_tenant_schema_name(TENANT, BASE_SCHEMA)

    # prepareandactivate returns before content nodes activate the schema;
    # probe until the first put lands.
    probe = Document(id="__ready__", content_type=ContentType.DOCUMENT, title="r")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            backend.put_document(
                probe, schema_name=schema_name, base_schema_name=BASE_SCHEMA
            )
            if (
                backend.get_document_fields(probe.id, schema_name=schema_name)
                is not None
            ):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        pytest.fail(f"{schema_name} not feedable within 90s of deploy")

    return backend, schema_name


@pytest.mark.integration
class TestDocumentMappingRoundTrip:
    def test_put_document_stores_mapped_fields(self, mapped_backend):
        backend, schema_name = mapped_backend
        doc = Document(
            id="mapped-1",
            content_type=ContentType.DOCUMENT,
            title="Quarterly Report",
            text_content="Revenue grew in every region.",
            created_at=1700000000,
            updated_at=1700000000,
            metadata={"page_count": 4, "document_path": "/reports/q3.pdf"},
        )

        backend.put_document(doc, schema_name=schema_name, base_schema_name=BASE_SCHEMA)

        fields = backend.get_document_fields("mapped-1", schema_name=schema_name)
        assert fields is not None
        assert fields["document_id"] == "mapped-1"
        assert fields["document_title"] == "Quarterly Report"
        assert fields["full_text"] == "Revenue grew in every region."
        assert fields["document_type"] == "document"
        assert fields["creation_timestamp"] == 1700000000
        assert fields["page_count"] == 4
        assert fields["document_path"] == "/reports/q3.pdf"

    def test_schema_without_mapping_refuses(self, mapped_backend):
        backend, _ = mapped_backend
        doc = Document(id="refused-1", title="t")

        with pytest.raises(ValueError, match="agent_memories"):
            backend.put_document(
                doc,
                schema_name="agent_memories",
                base_schema_name="agent_memories",
            )

        assert (
            backend.get_document_fields("refused-1", schema_name="agent_memories")
            is None
        )
