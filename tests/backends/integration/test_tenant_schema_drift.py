"""A registered tenant schema follows the shipped definition, against real Vespa.

A tenant schema deployed from an older definition stays registered under the
same full name. The next deploy for that tenant compares the stored
definition with the shipped one and redeploys it when they differ: fields the
shipped definition adds become feedable, and a change Vespa refuses fails
naming the schema while the registry keeps the definition that is live.
"""

import copy
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.exceptions import BackendDeploymentError
from cogniverse_core.registries.schema_registry import (
    SCHEMA_REGISTRY_SERVICE,
    SchemaRegistry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.ingestion_client import document_namespace

BASE_SCHEMA = "document_text"
CHUNK_FIELDS = ("chunk_index", "chunk_count", "chunk_start", "chunk_end")
SHIPPED_SCHEMAS = Path("configs/schemas")


def _shipped_definition() -> dict:
    return json.loads((SHIPPED_SCHEMAS / f"{BASE_SCHEMA}_schema.json").read_text())


def _older_loader(tmp_path: Path, definition: dict) -> FilesystemSchemaLoader:
    directory = tmp_path / "older_schemas"
    directory.mkdir()
    (directory / f"{BASE_SCHEMA}_schema.json").write_text(json.dumps(definition))
    return FilesystemSchemaLoader(directory)


def _without_chunk_fields() -> dict:
    definition = _shipped_definition()
    definition["document"]["fields"] = [
        field
        for field in definition["document"]["fields"]
        if field["name"] not in CHUNK_FIELDS
    ]
    return definition


def _chunk_index_as_string() -> dict:
    definition = _shipped_definition()
    for field in definition["document"]["fields"]:
        if field["name"] == "chunk_index":
            field["type"] = "string"
    return definition


def _ingestion_backend(vespa_instance, config_manager, schema_loader, tenant_id):
    return BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant_id,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _stored_definition(config_manager, tenant_id: str) -> dict:
    entry = config_manager.store.get_config(
        tenant_id=canonical_tenant_id(tenant_id),
        scope=ConfigScope.SCHEMA,
        service=SCHEMA_REGISTRY_SERVICE,
        config_key=f"schema_{BASE_SCHEMA}",
    )
    return json.loads(entry.config_value["schema_definition"])


def _named(definition: dict, full_name: str) -> dict:
    named = copy.deepcopy(definition)
    named["name"] = full_name
    return named


def _windowed_documents(source_id: str, text: str, window: int) -> list[Document]:
    spans = [
        (start, min(start + window, len(text))) for start in range(0, len(text), window)
    ]
    documents = []
    for index, (start, end) in enumerate(spans):
        document = Document(
            id=f"{source_id}_{source_id}_w{index:04d}",
            content_type=ContentType.DOCUMENT,
            content_id=source_id,
            status=ProcessingStatus.COMPLETED,
        )
        document.add_embedding(
            "embedding",
            np.full((3, 128), 0.25 * (index + 1), dtype=np.float32),
            {"type": "float", "raw": True},
        )
        document.add_metadata("document_id", source_id)
        document.add_metadata("document_title", "drift.md")
        document.add_metadata("document_type", "markdown")
        document.add_metadata("document_path", "/corpus/drift.md")
        document.add_metadata("full_text", text[start:end])
        document.add_metadata("page_count", 1)
        document.add_metadata("chunk_index", index)
        document.add_metadata("chunk_count", len(spans))
        document.add_metadata("chunk_start", start)
        document.add_metadata("chunk_end", end)
        documents.append(document)
    return documents


def _drop_tenant_schema(backend, tenant_id: str) -> None:
    backend.schema_manager.delete_schema(tenant_id, BASE_SCHEMA)


@pytest.mark.integration
class TestRegisteredSchemaFollowsTheShippedDefinition:
    def test_an_older_tenant_schema_gains_the_shipped_fields_and_takes_windows(
        self, vespa_instance, temp_config_manager, schema_loader, tmp_path
    ):
        tenant_id = f"drift{uuid.uuid4().hex[:8]}"
        backend = _ingestion_backend(
            vespa_instance, temp_config_manager, schema_loader, tenant_id
        )
        full_name = backend.get_tenant_schema_name(tenant_id, BASE_SCHEMA)
        older = _without_chunk_fields()
        SchemaRegistry(
            temp_config_manager, backend, _older_loader(tmp_path, older)
        ).deploy_schema(tenant_id, BASE_SCHEMA)
        try:
            assert _stored_definition(temp_config_manager, tenant_id) == _named(
                older, full_name
            )

            text = "".join(f"sentence {n} of the drift corpus. " for n in range(12))
            documents = _windowed_documents("driftsource", text, 150)
            assert len(documents) == 3

            outcome = backend.ingest_documents(documents, BASE_SCHEMA)

            assert outcome == {
                "success_count": 3,
                "failed_count": 0,
                "failed_documents": [],
                "total_documents": 3,
            }
            assert _stored_definition(temp_config_manager, tenant_id) == _named(
                _shipped_definition(), full_name
            )
            stored_rows = [
                {
                    key: fields[key]
                    for key in ("document_id", "full_text", *CHUNK_FIELDS)
                }
                for fields in (
                    backend.get_document_fields(
                        document.id,
                        schema_name=full_name,
                        namespace=document_namespace(full_name),
                    )
                    for document in documents
                )
            ]
            assert stored_rows == [
                {
                    "document_id": "driftsource",
                    "full_text": text[start : min(start + 150, len(text))],
                    "chunk_index": index,
                    "chunk_count": 3,
                    "chunk_start": start,
                    "chunk_end": min(start + 150, len(text)),
                }
                for index, start in enumerate(range(0, len(text), 150))
            ]
        finally:
            _drop_tenant_schema(backend, tenant_id)

    def test_concurrent_deploys_of_a_drifted_schema_activate_one_package(
        self, vespa_instance, temp_config_manager, schema_loader, tmp_path
    ):
        """Callers racing on the drifted schema, plus a peer registry whose
        cache still holds the older definition, redeploy it exactly once."""
        tenant_id = f"driftrace{uuid.uuid4().hex[:8]}"
        backend = _ingestion_backend(
            vespa_instance, temp_config_manager, schema_loader, tenant_id
        )
        full_name = backend.get_tenant_schema_name(tenant_id, BASE_SCHEMA)
        SchemaRegistry(
            temp_config_manager,
            backend,
            _older_loader(tmp_path, _without_chunk_fields()),
        ).deploy_schema(tenant_id, BASE_SCHEMA)
        peer = SchemaRegistry(temp_config_manager, backend, schema_loader)

        activations = []
        # The shared registry deploys through the backend that built it, the
        # peer through this tenant's backend; count activations on both.
        deployers = {
            id(owner): owner for owner in (backend, backend.schema_registry._backend)
        }

        def counting(real_deploy):
            def counting_deploy(schema_definitions, *args, **kwargs):
                activations.append(
                    sorted(
                        schema["name"]
                        for schema in schema_definitions
                        if schema["name"] == full_name
                    )
                )
                return real_deploy(schema_definitions, *args, **kwargs)

            return counting_deploy

        for owner in deployers.values():
            owner.deploy_schemas = counting(owner.deploy_schemas)
        try:
            callers = 4
            barrier = threading.Barrier(callers)
            names: list[str] = []
            errors: list[BaseException] = []

            def deploy():
                barrier.wait()
                try:
                    names.append(
                        backend.schema_registry.deploy_schema(tenant_id, BASE_SCHEMA)
                    )
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=deploy) for _ in range(callers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=600)

            assert errors == []
            assert names == [full_name] * callers
            assert activations == [[full_name]]

            assert peer.deploy_schema(tenant_id, BASE_SCHEMA) == full_name
            assert activations == [[full_name]]
            assert _stored_definition(temp_config_manager, tenant_id) == _named(
                _shipped_definition(), full_name
            )
        finally:
            for owner in deployers.values():
                del owner.deploy_schemas
            _drop_tenant_schema(backend, tenant_id)

    def test_a_change_vespa_refuses_fails_naming_the_schema(
        self, vespa_instance, temp_config_manager, schema_loader, tmp_path
    ):
        tenant_id = f"driftrefuse{uuid.uuid4().hex[:8]}"
        backend = _ingestion_backend(
            vespa_instance, temp_config_manager, schema_loader, tenant_id
        )
        full_name = backend.get_tenant_schema_name(tenant_id, BASE_SCHEMA)
        older = _chunk_index_as_string()
        SchemaRegistry(
            temp_config_manager, backend, _older_loader(tmp_path, older)
        ).deploy_schema(tenant_id, BASE_SCHEMA)
        try:
            with pytest.raises(BackendDeploymentError) as refused:
                backend.schema_registry.deploy_schema(tenant_id, BASE_SCHEMA)

            message = str(refused.value)
            assert message.startswith(
                f"Backend deployment failed for schema '{full_name}': "
                "Vespa refused the application package: "
                "Deployment failed with status 400: "
            ), message
            assert (
                "Field 'chunk_index' changed: data type: 'string' -> 'int'" in message
            ), message
            assert _stored_definition(temp_config_manager, tenant_id) == _named(
                older, full_name
            )
        finally:
            _drop_tenant_schema(backend, tenant_id)
