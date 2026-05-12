"""Shared fixtures for memory integration tests."""

import logging
from pathlib import Path

import pytest
import requests

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry

# Re-export the canonical session-scoped Vespa from the project root so
# memory tests pick it up via pytest's normal discovery walk.
from tests.conftest import shared_vespa  # noqa: F401
from tests.utils.async_polling import wait_for_service_startup, wait_for_vespa_indexing

logger = logging.getLogger(__name__)


def wait_for_backend_ready(config_port: int, timeout: int = 120) -> bool:
    """Wait for backend config server to be ready."""
    for _ in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus",
                timeout=2,
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Backend container startup")
    return False


def wait_for_data_port_ready(data_port: int, timeout: int = 120) -> bool:
    """Wait for Vespa HTTP container node (data port) to respond with 200.

    The config port becomes ready well before the HTTP container node, and
    after schema deployment the container node needs additional time to
    initialize. This probe uses GET /ApplicationStatus on the data port so
    it returns True only once the container node is fully up.
    """
    for _ in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{data_port}/ApplicationStatus",
                timeout=5,
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Data port readiness")
    return False


def _get_real_embedding(text: str = "readiness check") -> list:
    """Return a 768-dim probe vector for schema-readiness writes.

    The schema-readiness probe just needs Vespa to accept a valid write
    against the deployed schema; the embedding content doesn't matter
    (the document is deleted right after). A constant-valued vector
    avoids pulling a live embedding service into the readiness path.
    """
    return [0.01] * 768


def _namespace_for_schema(schema_name: str) -> str:
    """Return the Vespa namespace that matches the schema's content type.

    Must mirror the logic in VespaIngestionClient (ingestion_client.py).
    """
    if "agent_memories" in schema_name:
        return "memory_content"
    if "wiki_pages" in schema_name:
        return "wiki_content"
    if any(
        k in schema_name
        for k in ("config_metadata", "tenant_metadata", "organization_metadata")
    ):
        return "metadata"
    return "video"


def _readiness_doc_for_namespace(namespace: str) -> dict:
    """Return a minimal valid document body for the given Vespa namespace."""
    real_embedding = _get_real_embedding()
    if namespace == "wiki_content":
        return {
            "fields": {
                "doc_id": "readiness_check",
                "tenant_id": "test",
                "page_type": "topic",
                "title": "readiness check",
                "content": "test",
                "slug": "readiness_check",
                "entities": "[]",
                "sources": "[]",
                "cross_references": "[]",
                "update_count": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "embedding": real_embedding,
            }
        }
    # Default: memory schema fields
    return {
        "fields": {
            "id": "readiness_check",
            "text": "test",
            "user_id": "test",
            "agent_id": "test",
            "embedding": real_embedding,
            "metadata_": "{}",
            "created_at": 1234567890,
        }
    }


def wait_for_schema_ready(data_port: int, schema_name: str, timeout: int = 120) -> bool:
    """Wait for schema to be ready to accept documents.

    Uses the namespace that matches the schema's content type so the probe
    exercises the same code path as real document operations.
    """
    namespace = _namespace_for_schema(schema_name)
    test_doc = _readiness_doc_for_namespace(namespace)

    for _ in range(timeout):
        try:
            response = requests.post(
                f"http://localhost:{data_port}/document/v1/{namespace}/{schema_name}/docid/readiness_check",
                json=test_doc,
                timeout=5,
            )
            if response.status_code in [200, 201]:
                requests.delete(
                    f"http://localhost:{data_port}/document/v1/{namespace}/{schema_name}/docid/readiness_check",
                    timeout=5,
                )
                return True
        except Exception:
            pass
        wait_for_vespa_indexing(delay=1.0, description="schema readiness check")

    return False


@pytest.fixture(scope="session")
def shared_memory_vespa(shared_vespa):  # noqa: F811  (shared_vespa is the imported fixture)
    """Compatibility shim: yields the dict shape memory tests expect, but
    backed by the single project-wide ``shared_vespa`` container.

    Deploys the three data schemas memory tests need (``agent_memories``,
    ``wiki_pages``, ``provenance``) once per session, all under the
    historical ``test_tenant`` tenant_id so existing tests that hardcode
    that string keep working. Other packages use their own tenant_ids
    (see ``tests/utils/tenant_helpers.py``) so there's no cross-package
    collision on the shared container.

    Yields::

        {
            "http_port", "config_port", "container_name", "base_url",
            "tenant_schema_name": "agent_memories_test_tenant",
            "wiki_schema_name":   "wiki_pages_test_tenant",
            "config_manager":     <ConfigManager bound to shared_vespa>,
            "schema_loader":      <FilesystemSchemaLoader>,
        }
    """
    import json
    import time

    from vespa.package import ApplicationPackage

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_sdk.interfaces.config_store import ConfigScope
    from cogniverse_vespa.config.config_store import VespaConfigStore
    from cogniverse_vespa.json_schema_parser import JsonSchemaParser
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    config_port = shared_vespa["config_port"]
    http_port = shared_vespa["http_port"]

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()

    parser = JsonSchemaParser()
    schemas_dir = Path("configs/schemas")

    # Tenant-scope each schema by overwriting ``name`` and ``document.name``
    # before parsing — the canonical SchemaRegistry pattern, replicated here
    # because we deploy directly against the schema manager (faster than
    # going through SchemaRegistry for the session-scoped one-shot deploy).
    with open(schemas_dir / "agent_memories_schema.json") as f:
        memory_schema_json = json.load(f)
    memory_schema_json["name"] = "agent_memories_test_tenant"
    memory_schema_json["document"]["name"] = "agent_memories_test_tenant"
    memory_schema = parser.parse_schema(memory_schema_json)

    with open(schemas_dir / "wiki_pages_schema.json") as f:
        wiki_schema_json = json.load(f)
    wiki_schema_json["name"] = "wiki_pages_test_tenant"
    wiki_schema_json["document"]["name"] = "wiki_pages_test_tenant"
    wiki_schema = parser.parse_schema(wiki_schema_json)

    with open(schemas_dir / "provenance_schema.json") as f:
        provenance_schema_json = json.load(f)
    provenance_schema_json["name"] = "provenance_test_tenant"
    provenance_schema_json["document"]["name"] = "provenance_test_tenant"
    provenance_schema = parser.parse_schema(provenance_schema_json)

    # Deploy as a merged package alongside the metadata schemas the root
    # ``shared_vespa`` already deployed. ``_deploy_package`` does a
    # full-replace, so we must include the metadata schemas too or we'd
    # wipe them.
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )

    metadata_schemas = [
        create_organization_metadata_schema(),
        create_tenant_metadata_schema(),
        create_config_metadata_schema(),
        create_adapter_registry_schema(),
    ]

    all_schemas = metadata_schemas + [memory_schema, wiki_schema, provenance_schema]
    app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

    schema_mgr = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )
    schema_mgr._deploy_package(app_package)

    # Wait for the data port to converge after the redeploy.
    for _ in range(60):
        try:
            resp = requests.get(
                f"http://localhost:{http_port}/state/v1/health", timeout=2
            )
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)

    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=http_port,
    )
    config_manager = ConfigManager(store=config_store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
        )
    )

    # Register the deployed schemas in ConfigStore so any SchemaRegistry
    # created by downstream fixtures finds them and skips redeploy.
    # The schema_definition MUST be the real, parseable JSON — a stub
    # would fail parse_schema during a later merge-and-redeploy.
    tenant_schema_name = "agent_memories_test_tenant"
    wiki_schema_name = "wiki_pages_test_tenant"
    provenance_schema_name = "provenance_test_tenant"

    for schema_name, base_name, schema_dict in [
        (tenant_schema_name, "agent_memories", memory_schema_json),
        (wiki_schema_name, "wiki_pages", wiki_schema_json),
        (provenance_schema_name, "provenance", provenance_schema_json),
    ]:
        config_manager.set_config_value(
            tenant_id="test_tenant",
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=schema_name,
            config_value={
                "tenant_id": "test_tenant",
                "base_schema_name": base_name,
                "full_schema_name": schema_name,
                "schema_definition": json.dumps(schema_dict),
                "config": {},
                "deployment_time": "2026-04-06T00:00:00",
                "deleted": False,
            },
        )

    BackendRegistry._backend_instances.clear()

    if not wait_for_data_port_ready(http_port, timeout=120):
        pytest.fail(
            f"Vespa data port {http_port} not ready 120s after data-schema deploy"
        )

    if not wait_for_schema_ready(http_port, tenant_schema_name, timeout=120):
        pytest.fail(f"Schema {tenant_schema_name} not ready 120s after deploy")

    if not wait_for_schema_ready(http_port, wiki_schema_name, timeout=120):
        pytest.fail(f"Schema {wiki_schema_name} not ready 120s after deploy")

    yield {
        "http_port": http_port,
        "config_port": config_port,
        "container_name": shared_vespa["container_name"],
        "base_url": shared_vespa["base_url"],
        "tenant_schema_name": tenant_schema_name,
        "wiki_schema_name": wiki_schema_name,
        "config_manager": config_manager,
        "schema_loader": FilesystemSchemaLoader(schemas_dir),
    }
    # No teardown — shared_vespa owns the container lifecycle.
