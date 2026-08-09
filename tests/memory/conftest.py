"""Shared fixtures for memory integration tests."""

import logging
from pathlib import Path

import pytest
import requests

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from tests.utils.async_polling import wait_for_service_startup, wait_for_vespa_indexing
from tests.utils.tenant_helpers import MEM0_ROUNDTRIP_TENANT_ID, MEMORY_TENANT_ID
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    make_config_manager,
    schema_full_name,
)

logger = logging.getLogger(__name__)

# Schemas the fixture deploys, per tenant. SchemaRegistry canonicalizes a
# bare tenant id (``test_tenant`` → ``test_tenant:test_tenant``), so the
# derived names are computed here rather than written out.
MEMORY_BASE_SCHEMAS = ("agent_memories", "wiki_pages", "provenance")
MEM0_ROUNDTRIP_BASE_SCHEMAS = ("agent_memories",)

MEMORY_SCHEMA = schema_full_name("agent_memories", MEMORY_TENANT_ID)
WIKI_SCHEMA = schema_full_name("wiki_pages", MEMORY_TENANT_ID)
PROVENANCE_SCHEMA = schema_full_name("provenance", MEMORY_TENANT_ID)
MEM0_ROUNDTRIP_SCHEMA = schema_full_name("agent_memories", MEM0_ROUNDTRIP_TENANT_ID)


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


def deploy_memory_schemas(
    shared_vespa: dict,
    *,
    config_manager=None,
    force: bool = False,
) -> dict:
    """Deploy the memory suite's tenant schemas through SchemaRegistry.

    ``deploy_schema`` collects every schema already registered for every
    tenant and ships the complete list to Vespa, so a peer tenant's schema
    survives the redeploy. Returns ``{(tenant_id, base_schema_name):
    full_schema_name}``.
    """
    deployed = {}
    for tenant_id, base_names in (
        (MEMORY_TENANT_ID, MEMORY_BASE_SCHEMAS),
        (MEM0_ROUNDTRIP_TENANT_ID, MEM0_ROUNDTRIP_BASE_SCHEMAS),
    ):
        for base_schema_name in base_names:
            full_name = deploy_tenant_schema(
                shared_vespa,
                tenant_id=tenant_id,
                base_schema_name=base_schema_name,
                config_manager=config_manager,
                force=force,
            )
            expected = schema_full_name(base_schema_name, tenant_id)
            assert full_name == expected, (
                f"deploy_schema returned {full_name!r} for "
                f"{base_schema_name!r}/{tenant_id!r}; memory tests address "
                f"{expected!r}"
            )
            deployed[(tenant_id, base_schema_name)] = full_name
    return deployed


@pytest.fixture(scope="session")
def shared_memory_vespa(shared_vespa):
    """The session-wide Vespa with the memory suite's schemas deployed.

    Deploys ``agent_memories``, ``wiki_pages`` and ``provenance`` for
    ``MEMORY_TENANT_ID`` plus ``agent_memories`` for
    ``MEM0_ROUNDTRIP_TENANT_ID`` through SchemaRegistry, which merges every
    already-registered schema into the deployment package. Other packages
    hold their own tenants on the same container, and Vespa reads a
    deployment package as the cluster's complete schema list, so the merge
    is what keeps their schemas alive.

    Yields::

        {
            "http_port", "config_port", "container_name", "base_url",
            "tenant_schema_name": <agent_memories_...>,
            "wiki_schema_name":   <wiki_pages_...>,
            "provenance_schema_name": <provenance_...>,
            "mem0_roundtrip_schema_name": <agent_memories_... for the
                                           round-trip tenant>,
            "config_manager":     <ConfigManager bound to shared_vespa>,
            "schema_loader":      <FilesystemSchemaLoader>,
        }
    """
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    config_port = shared_vespa["config_port"]
    http_port = shared_vespa["http_port"]

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()

    config_manager = make_config_manager(shared_vespa)
    deployed = deploy_memory_schemas(shared_vespa, config_manager=config_manager)

    BackendRegistry._backend_instances.clear()

    if not wait_for_data_port_ready(http_port, timeout=120):
        pytest.fail(
            f"Vespa data port {http_port} not ready 120s after data-schema deploy"
        )

    for schema_name in (
        deployed[(MEMORY_TENANT_ID, "agent_memories")],
        deployed[(MEMORY_TENANT_ID, "wiki_pages")],
        deployed[(MEM0_ROUNDTRIP_TENANT_ID, "agent_memories")],
    ):
        if not wait_for_schema_ready(http_port, schema_name, timeout=120):
            pytest.fail(f"Schema {schema_name} not ready 120s after deploy")

    yield {
        "http_port": http_port,
        "config_port": config_port,
        "container_name": shared_vespa["container_name"],
        "base_url": shared_vespa["base_url"],
        "tenant_schema_name": deployed[(MEMORY_TENANT_ID, "agent_memories")],
        "wiki_schema_name": deployed[(MEMORY_TENANT_ID, "wiki_pages")],
        "provenance_schema_name": deployed[(MEMORY_TENANT_ID, "provenance")],
        "mem0_roundtrip_schema_name": deployed[
            (MEM0_ROUNDTRIP_TENANT_ID, "agent_memories")
        ],
        "config_manager": config_manager,
        "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
    }
    # No teardown — shared_vespa owns the container lifecycle.
