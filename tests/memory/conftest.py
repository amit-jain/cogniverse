"""Shared fixtures for memory integration tests."""

import logging
import platform
import subprocess
from pathlib import Path

import pytest
import requests

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from tests.utils.async_polling import wait_for_service_startup, wait_for_vespa_indexing
from tests.utils.docker_utils import generate_unique_ports

logger = logging.getLogger(__name__)

# Random ephemeral ports — never hardcode. Hardcoded ports collide with
# whatever else is running on the host (OpenShell on 8080, etc.) and silently
# route test traffic to the wrong service.
MEMORY_BACKEND_PORT, MEMORY_BACKEND_CONFIG_PORT = generate_unique_ports(
    "tests.memory.conftest"
)
MEMORY_BACKEND_CONTAINER = f"backend-memory-tests-{MEMORY_BACKEND_PORT}"


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


def deploy_memory_schema_for_tests(
    tenant_id: str,
    base_schema_name: str,
    backend_url: str,
    backend_config_port: int,
) -> str:
    """
    Deploy memory schema for tests using SchemaRegistry.

    Uses backend abstraction layer for all schema deployment.

    Returns:
        Tenant schema name that was deployed
    """
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import (
        create_default_config_manager,
        get_config,
    )

    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    # Get backend type from tenant's config (REQUIRED - no fallback)
    config = get_config(tenant_id, config_manager)
    backend_config = config.get("backend")
    if not backend_config or "type" not in backend_config:
        raise ValueError(
            f"Backend type not configured for tenant {tenant_id}. "
            "Config must have 'backend.type' field."
        )
    backend_type = backend_config["type"]

    # Get backend via BackendRegistry
    registry = BackendRegistry.get_instance()
    backend_config_dict = {
        "backend": {
            "url": backend_url,
            "config_port": backend_config_port,
            "port": MEMORY_BACKEND_PORT,
        }
    }
    backend = registry.get_search_backend(
        name=backend_type,
        config=backend_config_dict,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    return backend.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name=base_schema_name
    )


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


def _cleanup_leftover_memory_test_containers() -> None:
    """Remove any backend-memory-tests-* containers leaked by prior pytest runs.

    A container leaks when pytest is killed before its teardown can run
    (Ctrl-C, OOM, kill -9, etc.). The container names embed the random
    port so they don't collide with the current run, but they accumulate
    over time and consume Docker resources.
    """
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}",
            "--filter",
            "name=^backend-memory-tests-",
        ],
        capture_output=True,
        text=True,
    )
    for name in result.stdout.splitlines():
        name = name.strip()
        if not name:
            continue
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(scope="session")
def shared_memory_vespa():
    """Session-scoped Vespa backend container for all memory tests."""
    logger.info(
        f"Starting shared backend container {MEMORY_BACKEND_CONTAINER} "
        f"(data={MEMORY_BACKEND_PORT}, config={MEMORY_BACKEND_CONFIG_PORT})"
    )

    _cleanup_leftover_memory_test_containers()

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            MEMORY_BACKEND_CONTAINER,
            "-p",
            f"{MEMORY_BACKEND_PORT}:8080",
            "-p",
            f"{MEMORY_BACKEND_CONFIG_PORT}:19071",
            "--platform",
            docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        text=True,
    )

    try:
        if result.returncode != 0:
            pytest.fail(f"Failed to start backend container: {result.stderr}")

        if not wait_for_backend_ready(MEMORY_BACKEND_CONFIG_PORT, timeout=120):
            pytest.fail("Backend config port failed to start within 120 seconds")

        # Config port being ready doesn't mean data port is ready for document operations.
        import time

        time.sleep(10)

        Mem0MemoryManager._instances.clear()

        from cogniverse_core.registries.backend_registry import BackendRegistry

        BackendRegistry._backend_instances.clear()

        from pathlib import Path

        from vespa.package import ApplicationPackage

        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_sdk.interfaces.config_store import ConfigScope
        from cogniverse_vespa.config.config_store import VespaConfigStore
        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        parser = JsonSchemaParser()
        import json

        with open(Path("configs/schemas/agent_memories_schema.json")) as f:
            memory_schema_json = json.load(f)
        memory_schema_json["name"] = "agent_memories_test_tenant"
        memory_schema_json["document"]["name"] = "agent_memories_test_tenant"
        memory_schema = parser.parse_schema(memory_schema_json)

        with open(Path("configs/schemas/wiki_pages_schema.json")) as f:
            wiki_schema_json = json.load(f)
        wiki_schema_json["name"] = "wiki_pages_test_tenant"
        wiki_schema_json["document"]["name"] = "wiki_pages_test_tenant"
        wiki_schema = parser.parse_schema(wiki_schema_json)

        all_schemas = metadata_schemas + [memory_schema, wiki_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        schema_mgr = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=MEMORY_BACKEND_CONFIG_PORT,
        )
        schema_mgr._deploy_package(app_package)

        for _ in range(60):
            try:
                resp = requests.get(
                    f"http://localhost:{MEMORY_BACKEND_PORT}/state/v1/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)

        config_store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=MEMORY_BACKEND_PORT,
        )
        config_manager = ConfigManager(store=config_store)
        config_manager.set_system_config(
            SystemConfig(
                backend_url="http://localhost",
                backend_port=MEMORY_BACKEND_PORT,
            )
        )

        # Register deployed schemas in ConfigStore so any SchemaRegistry created
        # by downstream fixtures finds them and doesn't attempt redeployment.
        tenant_schema_name = "agent_memories_test_tenant"
        wiki_schema_name = "wiki_pages_test_tenant"

        for schema_name, base_name in [
            (tenant_schema_name, "agent_memories"),
            (wiki_schema_name, "wiki_pages"),
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
                    "schema_definition": "{}",
                    "config": {},
                    "deployment_time": "2026-04-06T00:00:00",
                    "deleted": False,
                },
            )

        BackendRegistry._backend_instances.clear()

        if not wait_for_data_port_ready(MEMORY_BACKEND_PORT, timeout=120):
            pytest.fail(
                f"Vespa HTTP container node (port {MEMORY_BACKEND_PORT}) not ready within "
                "120 seconds after schema deployment."
            )

        if not wait_for_schema_ready(
            MEMORY_BACKEND_PORT, tenant_schema_name, timeout=120
        ):
            pytest.fail(
                f"Schema {tenant_schema_name} not ready within 120 seconds — "
                "data port did not converge after schema deployment."
            )

        if not wait_for_schema_ready(
            MEMORY_BACKEND_PORT, wiki_schema_name, timeout=120
        ):
            pytest.fail(
                f"Schema {wiki_schema_name} not ready within 120 seconds — "
                "data port did not converge after schema deployment."
            )

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        backend_config = {
            "http_port": MEMORY_BACKEND_PORT,
            "config_port": MEMORY_BACKEND_CONFIG_PORT,
            "container_name": MEMORY_BACKEND_CONTAINER,
            "base_url": f"http://localhost:{MEMORY_BACKEND_PORT}",
            "tenant_schema_name": tenant_schema_name,
            "wiki_schema_name": wiki_schema_name,
            "config_manager": config_manager,
            "schema_loader": FilesystemSchemaLoader(Path("configs/schemas")),
        }

        yield backend_config

    finally:
        subprocess.run(
            ["docker", "rm", "-f", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
