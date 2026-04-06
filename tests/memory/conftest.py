"""
Shared fixtures for memory integration tests.

Provides session-scoped backend container that:
1. Starts once for entire test session
2. Deploys memory schemas once
3. Tests clean up documents (not schemas)
4. Stops after all tests complete
"""

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

# Shared backend configuration for all memory tests
MEMORY_BACKEND_PORT = 8081
MEMORY_BACKEND_CONFIG_PORT = 19072
MEMORY_BACKEND_CONTAINER = "backend-memory-tests"


def wait_for_backend_ready(config_port: int, timeout: int = 120) -> bool:
    """Wait for backend config server to be ready."""
    print(f"⏳ Waiting for backend config server on port {config_port}...")
    for i in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus",
                timeout=2,
            )
            if response.status_code == 200:
                print(f"✅ Backend config server ready after {i + 1} seconds")
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Backend container startup")

    print(f"❌ Backend config server not ready after {timeout} seconds")
    return False


def wait_for_data_port_ready(data_port: int, timeout: int = 120) -> bool:
    """Wait for Vespa HTTP container node (data port) to respond with 200.

    The config port (19071) becomes ready well before the HTTP container node
    (8080) starts. After schema deployment the container node needs additional
    time to initialize. This probe uses GET /ApplicationStatus on the data port
    so it returns True only once the container node is fully up.
    """
    print(f"⏳ Waiting for Vespa HTTP container node on port {data_port}...")
    for i in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{data_port}/ApplicationStatus",
                timeout=5,
            )
            if response.status_code == 200:
                print(f"✅ Vespa HTTP container node ready after {i + 1} seconds")
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Data port readiness")

    print(f"❌ Vespa HTTP container node not ready after {timeout} seconds")
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

    print(f"📦 Deploying {base_schema_name} for {tenant_id}...")

    # Create dependencies for backend abstraction
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

    # Deploy via SchemaRegistry
    tenant_schema_name = backend.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name=base_schema_name
    )

    print(f"✅ Deployed {tenant_schema_name}")
    return tenant_schema_name


def _get_real_embedding(text: str = "readiness check") -> list:
    """Get a real embedding from Ollama nomic-embed-text for schema probes."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["embeddings"][0]
    except Exception:
        pass
    # Fallback only for readiness probes — tests themselves must never use this
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
    print(
        f"⏳ Waiting for schema {schema_name} (namespace={namespace}) "
        f"to be ready on port {data_port}..."
    )

    test_doc = _readiness_doc_for_namespace(namespace)

    for i in range(timeout):
        try:
            response = requests.post(
                f"http://localhost:{data_port}/document/v1/{namespace}/{schema_name}/docid/readiness_check",
                json=test_doc,
                timeout=5,
            )
            if response.status_code in [200, 201]:
                # Cleanup test document
                requests.delete(
                    f"http://localhost:{data_port}/document/v1/{namespace}/{schema_name}/docid/readiness_check",
                    timeout=5,
                )
                print(f"✅ Schema {schema_name} ready after {i + 1} seconds")
                return True
            elif i % 10 == 0:  # Log non-success status codes every 10 attempts
                print(
                    f"   Attempt {i + 1}: Status {response.status_code}: {response.text[:100]}"
                )
        except Exception as e:
            # Log every 10th attempt to avoid spam
            if i % 10 == 0:
                print(
                    f"   Attempt {i + 1}: Readiness check error: {type(e).__name__}: {e}"
                )
        wait_for_vespa_indexing(delay=1.0, description="schema readiness check")

    print(f"❌ Schema {schema_name} not ready after {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def shared_memory_vespa():
    """
    Session-scoped backend instance for all memory tests.

    Starts once, deploys schemas once, used by all tests.
    Tests are responsible for cleaning up their own documents.
    """
    print("\n" + "=" * 70)
    print("🚀 Starting shared backend container for memory tests...")
    print(
        f"   Port: {MEMORY_BACKEND_PORT} (data), {MEMORY_BACKEND_CONFIG_PORT} (config)"
    )
    print("=" * 70)

    # Stop and remove any existing container
    subprocess.run(
        ["docker", "stop", MEMORY_BACKEND_CONTAINER],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", MEMORY_BACKEND_CONTAINER],
        capture_output=True,
    )

    # Determine platform for Docker
    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    # Start fresh backend container
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

    if result.returncode != 0:
        pytest.fail(f"Failed to start backend container: {result.stderr}")

    print(f"✅ Container started: {result.stdout.strip()}")

    # Wait for backend config port to be ready
    if not wait_for_backend_ready(MEMORY_BACKEND_CONFIG_PORT, timeout=120):
        # Cleanup on failure
        subprocess.run(
            ["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail("Backend config port failed to start within 120 seconds")

    # Give Vespa additional time to fully initialize all services
    # Config port being ready doesn't mean data port is ready for document operations
    import time

    print("⏳ Waiting additional 10 seconds for Vespa services to fully initialize...")
    time.sleep(10)
    print("✅ Vespa initialization complete")

    # Deploy memory schema using the SAME approach as working backend tests
    print("\n📦 Deploying agent_memories schema...")

    # Clear backend registry cache to ensure fresh state
    Mem0MemoryManager._instances.clear()

    # Clear backend registry cache to force recreation with profiles
    from cogniverse_core.registries.backend_registry import BackendRegistry

    BackendRegistry._backend_instances.clear()

    try:
        # Deploy schema via BackendRegistry + SchemaRegistry (correct pattern)
        from pathlib import Path

        # Deploy metadata schemas first — VespaConfigStore needs config_metadata
        # schema to exist before set_system_config() can write to it.
        from vespa.package import ApplicationPackage

        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_foundation.config.unified_config import SystemConfig
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

        # Also include agent_memories schema for test_tenant
        parser = JsonSchemaParser()
        memory_schema_file = Path("configs/schemas/agent_memories_schema.json")
        import json

        with open(memory_schema_file) as f:
            memory_schema_json = json.load(f)
        memory_schema_json["name"] = "agent_memories_test_tenant"
        memory_schema_json["document"]["name"] = "agent_memories_test_tenant"
        memory_schema = parser.parse_schema(memory_schema_json)

        # Include wiki_pages schema for test_tenant
        wiki_schema_file = Path("configs/schemas/wiki_pages_schema.json")
        with open(wiki_schema_file) as f:
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
        print("✅ Deployed metadata + agent_memories schemas")

        # Wait for application ready
        import time

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

        # Create ConfigManager backed by the TEST Vespa
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore

        config_store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=MEMORY_BACKEND_PORT,
        )
        config_manager = ConfigManager(store=config_store)

        system_config = SystemConfig(
            backend_url="http://localhost",
            backend_port=MEMORY_BACKEND_PORT,
        )
        config_manager.set_system_config(system_config)

        # Register deployed schemas in ConfigStore so any new SchemaRegistry
        # created by downstream fixtures (e.g., Mem0's get_ingestion_backend)
        # finds them and doesn't attempt redeployment.
        from cogniverse_sdk.interfaces.config_store import ConfigScope

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

    except Exception as e:
        # Cleanup on failure
        subprocess.run(
            ["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail(f"Failed to deploy schema: {e}")

    # After schema deployment the Vespa HTTP container node (data port) needs
    # additional time to start. Wait until it responds with 200 before testing.
    if not wait_for_data_port_ready(MEMORY_BACKEND_PORT, timeout=120):
        subprocess.run(
            ["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail(
            f"Vespa HTTP container node (port {MEMORY_BACKEND_PORT}) not ready within "
            "120 seconds after schema deployment."
        )

    # Wait for schemas to be fully ready — fail hard if readiness times out.
    # Tests depend on Vespa being able to accept document operations; silently
    # continuing with an unready backend causes cascading test failures.
    if not wait_for_schema_ready(MEMORY_BACKEND_PORT, tenant_schema_name, timeout=120):
        subprocess.run(
            ["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail(
            f"Schema {tenant_schema_name} not ready within 120 seconds — "
            "data port did not converge after schema deployment."
        )

    if not wait_for_schema_ready(MEMORY_BACKEND_PORT, wiki_schema_name, timeout=120):
        subprocess.run(
            ["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True
        )
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail(
            f"Schema {wiki_schema_name} not ready within 120 seconds — "
            "data port did not converge after schema deployment."
        )

    print("\n" + "=" * 70)
    print("✅ Shared backend ready for memory tests")
    print("=" * 70 + "\n")

    backend_config = {
        "http_port": MEMORY_BACKEND_PORT,
        "config_port": MEMORY_BACKEND_CONFIG_PORT,
        "container_name": MEMORY_BACKEND_CONTAINER,
        "base_url": f"http://localhost:{MEMORY_BACKEND_PORT}",
        "tenant_schema_name": tenant_schema_name,
        "wiki_schema_name": wiki_schema_name,
    }

    yield backend_config

    # Cleanup container after tests
    subprocess.run(["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
