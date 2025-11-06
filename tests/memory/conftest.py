"""
Shared fixtures for memory integration tests.

Provides session-scoped backend container that:
1. Starts once for entire test session
2. Deploys memory schemas once
3. Tests clean up documents (not schemas)
4. Stops after all tests complete
"""

import json
import platform
import subprocess
from pathlib import Path

import pytest
import requests
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

from tests.utils.async_polling import wait_for_service_startup, wait_for_vespa_indexing

# Shared backend configuration for all memory tests
MEMORY_BACKEND_PORT = 8081
MEMORY_BACKEND_CONFIG_PORT = 19072
MEMORY_BACKEND_CONTAINER = "backend-memory-tests"


def wait_for_backend_ready(config_port: int, timeout: int = 120) -> bool:
    """Wait for backend to be ready to accept requests"""
    print(f"⏳ Waiting for backend to be ready on port {config_port}...")
    for i in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus",
                timeout=2,
            )
            if response.status_code == 200:
                print(f"✅ Backend ready after {i + 1} seconds")
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Backend container startup")

    print(f"❌ Backend not ready after {timeout} seconds")
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
    from cogniverse_core.config.manager import ConfigManager
    from cogniverse_core.config.utils import get_config
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    print(f"📦 Deploying {base_schema_name} for {tenant_id}...")

    # Create dependencies for backend abstraction
    config_manager = ConfigManager()
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
        tenant_id=tenant_id,
        config=backend_config_dict,
        config_manager=config_manager,
        schema_loader=schema_loader
    )

    # Deploy via SchemaRegistry
    tenant_schema_name = backend.schema_registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name=base_schema_name
    )

    print(f"✅ Deployed {tenant_schema_name}")
    return tenant_schema_name


def wait_for_schema_ready(data_port: int, schema_name: str, timeout: int = 60) -> bool:
    """Wait for schema to be ready to accept documents"""
    print(f"⏳ Waiting for schema {schema_name} to be ready on port {data_port}...")

    test_doc = {
        "fields": {
            "id": "readiness_check",
            "text": "test",
            "user_id": "test",
            "agent_id": "test",
            "embedding": [0.0] * 768,
            "metadata_": "{}",
            "created_at": 1234567890,
        }
    }

    for i in range(timeout):
        try:
            # Use correct Vespa document ID format: id:<namespace>:<schema>::<id>
            response = requests.post(
                f"http://localhost:{data_port}/document/v1/video/{schema_name}/docid/readiness_check",
                json=test_doc,
                timeout=2,
            )
            if response.status_code in [200, 201]:
                # Cleanup test document
                requests.delete(
                    f"http://localhost:{data_port}/document/v1/video/{schema_name}/docid/readiness_check",
                    timeout=2,
                )
                print(f"✅ Schema {schema_name} ready after {i + 1} seconds")
                return True
            elif i % 10 == 0:  # Log non-success status codes every 10 attempts
                print(f"   Attempt {i+1}: Status {response.status_code}: {response.text[:100]}")
        except Exception as e:
            # Log every 10th attempt to avoid spam
            if i % 10 == 0:
                print(f"   Attempt {i+1}: Readiness check error: {type(e).__name__}: {e}")
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
    print(f"   Port: {MEMORY_BACKEND_PORT} (data), {MEMORY_BACKEND_CONFIG_PORT} (config)")
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
        subprocess.run(["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True)
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail("Backend config port failed to start within 120 seconds")

    # Give Vespa additional time to fully initialize all services
    # Config port being ready doesn't mean data port is ready for document operations
    import time
    print(f"⏳ Waiting additional 10 seconds for Vespa services to fully initialize...")
    time.sleep(10)
    print(f"✅ Vespa initialization complete")

    # Deploy memory schema using the SAME approach as working backend tests
    print("\n📦 Deploying agent_memories schema...")

    # Clear backend registry cache to ensure fresh state
    Mem0MemoryManager._instances.clear()

    # Clear backend registry cache to force recreation with profiles
    from cogniverse_core.registries.backend_registry import BackendRegistry
    BackendRegistry._backend_instances.clear()

    try:
        # Deploy application package with tenant-specific schema (same as working backend tests)
        from pathlib import Path
        from vespa.package import ApplicationPackage
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_core.config.manager import ConfigManager
        import tempfile

        # Create application package
        app_package = ApplicationPackage(name="memory")
        parser = JsonSchemaParser()

        # Load agent_memories schema and rename it to tenant-specific name
        schema_file = Path("configs/schemas/agent_memories_schema.json")
        schema = parser.load_schema_from_json_file(str(schema_file))

        # Rename schema to tenant-specific name
        tenant_schema_name = "agent_memories_test_tenant"
        schema.name = tenant_schema_name
        app_package.add_schema(schema)
        print(f"  Added schema: {schema.name}")

        # Deploy using VespaSchemaManager (same as working tests)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            config_manager = ConfigManager(db_path=Path(tmp_db.name))
            schema_manager = VespaSchemaManager(
                backend_endpoint="http://localhost",
                backend_port=MEMORY_BACKEND_CONFIG_PORT,
                config_manager=config_manager
            )
            schema_manager._deploy_package(app_package)
            print("✅ Schema deployment completed")

            # Cleanup temporary database
            Path(tmp_db.name).unlink(missing_ok=True)

        # Clear backend cache AGAIN after schema deployment
        # This ensures the memory manager gets a fresh backend WITH profiles
        print("🔄 Clearing backend cache after schema deployment...")
        BackendRegistry._backend_instances.clear()
        print("✅ Backend cache cleared - memory manager will create fresh instance with profiles")

    except Exception as e:
        # Cleanup on failure
        subprocess.run(["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True)
        subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
        pytest.fail(f"Failed to deploy schema: {e}")

    # Wait for schema to be fully ready
    if not wait_for_schema_ready(MEMORY_BACKEND_PORT, tenant_schema_name, timeout=60):
        print("⚠️  Warning: Schema readiness check failed, but continuing anyway...")
        print(
            "   This may indicate deployment issues or readiness check needs updating"
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
    }

    yield backend_config

    # Cleanup container after tests
    subprocess.run(["docker", "stop", MEMORY_BACKEND_CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", MEMORY_BACKEND_CONTAINER], capture_output=True)
