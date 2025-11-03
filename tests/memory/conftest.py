"""
Shared fixtures for memory integration tests.

Provides session-scoped Vespa container that:
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
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    Schema,
    Validation,
)

from tests.utils.async_polling import wait_for_service_startup, wait_for_vespa_indexing

# Shared Vespa configuration for all memory tests
MEMORY_VESPA_PORT = 8081
MEMORY_VESPA_CONFIG_PORT = 19072
MEMORY_VESPA_CONTAINER = "vespa-memory-tests"


def wait_for_vespa_ready(config_port: int, timeout: int = 120) -> bool:
    """Wait for Vespa to be ready to accept requests"""
    print(f"⏳ Waiting for Vespa to be ready on port {config_port}...")
    for i in range(timeout):
        try:
            response = requests.get(
                f"http://localhost:{config_port}/ApplicationStatus",
                timeout=2,
            )
            if response.status_code == 200:
                print(f"✅ Vespa ready after {i + 1} seconds")
                return True
        except Exception:
            pass
        wait_for_service_startup(delay=1.0, description="Vespa container startup")

    print(f"❌ Vespa not ready after {timeout} seconds")
    return False


def deploy_memory_schema_for_tests(
    tenant_id: str,
    base_schema_name: str,
    vespa_url: str,
    vespa_config_port: int,
) -> str:
    """
    Deploy memory schema for tests.

    Uses standard PyVespa deployment without explicit ContentCluster configuration,
    same as the rest of the codebase.

    Returns:
        Tenant schema name that was deployed
    """
    from datetime import datetime, timedelta

    # Construct tenant schema name
    tenant_suffix = tenant_id.replace(":", "_")
    tenant_schema_name = f"{base_schema_name}_{tenant_suffix}"

    print(f"📦 Deploying {tenant_schema_name}...")

    # Load base schema
    schema_file = Path("configs/schemas") / f"{base_schema_name}_schema.json"
    with open(schema_file, "r") as f:
        base_schema_json = json.load(f)

    # Transform schema for tenant
    tenant_schema_json = json.loads(json.dumps(base_schema_json))
    tenant_schema_json["name"] = tenant_schema_name
    if "document" in tenant_schema_json:
        tenant_schema_json["document"]["name"] = tenant_schema_name

    # Parse to Vespa Schema object
    parser = JsonSchemaParser()
    schema = parser.parse_schema(tenant_schema_json)

    # Create application package WITHOUT explicit ContentCluster
    # Let PyVespa use default behavior like the rest of the codebase
    app_package = ApplicationPackage(name="memorysearch")

    # Add the tenant schema
    app_package.add_schema(schema)

    # Add metadata schemas to prevent removal
    organization_metadata_schema = Schema(
        name="organization_metadata",
        document=Document(
            fields=[
                Field(
                    name="org_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(name="org_name", type="string", indexing=["summary", "index"]),
                Field(
                    name="created_at", type="long", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_by", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="status",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="tenant_count", type="int", indexing=["summary", "attribute"]
                ),
            ]
        ),
    )
    app_package.add_schema(organization_metadata_schema)

    tenant_metadata_schema = Schema(
        name="tenant_metadata",
        document=Document(
            fields=[
                Field(
                    name="tenant_full_id",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="org_id",
                    type="string",
                    indexing=["summary", "index", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="tenant_name", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_at", type="long", indexing=["summary", "attribute"]
                ),
                Field(
                    name="created_by", type="string", indexing=["summary", "attribute"]
                ),
                Field(
                    name="status",
                    type="string",
                    indexing=["summary", "attribute"],
                    attribute=["fast-search"],
                ),
                Field(
                    name="schemas_deployed",
                    type="array<string>",
                    indexing=["summary", "attribute"],
                ),
            ]
        ),
    )
    app_package.add_schema(tenant_metadata_schema)

    # Add validation overrides
    until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    schema_removal_validation = Validation(
        validation_id="schema-removal", until=until_date
    )
    content_cluster_validation = Validation(
        validation_id="content-cluster-removal", until=until_date
    )

    if app_package.validations is None:
        app_package.validations = []
    app_package.validations.append(schema_removal_validation)
    app_package.validations.append(content_cluster_validation)

    # Deploy via VespaSchemaManager
    schema_manager = VespaSchemaManager(
        vespa_endpoint=vespa_url, backend_port=vespa_config_port
    )
    schema_manager._deploy_package(app_package)

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
            response = requests.post(
                f"http://localhost:{data_port}/document/v1/{schema_name}/{schema_name}/docid/readiness_check",
                json=test_doc,
                timeout=2,
            )
            if response.status_code in [200, 201]:
                # Cleanup test document
                requests.delete(
                    f"http://localhost:{data_port}/document/v1/{schema_name}/{schema_name}/docid/readiness_check",
                    timeout=2,
                )
                print(f"✅ Schema {schema_name} ready after {i + 1} seconds")
                return True
        except Exception:
            pass
        wait_for_vespa_indexing(delay=1.0, description="schema readiness check")

    print(f"❌ Schema {schema_name} not ready after {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def shared_memory_vespa():
    """
    Session-scoped Vespa instance for all memory tests.

    Starts once, deploys schemas once, used by all tests.
    Tests are responsible for cleaning up their own documents.
    """
    print("\n" + "=" * 70)
    print("🚀 Starting shared Vespa container for memory tests...")
    print(f"   Port: {MEMORY_VESPA_PORT} (data), {MEMORY_VESPA_CONFIG_PORT} (config)")
    print("=" * 70)

    # Stop and remove any existing container
    subprocess.run(
        ["docker", "stop", MEMORY_VESPA_CONTAINER],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", MEMORY_VESPA_CONTAINER],
        capture_output=True,
    )

    # Determine platform for Docker
    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    # Start fresh Vespa container
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            MEMORY_VESPA_CONTAINER,
            "-p",
            f"{MEMORY_VESPA_PORT}:8080",
            "-p",
            f"{MEMORY_VESPA_CONFIG_PORT}:19071",
            "--platform",
            docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to start Vespa container: {result.stderr}")

    print(f"✅ Container started: {result.stdout.strip()}")

    # Wait for Vespa to be ready
    if not wait_for_vespa_ready(MEMORY_VESPA_CONFIG_PORT, timeout=120):
        # Cleanup on failure
        subprocess.run(["docker", "stop", MEMORY_VESPA_CONTAINER], capture_output=True)
        subprocess.run(["docker", "rm", MEMORY_VESPA_CONTAINER], capture_output=True)
        pytest.fail("Vespa failed to start within 120 seconds")

    # Deploy memory schema for test_tenant using test-specific deployment
    print("\n📦 Deploying agent_memories schema for test_tenant...")

    # Clear singletons to ensure fresh state
    TenantSchemaManager._instance = None
    Mem0MemoryManager._instances.clear()

    try:
        tenant_schema_name = deploy_memory_schema_for_tests(
            tenant_id="test_tenant",
            base_schema_name="agent_memories",
            backend_url="http://localhost",
            vespa_config_port=MEMORY_VESPA_CONFIG_PORT,
        )
        print("✅ Schema deployment completed")
    except Exception as e:
        # Cleanup on failure
        subprocess.run(["docker", "stop", MEMORY_VESPA_CONTAINER], capture_output=True)
        subprocess.run(["docker", "rm", MEMORY_VESPA_CONTAINER], capture_output=True)
        pytest.fail(f"Failed to deploy schema: {e}")

    # Wait for schema to be fully ready
    if not wait_for_schema_ready(MEMORY_VESPA_PORT, tenant_schema_name, timeout=60):
        print("⚠️  Warning: Schema readiness check failed, but continuing anyway...")
        print(
            "   This may indicate deployment issues or readiness check needs updating"
        )

    print("\n" + "=" * 70)
    print("✅ Shared Vespa ready for memory tests")
    print("=" * 70 + "\n")

    vespa_config = {
        "http_port": MEMORY_VESPA_PORT,
        "config_port": MEMORY_VESPA_CONFIG_PORT,
        "container_name": MEMORY_VESPA_CONTAINER,
        "base_url": f"http://localhost:{MEMORY_VESPA_PORT}",
        "tenant_schema_name": tenant_schema_name,
    }

    yield vespa_config

    # Cleanup
    print("\n" + "=" * 70)
    print("🧹 Cleaning up Vespa container...")
    print("=" * 70)
    subprocess.run(["docker", "stop", MEMORY_VESPA_CONTAINER], capture_output=True)
    subprocess.run(["docker", "rm", MEMORY_VESPA_CONTAINER], capture_output=True)
    print("✅ Cleanup complete")
