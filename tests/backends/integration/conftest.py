"""
Integration test configuration for backend tests.

Provides Vespa Docker instance fixture for testing schema lifecycle.
"""

import logging
import subprocess
import time

import pytest
import requests

from tests.utils.docker_utils import cleanup_vespa_container, generate_unique_ports

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for backend integration tests.

    Uses unique ports per test module to avoid conflicts with:
    - Main Vespa (8080)
    - System tests (different module, different ports)
    - Other test modules (deterministic hash-based port assignment)

    Yields:
        dict: Vespa connection info with keys:
            - http_port: Vespa HTTP port (unique per module)
            - config_port: Vespa config server port (unique per module)
            - base_url: Full HTTP URL
            - container_name: Docker container name

    Example:
        def test_schema_deployment(vespa_instance):
            manager = TenantSchemaManager(
                vespa_url="http://localhost",
                vespa_port=vespa_instance["http_port"]
            )
    """
    # Generate unique ports based on test module name
    http_port, config_port = generate_unique_ports(__name__)
    container_name = f"vespa-backend-test-{http_port}"

    logger.info(
        f"Backend test using unique ports: HTTP={http_port}, Config={config_port}"
    )

    # Stop and remove existing container if exists
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    # Detect platform
    import platform

    machine = platform.machine().lower()
    if machine in ["arm64", "aarch64"]:
        docker_platform = "linux/arm64"
        logger.info(f"Using ARM64 platform for {machine} architecture")
    else:
        docker_platform = "linux/amd64"
        logger.info(f"Using AMD64 platform for {machine} architecture")

    # Start Vespa Docker container
    logger.info(f"Starting Vespa container '{container_name}' on port {http_port}")
    try:
        docker_result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{http_port}:8080",  # Map container 8080 to test port
                "-p",
                f"{config_port}:19071",  # Map config server port
                "--platform",
                docker_platform,
                "vespaengine/vespa",
            ],
            capture_output=True,
            timeout=60,
        )

        if docker_result.returncode != 0:
            pytest.skip(
                f"Failed to start Docker container: {docker_result.stderr.decode()}"
            )

        logger.info(f"✅ Vespa Docker container '{container_name}' started")

        # Wait for config server to be ready
        logger.info("Waiting for Vespa config server...")
        for i in range(120):  # 2 minutes timeout
            try:
                response = requests.get(f"http://localhost:{config_port}/", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Config server ready on port {config_port}")
                    break
            except Exception:
                pass
            wait_for_vespa_indexing(delay=1)
            if i % 10 == 0 and i > 0:
                logger.info(f"  Still waiting... ({i}s)")
        else:
            # Cleanup and skip
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            pytest.skip("Config server not ready after 120 seconds")

        # Deploy base schemas before waiting for application endpoint
        logger.info("Deploying base schemas...")
        try:
            from datetime import datetime, timedelta
            from pathlib import Path

            from cogniverse_vespa.json_schema_parser import JsonSchemaParser
            from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
            from vespa.package import ApplicationPackage, Validation

            app_package = ApplicationPackage(name="videosearch")
            parser = JsonSchemaParser()

            # Load base video schemas
            schemas_dir = Path("configs/schemas")
            if schemas_dir.exists():
                for schema_file in schemas_dir.glob("*_schema.json"):
                    try:
                        schema = parser.load_schema_from_json_file(str(schema_file))
                        app_package.add_schema(schema)
                        logger.info(f"  Added schema: {schema.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {schema_file.name}: {e}")

            # Add metadata schemas (using consolidated module)
            from cogniverse_vespa.metadata_schemas import (
                add_metadata_schemas_to_package,
            )

            add_metadata_schemas_to_package(app_package)

            # Add validation overrides
            until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            app_package.validations = [
                Validation(validation_id="schema-removal", until=until_date),
                Validation(validation_id="content-cluster-removal", until=until_date),
            ]

            # Deploy schemas
            schema_manager = VespaSchemaManager(
                vespa_endpoint="http://localhost", vespa_port=config_port
            )
            schema_manager._deploy_package(app_package)
            logger.info("✅ Schemas deployed")

        except Exception as e:
            logger.warning(f"Schema deployment failed: {e}")
            # Continue anyway - some tests might not need schemas

        # Wait for application endpoint to be ready
        logger.info(f"Waiting for application endpoint on port {http_port}...")
        for i in range(60):  # 1 minute timeout
            try:
                response = requests.get(
                    f"http://localhost:{http_port}/ApplicationStatus", timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"✅ Vespa ready on port {http_port}")
                    break
            except Exception:
                pass
            wait_for_vespa_indexing(delay=2)
        else:
            # Cleanup and skip
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            pytest.skip("Application endpoint not ready after 60 seconds")

        # Yield instance info
        yield {
            "http_port": http_port,
            "config_port": config_port,
            "base_url": f"http://localhost:{http_port}",
            "container_name": container_name,
        }

    except Exception as e:
        logger.error(f"Failed to start Vespa instance: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        # Cleanup: Stop, remove, and wait for full resource release
        logger.info(f"Stopping Docker container '{container_name}'")
        if cleanup_vespa_container(container_name, timeout=30):
            logger.info("✅ Container fully removed and resources released")
        else:
            logger.warning("⚠️  Container cleanup may not have completed fully")

        # Clear singleton state to avoid interference with other test modules
        try:
            from cogniverse_core.config.manager import ConfigManager
            from cogniverse_core.registries.backend_registry import get_backend_registry
            from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

            # Clear TenantSchemaManager singleton
            TenantSchemaManager._clear_instance()

            # Clear backend registry instances
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()

            # Clear ConfigManager singleton
            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None

            logger.info("✅ Cleared singleton state")
        except Exception as e:
            logger.warning(f"⚠️  Error clearing singleton state: {e}")
