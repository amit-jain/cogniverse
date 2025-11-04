"""
Shared Vespa Docker container management for all tests.

Consolidates duplicate code from:
- tests/backends/integration/conftest.py
- tests/system/vespa_test_manager.py
"""

import logging
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import requests
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from cogniverse_vespa.metadata_schemas import add_metadata_schemas_to_package
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from vespa.package import ApplicationPackage, Validation

from tests.utils.async_polling import wait_for_vespa_indexing
from tests.utils.docker_utils import cleanup_vespa_container, generate_unique_ports

logger = logging.getLogger(__name__)


class VespaDockerManager:
    """Manages Vespa Docker containers for testing with automatic cleanup"""

    def __init__(self):
        self.container_name: Optional[str] = None
        self.http_port: Optional[int] = None
        self.config_port: Optional[int] = None

    def start_container(
        self,
        module_name: str,
        use_module_ports: bool = True
    ) -> Dict[str, any]:
        """
        Start isolated Vespa Docker container with unique ports.

        Args:
            module_name: Test module name (used for port generation and container naming)
            use_module_ports: If True, generate ports based on module name hash.
                            If False, use sequential ports (8081, 19072)

        Returns:
            dict: Container info with keys:
                - container_name: Docker container name
                - http_port: Vespa HTTP port
                - config_port: Vespa config server port
                - base_url: Full HTTP URL

        Raises:
            RuntimeError: If container fails to start
        """
        # Generate unique ports
        if use_module_ports:
            http_port, config_port = generate_unique_ports(module_name)
        else:
            # Sequential ports for system tests
            http_port = 8081
            config_port = 19072

        container_name = f"vespa-test-{http_port}"

        logger.info(f"Starting Vespa container '{container_name}' (HTTP={http_port}, Config={config_port})")

        # Stop and remove existing container if exists
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

        # Detect platform
        machine = platform.machine().lower()
        if machine in ["arm64", "aarch64"]:
            docker_platform = "linux/arm64"
            logger.info(f"Using ARM64 platform for {machine} architecture")
        else:
            docker_platform = "linux/amd64"
            logger.info(f"Using AMD64 platform for {machine} architecture")

        # Start Vespa Docker container
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
                raise RuntimeError(
                    f"Failed to start Docker container: {docker_result.stderr.decode()}"
                )

            logger.info(f"✅ Vespa Docker container '{container_name}' started")

            # Store container info
            self.container_name = container_name
            self.http_port = http_port
            self.config_port = config_port

            return {
                "container_name": container_name,
                "http_port": http_port,
                "config_port": config_port,
                "base_url": f"http://localhost:{http_port}",
            }

        except Exception as e:
            # Cleanup on failure
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            raise RuntimeError(f"Failed to start Vespa container: {e}") from e

    def wait_for_config_ready(
        self,
        container_info: Dict[str, any],
        timeout: int = 120
    ):
        """
        Wait for Vespa config server to be ready.

        Args:
            container_info: Container info dict from start_container()
            timeout: Timeout in seconds

        Raises:
            RuntimeError: If config server doesn't become ready within timeout
        """
        config_port = container_info["config_port"]

        logger.info("Waiting for Vespa config server...")
        for i in range(timeout):
            try:
                response = requests.get(f"http://localhost:{config_port}/", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Config server ready on port {config_port}")
                    return
            except Exception:
                pass
            wait_for_vespa_indexing(delay=1)
            if i % 10 == 0 and i > 0:
                logger.info(f"  Still waiting... ({i}s)")
        else:
            raise RuntimeError(
                f"Config server not ready after {timeout} seconds"
            )

    def wait_for_application_ready(
        self,
        container_info: Dict[str, any],
        timeout: int = 60
    ):
        """
        Wait for Vespa application endpoint to be ready.

        IMPORTANT: Must be called AFTER schemas are deployed!

        Args:
            container_info: Container info dict from start_container()
            timeout: Timeout in seconds

        Raises:
            RuntimeError: If application doesn't become ready within timeout
        """
        http_port = container_info["http_port"]

        logger.info(f"Waiting for application endpoint on port {http_port}...")
        for i in range(timeout):
            try:
                response = requests.get(
                    f"http://localhost:{http_port}/ApplicationStatus", timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"✅ Vespa ready on port {http_port}")
                    return
            except Exception:
                pass
            wait_for_vespa_indexing(delay=2)
        else:
            raise RuntimeError(
                f"Application endpoint not ready after {timeout} seconds"
            )

    def deploy_schemas(
        self,
        container_info: Dict[str, any],
        schemas_dir: Optional[Path] = None,
        include_metadata: bool = True
    ):
        """
        Deploy schemas to Vespa container.

        Args:
            container_info: Container info dict from start_container()
            schemas_dir: Directory containing schema JSON files.
                        Defaults to configs/schemas/
            include_metadata: If True, add metadata schemas

        Raises:
            RuntimeError: If schema deployment fails
        """
        config_port = container_info["config_port"]

        # Default to project schemas directory
        if schemas_dir is None:
            schemas_dir = Path("configs/schemas")

        if not schemas_dir.exists():
            raise RuntimeError(f"Schemas directory not found: {schemas_dir}")

        logger.info(f"Deploying schemas from {schemas_dir}...")

        try:
            app_package = ApplicationPackage(name="videosearch")
            parser = JsonSchemaParser()

            # Load base video schemas
            schema_files = list(schemas_dir.glob("*_schema.json"))
            if not schema_files:
                raise RuntimeError(f"No schema files found in {schemas_dir}")

            for schema_file in schema_files:
                try:
                    schema = parser.load_schema_from_json_file(str(schema_file))
                    app_package.add_schema(schema)
                    logger.info(f"  Added schema: {schema.name}")
                except Exception as e:
                    logger.warning(f"  Failed to load {schema_file.name}: {e}")

            # Add metadata schemas
            if include_metadata:
                add_metadata_schemas_to_package(app_package)
                logger.info("  Added metadata schemas")

            # Add validation overrides
            until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            app_package.validations = [
                Validation(validation_id="schema-removal", until=until_date),
                Validation(validation_id="content-cluster-removal", until=until_date),
            ]

            # Deploy schemas - create temporary ConfigManager for test infrastructure
            import tempfile

            from cogniverse_core.config.manager import ConfigManager

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
                config_manager = ConfigManager(db_path=Path(tmp_db.name))
                schema_manager = VespaSchemaManager(
                    vespa_endpoint="http://localhost",
                    vespa_port=config_port,
                    config_manager=config_manager
                )
                schema_manager._deploy_package(app_package)
                logger.info("✅ Schemas deployed successfully")

            # Cleanup temporary database
            Path(tmp_db.name).unlink(missing_ok=True)

        except Exception as e:
            raise RuntimeError(f"Schema deployment failed: {e}") from e

    def stop_container(self, container_info: Optional[Dict[str, any]] = None):
        """
        Stop and clean up Vespa container.

        Args:
            container_info: Container info dict from start_container().
                          If None, uses internally stored container name.
        """
        if container_info:
            container_name = container_info["container_name"]
        else:
            container_name = self.container_name

        if not container_name:
            logger.warning("No container to stop")
            return

        logger.info(f"Stopping Docker container '{container_name}'")
        if cleanup_vespa_container(container_name, timeout=30):
            logger.info("✅ Container fully removed and resources released")
        else:
            logger.warning("⚠️  Container cleanup may not have completed fully")

        # Clear stored state
        self.container_name = None
        self.http_port = None
        self.config_port = None
