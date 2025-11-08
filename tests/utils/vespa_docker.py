"""
Shared Vespa Docker container management for all tests.

Consolidates duplicate code from:
- tests/backends/integration/conftest.py
- tests/system/vespa_test_manager.py
"""

import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional

import requests

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
        include_metadata: bool = True,
        tenant_id: str = "test_tenant"
    ):
        """
        Deploy schemas to Vespa container with tenant-aware naming.

        Args:
            container_info: Container info dict from start_container()
            schemas_dir: Directory containing schema JSON files.
                        Defaults to configs/schemas/
            include_metadata: If True, add metadata schemas
            tenant_id: Tenant identifier for schema deployment (default: "test_tenant")

        Raises:
            RuntimeError: If schema deployment fails
        """
        http_port = container_info["http_port"]

        # Default to project schemas directory
        if schemas_dir is None:
            schemas_dir = Path("configs/schemas")

        if not schemas_dir.exists():
            raise RuntimeError(f"Schemas directory not found: {schemas_dir}")

        logger.info(f"Deploying schemas from {schemas_dir} for tenant '{tenant_id}'...")

        try:
            # Create temporary ConfigManager and SchemaLoader for test infrastructure
            import tempfile

            from cogniverse_core.config.unified_config import BackendConfig
            from cogniverse_core.config.utils import create_default_config_manager
            from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

            # Create temporary database for config
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
                config_manager = create_default_config_manager(db_path=Path(tmp_db.name))

                # Create SchemaLoader pointing to test schemas directory
                schema_loader = FilesystemSchemaLoader(base_path=schemas_dir)

                # Create backend instance with proper dependency injection
                backend_config = BackendConfig(
                    tenant_id=tenant_id,
                    backend_type="vespa",
                    url="http://localhost",
                    port=http_port,
                )

                # Import and instantiate VespaBackend
                from cogniverse_vespa.backend import VespaBackend

                backend = VespaBackend(
                    backend_config=backend_config,
                    schema_loader=schema_loader,
                    config_manager=config_manager
                )

                # Create and inject SchemaRegistry
                from cogniverse_core.registries.schema_registry import SchemaRegistry

                schema_registry = SchemaRegistry(
                    config_manager=config_manager,
                    backend=backend,
                    schema_loader=schema_loader
                )
                backend.schema_registry = schema_registry

                # Initialize backend
                backend.initialize({"tenant_id": tenant_id})

                # Deploy each schema using SchemaRegistry (automatically handles tenant suffixes)
                schema_files = list(schemas_dir.glob("*_schema.json"))
                if not schema_files:
                    raise RuntimeError(f"No schema files found in {schemas_dir}")

                deployed_schemas = []
                for schema_file in schema_files:
                    try:
                        # Extract base schema name from file (remove _schema.json suffix)
                        base_schema_name = schema_file.stem.replace("_schema", "")

                        # Use SchemaRegistry to deploy with tenant suffix
                        tenant_schema_name = schema_registry.deploy_schema(
                            tenant_id=tenant_id,
                            base_schema_name=base_schema_name
                        )
                        deployed_schemas.append(tenant_schema_name)
                        logger.info(f"  Deployed tenant schema: {tenant_schema_name}")
                    except Exception as e:
                        logger.warning(f"  Failed to deploy {schema_file.name}: {e}")
                        raise

                logger.info(f"✅ Deployed {len(deployed_schemas)} tenant-scoped schemas")

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
