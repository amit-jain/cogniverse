"""
Integration test configuration for backend tests.

Provides Vespa Docker instance fixture for testing schema lifecycle.
"""

import pytest
import subprocess
import time
import requests
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance():
    """
    Start isolated Vespa Docker instance for backend integration tests.

    Uses port 8082 to avoid conflicts with:
    - Main Vespa (8080)
    - System tests (8081)

    Yields:
        dict: Vespa connection info with keys:
            - http_port: Vespa HTTP port (8082)
            - config_port: Vespa config server port (19073)
            - base_url: Full HTTP URL
            - container_name: Docker container name

    Example:
        def test_schema_deployment(vespa_instance):
            manager = TenantSchemaManager(
                vespa_url="http://localhost",
                vespa_port=vespa_instance["http_port"]
            )
    """
    http_port = 8082
    config_port = 19073  # Config server port (19071 + port_offset)
    container_name = f"vespa-test-{http_port}"

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
            pytest.skip(f"Failed to start Docker container: {docker_result.stderr.decode()}")

        logger.info(f"✅ Vespa Docker container '{container_name}' started")

        # Wait for config server to be ready
        logger.info("Waiting for Vespa config server...")
        for i in range(120):  # 2 minutes timeout
            try:
                response = requests.get(f"http://localhost:{config_port}/", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Config server ready on port {config_port}")
                    break
            except:
                pass
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                logger.info(f"  Still waiting... ({i}s)")
        else:
            # Cleanup and skip
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            pytest.skip(f"Config server not ready after 120 seconds")

        # Wait for application endpoint to be ready
        logger.info(f"Waiting for application endpoint on port {http_port}...")
        for i in range(60):  # 1 minute timeout
            try:
                response = requests.get(f"http://localhost:{http_port}/ApplicationStatus", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Vespa ready on port {http_port}")
                    break
            except:
                pass
            time.sleep(2)
        else:
            # Cleanup and skip
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            pytest.skip(f"Application endpoint not ready after 60 seconds")

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
        # Cleanup: Stop and remove container
        logger.info(f"Stopping Docker container '{container_name}'")
        try:
            stop_result = subprocess.run(
                ["docker", "stop", container_name], capture_output=True, timeout=30
            )
            remove_result = subprocess.run(
                ["docker", "rm", container_name], capture_output=True, timeout=30
            )

            if stop_result.returncode == 0 and remove_result.returncode == 0:
                logger.info("✅ Stopped and removed Docker container")
            else:
                logger.warning(
                    f"⚠️  Issues stopping container: stop={stop_result.returncode}, rm={remove_result.returncode}"
                )
        except Exception as e:
            logger.warning(f"⚠️  Error stopping Docker container: {e}")
