"""Integration test configuration for backend tests.

``vespa_instance`` is a thin shim over the project-wide ``shared_vespa``
container (``tests/conftest.py``). Schema-lifecycle tests in this package
already deploy schemas under unique tenant_ids (``acme``, ``startup``,
etc.) so they coexist with other packages' tenant-scoped schemas in the
same Vespa without conflict — Vespa is multi-tenant by design.

The schema-mutation tests' ``wipe_non_protected_schemas`` fixture (in
``test_tenant_schema_lifecycle.py``) is the one place that needs care
under sharing — it must NOT touch tenants owned by other packages.
That scoping is done in the test file itself, not here.
"""

import logging
from pathlib import Path

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance(shared_vespa):
    """Compatibility shim: yields the dict shape backends/integration tests
    expect (``http_port``, ``config_port``, ``base_url``, ``container_name``)
    backed by the project-wide ``shared_vespa`` container.

    The original fixture started its own VespaDockerManager-managed
    container per module + deployed metadata schemas. ``shared_vespa``
    deploys the same metadata schemas at session start, so this shim
    just forwards the connection info — no setup work needed.
    """
    yield {
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
        "base_url": shared_vespa["base_url"],
        "container_name": shared_vespa["container_name"],
    }
    # Singleton clearing happens via the per-test autouse fixtures in
    # consumer conftests (e.g. tests/agents/integration/conftest.py).
    # No teardown here — shared_vespa owns the container lifecycle.


@pytest.fixture(autouse=True)
def _test_owned_telemetry():
    """Own the process-wide telemetry singleton a search touches.

    ``VespaSearchBackend.search`` opens encode spans through
    ``get_telemetry_manager()``, which first-builds the singleton from
    ``create_default_config_manager()``. Whether that reads a live store
    depends on what ``BACKEND_PORT`` an earlier module left behind (the
    root default is the dead 29071). Pre-build it disabled when unset and
    drop it afterwards, as ``tests/runtime/integration/conftest.py`` does.
    """
    import os

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        yield
        return
    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None


@pytest.fixture(scope="module")
def temp_config_manager(vespa_instance, tmp_path_factory):
    """
    Provide a temporary ConfigManager with real VespaConfigStore.

    Uses VespaConfigStore connected to the test Vespa instance.
    The config_metadata schema is automatically deployed as part of
    VespaSchemaManager.upload_metadata_schemas() during backend initialization.
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    http_port = vespa_instance["http_port"]
    logger.info(f"Creating VespaConfigStore with http_port={http_port}")

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=http_port,
    )
    logger.info(f"VespaConfigStore created, vespa_app URL: {store.vespa_app.url}")

    return ConfigManager(store=store)


@pytest.fixture(scope="module")
def schema_loader():
    """Provide FilesystemSchemaLoader for tests (module-scoped for reuse)."""
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture(scope="module")
def get_backend(vespa_instance, temp_config_manager, schema_loader):
    """
    Factory function to get backend for a tenant.

    Returns a function that creates backend instances for different tenants.
    Module-scoped to reuse backend instances across tests.
    """

    def _get_backend(tenant_id: str):
        registry = BackendRegistry.get_instance()
        config = {
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        }
        return registry.get_search_backend(
            name="vespa",
            config=config,
            config_manager=temp_config_manager,
            schema_loader=schema_loader,
        )

    return _get_backend


@pytest.fixture(scope="module")
def second_vespa():
    """A second, independent Vespa container.

    Endpoint identity can only be proved against two clusters that hold
    different documents: one cluster cannot show that a backend bound to
    endpoint A stopped answering from endpoint B.
    """
    import os
    import platform
    import subprocess
    import time

    from tests.conftest import (
        _shared_vespa_application_package,
        _shared_vespa_run_args,
        _vespa_wait_for_config_ready,
        _vespa_wait_for_data_port_ready,
        _vespa_wait_for_query_ready,
    )
    from tests.utils.docker_utils import start_docker_container_with_port_retry

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )
    container_name, http_port, config_port = start_docker_container_with_port_retry(
        "tests.backends.integration.second_vespa",
        name_prefix="backend-tests-second",
        image="vespaengine/vespa:8.668.5",
        container_ports=(8080, 19071),
        extra_run_args=_shared_vespa_run_args(
            owner_pid=os.getpid(), docker_platform=docker_platform
        ),
        max_attempts=5,
    )
    try:
        if not _vespa_wait_for_config_ready(config_port, timeout=180):
            pytest.fail(f"second_vespa config server (port {config_port}) not ready")
        time.sleep(10)

        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=config_port
        )._deploy_package(
            _shared_vespa_application_package(
                [
                    create_organization_metadata_schema(),
                    create_tenant_metadata_schema(),
                    create_config_metadata_schema(),
                    create_adapter_registry_schema(),
                ]
            )
        )
        if not _vespa_wait_for_data_port_ready(http_port, timeout=180):
            pytest.fail(f"second_vespa data port {http_port} not ready")
        if not _vespa_wait_for_query_ready(http_port, timeout=180):
            pytest.fail(f"second_vespa content cluster (port {http_port}) not ready")

        yield {
            "http_port": http_port,
            "config_port": config_port,
            "base_url": f"http://localhost:{http_port}",
            "container_name": container_name,
        }
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
