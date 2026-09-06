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
