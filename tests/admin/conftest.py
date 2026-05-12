"""Integration test configuration for admin profile tests.

Re-exports the project-wide ``shared_vespa`` container; admin tests
only need the metadata schemas (already deployed at session start) and
their own ConfigManager wiring (built per-test).
"""

import logging

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def vespa_instance(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape admin tests expect
    (``http_port``, ``config_port``, ``base_url``, ``container_name``)
    backed by ``shared_vespa``. No pre-deploy needed — admin tests
    drive the SchemaRegistry themselves under their own tenants.
    """
    yield {
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
        "base_url": shared_vespa["base_url"],
        "container_name": shared_vespa["container_name"],
    }
    # Clear singleton state so the next test module gets a fresh registry.
    try:
        from cogniverse_core.registries.backend_registry import BackendRegistry

        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _admin_singleton_reset_between_tests():
    """Per-test reset of every singleton admin tests touch.

    Earlier each test file's ``test_client`` fixture maintained its own
    list of singletons to clear; if any list missed
    ``BackendRegistry._shared_schema_registry`` (the registry that
    pins the active ``SchemaRegistry`` and its ``_schema_loader``) the
    next test inherited a registry pointing at the prior test's tmp
    schema dir and ``deploy_schema`` blew up with "Schema X not found
    at /tmp/.../schemas{N}/X_schema.json".

    Centralising the reset here means individual ``test_client``
    fixtures stop being responsible for catching every singleton.
    """
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_core.registries.schema_registry import SchemaRegistry

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    SchemaRegistry._instance = None

    # Also reset the admin / tenant_manager module-level state so a
    # test that forgets to call ``admin.reset_dependencies()`` in its
    # teardown doesn't leak into the next test.
    try:
        from cogniverse_runtime.admin import tenant_manager
        from cogniverse_runtime.routers import admin as admin_router

        admin_router.reset_dependencies()
        tenant_manager.backend = None
        tenant_manager._config_manager = None
        tenant_manager._schema_loader = None
    except Exception:
        pass

    yield

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    SchemaRegistry._instance = None
    try:
        from cogniverse_runtime.admin import tenant_manager
        from cogniverse_runtime.routers import admin as admin_router

        admin_router.reset_dependencies()
        tenant_manager.backend = None
        tenant_manager._config_manager = None
        tenant_manager._schema_loader = None
    except Exception:
        pass
