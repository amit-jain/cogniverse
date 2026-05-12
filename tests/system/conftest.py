"""System test configuration and fixtures.

The fixture is now a thin shim over the project-wide ``shared_vespa``
container. Each test module gets its own tenant-scoped video schema
(``video_colpali_smol500_mv_frame_<module>``) deployed via SchemaRegistry,
preserving per-module isolation without spinning up a separate container
per module.
"""

import os
from pathlib import Path

import pytest

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401


@pytest.fixture(scope="module")
def shared_system_vespa(shared_vespa, request):  # noqa: F811
    """Compatibility shim: yields the dict shape system tests expect,
    backed by the project-wide ``shared_vespa``.

    Sets ``COGNIVERSE_CONFIG`` to the system test config so tests that
    bootstrap their own ConfigManager pick up the right backend block.
    Deploys ``video_colpali_smol500_mv_frame`` for tenant ``test_unit``
    via SchemaRegistry (merge-safe). The ``manager`` field exposes a
    minimal adapter providing ``http_port``, ``config_port``, and
    ``default_test_schema`` so the few tests that read
    ``vespa_config["manager"].http_port`` keep working.
    """

    class _SharedVespaSystemAdapter:
        def __init__(self, http_port, config_port, container_name):
            self.http_port = http_port
            self.config_port = config_port
            self.container_name = container_name
            self.default_test_schema = "video_colpali_smol500_mv_frame"

        def cleanup(self) -> None:
            # No-op — shared_vespa owns the container lifecycle.
            pass

    # CRITICAL: Clear singletons so this module gets fresh state.
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,
        get_backend_registry,
    )
    from cogniverse_core.registries.registry import get_registry
    from cogniverse_foundation.config.manager import ConfigManager

    test_config_path = (
        Path(__file__).parent / "resources" / "configs" / "system_test_config.json"
    )
    original_config = os.environ.get("COGNIVERSE_CONFIG")
    os.environ["COGNIVERSE_CONFIG"] = str(test_config_path.absolute())

    strategy_registry = get_registry()
    if hasattr(strategy_registry, "_strategy_cache"):
        strategy_registry._strategy_cache.clear()
    strategy_registry.reload()

    Mem0MemoryManager._instances.clear()
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    # Pre-deploy the data schema for this module's tenant via SchemaRegistry.
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_vespa,
        tenant_id="test_unit",
        base_schema_name="video_colpali_smol500_mv_frame",
    )

    # Reset registries again so consumer tests start clean.
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()

    try:
        yield {
            "http_port": shared_vespa["http_port"],
            "config_port": shared_vespa["config_port"],
            "container_name": shared_vespa["container_name"],
            "base_url": shared_vespa["base_url"],
            "backend_url": "http://localhost",
            "default_schema": "video_colpali_smol500_mv_frame",
            "manager": _SharedVespaSystemAdapter(
                shared_vespa["http_port"],
                shared_vespa["config_port"],
                shared_vespa["container_name"],
            ),
        }
    finally:
        # Restore env vars; clear singletons.
        if original_config is not None:
            os.environ["COGNIVERSE_CONFIG"] = original_config
        elif "COGNIVERSE_CONFIG" in os.environ:
            del os.environ["COGNIVERSE_CONFIG"]
        try:
            Mem0MemoryManager._instances.clear()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()
            BackendRegistry._shared_schema_registry = None
            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None
        except Exception:
            pass
