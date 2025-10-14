"""
System test configuration and fixtures.

Provides module-scoped Vespa instance for system tests.
Similar pattern to memory tests:
1. Starts once per test module
2. Deploys video search schemas once
3. Ingests test videos once
4. Tests clean up documents (not schemas)
5. Stops after module tests complete

Uses unique ports per test module to avoid conflicts with:
- Main Vespa (8080)
- Backend integration tests (different module, different ports)
- Other test modules (deterministic hash-based port assignment)
"""

import pytest

from tests.utils.docker_utils import generate_unique_ports

# Generate unique ports based on this module name
SYSTEM_VESPA_PORT, SYSTEM_VESPA_CONFIG_PORT = generate_unique_ports(__name__)
SYSTEM_VESPA_CONTAINER = f"vespa-system-tests-{SYSTEM_VESPA_PORT}"


@pytest.fixture(scope="module")
def shared_system_vespa():
    """
    Module-scoped Vespa instance for system tests.

    Each test module gets a fresh Vespa instance with proper isolation.
    Starts, deploys schemas, and ingests test videos per module.
    Torn down after each module completes for proper cleanup.
    """
    from .vespa_test_manager import VespaTestManager

    print("\n" + "=" * 70)
    print("🚀 Starting shared Vespa container for system tests...")
    print(f"   Port: {SYSTEM_VESPA_PORT} (data), {SYSTEM_VESPA_CONFIG_PORT} (config)")
    print("=" * 70)

    # CRITICAL: Clear ALL singletons to ensure fresh state with test ports
    from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
    from cogniverse_core.config.manager import ConfigManager
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

    print("🧹 Clearing all singleton state before setup...")

    # Clear TenantSchemaManager singleton
    TenantSchemaManager._clear_instance()

    # Clear Mem0MemoryManager per-tenant instance cache
    Mem0MemoryManager._instances.clear()

    # Clear backend registry instances (critical - may have old port configs)
    registry = get_backend_registry()
    if hasattr(registry, '_backend_instances'):
        old_count = len(registry._backend_instances)
        registry._backend_instances.clear()
        print(f"   Cleared {old_count} cached backend instances")

    # Clear ConfigManager singleton (may have cached old config)
    if hasattr(ConfigManager, '_instance'):
        ConfigManager._instance = None
        print("   Cleared ConfigManager singleton")

    print("✅ Singleton state cleared")

    # Create manager with unique test ports for this module
    manager = VespaTestManager(
        http_port=SYSTEM_VESPA_PORT,
        config_port=SYSTEM_VESPA_CONFIG_PORT
    )

    try:
        # Setup: directory, deploy, ingest
        print("Setting up isolated Vespa instance with test data...")
        if not manager.full_setup():
            pytest.fail("Failed to setup Vespa test environment")

        print("\n" + "=" * 70)
        print("✅ Shared Vespa ready for system tests")
        print(f"   Search endpoint: http://localhost:{SYSTEM_VESPA_PORT}/search/")
        print(f"   Document API: http://localhost:{SYSTEM_VESPA_PORT}/document/v1/")
        print("=" * 70 + "\n")

        vespa_config = {
            "http_port": SYSTEM_VESPA_PORT,
            "config_port": SYSTEM_VESPA_CONFIG_PORT,
            "container_name": SYSTEM_VESPA_CONTAINER,
            "base_url": f"http://localhost:{SYSTEM_VESPA_PORT}",
            "vespa_url": "http://localhost",
            "default_schema": manager.default_test_schema,
            "manager": manager,  # Provide manager for tests that need it
        }

        yield vespa_config

    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("🧹 Cleaning up Vespa container...")
        print("=" * 70)
        manager.cleanup()

        # Clear singleton state to avoid interference with other test modules
        try:
            from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
            from cogniverse_core.config.manager import ConfigManager
            from cogniverse_core.registries.backend_registry import get_backend_registry
            from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

            # Clear TenantSchemaManager singleton
            TenantSchemaManager._clear_instance()

            # Clear Mem0MemoryManager instances
            Mem0MemoryManager._instances.clear()

            # Clear backend registry instances
            registry = get_backend_registry()
            if hasattr(registry, '_backend_instances'):
                registry._backend_instances.clear()

            # Clear ConfigManager singleton
            if hasattr(ConfigManager, '_instance'):
                ConfigManager._instance = None

            print("✅ Cleared singleton state for next module")
        except Exception as e:
            print(f"⚠️  Error clearing singleton state: {e}")

        print("✅ Cleanup complete")
