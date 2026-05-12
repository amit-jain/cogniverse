"""Integration test configuration for common tests.

Re-exports the project-wide ``shared_vespa`` container; common tests
(config persistence, dynamic config) only need the metadata schemas
that ``shared_vespa`` already deploys at session start.
"""

import logging

import pytest

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vespa_instance(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape common/integration tests
    expect (``http_port``, ``config_port``, ``base_url``,
    ``container_name``) backed by ``shared_vespa``.
    """
    yield {
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
        "base_url": shared_vespa["base_url"],
        "container_name": shared_vespa["container_name"],
    }
    # Clear singletons so the next module starts fresh.
    try:
        from cogniverse_core.registries.backend_registry import get_backend_registry
        from cogniverse_foundation.config.manager import ConfigManager

        registry = get_backend_registry()
        if hasattr(registry, "_backend_instances"):
            registry._backend_instances.clear()
        if hasattr(ConfigManager, "_instance"):
            ConfigManager._instance = None
    except Exception as e:
        logger.warning(f"singleton cleanup: {e}")
