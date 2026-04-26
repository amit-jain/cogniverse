"""Unit-test fixture: replace the ConfigManager singleton with an
InMemoryConfigStore-backed instance.

The default singleton in ``cogniverse_foundation.config.utils`` builds a
``VespaConfigStore`` pointed at ``$BACKEND_URL:$BACKEND_PORT`` (the
top-level conftest defaults these to ``http://localhost:8080`` for unit
tests). Vespa isn't running during unit tests, so every
``store.get_config(...)`` call waits ~25s for the TCP connection to
time out before the manager falls back to defaults. Tests that exercise
``TenantAwareAgentMixin`` (which reads system + routing + telemetry +
backend configs at agent construction time) blow the 15-minute CI
timeout on this single workflow.

Replacing the store with the real (functional) ``InMemoryConfigStore``
keeps the same ``ConfigStore`` interface and the same logical path —
``manager.get_X_config`` → ``store.get_config`` → fallback default —
just over a dict instead of HTTP. No code coverage is lost; only the
network round-trip is.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _fast_unit_test_config_singleton():
    from cogniverse_foundation.config import utils as config_utils
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    saved = config_utils._config_manager_singleton
    config_utils._config_manager_singleton = ConfigManager(
        store=InMemoryConfigStore(),
    )
    try:
        yield
    finally:
        config_utils._config_manager_singleton = saved
