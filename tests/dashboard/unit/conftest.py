"""Unit-test isolation for ``tests/dashboard/unit``.

Dashboard tabs read infrastructure config two ways, both of which fall back
to a Vespa-backed store built from the ``BACKEND_*`` env (the project-wide
dead-port default), so an un-isolated unit render leaks a Vespa read:

* ``get_telemetry_manager()`` (foundation/telemetry/manager) seeds its global
  singleton from ``create_default_config_manager()`` -> ``VespaConfigStore`` on
  first call.
* Several tabs call ``create_default_config_manager()`` directly to read
  ``SystemConfig`` (backend URL, agent registry, backend profiles).

Seed the telemetry singleton with an in-memory ``TelemetryManager`` and route
``create_default_config_manager()`` to a real ``ConfigManager`` backed by
``InMemoryConfigStore``, seeded with a ``SystemConfig`` pointed at the same
dead backend so each tab's reachability probe resolves deterministically
without touching a live Vespa.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _default_telemetry_singleton():
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
    telemetry_manager_module._telemetry_manager = TelemetryManager(TelemetryConfig())
    yield
    TelemetryManager.reset()


# Tabs that bind ``create_default_config_manager`` at import time (module-level
# ``from ... import``); their bound name must be patched alongside the source.
_CONFIG_MANAGER_CONSUMERS = (
    "cogniverse_dashboard.tabs.memory_management",
    "cogniverse_dashboard.tabs.config_management",
    "cogniverse_dashboard.tabs.backend_profile",
)


@pytest.fixture(autouse=True)
def _in_memory_config_manager(monkeypatch):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from tests.utils.memory_store import InMemoryConfigStore

    backend_url = os.environ.get("BACKEND_URL", "http://localhost")
    backend_port = int(os.environ.get("BACKEND_PORT", "29071"))

    def _factory() -> ConfigManager:
        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_system_config(
            SystemConfig(backend_url=backend_url, backend_port=backend_port)
        )
        return manager

    import cogniverse_foundation.config.utils as config_utils

    monkeypatch.setattr(config_utils, "create_default_config_manager", _factory)
    for name in _CONFIG_MANAGER_CONSUMERS:
        module = importlib.import_module(name)
        monkeypatch.setattr(
            module, "create_default_config_manager", _factory, raising=False
        )
    yield
