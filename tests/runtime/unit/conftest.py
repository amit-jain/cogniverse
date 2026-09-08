"""Unit-test isolation for ``tests/runtime/unit``.

These are pure-unit tests (dependencies mocked / injected via
``InMemoryConfigStore``). They must not touch Vespa. The one leak is the global
telemetry singleton: ``get_telemetry_manager()`` (foundation/telemetry/manager)
falls back to ``create_default_config_manager()`` → ``VespaConfigStore`` on its
first call, and the project-wide dead-port default then makes that read fail.

Seed the singleton with a default in-memory ``TelemetryManager`` before every
test (and reset after) so no code path triggers the Vespa fallback — the same
seed-the-singleton pattern ``telemetry_manager_with_phoenix`` uses, minus
Phoenix.
"""

from __future__ import annotations

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


@pytest.fixture
def harness_key_config_store(request, monkeypatch):
    """Bind tenant retirement to the test-owned credential store."""
    import subprocess

    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    available = int(
        subprocess.check_output(["free", "-g"], text=True).splitlines()[1].split()[-1]
    )
    assert available >= 30
    vespa = request.getfixturevalue("shared_vespa")
    store = VespaConfigStore(backend_port=vespa["http_port"])
    monkeypatch.setattr(tenant_manager, "_config_manager", ConfigManager(store=store))
    yield
    store.close()
