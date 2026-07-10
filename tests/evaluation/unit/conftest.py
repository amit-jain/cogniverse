"""Unit-test isolation for ``tests/evaluation/unit``.

These are pure-unit tests (storage/provider mocked or injected). They must not
touch Vespa. The leak is the global telemetry singleton:
``get_telemetry_manager()`` (foundation/telemetry/manager) falls back to
``create_default_config_manager()`` → ``VespaConfigStore`` on its first call,
and the project-wide dead-port default then makes that config read fail.

``TelemetryStorage._initialize_connection`` calls ``get_telemetry_manager()``
before the provider is resolved, so both ``DatasetManager()``/``TraceManager()``
construction and direct ``TelemetryStorage(config)`` construction trip the
fallback. Seed the singleton with a default in-memory ``TelemetryManager``
before every test (and reset after) so no code path triggers the Vespa
fallback.
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
