"""Unit-test fixtures for cogniverse_agents.

Unit tests construct ``RLMInference`` objects directly to verify wiring,
options, and serialization. The constructor probes for Deno (a real runtime
dependency for DSPy RLM REPL execution) and fails fast when missing — correct
behaviour at boot, but unhelpful for unit-only environments without Deno.

This autouse fixture sets ``COGNIVERSE_RLM_SKIP_DENO_CHECK=1`` for the unit
test session so construction succeeds. Integration tests at
``tests/agents/integration/`` deliberately do NOT set this — they want the
probe active so they exercise the real boot path.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _skip_rlm_deno_check_for_unit_tests():
    """Bypass RLMInference's Deno probe for unit-only test runs."""
    previous = os.environ.get("COGNIVERSE_RLM_SKIP_DENO_CHECK")
    os.environ["COGNIVERSE_RLM_SKIP_DENO_CHECK"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COGNIVERSE_RLM_SKIP_DENO_CHECK", None)
        else:
            os.environ["COGNIVERSE_RLM_SKIP_DENO_CHECK"] = previous


@pytest.fixture(autouse=True)
def _default_telemetry_singleton():
    """Seed the global telemetry singleton with an in-memory default so no
    agent construction path triggers ``get_telemetry_manager()``'s fallback to
    ``create_default_config_manager()`` → ``VespaConfigStore``, which would
    otherwise read the dead-port backend. Same seed-the-singleton pattern used
    by ``tests/runtime/unit/conftest.py``."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
    telemetry_manager_module._telemetry_manager = TelemetryManager(TelemetryConfig())
    yield
    TelemetryManager.reset()
