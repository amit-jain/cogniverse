"""GatewayAgent._load_artifact must distinguish the four outcomes.

The pre-fix code swallowed everything into a single ``logger.debug("No
gateway artifact to load (using defaults)")`` line — operators could
not tell whether a tenant was using defaults because (a) they had
never optimized, or (b) the telemetry provider was down. Both look
the same on the gateway. The fix records ``artifact_load_status`` on
the instance and logs the error path at WARNING.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cogniverse_agents.gateway_agent import GatewayAgent


@contextmanager
def _telemetry_with_blob(blob: str | None):
    """Yield a (GatewayAgent, ArtifactManager mock) pair whose ``load_blob``
    coroutine resolves to ``blob``."""
    agent = object.__new__(GatewayAgent)
    agent.deps = SimpleNamespace(
        fast_path_confidence_threshold=0.5, gliner_threshold=0.3
    )
    agent._artifact_tenant_id = "acme"
    provider = MagicMock()
    agent.telemetry_manager = MagicMock()
    agent.telemetry_manager.get_provider = MagicMock(return_value=provider)

    am = MagicMock()
    am.load_blob = AsyncMock(return_value=blob)
    # The function does a local ``from cogniverse_agents.optimizer.
    # artifact_manager import ArtifactManager``; patch at the source.
    with patch(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        return_value=am,
    ):
        yield agent, am


def test_no_telemetry_status() -> None:
    agent = object.__new__(GatewayAgent)
    agent.deps = SimpleNamespace(
        fast_path_confidence_threshold=0.5, gliner_threshold=0.3
    )
    # No telemetry_manager attribute at all.
    agent._load_artifact()
    assert agent.artifact_load_status == "no_telemetry"


def test_no_artifact_status_when_blob_is_none() -> None:
    with _telemetry_with_blob(blob=None) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "no_artifact"
    # Defaults preserved.
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.3


def test_loaded_status_applies_thresholds() -> None:
    blob = json.dumps({"fast_path_confidence_threshold": 0.8, "gliner_threshold": 0.6})
    with _telemetry_with_blob(blob=blob) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.8
    assert agent.deps.gliner_threshold == 0.6


def test_error_status_when_load_raises(caplog) -> None:
    import logging

    agent = object.__new__(GatewayAgent)
    agent.deps = SimpleNamespace(
        fast_path_confidence_threshold=0.5, gliner_threshold=0.3
    )
    agent._artifact_tenant_id = "acme"
    agent.telemetry_manager = MagicMock()
    # Provider raises on get_provider — simulates a telemetry outage.
    agent.telemetry_manager.get_provider = MagicMock(
        side_effect=ConnectionError("phoenix unreachable")
    )
    with caplog.at_level(logging.WARNING, logger="cogniverse_agents.gateway_agent"):
        agent._load_artifact()
    assert agent.artifact_load_status == "error"
    # Defaults preserved but a WARNING was emitted — operators can tell.
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert any(
        rec.levelno >= logging.WARNING and "artifact load failed" in rec.message
        for rec in caplog.records
    )


def test_missing_tenant_id_records_error_status() -> None:
    agent = object.__new__(GatewayAgent)
    agent.deps = SimpleNamespace(
        fast_path_confidence_threshold=0.5, gliner_threshold=0.3
    )
    agent.telemetry_manager = MagicMock()
    # _artifact_tenant_id intentionally unset.
    agent._load_artifact()
    assert agent.artifact_load_status == "error"
