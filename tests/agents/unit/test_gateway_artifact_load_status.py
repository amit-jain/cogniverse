"""GatewayAgent._load_artifact must distinguish the four outcomes.

The pre-fix code swallowed everything into a single ``logger.debug("No
gateway artifact to load (using defaults)")`` line — operators could
not tell whether a tenant was using defaults because (a) they had
never optimized, or (b) the telemetry provider was down. Both look
the same on the gateway. The fix records ``artifact_load_status`` on
the instance and logs the error path at WARNING.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps


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


# ---------------------------------------------------------------------------
# Threshold value validation (a non-numeric or out-of-range artifact value
# must not reach deps — deps has extra="allow"/no validate_assignment, so a
# bad value would sit silently and blow up at request time).
# ---------------------------------------------------------------------------


def test_non_numeric_fast_path_keeps_defaults_and_errors() -> None:
    blob = json.dumps({"fast_path_confidence_threshold": "high"})
    with _telemetry_with_blob(blob=blob) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "error"
    assert isinstance(agent.artifact_load_message, str) and agent.artifact_load_message
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.3


def test_null_gliner_threshold_keeps_defaults_and_errors() -> None:
    blob = json.dumps({"gliner_threshold": None})
    with _telemetry_with_blob(blob=blob) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "error"
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.3


def test_out_of_range_fast_path_keeps_defaults_and_errors() -> None:
    blob = json.dumps({"fast_path_confidence_threshold": 1.7})
    with _telemetry_with_blob(blob=blob) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "error"
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.3


def test_valid_floats_still_applied_after_validation() -> None:
    blob = json.dumps(
        {"fast_path_confidence_threshold": 0.72, "gliner_threshold": 0.18}
    )
    with _telemetry_with_blob(blob=blob) as (agent, _am):
        agent._load_artifact()
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.72
    assert agent.deps.gliner_threshold == 0.18


# ---------------------------------------------------------------------------
# Atomic threshold snapshot: the TTL reload thread mutates both thresholds
# while dispatches read them. Reads must never observe a torn (new, old) pair.
# ---------------------------------------------------------------------------

_PAIR_A = (0.11, 0.31)
_PAIR_B = (0.87, 0.67)


def _build_agent_with_toggling_artifact():
    """A GatewayAgent whose artifact ``load_blob`` alternates between _PAIR_A
    and _PAIR_B on each ``_load_artifact`` call, so successive reloads flip
    both thresholds together."""
    agent = _make_real_gateway()
    provider = MagicMock()
    agent.telemetry_manager = MagicMock()
    agent.telemetry_manager.get_provider = MagicMock(return_value=provider)
    agent._artifact_tenant_id = "acme"

    pairs = itertools.cycle([_PAIR_A, _PAIR_B])

    async def _load_blob(*_a, **_k):
        fast, gliner = next(pairs)
        return json.dumps(
            {"fast_path_confidence_threshold": fast, "gliner_threshold": gliner}
        )

    am = MagicMock()
    am.load_blob = AsyncMock(side_effect=_load_blob)
    return agent, am


def _make_real_gateway() -> GatewayAgent:
    from cogniverse_agents.gateway_agent import GatewayAgent as _GW

    return _GW(deps=GatewayDeps())


def test_reload_publishes_thresholds_as_one_atomic_pair() -> None:
    agent, am = _build_agent_with_toggling_artifact()
    with patch(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        return_value=am,
    ):
        agent._load_artifact()
        first = agent._thresholds
        agent._load_artifact()
        second = agent._thresholds

    assert (first.fast_path_confidence, first.gliner) in (_PAIR_A, _PAIR_B)
    assert (second.fast_path_confidence, second.gliner) in (_PAIR_A, _PAIR_B)
    # Consecutive reloads flipped BOTH values together, never a mix.
    assert (first.fast_path_confidence, first.gliner) != (
        second.fast_path_confidence,
        second.gliner,
    )


def test_concurrent_reads_never_see_a_torn_threshold_pair() -> None:
    agent, am = _build_agent_with_toggling_artifact()
    seen: set[tuple[float, float]] = set()
    bad: list[tuple[float, float]] = []
    stop = threading.Event()
    errors: list[BaseException] = []
    valid = frozenset((_PAIR_A, _PAIR_B))

    def _reader() -> None:
        try:
            while not stop.is_set():
                snap = agent._thresholds
                pair = (snap.fast_path_confidence, snap.gliner)
                if pair not in valid:
                    bad.append(pair)
                else:
                    seen.add(pair)
                # Yield so the writer's asyncio round trip isn't GIL-starved.
                time.sleep(0)
        except BaseException as exc:  # thread swallows; surface to the assertion
            errors.append(exc)

    def _writer() -> None:
        try:
            with patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                return_value=am,
            ):
                for _ in range(60):
                    agent._load_artifact()
        finally:
            stop.set()

    with patch(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        return_value=am,
    ):
        agent._load_artifact()  # seed the first snapshot

    readers = [threading.Thread(target=_reader) for _ in range(2)]
    writer = threading.Thread(target=_writer)
    for r in readers:
        r.start()
    writer.start()
    writer.join()
    for r in readers:
        r.join()

    assert not errors, errors
    # Every observed pair is one of the two valid pairs — never (new, old).
    assert not bad, f"torn threshold pairs observed: {bad[:5]}"
    # Both pairs actually occurred, so the reload really did interleave reads.
    assert seen == valid, f"expected both pairs to be observed, saw {seen}"
