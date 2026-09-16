"""A dead OpenShell gateway must fail fast, not stall on wait_ready.

The gateway breaker trips after a few failed dials; subsequent task sessions
then fail immediately (breaker open) instead of dialing and waiting out the
readiness budget, so one dead gateway can't stall the worker pool.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.utils.circuit_breaker import (
    BreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
)
from cogniverse_runtime.sandbox_manager import SandboxManager
from cogniverse_runtime.sandbox_pool import SandboxPoolConfig, SandboxSessionPool


@pytest.fixture(autouse=True)
def _reset_breakers():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def _breaker(name, threshold=2):
    # Long reset so the OPEN state persists across the test.
    return CircuitBreaker.get(
        BreakerConfig(name=name, failure_threshold=threshold, reset_timeout_s=10_000)
    )


def test_pool_create_fast_fails_after_breaker_opens():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ConnectionError("gateway down")

    client = MagicMock()
    client.create_session.side_effect = boom

    pool = SandboxSessionPool(
        client,
        config=SandboxPoolConfig(),
        gateway_breaker=_breaker("pool_gw"),
    )

    for _ in range(2):
        with pytest.raises(ConnectionError):
            pool._create_with_spans()
    assert calls["n"] == 2

    # Breaker is now open: the dial is not attempted.
    with pytest.raises(CircuitOpenError):
        pool._create_with_spans()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_task_session_fails_fast_once_gateway_breaker_open():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ConnectionError("gateway down")

    # Register the shared gateway breaker at threshold 2 before the manager
    # asks for it, so the manager uses this one.
    breaker = _breaker("openshell_gateway", threshold=2)
    mgr = SandboxManager(policy="disabled")
    assert mgr._gateway_breaker is breaker
    mgr._available = True
    mgr._client = MagicMock()
    mgr._client.create_session.side_effect = boom

    # First two dials fail and the breaker records them.
    for _ in range(2):
        with pytest.raises(ConnectionError, match="gateway down"):
            async with mgr.task_session("coding", "prodfixagents:breaker"):
                pass
    assert calls["n"] == 2

    # Third: breaker open -> fast-fail, no further dial.
    with pytest.raises(CircuitOpenError):
        async with mgr.task_session("coding", "prodfixagents:breaker"):
            pass
    assert calls["n"] == 2
