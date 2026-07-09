"""A dead OpenShell gateway must fail fast, not stall on wait_ready.

The gateway breaker trips after a few failed dials; subsequent sandbox exec
calls then return immediately (breaker open) instead of dialing and waiting the
120s wait_ready, so one dead gateway can't stall the worker pool.
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
        config=SandboxPoolConfig(enabled=False),
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


def test_exec_returns_none_fast_once_gateway_breaker_open():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ConnectionError("gateway down")

    mgr = object.__new__(SandboxManager)
    mgr._available = True
    mgr._client = MagicMock()
    mgr._client.create_session.side_effect = boom
    mgr._cert_rotator = None
    mgr._gateway_breaker = _breaker("openshell_gateway", threshold=2)
    # Force the non-pooled path.
    mgr._get_or_create_pool = lambda: None

    # First two dials fail (degrade dict), the breaker records them.
    for _ in range(2):
        result = mgr.exec_in_sandbox("coding", ["echo", "hi"])
        assert result == {"stdout": "", "stderr": "gateway down", "exit_code": -1}
    assert calls["n"] == 2

    # Third: breaker open -> fast-fail with None, no further dial.
    assert mgr.exec_in_sandbox("coding", ["echo", "hi"]) is None
    assert calls["n"] == 2
