"""The pool's health reaper must not close a checked-out connection.

_health_check_loop snapshots every connection (checked-out ones included) and
_remove_connection closed them unconditionally: a searcher holding a connection
mid-query hit a closed HTTP client, and its finally re-added the closed
connection to the available list, handing it to the next searcher. A
checked-out connection marked unhealthy is now closed only when its holder
returns it, and never re-added.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_vespa.search_backend import ConnectionPool

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_pool() -> ConnectionPool:
    pool = ConnectionPool.__new__(ConnectionPool)
    pool.url = "http://localhost:29071"
    pool._connections = []
    pool._available = []
    pool._removing = set()
    pool._lock = threading.Lock()
    pool._returned = threading.Condition(pool._lock)
    pool._closed = False
    pool._stop_health_check = threading.Event()
    pool.config = SimpleNamespace(
        max_connections=4,
        min_connections=1,
        connection_timeout=5.0,
        idle_timeout=300.0,
        health_check_interval=3600.0,
    )
    return pool


def _fake_conn() -> MagicMock:
    conn = MagicMock()
    conn.close = MagicMock()
    return conn


def test_checked_out_connection_is_not_closed_while_held():
    pool = _bare_pool()
    conn = _fake_conn()
    pool._connections.append(conn)
    pool._available.append(conn)

    with pool.get_connection() as held:
        assert held is conn
        assert conn not in pool._available  # checked out

        # The reaper finds it unhealthy while a searcher holds it.
        pool._remove_connection(conn)

        conn.close.assert_not_called()  # must NOT close a live in-flight conn
        assert conn in pool._removing

    # On return it is closed and dropped, not re-added to the available list.
    conn.close.assert_called_once()
    assert conn not in pool._available
    assert conn not in pool._connections
    assert conn not in pool._removing


def test_available_connection_is_closed_immediately():
    pool = _bare_pool()
    conn = _fake_conn()
    pool._connections.append(conn)
    pool._available.append(conn)

    # Not checked out — safe to close now.
    pool._remove_connection(conn)

    conn.close.assert_called_once()
    assert conn not in pool._available
    assert conn not in pool._connections


def test_returned_healthy_connection_is_reused_not_closed():
    pool = _bare_pool()
    conn = _fake_conn()
    pool._connections.append(conn)
    pool._available.append(conn)

    with pool.get_connection() as held:
        assert held is conn

    # Healthy return: back in the available list, not closed.
    conn.close.assert_not_called()
    assert conn in pool._available


class _SweepConnection:
    """Stands in for VespaConnection at the pool's construction seam."""

    def __init__(
        self, url: str = "", connection_id: str = "sweep-conn", *, healthy: bool = True
    ):
        self.connection_id = connection_id
        self.healthy = healthy
        self.idle_time = 0.0
        self.closed = False
        self.probes = 0
        self.block_probe = False
        self.probe_started = threading.Event()
        self.release_probe = threading.Event()

    def health_check(self) -> bool:
        self.probes += 1
        if self.block_probe:
            self.probe_started.set()
            assert self.release_probe.wait(timeout=10)
        return self.healthy

    def close(self) -> None:
        self.closed = True


class _OneSweepGate:
    """Stop-event stand-in that ends the real health loop after one sweep.

    The loop calls ``wait(interval)`` after each sweep; answering that call
    with an immediate stop bounds the loop to exactly one pass.
    """

    def __init__(self):
        self._stop = threading.Event()
        self.sweep_done = threading.Event()

    def is_set(self) -> bool:
        return self._stop.is_set()

    def wait(self, timeout=None) -> bool:
        self.sweep_done.set()
        self._stop.set()
        return True

    def set(self) -> None:
        self._stop.set()


def _start_one_sweep(pool: ConnectionPool):
    gate = _OneSweepGate()
    pool._stop_health_check = gate
    thread = threading.Thread(target=pool._health_check_loop, daemon=True)
    thread.start()
    return gate, thread


def test_sweep_defers_close_of_checked_out_unhealthy_connection():
    pool = _bare_pool()
    conn = _SweepConnection(healthy=False)
    pool._connections.append(conn)
    pool._available.append(conn)

    with pool.get_connection() as held:
        assert held is conn
        gate, thread = _start_one_sweep(pool)
        assert gate.sweep_done.wait(timeout=10)
        thread.join(timeout=10)
        assert not thread.is_alive()

        # Probing a checked-out connection is by design (the shared VespaSync
        # session is probe-safe); closing it while held is not.
        assert conn.probes == 1
        assert conn.closed is False
        assert conn in pool._removing

    # Holder returned it: closed exactly once, never re-added.
    assert conn.closed is True
    assert conn not in pool._available
    assert conn not in pool._connections
    assert conn not in pool._removing


def test_sweep_reaps_dead_connection_after_return():
    pool = _bare_pool()
    conn = _SweepConnection(healthy=False)
    pool._connections.append(conn)
    pool._available.append(conn)

    with pool.get_connection() as held:
        assert held is conn

    # Returned now — the sweep may probe and reap it immediately.
    gate, thread = _start_one_sweep(pool)
    assert gate.sweep_done.wait(timeout=10)
    thread.join(timeout=10)

    assert conn.probes == 1
    assert conn.closed is True
    assert conn not in pool._available
    assert conn not in pool._connections

    # The pool recovers: a new connection is created on demand.
    with patch("cogniverse_vespa.search_backend.VespaConnection", _SweepConnection):
        with pool.get_connection() as fresh:
            assert fresh is not conn
            assert fresh.closed is False


def test_sweep_keeps_healthy_idle_connection_available():
    pool = _bare_pool()
    conn = _SweepConnection(healthy=True)
    pool._connections.append(conn)
    pool._available.append(conn)

    gate, thread = _start_one_sweep(pool)
    assert gate.sweep_done.wait(timeout=10)
    thread.join(timeout=10)

    assert conn.probes == 1
    assert conn.closed is False
    with pool.get_connection() as again:
        assert again is conn


def test_checkout_during_sweep_probe_defers_close_until_return():
    """A searcher that checks out the connection the sweep is mid-probe on
    must keep it alive: the unhealthy verdict lands after checkout, so the
    close is deferred to the holder's return, never fired mid-query."""
    pool = _bare_pool()
    conn = _SweepConnection(healthy=False)
    conn.block_probe = True
    pool._connections.append(conn)
    pool._available.append(conn)

    gate, thread = _start_one_sweep(pool)
    assert conn.probe_started.wait(timeout=10)

    # Probe in flight: the connection is still available and is handed out.
    with pool.get_connection() as held:
        assert held is conn
        conn.release_probe.set()
        assert gate.sweep_done.wait(timeout=10)
        thread.join(timeout=10)
        assert not thread.is_alive()

        # Unhealthy verdict arrived while held — close deferred.
        assert conn.closed is False
        assert conn in pool._removing

    assert conn.closed is True
    assert conn not in pool._available
    assert conn not in pool._connections
