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
from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.search_backend import ConnectionPool

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _bare_pool() -> ConnectionPool:
    pool = ConnectionPool.__new__(ConnectionPool)
    pool._connections = []
    pool._available = []
    pool._removing = set()
    pool._lock = threading.Lock()
    pool._returned = threading.Condition(pool._lock)
    pool.config = SimpleNamespace(
        max_connections=4,
        min_connections=1,
        connection_timeout=5.0,
        idle_timeout=300.0,
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
