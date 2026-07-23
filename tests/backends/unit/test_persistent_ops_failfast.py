"""Persistent Vespa sessions fail fast instead of pyvespa's 120s default.

The document-API migration moved wiki/graph/back-ref writes (which carried
explicit 10-15s timeouts) onto PersistentVespaOps, whose underlying pyvespa
client hardcodes a 120s timeout — a hung Vespa would block every migrated
write 12x longer. The factory restores a fail-fast ceiling.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from cogniverse_vespa._vespa_factory import make_persistent_vespa_ops

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture()
def hanging_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def hold():
        while not stop.is_set():
            try:
                srv.settimeout(0.5)
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            stop.wait(30)
            conn.close()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    yield port
    stop.set()
    srv.close()


def test_ops_time_out_fast_against_a_hung_vespa(hanging_server):
    ops = make_persistent_vespa_ops(
        url="http://127.0.0.1", port=hanging_server, timeout_s=1.5
    )
    try:
        start = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            ops.get_data(schema="any_schema", data_id="x")
        elapsed = time.monotonic() - start
    finally:
        ops.close()

    assert elapsed < 10.0, (
        f"hung-Vespa op took {elapsed:.1f}s — the 120s pyvespa default is back "
        f"(raised {excinfo.type.__name__})"
    )


def test_search_connection_failfast():
    """The user-facing search session must carry the same fail-fast clamp as
    the document/config sessions — it kept pyvespa's 120s x 11-attempt
    defaults, so a hung Vespa blocked each query for minutes."""
    from cogniverse_vespa.search_backend import VespaConnection

    conn = VespaConnection("http://localhost:1", "conn-failfast")

    assert conn._sync.http_client.timeout == 15.0
    assert conn._sync.num_retries_429 == 2


def test_health_check_probes_through_the_clamped_session():
    """The reaper's health probe must go through the fail-fast-clamped session,
    not the raw un-clamped app — otherwise a hung Vespa blocks the health sweep
    for pyvespa's 120s default, delaying detection of the bad connection."""
    from unittest.mock import MagicMock

    from cogniverse_vespa.search_backend import VespaConnection

    conn = VespaConnection.__new__(VespaConnection)
    conn.connection_id = "c1"
    conn.is_healthy = True
    conn._sync = MagicMock()
    conn._sync.query.return_value = {"root": {"children": []}}
    conn.vespa = MagicMock()  # the un-clamped app — must NOT be probed

    assert conn.health_check() is True
    conn._sync.query.assert_called_once()
    conn.vespa.query.assert_not_called()
