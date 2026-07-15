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
