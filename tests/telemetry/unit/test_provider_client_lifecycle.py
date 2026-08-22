"""The per-loop Phoenix client memo must not leak loops or sockets under the
asyncio.run-per-call pattern (TelemetryStorage health check, the dashboard's
run-per-interaction sync facades).

The remote boundary is the only thing stubbed — a local keep-alive HTTP server
stands in for Phoenix so the real PhoenixTraceStore + memo run unchanged.
"""

import asyncio
import gc
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.unit


class _StubPhoenix(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like real Phoenix

    def _respond(self):
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._respond()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._respond()

    def log_message(self, *args):
        pass


def _socket_fds() -> int:
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith("socket:"):
                n += 1
        except OSError:
            pass
    return n


@pytest.fixture
def stub_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubPhoenix)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_run_per_call_does_not_leak_loops_or_sockets(stub_endpoint):
    from cogniverse_telemetry_phoenix.provider import (
        _CLIENTS_BY_LOOP,
        PhoenixTraceStore,
    )

    store = PhoenixTraceStore(http_endpoint=stub_endpoint)
    tenant_id = "acme:acme"
    project = f"cogniverse-{tenant_id}"

    gc.collect()
    base_sockets = _socket_fds()

    for _ in range(40):
        df = asyncio.run(store.get_spans(project=project, limit=1))
        assert df.empty
    gc.collect()
    gc.collect()

    # The memo must not accumulate one immortal entry per fresh loop, and the
    # per-request sockets must not survive their (closed) loops.
    assert len(_CLIENTS_BY_LOOP) <= 2, (
        f"memo leaked {len(_CLIENTS_BY_LOOP)} entries across 40 fresh loops"
    )
    leaked = _socket_fds() - base_sockets
    assert leaked <= 5, f"leaked {leaked} socket FDs across 40 run-per-call cycles"


@pytest.mark.asyncio
async def test_same_loop_reuses_one_client(stub_endpoint):
    """A genuinely long-lived loop still reuses a single client instance —
    the memo's original purpose is preserved."""
    from cogniverse_telemetry_phoenix.provider import (
        PhoenixTraceStore,
        _client_for_current_loop,
    )

    store = PhoenixTraceStore(http_endpoint=stub_endpoint)
    tenant_id = "acme:acme"
    project = f"cogniverse-{tenant_id}"
    await store.get_spans(project=project, limit=1)
    c1 = _client_for_current_loop(stub_endpoint)
    await store.get_spans(project=project, limit=1)
    c2 = _client_for_current_loop(stub_endpoint)
    assert c1 is c2
