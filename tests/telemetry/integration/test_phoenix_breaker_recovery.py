"""Deadline-cancelled Phoenix reads release endpoint recovery reservations."""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from cogniverse_core.common.utils.circuit_breaker import CircuitOpenError, CircuitState
from cogniverse_dashboard.tabs.optimization import _TELEMETRY_PROBE_TIMEOUT_S
from cogniverse_telemetry_phoenix.provider import PhoenixTraceStore

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture
def delayed_phoenix(phoenix_container):
    control = SimpleNamespace(
        mode="forward", entered=threading.Event(), release=threading.Event(), calls=0
    )

    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            control.calls += 1
            mode = control.mode
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if mode == "fail":
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":"Phoenix read unavailable"}')
                return
            response = httpx.request(
                self.command,
                phoenix_container["http_endpoint"] + self.path,
                content=body,
                headers={
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in {"host", "connection", "content-length"}
                },
                timeout=15,
            )
            if mode == "hold":
                control.entered.set()
                if not control.release.wait(10):
                    raise TimeoutError("test did not release Phoenix response")
            try:
                self.send_response(response.status_code)
                self.send_header(
                    "Content-Type",
                    response.headers.get("content-type", "application/json"),
                )
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)
            except (BrokenPipeError, ConnectionResetError):
                # Deadline cancellation closes this HTTP connection.
                return

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    control.endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        yield control
    finally:
        control.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def test_deadline_cancelled_probe_recovers_exact_tenant_spans(
    delayed_phoenix, telemetry_manager_with_phoenix
):
    manager = telemetry_manager_with_phoenix
    tenants = ("prodfixfoundation:recoverya", "prodfixfoundation:recoveryb")
    projects = [manager.config.get_project_name(tenant) for tenant in tenants]
    expected = []
    for tenant in tenants:
        with manager.span("recovery-span", tenant_id=tenant) as span:
            expected.append(f"{span.get_span_context().span_id:016x}")
    manager.force_flush()

    store = PhoenixTraceStore(delayed_phoenix.endpoint)
    peer = PhoenixTraceStore(delayed_phoenix.endpoint)
    assert store._breaker is peer._breaker
    for project, span_id in zip(projects, expected):
        for _ in range(30):
            frame = await store.get_spans(project=project)
            if len(frame) == 1:
                break
            await asyncio.sleep(0.1)
        assert frame["context.span_id"].tolist() == [span_id]
        assert frame["name"].tolist() == ["recovery-span"]

    breaker = store._breaker
    clock = [0.0]
    breaker.config.clock = lambda: clock[0]
    delayed_phoenix.mode = "fail"
    for _ in range(breaker.config.failure_threshold):
        with pytest.raises(httpx.HTTPStatusError) as failure:
            await store.get_spans(project=projects[0])
        assert failure.value.response.status_code == 503
    assert breaker.state is CircuitState.OPEN
    clock[0] += breaker.config.reset_timeout_s
    delayed_phoenix.mode = "hold"
    pending = asyncio.create_task(
        asyncio.wait_for(
            store.get_spans(project=projects[0]), _TELEMETRY_PROBE_TIMEOUT_S
        )
    )
    try:
        assert await asyncio.to_thread(delayed_phoenix.entered.wait, 5) is True
        calls = delayed_phoenix.calls
        with pytest.raises(CircuitOpenError) as rejected:
            await peer.get_spans(project=projects[1])
        assert rejected.value.name == f"phoenix:{delayed_phoenix.endpoint}"
        assert delayed_phoenix.calls == calls
        with pytest.raises(TimeoutError):
            await pending
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker._half_open_calls == 0
    finally:
        delayed_phoenix.release.set()
        await asyncio.gather(pending, return_exceptions=True)

    delayed_phoenix.mode = "fail"
    with pytest.raises(httpx.HTTPStatusError) as failure:
        await peer.get_spans(project=projects[1])
    assert failure.value.response.status_code == 503
    assert breaker.state is CircuitState.OPEN
    clock[0] += breaker.config.reset_timeout_s - 1
    with pytest.raises(CircuitOpenError):
        await store.get_spans(project=projects[0])
    clock[0] += 1
    delayed_phoenix.mode = "forward"
    for project, span_id in zip(projects, expected):
        frame = await peer.get_spans(project=project)
        assert frame["context.span_id"].tolist() == [span_id]
        assert frame["name"].tolist() == ["recovery-span"]
    assert breaker.state is CircuitState.CLOSED
    assert breaker._half_open_calls == 0
