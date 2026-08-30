"""RuntimeClient fault contracts against unreachable and hung runtimes.

A real listening-but-silent socket and a real dead port stand in for the
runtime. dispatch_agent retries a read timeout once, then returns a warming
status dict so the gateway can tell a cold start from a dead runtime without
raising into the global error handler; CRUD calls stay bounded by the 30s
client default; connects fail within seconds — never an unbounded stall.
"""

import asyncio
import contextlib

import httpx
import pytest
from cogniverse_messaging.runtime_client import RuntimeClient

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

DEAD_PORT = 29071


@pytest.fixture
async def hung_server():
    """A server that accepts connections and never responds."""

    async def _hold(reader, writer):
        # Never reply; exit when the client gives up and disconnects, so
        # Server.wait_closed() (which waits on handlers since 3.12) returns.
        try:
            while await reader.read(65536):
                pass
        finally:
            writer.close()

    server = await asyncio.start_server(_hold, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def timeout_server():
    """A server that reads each request and never sends a response."""

    request_count = 0

    async def _hold(reader, writer):
        nonlocal request_count
        buffer = b""
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\r\n\r\n" in buffer:
                    request_count += 1
                    buffer = buffer.split(b"\r\n\r\n", 1)[1]
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    server = await asyncio.start_server(_hold, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port, lambda: request_count
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_dispatch_read_timeout_retries_once_and_reports_warming(
    timeout_server,
):
    """A timed-out dispatch retries once, then returns a warming status dict."""
    port, request_count = timeout_server
    rc = RuntimeClient(f"http://127.0.0.1:{port}", dispatch_timeout=0.1)
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
    finally:
        await rc.close()

    assert result["status"] == "warming"
    assert result["message"] == "runtime warming up, try again: ReadTimeout after 0.1s"
    assert request_count() == 2


@pytest.mark.asyncio
async def test_crud_read_bounded_by_client_default(hung_server):
    """CRUD calls do not inherit the long dispatch budget: the shared client
    times out reads at 30s (asserted structurally; executing a 30s wait per
    run is not acceptable) and this call must raise, not hang forever."""
    rc = RuntimeClient(f"http://127.0.0.1:{hung_server}", timeout=1.0)
    try:
        client = await rc._get_client()
        assert client.timeout.read == 30.0
        assert client.timeout.connect == 5.0
        with pytest.raises(httpx.TimeoutException):
            await client.get("/health", timeout=httpx.Timeout(1.0, connect=5.0))
    finally:
        await rc.close()


@pytest.mark.asyncio
async def test_dead_port_dispatch_degrades_fast():
    """A dead runtime port degrades dispatch to an unavailable status dict
    fast (connect fails in seconds), never raising ConnectError at the caller."""
    rc = RuntimeClient(f"http://127.0.0.1:{DEAD_PORT}", dispatch_timeout=1.0)
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
    finally:
        await rc.close()

    assert result["status"] == "unavailable"
    assert result["message"] == "runtime unreachable: ConnectError"


@pytest.mark.asyncio
async def test_dispatch_non_json_200_is_an_error():
    """A 200 whose body is not JSON (proxy error page) must degrade to an error
    status dict, not raise a JSONDecodeError into the handler."""

    def _handler(request):
        return httpx.Response(200, text="<html>Bad Gateway</html>")

    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
    finally:
        await rc.close()

    assert result["status"] == "error"
    assert "non-JSON" in result["message"]


@pytest.mark.asyncio
async def test_crud_2xx_with_non_json_body_is_an_error():
    """A 200 whose body is not JSON (proxy error page, wrong port) must
    surface as an error — reporting "ok" turned it into false success
    messages like "Instructions updated." for writes that never happened."""
    handler_responses = {"n": 0}

    def _handler(request):
        handler_responses["n"] += 1
        return httpx.Response(200, text="<html>Bad Gateway</html>")

    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )
    try:
        result = await rc.set_instructions(tenant_id="acme:alice", text="be brief")
    finally:
        await rc.close()

    assert result["status"] == "error"
    assert "non-JSON" in result["message"]


@pytest.mark.asyncio
async def test_crud_204_empty_body_stays_ok():
    """A 204 No Content is a legitimate empty success, not a broken body."""

    def _handler(request):
        return httpx.Response(204)

    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )
    try:
        result = await rc.delete_job(tenant_id="acme:alice", job_id="j1")
    finally:
        await rc.close()

    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_dispatch_context_cannot_override_tenant():
    """A caller-supplied context["tenant_id"] must lose to the authoritative
    tenant argument — the old merge order let any context dict silently
    redirect the request to another tenant."""
    seen = {}

    def _handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": "ok"})

    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )
    try:
        await rc.dispatch_agent(
            "gateway_agent",
            "q",
            tenant_id="acme:alice",
            context={"tenant_id": "evil:other", "media_type": "photo"},
        )
    finally:
        await rc.close()

    assert seen["context"]["tenant_id"] == "acme:alice"
    assert seen["context"]["media_type"] == "photo"
