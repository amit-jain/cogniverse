"""RuntimeClient fault contracts against unreachable and hung runtimes.

A real listening-but-silent socket and a real dead port stand in for the
runtime. Dispatch reads are bounded by the constructor timeout, everything
else by the 30s client default, and connects fail within seconds — a hung
runtime must surface as a prompt TimeoutException in the handler (where the
gateway's error handler answers the user), never an unbounded stall.
"""

import asyncio
import time

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


@pytest.mark.asyncio
async def test_dispatch_read_bounded_by_constructor_timeout(hung_server):
    rc = RuntimeClient(f"http://127.0.0.1:{hung_server}", timeout=1.0)
    start = time.monotonic()
    try:
        with pytest.raises(httpx.TimeoutException):
            await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 10
    finally:
        await rc.close()


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
async def test_dead_port_raises_connect_error_fast():
    rc = RuntimeClient(f"http://127.0.0.1:{DEAD_PORT}", timeout=1.0)
    start = time.monotonic()
    try:
        with pytest.raises(httpx.ConnectError):
            await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 10
    finally:
        await rc.close()


@pytest.mark.asyncio
async def test_dispatch_timeout_overrides_client_default(hung_server):
    """The per-call dispatch timeout (constructor value) governs the agent
    call even though the shared client default is 30s — a 0.5s constructor
    timeout must fire in well under the client default."""
    rc = RuntimeClient(f"http://127.0.0.1:{hung_server}", timeout=0.5)
    start = time.monotonic()
    try:
        with pytest.raises(httpx.TimeoutException):
            await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 5
    finally:
        await rc.close()


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
