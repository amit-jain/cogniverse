"""RuntimeClient fault contracts against unreachable and hung runtimes.

A real listening-but-silent socket and a real dead port stand in for the
runtime. dispatch_agent degrades a dead / hung / non-JSON runtime to a status
dict (so the gateway renders a graceful "unavailable" reply instead of raising
into the global error handler), bounded by ``dispatch_timeout``; CRUD calls are
bounded by the 30s client default; connects fail within seconds — never an
unbounded stall.
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
async def test_dispatch_hung_runtime_degrades_within_budget(hung_server):
    """A hung runtime must not stall the chat: dispatch returns an 'unavailable'
    status dict within the dispatch budget, not raise or hang."""
    rc = RuntimeClient(f"http://127.0.0.1:{hung_server}", dispatch_timeout=1.0)
    start = time.monotonic()
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 10
        assert result["status"] == "unavailable"
        assert "unreachable" in result["message"]
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
async def test_dead_port_dispatch_degrades_fast():
    """A dead runtime port degrades dispatch to an 'unavailable' status dict
    fast (connect fails in seconds), never raising ConnectError at the caller."""
    rc = RuntimeClient(f"http://127.0.0.1:{DEAD_PORT}", dispatch_timeout=1.0)
    start = time.monotonic()
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 10
        assert result["status"] == "unavailable"
        assert "unreachable" in result["message"]
    finally:
        await rc.close()


@pytest.mark.asyncio
async def test_dispatch_timeout_governs_the_agent_call(hung_server):
    """dispatch_timeout (not the 300s stream timeout) governs the agent call —
    a 0.5s dispatch budget degrades in well under the client default."""
    rc = RuntimeClient(f"http://127.0.0.1:{hung_server}", dispatch_timeout=0.5)
    start = time.monotonic()
    try:
        result = await rc.dispatch_agent("gateway_agent", "q", "acme:alice")
        assert time.monotonic() - start < 5
        assert result["status"] == "unavailable"
    finally:
        await rc.close()


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
