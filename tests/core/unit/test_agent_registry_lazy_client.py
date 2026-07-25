"""AgentRegistry's httpx client is created lazily, not on construction.

A registry used only for local agent lookup should never open (or have to
close) an httpx client.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry


def _registry() -> AgentRegistry:
    return AgentRegistry(tenant_id="acme:unit", config_manager=object())


@pytest.mark.unit
@pytest.mark.ci_fast
def test_http_client_created_lazily_and_cached():
    reg = _registry()

    assert reg._http_client is None  # not created yet
    client = reg.http_client
    assert isinstance(client, httpx.AsyncClient)
    assert reg.http_client is client  # cached, not rebuilt
    asyncio.run(client.aclose())


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_close_without_use_creates_no_client():
    reg = _registry()

    await reg.close()  # must not construct a client just to close it

    assert reg._http_client is None


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_http_client_constructed_once_under_concurrent_first_access(monkeypatch):
    import cogniverse_core.registries.agent_registry as registry_module

    reg = _registry()
    caller_count = 24
    start = threading.Barrier(caller_count)
    created = []
    created_lock = threading.Lock()
    real_client = httpx.AsyncClient

    def slow_client(*args, **kwargs):
        time.sleep(0.02)
        client = real_client(*args, **kwargs)
        with created_lock:
            created.append(client)
        return client

    monkeypatch.setattr(registry_module.httpx, "AsyncClient", slow_client)

    def access_client(_):
        start.wait()
        return reg.http_client

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        clients = list(executor.map(access_client, range(caller_count)))

    try:
        assert len(created) == 1
        assert all(client is created[0] for client in clients)
        await reg.close()
        assert created[0].is_closed
    finally:
        await asyncio.gather(
            *(client.aclose() for client in created if not client.is_closed)
        )


@pytest.mark.unit
@pytest.mark.ci_fast
def test_http_client_construction_failure_allows_retry(monkeypatch):
    import cogniverse_core.registries.agent_registry as registry_module

    reg = _registry()
    real_client = httpx.AsyncClient
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("client construction failed")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(registry_module.httpx, "AsyncClient", fail_once)

    with pytest.raises(RuntimeError, match="client construction failed"):
        _ = reg.http_client
    assert reg._http_client is None

    client = reg.http_client
    assert attempts == 2
    assert reg.http_client is client
    asyncio.run(client.aclose())


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_close_detaches_client_before_waiting_for_shutdown(monkeypatch):
    import cogniverse_core.registries.agent_registry as registry_module

    class BlockingClient:
        def __init__(self):
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_count = 0

        async def aclose(self):
            self.close_count += 1
            self.close_started.set()
            await self.release_close.wait()

    reg = _registry()
    closing_client = BlockingClient()
    replacement_client = object()
    reg._http_client = closing_client
    monkeypatch.setattr(
        registry_module.httpx,
        "AsyncClient",
        lambda **_kwargs: replacement_client,
    )

    close_task = asyncio.create_task(reg.close())
    await asyncio.wait_for(closing_client.close_started.wait(), timeout=1)

    assert reg._http_client is None
    assert reg.http_client is replacement_client

    closing_client.release_close.set()
    await close_task
    assert closing_client.close_count == 1
    assert reg._http_client is replacement_client
