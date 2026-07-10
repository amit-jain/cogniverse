"""AgentRegistry's httpx client is created lazily, not on construction.

A registry used only for local agent lookup should never open (or have to
close) an httpx client.
"""

from __future__ import annotations

import httpx
import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry


@pytest.mark.unit
@pytest.mark.ci_fast
def test_http_client_created_lazily_and_cached():
    reg = object.__new__(AgentRegistry)
    reg._http_client = None

    assert reg._http_client is None  # not created yet
    client = reg.http_client
    assert isinstance(client, httpx.AsyncClient)
    assert reg.http_client is client  # cached, not rebuilt


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.asyncio
async def test_close_without_use_creates_no_client():
    reg = object.__new__(AgentRegistry)
    reg._http_client = None

    await reg.close()  # must not construct a client just to close it

    assert reg._http_client is None
