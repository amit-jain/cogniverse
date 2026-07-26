"""Lifecycle contracts for the orchestrator's loop-scoped HTTP client."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import cogniverse_agents.orchestrator_agent as orchestrator_module


@pytest.fixture(autouse=True)
def _clear_loop_resources():
    orchestrator_module._http_clients.clear()
    orchestrator_module._orch_semaphores.clear()
    yield
    orchestrator_module._http_clients.clear()
    orchestrator_module._orch_semaphores.clear()


@pytest.mark.asyncio
async def test_concurrent_cold_get_builds_one_client():
    client = AsyncMock()
    client.is_closed = False

    with patch.object(
        orchestrator_module.httpx,
        "AsyncClient",
        return_value=client,
    ) as client_type:
        clients = await asyncio.gather(
            *(orchestrator_module._get_http_client() for _ in range(20))
        )

    assert clients == [client] * 20
    client_type.assert_called_once_with(
        timeout=orchestrator_module._HTTP_CLIENT_TIMEOUT
    )


@pytest.mark.asyncio
async def test_concurrent_close_closes_current_loop_client_once():
    loop_key = id(asyncio.get_running_loop())
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    client = AsyncMock()
    client.is_closed = False

    async def _close():
        close_started.set()
        await allow_close.wait()

    client.aclose.side_effect = _close
    orchestrator_module._http_clients[loop_key] = client
    orchestrator_module._orch_semaphores[loop_key] = asyncio.Semaphore(1)

    first_close = asyncio.create_task(
        orchestrator_module.close_orchestrator_http_client()
    )
    await close_started.wait()
    second_close = asyncio.create_task(
        orchestrator_module.close_orchestrator_http_client()
    )
    await second_close
    allow_close.set()
    await first_close

    client.aclose.assert_awaited_once_with()
    assert loop_key not in orchestrator_module._http_clients
    assert loop_key not in orchestrator_module._orch_semaphores


@pytest.mark.asyncio
async def test_close_failure_clears_resources_and_raises_with_context():
    loop_key = id(asyncio.get_running_loop())
    client = AsyncMock()
    client.is_closed = False
    client.aclose.side_effect = OSError("socket close failed")
    orchestrator_module._http_clients[loop_key] = client
    orchestrator_module._orch_semaphores[loop_key] = asyncio.Semaphore(1)

    with pytest.raises(
        RuntimeError,
        match="Failed to close orchestrator HTTP client",
    ) as exc_info:
        await orchestrator_module.close_orchestrator_http_client()

    assert isinstance(exc_info.value.__cause__, OSError)
    assert loop_key not in orchestrator_module._http_clients
    assert loop_key not in orchestrator_module._orch_semaphores
