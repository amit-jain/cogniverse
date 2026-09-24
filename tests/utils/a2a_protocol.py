"""The ``/a2a`` protocol the runtime lifespan serves, for tests to serve too."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI


@asynccontextmanager
async def production_a2a(
    agent_registry: Any, dispatcher: Any, redis_url: str
) -> AsyncIterator[Any]:
    """The runtime's A2A protocol on ``redis_url`` with its production settings.

    Built in the caller's loop: the store's Redis client and the handler's
    cancel listener belong to the loop that runs them. ``cogniverse_runtime.main``
    is imported here, not at module import: importing it configures the
    runtime's logging for the whole test process.
    """
    from cogniverse_runtime.main import (
        _a2a_settings_from_env,
        _build_shared_a2a_protocol,
    )

    protocol = await _build_shared_a2a_protocol(
        agent_registry=agent_registry,
        dispatcher=dispatcher,
        redis_url=redis_url,
        replica_id=f"test-replica-{uuid.uuid4().hex}",
        **_a2a_settings_from_env({}),
    )
    try:
        yield protocol
    finally:
        await protocol.close()


def a2a_app(
    agent_registry: Any, dispatcher: Any, redis_url: str, *, path: str = "/a2a"
) -> FastAPI:
    """An app whose lifespan mounts that protocol at ``path``, as the runtime does."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        async with production_a2a(agent_registry, dispatcher, redis_url) as protocol:
            app.mount(path, protocol.app)
            yield

    return FastAPI(lifespan=_lifespan)
