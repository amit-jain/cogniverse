"""Serve ACP over stdio in a dedicated process."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cogniverse_runtime.acp import STDIN_LINE_LIMIT as _STDIN_LINE_LIMIT

logger = logging.getLogger(__name__)
_WRITE_CHUNK_BYTES = 64 * 1024


def _build_dispatcher() -> tuple[Any, Any]:
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.config_loader import get_config_loader
    from cogniverse_runtime.routers import agents

    config_manager = create_default_config_manager()
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    agent_registry = AgentRegistry(
        tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager
    )
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents(agent_registry=agent_registry)
    agents.set_agent_registry(agent_registry)
    agents.set_agent_dependencies(config_manager, schema_loader)
    return config_manager, agents.get_dispatcher()


class _LazyDispatcher:
    def __init__(self, build: Callable[[], tuple[Any, Any]]) -> None:
        self._build = build
        self._lock = threading.Lock()
        self._dispatcher: Any = None

    def __call__(self) -> Any:
        with self._lock:
            if self._dispatcher is None:
                _, self._dispatcher = self._build()
                logger.info("ACP dispatcher built")
            return self._dispatcher


def _resolve_agents(
    agent: str | None, coding_agent: str | None, config_path: Path
) -> tuple[str, str]:
    with config_path.open() as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("ACP requires a configuration object")
    harness = config.get("harness")
    if not isinstance(harness, dict) or not isinstance(harness.get("models"), dict):
        raise ValueError("ACP requires a harness.models object")
    models = harness["models"]
    agent = agent or models.get("cogniverse")
    coding_agent = coding_agent or models.get("cogniverse/coding")
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("ACP requires harness.models.cogniverse")
    if not isinstance(coding_agent, str) or not coding_agent.strip():
        raise ValueError("ACP requires harness.models.cogniverse/coding")
    return agent, coding_agent


class _WritePipeProtocol(asyncio.streams.FlowControlMixin):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(loop=loop)
        self._closed = loop.create_future()

    def connection_lost(self, exc: Exception | None) -> None:
        super().connection_lost(exc)
        if not self._closed.done():
            self._closed.set_result(None)

    def _get_close_waiter(self, stream: asyncio.StreamWriter) -> asyncio.Future:
        return self._closed


class _PipeWriter:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lock = asyncio.Lock()
        self._writer: asyncio.StreamWriter | None = None
        self._pending_write: asyncio.Task | None = None
        self._closed = False

    async def __call__(self, message: dict[str, Any]) -> None:
        await self._lock.acquire()
        if self._closed:
            self._lock.release()
            raise ConnectionError("ACP stdout is closed")
        task = asyncio.create_task(self._write_frame(message))
        self._pending_write = task
        task.add_done_callback(self._write_finished)
        await asyncio.shield(task)

    def _write_finished(self, task: asyncio.Task) -> None:
        self._pending_write = None
        self._lock.release()
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("ACP stdout write failed: %s", error)

    async def _write_frame(self, message: dict[str, Any]) -> None:
        if self._writer is None:
            loop = asyncio.get_running_loop()
            transport, protocol = await loop.connect_write_pipe(
                lambda: _WritePipeProtocol(loop), self._stream
            )
            self._writer = asyncio.StreamWriter(transport, protocol, None, loop)
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        for offset in range(0, len(data), _WRITE_CHUNK_BYTES):
            self._writer.write(data[offset : offset + _WRITE_CHUNK_BYTES])
            await self._writer.drain()

    async def close(self) -> None:
        self._closed = True
        pending = self._pending_write
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        if self._writer is None:
            self._stream.close()
            return
        if pending is not None:
            self._writer.transport.abort()
        else:
            self._writer.close()
        try:
            await asyncio.wait_for(self._writer.wait_closed(), timeout=1)
        except TimeoutError:
            self._writer.transport.abort()
            logger.warning("ACP stdout close timed out after 1 second")


def _make_writer(stream: Any) -> _PipeWriter:
    return _PipeWriter(stream)


async def _stdin_reader() -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_STDIN_LINE_LIMIT)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    return reader


async def _run() -> int:
    env = os.environ
    tenant = env.get("COGNIVERSE_ACP_TENANT")
    agent = env.get("COGNIVERSE_ACP_AGENT")
    coding_agent = env.get("COGNIVERSE_ACP_CODING_AGENT")
    log_level = env.get("LOG_LEVEL", "INFO")
    config_path = Path(env.get("COGNIVERSE_CONFIG", "configs/config.json"))
    if not tenant:
        print("cogniverse ACP: set COGNIVERSE_ACP_TENANT", file=sys.stderr)
        return 2

    protocol_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    write = _make_writer(protocol_out)
    try:
        from cogniverse_runtime.entrypoint_env import (
            configure_runtime_library_defaults,
            resolve_library_env_defaults,
        )

        configure_runtime_library_defaults(resolve_library_env_defaults())
        agent, coding_agent = await asyncio.to_thread(
            _resolve_agents, agent, coding_agent, config_path
        )
        from cogniverse_runtime.acp.server import ACPServer, ClientConnection, serve

        server = ACPServer(
            dispatcher_provider=_LazyDispatcher(_build_dispatcher),
            agent_provider=lambda: agent,
            coding_agent_provider=lambda: coding_agent,
            default_tenant=tenant,
        )
        reader = await _stdin_reader()
        logger.info("ACP serving tenant=%s", tenant)
        return await serve(
            server, ClientConnection(reader, write), line_limit=_STDIN_LINE_LIMIT
        )
    except Exception:
        logger.exception("ACP startup failed")
        return 2
    finally:
        await write.close()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
