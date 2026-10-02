"""A ``/v1`` turn suspended on one process resumes on another.

Two OS processes each serve the real ``/v1`` router over the real dispatcher on
a uvicorn socket, opening the shared continuation store from their own lifespan
as the runtime does. The client suspends a turn on one process and replays the
transcript with its tool result to the other. The deterministic agent echoes
the continuation state it was resumed with, so the resumed answer pins whether
the second process found the first one's state.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from tests.runtime.integration.test_openai_compat_endpoint import (
    KEY_A,
    KEY_B,
    MODEL_MAP,
    QUERY,
    TENANT_A,
    TENANT_A_RAW,
    TENANT_B,
    TOOL_CALL_ID,
    TOOL_DEFS,
    _body,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

BOOT_TIMEOUT_S = 120


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve(redis_url: str, prefix: str, port: int) -> None:
    """One runtime-shaped process: the /v1 router, the real dispatcher and the
    tool-echo agent, with the continuation store opened by its lifespan."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from cogniverse_core.common.agent_models import AgentEndpoint
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager
    from cogniverse_runtime.agent_dispatcher import AgentDispatcher
    from cogniverse_runtime.config_loader import ConfigLoader
    from cogniverse_runtime.routers import openai_compat
    from cogniverse_runtime.session_state import ContinuationStore, open_session_redis
    from tests.runtime.integration.test_openai_compat_endpoint import _AGENT_CLASSES
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    # As the runtime's startup does: the agents built here read this manager.
    get_telemetry_manager(config_manager)
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    for agent_name in _AGENT_CLASSES:
        registry.register_agent(
            AgentEndpoint(
                name=agent_name,
                url="http://localhost:8000",
                capabilities=[agent_name.removesuffix("_agent")],
            )
        )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )

    @asynccontextmanager
    async def lifespan(_app):
        redis = await open_session_redis(redis_url)
        openai_compat.set_continuation_store(
            ContinuationStore(redis, key_prefix=prefix)
        )
        try:
            yield
        finally:
            openai_compat.set_continuation_store(None)
            await redis.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(openai_compat.router, prefix="/v1")

    @app.get("/pid")
    async def pid():
        return JSONResponse({"pid": os.getpid()})

    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY_A: TENANT_A_RAW, KEY_B: TENANT_B})
    openai_compat.set_model_map(MODEL_MAP)
    openai_compat.set_key_resolver(None)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


@pytest.fixture
def two_processes(workflow_state_redis_url):
    context = multiprocessing.get_context("spawn")
    prefix = f"test:continuation:{uuid.uuid4().hex}"
    ports = [_free_port(), _free_port()]
    processes = [
        context.Process(target=_serve, args=(workflow_state_redis_url, prefix, port))
        for port in ports
    ]
    for process in processes:
        process.start()
    try:
        urls = [f"http://127.0.0.1:{port}" for port in ports]
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        pids = []
        for url, process in zip(urls, processes):
            while True:
                assert process.exitcode is None, f"{url} exited with {process.exitcode}"
                try:
                    pids.append(httpx.get(f"{url}/pid", timeout=2).json()["pid"])
                    break
                except httpx.TransportError:
                    assert time.monotonic() < deadline, f"{url} did not start"
                    time.sleep(0.2)
        assert pids == [process.pid for process in processes]
        yield urls, prefix
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=30)


def _resume_messages(tool_calls):
    return [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": TOOL_CALL_ID, "content": "wrote it"},
    ]


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


class TestContinuationAcrossProcesses:
    def test_a_turn_suspended_on_one_process_resumes_on_the_other(self, two_processes):
        (first, second), _ = two_processes

        suspended = httpx.post(
            f"{first}/v1/chat/completions",
            json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
            timeout=60,
        )
        assert suspended.status_code == 200, suspended.text
        choice = suspended.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"

        resumed = httpx.post(
            f"{second}/v1/chat/completions",
            json=_body(
                model="cogniverse/tools",
                tools=TOOL_DEFS,
                messages=_resume_messages(choice["message"]["tool_calls"]),
            ),
            headers=_auth(KEY_A),
            timeout=60,
        )

        assert resumed.status_code == 200, resumed.text
        assert json.loads(resumed.json()["choices"][0]["message"]["content"]) == {
            "tenant_id": TENANT_A,
            "resumed_state": {"plan": f"plan-for-{TENANT_A}"},
            "rounds": 1,
            "results": ["wrote it"],
        }

    def test_a_resumed_turn_is_gone_for_every_process(self, two_processes):
        """The state is taken once: replaying the same transcript to the
        process that suspended it re-derives instead of resuming twice."""
        (first, second), _ = two_processes
        suspended = httpx.post(
            f"{first}/v1/chat/completions",
            json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
            timeout=60,
        )
        resume = _body(
            model="cogniverse/tools",
            tools=TOOL_DEFS,
            messages=_resume_messages(
                suspended.json()["choices"][0]["message"]["tool_calls"]
            ),
        )

        answers = [
            json.loads(
                httpx.post(
                    f"{url}/v1/chat/completions",
                    json=resume,
                    headers=_auth(KEY_A),
                    timeout=60,
                ).json()["choices"][0]["message"]["content"]
            )["resumed_state"]
            for url in (second, first)
        ]

        assert answers == [{"plan": f"plan-for-{TENANT_A}"}, {}]

    def test_another_tenant_on_another_process_misses_the_state(self, two_processes):
        (first, second), _ = two_processes
        suspended = httpx.post(
            f"{first}/v1/chat/completions",
            json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
            headers=_auth(KEY_A),
            timeout=60,
        )
        resume = _body(
            model="cogniverse/tools",
            tools=TOOL_DEFS,
            messages=_resume_messages(
                suspended.json()["choices"][0]["message"]["tool_calls"]
            ),
        )

        thief = httpx.post(
            f"{second}/v1/chat/completions",
            json=resume,
            headers=_auth(KEY_B),
            timeout=60,
        )
        owner = httpx.post(
            f"{second}/v1/chat/completions",
            json=resume,
            headers=_auth(KEY_A),
            timeout=60,
        )

        assert json.loads(thief.json()["choices"][0]["message"]["content"]) == {
            "tenant_id": TENANT_B,
            "resumed_state": {},
            "rounds": 1,
            "results": ["wrote it"],
        }
        assert json.loads(owner.json()["choices"][0]["message"]["content"]) == {
            "tenant_id": TENANT_A,
            "resumed_state": {"plan": f"plan-for-{TENANT_A}"},
            "rounds": 1,
            "results": ["wrote it"],
        }
