"""A failed retrieval ends the turn on every transport that serves an answer.

Real ``/v1`` router and real ``A2AStarletteApplication`` + ``CogniverseAgentExecutor``
over real uvicorn sockets, in front of a real ``AgentDispatcher`` whose search
backend refuses one tenant while the answer path is otherwise healthy. Each
transport must produce exactly one attributable terminal error, the summarizer
must never be constructed for that tenant, and a tenant whose backend is up must
answer from its own hits at the same time.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from fastapi import FastAPI

from cogniverse_agents.search_agent import SearchAgent as RealSearchAgent
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher, _flatten_search_hit
from cogniverse_runtime.routers import openai_compat
from tests.utils.memory_store import InMemoryConfigStore, register_deployed_schema

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

FAILING_TENANT = "acme:acme"
HEALTHY_TENANT = "globex:globex"
FAILING_KEY = "grounding-key-failing"
HEALTHY_KEY = "grounding-key-healthy"
MODEL = "cogniverse/summary"
QUERY = "what did the speaker say about the launch?"
BACKEND_REFUSAL = "vespa search backend unreachable"

_SHIPPED_CONFIG = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)
_SHIPPED_PROFILES = {
    name: BackendProfileConfig.from_dict(name, data)
    for name, data in _SHIPPED_CONFIG["backend"]["profiles"].items()
}
_SHIPPED_SERVICE_URLS = {
    service: "http://inference.test"
    for data in _SHIPPED_CONFIG["backend"]["profiles"].values()
    for service in [(data.get("inference_services") or {}).get("embedding")]
    if service
}

HEALTHY_HIT = {
    "document_id": "vid_7",
    "score": 0.91,
    "metadata": {
        "source_url": "s3://cogniverse-ingest/globex:globex/launch.mp4",
        "video_id": "launch",
        "segment_id": 7,
    },
}
HEALTHY_SUMMARY = "The speaker said the launch slipped to the following quarter."


class _SearchBackend(RealSearchAgent):
    """The real search agent with only its backend leg replaced.

    ``FAILING_TENANT``'s backend refuses; every other tenant's returns a hit.
    Everything above it — profile resolution, the budget read, the grounding
    envelope — is production code.
    """

    def __init__(self, profile):
        self.telemetry_manager = None
        self._input_rails = None
        self._output_rails = None
        self._profile = profile

    async def _process_impl(self, inp):
        from cogniverse_agents.search_agent import SearchOutput

        if inp.tenant_id == FAILING_TENANT:
            raise ConnectionError(BACKEND_REFUSAL)
        return SearchOutput(
            query=inp.query,
            profile=self._profile,
            search_mode="single_profile",
            results=[HEALTHY_HIT],
            total_results=1,
        )


@pytest.fixture
def summarizer_builds(monkeypatch) -> List[str]:
    """Record every SummarizerAgent construction, by tenant."""
    import cogniverse_agents.summarizer_agent as summarizer_module
    from cogniverse_agents.summarizer_agent import (
        SummaryResult,
        ThinkingPhase,
    )

    built: List[str] = []

    class _RecordingSummarizer:
        def __init__(self, deps=None, config_manager=None, **kwargs):
            built.append(getattr(deps, "tenant_id", None) or "")

        async def summarize(self, request):
            assert request.search_results == [_flatten_search_hit(HEALTHY_HIT)], (
                request.search_results
            )
            return SummaryResult(
                summary=HEALTHY_SUMMARY,
                key_points=[],
                visual_insights=[],
                confidence_score=1.0,
                thinking_phase=ThinkingPhase(
                    key_themes=[],
                    content_categories=[],
                    relevance_scores={},
                    visual_elements=[],
                    reasoning="",
                ),
                metadata={},
            )

    monkeypatch.setattr(summarizer_module, "SummarizerAgent", _RecordingSummarizer)
    return built


@pytest.fixture
def dispatcher(monkeypatch):
    config_manager = MagicMock()
    config_manager.get_system_config.return_value = SystemConfig(
        backend_url="http://localhost",
        backend_port=8080,
        inference_service_urls=dict(_SHIPPED_SERVICE_URLS),
    )
    config_manager.list_backend_profiles.return_value = dict(_SHIPPED_PROFILES)
    config_manager.active_profile = None
    store = InMemoryConfigStore()
    store.initialize()
    config_manager.store = store
    for tenant_id in (FAILING_TENANT, HEALTHY_TENANT):
        for profile in _SHIPPED_PROFILES.values():
            register_deployed_schema(config_manager, tenant_id, profile.schema_name)

    registry = AgentRegistry(tenant_id=FAILING_TENANT, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="summarizer_agent",
            url="http://unused",
            capabilities=["summarization"],
        )
    )
    built = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    built._get_search_agent = _SearchBackend
    built.consult_egress_policy = lambda *a, **k: None
    built._verify_egress = lambda *a, **k: None
    built._init_agent_memory = lambda *a, **k: None
    built._apply_artefact_overlay = lambda *a, **k: None
    built._build_answer_agent = lambda cls, deps_cls, name, tenant_id: cls(
        deps=deps_cls(tenant_id=tenant_id)
    )
    built._conversation_store_factory = lambda tenant_id: None
    return built


@pytest.fixture
def compat_url(dispatcher):
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys(
        {FAILING_KEY: FAILING_TENANT, HEALTHY_KEY: HEALTHY_TENANT}
    )
    openai_compat.set_model_map({MODEL: "summarizer_agent"})
    openai_compat.set_key_resolver(None)
    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    with _serving(app) as url:
        yield url
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})


@pytest.fixture
def a2a_url(dispatcher):
    card = AgentCard(
        name="Cogniverse",
        description="Grounding failure fixture",
        url="http://localhost/a2a/",
        version="1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(
            agent_executor=CogniverseAgentExecutor(dispatcher),
            task_store=InMemoryTaskStore(),
        ),
    ).build(rpc_url="/a2a/")
    with _serving(app) as url:
        yield url


class _serving:
    """Run an ASGI app on a real uvicorn socket for the duration of a block."""

    def __init__(self, app):
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._port = self._listener.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="off")
        )
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._listener]}, daemon=True
        )

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 20
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self._server.started is True, "uvicorn did not start"
        return f"http://127.0.0.1:{self._port}"

    def __exit__(self, *_exc):
        self._server.should_exit = True
        self._thread.join(20)
        self._listener.close()
        assert self._thread.is_alive() is False


def _chat_body(stream: bool) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": QUERY}],
        "stream": stream,
    }


def _a2a_request(tenant_id: str, stream: bool) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": f"grounding-{tenant_id}-{stream}",
        "method": "message/stream" if stream else "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "message_id": f"m-{tenant_id}-{stream}",
                "parts": [{"kind": "text", "text": QUERY}],
            },
            "metadata": {
                "agent_name": "summarizer_agent",
                "query": QUERY,
                "tenant_id": tenant_id,
                "stream": stream,
            },
        },
    }


def _sse_frames(text: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]


@pytest.mark.asyncio
async def test_v1_nonstream_fails_the_turn_on_a_grounding_outage(
    compat_url, summarizer_builds
):
    async with httpx.AsyncClient(base_url=compat_url, timeout=60.0) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=False),
            headers={"Authorization": f"Bearer {FAILING_KEY}"},
        )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["code"] == "internal_error"
    assert error["agent"] == "summarizer_agent"
    assert error["error_type"] == "AnswerGroundingUnavailable"
    assert summarizer_builds == []


@pytest.mark.asyncio
async def test_v1_stream_ends_in_one_error_frame_on_a_grounding_outage(
    compat_url, summarizer_builds
):
    async with httpx.AsyncClient(base_url=compat_url, timeout=60.0) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=True),
            headers={"Authorization": f"Bearer {FAILING_KEY}"},
        )

    assert response.status_code == 200
    frames = _sse_frames(response.text)
    errors = [frame["error"] for frame in frames if "error" in frame]
    assert len(errors) == 1
    assert errors[0]["type"] == "server_error"
    assert errors[0]["code"] == "internal_error"
    assert errors[0]["agent"] == "summarizer_agent"
    assert errors[0]["error_type"] == "AnswerGroundingUnavailable"
    assert response.text.rstrip().endswith("data: [DONE]")
    # The role-priming chunk is the only choices frame: no answer token and no
    # finish_reason ever reached the client.
    assert [frame.get("choices") for frame in frames if frame.get("choices")] == [
        [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    ]
    assert summarizer_builds == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_a2a_fails_the_task_on_a_grounding_outage(
    a2a_url, summarizer_builds, stream
):
    async with httpx.AsyncClient(base_url=a2a_url, timeout=60.0) as client:
        response = await client.post("/a2a/", json=_a2a_request(FAILING_TENANT, stream))

    assert response.status_code == 200
    events = (
        [frame["result"] for frame in _sse_frames(response.text)]
        if stream
        else [response.json()["result"]]
    )
    terminals = [event for event in events if event.get("final", not stream)]
    assert len(terminals) == 1
    assert terminals[0]["status"]["state"] == "failed"
    payload = json.loads(terminals[0]["status"]["message"]["parts"][0]["text"])
    assert payload["type"] == "error"
    assert payload["agent"] == "summarizer_agent"
    assert payload["error_type"] == "AnswerGroundingUnavailable"
    assert BACKEND_REFUSAL not in response.text
    assert summarizer_builds == []


@pytest.mark.asyncio
async def test_a_healthy_tenant_answers_while_the_other_backend_is_down(
    compat_url, summarizer_builds
):
    """Both turns in flight at once: one tenant's outage must not touch the
    other's answer, and the failing turn must still never reach the
    summarizer."""
    async with httpx.AsyncClient(base_url=compat_url, timeout=60.0) as client:
        failing, healthy = await _gather(
            client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
                headers={"Authorization": f"Bearer {FAILING_KEY}"},
            ),
            client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False),
                headers={"Authorization": f"Bearer {HEALTHY_KEY}"},
            ),
        )

    assert failing.status_code == 500
    assert failing.json()["error"]["error_type"] == "AnswerGroundingUnavailable"
    assert healthy.status_code == 200
    assert healthy.json()["choices"][0]["message"]["content"].strip() == HEALTHY_SUMMARY
    assert summarizer_builds == [HEALTHY_TENANT]


async def _gather(*awaitables):
    return await asyncio.gather(*awaitables)
