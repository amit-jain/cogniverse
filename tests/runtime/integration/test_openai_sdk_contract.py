"""The /v1 surface parses through the official ``openai`` python SDK.

``test_openai_compat_endpoint`` pins the byte-level wire shape over raw
httpx. This module proves the same app against the client library the
harness clients actually speak (Pi's provider dialect is
``openai-completions``): object construction, stream accumulation, tool
calls, the SDK-native resume round trip, and the typed error classes.

Real router, real ``AgentDispatcher``, real ``AgentRegistry``, agents
registered through the production ``AGENT_CLASSES`` path. No LM: every
answer below is produced from the turn's own input, so the completion
text and the estimated usage are exact.
"""

from __future__ import annotations

import json
import re

import httpx
import openai
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.routers import openai_compat
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT = "test:unit"
KEY = "sdk-contract-key"
MODEL = "cogniverse/sdk-echo"
TOOL_CALL_ID = "call_sdk_write_file"

CONTENT_QUERY = "write the report"
TOOL_QUERY = "note the failure"
TOOL_RESULT = "file written ok"

CONTENT_ANSWER = f"answered: {CONTENT_QUERY}"
RESUMED_ANSWER = f"tool said: {TOOL_RESULT}; tenant={TENANT}; rounds=1; resumed=True"

# ``_finalize_usage`` has no LM tracker to read on these turns, so it falls
# back to the 4-chars-per-token estimate over (query + history) and the
# completion text: 16 prompt chars -> 4; the answers below at 26 / 68 chars
# -> 6 / 17; the serialized tool_calls at 136 chars -> 34.
CONTENT_USAGE = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
TOOL_USAGE = {"prompt_tokens": 4, "completion_tokens": 34, "total_tokens": 38}
RESUMED_USAGE = {"prompt_tokens": 4, "completion_tokens": 17, "total_tokens": 21}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the client workspace",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]

COMPLETION_ID = re.compile(r"chatcmpl-[0-9a-f]{32}")


class SdkEchoDeps(BaseModel):
    """The one Deps shape for this module."""


class SdkEchoInput(BaseModel):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []
    external_tools: list = []
    tool_results: list = []
    continuation_state: dict = {}
    tool_exchange: list = []


class SdkEchoOutput(BaseModel):
    status: str = "success"
    answer: str = ""
    pending_tool_calls: list = []
    continuation_state: dict = {}


class SdkEchoAgent:
    """Deterministic dual-loop agent with a fixed tool-call id.

    A turn carrying results answers from them; a turn that was handed tool
    definitions suspends on one call; a turn with neither answers directly.
    """

    def __init__(self, deps: SdkEchoDeps):
        self.deps = deps

    async def process(self, input: SdkEchoInput) -> SdkEchoOutput:
        if input.tool_results:
            results = "; ".join(result["content"] for result in input.tool_results)
            return SdkEchoOutput(
                answer=(
                    f"tool said: {results}; tenant={input.tenant_id}; "
                    f"rounds={len(input.tool_exchange)}; "
                    f"resumed={bool(input.continuation_state)}"
                )
            )
        if input.external_tools:
            return SdkEchoOutput(
                status="external_tool_calls",
                pending_tool_calls=[
                    {
                        "id": TOOL_CALL_ID,
                        "name": input.external_tools[0]["function"]["name"],
                        "arguments": {"text": input.query},
                    }
                ],
                continuation_state={"plan": f"plan-for-{input.tenant_id}"},
            )
        return SdkEchoOutput(answer=f"answered: {input.query}")


_AGENT_CLASSES = {"sdk_echo_agent": f"{__name__}:SdkEchoAgent"}
MODEL_MAP = {MODEL: "sdk_echo_agent"}


@pytest.fixture(scope="module")
def dispatcher():
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="sdk_echo_agent",
            url="http://localhost:8000",
            capabilities=["sdk_echo"],
        )
    )
    ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
    yield AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    for agent_name in _AGENT_CLASSES:
        ConfigLoader.AGENT_CLASSES.pop(agent_name, None)


@pytest.fixture()
def compat_app(dispatcher):
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    openai_compat.set_api_keys({KEY: TENANT})
    openai_compat.set_model_map(MODEL_MAP)
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()
    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    yield app
    openai_compat.set_dispatcher_provider(None)
    openai_compat.set_api_keys({})
    openai_compat.set_model_map({})
    openai_compat.set_key_resolver(None)
    openai_compat.clear_continuations()


def _sdk(app: FastAPI, key: str = KEY) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://testserver/v1",
        api_key=key,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=60.0,
        ),
        max_retries=0,
    )


@pytest.fixture()
async def sdk(compat_app):
    client = _sdk(compat_app)
    yield client
    await client.close()


class TestContentTurns:
    async def test_non_stream_completion_object(self, sdk):
        completion = await sdk.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": CONTENT_QUERY}]
        )

        assert COMPLETION_ID.fullmatch(completion.id)
        assert completion.object == "chat.completion"
        assert completion.model == MODEL
        assert len(completion.choices) == 1
        choice = completion.choices[0]
        assert (choice.index, choice.finish_reason) == (0, "stop")
        assert choice.message.role == "assistant"
        assert choice.message.content == CONTENT_ANSWER
        assert choice.message.tool_calls is None
        assert completion.usage.model_dump(include=set(CONTENT_USAGE)) == CONTENT_USAGE

    async def test_stream_accumulates_to_the_same_answer(self, sdk):
        stream = await sdk.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": CONTENT_QUERY}],
            stream=True,
        )
        roles: list[str] = []
        content_parts: list[str] = []
        finish_reasons: list[str] = []
        terminal_usage = None
        async for chunk in stream:
            assert chunk.object == "chat.completion.chunk"
            choice = chunk.choices[0]
            if choice.delta.role:
                roles.append(choice.delta.role)
            if choice.delta.content:
                content_parts.append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)
                terminal_usage = chunk.usage

        assert roles == ["assistant"]
        assert "".join(content_parts) == CONTENT_ANSWER
        assert finish_reasons == ["stop"]
        assert terminal_usage.model_dump(include=set(CONTENT_USAGE)) == CONTENT_USAGE


class TestToolCallTurns:
    async def test_non_stream_tool_calls_object(self, sdk):
        completion = await sdk.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TOOL_QUERY}],
            tools=TOOL_DEFS,
        )

        choice = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None
        (call,) = choice.message.tool_calls
        assert (call.id, call.type, call.function.name) == (
            TOOL_CALL_ID,
            "function",
            "write_file",
        )
        assert json.loads(call.function.arguments) == {"text": TOOL_QUERY}
        assert completion.usage.model_dump(include=set(TOOL_USAGE)) == TOOL_USAGE

    async def test_stream_tool_calls_accumulate(self, sdk):
        stream = await sdk.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": TOOL_QUERY}],
            tools=TOOL_DEFS,
            stream=True,
        )
        call_ids: list[str] = []
        indexes: list[int] = []
        name_parts: list[str] = []
        args_parts: list[str] = []
        finish_reasons: list[str] = []
        async for chunk in stream:
            choice = chunk.choices[0]
            for delta_call in choice.delta.tool_calls or []:
                indexes.append(delta_call.index)
                if delta_call.id:
                    call_ids.append(delta_call.id)
                if delta_call.function and delta_call.function.name:
                    name_parts.append(delta_call.function.name)
                if delta_call.function and delta_call.function.arguments:
                    args_parts.append(delta_call.function.arguments)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        assert call_ids == [TOOL_CALL_ID]
        assert set(indexes) == {0}
        assert "".join(name_parts) == "write_file"
        assert json.loads("".join(args_parts)) == {"text": TOOL_QUERY}
        assert finish_reasons == ["tool_calls"]

    async def test_sdk_native_resume_round_trip(self, sdk):
        """The assistant message the SDK parsed goes back verbatim with a
        tool result appended — the exact exchange an openai-dialect client
        performs between turns."""
        base = [{"role": "user", "content": CONTENT_QUERY}]
        first = await sdk.chat.completions.create(
            model=MODEL, messages=base, tools=TOOL_DEFS
        )
        assistant = first.choices[0].message.model_dump(exclude_none=True)

        second = await sdk.chat.completions.create(
            model=MODEL,
            messages=[
                *base,
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": TOOL_CALL_ID,
                    "content": TOOL_RESULT,
                },
            ],
            tools=TOOL_DEFS,
        )

        choice = second.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.content == RESUMED_ANSWER
        assert second.usage.model_dump(include=set(RESUMED_USAGE)) == RESUMED_USAGE


class TestModelsSurface:
    async def test_sdk_models_list(self, sdk):
        page = await sdk.models.list()

        assert [(model.id, model.object, model.owned_by) for model in page.data] == [
            (MODEL, "model", "cogniverse")
        ]


class TestErrorSurface:
    async def test_wrong_key_raises_authentication_error(self, compat_app):
        client = _sdk(compat_app, key="wrong-key")
        try:
            with pytest.raises(openai.AuthenticationError) as error:
                await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": CONTENT_QUERY}]
                )
        finally:
            await client.close()

        assert error.value.status_code == 401
        assert error.value.body == {
            "message": "Invalid or missing API key.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }

    async def test_unknown_model_raises_not_found(self, sdk):
        with pytest.raises(openai.NotFoundError) as error:
            await sdk.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": CONTENT_QUERY}]
            )

        assert error.value.status_code == 404
        assert error.value.body["code"] == "model_not_found"
