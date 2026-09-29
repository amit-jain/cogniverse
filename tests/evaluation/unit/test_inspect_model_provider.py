"""The ``cogniverse`` Inspect AI model provider calls the LM through litellm.

Each test drives a real litellm request against a local OpenAI-compatible
server that records what arrives and answers what the test scripts, so the
assertions cover the wire request (messages, sampling, auth, headers) and the
mapping of each response or failure onto Inspect AI's types.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import ToolInfo, ToolParams

from cogniverse_evaluation.core.inspect_model import (
    PROVIDER_NAME,
    CogniverseModelAPI,
    inspect_model,
)
from cogniverse_foundation.config.bootstrap import INFERENCE_API_KEY_ENV
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

pytestmark = pytest.mark.unit

IMAGE = "data:image/png;base64,iVBORw0KGgo="


def _completion(content, finish_reason="stop", prompt=11, completion=3):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class _Endpoint:
    """OpenAI-compatible chat endpoint answering from a scripted queue.

    An empty queue echoes the last user message back as the completion.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: list[tuple[int, dict, float]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                if endpoint.replies:
                    status, payload, delay = endpoint.replies.pop(0)
                else:
                    status, payload, delay = (
                        200,
                        _completion(body["messages"][-1]["content"]),
                        0.0,
                    )
                time.sleep(delay)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def endpoint():
    server = _Endpoint()
    try:
        yield server
    finally:
        server.close()


def _api(endpoint, **overrides) -> CogniverseModelAPI:
    config = LLMEndpointConfig(
        model="openai/stub-model", api_base=endpoint.url, api_key="sk-test"
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return inspect_model(config).api


def _error(message, kind):
    return {"error": {"message": message, "type": kind, "code": None}}


@pytest.mark.asyncio
async def test_messages_sampling_auth_and_headers_reach_the_endpoint(endpoint):
    endpoint.replies.append((200, _completion("hello"), 0.0))
    api = _api(endpoint, extra_headers={"x-vsr-task": "evaluation"}, seed=7)

    output = await api.generate(
        input=[
            ChatMessageSystem(content="be terse"),
            ChatMessageUser(content="first"),
            ChatMessageAssistant(content="earlier answer"),
            ChatMessageUser(
                content=[ContentText(text="describe"), ContentImage(image=IMAGE)]
            ),
        ],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(
            temperature=0.2, max_tokens=64, top_p=0.9, stop_seqs=["END"]
        ),
    )

    [request] = endpoint.requests
    body = request["body"]
    assert request["path"] == "/v1/chat/completions"
    assert body["model"] == "stub-model"
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": IMAGE}},
            ],
        },
    ]
    assert (body["temperature"], body["max_tokens"], body["top_p"]) == (0.2, 64, 0.9)
    assert (body["stop"], body["seed"]) == (["END"], 7)
    assert request["headers"]["authorization"] == "Bearer sk-test"
    assert request["headers"]["x-vsr-task"] == "evaluation"
    assert output.model == "openai/stub-model"
    assert output.completion == "hello"
    assert output.stop_reason == "stop"
    assert (
        output.usage.input_tokens,
        output.usage.output_tokens,
        output.usage.total_tokens,
    ) == (11, 3, 14)


@pytest.mark.asyncio
async def test_endpoint_sampling_defaults_apply_when_config_leaves_them_unset(
    endpoint,
):
    api = _api(endpoint, temperature=0.4, max_tokens=321)

    await api.generate(
        input=[ChatMessageUser(content="q")],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    body = endpoint.requests[0]["body"]
    assert (body["temperature"], body["max_tokens"]) == (0.4, 321)
    assert "top_p" not in body and "stop" not in body


@pytest.mark.asyncio
async def test_keyless_endpoint_sends_the_inference_bearer(endpoint, monkeypatch):
    monkeypatch.setenv(INFERENCE_API_KEY_ENV, "bearer-from-env")
    api = _api(endpoint, api_key=None)

    await api.generate(
        input=[ChatMessageUser(content="q")],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    assert endpoint.requests[0]["headers"]["authorization"] == "Bearer bearer-from-env"


@pytest.mark.asyncio
async def test_length_finish_maps_to_max_tokens(endpoint):
    endpoint.replies.append((200, _completion("trunc", finish_reason="length"), 0.0))

    output = await _api(endpoint).generate(
        input=[ChatMessageUser(content="q")],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    assert (output.completion, output.stop_reason) == ("trunc", "max_tokens")


@pytest.mark.asyncio
async def test_context_window_overflow_returns_a_model_length_output(endpoint):
    endpoint.replies.append(
        (
            400,
            _error(
                "This model's maximum context length is 8192 tokens. However, "
                "you requested 9000 tokens.",
                "invalid_request_error",
            ),
            0.0,
        )
    )

    output = await _api(endpoint).generate(
        input=[ChatMessageUser(content="q")],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
    )

    assert output.stop_reason == "model_length"
    assert output.completion == ""
    assert "maximum context length is 8192 tokens" in output.error


@pytest.mark.asyncio
async def test_tool_use_is_refused_before_any_request(endpoint):
    tool = ToolInfo(name="lookup", description="find", parameters=ToolParams())

    with pytest.raises(ValueError, match="does not support tool calls"):
        await _api(endpoint).generate(
            input=[ChatMessageUser(content="q")],
            tools=[tool],
            tool_choice="auto",
            config=GenerateConfig(),
        )
    with pytest.raises(ValueError, match="does not support tool calls"):
        await _api(endpoint).generate(
            input=[ChatMessageTool(content="r", tool_call_id="c1", function="f")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )
    assert endpoint.requests == []


@pytest.mark.asyncio
async def test_request_timeout_raises_and_is_a_transient_retry(endpoint):
    endpoint.replies.append((200, _completion("late"), 3.0))
    api = _api(endpoint)

    with pytest.raises(litellm.Timeout) as excinfo:
        await api.generate(
            input=[ChatMessageUser(content="q")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(timeout=1),
        )

    decision = api.should_retry(excinfo.value)
    assert (decision.retry, decision.kind) == (True, "transient")
    assert len(endpoint.requests) == 1


@pytest.mark.asyncio
async def test_rate_limit_is_a_rate_limit_retry(endpoint):
    endpoint.replies.append((429, _error("slow down", "rate_limit_error"), 0.0))
    api = _api(endpoint)

    with pytest.raises(litellm.RateLimitError) as excinfo:
        await api.generate(
            input=[ChatMessageUser(content="q")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

    decision = api.should_retry(excinfo.value)
    assert (decision.retry, decision.kind) == (True, "rate_limit")
    assert len(endpoint.requests) == 1


@pytest.mark.asyncio
async def test_server_error_is_retried_by_inspect_not_by_litellm(endpoint):
    endpoint.replies.append((503, _error("overloaded", "server_error"), 0.0))
    api = _api(endpoint)

    with pytest.raises(litellm.ServiceUnavailableError) as excinfo:
        await api.generate(
            input=[ChatMessageUser(content="q")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

    decision = api.should_retry(excinfo.value)
    assert (decision.retry, decision.kind) == (True, "transient")
    assert len(endpoint.requests) == 1


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried(endpoint):
    endpoint.replies.append((401, _error("bad key", "invalid_api_key"), 0.0))
    api = _api(endpoint)

    with pytest.raises(litellm.AuthenticationError) as excinfo:
        await api.generate(
            input=[ChatMessageUser(content="q")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

    assert bool(api.should_retry(excinfo.value)) is False
    assert api.is_auth_failure(excinfo.value) is True


@pytest.mark.asyncio
async def test_unreachable_endpoint_raises_a_transient_error():
    api = inspect_model(
        LLMEndpointConfig(
            model="openai/stub-model", api_base="http://127.0.0.1:1/v1", api_key="k"
        )
    ).api

    with pytest.raises(
        litellm.InternalServerError, match="Connection error"
    ) as excinfo:
        await api.generate(
            input=[ChatMessageUser(content="q")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
        )

    decision = api.should_retry(excinfo.value)
    assert (decision.retry, decision.kind) == (True, "transient")


@pytest.mark.asyncio
async def test_concurrent_generations_each_get_their_own_completion(endpoint):
    api = _api(endpoint)
    prompts = [f"prompt-{i}" for i in range(8)]

    outputs = await asyncio.gather(
        *[
            api.generate(
                input=[ChatMessageUser(content=prompt)],
                tools=[],
                tool_choice="none",
                config=GenerateConfig(),
            )
            for prompt in prompts
        ]
    )

    assert [output.completion for output in outputs] == prompts
    assert sorted(r["body"]["messages"][-1]["content"] for r in endpoint.requests) == (
        sorted(prompts)
    )


@pytest.mark.asyncio
async def test_provider_resolves_by_name_through_get_model(endpoint):
    model = get_model(
        f"{PROVIDER_NAME}/openai/stub-model",
        base_url=endpoint.url,
        api_key="sk-test",
    )

    output = await model.generate("named")

    assert isinstance(model.api, CogniverseModelAPI)
    assert output.completion == "named"
    assert endpoint.requests[0]["body"]["model"] == "stub-model"
