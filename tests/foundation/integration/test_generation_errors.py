"""Incomplete LM output ends the turn as an error, through the summary route.

Real FastAPI route -> real AgentDispatcher -> real SummarizerAgent -> real
DSPy/LiteLLM -> a real OpenAI-compatible HTTP provider. Nothing on that path
is mocked: the provider is an out-of-process-shaped HTTP server the test
drives, so a response that stops before the signature's required outputs is
the genuine wire condition a truncated generation produces.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from pathlib import Path

import dspy
import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.dspy import LenientJSONAdapter
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

MODEL = "openai/generation-contract"
TENANTS = ("prodfixfoundation:generationa", "prodfixfoundation:generationb")
COMPLETE = {
    "reasoning": "Both telescopes are named and the start time is given.",
    "summary": "The observatory runs two telescopes and starts observing at sunset.",
    "key_points": "two telescopes, observations start at sunset",
    "confidence_score": "0.9",
}
# What each case leaves out of the response the provider sends back, and the
# fields the adapter must then report as never generated.
MISSING_BY_CASE = {
    "missing": ("summary",),
    "unknown": ("summary",),
    "truncated": ("confidence_score", "key_points", "summary"),
}


class Provider:
    """OpenAI-compatible completion server driven by a marker in the prompt."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cases: list[str] = []
        self.barrier: threading.Barrier | None = None
        self.failures: set[str] = set()
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = json.dumps(body["messages"])
                case = next(
                    name
                    for name in ("missing", "unknown", "truncated", "valid")
                    if f"case:{name}" in prompt
                )
                with provider.lock:
                    provider.cases.append(case)
                    barrier = provider.barrier
                    failing = case in provider.failures
                if barrier is not None:
                    barrier.wait(timeout=30)
                if failing:
                    self.respond(
                        503,
                        {
                            "error": {
                                "message": "generation interrupted",
                                "type": "server_error",
                            }
                        },
                    )
                    return
                fields = dict(COMPLETE)
                if case == "missing":
                    del fields["summary"]
                    content = json.dumps(fields)
                elif case == "unknown":
                    fields["unrecognized_field"] = fields.pop("summary")
                    content = json.dumps(fields)
                elif case == "truncated":
                    # A max-token cut: the object stops inside the reasoning
                    # string that ChainOfThought emits first.
                    content = '{"reasoning": "' + fields["reasoning"][:24]
                else:
                    content = json.dumps(fields)
                self.respond(
                    200,
                    {
                        "id": f"chatcmpl-{case}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": (
                                    "length" if case == "truncated" else "stop"
                                ),
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                )

            def respond(self, status, payload):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


@pytest.fixture
def provider():
    server = Provider()
    server.thread.start()
    try:
        yield server
    finally:
        server.server.shutdown()
        server.server.server_close()
        server.thread.join(5)


@pytest.fixture
async def summary_route(provider, tmp_path, monkeypatch):
    config = json.loads(Path("configs/config.json").read_text())
    config["llm_config"] = {
        "primary": {
            "model": MODEL,
            "api_base": provider.url,
            "api_key": "not-required",
            "max_tokens": 256,
            "temperature": 0,
            "num_retries": 0,
            "request_timeout": 30,
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_file))

    manager = ConfigManager(store=InMemoryConfigStore())
    registry = AgentRegistry(tenant_id=TENANTS[0], config_manager=manager)
    registry.register_agent(
        AgentEndpoint(
            name="summarizer_agent",
            url="http://127.0.0.1:8000",
            capabilities=["summarization"],
            health_endpoint="/health",
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    monkeypatch.setattr(agents, "_dispatcher", dispatcher)
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    with dspy.context(adapter=LenientJSONAdapter()):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            yield client


def task(case: str, tenant: str = TENANTS[0]) -> dict:
    return {
        "agent_name": "summarizer_agent",
        "query": f"Summarize the observatory schedule. case:{case}",
        "context": {
            "tenant_id": tenant,
            "request_id": f"request-{case}",
            "search_results": [
                {
                    "id": "grounded-source",
                    "title": "Observatory schedule",
                    "text_content": (
                        "The observatory has two telescopes. "
                        "Observations start at sunset."
                    ),
                    "score": 0.9,
                }
            ],
        },
    }


def failed_turn(case: str) -> dict:
    return {
        "status": "error",
        "agent": "summarizer_agent",
        "error": (
            f"Agent 'summarizer_agent' generation for request 'request-{case}' "
            f"produced no {', '.join(MISSING_BY_CASE[case])}"
        ),
    }


@pytest.mark.parametrize("case", ["missing", "unknown", "truncated"])
async def test_incomplete_generation_is_a_failed_turn(summary_route, provider, case):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task(case)
    )
    assert response.status_code == 200, response.text
    assert response.json() == failed_turn(case)
    assert provider.cases == [case]


async def test_complete_generation_reaches_the_caller_byte_exact(
    summary_route, provider
):
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid")
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["agent"] == "summarizer_agent"
    assert body["result"]["summary"] == COMPLETE["summary"]
    assert body["answer"].startswith(COMPLETE["summary"])
    assert provider.cases == ["valid"]


async def test_concurrent_tenants_get_their_own_outcome(summary_route, provider):
    """Two tenants in flight at once: the incomplete one fails, the complete
    one answers, and neither borrows the other's fields."""
    provider.barrier = threading.Barrier(2)
    try:
        good, bad = await asyncio.gather(
            *[
                summary_route.post(
                    "/agents/summarizer_agent/process", json=task(case, tenant)
                )
                for case, tenant in zip(("valid", "missing"), TENANTS)
            ]
        )
    finally:
        provider.barrier = None
    assert sorted(provider.cases) == ["missing", "valid"]
    assert good.status_code == 200, good.text
    assert good.json()["status"] == "success"
    assert good.json()["result"]["summary"] == COMPLETE["summary"]
    assert bad.status_code == 200, bad.text
    assert bad.json() == failed_turn("missing")


async def test_provider_failure_is_not_answered_with_fabricated_output(
    summary_route, provider
):
    provider.failures = {"valid"}
    response = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid")
    )
    assert provider.cases == ["valid"]
    assert response.status_code == 500, response.text
    assert response.json() == {
        "detail": (
            "Agent 'summarizer_agent' failed with ServiceUnavailableError "
            "(request_id=request-valid). See runtime logs for detail."
        )
    }
    provider.failures = set()
    recovered = await summary_route.post(
        "/agents/summarizer_agent/process", json=task("valid", TENANTS[1])
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["result"]["summary"] == COMPLETE["summary"]
