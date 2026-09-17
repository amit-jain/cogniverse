"""An output whose blank answer is valid does not fail the generation.

Real FastAPI route / real AgentDispatcher -> real agent -> real DSPy/LiteLLM ->
a real OpenAI-compatible HTTP provider replaying the pro-tier model's recorded
responses: query enhancement leaves ``context`` blank when no contextual
addition applies, and the orchestration planner leaves ``parallel_steps``
blank when every step runs in sequence.
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

MODEL = "openai/blank-valid-outputs"
TENANTS = ("prodfixlmpaths:blanka", "prodfixlmpaths:blankb")

# Recorded from google/gemma-4-e4b-it serving a pro-tier tenant.
ENHANCEMENT = {
    "reasoning": (
        "The user is asking for basic information about Kubernetes networking. "
        "Since no source text or grounding context is provided, the query will "
        "be enhanced by keeping it broad but adding related fundamental concepts "
        "to improve search recall."
    ),
    "enhanced_query": "Kubernetes networking basics tutorial",
    "expansion_terms": "CNI, Pod networking, Service discovery, kube-proxy",
    "synonyms": "K8s networking, container orchestration networking",
    "context": "",
    "confidence": "0.8",
}
PLAN = {
    "reasoning": (
        "The user wants to find 'security incident briefings' and then "
        "'summarize the remediation steps'. The search and summarization are "
        "sequential (search first, then summarize), so they cannot run in "
        "parallel."
    ),
    "agent_sequence": "query_enhancement_agent,search_agent,summarizer_agent",
    "parallel_steps": "",
}


class Provider:
    """OpenAI-compatible server answering each signature with a recorded response
    edited by the ``case:`` marker in the prompt."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cases: list[str] = []
        self.barrier: threading.Barrier | None = None
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = json.dumps(body["messages"])
                case = next(
                    name
                    for name in ("blank_context", "blank_enhanced_query", "plan")
                    if f"case:{name}" in prompt
                )
                with provider.lock:
                    provider.cases.append(case)
                    barrier = provider.barrier
                if barrier is not None:
                    barrier.wait(timeout=30)
                if case == "plan":
                    fields = dict(PLAN)
                else:
                    fields = dict(ENHANCEMENT)
                    if case == "blank_enhanced_query":
                        fields["enhanced_query"] = ""
                encoded = json.dumps(
                    {
                        "id": f"chatcmpl-{case}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(fields),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                ).encode()
                self.send_response(200)
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
def telemetry_off():
    from cogniverse_foundation.telemetry import manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import (
        TelemetryConfig,
        TelemetryManager,
    )

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None


@pytest.fixture
def dispatcher(provider, tmp_path, monkeypatch, telemetry_off):
    config = json.loads(Path("configs/config.json").read_text())
    config.pop("active_video_profile", None)
    config["llm_config"] = {
        "primary": {
            "model": MODEL,
            "api_base": provider.url,
            "api_key": "not-required",
            "max_tokens": 512,
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
    for name, capability in (
        ("query_enhancement_agent", "query_enhancement"),
        ("search_agent", "search"),
        ("summarizer_agent", "summarization"),
    ):
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://127.0.0.1:8000",
                capabilities=[capability],
                health_endpoint="/health",
                process_endpoint=f"/agents/{name}/process",
            )
        )
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )


@pytest.fixture
def served_lm(provider):
    """The ambient LM and adapter the runtime lifespan binds, on the provider."""
    lm = dspy.LM(
        MODEL,
        api_base=provider.url,
        api_key="not-required",
        max_tokens=512,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
        yield


@pytest.fixture
async def route(dispatcher, served_lm, monkeypatch):
    monkeypatch.setattr(agents, "_dispatcher", dispatcher)
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    ) as client:
        yield client


QUERY = "kubernetes networking basics"


def enhancement_task(case: str, tenant: str = TENANTS[0]) -> dict:
    return {
        "agent_name": "query_enhancement_agent",
        "query": f"{QUERY} case:{case}",
        "context": {"tenant_id": tenant, "request_id": f"request-{case}"},
    }


def lm_enhancement(case: str) -> dict:
    query = f"{QUERY} case:{case}"
    enhanced = ENHANCEMENT["enhanced_query"]
    return {
        "status": "success",
        "agent": "query_enhancement_agent",
        "original_query": query,
        "enhanced_query": enhanced,
        "expansion_terms": ["CNI", "Pod networking", "Service discovery", "kube-proxy"],
        "synonyms": ["K8s networking", "container orchestration networking"],
        "context_additions": [],
        "query_variants": [
            enhanced,
            f"{query} CNI Pod networking Service discovery",
        ],
        "confidence": 0.8,
        "reasoning": ENHANCEMENT["reasoning"],
        "path_used": "lm",
    }


def fallback_enhancement(case: str) -> dict:
    query = f"{QUERY} case:{case}"
    return {
        "status": "success",
        "agent": "query_enhancement_agent",
        "original_query": query,
        "enhanced_query": f"{query} related content",
        "expansion_terms": [],
        "synonyms": [],
        "context_additions": [],
        "query_variants": [f"{query} related content"],
        "confidence": 0.5,
        "reasoning": "Fallback enhancement with heuristic expansion",
        "path_used": "heuristic_fallback",
    }


def without_answer(body: dict) -> dict:
    return {key: value for key, value in body.items() if key != "answer"}


async def test_blank_context_is_an_lm_enhancement(route, provider):
    response = await route.post(
        "/agents/query_enhancement_agent/process",
        json=enhancement_task("blank_context"),
    )
    assert response.status_code == 200, response.text
    assert without_answer(response.json()) == lm_enhancement("blank_context")
    assert provider.cases == ["blank_context"]


async def test_blank_required_output_still_falls_back(route, provider):
    response = await route.post(
        "/agents/query_enhancement_agent/process",
        json=enhancement_task("blank_enhanced_query"),
    )
    assert response.status_code == 200, response.text
    assert without_answer(response.json()) == fallback_enhancement(
        "blank_enhanced_query"
    )
    assert provider.cases == ["blank_enhanced_query"]


async def test_concurrent_tenants_keep_their_own_enhancement_path(route, provider):
    provider.barrier = threading.Barrier(2)
    try:
        lm, fallback = await asyncio.gather(
            *[
                route.post(
                    "/agents/query_enhancement_agent/process",
                    json=enhancement_task(case, tenant),
                )
                for case, tenant in zip(
                    ("blank_context", "blank_enhanced_query"), TENANTS
                )
            ]
        )
    finally:
        provider.barrier = None
    assert sorted(provider.cases) == ["blank_context", "blank_enhanced_query"]
    assert lm.status_code == 200, lm.text
    assert without_answer(lm.json()) == lm_enhancement("blank_context")
    assert fallback.status_code == 200, fallback.text
    assert without_answer(fallback.json()) == fallback_enhancement(
        "blank_enhanced_query"
    )


async def test_blank_parallel_steps_plan_runs_every_step_in_sequence(
    dispatcher, served_lm, provider
):
    orchestrator = await dispatcher._get_or_build_orchestrator(TENANTS[0])
    plan = await orchestrator._create_plan(
        "Look for security incident briefings and summarize the remediation "
        "steps case:plan"
    )
    assert provider.cases == ["plan"]
    assert [
        (step.agent_name, step.depends_on, step.reasoning) for step in plan.steps
    ] == [
        ("query_enhancement_agent", [], "Step 1: query_enhancement_agent processing"),
        ("search_agent", [0], "Step 2: search_agent processing"),
        ("summarizer_agent", [1], "Step 3: summarizer_agent processing"),
    ]
    assert plan.parallel_groups == []
    assert plan.reasoning == PLAN["reasoning"]
