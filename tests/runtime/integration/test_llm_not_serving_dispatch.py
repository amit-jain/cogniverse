"""A complex query against an undeployed chat LLM gets a 503 that names it.

The whole request path is real: the agents route, the dispatcher, the
orchestrator and its planner, the routed LM, and the shipped Envoy -> vLLM
Semantic Router -> stub upstream stack, whose student is put into the state
Modal answers for an undeployed app.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_foundation.config.lm_endpoint_availability import (
    NOT_SERVING_RECHECK_S,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SemanticRouterConfig
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents, health
from tests.foundation.integration._sr_stack.stub_upstream import STATE_DIR
from tests.foundation.integration.conftest import (
    semantic_router_stack as semantic_router_stack,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = pytest.mark.integration

TENANT = "llmdown:production"
CLASSIFICATION = SemanticRouterConfig().classification_model
PLAN_AGENTS = ("search_agent", "summarizer_agent")
COMPLEX_QUERY = "find machine learning videos and summarize them"
RETRY_AFTER = str(int(NOT_SERVING_RECHECK_S))


def _stub_exec(container: str, command: str) -> str:
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout


def _student_requests(stack, prompt_fragment: str) -> int:
    records = json.loads(
        _stub_exec(
            stack["stub_container"],
            'python -c "import urllib.request; print(urllib.request.urlopen('
            "'http://127.0.0.1:8000/requests').read().decode())\"",
        )
    )["requests"]
    return sum(1 for record in records if prompt_fragment in record["prompt"])


@pytest.fixture
def undeployed_student(semantic_router_stack):
    container = semantic_router_stack["stub_container"]
    _stub_exec(container, f"mkdir -p {STATE_DIR} && touch {STATE_DIR}/undeployed")
    yield semantic_router_stack
    _stub_exec(container, f"rm -rf {STATE_DIR}")


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
def app(undeployed_student, telemetry_off):
    """The agents and health routes over a dispatcher whose LM traffic is
    routed through the stack, the way the chart deploys the runtime."""
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        replace(
            manager.get_system_config(),
            semantic_router=SemanticRouterConfig(
                enabled=True,
                semantic_router_url=undeployed_student["base_url"],
            ),
        )
    )
    registry = AgentRegistry(tenant_id=TENANT, config_manager=manager)
    for name, capability in (
        ("orchestrator_agent", "orchestration"),
        *((name, name.removesuffix("_agent")) for name in PLAN_AGENTS),
    ):
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://127.0.0.1:9",
                capabilities=[capability],
                health_endpoint="/health",
                process_endpoint=f"/agents/{name}/process",
            )
        )
    agents._dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=manager, schema_loader=None
    )
    application = FastAPI()
    application.include_router(health.router)
    application.include_router(agents.router, prefix="/agents")
    yield application
    agents._dispatcher = None


def _task(query: str) -> dict:
    return {
        "agent_name": "orchestrator_agent",
        "query": query,
        "context": {"tenant_id": TENANT},
    }


def _expected_detail(request_id: str, *, retry_after_s: int) -> dict:
    return {
        "error": "llm_unavailable",
        "dependency": "llm",
        "agent": "orchestrator_agent",
        "failure": "UpstreamNotServing",
        "upstream_status": 404,
        "model": CLASSIFICATION,
        "retry_after_s": retry_after_s,
        "request_id": request_id,
        "message": (
            "Agent 'orchestrator_agent' could not complete: the chat LLM is not "
            "serving: nothing is deployed for the model (UpstreamNotServing, "
            f"upstream HTTP 404). Retry after {retry_after_s}s."
        ),
    }


def test_a_complex_query_gets_a_503_naming_the_llm_and_the_next_fails_fast(
    app, undeployed_student
):
    first_query = f"{COMPLEX_QUERY} {uuid.uuid4()}"
    second_query = f"{COMPLEX_QUERY} {uuid.uuid4()}"
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/agents/orchestrator_agent/process", json=_task(first_query)
        )
        started = time.perf_counter()
        second = client.post(
            "/agents/orchestrator_agent/process", json=_task(second_query)
        )
        second_s = time.perf_counter() - started
        health_body = client.get("/health").json()

    assert first.status_code == 503, first.text
    assert first.headers["retry-after"] == RETRY_AFTER
    first_detail = first.json()["detail"]
    assert first_detail == _expected_detail(
        first_detail["request_id"], retry_after_s=int(RETRY_AFTER)
    )
    assert _student_requests(undeployed_student, first_query) == 1

    assert second.status_code == 503, second.text
    second_detail = second.json()["detail"]
    retry_after_s = int(second.headers["retry-after"])
    assert retry_after_s <= int(RETRY_AFTER)
    assert second_detail == _expected_detail(
        second_detail["request_id"], retry_after_s=retry_after_s
    )
    assert _student_requests(undeployed_student, second_query) == 0
    assert second_s < 2.0, f"a refused planner call took {second_s:.2f}s"

    llm = health_body["dependencies"]["llm"]
    assert llm["status"] == "not_serving"
    assert [
        (e["endpoint"], e["model"], e["route"], e["state"], e["upstream_status"])
        for e in llm["endpoints"]
    ] == [
        (undeployed_student["base_url"], CLASSIFICATION, "default", "not_serving", 404)
    ]


async def test_concurrent_complex_queries_behind_the_404_all_fail_fast(
    app, undeployed_student
):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://runtime"
    ) as client:
        first = await client.post(
            "/agents/orchestrator_agent/process",
            json=_task(f"{COMPLEX_QUERY} {uuid.uuid4()}"),
        )
        queries = [f"{COMPLEX_QUERY} concurrent {uuid.uuid4()}" for _ in range(6)]
        responses = await asyncio.gather(
            *(
                client.post("/agents/orchestrator_agent/process", json=_task(query))
                for query in queries
            )
        )

    assert first.status_code == 503
    assert [(r.status_code, r.json()["detail"]["failure"]) for r in responses] == [
        (503, "UpstreamNotServing")
    ] * len(queries)
    assert len({r.json()["detail"]["request_id"] for r in responses}) == len(queries)
    assert sum(_student_requests(undeployed_student, q) for q in queries) == 0
