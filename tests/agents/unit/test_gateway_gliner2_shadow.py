"""The gateway's GLiNER2 shadow: recorded beside the rules, never routes.

GLiNER2 is the real inference server over HTTP, the way production runs it
(the runtime image has no torch), answering with a stand-in model
(``tests/utils/gliner2_models.py``). GLiNER is a stand-in returning no
entities, so the rule decision comes from the keyword rules alone.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import get_args
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.gateway_agent import (
    SHADOW_DECISIONS,
    SHADOW_ENTITY_TYPES,
    SIMPLE_ROUTE_MAP,
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    GatewayOutput,
    ShadowClassification,
    classifier_route,
)
from cogniverse_core.common.models import gliner2 as loader
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID
from cogniverse_foundation.common.entity_types import ENTITY_TYPES
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.utils.gliner2_models import QUERIES, SERVED_ANSWERS

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _fresh_client_cache():
    loader._client_cache.clear()
    yield
    loader._client_cache.clear()


@pytest.fixture(scope="module")
def computed(gliner2_service):
    """The server's answer for each query."""
    _, model = gliner2_service
    return {query: {"model": model, **SERVED_ANSWERS[query]} for query in QUERIES}


def _gateway(**deps) -> tuple[GatewayAgent, RecordingTelemetryManager]:
    agent = GatewayAgent(deps=GatewayDeps(**deps))
    gliner = MagicMock()
    gliner.predict_entities.return_value = []
    agent._gliner_model = gliner
    telemetry = RecordingTelemetryManager()
    agent.telemetry_manager = telemetry
    return agent, telemetry


def _shadowed(gliner2_service, **extra):
    url, model = gliner2_service
    return _gateway(
        gliner2_model=model, gliner2_inference_url=url, gliner2_timeout_s=30, **extra
    )


def _routing_outputs(telemetry: RecordingTelemetryManager) -> list[dict]:
    return [
        json.loads(span.attributes["output.value"])
        for call, span in zip(telemetry.calls, telemetry.spans)
        if call["name"] == "cogniverse.routing"
    ]


async def _route(agent: GatewayAgent, query: str):
    return await agent._process_impl(
        GatewayInput(query=query, tenant_id=TEST_TENANT_ID)
    )


RULE_FIELDS = ("complexity", "modality", "generation_type", "routed_to", "confidence")


def _rule(output) -> tuple:
    return tuple(getattr(output, field) for field in RULE_FIELDS)


def _expected_shadow(answer: dict, rule_routed_to: str) -> dict:
    """The shadow fields (minus latency) the gateway owes for ``answer``."""
    decisions = answer["classifications"]
    routed_to = classifier_route(
        decisions["complexity"]["label"],
        decisions["modality"]["label"],
        decisions["generation_type"]["label"],
    )
    return {
        "model": answer["model"],
        "complexity": decisions["complexity"]["label"],
        "complexity_score": decisions["complexity"]["score"],
        "modality": decisions["modality"]["label"],
        "modality_score": decisions["modality"]["score"],
        "generation_type": decisions["generation_type"]["label"],
        "generation_type_score": decisions["generation_type"]["score"],
        "entities": answer["entities"],
        "routed_to": routed_to,
        "agrees": routed_to == rule_routed_to,
        "error": None,
    }


def _without_latency(shadow: dict) -> dict:
    return {k: v for k, v in shadow.items() if k != "latency_ms"}


def test_shadow_tasks_are_the_gateway_decision_fields_and_entity_vocabulary():
    assert {task: list(labels) for task, labels in SHADOW_DECISIONS.items()} == {
        field: list(get_args(GatewayOutput.model_fields[field].annotation))
        for field in ("complexity", "modality", "generation_type")
    }
    assert set(SHADOW_ENTITY_TYPES) == set(ENTITY_TYPES)


def test_classifier_route_sends_complex_to_the_orchestrator_and_simple_by_map():
    assert classifier_route("complex", "video", "raw_results") == "orchestrator_agent"
    assert {
        key: classifier_route("simple", *key) for key in SIMPLE_ROUTE_MAP
    } == SIMPLE_ROUTE_MAP
    assert classifier_route("simple", "both", "raw_results") == "orchestrator_agent"


@pytest.mark.asyncio
async def test_shadow_records_what_gliner2_computes_and_routing_is_the_rules(
    gliner2_service, computed
):
    shadowed, telemetry = _shadowed(gliner2_service)
    plain, _ = _gateway()

    expected = []
    for query in QUERIES:
        with_shadow = await _route(shadowed, query)
        without = await _route(plain, query)
        assert _rule(with_shadow) == _rule(without)
        assert without.shadow_classification is None
        want = _expected_shadow(computed[query], without.routed_to)
        assert _without_latency(with_shadow.shadow_classification.model_dump()) == (
            want
        )
        expected.append(want)

    spans = _routing_outputs(telemetry)
    assert [_without_latency(s["shadow_classification"]) for s in spans] == expected
    assert [(s["routed_to"], s["agrees"]) for s in expected] == [
        ("search_agent", True),
        ("search_agent", True),
        ("orchestrator_agent", True),
        ("summarizer_agent", False),
    ]


@pytest.mark.asyncio
async def test_no_model_means_no_shadow_and_no_client_built():
    agent, telemetry = _gateway()
    output = await _route(agent, "show me cat videos")

    assert output.shadow_classification is None
    assert _routing_outputs(telemetry)[0]["shadow_classification"] is None
    assert loader._client_cache == {}


@pytest.mark.asyncio
async def test_a_dead_service_is_recorded_and_routing_still_served():
    agent, telemetry = _gateway(
        gliner2_model="fastino/GLiNER2.5-Decide",
        gliner2_inference_url="http://127.0.0.1:9",
    )
    plain, _ = _gateway()

    output = await _route(agent, "show me cat videos")

    assert _rule(output) == _rule(await _route(plain, "show me cat videos"))
    shadow = output.shadow_classification
    assert shadow.error.startswith("ConnectionError: ")
    assert "127.0.0.1" in shadow.error and "port=9" in shadow.error
    assert _without_latency(shadow.model_dump()) == {
        "model": "fastino/GLiNER2.5-Decide",
        "complexity": None,
        "complexity_score": None,
        "modality": None,
        "modality_score": None,
        "generation_type": None,
        "generation_type_score": None,
        "entities": [],
        "routed_to": None,
        "agrees": None,
        "error": shadow.error,
    }
    assert _routing_outputs(telemetry)[0]["shadow_classification"] == (
        shadow.model_dump()
    )


@pytest.mark.asyncio
async def test_a_model_without_an_inference_url_is_an_error():
    agent, _ = _gateway(gliner2_model="fastino/GLiNER2.5-Decide")
    output = await _route(agent, "show me cat videos")

    assert output.routed_to == "search_agent"
    assert output.shadow_classification.error == (
        "ValueError: inference_url must name the GLiNER2 inference service"
    )


class _Stub(BaseHTTPRequestHandler):
    body: bytes = b""
    delay_s: float = 0.0

    def do_POST(self) -> None:
        time.sleep(self.delay_s)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args) -> None:
        return


@contextmanager
def _stub_service(body: dict, delay_s: float = 0.0):
    handler = type(
        "Handler", (_Stub,), {"body": json.dumps(body).encode(), "delay_s": delay_s}
    )
    stub = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    stub.daemon_threads = True
    thread = threading.Thread(target=stub.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{stub.server_port}"
    finally:
        stub.shutdown()
        stub.server_close()


@pytest.mark.asyncio
async def test_a_hung_service_costs_at_most_its_timeout():
    with _stub_service({}, delay_s=3) as url:
        agent, _ = _gateway(
            gliner2_model="m", gliner2_inference_url=url, gliner2_timeout_s=0.3
        )
        started = time.perf_counter()
        output = await _route(agent, "show me cat videos")
        elapsed = time.perf_counter() - started

    assert output.routed_to == "search_agent"
    assert output.shadow_classification.error.startswith("ReadTimeout: ")
    assert 300 <= output.shadow_classification.latency_ms < 1500
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_an_entity_the_query_does_not_contain_is_an_error():
    body = {
        "model": "m",
        "entities": [
            {"text": "Paris", "label": "PLACE", "score": 0.9, "start": 0, "end": 5}
        ],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.9},
            "modality": {"label": "video", "score": 0.9},
            "generation_type": {"label": "raw_results", "score": 0.9},
        },
    }
    with _stub_service(body) as url:
        agent, _ = _gateway(gliner2_model="m", gliner2_inference_url=url)
        output = await _route(agent, "show me cat videos")

    assert output.routed_to == "search_agent"
    assert output.shadow_classification.error == (
        f"ValueError: GLiNER2 response from {url}/extract entity 0 text 'Paris' "
        "is not the query at [0, 5)"
    )
    assert output.shadow_classification.entities == []


@pytest.mark.asyncio
async def test_concurrent_requests_each_get_their_own_shadow(
    gliner2_service, computed, monkeypatch
):
    agent, telemetry = _shadowed(gliner2_service)
    builds = []
    real_init = loader.RemoteGliner2Client.__init__

    def counting_init(self, *args, **kwargs):
        time.sleep(0.05)  # hold the build open so racing first touches overlap
        builds.append(args)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(loader.RemoteGliner2Client, "__init__", counting_init)

    queries = list(QUERIES) * 3
    outputs = await asyncio.gather(*(_route(agent, q) for q in queries))

    assert len(builds) == 1
    assert [
        (o.query, _without_latency(o.shadow_classification.model_dump()))
        for o in outputs
    ] == [
        (q, _expected_shadow(computed[q], o.routed_to))
        for q, o in zip(queries, outputs)
    ]
    assert sorted(
        json.dumps(_without_latency(s["shadow_classification"]), sort_keys=True)
        for s in _routing_outputs(telemetry)
    ) == sorted(
        json.dumps(_expected_shadow(computed[q], o.routed_to), sort_keys=True)
        for q, o in zip(queries, outputs)
    )


def test_shadow_classification_fields_are_the_recorded_span_keys():
    assert set(ShadowClassification.model_fields) == {
        "model",
        "complexity",
        "complexity_score",
        "modality",
        "modality_score",
        "generation_type",
        "generation_type_score",
        "entities",
        "routed_to",
        "agrees",
        "latency_ms",
        "error",
    }
