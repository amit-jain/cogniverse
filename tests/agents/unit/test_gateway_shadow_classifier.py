"""The gateway's shadow query classifier: recorded beside the rules, never routes.

The classifier is a checkpoint trained by the production trainer and served
by the real inference server in a child process, the way production runs it
(the runtime image has no torch). GLiNER is a stand-in returning no entities,
so the rule decision comes from the keyword rules alone.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.gateway_agent import (
    CLASSIFIER_HEADS,
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    ShadowClassification,
)
from cogniverse_core.common.models import query_classifier as loader
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID
from cogniverse_foundation.query_classifier import QueryClassifier
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.utils.query_classifiers import (
    GATEWAY_EXAMPLES,
    query_classifier_server,
    tiny_encoder,
)

pytestmark = [pytest.mark.unit]

FULL = {
    e.text: e.labels for e in GATEWAY_EXAMPLES if set(e.labels) == set(CLASSIFIER_HEADS)
}


@pytest.fixture(autouse=True)
def _fresh_loader_cache():
    loader._classifier_cache.clear()
    yield
    loader._classifier_cache.clear()


def _gateway(**deps) -> tuple[GatewayAgent, RecordingTelemetryManager]:
    agent = GatewayAgent(deps=GatewayDeps(**deps))
    gliner = MagicMock()
    gliner.predict_entities.return_value = []
    agent._gliner_model = gliner
    telemetry = RecordingTelemetryManager()
    agent.telemetry_manager = telemetry
    return agent, telemetry


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


@pytest.mark.asyncio
async def test_shadow_labels_are_recorded_and_routing_is_the_rules(
    query_classifier_service,
):
    url, model = query_classifier_service
    shadowed, telemetry = _gateway(
        query_classifier_model=model,
        query_classifier_inference_url=url,
        query_classifier_timeout_s=30,
    )
    plain, _ = _gateway()

    rows = []
    for text in FULL:
        with_shadow = await _route(shadowed, text)
        without = await _route(plain, text)
        assert _rule(with_shadow) == _rule(without)
        assert without.shadow_classification is None
        shadow = with_shadow.shadow_classification
        rows.append(
            (
                text,
                with_shadow.routed_to,
                shadow.complexity,
                shadow.modality,
                shadow.generation_type,
                shadow.routed_to,
                shadow.agrees,
                shadow.error,
            )
        )
        assert (shadow.complexity, shadow.modality, shadow.generation_type) == (
            FULL[text]["complexity"],
            FULL[text]["modality"],
            FULL[text]["generation_type"],
        )
        assert shadow.model == model

    assert rows == [
        ("show me cat videos", "search_agent", "simple", "video", "raw_results",
         "search_agent", True, None),
        ("find dog photos", "image_search_agent", "simple", "image", "raw_results",
         "image_search_agent", True, None),
        ("summarize the podcast", "orchestrator_agent", "simple", "audio", "summary",
         "summarizer_agent", False, None),
        ("find cooking videos", "search_agent", "complex", "both", "detailed_report",
         "orchestrator_agent", False, None),
        ("compare reports on rockets", "orchestrator_agent", "complex", "document",
         "detailed_report", "orchestrator_agent", True, None),
    ]  # fmt: skip

    spans = _routing_outputs(telemetry)
    assert len(spans) == len(FULL)
    for span, text in zip(spans, FULL):
        recorded = ShadowClassification(**span["shadow_classification"])
        assert recorded.complexity == FULL[text]["complexity"]
        assert set(span["shadow_classification"]) == set(
            ShadowClassification.model_fields
        )


@pytest.mark.asyncio
async def test_no_model_means_no_shadow_and_no_classifier_built():
    agent, telemetry = _gateway()
    output = await _route(agent, "show me cat videos")

    assert output.shadow_classification is None
    assert _routing_outputs(telemetry)[0]["shadow_classification"] is None
    assert loader._classifier_cache == {}


@pytest.mark.asyncio
async def test_a_dead_classifier_is_recorded_and_routing_still_served():
    agent, telemetry = _gateway(
        query_classifier_model="gateway-classifier",
        query_classifier_inference_url="http://127.0.0.1:9",
    )
    plain, _ = _gateway()

    output = await _route(agent, "show me cat videos")

    assert _rule(output) == _rule(await _route(plain, "show me cat videos"))
    shadow = output.shadow_classification
    assert shadow.error.startswith("ConnectionError: ")
    assert "127.0.0.1" in shadow.error and "port=9" in shadow.error
    assert (
        shadow.complexity,
        shadow.modality,
        shadow.generation_type,
        shadow.routed_to,
        shadow.agrees,
    ) == (None, None, None, None, None)
    assert _routing_outputs(telemetry)[0]["shadow_classification"] == (
        shadow.model_dump()
    )


class _Hung(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        time.sleep(3)

    def log_message(self, format: str, *args) -> None:
        return


@contextmanager
def _hung_service():
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _Hung)
    stub.daemon_threads = True
    thread = threading.Thread(target=stub.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{stub.server_port}"
    finally:
        stub.shutdown()
        stub.server_close()


@pytest.mark.asyncio
async def test_a_hung_classifier_costs_at_most_its_timeout():
    with _hung_service() as url:
        agent, _ = _gateway(
            query_classifier_model="gateway-classifier",
            query_classifier_inference_url=url,
            query_classifier_timeout_s=0.3,
        )
        started = time.perf_counter()
        output = await _route(agent, "show me cat videos")
        elapsed = time.perf_counter() - started

    assert output.routed_to == "search_agent"
    assert output.shadow_classification.error.startswith("ReadTimeout: ")
    assert 300 <= output.shadow_classification.latency_ms < 1500
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_a_checkpoint_without_the_gateway_heads_is_an_error(tmp_path):
    encoder, tokenizer = tiny_encoder(["show me cat videos"])
    QueryClassifier(
        encoder, tokenizer, {"complexity": ["simple", "complex"]}, max_length=16
    ).save(tmp_path / "complexity-only")

    with query_classifier_server(tmp_path) as url:
        agent, _ = _gateway(
            query_classifier_model="complexity-only",
            query_classifier_inference_url=url,
            query_classifier_timeout_s=30,
        )
        output = await _route(agent, "show me cat videos")

    assert output.routed_to == "search_agent"
    assert output.shadow_classification.error == (
        "ValueError: Query classifier 'complexity-only' predictions cover heads "
        "['complexity'], expected ['complexity', 'generation_type', 'modality']"
    )


@pytest.mark.asyncio
async def test_concurrent_requests_each_get_their_own_shadow_labels(
    query_classifier_service, monkeypatch
):
    url, model = query_classifier_service
    agent, telemetry = _gateway(
        query_classifier_model=model,
        query_classifier_inference_url=url,
        query_classifier_timeout_s=30,
    )
    builds = []
    real_init = loader.RemoteQueryClassifierClient.__init__

    def counting_init(self, *args, **kwargs):
        time.sleep(0.05)  # hold the build open so racing first touches overlap
        builds.append(args)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(loader.RemoteQueryClassifierClient, "__init__", counting_init)

    texts = list(FULL) * 3
    outputs = await asyncio.gather(*(_route(agent, text) for text in texts))

    assert len(builds) == 1
    assert [
        (
            o.query,
            o.shadow_classification.complexity,
            o.shadow_classification.modality,
            o.shadow_classification.generation_type,
        )
        for o in outputs
    ] == [
        (t, FULL[t]["complexity"], FULL[t]["modality"], FULL[t]["generation_type"])
        for t in texts
    ]
    assert sorted(
        (span["shadow_classification"]["complexity"], span["complexity"])
        for span in _routing_outputs(telemetry)
    ) == sorted((FULL[t]["complexity"], o.complexity) for t, o in zip(texts, outputs))
