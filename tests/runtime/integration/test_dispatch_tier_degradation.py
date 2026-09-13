"""A teacher outage reaches the dispatch envelope and its tenant's LM span."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict

import pytest

from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummaryResult,
    ThinkingPhase,
)
from cogniverse_foundation.config.routed_lm import UpstreamAuthRejected
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.telemetry.span_contract import LLM_SERVED_MODEL_ATTRIBUTE
from cogniverse_foundation.telemetry.tenant_context import current_tenant_id
from tests.foundation.integration.conftest import (
    semantic_router_stack as semantic_router_stack,
)
from tests.runtime.integration.test_dispatch_binds_the_tenant_for_telemetry import (
    TENANT_A,
    TENANT_B,
    _task,
)
from tests.runtime.integration.test_dispatch_binds_the_tenant_for_telemetry import (
    dispatcher as dispatcher,
)
from tests.runtime.integration.test_dispatch_binds_the_tenant_for_telemetry import (
    route_client as route_client,
)
from tests.runtime.integration.test_dispatch_binds_the_tenant_for_telemetry import (
    tenant_exporters as tenant_exporters,
)

pytestmark = pytest.mark.integration

DEGRADATION = {
    "tier_degraded": "pro_model_unavailable",
    "upstream_status": 503,
    "upstream_exception_type": "UpstreamUnavailable",
}


def _summary(text):
    return SummaryResult(
        summary=text,
        key_points=[],
        visual_insights=[],
        confidence_score=1.0,
        thinking_phase=ThinkingPhase([], [], {}, [], "served"),
        metadata={},
    )


@pytest.fixture
def routed_summarizer(monkeypatch, semantic_router_stack, tenant_exporters):
    provider, _ = tenant_exporters
    tracer = provider.get_tracer("dspy")
    barrier = []
    lms = {
        tenant: create_routed_lm(
            LLMEndpointConfig(model="unused", num_retries=2, request_timeout=15.0),
            SemanticRouterConfig(
                enabled=True, semantic_router_url=semantic_router_stack["base_url"]
            ),
            tenant,
            tier,
            call_site="summarizer_agent",
        )
        for tenant, tier in ((TENANT_A, "pro"), (TENANT_B, "default"))
    }
    for lm in lms.values():
        lm.cache = False

    async def summarize(self, request):
        lm = lms[current_tenant_id()]

        def complete():
            with tracer.start_as_current_span(request.query):
                if barrier:
                    barrier[0].wait(timeout=30)
                response = lm.forward(request.query)
                reflection = json.loads(response.choices[0].message.content)
                return f"{reflection['backend_tag']}:{reflection['served_model']}"

        return _summary(await asyncio.to_thread(complete))

    monkeypatch.setattr(SummarizerAgent, "summarize", summarize)
    return barrier


def _expected(prompt, degraded):
    return {
        "status": "success",
        "agent": "summarizer_agent",
        "message": f"Generated summary for '{prompt}'",
        "grounding": {
            "state": "threaded_results",
            "modalities": [],
            "profiles": [],
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [],
            "result_count": 1,
        },
        "result": asdict(_summary("student:basic-chat")),
        "answer": "student:basic-chat",
        **(DEGRADATION if degraded else {}),
    }


def test_the_http_envelope_names_the_teacher_outage(
    route_client, routed_summarizer, tenant_exporters
):
    prompt = f"TEACHER_FAULT:status:503|{uuid.uuid4()}"
    response = route_client.post(
        "/agents/summarizer_agent/process", json=_task(TENANT_A, prompt)
    )
    assert response.status_code == 200, response.text
    assert response.json() == _expected(prompt, True)
    _, exporters = tenant_exporters
    assert {
        tenant: [(s.name, dict(s.attributes)) for s in exporter.get_finished_spans()]
        for tenant, exporter in exporters.items()
    } == {
        TENANT_A: [(prompt, {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat", **DEGRADATION})]
    }


async def test_concurrent_dispatches_do_not_share_degradation(
    dispatcher, routed_summarizer, tenant_exporters
):
    routed_summarizer.append(threading.Barrier(2))
    prompts = [f"TEACHER_FAULT:status:503|{uuid.uuid4()}" for _ in range(2)]
    results = await asyncio.gather(
        *[
            dispatcher.dispatch(
                "summarizer_agent", prompt, _task(tenant, prompt)["context"]
            )
            for tenant, prompt in zip((TENANT_A, TENANT_B), prompts)
        ]
    )
    assert results == [_expected(prompts[0], True), _expected(prompts[1], False)]
    _, exporters = tenant_exporters
    assert {
        tenant: [(s.name, dict(s.attributes)) for s in exporter.get_finished_spans()]
        for tenant, exporter in exporters.items()
    } == {
        TENANT_A: [
            (prompts[0], {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat", **DEGRADATION})
        ],
        TENANT_B: [(prompts[1], {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat"})],
    }
    assert current_tenant_id() is None


async def test_a_permanent_failure_does_not_mark_the_next_dispatch(
    dispatcher, routed_summarizer, tenant_exporters
):
    prompt = f"TEACHER_FAULT:misleading:401|{uuid.uuid4()}"
    with pytest.raises(UpstreamAuthRejected) as raised:
        await dispatcher.dispatch(
            "summarizer_agent", prompt, _task(TENANT_A, prompt)["context"]
        )
    assert raised.value.status == 401
    answered = f"default {uuid.uuid4()}"
    result = await dispatcher.dispatch(
        "summarizer_agent", answered, _task(TENANT_B, answered)["context"]
    )
    assert result == _expected(answered, False)
    _, exporters = tenant_exporters
    assert {
        tenant: [(s.name, dict(s.attributes)) for s in exporter.get_finished_spans()]
        for tenant, exporter in exporters.items()
    } == {
        TENANT_A: [(prompt, {})],
        TENANT_B: [(answered, {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat"})],
    }
