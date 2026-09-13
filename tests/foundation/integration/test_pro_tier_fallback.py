"""A failed teacher gets one named student attempt through the real router."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cogniverse_foundation.config.routed_lm import (
    RouterDecodeFailed,
    UpstreamAuthRejected,
)
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.telemetry.span_contract import LLM_SERVED_MODEL_ATTRIBUTE

pytestmark = pytest.mark.integration

DEGRADATION_KEYS = {"tier_degraded", "upstream_status", "upstream_exception_type"}


def _lm(stack, tier="pro", timeout=10.0, call_site="summarizer_agent"):
    lm = create_routed_lm(
        LLMEndpointConfig(
            model="unused",
            api_base="http://unused:1/v1",
            num_retries=2,
            request_timeout=timeout,
        ),
        SemanticRouterConfig(enabled=True, semantic_router_url=stack["base_url"]),
        tenant_id="fallback:production",
        tier=tier,
        call_site=call_site,
    )
    lm.cache = False
    return lm


async def _invoke(lm, prompt, mode):
    if mode == "sync":
        return await asyncio.to_thread(lm.forward, prompt)
    return await lm.aforward(prompt)


def _tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("pro-fallback")


def _counts(stack, prompts):
    counts = {}
    for backend, container in (
        ("student", "stub_container"),
        ("teacher", "teacher_container"),
    ):
        result = subprocess.run(
            [
                "docker",
                "exec",
                stack[container],
                "python",
                "-c",
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/requests').read().decode())",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        records = json.loads(result.stdout)["requests"]
        counts[backend] = Counter(
            r["prompt"] for r in records if r["prompt"] in prompts
        )
    return counts


def _degradation(status):
    return {
        "tier_degraded": "pro_model_unavailable",
        "upstream_status": status,
        "upstream_exception_type": "UpstreamUnavailable",
    }


def _assert_student(response, prompt, status):
    body = response.model_dump()
    assert {
        key: value for key, value in body.items() if key in DEGRADATION_KEYS
    } == _degradation(status)
    assert body["model"] == "basic-chat"
    reflection = json.loads(body["choices"][0]["message"]["content"])
    assert {
        key: reflection[key] for key in ("backend_tag", "served_model", "echo")
    } == {
        "backend_tag": "student",
        "served_model": "basic-chat",
        "echo": prompt,
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "fault,status",
    [
        ("status:503", 503),
        ("status:502", 502),
        ("status:504", 504),
        ("trickle:5", 408),
        ("reset", 503),
    ],
)
@pytest.mark.asyncio(loop_scope="module")
async def test_a_transient_teacher_failure_gets_one_named_student_attempt(
    semantic_router_stack, mode, fault, status
):
    stack = semantic_router_stack
    prompt = f"TEACHER_FAULT:{fault}|{uuid.uuid4()}"
    lm = _lm(stack, timeout=2.0 if fault == "trickle:5" else 10.0)
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        response = await _invoke(lm, prompt, mode)
    _assert_student(response, prompt, status)
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == {
        LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat",
        **_degradation(status),
    }
    assert _counts(stack, {prompt}) == {"teacher": {prompt: 1}, "student": {prompt: 1}}


@pytest.mark.parametrize(
    "status,kind",
    [
        (400, RouterDecodeFailed),
        (401, UpstreamAuthRejected),
        (403, UpstreamAuthRejected),
        (422, RouterDecodeFailed),
    ],
)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio(loop_scope="module")
async def test_a_permanent_teacher_refusal_never_reaches_the_student(
    semantic_router_stack, status, kind, mode
):
    stack = semantic_router_stack
    prompt = f"TEACHER_FAULT:status:{status}|{uuid.uuid4()}"
    lm = _lm(stack)
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        with pytest.raises(kind) as raised:
            await _invoke(lm, prompt, mode)
    assert type(raised.value) is kind
    assert raised.value.status == status
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == {}
    assert _counts(stack, {prompt}) == {"teacher": {prompt: 1}, "student": {}}


def test_transient_words_in_a_credential_error_do_not_change_its_type(
    semantic_router_stack,
):
    stack = semantic_router_stack
    prompt = f"TEACHER_FAULT:misleading:401|{uuid.uuid4()}"
    with pytest.raises(UpstreamAuthRejected) as raised:
        _lm(stack).forward(prompt)
    assert raised.value.status == 401
    assert type(raised.value) is UpstreamAuthRejected
    assert _counts(stack, {prompt}) == {"teacher": {prompt: 1}, "student": {}}


@pytest.mark.parametrize(
    "tier,call_site",
    [("default", "summarizer_agent"), ("pro", "entity_extraction_agent")],
)
def test_a_student_call_carries_no_degradation(semantic_router_stack, tier, call_site):
    stack = semantic_router_stack
    prompt = f"TEACHER_FAULT:status:503|{uuid.uuid4()}"
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        response = _lm(stack, tier=tier, call_site=call_site).forward(prompt)
    assert {
        key: value
        for key, value in response.model_dump().items()
        if key in DEGRADATION_KEYS
    } == {}
    assert response.model == "basic-chat"
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat"}
    assert _counts(stack, {prompt}) == {"teacher": {}, "student": {prompt: 1}}


def test_sixteen_concurrent_pro_calls_retry_once_without_marking_a_default_call(
    semantic_router_stack,
):
    stack = semantic_router_stack
    lm = _lm(stack)
    default_lm = _lm(stack, tier="default")
    barrier = threading.Barrier(17)
    exporter, tracer = _tracing()
    prompts = [f"TEACHER_FAULT:status:503|{uuid.uuid4()}" for _ in range(17)]

    def call(index):
        with tracer.start_as_current_span(str(index)):
            barrier.wait(timeout=30)
            response = (lm if index < 16 else default_lm).forward(prompts[index])
        if index < 16:
            _assert_student(response, prompts[index], 503)
        else:
            assert {
                key: value
                for key, value in response.model_dump().items()
                if key in DEGRADATION_KEYS
            } == {}
        return response.model

    with ThreadPoolExecutor(max_workers=17) as pool:
        assert list(pool.map(call, range(17))) == ["basic-chat"] * 17
    assert {
        span.name: dict(span.attributes) for span in exporter.get_finished_spans()
    } == {
        str(i): {
            LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat",
            **(_degradation(503) if i < 16 else {}),
        }
        for i in range(17)
    }
    assert _counts(stack, set(prompts)) == {
        "teacher": {prompt: 1 for prompt in prompts[:16]},
        "student": {prompt: 1 for prompt in prompts},
    }


@pytest.mark.parametrize("semantic_router_stack", [{"teacher_port": 1}], indirect=True)
def test_a_refused_teacher_connection_gets_one_student_attempt(semantic_router_stack):
    stack = semantic_router_stack
    prompt = f"teacher connection {uuid.uuid4()}"
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        response = _lm(stack).forward(prompt)
    _assert_student(response, prompt, 503)
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == {
        LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat",
        **_degradation(503),
    }
    assert _counts(stack, {prompt}) == {"teacher": {}, "student": {prompt: 1}}


def test_a_failed_student_attempt_propagates_without_another_retry(
    semantic_router_stack,
):
    from cogniverse_foundation.config.routed_lm import UpstreamUnavailable

    stack = semantic_router_stack
    prompt = "FAULT:status:503"
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        with pytest.raises(UpstreamUnavailable) as raised:
            _lm(stack).forward(prompt)
    assert raised.value.status == 503
    assert raised.value.routed_model == "openai/cogniverse-classification"
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == _degradation(503)
    assert _counts(stack, {prompt}) == {"teacher": {prompt: 1}, "student": {prompt: 1}}


def test_a_default_tier_outage_keeps_its_retry_policy_without_degradation(
    semantic_router_stack,
):
    from cogniverse_foundation.config.routed_lm import UpstreamUnavailable

    stack = semantic_router_stack
    prompt = "FAULT:status:502"
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("completion"):
        with pytest.raises(UpstreamUnavailable) as raised:
            _lm(stack, tier="default").forward(prompt)
    assert raised.value.status == 502
    assert raised.value.routed_model == "openai/auto"
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == {}
    assert _counts(stack, {prompt}) == {"teacher": {}, "student": {prompt: 3}}


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_degradation_does_not_mutate_the_cached_student_completion(
    semantic_router_stack, mode
):
    stack = semantic_router_stack
    prompt = f"TEACHER_FAULT:status:503|{uuid.uuid4()}"
    pro = _lm(stack)
    bounded = _lm(stack, call_site="entity_extraction_agent")
    pro.cache = True
    bounded.cache = True
    exporter, tracer = _tracing()
    with tracer.start_as_current_span("fallback"):
        degraded = await _invoke(pro, prompt, mode)
    _assert_student(degraded, prompt, 503)
    with tracer.start_as_current_span("bounded"):
        cached = await _invoke(bounded, prompt, mode)
    assert {
        key: value
        for key, value in cached.model_dump().items()
        if key in DEGRADATION_KEYS
    } == {}
    assert cached.model == "basic-chat"
    assert {
        span.name: dict(span.attributes) for span in exporter.get_finished_spans()
    } == {
        "fallback": {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat", **_degradation(503)},
        "bounded": {LLM_SERVED_MODEL_ATTRIBUTE: "basic-chat"},
    }
    assert _counts(stack, {prompt}) == {"teacher": {prompt: 1}, "student": {prompt: 1}}
