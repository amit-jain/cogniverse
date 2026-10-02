"""An LM endpoint with nothing deployed, as Modal answers for an undeployed app.

The direct path talks to an HTTP server that answers byte-for-byte what
Modal's edge answers for an app that is not deployed, and then like a
redeployed one, cold start included. The routed path runs the shipped stack
(Envoy -> vLLM Semantic Router -> stub upstream) with the stub put into the
same states through its state files, so the 404 travels the real router hop.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import litellm
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.lm_endpoint_availability import (
    LMEndpointNotServing,
    lm_endpoint_availability,
)
from cogniverse_foundation.config.routed_lm import UpstreamNotServing
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.telemetry.span_contract import (
    LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE,
    LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE,
    LLM_ENDPOINT_STATE_ATTRIBUTE,
)
from tests.foundation.integration._sr_stack.stub_upstream import STATE_DIR
from tests.utils.modal_app import ModalApp

pytestmark = pytest.mark.integration

MODEL = "openai/google/gemma-4-e4b-it"
CLASSIFICATION = SemanticRouterConfig().classification_model
# A refused call never leaves the process; anything near a network round trip
# means it was sent.
FAST_FAILURE_CEILING_S = 0.05
RECHECK_S = 0.5


@pytest.fixture
def availability():
    availability = lm_endpoint_availability()
    availability.clear()
    yield availability
    availability.clear()


@pytest.fixture
def modal_app():
    app = ModalApp()
    yield app
    app.close()


def _direct_lm(app: ModalApp, timeout: float = 10.0):
    return create_dspy_lm(
        LLMEndpointConfig(
            model=MODEL,
            api_base=app.api_base,
            api_key="modal-bearer",
            request_timeout=timeout,
        )
    )


def _tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("lm-not-serving")


def _call(lm, prompt):
    try:
        return "answered", lm.forward(prompt)
    except Exception as exc:  # noqa: BLE001 - the contract under test
        return "raised", exc


def _text(response) -> str:
    return response.choices[0].message.content


class TestDirectEndpoint:
    def test_the_404_is_named_once_and_never_resent_inside_the_window(
        self, availability, modal_app
    ):
        lm = _direct_lm(modal_app)
        first, second = f"first {uuid.uuid4()}", f"second {uuid.uuid4()}"
        exporter, tracer = _tracing()

        with tracer.start_as_current_span("first"):
            how, error = _call(lm, first)
        assert how == "raised" and type(error) is LMEndpointNotServing
        assert (error.status, error.failed_fast, error.endpoint.api_base) == (
            404,
            False,
            modal_app.api_base,
        )
        assert type(error.__cause__) is litellm.NotFoundError
        assert modal_app.count({first}) == 1

        started = time.perf_counter()
        with tracer.start_as_current_span("second"):
            how, refused = _call(lm, second)
        elapsed = time.perf_counter() - started

        assert how == "raised" and type(refused) is LMEndpointNotServing
        assert refused.failed_fast is True
        assert 0.0 < refused.recheck_in_s < error.recheck_in_s
        assert elapsed < FAST_FAILURE_CEILING_S
        assert modal_app.count({second}) == 0
        spans = {
            span.name: dict(span.attributes) for span in exporter.get_finished_spans()
        }
        assert spans["first"] == {
            LLM_ENDPOINT_STATE_ATTRIBUTE: "not_serving",
            LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE: False,
            LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE: error.recheck_in_s,
        }
        assert spans["second"] == {
            LLM_ENDPOINT_STATE_ATTRIBUTE: "not_serving",
            LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE: True,
            LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE: refused.recheck_in_s,
        }

    def test_a_redeployed_app_is_used_after_the_window_through_its_cold_start(
        self, availability, modal_app, monkeypatch
    ):
        monkeypatch.setattr(availability, "recheck_after_s", RECHECK_S)
        lm = _direct_lm(modal_app)
        assert _call(lm, f"down {uuid.uuid4()}")[0] == "raised"

        modal_app.deploy(cold_start_s=2.0)
        time.sleep(RECHECK_S)
        prompt = f"after redeploy {uuid.uuid4()}"
        started = time.perf_counter()
        how, response = _call(lm, prompt)

        assert how == "answered" and _text(response) == f"served:{prompt}"
        assert time.perf_counter() - started >= 2.0
        assert [
            (entry["endpoint"], entry["state"]) for entry in availability.snapshot()
        ] == [(modal_app.api_base, "serving")]

    def test_a_cold_start_on_a_deployed_app_never_fails_fast(
        self, availability, modal_app
    ):
        modal_app.deploy(cold_start_s=1.5)
        lm = _direct_lm(modal_app)
        prompts = [f"cold {i} {uuid.uuid4()}" for i in range(3)]

        outcomes = [_call(lm, prompt) for prompt in prompts]

        assert [(how, _text(r)) for how, r in outcomes] == [
            ("answered", f"served:{prompt}") for prompt in prompts
        ]

    def test_concurrent_callers_after_the_window_send_one_recheck(
        self, availability, modal_app, monkeypatch
    ):
        monkeypatch.setattr(availability, "recheck_after_s", RECHECK_S)
        lm = _direct_lm(modal_app)
        assert _call(lm, f"down {uuid.uuid4()}")[0] == "raised"
        time.sleep(RECHECK_S)
        threads = 12
        prompts = [f"race {i} {uuid.uuid4()}" for i in range(threads)]
        barrier = threading.Barrier(threads)

        def race(prompt):
            barrier.wait(timeout=10)
            how, error = _call(lm, prompt)
            return how, type(error).__name__, getattr(error, "failed_fast", None)

        with ThreadPoolExecutor(threads) as pool:
            outcomes = Counter(pool.map(race, prompts))

        assert modal_app.count(set(prompts)) == 1
        assert outcomes == Counter(
            {
                ("raised", "LMEndpointNotServing", False): 1,
                ("raised", "LMEndpointNotServing", True): threads - 1,
            }
        )

    async def test_concurrent_async_callers_share_the_fast_failure(
        self, availability, modal_app
    ):
        lm = _direct_lm(modal_app)
        with pytest.raises(LMEndpointNotServing):
            await lm.aforward(f"down {uuid.uuid4()}")
        prompts = [f"async {i} {uuid.uuid4()}" for i in range(8)]

        outcomes = await asyncio.gather(
            *(lm.aforward(prompt) for prompt in prompts), return_exceptions=True
        )

        assert [(type(o).__name__, o.failed_fast) for o in outcomes] == [
            ("LMEndpointNotServing", True)
        ] * len(prompts)
        assert modal_app.count(set(prompts)) == 0

    async def test_a_recheck_cancelled_mid_flight_hands_over_to_the_next_call(
        self, availability, modal_app, monkeypatch
    ):
        monkeypatch.setattr(availability, "recheck_after_s", RECHECK_S)
        lm = _direct_lm(modal_app)
        with pytest.raises(LMEndpointNotServing):
            await lm.aforward(f"down {uuid.uuid4()}")
        modal_app.deploy(cold_start_s=2.0)
        await asyncio.sleep(RECHECK_S)

        abandoned = asyncio.create_task(lm.aforward(f"abandoned {uuid.uuid4()}"))
        await asyncio.sleep(0.3)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        prompt = f"next {uuid.uuid4()}"
        response = await lm.aforward(prompt)

        assert _text(response) == f"served:{prompt}"
        assert availability.snapshot()[0]["state"] == "serving"


def _stub_exec(container: str, command: str) -> str:
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout


def _requests_at(container: str, prompts) -> int:
    records = json.loads(
        _stub_exec(
            container,
            'python -c "import urllib.request; print(urllib.request.urlopen('
            "'http://127.0.0.1:8000/requests').read().decode())\"",
        )
    )["requests"]
    return sum(1 for record in records if record["prompt"] in prompts)


@pytest.fixture
def stack(semantic_router_stack):
    containers = (
        semantic_router_stack["stub_container"],
        semantic_router_stack["teacher_container"],
    )
    yield semantic_router_stack
    for container in containers:
        _stub_exec(container, f"rm -rf {STATE_DIR}")


def _undeploy(container: str) -> None:
    _stub_exec(container, f"mkdir -p {STATE_DIR} && touch {STATE_DIR}/undeployed")


def _redeploy(container: str, cold_start_s: float) -> None:
    _stub_exec(
        container,
        f"rm -f {STATE_DIR}/undeployed && echo {cold_start_s} > {STATE_DIR}/cold_start_s",
    )


def _routed_lm(stack, *, tier: str, call_site: str, tenant: str = "notserving:prod"):
    lm = create_routed_lm(
        LLMEndpointConfig(model="unused", api_base="http://unused:1/v1"),
        SemanticRouterConfig(enabled=True, semantic_router_url=stack["base_url"]),
        tenant_id=tenant,
        tier=tier,
        call_site=call_site,
    )
    lm.cache = False
    return lm


class TestRoutedEndpoint:
    def test_an_undeployed_student_is_named_and_not_called_again(
        self, availability, stack
    ):
        _undeploy(stack["stub_container"])
        lm = _routed_lm(stack, tier="free", call_site="search_agent")
        first, second = f"first {uuid.uuid4()}", f"second {uuid.uuid4()}"

        how, error = _call(lm, first)

        assert how == "raised" and type(error) is UpstreamNotServing
        assert (
            error.status,
            error.failed_fast,
            error.tenant_id,
            error.tier,
            error.routed_model,
        ) == (404, False, "notserving:prod", "free", CLASSIFICATION)
        assert type(error.__cause__) is LMEndpointNotServing
        assert type(error.__cause__.__cause__) is litellm.NotFoundError
        assert _requests_at(stack["stub_container"], {first}) == 1

        started = time.perf_counter()
        how, refused = _call(lm, second)

        assert how == "raised" and type(refused) is UpstreamNotServing
        assert refused.failed_fast is True
        assert time.perf_counter() - started < FAST_FAILURE_CEILING_S
        assert _requests_at(stack["stub_container"], {second}) == 0

    def test_another_tier_on_the_same_entry_is_not_refused_for_it(
        self, availability, stack
    ):
        _undeploy(stack["stub_container"])
        free = _routed_lm(stack, tier="free", call_site="search_agent")
        assert _call(free, f"free {uuid.uuid4()}")[0] == "raised"
        prompt = f"pro {uuid.uuid4()}"

        how, error = _call(
            _routed_lm(stack, tier="pro", call_site="search_agent"), prompt
        )

        assert (how, type(error), error.failed_fast) == (
            "raised",
            UpstreamNotServing,
            False,
        )
        assert _requests_at(stack["stub_container"], {prompt}) == 1

    def test_a_redeployed_student_answers_after_its_cold_start(
        self, availability, stack, monkeypatch
    ):
        monkeypatch.setattr(availability, "recheck_after_s", RECHECK_S)
        _undeploy(stack["stub_container"])
        lm = _routed_lm(stack, tier="free", call_site="search_agent")
        assert _call(lm, f"down {uuid.uuid4()}")[0] == "raised"

        _redeploy(stack["stub_container"], cold_start_s=3)
        time.sleep(RECHECK_S)
        prompt = f"redeployed {uuid.uuid4()}"
        started = time.perf_counter()
        how, response = _call(lm, prompt)

        assert how == "answered"
        reflection = json.loads(_text(response))
        assert (reflection["backend_tag"], reflection["echo"]) == ("student", prompt)
        assert time.perf_counter() - started >= 3.0
        assert [entry["state"] for entry in availability.snapshot()] == ["serving"]

    def test_a_pro_call_with_its_teacher_undeployed_is_served_by_the_student(
        self, availability, stack
    ):
        _undeploy(stack["teacher_container"])
        lm = _routed_lm(stack, tier="pro", call_site="summarizer_agent")
        prompts = [f"pro free-form {i} {uuid.uuid4()}" for i in range(2)]

        responses = [lm.forward(prompt) for prompt in prompts]

        for prompt, response in zip(prompts, responses):
            body = response.model_dump()
            assert {
                key: body[key]
                for key in (
                    "tier_degraded",
                    "upstream_status",
                    "upstream_exception_type",
                )
            } == {
                "tier_degraded": "pro_model_unavailable",
                "upstream_status": 404,
                "upstream_exception_type": "UpstreamNotServing",
            }
            reflection = json.loads(_text(response))
            assert (reflection["backend_tag"], reflection["echo"]) == (
                "student",
                prompt,
            )
        assert _requests_at(stack["teacher_container"], {prompts[0]}) == 1
        assert _requests_at(stack["teacher_container"], {prompts[1]}) == 0
