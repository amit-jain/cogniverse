"""A request larger than the model's window is named as that, and an LM knows
the allowance of every endpoint that may serve its calls."""

from __future__ import annotations

import threading

import litellm
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.routed_lm import (
    RETRYABLE,
    ROUTED_PROMPT_MARGIN_TOKENS,
    ContextWindowExceeded,
    RouterDecodeFailed,
    ServingEndpoint,
    UpstreamRateLimited,
    UpstreamUnavailable,
    classify_routed_failure,
)
from cogniverse_foundation.config.semantic_router import (
    TEACHER_SERVICE,
    create_routed_lm,
    pro_tier_endpoint,
)
from cogniverse_foundation.config.token_budget import (
    ContextWindowUnavailableError,
    TokenBudget,
    TokenCountUnavailableError,
    clear_context_window_memo,
)
from cogniverse_foundation.config.unified_config import (
    LLMConfig,
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from tests.utils.llm_endpoint_stub import serve_llm_endpoint

pytestmark = [pytest.mark.unit]

TENANT = "acme:prod"
STUDENT = "openai/google/gemma-4-e4b-it"
CLASSIFICATION = "openai/cogniverse-classification"
KW = {"tenant_id": TENANT, "tier": "pro", "routed_model": CLASSIFICATION}

# What the Modal student answered an orchestrated request with, verbatim.
VLLM_OVERFLOW = (
    "This model's maximum context length is 8192 tokens. However, you "
    "requested 1000 output tokens and your prompt contains at least 7193 "
    "input tokens, for a total of at least 8193 tokens. Please reduce the "
    "length of the input prompt or the number of requested output tokens. "
    "(parameter=input_tokens, value=7193)"
)
OPENAI_OVERFLOW = (
    "This model's maximum context length is 4097 tokens. However, you "
    "requested 4200 tokens (3200 in the messages, 1000 in the completion). "
    "Please reduce the length of the messages or completion."
)


@pytest.fixture(autouse=True)
def _fresh_windows():
    clear_context_window_memo()
    yield
    clear_context_window_memo()


def _overflow(message: str) -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message=message, model="cogniverse-classification", llm_provider="openai"
    )


def _routed_lm(router_url: str, endpoint: LLMEndpointConfig, call_site: str):
    return create_routed_lm(
        endpoint=endpoint,
        config=SemanticRouterConfig(enabled=True, semantic_router_url=router_url),
        tenant_id=TENANT,
        tier="pro",
        call_site=call_site,
    )


class TestContextOverflowClassification:
    def test_a_context_overflow_is_not_a_router_refusal(self):
        failure = classify_routed_failure(_overflow(VLLM_OVERFLOW), **KW)

        assert type(failure) is ContextWindowExceeded
        assert not isinstance(failure, RouterDecodeFailed)
        assert failure.status == 400
        assert failure.context_window == 8192
        assert failure.prompt_tokens == 7193
        assert failure.requested_output_tokens == 1000
        assert str(failure) == (
            "the request does not fit the model's context window: "
            f"tenant={TENANT} tier=pro routed_model={CLASSIFICATION} "
            "status=400 router_code=None context_window=8192 "
            "prompt_tokens=7193 requested_output_tokens=1000"
        )

    def test_the_openai_wording_is_read_too(self):
        failure = classify_routed_failure(_overflow(OPENAI_OVERFLOW), **KW)

        assert type(failure) is ContextWindowExceeded
        assert (
            failure.context_window,
            failure.prompt_tokens,
            failure.requested_output_tokens,
        ) == (4097, 3200, 1000)

    def test_an_overflow_that_reports_no_numbers_carries_none(self):
        failure = classify_routed_failure(_overflow("context too long"), **KW)

        assert type(failure) is ContextWindowExceeded
        assert str(failure) == (
            "the request does not fit the model's context window: "
            f"tenant={TENANT} tier=pro routed_model={CLASSIFICATION} "
            "status=400 router_code=None context_window=None "
            "prompt_tokens=None requested_output_tokens=None"
        )

    def test_an_overflow_is_never_retried(self):
        assert RETRYABLE == (UpstreamRateLimited, UpstreamUnavailable)
        assert not issubclass(ContextWindowExceeded, RETRYABLE)

    @pytest.mark.parametrize(
        ("status", "kind"),
        [(400, RouterDecodeFailed), (422, RouterDecodeFailed)],
    )
    def test_any_other_refused_request_is_still_the_router_s(self, status, kind):
        refused = litellm.BadRequestError(
            message="unknown model", model="x", llm_provider="openai"
        )
        refused.status_code = status
        failure = classify_routed_failure(refused, **KW)

        assert type(failure) is kind
        assert failure.status == status

    def test_an_outage_is_still_an_outage(self):
        failure = classify_routed_failure(
            litellm.ServiceUnavailableError(
                message="overloaded", model="x", llm_provider="openai"
            ),
            **KW,
        )

        assert type(failure) is UpstreamUnavailable
        assert failure.status == 503


class TestContextOverflowOverTheWire:
    """litellm's own mapping of the endpoint's 400, through a ``RoutedLM``."""

    def _call(self, error_message: str):
        router = serve_llm_endpoint(
            chat_error=(
                400,
                {
                    "error": {
                        "message": error_message,
                        "type": "BadRequestError",
                        "param": None,
                        "code": 400,
                    }
                },
            )
        )
        lm = _routed_lm(
            router.api_base,
            LLMEndpointConfig(model=STUDENT, api_base=None, num_retries=1),
            "orchestrator_agent",
        )
        try:
            with pytest.raises(Exception) as raised:
                lm(f"overflow probe: {error_message[:24]}")
        finally:
            router.close()
        return raised.value, router.posted("/v1/chat/completions")

    def test_the_endpoint_s_overflow_answer_raises_the_typed_failure(self):
        failure, sent = self._call(VLLM_OVERFLOW)

        assert type(failure) is ContextWindowExceeded
        assert (
            failure.status,
            failure.context_window,
            failure.prompt_tokens,
            failure.requested_output_tokens,
            failure.routed_model,
            failure.tenant_id,
        ) == (400, 8192, 7193, 1000, CLASSIFICATION, TENANT)
        assert type(failure.__cause__) is litellm.ContextWindowExceededError
        # Sent once: the same request overflows the same way every time.
        assert len(sent) == 1

    def test_another_400_from_the_same_route_is_a_router_refusal(self):
        failure, sent = self._call("model 'x' is not in the catalog")

        assert type(failure) is RouterDecodeFailed
        assert failure.status == 400
        assert len(sent) == 1


class TestRoutedPromptAllowance:
    def test_the_classification_entry_is_budgeted_by_the_endpoint_serving_it(self):
        serving = serve_llm_endpoint(context_window=8192)
        endpoint = LLMEndpointConfig(
            model=STUDENT, api_base=serving.api_base, max_tokens=1000, num_retries=2
        )
        lm = _routed_lm("http://127.0.0.1:9/v1", endpoint, "orchestrator_agent")
        try:
            allowance = lm.prompt_allowance()
            counted = allowance.count_tokens(
                [{"role": "user", "content": "one two three"}]
            )
        finally:
            serving.close()

        assert lm.model == CLASSIFICATION
        assert lm.served_by == (
            ServingEndpoint(
                api_base=serving.api_base,
                model=STUDENT,
                context_window=None,
                num_retries=2,
                request_timeout=120.0,
            ),
        )
        assert allowance.budget == TokenBudget(
            model=STUDENT, context_window=8192, reserved_output=1000
        )
        assert ROUTED_PROMPT_MARGIN_TOKENS == 32
        assert allowance.margin == 32
        assert allowance.tokens == 7160
        assert counted == 7
        assert serving.received == [
            ("/v1/models", None),
            (
                "/tokenize",
                {
                    "model": "google/gemma-4-e4b-it",
                    "messages": [{"role": "user", "content": "one two three"}],
                },
            ),
        ]

    def test_an_entry_whose_upstream_the_router_picks_has_no_allowance(self):
        endpoint = LLMEndpointConfig(model=STUDENT, api_base="http://127.0.0.1:9/v1")
        lm = _routed_lm("http://127.0.0.1:10/v1", endpoint, "summarizer_agent")

        assert lm.model == "openai/auto"
        assert lm.served_by == ()
        assert lm.prompt_allowance() is None

    def test_an_endpoint_already_addressing_the_router_serves_nothing_itself(self):
        router_url = "http://127.0.0.1:10/v1"
        endpoint = LLMEndpointConfig(model=STUDENT, api_base=router_url)
        lm = _routed_lm(router_url, endpoint, "orchestrator_agent")

        assert lm.model == CLASSIFICATION
        assert lm.prompt_allowance() is None

    def test_a_copied_lm_keeps_its_allowance(self):
        serving = serve_llm_endpoint(context_window=4096)
        endpoint = LLMEndpointConfig(
            model=STUDENT, api_base=serving.api_base, max_tokens=500
        )
        lm = _routed_lm("http://127.0.0.1:9/v1", endpoint, "orchestrator_agent")
        try:
            allowance = lm.copy().prompt_allowance()
        finally:
            serving.close()

        assert allowance.budget == TokenBudget(
            model=STUDENT, context_window=4096, reserved_output=500
        )

    def test_the_declared_window_bounds_an_unreachable_serving_endpoint(self):
        dead = "http://127.0.0.1:9/v1"
        endpoint = LLMEndpointConfig(
            model=STUDENT, api_base=dead, max_tokens=1000, context_window=8192
        )
        lm = _routed_lm("http://127.0.0.1:10/v1", endpoint, "orchestrator_agent")

        assert lm.prompt_allowance().budget == TokenBudget(
            model=STUDENT, context_window=8192, reserved_output=1000
        )

    def test_an_unreachable_serving_endpoint_declaring_no_window_raises(self):
        dead = "http://127.0.0.1:9/v1"
        endpoint = LLMEndpointConfig(model=STUDENT, api_base=dead, max_tokens=1000)
        lm = _routed_lm("http://127.0.0.1:10/v1", endpoint, "orchestrator_agent")

        with pytest.raises(ContextWindowUnavailableError) as raised:
            lm.prompt_allowance()
        assert str(raised.value) == (
            f"{STUDENT} at {dead} publishes no context window and declares "
            "none, so no request budget can be derived"
        )


TEACHER = "openai/Qwen/Qwen3-14B-AWQ"
SUMMARIZER = "summarizer_agent"
ROUTER = "http://127.0.0.1:10/v1"


def _endpoint(api_base: str, model: str = STUDENT, **fields) -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model=model, api_base=api_base, max_tokens=1000, num_retries=0, **fields
    )


def _serving(api_base: str, model: str = STUDENT, **fields) -> ServingEndpoint:
    return ServingEndpoint(
        api_base=api_base,
        model=model,
        num_retries=0,
        request_timeout=120.0,
        **fields,
    )


def _tiered_lm(tier: str, call_site: str, student: str, teacher: str | None):
    return create_routed_lm(
        endpoint=_endpoint(student),
        config=SemanticRouterConfig(enabled=True, semantic_router_url=ROUTER),
        tenant_id=TENANT,
        tier=tier,
        call_site=call_site,
        pro_endpoint=None if teacher is None else _endpoint(teacher, TEACHER),
    )


class TestTheEndpointsServingARoutedLM:
    """Which endpoints a routed LM's prompt has to fit, per tier and entry."""

    STUDENT_BASE = "http://127.0.0.1:11/v1"
    TEACHER_BASE = "http://127.0.0.1:12/v1"

    @pytest.mark.parametrize(
        ("tier", "call_site", "teacher", "served"),
        [
            ("pro", SUMMARIZER, True, ("teacher", "student")),
            ("pro", "deep_research_decomposition", True, ("teacher", "student")),
            ("pro", SUMMARIZER, False, ()),
            ("pro", "orchestrator_agent", True, ("student",)),
            ("free", SUMMARIZER, True, ("student",)),
            ("default", SUMMARIZER, True, ("student",)),
            ("free", "orchestrator_agent", False, ("student",)),
        ],
    )
    def test_each_tier_and_entry_names_its_endpoints(
        self, tier, call_site, teacher, served
    ):
        lm = _tiered_lm(
            tier, call_site, self.STUDENT_BASE, self.TEACHER_BASE if teacher else None
        )
        endpoints = {
            "student": _serving(self.STUDENT_BASE),
            "teacher": _serving(self.TEACHER_BASE, TEACHER),
        }
        assert lm.served_by == tuple(endpoints[name] for name in served)

    def test_the_vision_entry_and_the_fallback_student_are_served_by_the_student(
        self,
    ):
        lm = _tiered_lm("pro", SUMMARIZER, self.STUDENT_BASE, self.TEACHER_BASE)

        assert (lm._vision.model, lm._vision.served_by) == (
            "openai/cogniverse-vision",
            (_serving(self.STUDENT_BASE),),
        )
        assert (lm._student.model, lm._student.served_by) == (
            CLASSIFICATION,
            (_serving(self.STUDENT_BASE),),
        )

    def test_a_student_serving_the_pro_tier_is_named_once(self):
        student = _endpoint(self.STUDENT_BASE)
        lm = create_routed_lm(
            endpoint=student,
            config=SemanticRouterConfig(enabled=True, semantic_router_url=ROUTER),
            tenant_id=TENANT,
            tier="pro",
            call_site=SUMMARIZER,
            pro_endpoint=student,
        )
        assert lm.served_by == (_serving(self.STUDENT_BASE),)


class TestAPromptMustFitEveryServingEndpoint:
    def test_a_pro_summary_is_budgeted_by_the_teacher_and_the_student(self):
        student = serve_llm_endpoint(context_window=8192)
        teacher = serve_llm_endpoint(model="Qwen/Qwen3-14B-AWQ", context_window=4096)
        lm = _tiered_lm("pro", SUMMARIZER, student.api_base, teacher.api_base)
        messages = [{"role": "user", "content": "one two three"}]
        try:
            allowance = lm.prompt_allowance_for(messages)
            counted = allowance.count_tokens(messages)
        finally:
            student.close()
            teacher.close()

        # The teacher's 4096 - 1000 - 32 binds; both stubs count 7.
        assert allowance.budget == TokenBudget(
            model=TEACHER, context_window=4096, reserved_output=1000
        )
        assert allowance.tokens == 3064
        assert counted == 7
        assert teacher.posted("/tokenize") == [
            {"model": "Qwen/Qwen3-14B-AWQ", "messages": messages}
        ]
        assert student.posted("/tokenize") == [
            {"model": "google/gemma-4-e4b-it", "messages": messages}
        ]

    def test_a_request_carrying_an_image_is_budgeted_by_the_student_alone(self):
        student = serve_llm_endpoint(context_window=8192)
        teacher = serve_llm_endpoint(model="Qwen/Qwen3-14B-AWQ", context_window=4096)
        lm = _tiered_lm("pro", SUMMARIZER, student.api_base, teacher.api_base)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
        try:
            allowance = lm.prompt_allowance_for(messages)
            allowance.count_tokens(messages)
        finally:
            student.close()
            teacher.close()

        assert allowance.budget == TokenBudget(
            model=STUDENT, context_window=8192, reserved_output=1000
        )
        assert allowance.tokens == 7160
        assert teacher.received == []
        assert [path for path, _ in student.received] == ["/v1/models", "/tokenize"]

    def test_a_pro_entry_with_an_unknown_upstream_has_no_allowance(self):
        lm = _tiered_lm("pro", SUMMARIZER, "http://127.0.0.1:11/v1", None)
        assert lm.prompt_allowance_for([{"role": "user", "content": "x"}]) is None

    def test_a_teacher_that_cannot_count_stops_the_count(self):
        student = serve_llm_endpoint(context_window=8192)
        lm = create_routed_lm(
            endpoint=_endpoint(student.api_base),
            config=SemanticRouterConfig(enabled=True, semantic_router_url=ROUTER),
            tenant_id=TENANT,
            tier="pro",
            call_site=SUMMARIZER,
            pro_endpoint=_endpoint(
                "http://127.0.0.1:9/v1", TEACHER, context_window=4096
            ),
        )
        try:
            allowance = lm.prompt_allowance()
            with pytest.raises(TokenCountUnavailableError) as raised:
                allowance.count_tokens([{"role": "user", "content": "x"}])
        finally:
            student.close()

        assert str(raised.value) == (
            "http://127.0.0.1:9/tokenize is unreachable, so the request cannot "
            f"be counted with the tokenizer {TEACHER} is served with"
        )

    def test_concurrent_first_counts_read_each_window_once(self):
        student = serve_llm_endpoint(context_window=8192)
        teacher = serve_llm_endpoint(model="Qwen/Qwen3-14B-AWQ", context_window=4096)
        lm = _tiered_lm("pro", SUMMARIZER, student.api_base, teacher.api_base)
        threads = 8
        barrier = threading.Barrier(threads)
        counts: dict[int, int] = {}

        def worker(index: int) -> None:
            messages = [{"role": "user", "content": "word " * index}]
            barrier.wait()
            counts[index] = lm.prompt_allowance().count_tokens(messages)

        pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        try:
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join()
        finally:
            student.close()
            teacher.close()

        assert counts == {index: 4 + index for index in range(threads)}
        assert student.received.count(("/v1/models", None)) == 1
        assert teacher.received.count(("/v1/models", None)) == 1
        assert len(student.posted("/tokenize")) == threads
        assert len(teacher.posted("/tokenize")) == threads


class TestTheProTierEndpoint:
    STUDENT_ENDPOINT = _endpoint("http://127.0.0.1:11/v1")
    TEACHER_ENDPOINT = _endpoint("http://127.0.0.1:12/v1", TEACHER)

    def test_a_deployed_teacher_serves_the_pro_tier(self):
        llm = LLMConfig(primary=self.STUDENT_ENDPOINT, teacher=self.TEACHER_ENDPOINT)
        assert (
            pro_tier_endpoint(
                llm, {TEACHER_SERVICE: "http://teacher"}, self.STUDENT_ENDPOINT
            )
            is self.TEACHER_ENDPOINT
        )
        assert TEACHER_SERVICE == "vllm_llm_teacher"

    def test_without_a_deployed_teacher_the_student_serves_the_pro_tier(self):
        llm = LLMConfig(primary=self.STUDENT_ENDPOINT, teacher=self.TEACHER_ENDPOINT)
        assert (
            pro_tier_endpoint(llm, {"vllm_asr": "http://asr"}, self.STUDENT_ENDPOINT)
            is self.STUDENT_ENDPOINT
        )

    def test_a_deployed_teacher_without_an_endpoint_is_unknown(self):
        llm = LLMConfig(primary=self.STUDENT_ENDPOINT)
        assert (
            pro_tier_endpoint(
                llm, {TEACHER_SERVICE: "http://teacher"}, self.STUDENT_ENDPOINT
            )
            is None
        )


class TestADirectLMKnowsItsEndpointsAllowance:
    def test_the_served_window_and_tokenizer_bound_a_direct_lm(self):
        serving = serve_llm_endpoint(context_window=8192)
        lm = create_dspy_lm(_endpoint(serving.api_base, request_timeout=45.0))
        try:
            allowance = lm.prompt_allowance()
            counted = allowance.count_tokens([{"role": "user", "content": "a b"}])
        finally:
            serving.close()

        assert allowance.budget == TokenBudget(
            model=STUDENT, context_window=8192, reserved_output=1000
        )
        assert (allowance.margin, allowance.tokens, counted) == (0, 7192, 6)
        assert [path for path, _ in serving.received] == ["/v1/models", "/tokenize"]

    def test_the_declared_window_bounds_a_listing_without_one(self):
        serving = serve_llm_endpoint()
        lm = create_dspy_lm(_endpoint(serving.api_base, context_window=4096))
        try:
            budget = lm.prompt_allowance().budget
        finally:
            serving.close()

        assert budget == TokenBudget(
            model=STUDENT, context_window=4096, reserved_output=1000
        )

    def test_an_lm_naming_no_endpoint_has_no_allowance(self):
        lm = create_dspy_lm(LLMEndpointConfig(model="openai/gpt-4o-mini"))
        assert lm.prompt_allowance() is None
