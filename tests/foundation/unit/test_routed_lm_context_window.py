"""A request larger than the model's window is named as that, and a routed LM
on the classification entry knows the allowance of the endpoint serving it."""

from __future__ import annotations

import litellm
import pytest

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
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.token_budget import (
    ContextWindowUnavailableError,
    TokenBudget,
    clear_context_window_memo,
)
from cogniverse_foundation.config.unified_config import (
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
        assert lm.served_by == ServingEndpoint(
            api_base=serving.api_base,
            model=STUDENT,
            context_window=None,
            num_retries=2,
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
        assert lm.served_by is None
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
