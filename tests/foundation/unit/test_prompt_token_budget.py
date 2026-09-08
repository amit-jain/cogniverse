"""The teacher endpoint's context window bounds what the optimizer may send.

The optimizer's teacher is served with ``--max-model-len 4096`` while its
endpoint reserves ``max_tokens=2048`` for the completion, leaving 2048 tokens
for the prompt. BootstrapFewShot grows that prompt one demonstration at a
time, so the request that overflows is the one after the last one that fit.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogniverse_foundation.config.token_budget import (
    ContextWindowUnavailableError,
    PromptBudgetExceededError,
    ResolvedContextWindow,
    TokenBudget,
    extract_context_window,
    fetch_context_window,
    fit_messages,
    litellm_message_counter,
    resolve_context_window,
    split_demonstrations,
)

# The served teacher: Qwen/Qwen3-14B-AWQ behind --max-model-len 4096 with the
# endpoint's max_tokens=2048 reservation (charts/cogniverse/files/config.json).
TEACHER_MODEL = "openai/Qwen/Qwen3-14B-AWQ"
TEACHER_CONTEXT_WINDOW = 4096
TEACHER_RESERVED_OUTPUT = 2048

# The Modal teacher's live /v1/models, recorded verbatim. The wrapper answers
# discovery itself so it never wakes the scale-to-zero GPU, and its listing
# carries no max_model_len.
MODAL_TEACHER_LISTING = {
    "data": [
        {
            "created": 0,
            "id": "Qwen/Qwen3-14B-AWQ",
            "object": "model",
            "owned_by": "cogniverse",
            "revision": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        }
    ],
    "object": "list",
}

# The same model served by an engine that does publish its window, and with a
# different one: the window belongs to the deployment, not to the model.
SERVED_LISTING = {
    "data": [
        {
            "created": 0,
            "id": "Qwen/Qwen3-14B-AWQ",
            "object": "model",
            "owned_by": "cogniverse",
            "revision": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
            "max_model_len": 8192,
        }
    ],
    "object": "list",
}


def _words(count: int) -> str:
    """A message body of exactly ``count`` whitespace-separated tokens."""

    return " ".join(f"w{index}" for index in range(count))


def _count_words(messages) -> int:
    return sum(len(str(message["content"]).split()) for message in messages)


@pytest.mark.unit
class TestTeacherBudgetArithmetic:
    def test_reserved_completion_leaves_half_the_served_window_for_input(self):
        budget = TokenBudget(
            model=TEACHER_MODEL,
            context_window=TEACHER_CONTEXT_WINDOW,
            reserved_output=TEACHER_RESERVED_OUTPUT,
        )

        assert budget.input_budget == 2048
        assert budget.context_window - budget.reserved_output == 2048

    def test_reserving_the_whole_window_is_rejected_at_construction(self):
        with pytest.raises(ValueError) as exc:
            TokenBudget(
                model=TEACHER_MODEL,
                context_window=4096,
                reserved_output=4096,
            )

        assert str(exc.value) == (
            "openai/Qwen/Qwen3-14B-AWQ reserves 4096 output tokens of a "
            "4096-token context window, leaving no room for input"
        )

    def test_zero_reservation_is_rejected_at_construction(self):
        with pytest.raises(ValueError) as exc:
            TokenBudget(
                model=TEACHER_MODEL,
                context_window=4096,
                reserved_output=0,
            )

        assert str(exc.value) == (
            "openai/Qwen/Qwen3-14B-AWQ reserved_output must be positive, got 0"
        )


@pytest.mark.unit
class TestFitMessagesToTheServedWindow:
    """2048 output + 2049 input = 4097 against a 4096-token window."""

    def _budget(self) -> TokenBudget:
        return TokenBudget(
            model=TEACHER_MODEL,
            context_window=TEACHER_CONTEXT_WINDOW,
            reserved_output=TEACHER_RESERVED_OUTPUT,
        )

    def _overflowing_request(self):
        # 1000 + 500 + 100 + 300 + 100 + 49 = 2049, one token past the 2048
        # the reserved completion leaves.
        return [
            {"role": "system", "content": _words(1000)},
            {"role": "user", "content": _words(500)},
            {"role": "assistant", "content": _words(100)},
            {"role": "user", "content": _words(300)},
            {"role": "assistant", "content": _words(100)},
            {"role": "user", "content": _words(49)},
        ]

    def test_the_captured_request_is_exactly_one_token_over_the_input_budget(self):
        budget = self._budget()
        messages = self._overflowing_request()

        assert _count_words(messages) == 2049
        assert _count_words(messages) == budget.input_budget + 1
        assert _count_words(messages) + budget.reserved_output == 4097

    def test_demonstrations_split_into_user_led_turns_around_the_live_query(self):
        preamble, demos, live = split_demonstrations(self._overflowing_request())

        assert [message["role"] for message in preamble] == ["system"]
        assert [[message["role"] for message in demo] for demo in demos] == [
            ["user", "assistant"],
            ["user", "assistant"],
        ]
        assert [message["role"] for message in live] == ["user"]

    def test_the_oldest_demonstration_is_dropped_and_the_request_then_fits(self):
        budget = self._budget()
        messages = self._overflowing_request()

        fitted = fit_messages(messages, budget=budget, count_tokens=_count_words)

        assert fitted.dropped_demos == 1
        assert fitted.total_demos == 2
        assert fitted.input_tokens == 1449
        assert fitted.input_tokens + budget.reserved_output == 3497
        assert fitted.messages == [
            {"role": "system", "content": _words(1000)},
            {"role": "user", "content": _words(300)},
            {"role": "assistant", "content": _words(100)},
            {"role": "user", "content": _words(49)},
        ]

    def test_a_request_already_inside_the_budget_is_sent_unchanged(self):
        budget = self._budget()
        messages = [
            {"role": "system", "content": _words(10)},
            {"role": "user", "content": _words(5)},
            {"role": "assistant", "content": _words(5)},
            {"role": "user", "content": _words(4)},
        ]

        fitted = fit_messages(messages, budget=budget, count_tokens=_count_words)

        assert fitted.dropped_demos == 0
        assert fitted.total_demos == 1
        assert fitted.input_tokens == 24
        assert fitted.messages == messages

    def test_a_prompt_that_cannot_fit_without_demonstrations_names_the_budget(self):
        budget = self._budget()
        messages = [
            {"role": "system", "content": _words(2000)},
            {"role": "user", "content": _words(400)},
            {"role": "assistant", "content": _words(100)},
            {"role": "user", "content": _words(400)},
            {"role": "assistant", "content": _words(100)},
            {"role": "user", "content": _words(100)},
        ]

        with pytest.raises(PromptBudgetExceededError) as exc:
            fit_messages(messages, budget=budget, count_tokens=_count_words)

        assert str(exc.value) == (
            "openai/Qwen/Qwen3-14B-AWQ prompt does not fit the served context "
            "window: context_window=4096 reserved_output=2048 input_budget=2048 "
            "input_tokens=2100 after dropping all 2 few-shot demonstrations"
        )
        assert exc.value.context_window == 4096
        assert exc.value.reserved_output == 2048
        assert exc.value.input_tokens == 2100
        assert exc.value.dropped_demos == 2


@pytest.mark.unit
class TestLitellmTokenCounter:
    """The default counter is the tokenizer litellm bills the request by."""

    def test_counts_a_chat_request_with_the_billing_tokenizer(self):
        import litellm

        messages = [
            {"role": "system", "content": "Extract entities from the query."},
            {"role": "user", "content": "Who founded Anthropic and when?"},
        ]
        count = litellm_message_counter(TEACHER_MODEL)(messages)

        assert count == litellm.token_counter(model=TEACHER_MODEL, messages=messages)
        assert count == 24


@pytest.mark.unit
class TestServedContextWindowDiscovery:
    def test_reads_max_model_len_from_the_models_listing(self):
        body = {
            "object": "list",
            "data": [
                {
                    "id": "Qwen/Qwen3-14B-AWQ",
                    "object": "model",
                    "max_model_len": 4096,
                }
            ],
        }

        assert extract_context_window(body) == 4096

    def test_a_listing_without_max_model_len_yields_no_window(self):
        body = {"object": "list", "data": [{"id": "qwen2.5:0.5b"}]}

        assert extract_context_window(body) is None

    def test_an_endpoint_that_declares_no_window_refuses_to_be_budgeted(self):
        server, base_url = _serve({"object": "list", "data": [{"id": "m"}]})
        try:
            with pytest.raises(ContextWindowUnavailableError) as exc:
                fetch_context_window(f"{base_url}/v1")
        finally:
            server.shutdown()

        assert str(exc.value) == (
            f"{base_url}/v1/models reports no max_model_len, so the served "
            f"context window is unknown and no request budget can be derived"
        )


@pytest.mark.unit
class TestContextWindowPrecedence:
    """What the endpoint serves wins; what it is configured with is the fallback.

    The Modal wrapper shadows the engine's listing to keep discovery off the
    GPU, so the teacher endpoint publishes no window at all. Refusing to budget
    there kills the optimizer outright, and guessing a window for an endpoint
    that publishes one would silently ignore the deployment it is talking to.
    """

    def test_a_listing_without_a_window_falls_back_to_the_declared_one(self):
        server, base_url = _serve(MODAL_TEACHER_LISTING)
        try:
            resolved = resolve_context_window(
                f"{base_url}/v1",
                declared=TEACHER_CONTEXT_WINDOW,
                model=TEACHER_MODEL,
            )
        finally:
            server.shutdown()

        assert resolved == ResolvedContextWindow(tokens=4096, source="declared")
        assert extract_context_window(MODAL_TEACHER_LISTING) is None

    def test_a_published_window_overrides_the_declared_one(self):
        server, base_url = _serve(SERVED_LISTING)
        try:
            resolved = resolve_context_window(
                f"{base_url}/v1",
                declared=TEACHER_CONTEXT_WINDOW,
                model=TEACHER_MODEL,
            )
        finally:
            server.shutdown()

        assert resolved == ResolvedContextWindow(tokens=8192, source="served")

    def test_neither_a_published_nor_a_declared_window_names_both_ends(self):
        server, base_url = _serve(MODAL_TEACHER_LISTING)
        try:
            with pytest.raises(ContextWindowUnavailableError) as exc:
                resolve_context_window(
                    f"{base_url}/v1",
                    declared=None,
                    model=TEACHER_MODEL,
                )
        finally:
            server.shutdown()

        assert str(exc.value) == (
            f"openai/Qwen/Qwen3-14B-AWQ at {base_url}/v1 publishes no context "
            f"window and declares none, so no request budget can be derived"
        )
        assert str(exc.value.__cause__) == (
            f"{base_url}/v1/models reports no max_model_len, so the served "
            f"context window is unknown and no request budget can be derived"
        )


@pytest.mark.unit
class TestBudgetedLMOverTheWire:
    """The served window is read from the endpoint and bounds the real request."""

    def test_the_teacher_request_reaches_the_endpoint_without_the_dropped_demo(self):
        from cogniverse_foundation.config.budgeted_lm import BudgetedLM

        server, base_url, received = _serve_openai(
            {
                "object": "list",
                "data": [{"id": "Qwen/Qwen3-14B-AWQ", "max_model_len": 4096}],
            }
        )
        lm = BudgetedLM(
            TEACHER_MODEL,
            api_base=f"{base_url}/v1",
            api_key="not-required",
            max_tokens=TEACHER_RESERVED_OUTPUT,
            temperature=0.7,
            cache=False,
            num_retries=0,
        )
        # Two demonstrations that assemble to exactly the captured 2049-token
        # prompt: 2049 input + 2048 reserved output = 4097 against 4096.
        demo_body = " ".join(["entity"] * 997)
        messages = [
            {"role": "system", "content": "Extract entities from the query."},
            {"role": "user", "content": f"first {demo_body}"},
            {"role": "assistant", "content": "first|CONCEPT|1.0"},
            {"role": "user", "content": f"second {demo_body}"},
            {"role": "assistant", "content": "second|CONCEPT|1.0"},
            {"role": "user", "content": "who founded Anthropic"},
        ]
        try:
            outputs = lm(messages=messages)
        finally:
            server.shutdown()

        assert outputs == ["entities: anthropic|ORGANIZATION|1.0"]
        assert lm.budget.context_window == 4096
        assert lm.budget.reserved_output == 2048
        assert lm.budget.input_budget == 2048
        assert received["paths"] == ["/v1/models", "/v1/chat/completions"]
        assert received["body"]["messages"] == [
            {"role": "system", "content": "Extract entities from the query."},
            {"role": "user", "content": f"second {demo_body}"},
            {"role": "assistant", "content": "second|CONCEPT|1.0"},
            {"role": "user", "content": "who founded Anthropic"},
        ]
        assert received["body"]["max_tokens"] == 2048
        counter = litellm_message_counter(TEACHER_MODEL)
        assert counter(messages) == 2049
        assert counter(messages) + TEACHER_RESERVED_OUTPUT == 4097
        assert counter(received["body"]["messages"]) == 1035
        assert counter(received["body"]["messages"]) + TEACHER_RESERVED_OUTPUT == 3083


@pytest.mark.unit
class TestBudgetResolutionUnderConcurrency:
    def test_eight_first_touches_read_the_served_window_once(self):
        from cogniverse_foundation.config.budgeted_lm import BudgetedLM

        server, base_url, received = _serve_openai(
            {
                "object": "list",
                "data": [{"id": "Qwen/Qwen3-14B-AWQ", "max_model_len": 4096}],
            }
        )
        lm = BudgetedLM(
            TEACHER_MODEL,
            api_base=f"{base_url}/v1",
            api_key="not-required",
            max_tokens=TEACHER_RESERVED_OUTPUT,
        )
        start = threading.Barrier(8)
        observed: list = []

        def resolve():
            start.wait(timeout=10)
            observed.append(lm.budget)

        threads = [threading.Thread(target=resolve) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        finally:
            server.shutdown()

        assert len(observed) == 8
        assert {id(budget) for budget in observed} == {id(observed[0])}
        assert observed[0].context_window == 4096
        assert observed[0].input_budget == 2048
        assert received["paths"] == ["/v1/models"]


@pytest.mark.unit
class TestDeclaredWindowBoundsTheModalTeacher:
    """The teacher's window is configuration, because its listing carries none."""

    def _lm(self, base_url: str):
        from cogniverse_foundation.config.budgeted_lm import BudgetedLM

        return BudgetedLM(
            TEACHER_MODEL,
            declared_context_window=TEACHER_CONTEXT_WINDOW,
            api_base=f"{base_url}/v1",
            api_key="not-required",
            max_tokens=TEACHER_RESERVED_OUTPUT,
            temperature=0.7,
            cache=False,
            num_retries=0,
        )

    def test_the_request_sheds_its_demo_against_a_listing_without_a_window(self):
        server, base_url, received = _serve_openai(MODAL_TEACHER_LISTING)
        lm = self._lm(base_url)
        demo_body = " ".join(["entity"] * 997)
        messages = [
            {"role": "system", "content": "Extract entities from the query."},
            {"role": "user", "content": f"first {demo_body}"},
            {"role": "assistant", "content": "first|CONCEPT|1.0"},
            {"role": "user", "content": f"second {demo_body}"},
            {"role": "assistant", "content": "second|CONCEPT|1.0"},
            {"role": "user", "content": "who founded Anthropic"},
        ]
        try:
            outputs = lm(messages=messages)
        finally:
            server.shutdown()

        assert outputs == ["entities: anthropic|ORGANIZATION|1.0"]
        assert lm.budget.context_window == 4096
        assert lm.budget.input_budget == 2048
        assert received["paths"] == ["/v1/models", "/v1/chat/completions"]
        assert received["body"]["messages"] == [
            {"role": "system", "content": "Extract entities from the query."},
            {"role": "user", "content": f"second {demo_body}"},
            {"role": "assistant", "content": "second|CONCEPT|1.0"},
            {"role": "user", "content": "who founded Anthropic"},
        ]
        assert "declared_context_window" not in received["body"]

    def test_a_copied_lm_still_carries_the_declared_window(self):
        server, base_url, received = _serve_openai(MODAL_TEACHER_LISTING)
        duplicate = self._lm(base_url).copy()
        try:
            budget = duplicate.budget
        finally:
            server.shutdown()

        assert duplicate.declared_context_window == 4096
        assert "declared_context_window" not in duplicate.kwargs
        assert budget.context_window == 4096
        assert budget.reserved_output == 2048
        assert budget.input_budget == 2048
        assert received["paths"] == ["/v1/models"]


@pytest.mark.unit
class TestBudgetFaultContract:
    def test_an_unreachable_endpoint_refuses_to_be_budgeted(self):
        from cogniverse_foundation.config.budgeted_lm import BudgetedLM

        server, base_url, _ = _serve_openai({"object": "list", "data": []})
        server.shutdown()
        lm = BudgetedLM(
            TEACHER_MODEL,
            api_base=f"{base_url}/v1",
            api_key="not-required",
            max_tokens=TEACHER_RESERVED_OUTPUT,
            num_retries=0,
        )

        with pytest.raises(ContextWindowUnavailableError) as exc:
            lm.budget

        assert str(exc.value) == (
            f"openai/Qwen/Qwen3-14B-AWQ at {base_url}/v1 publishes no context "
            f"window and declares none, so no request budget can be derived"
        )
        assert str(exc.value.__cause__) == (
            f"{base_url}/v1/models is unreachable, so the served context window "
            f"is unknown and no request budget can be derived"
        )

    def test_an_endpoint_reserving_no_completion_has_no_input_allowance(self):
        from cogniverse_foundation.config.budgeted_lm import BudgetedLM

        lm = BudgetedLM(TEACHER_MODEL, api_base="http://127.0.0.1:1/v1")

        with pytest.raises(ContextWindowUnavailableError) as exc:
            lm.budget

        assert str(exc.value) == (
            "openai/Qwen/Qwen3-14B-AWQ reserves no completion tokens, so the "
            "input allowance of its context window is undefined"
        )


class _StubHandler(BaseHTTPRequestHandler):
    models_body: dict = {}
    received: dict = {}

    def _json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        type(self).received.setdefault("paths", []).append(self.path)
        self._json(type(self).models_body)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        type(self).received.setdefault("paths", []).append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        type(self).received["body"] = json.loads(self.rfile.read(length))
        self._json(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-14B-AWQ",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "entities: anthropic|ORGANIZATION|1.0",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1035,
                    "completion_tokens": 9,
                    "total_tokens": 1044,
                },
            }
        )

    def log_message(self, *args):
        return


def _start(handler_attrs: dict):
    handler = type("Handler", (_StubHandler,), handler_attrs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _serve(models_body: dict):
    server, base_url = _start({"models_body": models_body, "received": {}})
    return server, base_url


def _serve_openai(models_body: dict):
    received: dict = {}
    server, base_url = _start({"models_body": models_body, "received": received})
    return server, base_url, received
