"""The teacher endpoint's context window bounds what the optimizer may send.

The optimizer's teacher is served with ``--max-model-len 4096`` while its
endpoint reserves ``max_tokens=2048`` for the completion, leaving 2048 tokens
for the prompt. BootstrapFewShot grows that prompt one demonstration at a
time, so the request that overflows is the one after the last one that fit.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogniverse_foundation.config.token_budget import (
    SINGLE_TURN_TEMPLATE_TOKENS,
    TOKENIZE_TIMEOUT_SECONDS,
    ContextWindowEndpointUnreachableError,
    ContextWindowUnavailableError,
    FittedLines,
    FittedText,
    PromptAllowance,
    PromptBudgetExceededError,
    ResolvedContextWindow,
    TokenBudget,
    TokenCountUnavailableError,
    combined_allowance,
    cut_notice,
    extract_context_window,
    fetch_context_window,
    fit_lines,
    fit_messages,
    fit_text,
    fitting_demonstrations,
    line_cut_notice,
    litellm_message_counter,
    resolve_context_window,
    served_message_counter,
    shared_served_message_counter,
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
class TestFittingDemonstrations:
    """A compiled program keeps the longest prefix of its demonstrations that
    fits the student's window with every input it will be called with."""

    # The served student: gemma behind --max-model-len 8192 with the
    # primary endpoint's max_tokens=1000 reservation.
    BUDGET = TokenBudget(
        model="openai/google/gemma-4-e4b-it",
        context_window=8192,
        reserved_output=1000,
    )

    @staticmethod
    def _render(demos, call):
        # A 300-word preamble, each demonstration as a user/assistant pair,
        # then the live turn of ``call`` words.
        return [
            {"role": "system", "content": _words(300)},
            *[
                message
                for size in demos
                for message in (
                    {"role": "user", "content": _words(size)},
                    {"role": "assistant", "content": _words(50)},
                )
            ],
            {"role": "user", "content": _words(call)},
        ]

    def test_demonstrations_are_kept_from_the_front_while_they_fit(self):
        # 300 + 4 x (1500 + 50) + 200 = 6700 fits the 7192 input budget;
        # five make 8250.
        kept = fitting_demonstrations(
            [1500] * 16,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=_count_words,
            calls=[200],
        )
        assert kept == 4

    def test_the_longest_call_decides(self):
        # With a 2000-word live turn: 300 + 2 x 1550 + 2000 = 5400, three
        # demonstrations 6950, four 8500 > 7192.
        kept = fitting_demonstrations(
            [1500] * 16,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=_count_words,
            calls=[200, 2000, 50],
        )
        assert kept == 3

    def test_every_demonstration_is_kept_when_all_fit(self):
        kept = fitting_demonstrations(
            [100] * 16,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=_count_words,
            calls=[200],
        )
        assert kept == 16

    def test_a_call_that_overflows_alone_keeps_no_demonstrations(self):
        kept = fitting_demonstrations(
            [10] * 4,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=_count_words,
            calls=[200, 7000],
        )
        assert kept == 0

    def test_no_calls_is_refused(self):
        with pytest.raises(ValueError) as exc:
            fitting_demonstrations(
                [10],
                budget=self.BUDGET,
                render=self._render,
                count_tokens=_count_words,
                calls=[],
            )
        assert str(exc.value) == "fitting demonstrations needs at least one call"

    def test_each_call_is_counted_once_when_every_demonstration_fits(self):
        """A served counter pays a round-trip per count, so a prefix that
        every call fits is confirmed with one count per call."""
        counted = []

        def count(messages):
            counted.append(len(messages))
            return _count_words(messages)

        kept = fitting_demonstrations(
            [100] * 16,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=count,
            calls=list(range(10, 310)),
        )
        assert kept == 16
        assert len(counted) == 300

    def test_a_call_that_lowers_the_prefix_does_not_raise_it_for_later_calls(self):
        # 300 + n x 1550 + call: the 2000-word call keeps three, the
        # 200-word call alone would keep four.
        counted = []

        def count(messages):
            counted.append(len(messages))
            return _count_words(messages)

        kept = fitting_demonstrations(
            [1500] * 16,
            budget=self.BUDGET,
            render=self._render,
            count_tokens=count,
            calls=[200, 2000],
        )
        assert kept == 3
        # 2000 first (longest request): 16, 8, 4, 2, 3 -> 3; then 200 at 3.
        assert counted == [34, 18, 10, 6, 8, 8]


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

        with pytest.raises(ContextWindowEndpointUnreachableError) as exc:
            lm.budget

        assert str(exc.value) == (
            f"openai/Qwen/Qwen3-14B-AWQ at {base_url}/v1 is unreachable and "
            f"declares no context window, so no request budget can be derived"
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


class _TokenizeHandler(BaseHTTPRequestHandler):
    """A ``POST /tokenize`` that counts one token per whitespace-separated
    word of every message, plus four per message for the chat template."""

    status: int = 200
    body: dict | None = None
    received: list = []
    # Statuses answered before ``status``, one per request, in order.
    failures: list = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length))
        type(self).received.append((self.path, request))
        if type(self).failures:
            failed = type(self).failures.pop(0)
            encoded = b"upstream unavailable"
            self.send_response(failed)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        payload = type(self).body
        if payload is None:
            count = sum(
                4 + len(str(message["content"]).split())
                for message in request["messages"]
            )
            payload = {"count": count, "max_model_len": 8192, "tokens": []}
        encoded = json.dumps(payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        return


def _serve_tokenize(
    status: int = 200, body: dict | None = None, failures: list | None = None
):
    received: list = []
    handler = type(
        "Handler",
        (_TokenizeHandler,),
        {
            "status": status,
            "body": body,
            "received": received,
            "failures": list(failures or []),
        },
    )
    server = _ConcurrentServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1", received


class _ConcurrentServer(ThreadingHTTPServer):
    # socketserver's listen backlog of 5 resets connections when the
    # concurrency tests connect eight clients at once on a loaded host.
    request_queue_size = 64


# Past httpx's default 5 s keep-alive, inside the counter's own.
IDLE_PAST_DEFAULT_KEEPALIVE_S = 6.0


def _serve_tokenize_kept_alive(
    *, close_after_response: bool = False, delay_s: float = 0.0
):
    """A ``/tokenize`` speaking HTTP/1.1 that records each connection it
    accepts. ``close_after_response`` drops the connection after answering
    without saying so, as an endpoint closing an idle connection does."""
    received: list = []
    connections: list = []

    class Handler(_TokenizeHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            connections.append(self.client_address)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
            time.sleep(delay_s)
            super().do_POST()
            if close_after_response:
                self.close_connection = True

    Handler.received = received
    Handler.failures = []
    server = _ConcurrentServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return (
        server,
        f"http://127.0.0.1:{server.server_address[1]}/v1",
        received,
        connections,
    )


@pytest.mark.unit
class TestServedMessageCounter:
    """The student is counted by the tokenizer and chat template it serves."""

    MESSAGES = [
        {"role": "system", "content": "Enhance the query."},
        {"role": "user", "content": "servo torque at 1,250 rpm"},
    ]

    def test_counts_through_the_endpoint_s_tokenize_route(self):
        server, api_base, received = _serve_tokenize()
        try:
            count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")
            assert count(self.MESSAGES) == 4 + 3 + 4 + 5
        finally:
            server.shutdown()
        assert received == [
            (
                "/tokenize",
                {"model": "google/gemma-4-e4b-it", "messages": self.MESSAGES},
            )
        ]

    def test_an_unreachable_endpoint_refuses_to_count(self):
        server, api_base, _ = _serve_tokenize()
        server.shutdown()
        server.server_close()
        count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")
        with pytest.raises(TokenCountUnavailableError) as exc:
            count(self.MESSAGES)
        root = api_base.removesuffix("/v1")
        assert str(exc.value) == (
            f"{root}/tokenize is unreachable, so the request cannot be counted "
            "with the tokenizer openai/google/gemma-4-e4b-it is served with"
        )

    def test_an_endpoint_without_the_route_refuses_to_count(self):
        server, api_base, received = _serve_tokenize(
            status=404, body={"detail": "Not Found"}
        )
        slept: list = []
        try:
            count = served_message_counter(
                api_base, "openai/google/gemma-4-e4b-it", retries=1, sleep=slept.append
            )
            with pytest.raises(TokenCountUnavailableError) as exc:
                count(self.MESSAGES)
        finally:
            server.shutdown()
        # A missing route is not transient: it is asked once.
        assert (len(received), slept) == (1, [])
        root = api_base.removesuffix("/v1")
        assert str(exc.value) == (
            f"{root}/tokenize answered HTTP 404, so the request cannot be "
            "counted with the tokenizer openai/google/gemma-4-e4b-it is served with"
        )

    def test_an_answer_without_a_count_refuses_to_count(self):
        server, api_base, _ = _serve_tokenize(body={"tokens": [1, 2, 3]})
        try:
            count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")
            with pytest.raises(TokenCountUnavailableError) as exc:
                count(self.MESSAGES)
        finally:
            server.shutdown()
        root = api_base.removesuffix("/v1")
        assert str(exc.value) == (
            f"{root}/tokenize answered without an integer count, so the request "
            "cannot be counted with the tokenizer openai/google/gemma-4-e4b-it "
            "is served with"
        )

    def test_a_transient_gateway_failure_is_retried(self):
        """One count of hundreds meeting a 502 from the endpoint's gateway is
        asked again after a pause, as the LM's own calls are."""
        server, api_base, received = _serve_tokenize(failures=[502])
        slept: list = []
        try:
            count = served_message_counter(
                api_base, "openai/google/gemma-4-e4b-it", retries=1, sleep=slept.append
            )
            assert count(self.MESSAGES) == 4 + 3 + 4 + 5
        finally:
            server.shutdown()
        assert (len(received), slept) == (2, [1.0])

    def test_a_failure_that_outlasts_the_retries_refuses_to_count(self):
        server, api_base, received = _serve_tokenize(failures=[503, 502, 502])
        slept: list = []
        try:
            count = served_message_counter(
                api_base, "openai/google/gemma-4-e4b-it", retries=2, sleep=slept.append
            )
            with pytest.raises(TokenCountUnavailableError) as exc:
                count(self.MESSAGES)
        finally:
            server.shutdown()
        assert (len(received), slept) == (3, [1.0, 2.0])
        root = api_base.removesuffix("/v1")
        assert str(exc.value) == (
            f"{root}/tokenize answered HTTP 502, so the request cannot be "
            "counted with the tokenizer openai/google/gemma-4-e4b-it is served with"
        )

    def test_concurrent_counts_each_get_their_own_request_s_count(self):
        server, api_base, received = _serve_tokenize()
        count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")
        threads = 16
        barrier = threading.Barrier(threads)
        results: dict[int, int] = {}

        def worker(index: int) -> None:
            messages = [{"role": "user", "content": _words(index + 1)}]
            barrier.wait()
            results[index] = count(messages)

        try:
            pool = [
                threading.Thread(target=worker, args=(index,))
                for index in range(threads)
            ]
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join()
        finally:
            server.shutdown()
        assert results == {index: 4 + index + 1 for index in range(threads)}
        assert len(received) == threads

    def test_a_count_after_an_idle_pause_reuses_the_connection(self):
        server, api_base, received, connections = _serve_tokenize_kept_alive()
        try:
            count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")
            first = count(self.MESSAGES)
            time.sleep(IDLE_PAST_DEFAULT_KEEPALIVE_S)
            second = count(self.MESSAGES)
        finally:
            server.shutdown()
        assert (first, second, len(received), len(connections)) == (16, 16, 2, 1)

    def test_a_connection_the_endpoint_closed_is_replaced_without_a_failed_count(
        self,
    ):
        server, api_base, received, connections = _serve_tokenize_kept_alive(
            close_after_response=True
        )
        slept: list = []
        try:
            count = served_message_counter(
                api_base, "openai/google/gemma-4-e4b-it", retries=0, sleep=slept.append
            )
            first = count(self.MESSAGES)
            time.sleep(0.5)
            second = count(self.MESSAGES)
        finally:
            server.shutdown()
        assert (first, second, len(received), len(connections), slept) == (
            16,
            16,
            2,
            2,
            [],
        )

    def test_concurrent_counts_after_an_idle_pause_reuse_their_connections(self):
        threads = 16
        server, api_base, received, connections = _serve_tokenize_kept_alive(
            delay_s=0.3
        )
        count = served_message_counter(api_base, "openai/google/gemma-4-e4b-it")

        def round_of_counts() -> dict[int, int]:
            barrier = threading.Barrier(threads)
            results: dict[int, int] = {}

            def worker(index: int) -> None:
                messages = [{"role": "user", "content": _words(index + 1)}]
                barrier.wait()
                results[index] = count(messages)

            pool = [
                threading.Thread(target=worker, args=(index,))
                for index in range(threads)
            ]
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join()
            return results

        try:
            first = round_of_counts()
            opened = len(connections)
            time.sleep(IDLE_PAST_DEFAULT_KEEPALIVE_S)
            second = round_of_counts()
        finally:
            server.shutdown()
        expected = {index: 4 + index + 1 for index in range(threads)}
        assert (first, second) == (expected, expected)
        assert (opened, len(connections), len(received)) == (
            threads,
            threads,
            2 * threads,
        )


def _allowance(api_base: str, *, window: int, reserved: int, margin: int = 0):
    return PromptAllowance(
        budget=TokenBudget(
            model=STUDENT_MODEL, context_window=window, reserved_output=reserved
        ),
        count_tokens=served_message_counter(api_base, STUDENT_MODEL, retries=0),
        margin=margin,
    )


def _fixed_words(count: int) -> str:
    """``count`` five-character words: ``text[:k]`` holds ceil(k / 5) of them."""

    return "word " * count


STUDENT_MODEL = "openai/google/gemma-4-e4b-it"
# The stub counts four template tokens for the message and one per word; the
# cut notice is fifteen words.
NOTICE_WORDS = 15


@pytest.mark.unit
class TestFitTextToTheServedWindow:
    """A free-text prompt has no demonstrations to drop: its tail is cut."""

    def test_the_notice_names_what_was_kept(self):
        assert cut_notice(405, 2000) == (
            "\n\n[Cut to fit the model's context window: the first 405 of "
            "2000 characters are shown.]"
        )
        assert len(cut_notice(405, 2000).split()) == NOTICE_WORDS

    def test_a_text_small_enough_by_its_bytes_is_admitted_without_a_count(self):
        server, api_base, received = _serve_tokenize()
        text = "summarize the evidence"
        try:
            fitted = fit_text(
                text, allowance=_allowance(api_base, window=200, reserved=100)
            )
        finally:
            server.shutdown()
        assert fitted == FittedText(
            text=text, kept_chars=22, total_chars=22, input_tokens=None
        )
        assert fitted.text is text
        assert SINGLE_TURN_TEMPLATE_TOKENS == 32
        assert received == []

    def test_a_text_within_the_allowance_is_counted_once_and_sent_unchanged(self):
        server, api_base, received = _serve_tokenize()
        text = _fixed_words(90)
        try:
            fitted = fit_text(
                text, allowance=_allowance(api_base, window=200, reserved=100)
            )
        finally:
            server.shutdown()
        assert fitted == FittedText(
            text=text, kept_chars=450, total_chars=450, input_tokens=94
        )
        assert fitted.text is text
        assert received == [
            (
                "/tokenize",
                {
                    "model": "google/gemma-4-e4b-it",
                    "messages": [{"role": "user", "content": text}],
                },
            )
        ]

    def test_an_overflowing_text_keeps_its_longest_leading_run_that_fits(self):
        server, api_base, received = _serve_tokenize()
        text = _fixed_words(400)
        allowance = _allowance(api_base, window=200, reserved=100)
        try:
            fitted = fit_text(text, allowance=allowance)
            fit_requests = len(received)
            recount = allowance.count_tokens([{"role": "user", "content": fitted.text}])
            one_more = allowance.count_tokens(
                [{"role": "user", "content": text[:406] + cut_notice(406, 2000)}]
            )
        finally:
            server.shutdown()
        # 4 template + ceil(k / 5) words + 15 notice words <= 100  =>  k <= 405.
        assert fitted == FittedText(
            text=text[:405] + cut_notice(405, 2000),
            kept_chars=405,
            total_chars=2000,
            input_tokens=100,
        )
        assert recount == 100
        assert one_more == 101
        # One count of the whole text, then a bisection over 2000 lengths.
        assert fit_requests == 11

    def test_the_margin_is_held_back_from_the_allowance(self):
        server, api_base, _ = _serve_tokenize()
        text = _fixed_words(400)
        try:
            fitted = fit_text(
                text,
                allowance=_allowance(api_base, window=200, reserved=100, margin=10),
            )
        finally:
            server.shutdown()
        assert fitted == FittedText(
            text=text[:355] + cut_notice(355, 2000),
            kept_chars=355,
            total_chars=2000,
            input_tokens=90,
        )

    def test_an_allowance_smaller_than_the_notice_names_the_budget(self):
        server, api_base, _ = _serve_tokenize()
        try:
            with pytest.raises(PromptBudgetExceededError) as exc:
                fit_text(
                    _fixed_words(400),
                    allowance=_allowance(api_base, window=110, reserved=100),
                )
        finally:
            server.shutdown()
        assert str(exc.value) == (
            f"{STUDENT_MODEL} prompt does not fit the served context window: "
            "context_window=110 reserved_output=100 input_budget=10 "
            "input_tokens=19 after dropping all 0 few-shot demonstrations"
        )

    def test_an_unreachable_tokenizer_refuses_to_bound_an_uncounted_text(self):
        server, api_base, _ = _serve_tokenize()
        server.shutdown()
        server.server_close()
        root = api_base.removesuffix("/v1")
        allowance = _allowance(api_base, window=200, reserved=100)
        with pytest.raises(TokenCountUnavailableError) as exc:
            fit_text(_fixed_words(400), allowance=allowance)
        assert str(exc.value) == (
            f"{root}/tokenize is unreachable, so the request cannot be counted "
            f"with the tokenizer {STUDENT_MODEL} is served with"
        )
        # A text admitted by its bytes needs no count, so it still passes.
        assert fit_text("ok", allowance=allowance) == FittedText(
            text="ok", kept_chars=2, total_chars=2, input_tokens=None
        )

    def test_concurrent_fits_each_keep_their_own_text(self):
        server, api_base, _ = _serve_tokenize()
        allowance = _allowance(api_base, window=200, reserved=100)
        threads = 8
        barrier = threading.Barrier(threads)
        results: dict[int, FittedText] = {}

        def worker(index: int) -> None:
            text = f"w{index}xx " * 400
            barrier.wait()
            results[index] = fit_text(text, allowance=allowance)

        try:
            pool = [
                threading.Thread(target=worker, args=(index,))
                for index in range(threads)
            ]
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join()
        finally:
            server.shutdown()
        assert results == {
            index: FittedText(
                text=(f"w{index}xx " * 400)[:405] + cut_notice(405, 2000),
                kept_chars=405,
                total_chars=2000,
                input_tokens=100,
            )
            for index in range(threads)
        }


@pytest.mark.unit
class TestSharedServedMessageCounter:
    def test_eight_first_touches_build_one_counter(self, monkeypatch):
        from cogniverse_foundation.config import token_budget

        built: list[tuple] = []
        build = token_budget.served_message_counter

        def counting(api_base, model, **kwargs):
            built.append((api_base, model, kwargs))
            return build(api_base, model, **kwargs)

        monkeypatch.setattr(token_budget, "served_message_counter", counting)
        monkeypatch.setattr(token_budget, "_COUNTER_MEMO", {})
        api_base = "http://127.0.0.1:9/v1"
        threads = 8
        barrier = threading.Barrier(threads)
        counters: list = []

        def worker() -> None:
            barrier.wait()
            counters.append(
                shared_served_message_counter(api_base, STUDENT_MODEL, retries=2)
            )

        pool = [threading.Thread(target=worker) for _ in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
        assert built == [
            (
                api_base,
                STUDENT_MODEL,
                {"retries": 2, "timeout_seconds": TOKENIZE_TIMEOUT_SECONDS},
            )
        ]
        assert TOKENIZE_TIMEOUT_SECONDS == 30.0
        assert len(counters) == threads
        assert {id(counter) for counter in counters} == {id(counters[0])}

    def test_a_different_endpoint_gets_its_own_counter(self, monkeypatch):
        from cogniverse_foundation.config import token_budget

        monkeypatch.setattr(token_budget, "_COUNTER_MEMO", {})
        first = shared_served_message_counter("http://127.0.0.1:9/v1", STUDENT_MODEL)
        again = shared_served_message_counter("http://127.0.0.1:9/v1/", STUDENT_MODEL)
        other = shared_served_message_counter("http://127.0.0.1:10/v1", STUDENT_MODEL)
        assert again is first
        assert other is not first


def _word_counter(per_word: int = 1, calls: list | None = None):
    """Four tokens per message plus ``per_word`` per whitespace-separated word."""

    def count(messages):
        if calls is not None:
            calls.append(messages)
        return sum(
            4 + per_word * len(str(message["content"]).split()) for message in messages
        )

    return count


def _words_allowance(tokens: int, *, per_word: int = 1, calls=None, margin=0):
    return PromptAllowance(
        budget=TokenBudget(
            model=STUDENT_MODEL,
            context_window=tokens + 200 + margin,
            reserved_output=200,
        ),
        count_tokens=_word_counter(per_word, calls),
        margin=margin,
    )


def _summary_request(text: str) -> list[dict]:
    # 4 + 3 words for the system message, 4 + words of the text for the user.
    return [
        {"role": "system", "content": "Summarize the results."},
        {"role": "user", "content": text},
    ]


def _hit_line(index: int, words: int = 30, tag: str = "h") -> str:
    """``words`` five-character words: ``line[:6 * w]`` holds exactly ``w``."""

    return " ".join(f"{tag}{index}w{word:02d}" for word in range(words))


HEAD = ["Total Results: 4"]
TAIL = ["Key themes: video"]
# 7 (system) + 4 (user) + 3 (head) + 3 (tail) before any hit line.
FIXED_TOKENS = 17
LINE_NOTICE_WORDS = 15


@pytest.mark.unit
class TestFitLinesToTheAllowance:
    """A prompt field of lines keeps the most important ones that fit."""

    LINES = [_hit_line(index) for index in range(4)]

    def test_the_line_notice_names_what_was_kept(self):
        assert line_cut_notice(48, 179) == (
            " [cut to fit the model's context window: the first 48 of 179 "
            "characters are shown]"
        )
        assert len(line_cut_notice(48, 179).split()) == LINE_NOTICE_WORDS
        assert len(self.LINES[0]) == 179

    def test_every_line_that_fits_is_sent_after_one_count(self):
        calls: list = []
        fitted = fit_lines(
            self.LINES,
            allowance=_words_allowance(FIXED_TOKENS + 120, calls=calls),
            render=_summary_request,
            head=HEAD,
            tail=TAIL,
        )

        text = "\n".join([*HEAD, *self.LINES, *TAIL])
        assert fitted == FittedLines(
            text=text,
            kept_lines=4,
            cut_chars=None,
            total_lines=4,
            input_tokens=FIXED_TOKENS + 120,
        )
        assert calls == [_summary_request(text)]

    def test_an_overflow_keeps_the_leading_lines_and_cuts_the_next(self):
        # 100 tokens: two whole lines (77), then eight words of the third
        # with the fifteen-word notice reach exactly 100.
        fitted = fit_lines(
            self.LINES,
            allowance=_words_allowance(100),
            render=_summary_request,
            head=HEAD,
            tail=TAIL,
        )

        assert fitted == FittedLines(
            text="\n".join(
                [
                    *HEAD,
                    self.LINES[0],
                    self.LINES[1],
                    self.LINES[2][:48] + line_cut_notice(48, 179),
                    *TAIL,
                ]
            ),
            kept_lines=2,
            cut_chars=48,
            total_lines=4,
            input_tokens=100,
        )
        assert self.LINES[2][:48] == (
            "h2w00 h2w01 h2w02 h2w03 h2w04 h2w05 h2w06 h2w07 "
        )

    def test_a_line_too_long_to_fit_whole_is_cut_even_when_it_is_the_first(self):
        fitted = fit_lines(
            self.LINES,
            allowance=_words_allowance(FIXED_TOKENS + LINE_NOTICE_WORDS + 5),
            render=_summary_request,
            head=HEAD,
            tail=TAIL,
        )

        assert fitted == FittedLines(
            text="\n".join(
                [*HEAD, self.LINES[0][:30] + line_cut_notice(30, 179), *TAIL]
            ),
            kept_lines=0,
            cut_chars=30,
            total_lines=4,
            input_tokens=FIXED_TOKENS + LINE_NOTICE_WORDS + 5,
        )

    def test_a_next_line_with_no_room_for_its_notice_is_left_out(self):
        fitted = fit_lines(
            self.LINES,
            allowance=_words_allowance(FIXED_TOKENS + 60 + LINE_NOTICE_WORDS - 1),
            render=_summary_request,
            head=HEAD,
            tail=TAIL,
        )

        assert fitted == FittedLines(
            text="\n".join([*HEAD, *self.LINES[:2], *TAIL]),
            kept_lines=2,
            cut_chars=None,
            total_lines=4,
            input_tokens=FIXED_TOKENS + 60,
        )

    def test_a_head_and_tail_that_overflow_alone_name_the_budget(self):
        allowance = _words_allowance(FIXED_TOKENS - 1)

        with pytest.raises(PromptBudgetExceededError) as raised:
            fit_lines(
                self.LINES,
                allowance=allowance,
                render=_summary_request,
                head=HEAD,
                tail=TAIL,
            )
        assert (raised.value.input_tokens, raised.value.context_window) == (
            FIXED_TOKENS,
            FIXED_TOKENS - 1 + 200,
        )

    def test_a_counter_that_cannot_count_stops_the_fit(self):
        def unavailable(messages):
            raise TokenCountUnavailableError("tokenize is down")

        allowance = PromptAllowance(
            budget=TokenBudget(
                model=STUDENT_MODEL, context_window=300, reserved_output=200
            ),
            count_tokens=unavailable,
        )
        with pytest.raises(TokenCountUnavailableError, match="tokenize is down"):
            fit_lines(self.LINES, allowance=allowance, render=_summary_request)

    def test_concurrent_fits_each_keep_their_own_lines(self):
        threads = 8
        barrier = threading.Barrier(threads)
        corpora = {
            index: [_hit_line(line, tag="abcdefgh"[index]) for line in range(4)]
            for index in range(threads)
        }
        results: dict[int, FittedLines] = {}
        allowance = _words_allowance(100)

        def worker(index: int) -> None:
            barrier.wait()
            results[index] = fit_lines(
                corpora[index],
                allowance=allowance,
                render=_summary_request,
                head=HEAD,
                tail=TAIL,
            )

        pool = [
            threading.Thread(target=worker, args=(index,)) for index in range(threads)
        ]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

        assert {index: fitted.text for index, fitted in results.items()} == {
            index: "\n".join(
                [
                    *HEAD,
                    *lines[:2],
                    lines[2][:48] + line_cut_notice(48, 179),
                    *TAIL,
                ]
            )
            for index, lines in corpora.items()
        }


@pytest.mark.unit
class TestCombinedAllowance:
    """A request any of several endpoints may serve fits every one of them."""

    MESSAGES_20 = [{"role": "user", "content": "word " * 20}]
    MESSAGES_30 = [{"role": "user", "content": "word " * 30}]

    def test_a_request_within_every_allowance_is_within_the_combined_one(self):
        wide = _words_allowance(100)
        narrow = _words_allowance(50, per_word=2)
        combined = combined_allowance([wide, narrow])

        assert combined.budget is narrow.budget
        assert combined.tokens == 50
        # wide: 24 of 100, narrow: 44 of 50 -> 6 under the narrow one.
        assert combined.count_tokens(self.MESSAGES_20) == 44

    def test_an_overrun_of_either_allowance_is_an_overrun_of_the_combined_one(self):
        # The wider allowance counts with the costlier tokenizer.
        wide = _words_allowance(70, per_word=2)
        narrow = _words_allowance(50)
        combined = combined_allowance([wide, narrow])

        assert combined.tokens == 50
        # wide: 64 of 70 (6 under), narrow: 34 of 50 (16 under).
        assert combined.count_tokens(self.MESSAGES_30) == 44
        # 40 words: wide 84 of 70 (14 over), narrow 44 of 50.
        assert combined.count_tokens([{"role": "user", "content": "w " * 40}]) == 64

    def test_one_allowance_is_its_own_combination(self):
        only = _words_allowance(100)
        assert combined_allowance([only]) is only

    def test_no_allowance_is_refused(self):
        with pytest.raises(ValueError, match="at least one allowance"):
            combined_allowance([])

    def test_an_endpoint_that_cannot_count_stops_the_combined_count(self):
        def unavailable(messages):
            raise TokenCountUnavailableError("the teacher cannot count")

        teacher = PromptAllowance(
            budget=TokenBudget(
                model="openai/Qwen/Qwen3-14B-AWQ",
                context_window=4096,
                reserved_output=1000,
            ),
            count_tokens=unavailable,
        )
        combined = combined_allowance([_words_allowance(7000), teacher])
        with pytest.raises(TokenCountUnavailableError, match="cannot count"):
            combined.count_tokens(self.MESSAGES_20)
