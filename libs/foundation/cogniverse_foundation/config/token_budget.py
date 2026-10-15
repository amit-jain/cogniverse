"""Fit an assembled chat request inside the context window its endpoint serves.

A chat request costs ``prompt tokens + max_tokens`` against the model's
context window. ``max_tokens`` is the completion an endpoint reserves; the
remainder is everything the adapter may send. This module reads the window
the endpoint reports for the model it serves, derives that remainder, and
fits the assembled messages into it by dropping whole few-shot
demonstrations. A request that still overflows with no demonstrations left
raises with the window, the reservation, the measured input and the number
of demonstrations dropped. ``fitting_demonstrations`` sizes a compiled
program's demonstrations to the window before the program is sent at all.
``fit_text`` bounds one free-text prompt, which has no demonstrations to drop,
by cutting its tail and saying so in the prompt. ``fit_lines`` keeps the
leading lines of a prompt field that fit, most important first.
``combined_allowance`` is the allowance of a request any of several endpoints
may serve.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import httpx

from cogniverse_foundation.config.inference_auth import (
    endpoint_root,
    inference_headers,
)

logger = logging.getLogger(__name__)

Messages = list[dict[str, Any]]
TokenCounter = Callable[[Messages], int]

CONTEXT_WINDOW_TIMEOUT_SECONDS = 5.0
TOKENIZE_TIMEOUT_SECONDS = 30.0

# A window is a property of a deployment, so it is read once per
# (endpoint, model) and held for the process. Ingestion builds a fresh LM per
# segment; without this every segment would pay a listing round-trip.
_MAX_MEMOIZED_WINDOWS = 32
_WINDOW_MEMO: "dict[tuple[str, str], ResolvedContextWindow]" = {}
_WINDOW_MEMO_LOCK = threading.Lock()


# One counter, and so one HTTP client, per (endpoint, model, retries, timeout).
_MAX_MEMOIZED_COUNTERS = 32
_COUNTER_MEMO: "dict[tuple[str, str, int, float], TokenCounter]" = {}
_COUNTER_MEMO_LOCK = threading.Lock()

# Tokens a served chat template adds around one user message, at most.
SINGLE_TURN_TEMPLATE_TOKENS = 32


def clear_context_window_memo() -> None:
    """Drop every memoized window so the next resolve re-reads its endpoint."""

    with _WINDOW_MEMO_LOCK:
        _WINDOW_MEMO.clear()


class ContextWindowUnavailableError(RuntimeError):
    """The endpoint does not report a context window for the model it serves."""


class TokenCountUnavailableError(RuntimeError):
    """The endpoint cannot count a request with the tokenizer it serves."""


class PromptBudgetExceededError(RuntimeError):
    """The prompt overflows the window with every demonstration dropped."""

    def __init__(
        self,
        *,
        model: str,
        context_window: int,
        reserved_output: int,
        input_tokens: int,
        dropped_demos: int,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self.reserved_output = reserved_output
        self.input_tokens = input_tokens
        self.dropped_demos = dropped_demos
        super().__init__(
            f"{model} prompt does not fit the served context window: "
            f"context_window={context_window} reserved_output={reserved_output} "
            f"input_budget={context_window - reserved_output} "
            f"input_tokens={input_tokens} after dropping all {dropped_demos} "
            f"few-shot demonstrations"
        )


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """The input allowance left by a reserved completion in a served window."""

    model: str
    context_window: int
    reserved_output: int

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError(
                f"{self.model} context_window must be positive, "
                f"got {self.context_window}"
            )
        if self.reserved_output <= 0:
            raise ValueError(
                f"{self.model} reserved_output must be positive, "
                f"got {self.reserved_output}"
            )
        if self.reserved_output >= self.context_window:
            raise ValueError(
                f"{self.model} reserves {self.reserved_output} output tokens of a "
                f"{self.context_window}-token context window, leaving no room for "
                f"input"
            )

    @property
    def input_budget(self) -> int:
        """Tokens the prompt may occupy once the completion is reserved."""

        return self.context_window - self.reserved_output


@dataclass(frozen=True, slots=True)
class FittedPrompt:
    """Messages that fit the budget, and what it cost to get there."""

    messages: Messages
    input_tokens: int
    dropped_demos: int
    total_demos: int


def split_demonstrations(
    messages: Messages,
) -> tuple[Messages, list[Messages], Messages]:
    """Split ``messages`` into (preamble, demonstration turns, live turn).

    Chat adapters emit a system preamble, one user/assistant pair per
    few-shot demonstration, then the turn being answered. A demonstration
    turn starts at a ``user`` message and runs until the next one.
    """

    if not messages:
        raise ValueError("messages must not be empty")

    preamble_end = 0
    while (
        preamble_end < len(messages) - 1
        and messages[preamble_end].get("role") == "system"
    ):
        preamble_end += 1

    preamble = messages[:preamble_end]
    live = messages[-1:]
    middle = messages[preamble_end:-1]

    demos: list[Messages] = []
    for message in middle:
        if message.get("role") == "user" or not demos:
            demos.append([message])
        else:
            demos[-1].append(message)
    return preamble, demos, live


def fit_messages(
    messages: Messages,
    *,
    budget: TokenBudget,
    count_tokens: TokenCounter,
) -> FittedPrompt:
    """Drop whole demonstrations, oldest first, until the prompt fits ``budget``.

    Raises ``PromptBudgetExceededError`` when the preamble and the live turn
    alone overflow the input allowance: there is nothing left to drop and a
    truncated prompt would silently change the request.
    """

    preamble, demos, live = split_demonstrations(messages)
    kept = list(demos)
    while True:
        candidate = [*preamble, *[m for demo in kept for m in demo], *live]
        input_tokens = count_tokens(candidate)
        if input_tokens <= budget.input_budget:
            return FittedPrompt(
                messages=candidate,
                input_tokens=input_tokens,
                dropped_demos=len(demos) - len(kept),
                total_demos=len(demos),
            )
        if not kept:
            raise PromptBudgetExceededError(
                model=budget.model,
                context_window=budget.context_window,
                reserved_output=budget.reserved_output,
                input_tokens=input_tokens,
                dropped_demos=len(demos),
            )
        kept.pop(0)


def litellm_message_counter(model: str) -> TokenCounter:
    """Count chat messages with the tokenizer litellm bills the request by."""

    import litellm

    def count(messages: Messages) -> int:
        return int(litellm.token_counter(model=model, messages=messages))

    return count


def served_message_counter(
    api_base: str,
    model: str,
    *,
    retries: int = 1,
    timeout_seconds: float = TOKENIZE_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> TokenCounter:
    """Count chat messages with the tokenizer and chat template ``model`` is
    served with, through the endpoint's ``POST /tokenize``.

    litellm counts a model it has no tokenizer for with OpenAI's
    ``cl100k_base``. For the gemma student that undercounts transcript text
    by about a quarter, so a request sized to the window by litellm's count is
    refused. vLLM's ``/tokenize`` renders the messages with the served chat
    template and counts them with the served tokenizer: its ``count`` is the
    ``prompt_tokens`` the completion reports. A count is idempotent and a
    fitting pass asks hundreds, so an unreachable endpoint or a 5xx answer is
    asked again up to ``retries`` times, after 1 s, 2 s, 4 s ... (the
    endpoint's ``num_retries``, as its LM calls are retried). Raises
    ``TokenCountUnavailableError`` when the endpoint still cannot answer, or
    answers 4xx or without an integer count; a count by another tokenizer is
    the defect this counter exists to close.
    """

    from cogniverse_foundation.dspy.model_format import bare_model_name

    root = endpoint_root(api_base)
    url = f"{root}/tokenize"
    served = bare_model_name(model)
    unavailable = (
        f"so the request cannot be counted with the tokenizer {model} is served with"
    )
    http = httpx.Client(headers=dict(inference_headers(root)), timeout=timeout_seconds)

    def ask(messages: Messages) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = http.post(url, json={"model": served, "messages": messages})
            except httpx.HTTPError as exc:
                if attempt == retries:
                    raise TokenCountUnavailableError(
                        f"{url} is unreachable, {unavailable}"
                    ) from exc
            else:
                if response.status_code < 500 or attempt == retries:
                    return response
            attempt += 1
            sleep(float(2 ** (attempt - 1)))

    def count(messages: Messages) -> int:
        response = ask(messages)
        if response.status_code != 200:
            raise TokenCountUnavailableError(
                f"{url} answered HTTP {response.status_code}, {unavailable}"
            )
        try:
            tokens = response.json().get("count")
        except (ValueError, AttributeError):
            tokens = None
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TokenCountUnavailableError(
                f"{url} answered without an integer count, {unavailable}"
            )
        return tokens

    return count


def shared_served_message_counter(
    api_base: str,
    model: str,
    *,
    retries: int = 1,
    timeout_seconds: float = TOKENIZE_TIMEOUT_SECONDS,
) -> TokenCounter:
    """The process's ``served_message_counter`` for an endpoint and model.

    Built once per ``(api_base, model, retries, timeout_seconds)`` and shared
    by every caller, so a serving path that counts on each request reuses one
    HTTP client.
    """

    key = (api_base.rstrip("/"), model, retries, timeout_seconds)
    with _COUNTER_MEMO_LOCK:
        counter = _COUNTER_MEMO.get(key)
        if counter is None:
            counter = served_message_counter(
                api_base, model, retries=retries, timeout_seconds=timeout_seconds
            )
            if len(_COUNTER_MEMO) >= _MAX_MEMOIZED_COUNTERS:
                _COUNTER_MEMO.pop(next(iter(_COUNTER_MEMO)))
            _COUNTER_MEMO[key] = counter
        return counter


@dataclass(frozen=True, slots=True)
class PromptAllowance:
    """A budget, the counter that measures against it, and tokens held back."""

    budget: TokenBudget
    count_tokens: TokenCounter
    margin: int = 0

    @property
    def tokens(self) -> int:
        """Tokens a prompt may occupy."""

        return self.budget.input_budget - self.margin


@dataclass(frozen=True, slots=True)
class FittedText:
    """A prompt text that fits its allowance, and how much of it was kept.

    ``input_tokens`` is ``None`` when the text was admitted by its size alone.
    """

    text: str
    kept_chars: int
    total_chars: int
    input_tokens: int | None


def cut_notice(kept_chars: int, total_chars: int) -> str:
    """The line ``fit_text`` ends a cut prompt with."""

    return (
        f"\n\n[Cut to fit the model's context window: the first {kept_chars} "
        f"of {total_chars} characters are shown.]"
    )


def fit_text(text: str, *, allowance: PromptAllowance) -> FittedText:
    """Bound one user prompt to ``allowance`` by cutting its tail.

    A text that fits is returned unchanged. One whose UTF-8 bytes plus
    ``SINGLE_TURN_TEMPLATE_TOKENS`` are within the allowance is admitted
    without a count, since a token spans at least one byte. Otherwise the
    text is counted as a single user message, and one over the allowance is
    cut to its longest leading run of characters that fits together with
    ``cut_notice``, found by bisection, each probe counted the same way.

    Raises ``PromptBudgetExceededError`` when the notice alone overflows, and
    whatever ``allowance.count_tokens`` raises when it cannot count.
    """

    limit = allowance.tokens
    total = len(text)
    if len(text.encode("utf-8")) + SINGLE_TURN_TEMPLATE_TOKENS <= limit:
        return FittedText(
            text=text, kept_chars=total, total_chars=total, input_tokens=None
        )

    def count(candidate: str) -> int:
        return allowance.count_tokens([{"role": "user", "content": candidate}])

    tokens = count(text)
    if tokens <= limit:
        return FittedText(
            text=text, kept_chars=total, total_chars=total, input_tokens=tokens
        )

    def cut(kept: int) -> str:
        return text[:kept] + cut_notice(kept, total)

    fitted: tuple[int, int] | None = None
    low, high = 0, total - 1
    while low <= high:
        middle = (low + high) // 2
        tokens = count(cut(middle))
        if tokens <= limit:
            fitted = (middle, tokens)
            low = middle + 1
        else:
            high = middle - 1
    if fitted is None:
        raise PromptBudgetExceededError(
            model=allowance.budget.model,
            context_window=allowance.budget.context_window,
            reserved_output=allowance.budget.reserved_output,
            input_tokens=tokens,
            dropped_demos=0,
        )
    kept, tokens = fitted
    return FittedText(
        text=cut(kept), kept_chars=kept, total_chars=total, input_tokens=tokens
    )


def combined_allowance(allowances: Sequence[PromptAllowance]) -> PromptAllowance:
    """One allowance a request meets exactly when it meets each of ``allowances``.

    A request any of several endpoints may serve has to fit every one of their
    windows, each counted with its own tokenizer. The result carries the
    budget and margin of the allowance with the fewest tokens; its counter
    asks every endpoint and reports the largest overrun of its own allowance,
    shifted onto that one, so a count is within ``tokens`` exactly when it is
    within every endpoint's.
    """

    if not allowances:
        raise ValueError("a combined allowance needs at least one allowance")
    if len(allowances) == 1:
        return allowances[0]
    binding = min(allowances, key=lambda allowance: allowance.tokens)

    def count(messages: Messages) -> int:
        return binding.tokens + max(
            allowance.count_tokens(messages) - allowance.tokens
            for allowance in allowances
        )

    return PromptAllowance(
        budget=binding.budget, count_tokens=count, margin=binding.margin
    )


@dataclass(frozen=True, slots=True)
class FittedLines:
    """A text of whole lines that fits its allowance, and what it kept.

    ``kept_lines`` lines are whole; ``cut_chars`` is how much of the line
    after them was kept, ``None`` when none was.
    """

    text: str
    kept_lines: int
    cut_chars: int | None
    total_lines: int
    input_tokens: int


def line_cut_notice(kept_chars: int, total_chars: int) -> str:
    """What ``fit_lines`` appends to the line it cuts."""

    return (
        f" [cut to fit the model's context window: the first {kept_chars} of "
        f"{total_chars} characters are shown]"
    )


def fit_lines(
    lines: Sequence[str],
    *,
    allowance: PromptAllowance,
    render: Callable[[str], Messages],
    head: Sequence[str] = (),
    tail: Sequence[str] = (),
) -> FittedLines:
    """The leading ``lines`` that fit ``allowance`` in the request ``render``
    builds around them.

    ``render(text)`` is the request a text is sent in, counted with
    ``allowance.count_tokens``. The text is ``head``, the kept lines, then
    ``tail``, joined by newlines; ``lines`` come most important first. When
    every line fits, the text is returned after one count. Otherwise the
    longest leading run of whole lines that fits is found by bisection, and
    the line after it is cut to its longest leading run of characters that
    fits together with ``line_cut_notice``, also by bisection; it is left out
    when no character of it fits.

    Raises ``PromptBudgetExceededError`` when ``head`` and ``tail`` alone
    overflow, and whatever ``allowance.count_tokens`` raises when it cannot
    count.
    """

    limit = allowance.tokens
    total = len(lines)

    def compose(kept: int, cut: int | None = None) -> str:
        parts = [*head, *lines[:kept]]
        if cut is not None:
            line = lines[kept]
            parts.append(line[:cut] + line_cut_notice(cut, len(line)))
        parts.extend(tail)
        return "\n".join(parts)

    def count(text: str) -> int:
        return allowance.count_tokens(render(text))

    tokens = count(compose(total))
    if tokens <= limit:
        return FittedLines(
            text=compose(total),
            kept_lines=total,
            cut_chars=None,
            total_lines=total,
            input_tokens=tokens,
        )

    whole: tuple[int, int] | None = None
    low, high = 0, total - 1
    while low <= high:
        middle = (low + high) // 2
        tokens = count(compose(middle))
        if tokens <= limit:
            whole = (middle, tokens)
            low = middle + 1
        else:
            high = middle - 1
    if whole is None:
        raise PromptBudgetExceededError(
            model=allowance.budget.model,
            context_window=allowance.budget.context_window,
            reserved_output=allowance.budget.reserved_output,
            input_tokens=tokens,
            dropped_demos=0,
        )
    kept, tokens = whole

    cut: int | None = None
    low, high = 1, len(lines[kept]) - 1
    while low <= high:
        middle = (low + high) // 2
        probed = count(compose(kept, middle))
        if probed <= limit:
            cut, tokens = middle, probed
            low = middle + 1
        else:
            high = middle - 1
    return FittedLines(
        text=compose(kept, cut),
        kept_lines=kept,
        cut_chars=cut,
        total_lines=total,
        input_tokens=tokens,
    )


def extract_context_window(body: Any) -> int | None:
    """The ``max_model_len`` an OpenAI-compatible ``/v1/models`` body reports."""

    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        window = entry.get("max_model_len")
        if isinstance(window, bool) or not isinstance(window, int):
            continue
        if window > 0:
            return window
    return None


def fetch_context_window(
    api_base: str,
    *,
    client: httpx.Client | None = None,
    timeout_seconds: float = CONTEXT_WINDOW_TIMEOUT_SECONDS,
) -> int:
    """The context window the endpoint at ``api_base`` serves.

    Raises ``ContextWindowUnavailableError`` when the endpoint is unreachable
    or answers without ``max_model_len``: a budget guessed for an endpoint
    that never declared its window is the defect this module exists to close.
    """

    url = f"{api_base.rstrip('/')}/models"
    headers = dict(inference_headers(endpoint_root(api_base)))
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http.get(url, headers=headers, timeout=timeout_seconds)
    except httpx.HTTPError as exc:
        raise ContextWindowUnavailableError(
            f"{url} is unreachable, so the served context window is unknown and "
            f"no request budget can be derived"
        ) from exc
    finally:
        if owns_client:
            http.close()

    if response.status_code != 200:
        raise ContextWindowUnavailableError(
            f"{url} answered HTTP {response.status_code}, so the served context "
            f"window is unknown and no request budget can be derived"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ContextWindowUnavailableError(
            f"{url} answered with a non-JSON body, so the served context window "
            f"is unknown and no request budget can be derived"
        ) from exc
    window = extract_context_window(body)
    if window is None:
        raise ContextWindowUnavailableError(
            f"{url} reports no max_model_len, so the served context window is "
            f"unknown and no request budget can be derived"
        )
    return window


@dataclass(frozen=True, slots=True)
class ResolvedContextWindow:
    """A context window and which of the two sources supplied it."""

    tokens: int
    source: Literal["served", "declared"]


def resolve_context_window(
    api_base: str,
    *,
    declared: int | None,
    model: str,
) -> ResolvedContextWindow:
    """The window to budget against: what the endpoint serves, else ``declared``.

    An endpoint that publishes ``max_model_len`` is authoritative — the window
    is a property of the deployment, not of the model, and only the endpoint
    knows which one it launched with. A listing that carries no window (a
    shadowed discovery route that answers without waking the engine) falls
    back to the window the endpoint's configuration declares. With neither,
    there is nothing to budget against and no guess is safe.
    """

    key = (api_base.rstrip("/"), model)
    with _WINDOW_MEMO_LOCK:
        memoized = _WINDOW_MEMO.get(key)
        if memoized is not None:
            return memoized
        resolved = _read_context_window(api_base, declared=declared, model=model)
        if len(_WINDOW_MEMO) >= _MAX_MEMOIZED_WINDOWS:
            _WINDOW_MEMO.pop(next(iter(_WINDOW_MEMO)))
        _WINDOW_MEMO[key] = resolved
        return resolved


def _read_context_window(
    api_base: str,
    *,
    declared: int | None,
    model: str,
) -> ResolvedContextWindow:
    """Read the window from the endpoint, falling back to ``declared``."""

    try:
        served = fetch_context_window(api_base)
    except ContextWindowUnavailableError as exc:
        if declared is None:
            raise ContextWindowUnavailableError(
                f"{model} at {api_base} publishes no context window and declares "
                f"none, so no request budget can be derived"
            ) from exc
        logger.warning(
            "%s at %s publishes no context window; budgeting against the "
            "declared %d tokens instead: %s",
            model,
            api_base,
            declared,
            exc,
        )
        return ResolvedContextWindow(tokens=declared, source="declared")

    logger.info(
        "%s at %s serves a %d-token context window",
        model,
        api_base,
        served,
    )
    return ResolvedContextWindow(tokens=served, source="served")


def fitting_demonstrations(
    demos: Sequence[Any],
    *,
    budget: TokenBudget,
    render: Callable[[Sequence[Any], Any], Messages],
    count_tokens: TokenCounter,
    calls: Sequence[Any],
) -> int:
    """How many of ``demos``, kept from the front, fit ``budget`` in every
    one of ``calls``.

    ``render(demos, call)`` assembles the chat request ``call`` sends with
    those demonstrations. A longer prefix never costs fewer tokens, so the
    prefix only shrinks: each call is counted once at the longest prefix every
    earlier call fit, and a call that overflows it bisects below it. Calls
    are taken longest request first, so the prefix usually settles on the
    first. Zero when a call overflows with no demonstrations: none is then the
    only prefix that can be sent.
    """

    if not calls:
        raise ValueError("fitting demonstrations needs at least one call")
    ranked = sorted(
        calls, key=lambda call: -len(json.dumps(render([], call), default=str))
    )

    def fits(kept: int, call: Any) -> bool:
        return count_tokens(render(demos[:kept], call)) <= budget.input_budget

    longest = len(demos)
    for call in ranked:
        if fits(longest, call):
            continue
        low, high = 0, longest - 1
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle, call):
                low = middle
            else:
                high = middle - 1
        longest = low
    return longest
