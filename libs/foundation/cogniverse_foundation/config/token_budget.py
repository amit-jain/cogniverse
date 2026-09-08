"""Fit an assembled chat request inside the context window its endpoint serves.

A chat request costs ``prompt tokens + max_tokens`` against the model's
context window. ``max_tokens`` is the completion an endpoint reserves; the
remainder is everything the adapter may send. This module reads the window
the endpoint reports for the model it serves, derives that remainder, and
fits the assembled messages into it by dropping whole few-shot
demonstrations. A request that still overflows with no demonstrations left
raises with the window, the reservation, the measured input and the number
of demonstrations dropped.
"""

from __future__ import annotations

import logging
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


class ContextWindowUnavailableError(RuntimeError):
    """The endpoint does not report a context window for the model it serves."""


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
