"""The completion budget a request places on the LM calls made for it.

A dispatched request may carry ``context["max_output_tokens"]``. Every LM call
made while it is bound sends ``max_tokens`` no larger than that budget; the
endpoint's configured ``max_tokens`` still applies when it is smaller. An
unbound call sends the configured value unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Optional

MAX_OUTPUT_TOKENS_KEY = "max_output_tokens"

_CURRENT: ContextVar[Optional[int]] = ContextVar("lm_output_budget", default=None)


def output_token_budget_from(context: Mapping[str, Any]) -> Optional[int]:
    """The request's completion budget, or ``None`` when it names none.

    Raises:
        ValueError: The value is not a positive integer.
    """
    if MAX_OUTPUT_TOKENS_KEY not in context:
        return None
    budget = context[MAX_OUTPUT_TOKENS_KEY]
    if type(budget) is not int or budget < 1:
        raise ValueError(
            f"context.{MAX_OUTPUT_TOKENS_KEY} must be a positive integer, "
            f"got {budget!r}"
        )
    return budget


@contextmanager
def bound_output_token_budget(budget: Optional[int]) -> Iterator[None]:
    """Bind ``budget`` for LM calls made in this context; ``None`` binds nothing."""
    if budget is None:
        yield
        return
    token = _CURRENT.set(budget)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def budgeted_call_kwargs(
    configured: Mapping[str, Any], call_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """``call_kwargs`` with ``max_tokens`` held within the bound budget.

    ``configured`` is the LM's own request parameters, which a call's kwargs
    override.
    """
    kwargs = dict(call_kwargs)
    budget = _CURRENT.get()
    if budget is None:
        return kwargs
    requested = kwargs.get("max_tokens", configured.get("max_tokens"))
    kwargs["max_tokens"] = budget if requested is None else min(budget, requested)
    return kwargs
