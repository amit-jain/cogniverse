"""A dspy.LM that keeps every request inside the window its endpoint serves."""

from __future__ import annotations

import logging
import threading
from typing import Any

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.token_budget import (
    ContextWindowUnavailableError,
    FittedPrompt,
    PromptBudgetExceededError,
    TokenBudget,
    fit_messages,
    litellm_message_counter,
    resolve_context_window,
)

logger = logging.getLogger(__name__)

# Module scope, not per instance: dspy.LM.copy() deepcopies the LM and a lock
# is not deepcopy-able. Resolution happens once per LM, so the coarse grain
# costs nothing.
_BUDGET_LOCK = threading.Lock()


class BudgetedLM(BodyBoundedLM):
    """Fit the assembled prompt to ``context_window - max_tokens`` before sending.

    The window is read from the endpoint's ``/v1/models`` on first use and
    held for the life of the instance, falling back to
    ``declared_context_window`` when the listing carries none. Requests over
    the allowance shed whole few-shot demonstrations, oldest first; one that
    still overflows with none left raises ``PromptBudgetExceededError`` rather
    than reaching the provider as an opaque context-window rejection.
    """

    _budget: TokenBudget | None = None
    _counter = None

    def __init__(
        self,
        model: str,
        *,
        declared_context_window: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        # A plain attribute, never a dspy kwarg: everything in self.kwargs is
        # forwarded to litellm as a request parameter.
        self.declared_context_window = declared_context_window

    @property
    def budget(self) -> TokenBudget:
        """The endpoint's input allowance, read from the served model once."""

        if self._budget is not None:
            return self._budget
        with _BUDGET_LOCK:
            if self._budget is not None:
                return self._budget
            api_base = self.kwargs.get("api_base")
            if not api_base:
                raise ContextWindowUnavailableError(
                    f"{self.model} declares no api_base, so the served context "
                    f"window is unknown and no request budget can be derived"
                )
            reserved = self.kwargs.get("max_tokens") or self.kwargs.get(
                "max_completion_tokens"
            )
            if not reserved:
                raise ContextWindowUnavailableError(
                    f"{self.model} reserves no completion tokens, so the input "
                    f"allowance of its context window is undefined"
                )
            resolved = resolve_context_window(
                api_base,
                declared=self.declared_context_window,
                model=self.model,
            )
            logger.info(
                "Budgeting %s against its %s context_window=%d with "
                "reserved_output=%d at api_base=%s",
                self.model,
                resolved.source,
                resolved.tokens,
                int(reserved),
                api_base,
            )
            self._budget = TokenBudget(
                model=self.model,
                context_window=resolved.tokens,
                reserved_output=int(reserved),
            )
            return self._budget

    def _fit(self, prompt: str | None, messages: list[dict[str, Any]] | None):
        budget = self.budget
        if self._counter is None:
            self._counter = litellm_message_counter(self.model)
        fitted = fit_messages(
            messages or [{"role": "user", "content": prompt}],
            budget=budget,
            count_tokens=self._counter,
        )
        if fitted.dropped_demos:
            logger.warning(
                "Dropped %d of %d few-shot demonstrations for %s to fit "
                "context_window=%d reserved_output=%d: input_tokens=%d",
                fitted.dropped_demos,
                fitted.total_demos,
                self.model,
                budget.context_window,
                budget.reserved_output,
                fitted.input_tokens,
            )
        return fitted

    def _overflow(self, fitted: FittedPrompt) -> PromptBudgetExceededError:
        return PromptBudgetExceededError(
            model=self.model,
            context_window=self.budget.context_window,
            reserved_output=self.budget.reserved_output,
            input_tokens=fitted.input_tokens,
            dropped_demos=fitted.dropped_demos,
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        from litellm.exceptions import ContextWindowExceededError

        fitted = self._fit(prompt, messages)
        try:
            return super().forward(messages=fitted.messages, **kwargs)
        except ContextWindowExceededError as exc:
            raise self._overflow(fitted) from exc

    async def aforward(self, prompt=None, messages=None, **kwargs):
        from litellm.exceptions import ContextWindowExceededError

        fitted = self._fit(prompt, messages)
        try:
            return await super().aforward(messages=fitted.messages, **kwargs)
        except ContextWindowExceededError as exc:
            raise self._overflow(fitted) from exc
