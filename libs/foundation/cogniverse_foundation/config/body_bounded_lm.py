"""A dspy.LM that reports what every request weighs on the wire."""

from __future__ import annotations

import logging
from typing import Any, Optional

import dspy

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.lm_deadline import (
    LMCallDeadline,
    LMCallDeadlineExceeded,
    current_lm_call_deadline,
    deadline_bound_openai_client,
)
from cogniverse_foundation.config.lm_response_cache import (
    TenantScopedLMCache,
    lm_response_cache,
    request_cache_key,
)
from cogniverse_foundation.config.request_body import (
    LLM_REQUEST_BODY_LIMIT_BYTES,
    http_status_of,
    messages_from,
    request_body_metrics,
)

logger = logging.getLogger(__name__)


class BodyBoundedLM(dspy.LM):
    """Measure requests and cache responses under the bound canonical tenant.

    DSPy's caches are disabled. An unbound LM reaches the provider each time.
    """

    def __init__(
        self,
        model: str,
        *,
        cache_tenant_id: str | None = None,
        response_cache: TenantScopedLMCache | None = None,
        **kwargs: Any,
    ) -> None:
        if cache_tenant_id:
            if kwargs.get("cache", False):
                raise ValueError(
                    f"{model} binds cache_tenant_id={cache_tenant_id!r} and "
                    f"cache=True; a tenant-scoped LM is served from "
                    f"TenantScopedLMCache and DSPy's shared cache is off"
                )
        kwargs["cache"] = False
        super().__init__(model, **kwargs)
        # Plain attributes, never dspy kwargs: everything in self.kwargs is
        # forwarded to litellm as a request parameter.
        self.cache_tenant_id = (
            canonical_tenant_id(cache_tenant_id) if cache_tenant_id else None
        )
        self.response_cache = (
            response_cache
            if response_cache is not None or self.cache_tenant_id is None
            else lm_response_cache()
        )

    def for_tenant(self, tenant_id: str) -> "BodyBoundedLM":
        """This LM, answering for ``tenant_id``.

        The process-global LM a runtime configures at startup serves every
        tenant. Binding it per request is what puts the tenant in the response
        cache key on the direct-to-backend path, where no routing header
        carries one.
        """
        canonical = canonical_tenant_id(tenant_id)
        if self.cache_tenant_id == canonical:
            return self
        bound = self.copy(cache_tenant_id=canonical, cache=False)
        if bound.response_cache is None:
            bound.response_cache = lm_response_cache()
        return bound

    def _report(self, messages: list[dict[str, Any]]) -> str:
        metrics = request_body_metrics(messages)
        detail = (
            f"model={self.model} body_bytes={metrics.body_bytes} "
            f"image_parts={metrics.image_part_count} "
            f"image_bytes={metrics.image_part_bytes} "
            f"text_bytes={metrics.text_bytes} "
            f"limit={LLM_REQUEST_BODY_LIMIT_BYTES}"
        )
        logger.debug("LM request body: %s", detail)
        return detail

    def _reraise(self, exc: BaseException, detail: str) -> None:
        status = http_status_of(exc)
        if status is not None and 400 <= status < 500:
            logger.error("LM rejected the request %d: %s", status, detail)

    def cache_key(
        self, messages: list[dict[str, Any]], call_kwargs: dict[str, Any]
    ) -> str:
        """The key this call is stored under for the bound tenant."""
        request = {
            "model": self.model,
            "messages": messages,
            **{
                key: value
                for key, value in {**self.kwargs, **call_kwargs}.items()
                if key != "cache"
            },
        }
        return request_cache_key(self.cache_tenant_id, request)

    def _deadline_exceeded(self, deadline: LMCallDeadline) -> LMCallDeadlineExceeded:
        return deadline.exceeded(endpoint=self.kwargs.get("api_base"), model=self.model)

    def _refuse_past_deadline(self) -> Optional[LMCallDeadline]:
        """The bound deadline, or raise once it is spent."""
        deadline = current_lm_call_deadline()
        if deadline is not None and deadline.expired:
            raise self._deadline_exceeded(deadline)
        return deadline

    def _within_deadline(self, kwargs: dict[str, Any]) -> None:
        """Refuse a request once its caller's deadline is spent. An
        OpenAI-compatible request goes through the client that sends it with
        the time left at that moment as its timeout."""
        deadline = self._refuse_past_deadline()
        api_base = self.kwargs.get("api_base")
        if deadline is None or not api_base or not self.model.startswith("openai/"):
            return
        kwargs["client"] = deadline_bound_openai_client(
            api_base, self.kwargs.get("api_key")
        )

    def _upstream(self, messages: list[dict[str, Any]], **kwargs):
        detail = self._report(messages)
        kwargs["cache"] = False
        self._within_deadline(kwargs)
        try:
            return super().forward(messages=messages, **kwargs)
        except Exception as exc:
            self._reraise(exc, detail)
            raise

    async def _aupstream(self, messages: list[dict[str, Any]], **kwargs):
        detail = self._report(messages)
        kwargs["cache"] = False
        self._refuse_past_deadline()
        try:
            return await super().aforward(messages=messages, **kwargs)
        except Exception as exc:
            self._reraise(exc, detail)
            raise

    def forward(self, prompt=None, messages=None, **kwargs):
        assembled = messages_from(prompt, messages)
        if self.cache_tenant_id is None:
            return self._upstream(assembled, **kwargs)
        deadline = current_lm_call_deadline()
        try:
            return self.response_cache.get_or_call(
                self.cache_key(assembled, kwargs),
                lambda: self._upstream(assembled, **kwargs),
                tenant_id=self.cache_tenant_id,
                model=self.model,
                wait_s=None if deadline is None else deadline.remaining_s(),
            )
        except TimeoutError as exc:
            if deadline is None or isinstance(exc, LMCallDeadlineExceeded):
                raise
            raise self._deadline_exceeded(deadline) from exc

    async def aforward(self, prompt=None, messages=None, **kwargs):
        assembled = messages_from(prompt, messages)
        if self.cache_tenant_id is None:
            return await self._aupstream(assembled, **kwargs)
        deadline = current_lm_call_deadline()
        try:
            return await self.response_cache.aget_or_call(
                self.cache_key(assembled, kwargs),
                lambda: self._aupstream(assembled, **kwargs),
                tenant_id=self.cache_tenant_id,
                model=self.model,
                wait_s=None if deadline is None else deadline.remaining_s(),
            )
        except TimeoutError as exc:
            if deadline is None or isinstance(exc, LMCallDeadlineExceeded):
                raise
            raise self._deadline_exceeded(deadline) from exc
