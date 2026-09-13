"""Typed failures for a chat completion sent through the semantic router.

The router preserves the upstream's HTTP status and error body, so the status
is the contract: 401/403 is a credential or tenant refusal, 429 a quota, 5xx an
outage, and any other 4xx the router refusing the request cogniverse built.
litellm collapses several of those onto one exception class -- a 403 from the
upstream and a 400 from the router both arrive as ``BadRequestError`` -- so a
caller reading the litellm class alone cannot tell "this tenant may not use
this model" from "cogniverse sent something the router would not accept".

``RoutedLM`` classifies each failure by the status and the error ``code`` the
provider sent, both read from the exception's typed fields, and raises one of
the classes below chained from the original. Every one names the tenant, the
tier and the routed model alias the call was made for. A timeout or a refused
connection answers no status at all and is an ``UpstreamUnavailable``.

The router does not report which decision it selected on a failed call (the
``x-vsr-selected-decision`` header is present only on an answered request, and
litellm does not carry response headers onto its exceptions), so the routing
inputs are what travels with the error.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

import openai
from opentelemetry import trace

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.request_body import (
    http_status_of,
    messages_from,
    request_body_metrics,
)
from cogniverse_foundation.config.semantic_router import record_served_model
from cogniverse_foundation.telemetry.span_contract import (
    LLM_TIER_DEGRADED_ATTRIBUTE,
    LLM_UPSTREAM_EXCEPTION_TYPE_ATTRIBUTE,
    LLM_UPSTREAM_STATUS_ATTRIBUTE,
    PRO_MODEL_UNAVAILABLE,
)

logger = logging.getLogger(__name__)


class RoutedLMCallFailed(RuntimeError):
    """A routed chat completion produced no completion.

    Never raised directly -- a caller catches this to mean "no answer" and one
    of the subclasses to decide whether to retry, re-credential or give up.
    """

    def __init__(
        self,
        summary: str,
        *,
        status: Optional[int],
        router_code: Optional[str],
        tenant_id: str,
        tier: str,
        routed_model: str,
    ) -> None:
        super().__init__(
            f"{summary}: tenant={tenant_id} tier={tier} "
            f"routed_model={routed_model} status={status} "
            f"router_code={router_code}"
        )
        self.summary = summary
        self.status = status
        self.router_code = router_code
        self.tenant_id = tenant_id
        self.tier = tier
        self.routed_model = routed_model


class UpstreamAuthRejected(RoutedLMCallFailed):
    """The model endpoint refused the credentials (401) or this tenant (403)."""


class UpstreamRateLimited(RoutedLMCallFailed):
    """The model endpoint refused the request for quota reasons (429)."""


class UpstreamUnavailable(RoutedLMCallFailed):
    """The model endpoint did not answer: 5xx, a reset, or a timeout."""


class RouterDecodeFailed(RoutedLMCallFailed):
    """The router would not accept the request cogniverse sent (other 4xx)."""


_SUMMARIES = {
    UpstreamAuthRejected: "the model endpoint rejected the credentials or the tenant",
    UpstreamRateLimited: "the model endpoint refused the request for quota",
    UpstreamUnavailable: "the model endpoint did not answer",
    RouterDecodeFailed: "the semantic router would not accept the request",
}


def router_error_code(exc: BaseException) -> Optional[str]:
    """The provider's own error ``code`` for a failed call, or ``None``.

    Read from the exception's ``code`` field, which litellm fills from the
    provider's error body, never matched out of the message. litellm fills it
    with the stringified status when the provider sent none, which is not a
    code.
    """
    code = getattr(exc, "code", None)
    status = http_status_of(exc)
    if isinstance(code, str) and code and code != str(status):
        return code
    return None


def classify_routed_failure(
    exc: BaseException,
    *,
    tenant_id: str,
    tier: str,
    routed_model: str,
) -> Optional[RoutedLMCallFailed]:
    """The typed failure for ``exc``, or ``None`` when it is not a call failure.

    Anything that is not a provider/transport error -- a bug in cogniverse's
    own assembly, a cancellation -- returns ``None`` so it propagates as
    itself instead of being reported as an endpoint fault.
    """
    if not isinstance(exc, openai.APIError):
        return None
    status = http_status_of(exc)
    if isinstance(exc, openai.APIConnectionError):
        # A timeout or a refused/reset connection: no status was answered, and
        # the one litellm stamps on (408 / 500) is synthetic.
        kind: type[RoutedLMCallFailed] = UpstreamUnavailable
    elif status in (401, 403):
        kind = UpstreamAuthRejected
    elif status == 429:
        kind = UpstreamRateLimited
    elif status is not None and status >= 500:
        kind = UpstreamUnavailable
    elif status is not None:
        kind = RouterDecodeFailed
    else:
        kind = UpstreamUnavailable
    return kind(
        _SUMMARIES[kind],
        status=status,
        router_code=router_error_code(exc),
        tenant_id=tenant_id,
        tier=tier,
        routed_model=routed_model,
    )


# Failures a second attempt can plausibly answer. A refused credential, a
# forbidden tenant and a request the router will not accept answer the same way
# every time, so retrying one only doubles the latency and the load.
RETRYABLE = (UpstreamRateLimited, UpstreamUnavailable)


_request_degradation: ContextVar[dict[str, Any] | None] = ContextVar(
    "routed_lm_degradation", default=None
)


@contextmanager
def tier_degradation_context() -> Iterator[dict[str, Any]]:
    """Collect degradation fields for one response, including worker-thread calls."""
    metadata: dict[str, Any] = {}
    token = _request_degradation.set(metadata)
    try:
        yield metadata
    finally:
        _request_degradation.reset(token)


def _record_degradation(failure: RoutedLMCallFailed) -> dict[str, Any]:
    metadata = {
        LLM_TIER_DEGRADED_ATTRIBUTE: PRO_MODEL_UNAVAILABLE,
        LLM_UPSTREAM_STATUS_ATTRIBUTE: failure.status,
        LLM_UPSTREAM_EXCEPTION_TYPE_ATTRIBUTE: type(failure).__name__,
    }
    trace.get_current_span().set_attributes(
        {key: value for key, value in metadata.items() if value is not None}
    )
    request = _request_degradation.get()
    if request is not None:
        request.update(metadata)
    logger.warning("pro model unavailable; trying the student: %s", failure)
    return metadata


def _annotate_completion(response, metadata: dict[str, Any]):
    return response.model_copy(update=metadata)


class RoutedLM(BodyBoundedLM):
    """A ``dspy.LM`` addressing the semantic router that names its failures.

    Carries the routing inputs of the request so a failure says which tenant,
    tier and routed model it belongs to, and owns the retry policy: the
    endpoint's ``num_retries`` applies only to statuses a retry can answer.
    """

    def __init__(
        self,
        model: str,
        *,
        tenant_id: str,
        tier: str,
        vision_model: str | None = None,
        student_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.tenant_id = tenant_id
        self.tier = tier
        # The endpoint's retry allowance, spent here rather than inside
        # litellm, which retries by status class and would re-send a rejected
        # credential.
        self.call_attempts = self.num_retries + 1
        self.num_retries = 0
        self._student: RoutedLM | None = None
        if tier == "pro" and student_model and student_model != model:
            self._student = RoutedLM(
                student_model,
                tenant_id=tenant_id,
                tier=tier,
                **{**kwargs, "num_retries": 0},
            )
        self._vision: RoutedLM | None = None
        if vision_model and vision_model != model:
            self._vision = RoutedLM(
                vision_model, tenant_id=tenant_id, tier=tier, **kwargs
            )

    def _carrier(self, prompt, messages):
        if self._vision is None:
            return self
        if request_body_metrics(messages_from(prompt, messages)).image_part_count:
            return self._vision
        return self

    def _classified(self, exc: BaseException) -> Optional[RoutedLMCallFailed]:
        failure = classify_routed_failure(
            exc,
            tenant_id=self.tenant_id,
            tier=self.tier,
            routed_model=self.model,
        )
        if failure is not None:
            logger.error("routed LM call failed: %s", failure)
        return failure

    def _can_use_student(self, failure: RoutedLMCallFailed) -> bool:
        return (
            self._student is not None
            and isinstance(failure, UpstreamUnavailable)
            and failure.status in (None, 408, 500, 502, 503, 504)
        )

    def _retryable(self, failure: RoutedLMCallFailed, attempt: int) -> bool:
        return isinstance(failure, RETRYABLE) and attempt < self.call_attempts

    def forward(self, prompt=None, messages=None, **kwargs):
        carrier = self._carrier(prompt, messages)
        if carrier is not self:
            return carrier.forward(prompt=prompt, messages=messages, **kwargs)
        for attempt in range(1, self.call_attempts + 1):
            try:
                response = super().forward(prompt=prompt, messages=messages, **kwargs)
                record_served_model(response)
                return response
            except Exception as exc:
                failure = self._classified(exc)
                if failure is None:
                    raise
                if self._can_use_student(failure):
                    metadata = _record_degradation(failure)
                    response = self._student.forward(
                        prompt=prompt,
                        messages=messages,
                        **{"cache": self.cache, **kwargs},
                    )
                    return _annotate_completion(response, metadata)
                if self._retryable(failure, attempt):
                    continue
                raise failure from exc

    async def aforward(self, prompt=None, messages=None, **kwargs):
        carrier = self._carrier(prompt, messages)
        if carrier is not self:
            return await carrier.aforward(prompt=prompt, messages=messages, **kwargs)
        for attempt in range(1, self.call_attempts + 1):
            try:
                response = await super().aforward(
                    prompt=prompt, messages=messages, **kwargs
                )
                record_served_model(response)
                return response
            except Exception as exc:
                failure = self._classified(exc)
                if failure is None:
                    raise
                if self._can_use_student(failure):
                    metadata = _record_degradation(failure)
                    response = await self._student.aforward(
                        prompt=prompt,
                        messages=messages,
                        **{"cache": self.cache, **kwargs},
                    )
                    return _annotate_completion(response, metadata)
                if self._retryable(failure, attempt):
                    continue
                raise failure from exc
