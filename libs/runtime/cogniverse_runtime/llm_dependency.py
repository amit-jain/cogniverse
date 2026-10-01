"""How a request that failed on the chat LLM is answered over HTTP.

The chat LLM is a dependency the caller cannot fix, so its failure is never an
opaque 500. An LLM that is unavailable -- nothing deployed (404), not
answering, overloaded -- is a 503; one that rejected the request cogniverse
sent -- a refused credential, a body the router would not accept -- is a 502.
A not-serving endpoint also says when it is rechecked, as ``Retry-After``.

The answer is built from the failure's typed fields, never from its text,
which carries endpoint URLs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import openai

from cogniverse_core.agents.base import leaf_exceptions
from cogniverse_foundation.config.lm_endpoint_availability import (
    LMEndpointNotServing,
)
from cogniverse_foundation.config.routed_lm import (
    RoutedLMCallFailed,
    RouterDecodeFailed,
    UpstreamAuthRejected,
    UpstreamNotServing,
)

LLM_UNAVAILABLE = "llm_unavailable"
LLM_REQUEST_REJECTED = "llm_request_rejected"

_NOT_SERVING = "is not serving: nothing is deployed for the model"
_UNAVAILABLE = "is unavailable"
_REJECTED = "rejected the request"


@dataclass(frozen=True)
class LLMDependencyFailure:
    """A request's failure on the chat LLM, as the HTTP answer it gets."""

    condition: str
    failure: str
    upstream_status: Optional[int]
    model: Optional[str]
    retry_after_s: Optional[int]

    @property
    def http_status(self) -> int:
        return 502 if self.condition == _REJECTED else 503

    @property
    def error(self) -> str:
        return LLM_REQUEST_REJECTED if self.condition == _REJECTED else LLM_UNAVAILABLE

    def message(self, agent: str) -> str:
        status = (
            f", upstream HTTP {self.upstream_status}" if self.upstream_status else ""
        )
        retry = (
            f" Retry after {self.retry_after_s}s."
            if self.retry_after_s is not None
            else ""
        )
        return (
            f"Agent '{agent}' could not complete: the chat LLM {self.condition} "
            f"({self.failure}{status}).{retry}"
        )

    def body(self, *, agent: str, request_id: str) -> Dict[str, Any]:
        return {
            "error": self.error,
            "dependency": "llm",
            "agent": agent,
            "failure": self.failure,
            "upstream_status": self.upstream_status,
            "model": self.model,
            "retry_after_s": self.retry_after_s,
            "request_id": request_id,
            "message": self.message(agent),
        }

    def headers(self) -> Optional[Dict[str, str]]:
        if self.retry_after_s is None:
            return None
        return {"Retry-After": str(self.retry_after_s)}


def _retry_after(recheck_in_s: float) -> int:
    return max(1, math.ceil(recheck_in_s))


def _routed(failure: RoutedLMCallFailed) -> LLMDependencyFailure:
    if isinstance(failure, UpstreamNotServing):
        condition = _NOT_SERVING
    elif isinstance(failure, (UpstreamAuthRejected, RouterDecodeFailed)):
        condition = _REJECTED
    else:
        condition = _UNAVAILABLE
    return LLMDependencyFailure(
        condition=condition,
        failure=type(failure).__name__,
        upstream_status=failure.status,
        model=failure.routed_model,
        retry_after_s=(
            _retry_after(failure.recheck_in_s)
            if isinstance(failure, UpstreamNotServing)
            else None
        ),
    )


def _not_serving(failure: LMEndpointNotServing) -> LLMDependencyFailure:
    return LLMDependencyFailure(
        condition=_NOT_SERVING,
        failure=type(failure).__name__,
        upstream_status=failure.status,
        model=failure.endpoint.model,
        retry_after_s=_retry_after(failure.recheck_in_s),
    )


def _provider(failure: openai.APIError) -> LLMDependencyFailure:
    status = failure.status_code if isinstance(failure, openai.APIStatusError) else None
    unavailable = status is None or status >= 500 or status == 429
    return LLMDependencyFailure(
        condition=_UNAVAILABLE if unavailable else _REJECTED,
        failure=type(failure).__name__,
        upstream_status=status,
        model=getattr(failure, "model", None),
        retry_after_s=None,
    )


def llm_dependency_failure(exc: BaseException) -> Optional[LLMDependencyFailure]:
    """The LLM failure ``exc`` is or was explicitly raised from, else ``None``.

    Follows each leaf's ``__cause__`` chain, nearest first; a failure that was
    handled before an unrelated one was raised is not that one's cause.
    """
    for leaf in leaf_exceptions(exc):
        seen: set[int] = set()
        current: Optional[BaseException] = leaf
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, RoutedLMCallFailed):
                return _routed(current)
            if isinstance(current, LMEndpointNotServing):
                return _not_serving(current)
            if isinstance(current, openai.APIError):
                return _provider(current)
            current = current.__cause__
    return None


__all__ = [
    "LLM_REQUEST_REJECTED",
    "LLM_UNAVAILABLE",
    "LLMDependencyFailure",
    "llm_dependency_failure",
]
