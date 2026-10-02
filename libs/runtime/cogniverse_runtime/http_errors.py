"""HTTP answers for failed requests, built from typed fields.

An exception's text can name backend URLs, credentials or file paths, so an
error body never carries it. The body carries a stable ``error`` code, a
``message`` the route builds from values it owns, and ``failure``, the
exception's type name. The cause itself goes to the runtime log, with its
traceback, and to the active span.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Union

from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from cogniverse_core.common.models.model_loaders import (
    INFERENCE_BREAKER_RESET_TIMEOUT_S,
)
from cogniverse_core.query.encoders import (
    EncoderNotConfiguredError,
    EncoderUnavailableError,
)

logger = logging.getLogger(__name__)


def record_failure(
    exc: BaseException, error: str, *, level: int = logging.ERROR
) -> None:
    """Log ``exc`` with its traceback and record it on the active span."""
    logger.log(
        level,
        "%s: %s: %s",
        error,
        type(exc).__name__,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, f"{error}: {type(exc).__name__}"))


def failure_body(
    error: str, message: str, exc: BaseException, **fields: Any
) -> Dict[str, Any]:
    """The typed body for a failure: code, message, failure type, fields."""
    return {"error": error, "message": message, "failure": type(exc).__name__, **fields}


def failure_response(
    status_code: int,
    error: str,
    message: str,
    exc: BaseException,
    *,
    headers: Optional[Dict[str, str]] = None,
    **fields: Any,
) -> HTTPException:
    """Record ``exc`` and return the HTTPException that answers it.

    ``message`` and ``fields`` must come from values the route owns (request
    fields, tenant and profile names), never from ``exc``. A 4xx is logged as
    a warning, anything else as an error.
    """
    record_failure(
        exc, error, level=logging.WARNING if status_code < 500 else logging.ERROR
    )
    return HTTPException(
        status_code=status_code,
        detail=failure_body(error, message, exc, **fields),
        headers=headers,
    )


QUERY_ENCODER_NOT_CONFIGURED = "query_encoder_not_configured"
QUERY_ENCODER_UNAVAILABLE = "query_encoder_unavailable"
# A tripped inference-endpoint breaker admits a trial call after this long.
QUERY_ENCODER_RETRY_AFTER_S = math.ceil(INFERENCE_BREAKER_RESET_TIMEOUT_S)


def query_encoder_failure(
    exc: Union[EncoderNotConfiguredError, EncoderUnavailableError],
    *,
    profile: Optional[str],
    strategy: Optional[str],
    **fields: Any,
) -> tuple[int, Dict[str, Any], Optional[Dict[str, str]]]:
    """Status, body and headers for a request whose query encoder failed.

    Built from the failure's typed fields, never its text, which names the
    sidecar URL. A configuration gap no retry fixes is a 500; a configured
    encoder whose service did not serve the request is a 503 with
    ``Retry-After``. ``profile`` stands in when the failure names none.
    """
    if isinstance(exc, EncoderUnavailableError):
        cause = exc.__cause__
        failure = type(cause if cause is not None else exc).__name__
        where = (
            f"inference service '{exc.service}'" if exc.service else "the local encoder"
        )
        retry_after = QUERY_ENCODER_RETRY_AFTER_S
        body = {
            "error": QUERY_ENCODER_UNAVAILABLE,
            "dependency": "query_encoder",
            "profile": exc.profile,
            "strategy": strategy,
            "service": exc.service,
            "failure": failure,
            "retry_after_s": retry_after,
            "message": (
                f"The query encoder for profile '{exc.profile}' is unavailable: "
                f"{where} did not serve the request ({failure}). "
                f"Retry after {retry_after}s."
            ),
            **fields,
        }
        return 503, body, {"Retry-After": str(retry_after)}
    profile = exc.profile or profile
    needs = f"Strategy '{strategy}' needs" if strategy else "The search needs"
    body = {
        "error": QUERY_ENCODER_NOT_CONFIGURED,
        "dependency": "query_encoder",
        "profile": profile,
        "strategy": strategy,
        "message": (
            f"{needs} a query encoder, and profile '{profile}' has none "
            "configured in this deployment."
        ),
        **fields,
    }
    return 500, body, None


def upstream_rejection(
    status_code: int,
    error: str,
    message: str,
    *,
    upstream_status: int,
    upstream_body: str,
    **fields: Any,
) -> HTTPException:
    """The HTTPException for an upstream that answered with an error status.

    The upstream's body is logged and never served: it can carry the
    upstream's internal names and policy text.
    """
    logger.error(
        "%s: upstream HTTP %s: %s", error, upstream_status, upstream_body[:500]
    )
    span = trace.get_current_span()
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, f"{error}: HTTP {upstream_status}"))
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "message": message,
            "upstream_status": upstream_status,
            **fields,
        },
    )


__all__ = [
    "QUERY_ENCODER_NOT_CONFIGURED",
    "QUERY_ENCODER_RETRY_AFTER_S",
    "QUERY_ENCODER_UNAVAILABLE",
    "failure_body",
    "failure_response",
    "query_encoder_failure",
    "record_failure",
    "upstream_rejection",
]
