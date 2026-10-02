"""HTTP answers for failed requests, built from typed fields.

An exception's text can name backend URLs, credentials or file paths, so an
error body never carries it. The body carries a stable ``error`` code, a
``message`` the route builds from values it owns, and ``failure``, the
exception's type name. The cause itself goes to the runtime log, with its
traceback, and to the active span.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

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
    "failure_body",
    "failure_response",
    "record_failure",
    "upstream_rejection",
]
