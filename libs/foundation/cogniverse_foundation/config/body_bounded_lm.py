"""A dspy.LM that reports what every request weighs on the wire."""

from __future__ import annotations

import logging
from typing import Any

import dspy

from cogniverse_foundation.config.request_body import (
    LLM_REQUEST_BODY_LIMIT_BYTES,
    http_status_of,
    messages_from,
    request_body_metrics,
)

logger = logging.getLogger(__name__)


class BodyBoundedLM(dspy.LM):
    """Measure the assembled request body of every call, and name its size on
    any 4xx the provider returns.

    A body over the proxy's ``per_connection_buffer_limit_bytes`` comes back as
    a bare 413 with no indication of what filled it, so the size travels with
    the error instead of having to be reconstructed from a machine where the
    request happens to fit.
    """

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

    def forward(self, prompt=None, messages=None, **kwargs):
        assembled = messages_from(prompt, messages)
        detail = self._report(assembled)
        try:
            return super().forward(messages=assembled, **kwargs)
        except Exception as exc:
            self._reraise(exc, detail)
            raise

    async def aforward(self, prompt=None, messages=None, **kwargs):
        assembled = messages_from(prompt, messages)
        detail = self._report(assembled)
        try:
            return await super().aforward(messages=assembled, **kwargs)
        except Exception as exc:
            self._reraise(exc, detail)
            raise
