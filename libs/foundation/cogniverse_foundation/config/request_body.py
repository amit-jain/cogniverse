"""Size of the chat-completions request body an LM call puts on the wire.

An answer LM call reaches the model through the semantic-router Envoy, whose
ext_proc filter buffers the whole request body (``request_body_mode: BUFFERED``)
and answers 413 "Payload Too Large" above its ``per_connection_buffer_limit_bytes``.
The chart pins the listener to ``LLM_REQUEST_BODY_LIMIT_BYTES``:
charts/cogniverse/files/semantic-router/envoy.yaml.

Measuring the body needs the assembled messages, so every number here is taken
from what is about to be sent, never predicted from what went into it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

LLM_REQUEST_BODY_LIMIT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class BodyMetrics:
    """What one assembled chat request weighs, split by what carries the bytes."""

    body_bytes: int
    image_part_count: int
    image_part_bytes: int

    @property
    def text_bytes(self) -> int:
        """Body bytes that are not image payload: instructions, the grounding
        content block, the adapter preamble and every JSON wrapper."""
        return self.body_bytes - self.image_part_bytes


def _image_part_url(part: Any) -> str | None:
    """The data URL an OpenAI-style ``image_url`` content part carries."""
    if not isinstance(part, Mapping) or part.get("type") != "image_url":
        return None
    holder = part.get("image_url")
    if isinstance(holder, Mapping):
        url = holder.get("url")
    else:
        url = holder
    return url if isinstance(url, str) else None


def serialize_messages(messages: Sequence[Mapping[str, Any]]) -> bytes:
    """The messages array as it is serialized into the request body.

    Base64 image payloads dominate the body and are ASCII, so the separator and
    ``ensure_ascii`` choices move the total by well under a tenth of a percent;
    the constant that matters is measured, not assumed --
    ``MEASURED_PARAM_OVERHEAD_BYTES``.
    """
    return json.dumps(list(messages), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def request_body_metrics(messages: Sequence[Mapping[str, Any]]) -> BodyMetrics:
    """Measure an assembled messages array."""
    image_count = 0
    image_bytes = 0
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            url = _image_part_url(part)
            if url is None:
                continue
            image_count += 1
            image_bytes += len(url.encode("utf-8"))
    return BodyMetrics(
        body_bytes=len(serialize_messages(messages)),
        image_part_count=image_count,
        image_part_bytes=image_bytes,
    )


def messages_from(prompt: Any, messages: Any) -> list[dict[str, Any]]:
    """The messages a dspy.LM call sends, whichever form the caller used."""
    if messages:
        return list(messages)
    return [{"role": "user", "content": prompt}]


def http_status_of(exc: BaseException) -> int | None:
    """The HTTP status an LM exception carries, or None when it carries none.

    Read from the exception's own typed fields -- never matched out of its
    message, which embeds the request URL and the provider's body.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def iter_image_urls(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every image data URL in an assembled messages array, in wire order."""
    urls: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            url = _image_part_url(part)
            if url is not None:
                urls.append(url)
    return urls
