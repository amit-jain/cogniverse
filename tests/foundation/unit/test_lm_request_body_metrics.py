"""Every LM call reports what its request body weighs.

A body over the semantic-router Envoy's ``per_connection_buffer_limit_bytes``
comes back as a bare 413 naming nothing, so the size has to travel with the
call: at DEBUG on every request, and on the ERROR line for any 4xx the provider
returns. Pinned here against a real HTTP chat-completions endpoint, so the
numbers are measured on the bytes that actually went out.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.request_body import (
    LLM_REQUEST_BODY_LIMIT_BYTES,
    BodyMetrics,
    http_status_of,
    iter_image_urls,
    request_body_metrics,
    serialize_messages,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

_PIXEL = base64.b64encode(b"\xff\xd8\xff" + b"k" * 997).decode("ascii")
_IMAGE_URL = f"data:image/jpeg;base64,{_PIXEL}"


def _messages(image_count: int, text: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    content += [
        {"type": "image_url", "image_url": {"url": _IMAGE_URL}}
        for _ in range(image_count)
    ]
    return [{"role": "user", "content": content}]


class _RecordingChatEndpoint:
    """A chat-completions endpoint that records each request's exact body size
    and answers with the status the test asks for."""

    def __init__(self, status: int = 200):
        self.request_bytes: list[int] = []
        recorded = self.request_bytes
        wanted = status

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                recorded.append(len(raw))
                if wanted != 200:
                    payload = json.dumps({"error": {"message": "refused"}}).encode()
                    self.send_response(wanted)
                else:
                    payload = json.dumps(
                        {
                            "id": "stub",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "stub",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "ok"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"


@pytest.mark.unit
class TestRequestBodyMetrics:
    """The measurement splits a body into exactly what carries its bytes."""

    def test_splits_image_payload_from_everything_else(self):
        messages = _messages(3, "ground the summary")
        metrics = request_body_metrics(messages)
        image_bytes = 3 * len(_IMAGE_URL.encode())

        assert metrics == BodyMetrics(
            body_bytes=len(serialize_messages(messages)),
            image_part_count=3,
            image_part_bytes=image_bytes,
        )
        assert metrics.text_bytes == metrics.body_bytes - image_bytes
        assert iter_image_urls(messages) == [_IMAGE_URL] * 3

    def test_a_text_only_body_carries_no_image_bytes(self):
        metrics = request_body_metrics([{"role": "user", "content": "plain"}])
        assert metrics.image_part_count == 0
        assert metrics.image_part_bytes == 0
        assert metrics.text_bytes == metrics.body_bytes

    def test_status_is_read_from_the_exception_never_its_message(self):
        """Classification by type/field, never by substring: a message that
        merely names a status must not be read as one."""

        class Typed(Exception):
            status_code = 413

        class Untyped(Exception):
            pass

        assert http_status_of(Typed()) == 413
        assert (
            http_status_of(Untyped("OpenAIException - 413 Payload Too Large")) is None
        )


@pytest.mark.unit
class TestBodyBoundedLMReportsWhatItSent:
    """The four numbers are measured against the bytes the endpoint received."""

    def _lm(self, endpoint, port_tag: str) -> BodyBoundedLM:
        return create_dspy_lm(
            LLMEndpointConfig(
                model=f"openai/stub-{port_tag}",
                api_base=endpoint.api_base,
                api_key="stub-key",
                temperature=0.0,
                max_tokens=8,
                num_retries=0,
            )
        )

    def test_factory_builds_the_measuring_lm(self):
        with _RecordingChatEndpoint() as endpoint:
            assert isinstance(self._lm(endpoint, "factory"), BodyBoundedLM)

    def test_debug_line_names_the_bytes_the_endpoint_received(self, caplog):
        messages = _messages(2, "ground the summary")
        expected = request_body_metrics(messages)
        with _RecordingChatEndpoint() as endpoint:
            lm = self._lm(endpoint, "debug")
            with caplog.at_level(
                logging.DEBUG, logger="cogniverse_foundation.config.body_bounded_lm"
            ):
                lm(messages=messages, cache=False)

            assert endpoint.request_bytes == [endpoint.request_bytes[0]]
            sent = endpoint.request_bytes[0]

        lines = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(lines) == 1
        assert f"body_bytes={expected.body_bytes}" in lines[0]
        assert "image_parts=2" in lines[0]
        assert f"image_bytes={expected.image_part_bytes}" in lines[0]
        assert f"text_bytes={expected.text_bytes}" in lines[0]
        assert f"limit={LLM_REQUEST_BODY_LIMIT_BYTES}" in lines[0]

        # The measurement is the messages array; the rest of the wire body is
        # the request parameters, and it is the smaller, bounded part.
        assert 0 < sent - expected.body_bytes < 4096

    def test_a_413_names_the_size_that_caused_it(self, caplog):
        """The failure this exists for: a refused call must never leave the
        operator to reconstruct the size from a machine where it fits."""
        messages = _messages(2, "ground the summary")
        expected = request_body_metrics(messages)
        with _RecordingChatEndpoint(status=413) as endpoint:
            lm = self._lm(endpoint, "refused")
            with caplog.at_level(
                logging.ERROR, logger="cogniverse_foundation.config.body_bounded_lm"
            ):
                with pytest.raises(Exception) as raised:
                    lm(messages=messages, cache=False)

        assert http_status_of(raised.value) == 413
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert errors[0] == (
            "LM rejected the request 413: "
            f"model=openai/stub-refused body_bytes={expected.body_bytes} "
            f"image_parts=2 image_bytes={expected.image_part_bytes} "
            f"text_bytes={expected.text_bytes} "
            f"limit={LLM_REQUEST_BODY_LIMIT_BYTES}"
        )

    def test_control_a_success_logs_no_error_line(self, caplog):
        """CONTROL: the ERROR line is produced by the 4xx, not by every call."""
        with _RecordingChatEndpoint(status=200) as endpoint:
            lm = self._lm(endpoint, "control")
            with caplog.at_level(
                logging.DEBUG, logger="cogniverse_foundation.config.body_bounded_lm"
            ):
                lm(messages=_messages(2, "ground the summary"), cache=False)

        assert [
            r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
        ] == []
