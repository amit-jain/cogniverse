"""A worker reads the context windows of its chat LM's endpoints at startup.

``preload_served_context_windows`` reads the window of every endpoint the
chat LM is served from — the student, and the teacher when it serves the pro
tier — so no request budgets a prompt and pays that read inside its own
deadline. Each endpoint is a ``/v1/models`` listing on a local stub that
records what it is asked.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogniverse_foundation.config.semantic_router import TEACHER_SERVICE
from cogniverse_foundation.config.token_budget import (
    ContextWindowEndpointUnreachableError,
    clear_context_window_memo,
    resolve_context_window,
)
from cogniverse_foundation.config.unified_config import LLMConfig, LLMEndpointConfig
from cogniverse_runtime.main import preload_served_context_windows

pytestmark = [pytest.mark.unit]

STUDENT_MODEL = "openai/google/gemma-4-e4b-it"
TEACHER_MODEL = "openai/google/gemma-4-31b-it"


@pytest.fixture(autouse=True)
def _fresh_window_memo():
    clear_context_window_memo()
    yield
    clear_context_window_memo()


class _ListingHandler(BaseHTTPRequestHandler):
    window: int = 0
    served: str = ""
    received: list = []
    release: threading.Event | None = None

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        type(self).received.append(self.path)
        if type(self).release is not None:
            type(self).release.wait(timeout=10)
        encoded = json.dumps(
            {
                "object": "list",
                "data": [{"id": type(self).served, "max_model_len": type(self).window}],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        return


def _serve_listing(served: str, window: int, release=None):
    received: list = []
    handler = type(
        "Handler",
        (_ListingHandler,),
        {"window": window, "served": served, "received": received, "release": release},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1", received


def _dead_api_base() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}/v1"


def _llm_config(student_base: str, teacher_base: str | None = None) -> LLMConfig:
    return LLMConfig(
        primary=LLMEndpointConfig(model=STUDENT_MODEL, api_base=student_base),
        teacher=(
            LLMEndpointConfig(model=TEACHER_MODEL, api_base=teacher_base)
            if teacher_base
            else None
        ),
    )


class TestStartupReadsTheServingWindows:
    def test_the_student_s_window_is_read_once_and_no_request_reads_it_again(self):
        server, student_base, received = _serve_listing("google/gemma-4-e4b-it", 8192)
        try:
            preload_served_context_windows(_llm_config(student_base), {})
            window = resolve_context_window(
                student_base, declared=None, model=STUDENT_MODEL
            )
        finally:
            server.shutdown()
        assert (window.tokens, window.source, received) == (
            8192,
            "served",
            ["/v1/models"],
        )

    def test_a_deployed_teacher_s_window_is_read_beside_the_student_s(self):
        student, student_base, student_received = _serve_listing(
            "google/gemma-4-e4b-it", 8192
        )
        teacher, teacher_base, teacher_received = _serve_listing(
            "google/gemma-4-31b-it", 32768
        )
        try:
            preload_served_context_windows(
                _llm_config(student_base, teacher_base),
                {TEACHER_SERVICE: teacher_base},
            )
            windows = (
                resolve_context_window(
                    student_base, declared=None, model=STUDENT_MODEL
                ).tokens,
                resolve_context_window(
                    teacher_base, declared=None, model=TEACHER_MODEL
                ).tokens,
            )
        finally:
            student.shutdown()
            teacher.shutdown()
        assert (windows, student_received, teacher_received) == (
            (8192, 32768),
            ["/v1/models"],
            ["/v1/models"],
        )

    def test_a_teacher_that_does_not_serve_the_pro_tier_is_not_read(self):
        student, student_base, student_received = _serve_listing(
            "google/gemma-4-e4b-it", 8192
        )
        teacher, teacher_base, teacher_received = _serve_listing(
            "google/gemma-4-31b-it", 32768
        )
        try:
            preload_served_context_windows(_llm_config(student_base, teacher_base), {})
        finally:
            student.shutdown()
            teacher.shutdown()
        assert (student_received, teacher_received) == (["/v1/models"], [])


class TestAnEndpointThatCannotAnswerAtStartup:
    def test_it_is_named_and_left_to_the_first_call_which_still_raises(self, caplog):
        dead = _dead_api_base()

        with caplog.at_level(logging.WARNING, logger="cogniverse_runtime.main"):
            preload_served_context_windows(_llm_config(dead), {})

        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.main"
        ] == [
            f"Context window of {STUDENT_MODEL} at {dead} was not read at startup; "
            f"the first call that needs it reads it: {STUDENT_MODEL} at {dead} "
            "is unreachable and declares no context window, so no request budget "
            "can be derived"
        ]
        with pytest.raises(ContextWindowEndpointUnreachableError) as exc:
            resolve_context_window(dead, declared=None, model=STUDENT_MODEL)
        assert str(exc.value) == (
            f"{STUDENT_MODEL} at {dead} is unreachable and declares no context "
            "window, so no request budget can be derived"
        )


class TestARequestDuringTheStartupRead:
    def test_it_waits_on_that_read_instead_of_making_its_own(self):
        release = threading.Event()
        server, student_base, received = _serve_listing(
            "google/gemma-4-e4b-it", 8192, release=release
        )
        windows: list = []

        def first_request() -> None:
            windows.append(
                resolve_context_window(
                    student_base, declared=None, model=STUDENT_MODEL
                ).tokens
            )

        preload = threading.Thread(
            target=preload_served_context_windows,
            args=(_llm_config(student_base), {}),
        )
        request = threading.Thread(target=first_request)
        try:
            preload.start()
            while not received:
                threading.Event().wait(0.01)
            request.start()
            release.set()
            preload.join(timeout=10)
            request.join(timeout=10)
        finally:
            server.shutdown()
        assert (windows, received) == ([8192], ["/v1/models"])
