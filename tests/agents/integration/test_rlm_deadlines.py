"""An RLM deadline stops the work instead of abandoning it.

Real ``dspy.RLM`` over the real Deno/Pyodide interpreter, driven by an
OpenAI-compatible model this module serves on loopback. The model records
every completion it is asked for, which is how "the computation stopped" is
observed: after the deadline raises, no further completions arrive.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest

from cogniverse_agents.inference.rlm_inference import RLMInference, RLMTimeoutError
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

pytestmark = pytest.mark.integration

# Never calls SUBMIT, so the REPL loop runs until max_iterations or the
# deadline — the loop is what the deadline has to stop.
LOOPING_CODE = "```python\nprint('still working')\n```"

SUB_CALLS = 6

# One iteration's program spending the rest of the sub-LLM budget. Every
# llm_query is a model call the deadline has to refuse.
SUBCALL_CODE = (
    "```python\n"
    f"for _i in range({SUB_CALLS}):\n"
    "    llm_query('sub-query')\n"
    "print('done')\n"
    "```"
)


@contextmanager
def scripted_model(
    hold_seconds: float,
    *,
    status: int = 200,
    code=LOOPING_CODE,
    arrivals: threading.Barrier | None = None,
):
    """Serve chat completions, holding each one for ``hold_seconds``.

    With ``arrivals``, the first completions are answered only once every
    party of the barrier has arrived.
    """
    calls: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                calls.append(body)
                arrival = len(calls)
            if arrivals is not None and arrival <= arrivals.parties:
                arrivals.wait(timeout=300)
            time.sleep(hold_seconds)
            if status != 200:
                payload = json.dumps(
                    {"error": {"message": "model backend refused the request"}}
                ).encode()
            else:
                payload = json.dumps(
                    {
                        "id": "deadline-fixture",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "deadline-fixture",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "reasoning": "Keep exploring.",
                                            "code": code,
                                            "answer": "incomplete",
                                        }
                                    ),
                                },
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
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield {
            "api_base": f"http://127.0.0.1:{server.server_port}/v1",
            "calls": calls,
        }
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _inference(api_base: str, timeout_seconds, *, max_iterations: int = 4):
    return RLMInference(
        llm_config=LLMEndpointConfig(
            model="openai/deadline-fixture",
            api_base=api_base,
            api_key="local",
            max_tokens=256,
            temperature=0,
            num_retries=0,
            request_timeout=60,
        ),
        max_iterations=max_iterations,
        max_llm_calls=8,
        timeout_seconds=timeout_seconds,
    )


def _run(inference: RLMInference, context: str = "deadline-context"):
    with dspy.context(adapter=dspy.JSONAdapter()):
        return inference.process(query="Summarize the context.", context=context)


def test_expired_deadline_stops_further_iterations():
    """Past the deadline the loop stops; the model sees no further calls."""
    hold = 4.0
    with scripted_model(hold) as model:
        # The budget covers interpreter startup plus the first model call and
        # nothing more, so the second iteration boundary is past the deadline.
        inference = _inference(model["api_base"], timeout_seconds=hold / 2)
        started = time.monotonic()
        with pytest.raises(RLMTimeoutError) as raised:
            _run(inference)
        elapsed = time.monotonic() - started

        assert str(raised.value) == f"RLM processing exceeded timeout of {hold / 2}s"
        calls_at_raise = len(model["calls"])
        assert calls_at_raise == 1
        # One in-flight model call is the whole overrun: two more holds is
        # ample time for an abandoned loop to issue its next call.
        time.sleep(hold * 2)
        assert len(model["calls"]) == calls_at_raise
        assert elapsed < hold * 2


def test_expired_deadline_stops_in_repl_model_calls():
    """A generated program cannot spend its sub-LLM budget past the deadline."""
    hold = 2.0
    with scripted_model(hold, code=SUBCALL_CODE) as model:
        inference = _inference(model["api_base"], timeout_seconds=hold / 2)
        started = time.monotonic()
        with pytest.raises(RLMTimeoutError) as raised:
            _run(inference)
        elapsed = time.monotonic() - started

        assert str(raised.value) == f"RLM processing exceeded timeout of {hold / 2}s"
        # The action call is the whole overrun: it was in flight when the
        # deadline passed, and every llm_query after it is refused.
        assert len(model["calls"]) == 1
        time.sleep(hold * 2)
        assert len(model["calls"]) == 1
        # Serving the sub-calls would have taken SUB_CALLS more holds.
        assert elapsed < hold * SUB_CALLS


def test_one_callers_deadline_does_not_truncate_another():
    """Two budgets on one module: neither caller sees the other's deadline."""
    hold = 2.0
    with scripted_model(hold) as model:
        inference = _inference(model["api_base"], timeout_seconds=None)
        rlm = inference._get_rlm()
        assert inference._get_rlm() is rlm
        both_inside = threading.Barrier(2)

        def call(context: str, timeout_seconds):
            with dspy.context(adapter=dspy.JSONAdapter()):
                with rlm.deadline(timeout_seconds):
                    both_inside.wait(timeout=60)
                    return inference._execute_rlm(
                        rlm, "Summarize the context.", context
                    )

        with ThreadPoolExecutor(max_workers=2) as pool:
            expiring = pool.submit(call, "tenant-a", hold / 2)
            surviving = pool.submit(call, "tenant-b", None)
            with pytest.raises(RLMTimeoutError) as raised:
                expiring.result(timeout=180)
            result, _tokens = surviving.result(timeout=180)

    assert str(raised.value) == f"RLM processing exceeded timeout of {hold / 2}s"
    assert result.answer == "incomplete"
    # One action call for the expiring caller; four action calls plus the
    # extract fallback for the caller that ran to max_iterations.
    assert len(model["calls"]) == 6


def test_model_outage_raises_instead_of_answering():
    """A failing model boundary propagates rather than returning an answer."""
    with scripted_model(0.0, status=503) as model:
        inference = _inference(model["api_base"], timeout_seconds=None)
        with pytest.raises(Exception) as raised:
            _run(inference)

    assert not isinstance(raised.value, RLMTimeoutError)
    assert "model backend refused the request" in str(raised.value)
    assert len(model["calls"]) == 1
