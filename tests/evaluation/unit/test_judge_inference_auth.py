"""The LLM and visual judges authenticate their chat-completions POSTs.

Both shipped with ``evaluators.*.api_key: null``: ``LLMJudgeCore`` then sent
``Bearer not-required`` and ``ConfigurableVisualJudge`` sent no Authorization
at all, so a Modal-hosted judge model answered 401 — read as "Evaluation
failed" (score None) or a raised "Vision API error" for every sample. Each
resolves its key by the same rule as ``create_dspy_lm``.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from cogniverse_core.common.media import MediaLocator
from cogniverse_evaluation.evaluators.configurable_visual_judge import (
    ConfigurableVisualJudge,
)
from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore

pytestmark = pytest.mark.unit

MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
IN_CLUSTER = "http://cogniverse-vllm-llm-student:8000/v1"


class _ChatCompletionsServer:
    """Real HTTP boundary that records the one request the judge sends."""

    def __init__(self, content: str):
        recorded: list[dict] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                recorded.append(
                    {
                        "path": self.path,
                        "authorization": self.headers["Authorization"],
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                payload = json.dumps(
                    {"choices": [{"message": {"content": content}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.recorded = recorded
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


class TestLLMJudgeCore:
    def test_modal_endpoint_gets_the_environment_bearer(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

        assert LLMJudgeCore(model_name="m", base_url=MODAL).api_key == "real-bearer"

    def test_in_cluster_endpoint_without_a_bearer_keeps_the_keyless_sentinel(
        self, monkeypatch
    ):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        assert (
            LLMJudgeCore(model_name="m", base_url=IN_CLUSTER).api_key == "not-required"
        )

    def test_an_explicit_key_is_not_overridden(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

        judge = LLMJudgeCore(model_name="m", base_url=MODAL, api_key="explicit-key")

        assert judge.api_key == "explicit-key"

    def test_modal_endpoint_without_a_bearer_fails_at_construction(self, monkeypatch):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            LLMJudgeCore(model_name="m", base_url=MODAL)

    @pytest.mark.asyncio
    async def test_the_bearer_reaches_the_wire(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

        with _ChatCompletionsServer("Score: 8/10. Relevant.") as server:
            judge = LLMJudgeCore(model_name="m", base_url=f"{server.base_url}/v1")
            reply = await judge._call_llm("rate this", system_prompt="be strict")

        assert reply == "Score: 8/10. Relevant."
        assert judge._extract_score_from_response(reply) == (
            0.8,
            "Score: 8/10. Relevant.",
        )
        assert server.recorded == [
            {
                "path": "/v1/chat/completions",
                "authorization": "Bearer real-bearer",
                "body": {
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": "be strict"},
                        {"role": "user", "content": "rate this"},
                    ],
                },
            }
        ]


def _visual_judge(base_url: str, api_key: str | None = None) -> ConfigurableVisualJudge:
    config = {
        "evaluators": {
            "visual_judge": {
                "provider": "vllm",
                "model": "m",
                "base_url": base_url,
                "api_key": api_key,
            }
        }
    }
    with patch(
        "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
        return_value=config,
    ):
        return ConfigurableVisualJudge(locator=MediaLocator.__new__(MediaLocator))


class TestConfigurableVisualJudge:
    def test_modal_endpoint_gets_the_environment_bearer(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

        assert _visual_judge(MODAL).api_key == "real-bearer"

    def test_in_cluster_endpoint_without_a_bearer_keeps_the_keyless_sentinel(
        self, monkeypatch
    ):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        assert _visual_judge(IN_CLUSTER).api_key == "not-required"

    def test_an_explicit_key_is_not_overridden(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

        assert _visual_judge(MODAL, api_key="explicit-key").api_key == "explicit-key"

    def test_modal_endpoint_without_a_bearer_fails_at_construction(self, monkeypatch):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

        with pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ):
            _visual_judge(MODAL)

    def test_the_bearer_reaches_the_wire(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xd9")
        encoded = base64.b64encode(frame.read_bytes()).decode("utf-8")

        with _ChatCompletionsServer("SCORE: 7/10, REASONING: frames match") as server:
            judge = _visual_judge(server.base_url)
            scored = judge._score_frames("a red car", [str(frame)])

        assert scored == (0.7, "frames match")
        assert server.recorded == [
            {
                "path": "/v1/chat/completions",
                "authorization": "Bearer real-bearer",
                "body": {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Do these video frames match the search "
                                        "query 'a red car'? Rate 0-10. Format: "
                                        "SCORE: X/10, REASONING: explanation"
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 300,
                },
            }
        ]
