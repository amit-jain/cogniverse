"""The answer LM request fits the body limit its transport buffers.

Frames reach the summary/report/research LM as base64 data URLs. Counting them
does not bound the request: the semantic-router Envoy in front of every agent
LM call buffers the whole body for its ext_proc routing decision and answers
413 "Payload Too Large" above ``per_connection_buffer_limit_bytes``, so four
768 px frames refuse the call outright and the turn produces no answer.

Pinned here: which images survive the allowance and how many are shed, the
chart and the agents naming one limit, and the streaming path answering over a
real HTTP endpoint that enforces that limit exactly as Envoy does.
"""

from __future__ import annotations

import base64
import json
import random
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import dspy
import pytest
import yaml
from PIL import Image

from cogniverse_agents.multimodal import (
    LLM_REQUEST_BODY_LIMIT_BYTES,
    LLM_TEXT_RESERVE_BYTES,
    MAX_IMAGE_PAYLOAD_BYTES,
    FittedImages,
    KeyframeImageResolver,
    fit_answer_images,
    image_payload_bytes,
)
from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummarizerDeps,
    SummarizerInput,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
CHART_VALUES = REPO_ROOT / "charts" / "cogniverse" / "values.yaml"
CHART_ENVOY = (
    REPO_ROOT / "charts" / "cogniverse" / "files" / "semantic-router" / "envoy.yaml"
)


def _noise_jpeg(path: Path, edge: int) -> Path:
    """A JPEG that does not compress away, so its payload is frame-sized."""
    rnd = random.Random(edge)
    image = Image.new("RGB", (edge, edge))
    image.putdata(
        [
            (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
            for _ in range(edge * edge)
        ]
    )
    image.save(path, format="JPEG", quality=85)
    return path


@pytest.fixture(scope="module")
def frame_jpeg(tmp_path_factory) -> Path:
    # Sized so three frames fit the allowance and four overflow the transport
    # limit; both are asserted where they are relied on.
    return _noise_jpeg(tmp_path_factory.mktemp("frames") / "frame.jpg", 560)


@pytest.fixture(scope="module")
def frame_image(frame_jpeg) -> dspy.Image:
    raw = frame_jpeg.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return dspy.Image(url=f"data:image/jpeg;base64,{encoded}")


def _video_hit(index: int) -> dict[str, Any]:
    """A hit carrying exactly the fields keyframe derivation reads."""
    return {
        "id": f"hit-{index}",
        "title": f"Clip {index}",
        "content_type": "video",
        "relevance": 1.0 - index / 100,
        "metadata": {
            "source_url": "s3://cogniverse-media/flywheel_org:production/vid.mp4",
            "video_id": f"vid{index}",
            "segment_id": index,
        },
    }


class _DiskLocator:
    """MediaLocator stand-in: every keyframe URI resolves to one real JPEG."""

    def __init__(self, jpeg: Path):
        self._jpeg = jpeg
        self.localized: list[str] = []

    def localize(self, uri: str) -> str:
        self.localized.append(uri)
        return str(self._jpeg)


@pytest.mark.unit
class TestFitAnswerImages:
    """Whole images are shed, in rank order, to fit the body allowance."""

    def test_allowance_is_the_transport_limit_less_the_text_reserve(self):
        assert MAX_IMAGE_PAYLOAD_BYTES == (
            LLM_REQUEST_BODY_LIMIT_BYTES - LLM_TEXT_RESERVE_BYTES
        )

    def test_sheds_the_frames_that_do_not_fit(self, frame_image):
        cost = image_payload_bytes(frame_image)
        fits = MAX_IMAGE_PAYLOAD_BYTES // cost
        assert fits == 3, (
            "fixture must be sized so four frames overflow the allowance: "
            f"cost={cost} allowance={MAX_IMAGE_PAYLOAD_BYTES}"
        )

        fitted = fit_answer_images([], [frame_image] * 4, max_images=4)

        assert fitted == FittedImages(images=[frame_image] * fits, shed=4 - fits)
        assert sum(image_payload_bytes(i) for i in fitted.images) == fits * cost

    def test_control_raising_the_allowance_keeps_every_frame(self, frame_image):
        """Same input, allowance above the payload: nothing is shed.

        Without this the shed count above could come from the count cap rather
        than from the byte bound.
        """
        cost = image_payload_bytes(frame_image)
        fitted = fit_answer_images(
            [], [frame_image] * 4, max_images=4, max_total_bytes=4 * cost
        )
        assert fitted == FittedImages(images=[frame_image] * 4, shed=0)

    def test_attachments_are_kept_before_retrieved_frames(
        self, frame_image, frame_jpeg
    ):
        attachment = dspy.Image(str(frame_jpeg))
        fitted = fit_answer_images(
            [attachment], [frame_image] * 4, max_images=2, max_total_bytes=10**9
        )
        assert fitted == FittedImages(images=[attachment, frame_image], shed=0)

    def test_one_frame_over_the_whole_allowance_degrades_to_text_only(
        self, frame_image
    ):
        fitted = fit_answer_images(
            [],
            [frame_image, frame_image],
            max_images=2,
            max_total_bytes=image_payload_bytes(frame_image) - 1,
        )
        assert fitted == FittedImages(images=[], shed=2)


@pytest.mark.unit
class TestChartAndAgentsNameOneLimit:
    """The proxy's buffer limit and the agents' allowance cannot drift apart."""

    def test_chart_value_equals_the_agent_limit(self):
        values = yaml.safe_load(CHART_VALUES.read_text())
        assert (
            values["semanticRouter"]["envoy"]["maxRequestBytes"]
            == LLM_REQUEST_BODY_LIMIT_BYTES
        )

    def test_envoy_listener_reads_that_value(self):
        assert (
            "per_connection_buffer_limit_bytes: "
            "{{ int .Values.semanticRouter.envoy.maxRequestBytes }}"
        ) in CHART_ENVOY.read_text()


_STUB_SUMMARY = "Machine learning fits patterns from data without explicit rules."
_STUB_COMPLETION = (
    "[[ ## reasoning ## ]]\n"
    "The frames show a lecture slide.\n\n"
    "[[ ## summary ## ]]\n"
    f"{_STUB_SUMMARY}\n\n"
    "[[ ## key_points ## ]]\n"
    "learns from data, needs no explicit rules\n\n"
    "[[ ## confidence_score ## ]]\n"
    "0.9\n\n"
    "[[ ## completed ## ]]\n"
)


class _BodyLimitedChatEndpoint:
    """A real chat-completions endpoint with Envoy's buffered-body contract.

    ``body_limit=None`` refuses every request, which is the 413 the cluster
    served before the allowance existed.
    """

    def __init__(self, *, body_limit: Optional[int]):
        self._body_limit = body_limit
        self.request_bytes: list[int] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: A003 - silence the stdlib logger
                return

            def do_POST(self):  # noqa: N802 - stdlib handler name
                body = self.rfile.read(int(self.headers["Content-Length"]))
                endpoint.request_bytes.append(len(body))
                limit = endpoint._body_limit
                if limit is None or len(body) > limit:
                    payload = b"Payload Too Large"
                    self.send_response(413)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if json.loads(body).get("stream"):
                    self._stream()
                else:
                    self._complete()

            def _complete(self):
                payload = json.dumps(
                    {
                        "id": "stub-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "stub-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": _STUB_COMPLETION,
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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in (_STUB_COMPLETION, None):
                    delta = {} if chunk is None else {"content": chunk}
                    frame = json.dumps(
                        {
                            "id": "stub-1",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "stub-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None if chunk else "stop",
                                }
                            ],
                        }
                    )
                    self._send_chunk(f"data: {frame}\n\n".encode())
                self._send_chunk(b"data: [DONE]\n\n")
                self._send_chunk(b"")

            def _send_chunk(self, data: bytes):
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_BodyLimitedChatEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def _summarizer(endpoint: "_BodyLimitedChatEndpoint", locator) -> SummarizerAgent:
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    with (
        patch.object(SummarizerAgent, "_initialize_vlm_client"),
        patch("cogniverse_agents.summarizer_agent.VLMInterface"),
    ):
        agent = SummarizerAgent(
            deps=SummarizerDeps(
                multimodal_generation_enabled=True, max_keyframes_to_llm=4
            ),
            config_manager=ConfigManager(store=store),
        )
    agent._llm_config = LLMEndpointConfig(
        # One model name per endpoint instance: DSPy's disk cache keys by
        # model + prompt and outlives the process, and every case here sends
        # the same prompt, so a port-derived name can replay another run's
        # answer without reaching this endpoint.
        model=f"openai/stub-{uuid.uuid4().hex}",
        api_base=endpoint.api_base,
        api_key="stub-key",
        temperature=0.0,
        max_tokens=256,
        num_retries=0,
    )
    agent._keyframe_resolver = KeyframeImageResolver(locator)
    return agent


async def _stream_events(agent: SummarizerAgent) -> list[dict[str, Any]]:
    typed_input = SummarizerInput(
        query="summarize what machine learning is",
        search_results=[_video_hit(i) for i in range(4)],
        summary_type="comprehensive",
        include_visual_analysis=True,
    )
    events: list[dict[str, Any]] = []
    async for event in await agent.process(typed_input, stream=True):
        events.append(event)
    return events


# Every progress event a refused summary stream emits, in order: the thinking
# pass, its partial, the visual pass that finds no visual elements, the
# summarization pass, then the error that replaces the result.
_REFUSED_STREAM = [
    ("status", "thinking"),
    ("partial", "thinking"),
    ("status", "visual_analysis"),
    ("status", "summarization"),
    ("error", None),
]


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamingAnswerFitsTheTransportLimit:
    """The streaming path answers over an endpoint that enforces the limit."""

    async def test_stream_answers_and_reports_the_shed_frames(
        self, frame_jpeg, frame_image
    ):
        locator = _DiskLocator(frame_jpeg)
        fits = MAX_IMAGE_PAYLOAD_BYTES // image_payload_bytes(frame_image)
        with _BodyLimitedChatEndpoint(
            body_limit=LLM_REQUEST_BODY_LIMIT_BYTES
        ) as endpoint:
            dspy.configure(adapter=dspy.ChatAdapter())
            events = await _stream_events(_summarizer(endpoint, locator))

            # Non-empty (the set would be empty otherwise) and every attempt
            # inside the limit.
            assert {
                n <= LLM_REQUEST_BODY_LIMIT_BYTES for n in endpoint.request_bytes
            } == {True}

        finals = [e for e in events if e["type"] == "final"]
        assert len(finals) == 1
        assert finals[0]["data"]["summary"] == _STUB_SUMMARY
        assert finals[0]["data"]["key_points"] == [
            "learns from data",
            "needs no explicit rules",
        ]
        assert finals[0]["data"]["metadata"]["keyframes_attached"] == fits
        assert finals[0]["data"]["metadata"]["keyframes_shed"] == 4 - fits

    async def test_control_unbounded_frames_lose_the_whole_answer(
        self, frame_jpeg, frame_image
    ):
        """Pre-fix assembly against the same endpoint: 413, no answer.

        Counting frames without costing them is exactly what produced a stream
        of progress events and no result at all.
        """
        locator = _DiskLocator(frame_jpeg)
        with _BodyLimitedChatEndpoint(
            body_limit=LLM_REQUEST_BODY_LIMIT_BYTES
        ) as endpoint:
            dspy.configure(adapter=dspy.ChatAdapter())
            agent = _summarizer(endpoint, locator)
            with patch(
                "cogniverse_agents.summarizer_agent.fit_answer_images",
                lambda attachments, retrieved, *, max_images, **kwargs: FittedImages(
                    images=[*attachments, *retrieved][:max_images], shed=0
                ),
            ):
                events = await _stream_events(agent)

            assert {
                n > LLM_REQUEST_BODY_LIMIT_BYTES for n in endpoint.request_bytes
            } == {True}

        assert [(e["type"], e.get("phase")) for e in events] == _REFUSED_STREAM
        assert events[-1]["error_type"] == "APIError"
        assert events[-1]["agent"] == "SummarizerAgent"

    async def test_endpoint_refusing_every_request_yields_a_named_error_event(
        self, frame_jpeg
    ):
        """Fault contract: a refusing LM produces an error event, never silence."""
        locator = _DiskLocator(frame_jpeg)
        with _BodyLimitedChatEndpoint(body_limit=None) as endpoint:
            dspy.configure(adapter=dspy.ChatAdapter())
            events = await _stream_events(_summarizer(endpoint, locator))

            assert {
                n <= LLM_REQUEST_BODY_LIMIT_BYTES for n in endpoint.request_bytes
            } == {True}

        assert [(e["type"], e.get("phase")) for e in events] == _REFUSED_STREAM
        assert events[-1]["error_type"] == "APIError"
