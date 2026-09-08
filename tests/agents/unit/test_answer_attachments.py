"""Client content reaches answer modules with ordered images and explicit failures."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest
from PIL import Image

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchDeps,
    DeepResearchInput,
)
from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
    DetailedReportInput,
)
from cogniverse_agents.multimodal import attachments_to_images, hit_keyframe_uri
from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummarizerDeps,
    SummarizerInput,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig, SystemConfig
from tests.agents.unit.test_multimodal_attachments import (
    _data_uri,
    _decoded_jpeg,
    _encoded,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]
DISABLED_ERROR = "attachments_disabled: visual inputs are disabled for this request"
DEAD_IMAGE = "http://127.0.0.1:29071/photo.png"
FAILURE = "attachment[0]: URLError: <urlopen error [Errno 111] Connection refused>"
TEXT = "The Eiffel Tower was completed in 1889. Summarize this text."
ANSWER = "The Eiffel Tower was completed in 1889."


class ContentDecision(dspy.Module):
    """Controlled DSPy decision; real agents prepare and pass every input."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, **kwargs):
        self.inputs.append(kwargs)
        return dspy.Prediction(
            summary=ANSWER,
            executive_summary=ANSWER,
            key_points="Completed in 1889, Located in Paris",
            recommendations="Visit the tower, Read its history",
            sub_questions=["When was it completed?"],
            has_sufficient_evidence=True,
            gaps=[],
            confidence=0.9,
        )


@pytest.fixture
def build_agent(config_manager_memory):
    config_manager_memory.set_system_config(SystemConfig())

    def build(kind, **options):
        decision = ContentDecision()
        if kind == "summary":
            agent = SummarizerAgent(
                SummarizerDeps(thinking_enabled=False, **options),
                config_manager=config_manager_memory,
            )
            agent.summarization_module = decision
        elif kind == "report":
            agent = DetailedReportAgent(
                DetailedReportDeps(thinking_enabled=False, **options),
                config_manager=config_manager_memory,
            )
            agent.report_module = decision
        else:

            async def search_fn(**kwargs):
                return []

            agent = DeepResearchAgent(
                DeepResearchDeps(tenant_id="test:attachments", **options),
                search_fn=search_fn,
                config_manager=config_manager_memory,
            )
            agent._decomposer = decision
            agent._evaluator = decision
            agent._synthesizer = decision
        return agent, decision

    return build


async def answer(agent, kind, attachments, *, query=TEXT, hits=None, **kwargs):
    if kind == "summary":
        return await agent._process_impl(
            SummarizerInput(
                query=query,
                attachments=attachments,
                search_results=hits or [],
                **kwargs,
            )
        )
    if kind == "report":
        return await agent._process_impl(
            DetailedReportInput(
                query=query,
                attachments=attachments,
                search_results=hits or [],
                **kwargs,
            )
        )
    if hits:

        async def search_fn(**kwargs):
            return hits

        agent._search_fn = search_fn
    return await agent._process_impl(
        DeepResearchInput(
            query=query, attachments=attachments, tenant_id="test:attachments", **kwargs
        )
    )


def generation_input(decision):
    return [item for item in decision.inputs if "keyframes" in item][-1]


@pytest.mark.parametrize("kind", ["summary", "report", "research"])
async def test_attachments_precede_cached_keyframes_and_share_cap(build_agent, kind):
    agent, decision = build_agent(kind, max_keyframes_to_llm=3)
    uris = [
        _data_uri(Image.new("RGB", size, color), "PNG")
        for size, color in [((1600, 1200), "red"), ((100, 40), "blue")]
    ]
    hit = {
        "source_url": "s3://media/test:attachments/clip.mp4",
        "video_id": "clip",
        "segment_id": 1,
    }
    second_hit = {**hit, "segment_id": 2}
    frame = attachments_to_images(
        [_data_uri(Image.new("RGB", (90, 30), "green"), "PNG")]
    ).images[0]
    for cached_hit in (hit, second_hit):
        agent._keyframe_resolver._cache[hit_keyframe_uri(cached_hit)] = frame

    result = await answer(agent, kind, uris, hits=[hit, second_hit])

    assert [
        _decoded_jpeg(image).size for image in generation_input(decision)["keyframes"]
    ] == [(768, 576), (100, 40), (90, 30)]
    assert (result.executive_summary if kind == "report" else result.summary) == ANSWER
    if kind == "summary":
        assert result.key_points == ["Completed in 1889", "Located in Paris"]
    elif kind == "report":
        assert result.recommendations == ["Visit the tower", "Read its history"]
        assert result.metadata["keyframes_attached"] == 3


@pytest.mark.parametrize("summary_type", ["brief", "comprehensive", "bullet_points"])
async def test_zero_hits_supplied_text_has_exact_summary(build_agent, summary_type):
    agent, decision = build_agent("summary")
    result = await answer(agent, "summary", [], summary_type=summary_type)
    assert result.summary == ANSWER
    assert result.key_points == ["Completed in 1889", "Located in Paris"]
    assert generation_input(decision)["content"] == TEXT
    assert generation_input(decision)["summary_type"] == summary_type


@pytest.mark.parametrize("summary_type", ["brief", "comprehensive", "bullet_points"])
async def test_image_only_zero_hits_reaches_summary(build_agent, summary_type):
    agent, decision = build_agent("summary")
    result = await answer(
        agent,
        "summary",
        [_data_uri(Image.new("RGB", (100, 40), "blue"), "PNG")],
        query="",
        summary_type=summary_type,
    )
    assert result.summary == ANSWER
    assert [
        _decoded_jpeg(image).size for image in generation_input(decision)["keyframes"]
    ] == [(100, 40)]


async def test_enriched_summary_retains_attachments(build_agent):
    agent, decision = build_agent("summary")
    result = await answer(
        agent,
        "summary",
        [_data_uri(Image.new("RGB", (100, 40), "blue"), "PNG")],
        enhanced_query="Enhanced tower question",
        entities=[{"name": "Eiffel Tower"}],
    )
    assert result.summary == ANSWER
    assert generation_input(decision)["query"] == "Enhanced tower question"
    assert [
        _decoded_jpeg(image).size for image in generation_input(decision)["keyframes"]
    ] == [(100, 40)]


@pytest.mark.parametrize(
    ("kind", "options", "request_options"),
    [
        ("summary", {"multimodal_generation_enabled": False}, {}),
        ("summary", {"visual_analysis_enabled": False}, {}),
        ("summary", {}, {"include_visual_analysis": False}),
        ("report", {"multimodal_generation_enabled": False}, {}),
        ("report", {"visual_analysis_enabled": False}, {}),
        ("report", {}, {"include_visual_analysis": False}),
        ("research", {"multimodal_generation_enabled": False}, {}),
    ],
)
async def test_disabled_visuals_reject_attachments_with_named_error(
    build_agent, kind, options, request_options
):
    agent, decision = build_agent(kind, **options)
    with pytest.raises(ValueError) as exc:
        await answer(agent, kind, [DEAD_IMAGE], **request_options)
    assert str(exc.value) == DISABLED_ERROR
    assert decision.inputs == []


@pytest.mark.parametrize("kind", ["summary", "report", "research"])
async def test_dead_attachment_returns_exact_degradation(build_agent, kind):
    agent, decision = build_agent(kind)
    result = await answer(agent, kind, [DEAD_IMAGE])
    assert (result.executive_summary if kind == "report" else result.summary) == ANSWER
    assert generation_input(decision)["keyframes"] == []
    if kind == "report":
        assert result.metadata["report_degraded"] is True
        assert result.metadata["report_degraded_reason"] == FAILURE
    elif kind == "summary":
        assert result.metadata["attachments_degraded"] is True
        assert result.metadata["attachment_failures"] == [FAILURE]
    else:
        assert result.attachments_degraded is True
        assert result.attachment_failures == [FAILURE]


@pytest.fixture
def overlapping_images():
    barrier = threading.Barrier(2)
    payloads = {
        "/wide": _encoded(Image.new("RGB", (100, 40), "red"), "PNG"),
        "/tall": _encoded(Image.new("RGB", (40, 100), "blue"), "PNG"),
    }
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            barrier.wait(timeout=3)
            body = payloads[self.path]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", paths
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.mark.parametrize("kind", ["summary", "report", "research"])
async def test_concurrent_http_attachments_keep_request_state_and_free_loop(
    build_agent, overlapping_images, kind
):
    agent, decision = build_agent(kind)
    url, paths = overlapping_images
    loop_thread = threading.get_ident()
    completions = []
    original_forward = decision.forward

    def record_thread(**kwargs):
        completions.append(threading.get_ident() == loop_thread)
        return original_forward(**kwargs)

    decision.forward = record_thread
    left, right = await asyncio.wait_for(
        asyncio.gather(
            answer(agent, kind, [url + "/wide", DEAD_IMAGE], query="left"),
            answer(agent, kind, [url + "/tall"], query="right"),
        ),
        timeout=8,
    )
    generated = {
        item["query"]: [_decoded_jpeg(image).size for image in item["keyframes"]]
        for item in decision.inputs
        if "keyframes" in item
    }
    assert sorted(paths) == ["/tall", "/wide"]
    assert generated == {"left": [(100, 40)], "right": [(40, 100)]}
    assert completions == ([False] * 6 if kind == "research" else [False, False])
    if kind == "report":
        assert [
            left.metadata["report_degraded"],
            right.metadata["report_degraded"],
        ] == [True, False]
    elif kind == "summary":
        assert [
            left.metadata["attachments_degraded"],
            right.metadata["attachments_degraded"],
        ] == [True, False]
    else:
        assert [left.attachments_degraded, right.attachments_degraded] == [True, False]


async def test_zero_hit_text_does_not_hide_real_lm_connection_failure(build_agent):
    agent, _ = build_agent("summary")
    from cogniverse_agents.summarizer_agent import SummarizationModule

    agent.summarization_module = SummarizationModule()
    agent._llm_config = LLMEndpointConfig(
        model="openai/test",
        api_base="http://127.0.0.1:29071/v1",
        api_key="test",
        request_timeout=0.5,
        num_retries=0,
    )
    with pytest.raises(Exception) as exc:
        await answer(agent, "summary", [])
    assert type(exc.value).__name__ == "InternalServerError"
    assert (
        str(exc.value)
        == "litellm.InternalServerError: InternalServerError: OpenAIException - Connection error."
    )


async def test_report_attachment_and_real_lm_failure_reasons_accumulate(build_agent):
    from cogniverse_agents.detailed_report_agent import ReportGenerationModule

    agent, _ = build_agent("report")
    agent.report_module = ReportGenerationModule()
    agent._llm_config = LLMEndpointConfig(
        model="openai/test",
        api_base="http://127.0.0.1:29071/v1",
        api_key="test",
        request_timeout=0.5,
        num_retries=0,
    )
    result = await answer(agent, "report", [DEAD_IMAGE])
    assert result.executive_summary == (
        "Analysis of 0 results for 'The Eiffel Tower was completed in 1889. "
        "Summarize this text.' with average relevance of 0.00."
    )
    assert result.metadata["report_degraded"] is True
    assert result.metadata["report_degraded_reason"] == (
        FAILURE + "; InternalServerError: litellm.InternalServerError: "
        "InternalServerError: OpenAIException - Connection error."
    )


@pytest.mark.parametrize("kind", ["summary", "report", "research"])
def test_public_attachment_validation_rejects_typed_stream_input(build_agent, kind):
    agent, decision = build_agent(kind, multimodal_generation_enabled=False)
    input_cls = {
        "summary": SummarizerInput,
        "report": DetailedReportInput,
        "research": DeepResearchInput,
    }[kind]
    request = input_cls(
        query=TEXT, tenant_id="test:attachments", attachments=[DEAD_IMAGE]
    )
    with pytest.raises(ValueError) as exc:
        agent.validate_attachments(request)
    assert str(exc.value) == DISABLED_ERROR
    assert decision.inputs == []
