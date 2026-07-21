"""Answer-time keyframe injection: the read path.

Pins three seams that together close the "pixels never reach the answer LLM"
gap for the detailed-report agent:

* the shared resolver derives the exact keyframe URI from a hit and encodes
  each frame at most once (bounded cache), skipping unfetchable frames;
* the report signature/module actually emits ``image_url`` parts to the LM
  (not text) when frames are attached;
* the agent's assembly forwards exactly K frames to the LM only when the
  rollout flag AND the visual toggles are on.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, Mock, patch

import dspy
import pytest
from PIL import Image

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchDeps,
    SynthesisSignature,
)
from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
    ReportGenerationModule,
    ReportRequest,
    ThinkingPhase,
)
from cogniverse_agents.multimodal import KeyframeImageResolver, hit_keyframe_uri
from cogniverse_agents.summarizer_agent import (
    SummarizationModule,
    SummarizerAgent,
    SummarizerDeps,
    SummaryRequest,
)


@pytest.fixture
def jpg_path(tmp_path):
    p = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(p)
    return str(p)


class FakeLocator:
    """Stands in for MediaLocator: records every localize() and returns a real
    on-disk jpg so dspy.Image(path) produces a genuine image."""

    def __init__(self, jpg, missing=(), errors=None):
        self._jpg = jpg
        self.calls = []
        self._missing = set(missing)
        self._errors = errors or {}

    def localize(self, uri):
        self.calls.append(uri)
        if uri in self._errors:
            raise self._errors[uri]
        if uri in self._missing:
            raise FileNotFoundError(uri)
        return self._jpg


def _video_hit(segment_id, video_id="vid123", bucket="media", tenant="acme:acme"):
    """A hit in the canonical ``SearchResult.to_dict`` shape: the keyframe
    fields live under ``metadata``, not the top level. This is what every agent
    actually receives from ``SearchService.search``."""
    return {
        "document_id": f"{video_id}_{segment_id}",
        "score": 0.9,
        "highlights": {},
        "metadata": {
            "source_url": f"s3://{bucket}/{tenant}/somefile.mp4",
            "video_id": video_id,
            "segment_id": segment_id,
            "video_title": "clip",
        },
    }


def _flat_video_hit(segment_id, video_id="vid123", bucket="media", tenant="acme:acme"):
    """A hit whose keyframe fields were flattened to the top level by an agent
    pipeline — the resolver must derive the same URI from either shape."""
    return {
        "source_url": f"s3://{bucket}/{tenant}/somefile.mp4",
        "video_id": video_id,
        "segment_id": segment_id,
        "score": 0.9,
    }


@pytest.mark.unit
class TestHitKeyframeUri:
    def test_derives_exact_uri(self):
        uri = hit_keyframe_uri(_video_hit(7))
        assert uri == "s3://media/acme:acme/keyframes/vid123/0007.jpg"

    def test_derives_exact_uri_from_flattened_hit(self):
        uri = hit_keyframe_uri(_flat_video_hit(7))
        assert uri == "s3://media/acme:acme/keyframes/vid123/0007.jpg"

    def test_nested_and_flat_derive_identical_uri(self):
        assert hit_keyframe_uri(_video_hit(7)) == hit_keyframe_uri(_flat_video_hit(7))

    def test_non_s3_source_url_returns_none(self):
        h = _video_hit(7)
        h["metadata"]["source_url"] = "file:///x.mp4"
        assert hit_keyframe_uri(h) is None

    def test_missing_source_url_returns_none(self):
        h = _video_hit(7)
        del h["metadata"]["source_url"]
        assert hit_keyframe_uri(h) is None

    def test_missing_video_id_returns_none(self):
        h = _video_hit(7)
        del h["metadata"]["video_id"]
        assert hit_keyframe_uri(h) is None

    def test_missing_segment_id_returns_none(self):
        h = _video_hit(7)
        del h["metadata"]["segment_id"]
        assert hit_keyframe_uri(h) is None

    def test_string_segment_id_matches_int(self):
        assert hit_keyframe_uri(_video_hit("7")) == hit_keyframe_uri(_video_hit(7))


@pytest.mark.unit
class TestKeyframeImageResolver:
    def test_collect_returns_one_image_per_hit(self, jpg_path):
        r = KeyframeImageResolver(FakeLocator(jpg_path))
        imgs = r.collect([_video_hit(1), _video_hit(2)], max_images=8)
        assert len(imgs) == 2
        assert all(isinstance(i, dspy.Image) for i in imgs)

    def test_respects_max_images(self, jpg_path):
        r = KeyframeImageResolver(FakeLocator(jpg_path))
        imgs = r.collect([_video_hit(i) for i in range(5)], max_images=2)
        assert len(imgs) == 2

    def test_max_images_zero_returns_empty(self, jpg_path):
        r = KeyframeImageResolver(FakeLocator(jpg_path))
        assert r.collect([_video_hit(1)], max_images=0) == []

    def test_skips_hits_without_derivable_uri(self, jpg_path):
        loc = FakeLocator(jpg_path)
        r = KeyframeImageResolver(loc)
        hits = [_video_hit(1), {"title": "no source_url"}, _video_hit(2)]
        imgs = r.collect(hits, max_images=8)
        assert len(imgs) == 2
        assert len(loc.calls) == 2  # the fieldless hit never reached the locator

    def test_encode_cache_avoids_second_localize(self, jpg_path):
        loc = FakeLocator(jpg_path)
        r = KeyframeImageResolver(loc)
        r.collect([_video_hit(1)], max_images=8)
        r.collect([_video_hit(1)], max_images=8)
        uri = "s3://media/acme:acme/keyframes/vid123/0001.jpg"
        assert loc.calls.count(uri) == 1  # second collect served from cache

    def test_distinct_uris_each_localized(self, jpg_path):
        loc = FakeLocator(jpg_path)
        r = KeyframeImageResolver(loc)
        r.collect([_video_hit(1), _video_hit(2)], max_images=8)
        assert len(set(loc.calls)) == 2

    def test_missing_frame_skipped_others_kept(self, jpg_path):
        missing = "s3://media/acme:acme/keyframes/vid123/0002.jpg"
        loc = FakeLocator(jpg_path, missing=[missing])
        r = KeyframeImageResolver(loc)
        imgs = r.collect([_video_hit(1), _video_hit(2), _video_hit(3)], max_images=8)
        assert len(imgs) == 2  # the missing one dropped, not raised

    def test_transient_io_error_skipped_others_kept(self, jpg_path):
        bad = "s3://media/acme:acme/keyframes/vid123/0002.jpg"
        loc = FakeLocator(jpg_path, errors={bad: OSError("connection reset")})
        r = KeyframeImageResolver(loc)
        imgs = r.collect([_video_hit(1), _video_hit(2), _video_hit(3)], max_images=8)
        assert len(imgs) == 2  # transient IO error degrades that frame, not raised

    def test_unexpected_error_propagates_not_swallowed(self, jpg_path):
        bad = "s3://media/acme:acme/keyframes/vid123/0001.jpg"
        loc = FakeLocator(jpg_path, errors={bad: ValueError("unsupported scheme")})
        r = KeyframeImageResolver(loc)
        with pytest.raises(ValueError):
            r.collect([_video_hit(1)], max_images=8)

    def test_cache_bounded_by_size(self, jpg_path):
        loc = FakeLocator(jpg_path)
        r = KeyframeImageResolver(loc, cache_size=2)
        r.collect([_video_hit(i) for i in range(4)], max_images=8)
        assert len(r._cache) == 2  # evicted oldest


def _count_image_parts(messages) -> int:
    n = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(
                1
                for part in c
                if isinstance(part, dict) and part.get("type") == "image_url"
            )
    return n


class _CapturingLM(dspy.LM):
    """Captures the litellm message payload and returns a completion carrying
    every output field of both the report and summary signatures, so one LM
    serves both modules (dspy ignores the markers a given signature doesn't
    declare)."""

    def __init__(self):
        super().__init__(model="capturing")
        self.messages = []

    def __call__(self, prompt=None, messages=None, **kw):
        self.messages.append(messages or prompt)
        return [
            "[[ ## reasoning ## ]]\nbecause\n"
            "[[ ## executive_summary ## ]]\nA summary.\n"
            "[[ ## key_findings ## ]]\none\n"
            "[[ ## recommendations ## ]]\ndo x\n"
            "[[ ## summary ## ]]\nA summary.\n"
            "[[ ## key_points ## ]]\none, two\n"
            "[[ ## confidence_score ## ]]\n0.8\n"
            "[[ ## completed ## ]]"
        ]


@pytest.mark.unit
class TestReportModuleEmitsImageParts:
    """The signature+module actually send image_url parts to the LM."""

    def test_two_frames_produce_two_image_parts(self, jpg_path):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        module = ReportGenerationModule()
        frames = [dspy.Image(jpg_path), dspy.Image(jpg_path)]
        module.forward(
            content="c", query="q", report_type="comprehensive", keyframes=frames
        )
        assert _count_image_parts(lm.messages[-1]) == 2

    def test_no_frames_produce_zero_image_parts(self):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        module = ReportGenerationModule()
        module.forward(
            content="c", query="q", report_type="comprehensive", keyframes=[]
        )
        assert _count_image_parts(lm.messages[-1]) == 0


def _thinking_phase():
    return ThinkingPhase(
        content_analysis={
            "total_results": 3,
            "avg_relevance": 0.8,
            "content_types": {},
        },
        visual_assessment={"has_visual_content": True},
        technical_findings=[],
        patterns_identified=[],
        gaps_and_limitations=[],
        reasoning="",
    )


@pytest.fixture
def report_agent(jpg_path):
    with (
        patch.object(DetailedReportAgent, "_initialize_vlm_client"),
        patch("cogniverse_agents.detailed_report_agent.VLMInterface"),
    ):
        agent = DetailedReportAgent(
            deps=DetailedReportDeps(
                multimodal_generation_enabled=True, max_keyframes_to_llm=4
            ),
            config_manager=Mock(),
        )
    agent._keyframe_resolver = KeyframeImageResolver(FakeLocator(jpg_path))
    return agent


@pytest.mark.unit
@pytest.mark.asyncio
class TestReportAgentAssemblyGate:
    """The assembly forwards exactly the frames the gate allows to call_dspy."""

    async def _run(self, agent, request):
        agent.call_dspy = AsyncMock(return_value=Mock(executive_summary="ok"))
        await agent._generate_executive_summary(request, _thinking_phase())
        return agent.call_dspy.call_args.kwargs["keyframes"]

    async def test_enabled_forwards_k_frames(self, report_agent):
        req = ReportRequest(
            query="q",
            search_results=[_video_hit(1), _video_hit(2)],
            include_visual_analysis=True,
        )
        frames = await self._run(report_agent, req)
        assert len(frames) == 2
        assert all(isinstance(f, dspy.Image) for f in frames)

    async def test_capped_at_max_keyframes(self, report_agent):
        report_agent.max_keyframes_to_llm = 1
        req = ReportRequest(
            query="q",
            search_results=[_video_hit(1), _video_hit(2), _video_hit(3)],
            include_visual_analysis=True,
        )
        assert len(await self._run(report_agent, req)) == 1

    async def test_disabled_flag_forwards_no_frames(self, report_agent):
        report_agent.multimodal_generation_enabled = False
        req = ReportRequest(
            query="q",
            search_results=[_video_hit(1), _video_hit(2)],
            include_visual_analysis=True,
        )
        assert await self._run(report_agent, req) == []

    async def test_request_visual_off_forwards_no_frames(self, report_agent):
        req = ReportRequest(
            query="q",
            search_results=[_video_hit(1)],
            include_visual_analysis=False,
        )
        assert await self._run(report_agent, req) == []

    async def test_non_visual_hits_yield_no_frames(self, report_agent):
        req = ReportRequest(
            query="q",
            search_results=[{"title": "text-only", "score": 0.5}],
            include_visual_analysis=True,
        )
        assert await self._run(report_agent, req) == []


@pytest.mark.unit
class TestSummaryModuleEmitsImageParts:
    """The summarizer signature+module send image_url parts to the LM."""

    def test_two_frames_produce_two_image_parts(self, jpg_path):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        SummarizationModule().forward(
            content="c",
            query="q",
            summary_type="brief",
            keyframes=[dspy.Image(jpg_path), dspy.Image(jpg_path)],
        )
        assert _count_image_parts(lm.messages[-1]) == 2

    def test_no_frames_produce_zero_image_parts(self):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        SummarizationModule().forward(
            content="c", query="q", summary_type="brief", keyframes=[]
        )
        assert _count_image_parts(lm.messages[-1]) == 0


@pytest.fixture
def summarizer_agent(jpg_path):
    with (
        patch.object(SummarizerAgent, "_initialize_vlm_client"),
        patch("cogniverse_agents.summarizer_agent.VLMInterface"),
    ):
        agent = SummarizerAgent(
            deps=SummarizerDeps(
                multimodal_generation_enabled=True, max_keyframes_to_llm=4
            ),
            config_manager=Mock(),
        )
    agent._keyframe_resolver = KeyframeImageResolver(FakeLocator(jpg_path))
    return agent


@pytest.mark.unit
class TestSummarizerFrameGate:
    """_collect_keyframes yields frames only under the rollout + visual gates."""

    def test_collects_when_enabled(self, summarizer_agent):
        req = SummaryRequest(
            query="q",
            search_results=[_video_hit(1), _video_hit(2)],
            include_visual_analysis=True,
        )
        frames = summarizer_agent._collect_keyframes(req, req.search_results)
        assert len(frames) == 2
        assert all(isinstance(f, dspy.Image) for f in frames)

    def test_capped_at_max_keyframes(self, summarizer_agent):
        summarizer_agent.max_keyframes_to_llm = 1
        req = SummaryRequest(
            query="q",
            search_results=[_video_hit(1), _video_hit(2), _video_hit(3)],
            include_visual_analysis=True,
        )
        assert len(summarizer_agent._collect_keyframes(req, req.search_results)) == 1

    def test_disabled_flag_yields_no_frames(self, summarizer_agent):
        summarizer_agent.multimodal_generation_enabled = False
        req = SummaryRequest(
            query="q", search_results=[_video_hit(1)], include_visual_analysis=True
        )
        assert summarizer_agent._collect_keyframes(req, req.search_results) == []

    def test_request_visual_off_yields_no_frames(self, summarizer_agent):
        req = SummaryRequest(
            query="q", search_results=[_video_hit(1)], include_visual_analysis=False
        )
        assert summarizer_agent._collect_keyframes(req, req.search_results) == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestSummarizerRunForwardsFrames:
    """The summarization funnel forwards collected frames to the LM call."""

    async def test_run_summarization_forwards_keyframes(
        self, summarizer_agent, jpg_path
    ):
        summarizer_agent.call_dspy = AsyncMock(return_value=Mock(summary="ok"))
        frames = [dspy.Image(jpg_path)]
        await summarizer_agent._run_summarization("c", "q", "brief", keyframes=frames)
        assert summarizer_agent.call_dspy.call_args.kwargs["keyframes"] == frames


@pytest.mark.unit
class TestSynthesisModuleEmitsImageParts:
    """The deep-research synthesis signature sends image_url parts to the LM."""

    def test_two_frames_produce_two_image_parts(self, jpg_path):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        dspy.ChainOfThought(SynthesisSignature)(
            query="q",
            evidence="e",
            keyframes=[dspy.Image(jpg_path), dspy.Image(jpg_path)],
        )
        assert _count_image_parts(lm.messages[-1]) == 2

    def test_no_frames_produce_zero_image_parts(self):
        lm = _CapturingLM()
        dspy.configure(lm=lm)
        dspy.ChainOfThought(SynthesisSignature)(query="q", evidence="e", keyframes=[])
        assert _count_image_parts(lm.messages[-1]) == 0


@pytest.fixture
def deep_research_agent(jpg_path):
    agent = DeepResearchAgent(
        deps=DeepResearchDeps(
            tenant_id="acme:acme",
            multimodal_generation_enabled=True,
            max_keyframes_to_llm=4,
        )
    )
    agent._keyframe_resolver = KeyframeImageResolver(FakeLocator(jpg_path))
    return agent


@pytest.mark.unit
@pytest.mark.asyncio
class TestDeepResearchSynthesizeGate:
    """_synthesize flattens evidence hits and forwards frames under the gate."""

    async def _run(self, agent, evidence):
        agent.call_dspy = AsyncMock(return_value=Mock(summary="ok"))
        await agent._synthesize("q", evidence)
        return agent.call_dspy.call_args.kwargs["keyframes"]

    async def test_enabled_forwards_frames_from_nested_hits(self, deep_research_agent):
        evidence = [{"question": "q", "results": [_video_hit(1), _video_hit(2)]}]
        frames = await self._run(deep_research_agent, evidence)
        assert len(frames) == 2
        assert all(isinstance(f, dspy.Image) for f in frames)

    async def test_capped_at_max_keyframes(self, deep_research_agent):
        deep_research_agent.max_keyframes_to_llm = 1
        evidence = [
            {"question": "q", "results": [_video_hit(1), _video_hit(2), _video_hit(3)]}
        ]
        assert len(await self._run(deep_research_agent, evidence)) == 1

    async def test_disabled_flag_forwards_no_frames(self, deep_research_agent):
        deep_research_agent.multimodal_generation_enabled = False
        evidence = [{"question": "q", "results": [_video_hit(1)]}]
        assert await self._run(deep_research_agent, evidence) == []

    async def test_non_dict_results_forward_no_frames(self, deep_research_agent):
        evidence = [{"question": "q", "results": "a plain string, not hits"}]
        assert await self._run(deep_research_agent, evidence) == []


@pytest.mark.unit
class TestAnswerAgentResolverTargetsObjectStore:
    """Each answer agent must build its keyframe resolver against the
    SystemConfig object-store endpoint so s3:// keyframes are fetchable at answer
    time; a bare MediaConfig only handles file:// and errors on every keyframe."""

    @staticmethod
    def _config_manager_with_minio(endpoint):
        cm = Mock()
        sys_cfg = Mock()
        sys_cfg.minio_endpoint = endpoint
        cm.get_system_config.return_value = sys_cfg
        return cm

    def test_deep_research_resolver_targets_object_store(self):
        agent = DeepResearchAgent(
            deps=DeepResearchDeps(tenant_id="acme:acme"),
            config_manager=self._config_manager_with_minio("http://minio-x:9000"),
        )
        s3 = agent._keyframe_resolver._locator.config.s3
        assert s3.endpoint_url == "http://minio-x:9000"

    def test_summarizer_resolver_targets_object_store(self):
        with patch("cogniverse_agents.summarizer_agent.VLMInterface"):
            agent = SummarizerAgent(
                deps=SummarizerDeps(),
                config_manager=self._config_manager_with_minio("http://minio-x:9000"),
            )
        s3 = agent._keyframe_resolver._locator.config.s3
        assert s3.endpoint_url == "http://minio-x:9000"

    def test_detailed_report_resolver_targets_object_store(self):
        with patch("cogniverse_agents.detailed_report_agent.VLMInterface"):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(),
                config_manager=self._config_manager_with_minio("http://minio-x:9000"),
            )
        s3 = agent._keyframe_resolver._locator.config.s3
        assert s3.endpoint_url == "http://minio-x:9000"

    def test_no_minio_endpoint_yields_file_only_resolver(self):
        agent = DeepResearchAgent(
            deps=DeepResearchDeps(tenant_id="acme:acme"),
            config_manager=self._config_manager_with_minio(""),
        )
        assert agent._keyframe_resolver._locator.config.s3.endpoint_url is None


class _ThreadRecordingLocator(FakeLocator):
    """Records the thread each localize() runs on, so tests can prove frame
    fetches happen off the event loop."""

    def __init__(self, jpg):
        super().__init__(jpg)
        self.threads: list = []

    def localize(self, uri):
        self.threads.append(threading.get_ident())
        return super().localize(uri)


@pytest.mark.unit
class TestResolverCacheThreadSafety:
    def test_concurrent_collect_is_threadsafe_and_cache_stays_bounded(self, jpg_path):
        """collect() runs from asyncio.to_thread workers, so concurrent
        requests hammer one resolver from many threads. With the cache bound
        far below the working set, every thread constantly evicts while others
        read/insert — the cache must stay consistent and bounded, and every
        collect must still return one image per hit."""
        resolver = KeyframeImageResolver(FakeLocator(jpg_path), cache_size=8)
        hits = [_video_hit(i) for i in range(24)]
        n_threads, rounds = 8, 10
        barrier = threading.Barrier(n_threads)
        failures: list = []
        counts: list = []

        def _work():
            barrier.wait()
            try:
                for _ in range(rounds):
                    counts.append(len(resolver.collect(hits, max_images=24)))
            except Exception as exc:  # noqa: BLE001 — any exception is the bug
                failures.append(exc)

        threads = [threading.Thread(target=_work) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert failures == []
        assert counts == [24] * (n_threads * rounds)
        assert len(resolver._cache) <= 8


@pytest.mark.unit
@pytest.mark.asyncio
class TestCollectRunsOffTheEventLoop:
    """Keyframe collection does blocking object-store downloads; every agent
    answer path must run it in a worker thread — on the loop it stalls every
    concurrent request (and /health/live) for the duration of the downloads."""

    async def test_report_agent_collects_off_loop(self, report_agent, jpg_path):
        locator = _ThreadRecordingLocator(jpg_path)
        report_agent._keyframe_resolver = KeyframeImageResolver(locator)
        report_agent.call_dspy = AsyncMock(return_value=Mock(executive_summary="ok"))
        req = ReportRequest(
            query="q", search_results=[_video_hit(1)], include_visual_analysis=True
        )
        await report_agent._generate_executive_summary(req, _thinking_phase())
        loop_thread = threading.get_ident()
        assert locator.threads, "no keyframe was fetched"
        assert all(t != loop_thread for t in locator.threads)

    async def test_summarizer_collects_off_loop(self, summarizer_agent, jpg_path):
        locator = _ThreadRecordingLocator(jpg_path)
        summarizer_agent._keyframe_resolver = KeyframeImageResolver(locator)
        summarizer_agent._generate_brief_summary = AsyncMock(return_value=("s", ["k"]))
        req = SummaryRequest(
            query="q",
            search_results=[_video_hit(1)],
            include_visual_analysis=True,
            summary_type="brief",
        )
        await summarizer_agent._generate_summary(
            req, Mock(relevance_scores={}), visual_insights=[]
        )
        loop_thread = threading.get_ident()
        assert locator.threads, "no keyframe was fetched"
        assert all(t != loop_thread for t in locator.threads)

    async def test_deep_research_collects_off_loop(self, deep_research_agent, jpg_path):
        locator = _ThreadRecordingLocator(jpg_path)
        deep_research_agent._keyframe_resolver = KeyframeImageResolver(locator)
        deep_research_agent.call_dspy = AsyncMock(return_value=Mock(summary="ok"))
        await deep_research_agent._synthesize(
            "q", [{"question": "q", "results": [_video_hit(1)]}]
        )
        loop_thread = threading.get_ident()
        assert locator.threads, "no keyframe was fetched"
        assert all(t != loop_thread for t in locator.threads)
