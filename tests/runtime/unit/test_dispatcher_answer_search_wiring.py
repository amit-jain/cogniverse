"""The dispatcher feeds answer agents real search results, not an empty list.

``_execute_detailed_report_task`` / ``_execute_summarization_task`` previously
built their ``ReportRequest`` / ``SummaryRequest`` with ``search_results=[]``, so
a directly-dispatched report/summary was ungrounded and answer-time keyframe
injection had no hits to resolve. These tests pin that the dispatch now grounds
the answer in results the caller threaded through ``context["search_results"]``
when present, else in a fresh search, and that a threaded set skips the redundant
search.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher, _flatten_search_hit


@pytest.fixture
def dispatcher():
    sys_cfg = MagicMock()
    sys_cfg.backend_url = "http://localhost"
    sys_cfg.backend_port = 8080
    config_manager = MagicMock()
    config_manager.get_system_config.return_value = sys_cfg
    return AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )


@dataclass
class _FakeReport:
    executive_summary: str = "ok"
    detailed_findings: list = None
    visual_analysis: list = None
    technical_details: list = None
    recommendations: list = None
    confidence_assessment: dict = None
    thinking_phase: dict = None
    metadata: dict = None


def _s3_hit(seg):
    return {
        "document_id": f"v_{seg}",
        "score": 0.9,
        "metadata": {
            "source_url": "s3://cogniverse-ingest/acme:acme/vid.mp4",
            "video_id": "vid",
            "segment_id": seg,
        },
    }


class _CaptureAgent:
    """Stands in for DetailedReportAgent/SummarizerAgent, capturing the request
    its generate/summarize method receives so the test can assert what the
    dispatch fed it."""

    captured = {}

    def __init__(self, *args, **kwargs):
        pass

    async def generate_report(self, request):
        _CaptureAgent.captured["request"] = request
        return _FakeReport()

    async def summarize(self, request):
        _CaptureAgent.captured["request"] = request
        return _FakeReport()


@pytest.mark.unit
@pytest.mark.asyncio
class TestResolveAnswerSearchResults:
    async def test_uses_threaded_context_results_without_searching(self, dispatcher):
        threaded = [_s3_hit(0), _s3_hit(1)]
        searched = []

        async def _no_search(*a, **k):
            searched.append(1)
            return {"results": []}

        dispatcher._execute_search_task = _no_search
        out = await dispatcher._resolve_answer_search_results(
            "q", "acme:acme", {"search_results": threaded}, top_k=20
        )
        assert out == [_flatten_search_hit(h) for h in threaded]
        assert searched == [], (
            "threaded results present — must not run a redundant search"
        )

    async def test_searches_when_context_has_no_results(self, dispatcher):
        hits = [_s3_hit(3)]

        async def _search(query, tenant_id, top_k, **kw):
            assert top_k == 20
            return {"results": hits}

        dispatcher._execute_search_task = _search
        out = await dispatcher._resolve_answer_search_results(
            "q", "acme:acme", None, top_k=20
        )
        assert out == [_flatten_search_hit(h) for h in hits]

    async def test_empty_threaded_list_falls_through_to_search(self, dispatcher):
        hits = [_s3_hit(4)]

        async def _search(*a, **k):
            return {"results": hits}

        dispatcher._execute_search_task = _search
        out = await dispatcher._resolve_answer_search_results(
            "q", "acme:acme", {"search_results": []}, top_k=10
        )
        assert out == [_flatten_search_hit(h) for h in hits]

    async def test_non_dict_threaded_items_fall_through_to_search(self, dispatcher):
        """An external caller's context["search_results"] of non-dicts must not
        crash the answer agent — filter them out and run a fresh search."""
        hits = [_s3_hit(6)]

        async def _search(*a, **k):
            return {"results": hits}

        dispatcher._execute_search_task = _search
        out = await dispatcher._resolve_answer_search_results(
            "q", "acme:acme", {"search_results": ["foo", "bar", 1]}, top_k=10
        )
        assert out == [_flatten_search_hit(h) for h in hits]

    async def test_search_failure_degrades_to_empty(self, dispatcher):
        """A report/summary request is not inherently a video-search request —
        an unreachable search backend must degrade to an ungrounded answer over
        [], not hard-fail the whole request (e.g. a plain conversational summary
        when the video-search embedding service is down)."""

        async def _boom(*a, **k):
            raise RuntimeError("embedding service unreachable")

        dispatcher._execute_search_task = _boom
        out = await dispatcher._resolve_answer_search_results(
            "explain deep learning", "acme:acme", None, top_k=10
        )
        assert out == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchUsesActiveVideoProfile:
    """_execute_search_task must search the tenant's configured
    ``active_video_profile``, not the hardcoded fallback. The prior code read a
    phantom ``default_profile`` key that never existed in config, so every
    dispatcher-driven search silently ignored the tenant's profile."""

    async def test_search_uses_configured_active_video_profile(
        self, dispatcher, monkeypatch
    ):
        fake_config = MagicMock()
        fake_config.get = lambda key, default=None: (
            "video_custom_mv_frame" if key == "active_video_profile" else default
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config",
            lambda **kwargs: fake_config,
        )
        captured = {}

        def _capture(profile):
            captured["profile"] = profile
            agent = MagicMock()

            async def _process(_inp):
                out = MagicMock()
                out.results = []
                out.enhanced_query = None
                out.profile = profile
                out.search_mode = "single_profile"
                return out

            agent._process_impl = _process
            return agent

        dispatcher._get_search_agent = _capture
        dispatcher.consult_egress_policy = lambda *a, **k: None
        dispatcher._verify_search_egress = lambda *a, **k: None
        dispatcher._apply_artefact_overlay = lambda *a, **k: None

        await dispatcher._execute_search_task("robots", "acme:acme", top_k=5)
        assert captured["profile"] == "video_custom_mv_frame"


@pytest.mark.unit
@pytest.mark.asyncio
class TestAnswerTasksFeedSearchedResults:
    async def test_detailed_report_task_feeds_searched_hits(
        self, dispatcher, monkeypatch
    ):
        hits = [_s3_hit(0), _s3_hit(1)]

        async def _search(query, tenant_id, top_k, **kw):
            return {"results": hits}

        dispatcher._execute_search_task = _search
        dispatcher._init_agent_memory = lambda *a, **k: None
        _CaptureAgent.captured = {}
        monkeypatch.setattr(
            "cogniverse_agents.detailed_report_agent.DetailedReportAgent",
            _CaptureAgent,
        )
        await dispatcher._execute_detailed_report_task("q", "acme:acme")
        assert _CaptureAgent.captured["request"].search_results == [
            _flatten_search_hit(h) for h in hits
        ]

    async def test_summarization_task_feeds_searched_hits(
        self, dispatcher, monkeypatch
    ):
        hits = [_s3_hit(2)]

        async def _search(query, tenant_id, top_k, **kw):
            return {"results": hits}

        dispatcher._execute_search_task = _search
        dispatcher._init_agent_memory = lambda *a, **k: None
        dispatcher.consult_egress_policy = lambda *a, **k: None
        dispatcher._verify_summarizer_egress = lambda *a, **k: None
        dispatcher._apply_artefact_overlay = lambda *a, **k: None
        _CaptureAgent.captured = {}
        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent", _CaptureAgent
        )
        await dispatcher._execute_summarization_task("q", "acme:acme")
        assert _CaptureAgent.captured["request"].search_results == [
            _flatten_search_hit(h) for h in hits
        ]

    async def test_detailed_report_task_prefers_threaded_context_hits(
        self, dispatcher, monkeypatch
    ):
        threaded = [_s3_hit(5)]

        async def _search(*a, **k):
            raise AssertionError("must not search when context carries results")

        dispatcher._execute_search_task = _search
        dispatcher._init_agent_memory = lambda *a, **k: None
        _CaptureAgent.captured = {}
        monkeypatch.setattr(
            "cogniverse_agents.detailed_report_agent.DetailedReportAgent",
            _CaptureAgent,
        )
        await dispatcher._execute_detailed_report_task(
            "q", "acme:acme", context={"search_results": threaded}
        )
        assert _CaptureAgent.captured["request"].search_results == [
            _flatten_search_hit(h) for h in threaded
        ]


@pytest.mark.unit
class TestFlattenSearchHit:
    """The answer agents' text helpers read title/description/video_id at the top
    level; a SearchResult/gateway hit nests them (renamed) under metadata.
    _flatten_search_hit lifts + aliases them so the retrieved text is not lost."""

    def test_lifts_metadata_and_aliases_names(self):
        hit = {
            "id": "vidX_3",
            "document_id": "vidX_3",
            "score": 0.87,
            "metadata": {
                "video_id": "vidX",
                "video_title": "Bushcraft Basics",
                "audio_transcript": "carving a feather stick",
                "source_url": "s3://bucket/acme:acme/vidX.mp4",
                "segment_id": 3,
            },
        }
        flat = _flatten_search_hit(hit)
        # top-level reads the answer agents perform now resolve
        assert flat["video_id"] == "vidX"
        assert flat["title"] == "Bushcraft Basics"
        assert flat["description"] == "carving a feather stick"
        assert flat["text_content"] == "carving a feather stick"
        assert flat["source_url"] == "s3://bucket/acme:acme/vidX.mp4"
        # identity/score preserved, and metadata kept for keyframe resolution
        assert flat["score"] == 0.87
        assert flat["metadata"]["source_url"] == "s3://bucket/acme:acme/vidX.mp4"

    def test_top_level_fields_win_over_metadata(self):
        hit = {"title": "explicit", "score": 0.5, "metadata": {"video_title": "meta"}}
        assert _flatten_search_hit(hit)["title"] == "explicit"

    def test_hit_without_metadata_returned_unchanged(self):
        hit = {"video_id": "v", "source_url": "s3://b/t/v.mp4", "segment_id": 0}
        assert _flatten_search_hit(hit) is hit


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamingCapabilityOrdering:
    """A streamed detailed_report request must build a DetailedReportAgent, not a
    SummarizerAgent — detailed_report_agent is registered (config.json) with a
    'text_generation' capability, which the summarization branch also matches."""

    async def test_detailed_report_capabilities_build_report_agent(
        self, dispatcher, monkeypatch
    ):
        class _ReportStub:
            def __init__(self, *a, **k):
                pass

        class _SummaryStub:
            def __init__(self, *a, **k):
                pass

        async def _no_search(*a, **k):
            return {"results": []}

        dispatcher._execute_search_task = _no_search
        entry = MagicMock()
        entry.capabilities = ["detailed_report", "analysis", "text_generation"]
        dispatcher._registry.get_agent = MagicMock(return_value=entry)
        monkeypatch.setattr(
            "cogniverse_agents.detailed_report_agent.DetailedReportAgent", _ReportStub
        )
        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent", _SummaryStub
        )

        agent, typed_input = await dispatcher.create_streaming_agent(
            "detailed_report_agent", "report on the clip", "acme:acme"
        )
        assert isinstance(agent, _ReportStub)
        assert type(typed_input).__name__ == "DetailedReportInput"


class TestAgentBehaviorConfigWiring:
    """The dispatcher threads per-tenant thinking_enabled / visual_analysis_enabled
    from the persisted AgentConfig into the summarizer / detailed-report Deps.
    Before this, those config fields were persistable but never reached the
    agents (the dispatcher built the Deps with tenant_id only)."""

    def _dispatcher_with_real_config(self):
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        cm = ConfigManager(store=store)
        dispatcher = AgentDispatcher(
            agent_registry=MagicMock(),
            config_manager=cm,
            schema_loader=MagicMock(),
        )
        return dispatcher, cm

    def _agent_config(self, name, **behavior):
        from cogniverse_foundation.config.agent_config import (
            AgentConfig,
            DSPyModuleType,
            ModuleConfig,
        )

        return AgentConfig(
            agent_name=name,
            agent_version="1.0.0",
            agent_description="test",
            agent_url="http://x",
            capabilities=["summarization"],
            skills=[],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.PREDICT, signature="S"
            ),
            **behavior,
        )

    def test_per_tenant_toggles_reach_deps_kwargs(self):
        dispatcher, cm = self._dispatcher_with_real_config()
        cm.set_agent_config(
            tenant_id="acme:acme",
            agent_name="summarizer_agent",
            agent_config=self._agent_config(
                "summarizer_agent",
                thinking_enabled=False,
                visual_analysis_enabled=False,
            ),
        )

        kwargs = dispatcher._agent_behavior_kwargs("acme:acme", "summarizer_agent")
        assert kwargs == {
            "thinking_enabled": False,
            "visual_analysis_enabled": False,
        }

    def test_no_per_tenant_config_yields_defaults(self):
        dispatcher, _ = self._dispatcher_with_real_config()
        # No config set → empty kwargs so the Deps field defaults (True) apply.
        assert (
            dispatcher._agent_behavior_kwargs("acme:acme", "detailed_report_agent")
            == {}
        )
