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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.gateway_agent import GatewayAgent as RealGatewayAgent
from cogniverse_agents.search_agent import SearchAgent as RealSearchAgent
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

    async def process(self, typed_input):
        from cogniverse_agents.detailed_report_agent import DetailedReportOutput

        _CaptureAgent.captured["request"] = typed_input
        return DetailedReportOutput(executive_summary="ok")

    async def summarize(self, request):
        _CaptureAgent.captured["request"] = request
        return _FakeReport()


class _SearchAgentStub(RealSearchAgent):
    def __init__(self, profile):
        self.telemetry_manager = None
        self._input_rails = None
        self._output_rails = None
        self._profile = profile

    async def _process_impl(self, _inp):
        return SimpleNamespace(
            results=[],
            enhanced_query=None,
            profile=self._profile,
            search_mode="single_profile",
        )


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
            return _SearchAgentStub(profile)

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

    def test_lifts_document_title_and_full_text(self):
        """A document-profile hit nests document_title/full_text; without lifting
        them the summarizer rendered every source as 'Unknown' with no content."""
        hit = {
            "id": "a1",
            "score": 5.04,
            "metadata": {
                "document_id": "a1",
                "document_title": "Quarterly Report",
                "full_text": "Revenue grew 12% in the zephyrite division.",
            },
        }
        flat = _flatten_search_hit(hit)
        assert flat["title"] == "Quarterly Report"
        assert flat["description"] == "Revenue grew 12% in the zephyrite division."
        assert flat["text_content"] == "Revenue grew 12% in the zephyrite division."

    def test_lifts_audio_title(self):
        hit = {
            "id": "aud1",
            "score": 0.9,
            "metadata": {"audio_title": "Keynote", "audio_transcript": "welcome all"},
        }
        flat = _flatten_search_hit(hit)
        assert flat["title"] == "Keynote"
        assert flat["description"] == "welcome all"

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

    @pytest.mark.asyncio
    async def test_dispatch_threads_persisted_toggles_into_summarizer_deps(
        self, monkeypatch
    ):
        """Dispatch-level pin: the persisted per-tenant toggles must land in
        the CONSTRUCTED SummarizerDeps — dropping the kwargs splat at the
        construction site would keep the helper test green while every
        dispatch silently used the Deps defaults."""
        from unittest.mock import AsyncMock

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
        entry = MagicMock()
        entry.capabilities = ["summarization"]
        dispatcher._registry.get_agent.return_value = entry

        captured = {}

        class _CapturingSummarizer:
            def __init__(self, deps, config_manager=None):
                captured["deps"] = deps

        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent",
            _CapturingSummarizer,
        )
        monkeypatch.setattr(
            dispatcher, "_resolve_answer_search_results", AsyncMock(return_value=[])
        )

        agent, typed_input = await dispatcher.create_streaming_agent(
            "summarizer_agent", "summarize the clips", "acme:acme"
        )

        assert isinstance(agent, _CapturingSummarizer)
        assert captured["deps"].thinking_enabled is False
        assert captured["deps"].visual_analysis_enabled is False
        assert captured["deps"].tenant_id == "acme:acme"


@pytest.mark.unit
@pytest.mark.asyncio
class TestGatewayTopKForwarding:
    """A caller's top_k must reach the downstream agent through the gateway
    routing branch. It was dropped there — _execute_gateway_task read
    context['top_k'] (a key nobody writes) and defaulted to 10, so a request
    for 100 results was silently capped at 10."""

    async def test_dispatch_forwards_top_k_through_gateway_branch(self, dispatcher):
        from cogniverse_agents.gateway_agent import GatewayOutput

        d = dispatcher
        gateway_agent = MagicMock()
        gateway_agent.capabilities = {"gateway"}
        d._registry.get_agent.return_value = gateway_agent

        d.consult_egress_policy = lambda *a, **k: None
        d._verify_routing_egress = lambda *a, **k: None
        d._get_rail_chains = lambda tenant_id: None
        d._spawn_background = lambda coro: coro.close()

        routed = GatewayOutput(
            query="find every matching clip",
            complexity="simple",
            routed_to="search_agent",
            modality="video",
            generation_type="raw_results",
            confidence=0.9,
            fast_path_confidence_threshold=0.4,
            gliner_threshold=0.3,
            reasoning="keyword route",
        )

        class _GW(RealGatewayAgent):
            def __init__(self):
                self.telemetry_manager = None
                self._input_rails = None
                self._output_rails = None

            async def _process_impl(self, _input):
                return routed

        async def _build_gw(_tenant_id):
            return _GW()

        d._get_or_build_gateway_agent = _build_gw

        seen = {}

        async def _downstream(
            agent_name,
            query,
            tenant_id,
            top_k,
            conversation_history,
            enrichment=None,
            image_b64=None,
            context=None,
        ):
            seen["top_k"] = top_k
            seen["agent_name"] = agent_name
            return {"status": "success", "message": "ok"}

        d._execute_downstream_agent = _downstream

        await d.dispatch(
            agent_name="gateway_agent",
            query="find every matching clip",
            context={"tenant_id": "acme:acme"},
            top_k=37,
        )

        assert seen["agent_name"] == "search_agent"
        assert seen["top_k"] == 37  # not the dropped default of 10


@pytest.mark.unit
@pytest.mark.asyncio
class TestHistoryRewriteOnModalityBranches:
    """The image/audio/document/deep-research branches must resolve an
    anaphoric follow-up against conversation history — they dropped it before,
    so "more of those" searched for the literal words, not the prior entity."""

    async def _run_branch(self, dispatcher, capability, exec_attr):
        d = dispatcher
        agent = MagicMock()
        agent.capabilities = {capability}
        d._registry.get_agent.return_value = agent
        d._spawn_background = lambda coro: coro.close()

        async def _fake_rewrite(query, history):
            return f"resolved:{query}:from:{history[-1]['content']}"

        d._rewrite_query_with_history = _fake_rewrite

        seen = {}

        async def _exec(query, tenant_id, top_k, image_b64=None):
            seen["query"] = query
            return {"status": "success", "message": "ok"}

        setattr(d, exec_attr, _exec)

        await d.dispatch(
            agent_name="x",
            query="more of those",
            context={
                "tenant_id": "acme:acme",
                "conversation_history": [{"role": "user", "content": "cats"}],
            },
        )
        return seen

    async def test_image_branch_rewrites_query_with_history(self, dispatcher):
        seen = await self._run_branch(
            dispatcher, "image_search", "_execute_image_search_task"
        )
        assert seen["query"] == "resolved:more of those:from:cats"

    async def test_audio_branch_rewrites_query_with_history(self, dispatcher):
        seen = await self._run_branch(
            dispatcher, "audio_analysis", "_execute_audio_search_task"
        )
        assert seen["query"] == "resolved:more of those:from:cats"

    async def test_document_branch_rewrites_query_with_history(self, dispatcher):
        seen = await self._run_branch(
            dispatcher, "document_analysis", "_execute_document_search_task"
        )
        assert seen["query"] == "resolved:more of those:from:cats"

    async def test_no_history_leaves_query_unchanged(self, dispatcher):
        d = dispatcher
        agent = MagicMock()
        agent.capabilities = {"image_search"}
        d._registry.get_agent.return_value = agent
        d._spawn_background = lambda coro: coro.close()

        async def _fail_rewrite(query, history):
            raise AssertionError("must not rewrite when there is no history")

        d._rewrite_query_with_history = _fail_rewrite
        seen = {}

        async def _img(query, tenant_id, top_k, image_b64=None):
            seen["query"] = query
            return {"status": "success", "message": "ok"}

        d._execute_image_search_task = _img
        await d.dispatch(
            agent_name="x", query="find cats", context={"tenant_id": "acme:acme"}
        )
        assert seen["query"] == "find cats"

    async def test_downstream_agent_threads_enrichment_to_search(self, dispatcher):
        """The router's enrichment (entities, enhanced_query, ...) must reach
        the search task through _execute_downstream_agent, not be dropped."""
        d = dispatcher
        agent = MagicMock()
        agent.capabilities = {"search"}
        d._registry.get_agent.return_value = agent

        seen = {}

        async def _search(
            query,
            tenant_id,
            top_k,
            conversation_history=None,
            enrichment=None,
            context=None,
        ):
            seen["enrichment"] = enrichment
            return {"status": "success", "message": "ok"}

        d._execute_search_task = _search
        enrichment = {"entities": ["cats"], "enhanced_query": "cats playing"}

        await d._execute_downstream_agent(
            agent_name="search_agent",
            query="q",
            tenant_id="acme:acme",
            enrichment=enrichment,
        )
        assert seen["enrichment"] == enrichment


@pytest.mark.unit
@pytest.mark.asyncio
class TestDownstreamDispatchThreadsRequestContext:
    """The gateway "simple" path routes through ``_execute_downstream_agent``.
    It must pass the request context to the execution tasks: the canary/variant
    artefact overlay travels in ``context["_artefact_overlay"]`` and threaded
    ``context["search_results"]`` ground the answer agents. Dropping the
    context makes both silently no-op for every gateway-routed request while
    direct ``/agents/{name}/process`` dispatch still applies them.
    """

    async def test_overlay_reaches_search_agent_via_downstream_dispatch(
        self, dispatcher, monkeypatch
    ):
        fake_config = MagicMock()
        fake_config.get = lambda key, default=None: default
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config",
            lambda **kwargs: fake_config,
        )
        stub = _SearchAgentStub("video_colpali_smol500_mv_frame")
        dispatcher._get_search_agent = lambda profile: stub
        dispatcher.consult_egress_policy = lambda *a, **k: None
        dispatcher._verify_search_egress = lambda *a, **k: None
        dispatcher._registry.get_agent = lambda name: SimpleNamespace(
            capabilities=["search"]
        )

        overlay = {
            "prompts": {"query_enhancement": "canary prompt"},
            "served_from": "canary",
            "version": 7,
            "variant_id": "v-canary-7",
        }
        try:
            await dispatcher._execute_downstream_agent(
                agent_name="search_agent",
                query="robots",
                tenant_id="acme:acme",
                top_k=5,
                context={"_artefact_overlay": overlay},
            )
            # Real MemoryAwareMixin state, set by the real
            # _apply_artefact_overlay inside the real _execute_search_task.
            assert stub.get_dispatched_artefact() is overlay
        finally:
            stub.set_dispatched_artefact(None)

    async def test_downstream_report_dispatch_threads_request_context(self, dispatcher):
        captured = {}

        async def _spy(query, tenant_id, context=None):
            captured["context"] = context
            return {"status": "success", "agent": "detailed_report_agent"}

        dispatcher._execute_detailed_report_task = _spy
        dispatcher._registry.get_agent = lambda name: SimpleNamespace(
            capabilities=["detailed_report"]
        )
        ctx = {"search_results": [_s3_hit(9)], "_artefact_overlay": None}
        await dispatcher._execute_downstream_agent(
            agent_name="detailed_report_agent",
            query="q",
            tenant_id="acme:acme",
            context=ctx,
        )
        assert captured["context"] is ctx


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamingInputCarriesContext:
    """create_streaming_agent must build the same typed input from context
    that the non-streaming dispatch builds. The search, coding, and
    orchestrator branches constructed inputs from query and tenant_id only,
    so a streamed request lost top_k, enrichment, codebase_path,
    max_iterations, language, session_id, conversation_history and the
    gateway-shaped modality hints its caller threaded through context."""

    async def test_streaming_search_input_carries_context_fields(
        self, dispatcher, monkeypatch
    ):
        d = dispatcher
        d._registry.get_agent.return_value = SimpleNamespace(capabilities=["search"])

        class _StubSearchAgent:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            "cogniverse_agents.search_agent.SearchAgent", _StubSearchAgent
        )

        agent, typed_input = await d.create_streaming_agent(
            "search_agent",
            "find the demo",
            "acme:acme",
            context={
                "top_k": 25,
                "enhanced_query": "find the product demo",
                "entities": [{"text": "demo", "label": "PRODUCT"}],
                "relationships": [
                    {"subject": "demo", "relation": "shows", "object": "product"}
                ],
                "query_variants": [{"query": "product demo video"}],
                "profiles": ["video_colpali_smol500_mv_frame"],
            },
        )

        assert isinstance(agent, _StubSearchAgent)
        assert typed_input.query == "find the demo"
        assert typed_input.tenant_id == "acme:acme"
        assert typed_input.top_k == 25
        assert typed_input.enhanced_query == "find the product demo"
        assert typed_input.entities == [{"text": "demo", "label": "PRODUCT"}]
        assert typed_input.relationships == [
            {"subject": "demo", "relation": "shows", "object": "product"}
        ]
        assert typed_input.query_variants == [{"query": "product demo video"}]
        assert typed_input.profiles == ["video_colpali_smol500_mv_frame"]

    async def test_streaming_coding_input_carries_context_fields(
        self, dispatcher, monkeypatch
    ):
        d = dispatcher
        d._registry.get_agent.return_value = SimpleNamespace(capabilities=["coding"])

        class _StubCodingAgent:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            "cogniverse_agents.coding_agent.CodingAgent", _StubCodingAgent
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config",
            lambda tenant_id=None, config_manager=None: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.create_routed_lm",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.resolve_semantic_router_config",
            lambda *args, **kwargs: MagicMock(),
        )

        agent, typed_input = await d.create_streaming_agent(
            "coding_agent",
            "add a retry helper",
            "acme:acme",
            context={
                "codebase_path": "/repo/svc",
                "max_iterations": 9,
                "language": "go",
            },
        )

        assert isinstance(agent, _StubCodingAgent)
        assert typed_input.task == "add a retry helper"
        assert typed_input.tenant_id == "acme:acme"
        assert typed_input.codebase_path == "/repo/svc"
        assert typed_input.max_iterations == 9
        assert typed_input.language == "go"

    async def test_streaming_orchestrator_input_carries_context_fields(
        self, dispatcher, monkeypatch
    ):
        from unittest.mock import AsyncMock

        d = dispatcher
        d._registry.get_agent.return_value = SimpleNamespace(
            capabilities=["orchestration"]
        )
        stub_agent = MagicMock()
        monkeypatch.setattr(
            d, "_get_or_build_orchestrator", AsyncMock(return_value=stub_agent)
        )

        history = [{"role": "user", "content": "show me the two demos"}]
        agent, typed_input = await d.create_streaming_agent(
            "orchestrator_agent",
            "compare the two demos",
            "acme:acme",
            context={
                "session_id": "s-1",
                "conversation_history": history,
                "detected_modalities": ["video", "text"],
                "gateway_context": {
                    "modality": "video",
                    "generation_type": "raw_results",
                    "synthesis_depth": "detailed",
                },
            },
        )

        assert agent is stub_agent
        assert typed_input.query == "compare the two demos"
        assert typed_input.tenant_id == "acme:acme"
        assert typed_input.session_id == "s-1"
        assert typed_input.conversation_history == history
        assert typed_input.detected_modalities == ["video", "text"]
        assert typed_input.modality == "video"
        assert typed_input.generation_type == "raw_results"
        assert typed_input.synthesis_depth == "detailed"

    async def test_streaming_orchestrator_plain_synthesis_depth(
        self, dispatcher, monkeypatch
    ):
        from unittest.mock import AsyncMock

        d = dispatcher
        d._registry.get_agent.return_value = SimpleNamespace(
            capabilities=["orchestration"]
        )
        monkeypatch.setattr(
            d, "_get_or_build_orchestrator", AsyncMock(return_value=MagicMock())
        )

        _, typed_input = await d.create_streaming_agent(
            "orchestrator_agent",
            "compare the two demos",
            "acme:acme",
            context={"synthesis_depth": "exhaustive"},
        )

        assert typed_input.synthesis_depth == "exhaustive"


@pytest.mark.unit
@pytest.mark.asyncio
class TestRlmThreadsIntoTypedInputs:
    """The orchestrator's RLM promotion stamps ``rlm`` on the sub-agent
    payload; each promotable execution seam must feed context["rlm"] into
    the typed input, or the promoted flag dies between the HTTP boundary
    and the agent and RLM synthesis never activates."""

    _RLM = {"enabled": True, "auto_detect": True, "context_threshold": 50_000}

    def _expected_options(self):
        from cogniverse_core.agents.rlm_options import RLMOptions

        return RLMOptions(enabled=True, auto_detect=True, context_threshold=50_000)

    async def test_search_task_threads_rlm_into_input(self, dispatcher, monkeypatch):
        fake_config = MagicMock()
        fake_config.get = lambda key, default=None: default
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config",
            lambda **kwargs: fake_config,
        )
        captured = {}

        class _CapturingSearchStub(_SearchAgentStub):
            async def _process_impl(self, inp):
                captured["input"] = inp
                return await super()._process_impl(inp)

        dispatcher._get_search_agent = lambda profile: _CapturingSearchStub(profile)
        dispatcher.consult_egress_policy = lambda *a, **k: None
        dispatcher._verify_search_egress = lambda *a, **k: None

        await dispatcher._execute_search_task(
            "robots", "acme:acme", 5, context={"rlm": dict(self._RLM)}
        )

        assert captured["input"].rlm == self._expected_options()

    async def test_coding_task_threads_rlm_into_input(self, dispatcher, monkeypatch):
        captured = {}

        class _StubCodingAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def process(self, input):
                captured["input"] = input
                return SimpleNamespace(model_dump=lambda: {"ok": True})

        monkeypatch.setattr(
            "cogniverse_agents.coding_agent.CodingAgent", _StubCodingAgent
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config",
            lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.create_routed_lm",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "cogniverse_foundation.config.semantic_router.resolve_semantic_router_config",
            lambda *args, **kwargs: MagicMock(),
        )
        dispatcher._init_agent_memory = lambda *args, **kwargs: None

        await dispatcher._execute_coding_task(
            "add a retry helper", "acme:acme", context={"rlm": dict(self._RLM)}
        )

        assert captured["input"].rlm == self._expected_options()

    async def test_deep_research_task_threads_rlm_into_input(
        self, dispatcher, monkeypatch
    ):
        captured = {}

        class _StubResearchAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def process(self, input):
                captured["input"] = input
                return SimpleNamespace(model_dump=lambda: {"ok": True})

        monkeypatch.setattr(
            "cogniverse_agents.deep_research_agent.DeepResearchAgent",
            _StubResearchAgent,
        )
        dispatcher._init_agent_memory = lambda *args, **kwargs: None

        await dispatcher._execute_deep_research_task(
            "trace the supply chain", "acme:acme", context={"rlm": dict(self._RLM)}
        )

        assert captured["input"].rlm == self._expected_options()


@pytest.mark.unit
class TestTypedInputFromContext:
    """Every context key naming a declared input field reaches the agent."""

    def test_search_input_modality_rrf_k_and_dates_reachable(self):
        from cogniverse_agents.search_agent import SearchInput
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            SearchInput,
            query="robots",
            tenant_id="acme:prod",
            context={
                "query": "context copy must lose",
                "tenant_id": "evil:tenant",
                "modality": "image",
                "rrf_k": 30,
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "top_k": 25,
                "profiles": ["image_colpali_mv"],
                "unknown_key": "dropped",
            },
        )
        assert typed == SearchInput(
            query="robots",
            tenant_id="acme:prod",
            modality="image",
            rrf_k=30,
            start_date="2026-01-01",
            end_date="2026-02-01",
            top_k=25,
            profiles=["image_colpali_mv"],
        )

    def test_summarizer_summary_type_reachable_and_defaults(self):
        from cogniverse_agents.summarizer_agent import SummarizerInput
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            SummarizerInput,
            query="q",
            tenant_id="t:t",
            context={"summary_type": "bullet_points"},
            search_results=[{"video_id": "v1"}],
        )
        assert typed.summary_type == "bullet_points"
        assert typed.search_results == [{"video_id": "v1"}]

        defaulted = typed_input_from_context(
            SummarizerInput, query="q", tenant_id="t:t", context={}
        )
        assert defaulted.summary_type == "comprehensive"

    def test_detailed_report_rlm_and_enrichment_reachable(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput
        from cogniverse_core.agents.rlm_options import RLMOptions
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            DetailedReportInput,
            query="q",
            tenant_id="t:t",
            context={
                "rlm": {
                    "enabled": True,
                    "auto_detect": True,
                    "context_threshold": 50000,
                },
                "entities": [{"text": "GPU"}],
                "relationships": [{"subject": "a", "object": "b"}],
                "enhanced_query": "better q",
                "report_type": "technical",
            },
            search_results=[{"video_id": "v1"}],
        )
        assert typed.rlm == RLMOptions(
            enabled=True, auto_detect=True, context_threshold=50000
        )
        assert typed.entities == [{"text": "GPU"}]
        assert typed.relationships == [{"subject": "a", "object": "b"}]
        assert typed.enhanced_query == "better q"
        assert typed.report_type == "technical"
        assert typed.search_results == [{"video_id": "v1"}]

    def test_coding_input_built_via_task_override(self):
        from cogniverse_agents.coding_agent import CodingInput
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            CodingInput,
            query="fix the bug",
            tenant_id="t:t",
            context={"codebase_path": "/repo", "max_iterations": 3, "language": "go"},
            task="fix the bug",
        )
        assert typed == CodingInput(
            task="fix the bug",
            tenant_id="t:t",
            codebase_path="/repo",
            max_iterations=3,
            language="go",
        )

    def test_profiles_context_key_fills_available_profiles(self):
        from cogniverse_agents.profile_selection_agent import ProfileSelectionInput
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            ProfileSelectionInput,
            query="q",
            tenant_id="t:t",
            context={"profiles": ["p1", "p2"]},
        )
        assert typed.available_profiles == ["p1", "p2"]

    def test_none_context_values_fall_to_model_defaults(self):
        from cogniverse_agents.summarizer_agent import SummarizerInput
        from cogniverse_runtime.agent_dispatcher import typed_input_from_context

        typed = typed_input_from_context(
            SummarizerInput,
            query="q",
            tenant_id="t:t",
            context={"summary_type": None, "enhanced_query": None},
        )
        assert typed.summary_type == "comprehensive"
        assert typed.enhanced_query is None


class _ProcessCaptureReportAgent:
    """Stands in for DetailedReportAgent, capturing the typed input process()
    receives and returning a real DetailedReportOutput."""

    captured = {}

    def __init__(self, *args, **kwargs):
        pass

    async def process(self, typed_input):
        from cogniverse_agents.detailed_report_agent import DetailedReportOutput

        _ProcessCaptureReportAgent.captured["input"] = typed_input
        return DetailedReportOutput(executive_summary="ES from process")


@pytest.mark.unit
@pytest.mark.asyncio
class TestDetailedReportDispatchThroughProcess:
    """The dispatch task must go through process() (spans, rails, memory
    enrichment, RLM synthesis), not the bare generate_report() bypass, and the
    context must reach the typed input."""

    async def test_task_calls_process_with_context_derived_input(
        self, dispatcher, monkeypatch
    ):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput
        from cogniverse_core.agents.rlm_options import RLMOptions

        hits = [_s3_hit(7)]

        async def _search(query, tenant_id, top_k, **kw):
            return {"results": hits}

        dispatcher._execute_search_task = _search
        dispatcher._init_agent_memory = lambda *a, **k: None
        dispatcher._apply_artefact_overlay = lambda *a, **k: None
        _ProcessCaptureReportAgent.captured = {}
        monkeypatch.setattr(
            "cogniverse_agents.detailed_report_agent.DetailedReportAgent",
            _ProcessCaptureReportAgent,
        )

        result = await dispatcher._execute_detailed_report_task(
            "q",
            "acme:acme",
            context={
                "rlm": {"enabled": True, "auto_detect": True},
                "enhanced_query": "better q",
            },
        )

        typed = _ProcessCaptureReportAgent.captured["input"]
        assert isinstance(typed, DetailedReportInput)
        assert typed.tenant_id == "acme:acme"
        assert typed.rlm == RLMOptions(enabled=True, auto_detect=True)
        assert typed.enhanced_query == "better q"
        assert typed.search_results == [_flatten_search_hit(h) for h in hits]
        assert result["status"] == "success"
        assert result["result"]["executive_summary"] == "ES from process"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSummaryTypeReachableFromContext:
    async def test_summarization_task_threads_summary_type(
        self, dispatcher, monkeypatch
    ):
        async def _search(*a, **k):
            return {"results": []}

        dispatcher._execute_search_task = _search
        dispatcher._init_agent_memory = lambda *a, **k: None
        dispatcher.consult_egress_policy = lambda *a, **k: None
        dispatcher._verify_summarizer_egress = lambda *a, **k: None
        dispatcher._apply_artefact_overlay = lambda *a, **k: None
        _CaptureAgent.captured = {}
        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent", _CaptureAgent
        )
        await dispatcher._execute_summarization_task(
            "q", "acme:acme", context={"summary_type": "brief"}
        )
        assert _CaptureAgent.captured["request"].summary_type == "brief"

    async def test_summarization_task_defaults_to_comprehensive(
        self, dispatcher, monkeypatch
    ):
        async def _search(*a, **k):
            return {"results": []}

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
        assert _CaptureAgent.captured["request"].summary_type == "comprehensive"


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamingInputsDeriveFromContext:
    async def test_streaming_search_input_gets_modality_dates_rrf_k(
        self, dispatcher, monkeypatch
    ):
        entry = MagicMock()
        entry.capabilities = ["search"]
        dispatcher._registry.get_agent = MagicMock(return_value=entry)

        class _SearchStub:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr("cogniverse_agents.search_agent.SearchAgent", _SearchStub)

        agent, typed_input = await dispatcher.create_streaming_agent(
            "search_agent",
            "robots",
            "acme:acme",
            context={
                "modality": "image",
                "rrf_k": 30,
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "top_k": 7,
            },
        )
        assert typed_input.modality == "image"
        assert typed_input.rrf_k == 30
        assert typed_input.start_date == "2026-01-01"
        assert typed_input.end_date == "2026-02-01"
        assert typed_input.top_k == 7
        assert typed_input.tenant_id == "acme:acme"

    async def test_streaming_summarizer_input_gets_summary_type(
        self, dispatcher, monkeypatch
    ):
        async def _no_search(*a, **k):
            return {"results": []}

        dispatcher._execute_search_task = _no_search
        entry = MagicMock()
        entry.capabilities = ["summarization"]
        dispatcher._registry.get_agent = MagicMock(return_value=entry)

        class _SummaryStub:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr(
            "cogniverse_agents.summarizer_agent.SummarizerAgent", _SummaryStub
        )

        agent, typed_input = await dispatcher.create_streaming_agent(
            "summarizer_agent",
            "sum it up",
            "acme:acme",
            context={"summary_type": "bullet_points"},
        )
        assert typed_input.summary_type == "bullet_points"

        agent, defaulted = await dispatcher.create_streaming_agent(
            "summarizer_agent", "sum it up", "acme:acme"
        )
        assert defaulted.summary_type == "comprehensive"
