"""The dispatch envelope carries the answer every consumer reads.

Built on the real dispatcher (real ConfigManager, real AgentRegistry) the way
``test_dispatch_egress_enforcement`` builds it. The module owns no Vespa: the
agent execution seam is driven with envelopes built from the shipped output
types, and the wiki round-trip runs the real ``WikiManager`` body against a
recording document backend. The LM and the embedding sidecar are the only
boundaries stood in for, and neither is provisioned by this lane.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pytest

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager, get_config
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT = "acme:acme"
ANSWER = "The clip shows a cyclist crossing a bridge at dusk."

# Six prior turns put the dispatch at turn 4, which is WikiManager's
# auto-file threshold for an agent that is not on its always-file list.
AUTO_FILE_HISTORY = [
    {"role": "user", "content": "find the red bike"},
    {"role": "assistant", "content": "Clip v_001."},
    {"role": "user", "content": "and the bridge"},
    {"role": "assistant", "content": "Clip v_002."},
    {"role": "user", "content": "at dusk"},
    {"role": "assistant", "content": "Clip v_003."},
]


class _StubSandboxManager:
    """Returns a fixed policy per agent, as SandboxManager does."""

    def __init__(self, policies: Dict[str, Dict[str, Any]]):
        self._policies = policies

    def get_policy(self, agent_name: str) -> Optional[Dict[str, Any]]:
        return self._policies.get(agent_name)


class _BrokenConfigManager:
    """A config manager whose system config carries an unparseable address."""

    def __init__(self, system_config: Any):
        self._system_config = system_config

    def get_system_config(self) -> Any:
        return self._system_config


class _RecordingBackend:
    """Records the documents a WikiManager feeds; owns no schema."""

    def __init__(self) -> None:
        self.fed: List[Dict[str, Any]] = []

    def schema_exists(self, base_schema: str, tenant_id: str) -> bool:
        return False

    def get_document_fields(self, *args, **kwargs):
        return None

    def put_document(self, document, **kwargs):
        self.fed.append({"document": document, "kwargs": kwargs})
        return True


def _dispatcher(
    policies: Optional[Dict[str, Dict[str, Any]]] = None,
) -> AgentDispatcher:
    config_manager = create_default_config_manager()
    return AgentDispatcher(
        agent_registry=AgentRegistry(tenant_id=TENANT, config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=None,
        sandbox_manager=_StubSandboxManager(policies or {}),
    )


def _register(dispatcher: AgentDispatcher, name: str, capabilities: List[str]) -> None:
    dispatcher._registry.register_agent(
        AgentEndpoint(
            name=name,
            url="http://localhost:8003",
            capabilities=capabilities,
        )
    )


def _summarizer_envelope(query: str = "summarize the clip") -> Dict[str, Any]:
    """The envelope ``_execute_summarization_task`` returns, built from the
    shipped ``SummaryResult``."""
    from cogniverse_agents.summarizer_agent import SummaryResult, ThinkingPhase

    result = SummaryResult(
        summary=ANSWER,
        key_points=["cyclist", "bridge"],
        visual_insights=["dusk light"],
        confidence_score=0.77,
        thinking_phase=ThinkingPhase(
            key_themes=["cycling"],
            content_categories=["video"],
            relevance_scores={"v_001": 0.9},
            visual_elements=["bridge"],
            reasoning="Both clips share the bridge setting.",
        ),
        metadata={"result_count": 2},
    )
    return {
        "status": "success",
        "agent": "summarizer_agent",
        "message": f"Generated summary for '{query}'",
        "result": dataclasses.asdict(result),
    }


async def _drain_background(dispatcher: AgentDispatcher) -> None:
    while dispatcher._background_tasks:
        await asyncio.gather(*list(dispatcher._background_tasks))


class TestCanonicalAnswerOnTheEnvelope:
    async def test_dispatch_stamps_the_answer_text(self):
        dispatcher = _dispatcher()
        _register(dispatcher, "summarizer_agent", ["summarization"])
        envelope = _summarizer_envelope()

        async def execute(query, tenant_id, context=None):
            return dict(envelope)

        dispatcher._execute_summarization_task = execute
        dispatcher._spawn_background = lambda coro: coro.close()

        result = await dispatcher.dispatch(
            "summarizer_agent", "summarize the clip", context={"tenant_id": TENANT}
        )

        assert result["answer"] == ANSWER
        assert set(result) == {"status", "agent", "message", "result", "answer"}
        assert result["message"] == "Generated summary for 'summarize the clip'"
        assert result["result"] == envelope["result"]

    async def test_error_envelope_gets_no_answer_and_is_preserved(self):
        dispatcher = _dispatcher()
        _register(dispatcher, "summarizer_agent", ["summarization"])
        error_envelope = {
            "status": "error",
            "agent": "summarizer_agent",
            "error": "SummarizerError: LM endpoint refused the connection",
        }

        async def execute(query, tenant_id, context=None):
            return dict(error_envelope)

        dispatcher._execute_summarization_task = execute
        dispatcher._spawn_background = lambda coro: coro.close()

        result = await dispatcher.dispatch(
            "summarizer_agent", "summarize the clip", context={"tenant_id": TENANT}
        )

        assert result == error_envelope

    async def test_raising_agent_produces_no_answer(self):
        dispatcher = _dispatcher()
        _register(dispatcher, "summarizer_agent", ["summarization"])
        seen: Dict[str, Any] = {}

        async def execute(query, tenant_id, context=None):
            raise RuntimeError("LM endpoint refused the connection")

        def spawn(coro):
            seen["background"] = True
            coro.close()

        dispatcher._execute_summarization_task = execute
        dispatcher._spawn_background = spawn

        with pytest.raises(RuntimeError) as excinfo:
            await dispatcher.dispatch(
                "summarizer_agent", "summarize the clip", context={"tenant_id": TENANT}
            )

        assert str(excinfo.value) == "LM endpoint refused the connection"
        assert seen == {}


class TestWikiAutoFileReadsTheAnswer:
    async def test_filed_page_body_is_the_answer_text(self, monkeypatch):
        from cogniverse_agents.wiki.wiki_manager import WikiManager
        from cogniverse_runtime.routers import wiki as wiki_router
        from cogniverse_sdk.document import DocumentFieldMapping

        backend = _RecordingBackend()
        manager = WikiManager(
            backend_resolver=lambda: backend,
            tenant_id=TENANT,
            schema_name="wiki_pages_acme_acme",
        )
        # The embedding sidecar is the one boundary this module does not
        # provision; every other step of save_session runs for real.
        monkeypatch.setattr(manager, "_generate_embedding", lambda *a, **k: [0.1] * 768)
        monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: manager)

        dispatcher = _dispatcher()
        _register(dispatcher, "summarizer_agent", ["summarization"])

        async def execute(query, tenant_id, context=None):
            return _summarizer_envelope()

        dispatcher._execute_summarization_task = execute

        await dispatcher.dispatch(
            "summarizer_agent",
            "summarize the clip",
            context={
                "tenant_id": TENANT,
                "conversation_history": AUTO_FILE_HISTORY,
            },
        )
        await _drain_background(dispatcher)

        mapping = DocumentFieldMapping.from_dict(
            json.loads(Path("configs/schemas/wiki_pages_schema.json").read_text())[
                "document_mapping"
            ]
        )
        session_pages = [
            fed["document"].to_schema_fields(mapping)
            for fed in backend.fed
            if fed["document"].to_schema_fields(mapping).get("page_type") == "session"
        ]
        assert len(session_pages) == 1
        page = session_pages[0]
        assert page["content"] == ANSWER
        assert page["title"] == "Session — summarize the clip"
        assert page["agent_used"] == "summarizer_agent"

    async def test_error_envelope_files_nothing(self, monkeypatch):
        from cogniverse_agents.wiki.wiki_manager import WikiManager
        from cogniverse_runtime.routers import wiki as wiki_router

        backend = _RecordingBackend()
        manager = WikiManager(
            backend_resolver=lambda: backend,
            tenant_id=TENANT,
            schema_name="wiki_pages_acme_acme",
        )
        monkeypatch.setattr(manager, "_generate_embedding", lambda *a, **k: [0.1] * 768)
        monkeypatch.setattr(wiki_router, "_wiki_manager_factory", lambda tid: manager)

        dispatcher = _dispatcher()
        _register(dispatcher, "summarizer_agent", ["summarization"])

        async def execute(query, tenant_id, context=None):
            return {
                "status": "error",
                "agent": "summarizer_agent",
                "error": "SummarizerError: LM endpoint refused the connection",
            }

        dispatcher._execute_summarization_task = execute

        await dispatcher.dispatch(
            "summarizer_agent",
            "summarize the clip",
            context={
                "tenant_id": TENANT,
                "conversation_history": AUTO_FILE_HISTORY,
            },
        )
        await _drain_background(dispatcher)

        assert backend.fed == []


class TestEgressMapIsDerivedFromTheRegistry:
    def test_map_covers_a_newly_registered_agent(self):
        dispatcher = _dispatcher()
        _register(dispatcher, "synthetic_probe_agent", ["retrieval", "summarization"])
        _register(dispatcher, "synthetic_talker_agent", ["text_generation"])

        kinds = dispatcher.egress_endpoint_kinds()

        assert kinds["synthetic_probe_agent"] == frozenset({"vespa", "llm"})
        assert kinds["synthetic_talker_agent"] == frozenset({"llm"})

    def test_shipped_agents_keep_their_endpoint_kinds(self):
        dispatcher = _dispatcher()
        _register(dispatcher, "search_agent", ["search", "video_search", "retrieval"])
        _register(dispatcher, "summarizer_agent", ["summarization", "text_generation"])
        _register(dispatcher, "coding_agent", ["coding", "code_generation"])
        _register(dispatcher, "orchestrator_agent", ["orchestration", "planning"])

        kinds = dispatcher.egress_endpoint_kinds()

        assert kinds["search_agent"] == frozenset({"vespa", "llm"})
        assert kinds["summarizer_agent"] == frozenset({"llm"})
        assert kinds["coding_agent"] == frozenset({"vespa", "llm"})
        assert kinds["orchestrator_agent"] == frozenset({"llm"})
        # The gateway path consults the routing_agent policy, which is not a
        # registered agent name: it still reaches the LM.
        assert dispatcher._egress_kinds_for("routing_agent") == frozenset({"llm"})


class TestEgressRefusalIsEnforcedForCodingAndOrchestrator:
    ALLOW_VESPA_ONLY = {
        "network_policies": {
            "egress": [{"host": "localhost", "port": 8080, "protocol": "tcp"}],
            "deny_all_other": True,
        }
    }

    def _dispatcher_with_endpoints(self, agent_name: str) -> AgentDispatcher:
        dispatcher = _dispatcher({agent_name: self.ALLOW_VESPA_ONLY})
        dispatcher._system_endpoints = lambda tenant_id: {
            "vespa": {"host": "localhost", "port": 8080, "protocol": "tcp"},
            "llm": {"host": "llm.example.com", "port": 11434, "protocol": "tcp"},
        }
        return dispatcher

    def test_coding_agent_llm_hop_off_the_allowlist_is_reported(self):
        dispatcher = self._dispatcher_with_endpoints("coding_agent")
        _register(dispatcher, "coding_agent", ["coding", "code_generation"])

        assert dispatcher._verify_egress("coding_agent", TENANT) == [
            {
                "host": "llm.example.com",
                "port": 11434,
                "protocol": "tcp",
                "reason": (
                    "host=llm.example.com port=11434 protocol=tcp "
                    "not in egress allowlist for agent=coding_agent"
                ),
            }
        ]

    def test_orchestrator_agent_llm_hop_off_the_allowlist_is_reported(self):
        dispatcher = self._dispatcher_with_endpoints("orchestrator_agent")
        _register(dispatcher, "orchestrator_agent", ["orchestration", "planning"])

        assert dispatcher._verify_egress("orchestrator_agent", TENANT) == [
            {
                "host": "llm.example.com",
                "port": 11434,
                "protocol": "tcp",
                "reason": (
                    "host=llm.example.com port=11434 protocol=tcp "
                    "not in egress allowlist for agent=orchestrator_agent"
                ),
            }
        ]

    async def test_coding_dispatch_verifies_before_any_other_work(self):
        """The coding path ran unchecked: pin that the verification happens,
        and that it happens before the agent is built."""
        dispatcher = self._dispatcher_with_endpoints("coding_agent")
        _register(dispatcher, "coding_agent", ["coding", "code_generation"])
        calls: List[Any] = []

        class _Verified(RuntimeError):
            pass

        def record(agent_name, endpoints):
            calls.append((agent_name, endpoints))
            raise _Verified("verified")

        dispatcher.validate_dispatch_endpoints = record

        with pytest.raises(_Verified):
            await dispatcher._execute_coding_task("fix the test", TENANT, {})

        assert calls == [
            (
                "coding_agent",
                [
                    {"host": "localhost", "port": 8080, "protocol": "tcp"},
                    {"host": "llm.example.com", "port": 11434, "protocol": "tcp"},
                ],
            )
        ]

    async def test_orchestration_dispatch_verifies_before_any_other_work(self):
        dispatcher = self._dispatcher_with_endpoints("orchestrator_agent")
        _register(dispatcher, "orchestrator_agent", ["orchestration", "planning"])
        calls: List[Any] = []

        class _Verified(RuntimeError):
            pass

        def record(agent_name, endpoints):
            calls.append((agent_name, endpoints))
            raise _Verified("verified")

        dispatcher.validate_dispatch_endpoints = record

        with pytest.raises(_Verified):
            await dispatcher._execute_orchestration_task("plan it", {}, TENANT)

        assert calls == [
            (
                "orchestrator_agent",
                [{"host": "llm.example.com", "port": 11434, "protocol": "tcp"}],
            )
        ]

    async def test_unparseable_endpoint_does_not_fail_the_dispatch(self):
        """The pre-flight is a drift detector: an address that does not parse
        drops out of the check instead of turning the dispatch into an error."""
        dispatcher = _dispatcher({"summarizer_agent": self.ALLOW_VESPA_ONLY})
        _register(dispatcher, "summarizer_agent", ["summarization"])

        class _BrokenSystemConfig:
            backend_url = "http://localhost"
            backend_port = "not-a-port"
            inference_service_urls: Dict[str, str] = {}

        dispatcher._config_manager = _BrokenConfigManager(_BrokenSystemConfig())

        api_base = (
            get_config(tenant_id=TENANT, config_manager=create_default_config_manager())
            .get_llm_config()
            .primary.api_base
        )
        resolved = urlparse(api_base if "://" in api_base else f"http://{api_base}")

        endpoints = dispatcher._system_endpoints(TENANT)
        # The unparseable backend address drops out; the LM endpoint the
        # tenant's config resolves is still checked.
        assert set(endpoints) == {"llm"}
        assert endpoints["llm"] == {
            "host": resolved.hostname,
            "port": resolved.port or 11434,
            "protocol": "tcp",
        }

        llm = endpoints["llm"]
        assert dispatcher._verify_egress("summarizer_agent", TENANT) == [
            {
                **llm,
                "reason": (
                    f"host={llm['host']} port={llm['port']} protocol=tcp "
                    "not in egress allowlist for agent=summarizer_agent"
                ),
            }
        ]

    def test_unreadable_system_config_skips_the_check_and_says_so(self, caplog):
        """A config-store outage must not fail the dispatch, and must not pass
        the drift check in silence either: the skip names the tenant."""

        class _DownConfigManager:
            def get_system_config(self):
                raise ConnectionError("config store refused the connection")

        dispatcher = _dispatcher({"summarizer_agent": self.ALLOW_VESPA_ONLY})
        _register(dispatcher, "summarizer_agent", ["summarization"])
        dispatcher._config_manager = _DownConfigManager()

        with caplog.at_level(logging.WARNING):
            assert dispatcher._system_endpoints(TENANT) == {}
            assert dispatcher._verify_egress("summarizer_agent", TENANT) == []

        skipped = (
            "Egress pre-flight skipped for tenant acme:acme: the system config "
            "is unreadable (config store refused the connection)"
        )
        # Once for the direct call above, once for the call inside the verify.
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.agent_dispatcher"
        ] == [skipped, skipped]

    def test_search_agent_vespa_hop_on_the_allowlist_is_not_reported(self):
        dispatcher = self._dispatcher_with_endpoints("search_agent")
        _register(dispatcher, "search_agent", ["search", "retrieval"])
        dispatcher._system_endpoints = lambda tenant_id: {
            "vespa": {"host": "localhost", "port": 8080, "protocol": "tcp"},
        }

        assert dispatcher._verify_egress("search_agent", TENANT) == []


class TestHistoryRewriteAcceptsAReplayedToolCallTurn:
    async def test_assistant_turn_without_content_round_trips(self):
        dispatcher = _dispatcher()
        seen: Dict[str, Any] = {}

        class _RecordingRewriter:
            async def acall(self, query, conversation_history):
                seen["history"] = conversation_history
                seen["query"] = query
                return dataclasses.make_dataclass(
                    "_Prediction", [("rewritten_query", str)]
                )("where is the blue bike")

        dispatcher._query_rewriter = _RecordingRewriter()

        rewritten = await dispatcher._rewrite_query_with_history(
            "and the blue one",
            [
                {"role": "user", "content": "find the red bike"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "content": "read_file -> ok"},
            ],
        )

        assert rewritten == "where is the blue bike"
        assert seen["query"] == "and the blue one"
        assert seen["history"] == (
            "user: find the red bike\nassistant: \ntool: read_file -> ok"
        )
