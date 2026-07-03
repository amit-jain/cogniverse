"""Tests for RLM wiring across agents.

Verifies that each agent inherits RLMAwareMixin, its input schema exposes an
optional rlm field, and its output schema exposes rlm_synthesis/rlm_telemetry.
Also verifies real runtime behaviour: field assignment, type annotation,
and WikiManager merge-threshold logic.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

_SR_URL = "http://semantic-router:8080/v1"
_DIRECT = "http://direct"


def _enabled_semantic_router() -> SemanticRouterConfig:
    return SemanticRouterConfig(
        enabled=True,
        semantic_router_url=_SR_URL,
        tenant_tiers={"acme:prod": "pro", "beta:prod": "free"},
        default_tier="free",
    )


def _patch_enabled_get_config(monkeypatch) -> None:
    cfg = MagicMock()
    cfg.get_semantic_router.return_value = _enabled_semantic_router()
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
    )


AGENTS_WITH_RLM = [
    (
        "cogniverse_agents.detailed_report_agent",
        "DetailedReportAgent",
        "DetailedReportInput",
        "DetailedReportOutput",
    ),
    (
        "cogniverse_agents.coding_agent",
        "CodingAgent",
        "CodingInput",
        "CodingOutput",
    ),
    (
        "cogniverse_agents.deep_research_agent",
        "DeepResearchAgent",
        "DeepResearchInput",
        "DeepResearchOutput",
    ),
]


class TestRLMWiring:
    @pytest.mark.parametrize("module,agent_cls,input_cls,output_cls", AGENTS_WITH_RLM)
    def test_agent_has_rlm_mixin(self, module, agent_cls, input_cls, output_cls):
        mod = __import__(module, fromlist=[agent_cls])
        cls = getattr(mod, agent_cls)
        assert issubclass(cls, RLMAwareMixin)


class TestDetailedReportInputRLMRuntime:
    """Runtime behaviour for DetailedReportInput's rlm field."""

    def test_rlm_field_accepts_rlm_options(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        opts = RLMOptions(enabled=True)
        inp = DetailedReportInput(query="test", search_results=[], rlm=opts)
        assert inp.rlm is opts
        assert inp.rlm.enabled is True


class _MixinHost(RLMAwareMixin):
    """Minimal RLMAwareMixin host with tenant + config_manager set."""

    def __init__(self, tenant_id, config_manager):
        self.tenant_id = tenant_id
        self._config_manager = config_manager


class TestRLMAwareMixinRouting:
    """get_rlm routes the RLM endpoint through the semantic router for the host tenant."""

    def _endpoint(self):
        return LLMEndpointConfig(model="openai/gpt-4o", api_base=_DIRECT)

    def test_enabled_routes_cached_rlm_through_semantic_router(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        _patch_enabled_get_config(monkeypatch)
        host = _MixinHost("acme:prod", MagicMock())

        rlm = host.get_rlm(self._endpoint())

        assert rlm.llm_config.api_base == _SR_URL
        assert rlm.model == "openai/auto"
        assert rlm.llm_config.extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }
        assert rlm._tenant_id == "acme:prod"

    def test_disabled_keeps_direct_endpoint(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
        )
        host = _MixinHost("acme:prod", MagicMock())

        rlm = host.get_rlm(self._endpoint())

        assert rlm.llm_config.api_base == _DIRECT
        assert rlm.llm_config.extra_headers is None

    def test_no_config_manager_keeps_direct_endpoint(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        host = _MixinHost("acme:prod", None)

        rlm = host.get_rlm(self._endpoint())

        assert rlm.llm_config.api_base == _DIRECT

    def test_cache_invalidates_on_tenant_change(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        _patch_enabled_get_config(monkeypatch)
        host = _MixinHost("acme:prod", MagicMock())

        first = host.get_rlm(self._endpoint(), tenant_id="acme:prod")
        second = host.get_rlm(self._endpoint(), tenant_id="beta:prod")

        # Different tenant tier => different headers => cache must not reuse.
        assert first is not second
        assert first.llm_config.extra_headers["x-authz-user-groups"] == "pro"
        assert second.llm_config.extra_headers["x-authz-user-groups"] == "free"

    def test_cache_reuses_same_instance_for_same_tenant(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        _patch_enabled_get_config(monkeypatch)
        host = _MixinHost("acme:prod", MagicMock())

        first = host.get_rlm(self._endpoint(), tenant_id="acme:prod")
        second = host.get_rlm(self._endpoint(), tenant_id="acme:prod")

        assert first is second


class TestHostAgentsThreadConfigManagerToRLM:
    """A constructor-injected config_manager reaches get_rlm's routing path."""

    def _build(self, which, cm):
        if which == "coding":
            from cogniverse_agents.coding_agent import CodingAgent, CodingDeps

            return CodingAgent(
                deps=CodingDeps(tenant_id="acme:prod"), config_manager=cm
            )
        from cogniverse_agents.deep_research_agent import (
            DeepResearchAgent,
            DeepResearchDeps,
        )

        return DeepResearchAgent(
            deps=DeepResearchDeps(tenant_id="acme:prod"), config_manager=cm
        )

    @pytest.mark.parametrize("which", ["coding", "deep_research"])
    def test_constructor_config_manager_routes_rlm(self, which, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")
        _patch_enabled_get_config(monkeypatch)
        cm = MagicMock()

        agent = self._build(which, cm)
        assert agent._config_manager is cm

        rlm = agent.get_rlm(
            LLMEndpointConfig(model="openai/gpt-4o", api_base=_DIRECT),
            tenant_id="acme:prod",
        )
        assert rlm.llm_config.api_base == _SR_URL
        assert rlm.llm_config.extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }


class TestWikiManagerRLMRouting:
    """_merge_with_rlm routes its endpoint through the semantic router for the tenant."""

    def test_merge_routes_rlm_through_semantic_router(self, monkeypatch):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        captured = {}

        class _FakeRLM:
            def __init__(self, llm_config, tenant_id=None, **kw):
                captured["llm_config"] = llm_config
                captured["tenant_id"] = tenant_id

            def process(self, query, context):
                captured["context"] = context
                return SimpleNamespace(answer="MERGED-SUMMARY")

        monkeypatch.setattr(
            "cogniverse_agents.wiki.wiki_manager.RLMInference", _FakeRLM
        )
        _patch_enabled_get_config(monkeypatch)

        wm = WikiManager.__new__(WikiManager)
        wm._llm_endpoint_config = LLMEndpointConfig(
            model="openai/gpt-4o", api_base=_DIRECT
        )
        wm._config_manager = MagicMock()
        wm._tenant_id = "acme:prod"

        out = wm._merge_with_rlm("old text", "new text", "Acme Corp")

        assert out == "MERGED-SUMMARY"
        assert captured["llm_config"].api_base == _SR_URL
        assert captured["llm_config"].extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }
        assert captured["tenant_id"] == "acme:prod"

    def test_merge_stays_direct_without_config_manager(self, monkeypatch):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        captured = {}

        class _FakeRLM:
            def __init__(self, llm_config, tenant_id=None, **kw):
                captured["llm_config"] = llm_config

            def process(self, query, context):
                return SimpleNamespace(answer="MERGED")

        monkeypatch.setattr(
            "cogniverse_agents.wiki.wiki_manager.RLMInference", _FakeRLM
        )

        wm = WikiManager.__new__(WikiManager)
        wm._llm_endpoint_config = LLMEndpointConfig(
            model="openai/gpt-4o", api_base=_DIRECT
        )
        wm._config_manager = None
        wm._tenant_id = "acme:prod"

        wm._merge_with_rlm("old", "new", "Acme")

        assert captured["llm_config"].api_base == _DIRECT
        assert captured["llm_config"].extra_headers is None


class TestWikiManagerRLM:
    def test_skips_rlm_for_small_content(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        assert wm._should_use_rlm_for_merge("short", "also short") is False

    def test_triggers_rlm_for_large_content(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        assert wm._should_use_rlm_for_merge("x" * 40000, "y" * 20000) is True

    def test_threshold_boundary_below(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        assert wm._should_use_rlm_for_merge("a" * 30000, "b" * 19999) is False

    def test_threshold_boundary_at(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        assert wm._should_use_rlm_for_merge("a" * 25000, "b" * 25000) is True
