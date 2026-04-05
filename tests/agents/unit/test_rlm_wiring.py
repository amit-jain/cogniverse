"""Tests for RLM wiring across agents.

Verifies that each agent inherits RLMAwareMixin, its input schema exposes an
optional rlm field, and its output schema exposes rlm_synthesis/rlm_telemetry.
Also verifies real runtime behaviour: field assignment, type annotation,
and WikiManager merge-threshold logic.
"""

import typing

import pytest

from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_core.agents.rlm_options import RLMOptions

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

    @pytest.mark.parametrize("module,agent_cls,input_cls,output_cls", AGENTS_WITH_RLM)
    def test_input_has_rlm_field(self, module, agent_cls, input_cls, output_cls):
        mod = __import__(module, fromlist=[input_cls])
        cls = getattr(mod, input_cls)
        assert "rlm" in cls.model_fields
        field = cls.model_fields["rlm"]
        assert field.default is None

    @pytest.mark.parametrize("module,agent_cls,input_cls,output_cls", AGENTS_WITH_RLM)
    def test_output_has_rlm_fields(self, module, agent_cls, input_cls, output_cls):
        mod = __import__(module, fromlist=[output_cls])
        cls = getattr(mod, output_cls)
        assert "rlm_synthesis" in cls.model_fields
        assert "rlm_telemetry" in cls.model_fields


class TestDetailedReportInputRLMRuntime:
    """Runtime behaviour for DetailedReportInput's rlm field."""

    def test_rlm_field_accepts_rlm_options(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        opts = RLMOptions(enabled=True)
        inp = DetailedReportInput(query="test", search_results=[], rlm=opts)
        assert inp.rlm is opts
        assert inp.rlm.enabled is True

    def test_rlm_field_type_annotation(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        annotation = DetailedReportInput.model_fields["rlm"].annotation
        args = typing.get_args(annotation)
        assert RLMOptions in args, (
            f"rlm field annotation should include RLMOptions, got {annotation}"
        )


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
