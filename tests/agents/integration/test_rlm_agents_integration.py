"""Integration tests for RLM wiring across agents."""

import httpx
import pytest

from cogniverse_core.agents.rlm_options import RLMOptions


def _ollama_available():
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not available")


@pytest.mark.integration
@skip_if_no_ollama
class TestRLMAgentIntegration:
    def test_detailed_report_accepts_rlm_options(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        inp = DetailedReportInput(
            query="analyze results",
            search_results=[{"title": "test", "summary": "test result"}],
            rlm=RLMOptions(enabled=True, max_iterations=2),
        )
        assert inp.rlm.enabled is True
        assert inp.rlm.max_iterations == 2

    def test_coding_agent_accepts_rlm_options(self):
        from cogniverse_agents.coding_agent import CodingInput

        inp = CodingInput(
            task="analyze code",
            codebase_path="/tmp",
            rlm=RLMOptions(enabled=True, auto_detect=True),
        )
        assert inp.rlm.enabled is True

    def test_deep_research_accepts_rlm_options(self):
        from cogniverse_agents.deep_research_agent import DeepResearchInput

        inp = DeepResearchInput(
            query="test research",
            rlm=RLMOptions(enabled=True, context_threshold=30000),
        )
        assert inp.rlm.enabled is True
        assert inp.rlm.context_threshold == 30000

    def test_wiki_merge_threshold_logic(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        assert wm._should_use_rlm_for_merge("x" * 40000, "y" * 20000) is True
        assert wm._should_use_rlm_for_merge("short", "also short") is False
        # 49998 chars combined — just under threshold
        assert wm._should_use_rlm_for_merge("x" * 30000, "y" * 19998) is False

    def test_rlm_options_disabled_by_default(self):
        """All agents default to rlm=None (disabled)."""
        from cogniverse_agents.coding_agent import CodingInput
        from cogniverse_agents.deep_research_agent import DeepResearchInput
        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        assert DetailedReportInput(query="x", search_results=[]).rlm is None
        assert CodingInput(task="x", codebase_path="/tmp").rlm is None
        assert DeepResearchInput(query="x").rlm is None
