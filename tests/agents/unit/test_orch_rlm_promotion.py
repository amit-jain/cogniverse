"""Unit tests for Orchestrator's RLM promotion of sub-agent payloads."""

from __future__ import annotations

import pytest

from cogniverse_agents._rlm_promotion import (
    _RLM_PROMOTION_DEFAULT_THRESHOLD,
    _projected_payload_chars,
)
from cogniverse_agents._rlm_promotion import (
    maybe_promote_to_rlm as _maybe_promote_to_rlm,
)


class TestProjectedPayloadChars:
    def test_sums_string_field_lengths(self):
        agent_input = {
            "query": "what is X" * 10,  # 90 chars
            "documents": "abc" * 100,  # 300 chars
            "context": "y" * 50,  # 50 chars
        }
        # Approximate: 90 + 300 + 50 = 440. Allow ±20 for str() overhead.
        assert 400 < _projected_payload_chars(agent_input) < 500

    def test_handles_non_string_values(self):
        agent_input = {
            "max_iterations": 5,
            "ratios": [0.1, 0.2],
            "flag": True,
        }
        # Should not raise and produce a non-negative count.
        assert _projected_payload_chars(agent_input) > 0

    def test_empty_input_zero(self):
        assert _projected_payload_chars({}) == 0


class TestPromotionDecision:
    def test_promotes_when_projected_exceeds_cutoff(self):
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)  # well above 75%
        agent_input = {"query": "q", "context": big_value}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert agent_input["rlm"]["enabled"] is True
        assert agent_input["rlm"]["auto_detect"] is True
        assert (
            agent_input["rlm"]["context_threshold"] == _RLM_PROMOTION_DEFAULT_THRESHOLD
        )

    def test_does_not_promote_when_under_cutoff(self):
        agent_input = {"query": "small", "context": "small context"}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert "rlm" not in agent_input

    def test_promotion_only_for_promotable_agents(self):
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)
        agent_input = {"query": "q", "context": big_value}
        # Routing / gateway / orchestrator agents are NOT in the promotable
        # set — they do not accept an rlm field on their input schema.
        _maybe_promote_to_rlm("orchestrator_agent", agent_input)
        assert "rlm" not in agent_input
        _maybe_promote_to_rlm("entity_extraction_agent", agent_input)
        assert "rlm" not in agent_input

    def test_caller_explicit_rlm_wins(self):
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)
        # Caller already opted out (rlm=None).
        agent_input = {"query": "q", "context": big_value, "rlm": None}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert agent_input["rlm"] is None  # untouched

        # Caller already opted in with custom params.
        agent_input2 = {
            "query": "q",
            "context": big_value,
            "rlm": {"enabled": True, "max_iterations": 7},
        }
        _maybe_promote_to_rlm("search_agent", agent_input2)
        assert agent_input2["rlm"] == {"enabled": True, "max_iterations": 7}

    def test_canonical_name_lookup_handles_suffix_strip(self):
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)
        # Pass without _agent suffix — should still match search_agent.
        agent_input = {"query": "q", "context": big_value}
        _maybe_promote_to_rlm("search", agent_input)
        assert "rlm" in agent_input

    def test_env_disabled_skips_promotion(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_ORCH_RLM_PROMOTION", "disabled")
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)
        agent_input = {"query": "q", "context": big_value}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert "rlm" not in agent_input

    def test_custom_fraction_via_env(self, monkeypatch):
        # Tighten the fraction so a smaller payload triggers.
        monkeypatch.setenv("COGNIVERSE_ORCH_RLM_PROMOTION_FRACTION", "0.1")
        # 10% of 50_000 = 5_000 chars; our payload is 6000.
        agent_input = {"query": "q" * 6000}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert "rlm" in agent_input

    def test_env_invalid_fraction_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_ORCH_RLM_PROMOTION_FRACTION", "not-a-float")
        # Below default cutoff; should NOT promote.
        agent_input = {"query": "small"}
        _maybe_promote_to_rlm("search_agent", agent_input)
        assert "rlm" not in agent_input

    @pytest.mark.parametrize(
        "agent",
        [
            "search_agent",
            "deep_research_agent",
            "detailed_report_agent",
            "coding_agent",
        ],
    )
    def test_all_documented_promotable_agents(self, agent):
        big_value = "X" * (_RLM_PROMOTION_DEFAULT_THRESHOLD)
        agent_input = {"query": "q", "context": big_value}
        _maybe_promote_to_rlm(agent, agent_input)
        assert agent_input.get("rlm", {}).get("enabled") is True
