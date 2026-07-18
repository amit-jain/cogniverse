"""Reflective recompile against real Phoenix + real LM.

An all-failure search agent (only low_scoring rows) has no positives to
bootstrap, so ``_optimize_agent`` routes it through a real ``dspy.GEPA``
reflective compile. The candidate STILL goes through ``promote_if_better``:
either it beats the baseline on the held-out failures and flips the active
prompt, or it is rejected and the base prompt is left byte-for-byte unchanged.

Uses the shared vespa_instance + config_manager + real_telemetry from conftest.
"""

import logging

import pandas as pd
import pytest

from tests.fixtures.llm import is_test_lm_available

logger = logging.getLogger(__name__)


skip_if_no_lm = pytest.mark.skipif(
    not is_test_lm_available(),
    reason="Test LM not available for reflective recompile",
)


@pytest.mark.integration
@skip_if_no_lm
class TestReflectiveRecompile:
    """Run the real reflective compile through _optimize_agent end-to-end."""

    @pytest.mark.asyncio
    async def test_all_failure_search_reflective_compile_is_gated(
        self, config_manager, real_telemetry, phoenix_container, monkeypatch
    ):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_agents.routing.config import (
            AutomationRulesConfig,
            OptimizationTriggersConfig,
        )
        from cogniverse_foundation.config.utils import get_config
        from cogniverse_runtime.optimization_cli import _optimize_agent

        tenant_id = "test:unit"

        # Small GEPA budget so the reflective compile runs quickly against the
        # tiny test LM, but keep the real promotion gate (default 0.05 margin).
        rules = AutomationRulesConfig(
            optimization_triggers=OptimizationTriggersConfig(
                enable_reflective_recompile=True,
                min_reflective_failures=3,
                reflective_max_metric_calls=18,
            )
        )
        monkeypatch.setattr(
            "cogniverse_runtime.quality_monitor_cli._load_automation_rules",
            lambda tenant_id, config_manager=None: rules,
        )

        # All failures, zero positives — the positives trainset is empty and the
        # reflective branch takes over.
        low = pd.DataFrame(
            [
                {"query": q, "output": '{"results": []}', "score": 0.1}
                for q in [
                    "when did the event happen after the explosion",
                    "timeline of events before the crash",
                    "sequence during the performance",
                    "what happened right after the goal",
                    "before the sunrise over the mountains",
                    "who scored just before halftime",
                    "events leading up to the finale",
                    "moments after the announcement",
                ]
            ]
        )

        telemetry_provider = real_telemetry.get_provider(tenant_id=tenant_id)
        artifact_manager = ArtifactManager(telemetry_provider, tenant_id)
        before = await artifact_manager.load_prompts("search_agent")

        config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        llm_endpoint = config.get_llm_config().resolve("optimization")

        result = await _optimize_agent(
            agent_name="search",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=llm_endpoint,
            config_manager=config_manager,
            telemetry_provider=telemetry_provider,
            tenant_id=tenant_id,
        )

        assert result["status"] == "success", result
        assert result["reflective"] is True
        # No positives held out — the reflective path scores on negatives only.
        assert result["holdout_examples"] == 0
        # 8 failures split deterministically -> 6 reflect-train (>= min 3).
        assert result["training_examples"] == 6
        assert result["negative_probes"] == 2

        served = result["served"]
        assert served is not None, result
        assert served["served_agent"] == "search_agent"
        # Either the GEPA candidate beat the baseline on the held-out failures
        # and flipped active, or it was rejected — both are valid, but the
        # active prompt state must match the verdict exactly.
        assert isinstance(served["promoted"], bool)
        assert isinstance(served["baseline_score"], float)
        assert isinstance(served["candidate_score"], float)

        after = await artifact_manager.load_prompts("search_agent")
        if served["promoted"]:
            assert served["active"] is True
            assert isinstance(served["version"], int) and served["version"] >= 1
            assert after is not None
            assert after.get("search_optimizer")
            assert after != before
        else:
            assert served["active"] is False
            # A rejected reflective candidate must not touch the base prompt.
            assert after == before
