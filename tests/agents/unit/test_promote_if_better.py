"""Unit tests for the regression-reject gate (``promote_if_better``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


class FakeDatasetStore:
    """Captures saves so tests can assert what was (or wasn't) persisted."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.append_calls: list[tuple[str, pd.DataFrame]] = []
        self.create_calls: list[str] = []  # ordered creation history

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.created[name] = data.copy()
        self.create_calls.append(name)
        return f"id::{name}"

    async def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.created:
            raise KeyError(name)
        return self.created[name]

    async def delete_dataset(self, name: str) -> bool:
        return self.created.pop(name, None) is not None

    async def append_to_dataset(
        self, name: str, data: pd.DataFrame, metadata: dict | None = None
    ) -> None:
        if name not in self.created:
            raise KeyError(name)
        self.append_calls.append((name, data))
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self):
        self.datasets = FakeDatasetStore()


@pytest.fixture
def manager_and_provider():
    provider = FakeProvider()
    return ArtifactManager(provider, tenant_id="acme"), provider


CANDIDATE_PROMPTS = {"system": "candidate-system", "tool_use": "use carefully"}
CANDIDATE_DEMOS: List[Dict[str, Any]] = [
    {"input": "q1", "output": "a1"},
    {"input": "q2", "output": "a2"},
]


def _ds_name_prompts(agent: str) -> str:
    return f"dspy-prompts-acme:acme-{agent}"


def _ds_name_demos(agent: str) -> str:
    return f"dspy-demos-acme:acme-{agent}"


def _ds_name_experiments(agent: str) -> str:
    return f"dspy-experiments-acme:acme-{agent}"


class TestPromotionPath:
    @pytest.mark.asyncio
    async def test_winning_candidate_persists_artefacts_and_logs_promoted(
        self, manager_and_provider
    ):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=CANDIDATE_DEMOS,
            baseline_score=0.50,
            candidate_score=0.60,
            optimizer="BootstrapFewShot",
            train_examples=64,
        )

        # Artefacts persisted under the agent's prompts/demos datasets.
        assert _ds_name_prompts("search_agent") in provider.datasets.created
        assert _ds_name_demos("search_agent") in provider.datasets.created

        # Experiment row written, promoted flag true.
        assert record.promoted is True
        assert record.candidate_score == 0.60
        assert record.baseline_score == 0.50
        assert pytest.approx(record.improvement, rel=1e-9) == 0.10
        assert record.extra_metrics["tolerance"] == 0.0
        assert "rejection_reason" not in record.extra_metrics

    @pytest.mark.asyncio
    async def test_tolerance_band_allows_marginal_regression(
        self, manager_and_provider
    ):
        mgr, provider = manager_and_provider

        # candidate is 0.005 below baseline; with tolerance=0.01, still promoted.
        record = await mgr.promote_if_better(
            agent_type="summary",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.80,
            candidate_score=0.795,
            tolerance=0.01,
        )

        assert record.promoted is True
        assert _ds_name_prompts("summary") in provider.datasets.created
        # No demos passed → demos dataset must NOT be created.
        assert _ds_name_demos("summary") not in provider.datasets.created


class TestRejectionPath:
    @pytest.mark.asyncio
    async def test_losing_candidate_does_not_save_prompts_or_demos(
        self, manager_and_provider
    ):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=CANDIDATE_DEMOS,
            baseline_score=0.80,
            candidate_score=0.50,
        )

        # Prompts/demos NOT saved.
        assert _ds_name_prompts("search_agent") not in provider.datasets.created
        assert _ds_name_demos("search_agent") not in provider.datasets.created

        # ExperimentMetrics row recorded with promoted=False + rejection_reason.
        assert record.promoted is False
        assert record.candidate_score == 0.50
        assert "rejection_reason" in record.extra_metrics
        assert "regression" in record.extra_metrics["rejection_reason"]

        # Experiment dataset still gets created so the rejection is queryable.
        assert _ds_name_experiments("search_agent") in provider.datasets.created

    @pytest.mark.asyncio
    async def test_zero_tolerance_strict_loss_rejected(self, manager_and_provider):
        mgr, _ = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="x",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.4999,
        )
        assert record.promoted is False

    @pytest.mark.asyncio
    async def test_outside_tolerance_band_rejected(self, manager_and_provider):
        mgr, _ = manager_and_provider

        # Candidate 0.02 below baseline; tolerance is only 0.01 → rejected.
        record = await mgr.promote_if_better(
            agent_type="x",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.50,
            candidate_score=0.48,
            tolerance=0.01,
        )
        assert record.promoted is False
        assert record.improvement == pytest.approx(-0.02, rel=1e-9)


class TestExperimentLedgerComposition:
    @pytest.mark.asyncio
    async def test_promotion_and_rejection_both_appear_in_history(
        self, manager_and_provider
    ):
        mgr, _ = manager_and_provider

        # First run: promoted.
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.6,
            run_id="run-001",
        )
        # Second run: rejected.
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v2"},
            candidate_demos=None,
            baseline_score=0.6,
            candidate_score=0.4,
            run_id="run-002",
        )

        history = await mgr.load_experiments("agent_a")
        ids = [r.run_id for r in history]
        assert ids == ["run-001", "run-002"]
        assert history[0].promoted is True
        assert history[1].promoted is False

        # Active prompts must reflect ONLY the first (winning) save.
        active = await mgr.load_prompts("agent_a")
        assert active == {"system": "v1"}


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_negative_tolerance_rejected(self, manager_and_provider):
        mgr, _ = manager_and_provider
        with pytest.raises(ValueError, match="tolerance"):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts={},
                candidate_demos=None,
                baseline_score=0.5,
                candidate_score=0.4,
                tolerance=-0.1,
            )

    @pytest.mark.asyncio
    async def test_negative_min_improvement_rejected(self, manager_and_provider):
        mgr, _ = manager_and_provider
        with pytest.raises(ValueError, match="min_improvement"):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts={},
                candidate_demos=None,
                baseline_score=0.5,
                candidate_score=0.6,
                min_improvement=-0.01,
            )


class TestMinImprovement:
    """``min_improvement`` demands a real win, not just parity."""

    @pytest.mark.asyncio
    async def test_below_min_improvement_rejected(self, manager_and_provider):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.50,
            candidate_score=0.52,
            min_improvement=0.05,
        )

        assert record.promoted is False
        assert record.extra_metrics["min_improvement"] == 0.05
        assert "min_improvement" in record.extra_metrics["rejection_reason"]
        assert _ds_name_prompts("search_agent") not in provider.datasets.created

    @pytest.mark.asyncio
    async def test_meeting_min_improvement_promoted(self, manager_and_provider):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.50,
            candidate_score=0.55,
            min_improvement=0.05,
        )

        assert record.promoted is True
        assert _ds_name_prompts("search_agent") in provider.datasets.created


class TestServeVersioned:
    """``serve_versioned=True`` routes a win through the canary state machine
    so BOTH read seams (init-time un-versioned active + request-time overlay
    keyed off the state blob) serve the winning candidate."""

    @pytest.mark.asyncio
    async def test_winning_candidate_lands_versioned_canary_then_active(
        self, manager_and_provider
    ):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=CANDIDATE_DEMOS,
            baseline_score=0.50,
            candidate_score=0.60,
            serve_versioned=True,
        )

        assert record.promoted is True
        assert record.extra_metrics["served_version"] == 1
        assert "dspy-prompts-acme:acme-search_agent-v1" in provider.datasets.created
        assert "dspy-demos-acme:acme-search_agent-v1" in provider.datasets.created
        state = await mgr.get_artefact_state("search_agent")
        assert state["active"]["version"] == 1
        assert state["canary"] is None
        assert await mgr.load_prompts("search_agent") == CANDIDATE_PROMPTS

    @pytest.mark.asyncio
    async def test_losing_candidate_serve_versioned_writes_nothing(
        self, manager_and_provider
    ):
        mgr, provider = manager_and_provider

        record = await mgr.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.80,
            candidate_score=0.50,
            serve_versioned=True,
        )

        assert record.promoted is False
        assert "dspy-prompts-acme:acme-search_agent-v1" not in provider.datasets.created
        assert await mgr.load_prompts("search_agent") is None
        state = await mgr.get_artefact_state("search_agent")
        assert state == {"active": None, "canary": None, "retired": []}
