"""Unit tests for the regression-reject gate (``promote_if_better``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class FakeDatasetStore:
    """Captures saves so tests can assert what was (or wasn't) persisted."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.append_calls: list[tuple[str, pd.DataFrame]] = []
        self.create_calls: list[str] = []  # ordered creation history

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

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


def _gate(threshold=0.05, share=0.5, ceiling=0.95):
    from cogniverse_agents.optimizer.artifact_manager import PromotionGate

    return PromotionGate(
        improvement_threshold=threshold, gap_share=share, skip_above_score=ceiling
    )


class TestPromotionGate:
    """The gain a candidate owes is capped by a share of the gap to the maximum."""

    @pytest.mark.parametrize(
        ("served", "candidate", "threshold", "share", "max_score", "required", "wins"),
        [
            # Below 0.90 the gap share exceeds the threshold: the flat rule.
            (0.5, 0.54, 0.05, 0.5, 1.0, 0.05, False),
            (0.5, 0.55, 0.05, 0.5, 1.0, 0.05, True),
            (0.0, 0.05, 0.05, 0.5, 1.0, 0.05, True),
            (0.0, 0.04, 0.05, 0.5, 1.0, 0.05, False),
            # Exactly 0.90: half the gap equals the threshold.
            (0.9, 0.95, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.9), True),
            (0.9, 0.94, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.9), False),
            # Above 0.90 the candidate must close half the remaining gap.
            (0.983, 1.0, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.983), True),
            (0.983, 0.99, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.983), False),
            (0.983, 0.9915, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.983), True),
            (0.96, 0.98, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.96), True),
            (0.96, 0.979, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.96), False),
            # A whole-gap share demands the maximum itself.
            (0.98, 0.99, 0.05, 1.0, 1.0, 1.0 - 0.98, False),
            (0.98, 1.0, 0.05, 1.0, 1.0, 1.0 - 0.98, True),
            # Nothing left to gain: an equal candidate never promotes.
            (1.0, 1.0, 0.05, 0.5, 1.0, 0.0, False),
            (1.0, 1.0, 0.0, 0.5, 1.0, 0.0, False),
            (2.0, 2.0, 0.05, 0.5, 2.0, 0.0, False),
            # A regression never promotes, whatever the gap.
            (0.983, 0.9, 0.05, 0.5, 1.0, 0.5 * (1.0 - 0.983), False),
            # The gap is measured against the metric's own maximum.
            (1.0, 1.04, 0.05, 0.5, 2.0, 0.05, False),
            (1.0, 1.05, 0.05, 0.5, 2.0, 0.05, True),
            (1.96, 1.98, 0.05, 0.5, 2.0, 0.5 * (2.0 - 1.96), True),
            # A zero threshold accepts an equal candidate only below the maximum.
            (0.5, 0.5, 0.0, 0.5, 1.0, 0.0, True),
        ],
    )
    def test_decision_table(
        self, served, candidate, threshold, share, max_score, required, wins
    ):
        gate = _gate(threshold, share)

        assert gate.required_gain(served, max_score) == required
        assert gate.promotes(candidate, served, max_score) is wins

    @pytest.mark.parametrize(
        ("holdout", "served_hits", "candidate_hits", "wins"),
        [
            # Half the gap, exactly: 0.5 * (1 - 58/60) rounds a hair above 1/60.
            (60, 58, 59, True),
            (27, 25, 26, True),
            (54, 50, 52, True),
            (54, 50, 51, False),
            # The flat threshold, exactly: 2/20 + 0.05 rounds above 3/20.
            (20, 2, 3, True),
            (40, 21, 23, True),
            (40, 21, 22, False),
            (60, 60, 60, False),
        ],
    )
    def test_a_candidate_exactly_on_the_bar_is_not_lost_to_rounding(
        self, holdout, served_hits, candidate_hits, wins
    ):
        assert (
            _gate().promotes(candidate_hits / holdout, served_hits / holdout, 1.0)
            is wins
        )

    @pytest.mark.parametrize(
        ("served", "ceiling", "skips"),
        [
            (0.983, 0.95, True),
            (0.95, 0.95, True),
            (0.9499999, 0.95, False),
            (0.5, 0.95, False),
            (1.0, 1.0, True),
            (0.999, 1.0, False),
        ],
    )
    def test_skips_compile_at_or_above_the_ceiling(self, served, ceiling, skips):
        assert _gate(ceiling=ceiling).skips_compile(served, 1.0) is skips

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"threshold": -0.01}, "improvement_threshold must be >= 0; got -0.01"),
            ({"share": 0.0}, "gap_share must be in (0, 1]; got 0.0"),
            ({"share": 1.01}, "gap_share must be in (0, 1]; got 1.01"),
            ({"share": -0.5}, "gap_share must be in (0, 1]; got -0.5"),
            ({"ceiling": 0.0}, "skip_above_score must be > 0; got 0.0"),
            ({"ceiling": -1.0}, "skip_above_score must be > 0; got -1.0"),
        ],
    )
    def test_out_of_range_values_are_refused(self, kwargs, message):
        with pytest.raises(ValueError) as raised:
            _gate(**kwargs)
        assert str(raised.value) == message

    def test_a_ceiling_above_the_metric_maximum_is_refused(self):
        gate = _gate(ceiling=1.5)

        with pytest.raises(ValueError) as raised:
            gate.skips_compile(1.0, 1.0)
        assert str(raised.value) == (
            "skip_above_score 1.5 is above the metric's maximum 1.0"
        )
        with pytest.raises(ValueError) as raised:
            gate.promotes(1.0, 0.5, 0.0)
        assert str(raised.value) == "max_score must be > 0; got 0.0"


class TestGatedPromotion:
    """``promote_if_better`` decides through the tenant's ``PromotionGate``."""

    @pytest.mark.asyncio
    async def test_a_candidate_closing_half_the_gap_is_promoted(
        self, manager_and_provider
    ):
        manager, provider = manager_and_provider

        record = await manager.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.983,
            candidate_score=1.0,
            gate=_gate(),
            max_score=1.0,
        )

        assert record.promoted is True
        assert record.extra_metrics["min_improvement"] == 0.5 * (1.0 - 0.983)
        assert record.extra_metrics["improvement_threshold"] == 0.05
        assert record.extra_metrics["gap_share"] == 0.5
        assert record.extra_metrics["max_score"] == 1.0
        assert provider.datasets.created[_ds_name_prompts("search_agent")].to_dict(
            "records"
        ) == [{"name": key, "value": text} for key, text in CANDIDATE_PROMPTS.items()]

    @pytest.mark.asyncio
    async def test_a_candidate_short_of_half_the_gap_is_rejected(
        self, manager_and_provider
    ):
        manager, provider = manager_and_provider

        record = await manager.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=0.983,
            candidate_score=0.99,
            gate=_gate(),
            max_score=1.0,
        )

        assert record.promoted is False
        assert record.extra_metrics["rejection_reason"] == (
            "candidate_score=0.9900 < required 0.9915 (baseline 0.9830 + "
            "min_improvement 0.0085 - tolerance 0.0000); improvement=0.0070 — "
            "regression-rejected"
        )
        assert _ds_name_prompts("search_agent") not in provider.datasets.created

    @pytest.mark.asyncio
    async def test_a_baseline_at_the_maximum_never_promotes_an_equal_candidate(
        self, manager_and_provider
    ):
        manager, provider = manager_and_provider

        record = await manager.promote_if_better(
            agent_type="search_agent",
            candidate_prompts=CANDIDATE_PROMPTS,
            candidate_demos=None,
            baseline_score=1.0,
            candidate_score=1.0,
            gate=_gate(),
            max_score=1.0,
        )

        assert record.promoted is False
        assert record.extra_metrics["min_improvement"] == 0.0
        assert record.extra_metrics["rejection_reason"] == (
            "baseline 1.0000 is at the metric's maximum 1.0000; "
            "candidate_score=1.0000 cannot improve on it"
        )
        assert _ds_name_prompts("search_agent") not in provider.datasets.created

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"gate": "gate"}, "gate and max_score are passed together"),
            ({"max_score": 1.0}, "gate and max_score are passed together"),
            (
                {"gate": "gate", "max_score": 1.0, "min_improvement": 0.05},
                "min_improvement is derived from the gate; pass one or the other",
            ),
        ],
    )
    async def test_gate_arguments_are_checked(
        self, manager_and_provider, kwargs, message
    ):
        manager, provider = manager_and_provider
        if "gate" in kwargs:
            kwargs = {**kwargs, "gate": _gate()}

        with pytest.raises(ValueError) as raised:
            await manager.promote_if_better(
                agent_type="search_agent",
                candidate_prompts=CANDIDATE_PROMPTS,
                candidate_demos=None,
                baseline_score=0.5,
                candidate_score=0.9,
                **kwargs,
            )
        assert str(raised.value) == message
        assert provider.datasets.created == {}
