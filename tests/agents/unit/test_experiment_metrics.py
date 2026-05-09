"""Unit tests for the typed ExperimentMetrics + ArtifactManager save/load.

The typed experiment ledger covers:
  * dataclass round-trip via ``to_row`` / ``from_row``;
  * back-compat shim ``log_optimization_run`` translates dict → typed record;
  * tenant_id mismatch is rejected at write time.

Real-Phoenix integration coverage lives in
``tests/agents/integration/test_artifact_manager_experiments.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    ArtifactManager,
    ExperimentMetrics,
)


class FakeDatasetStore:
    """In-memory DatasetStore stub so unit tests do not need Phoenix."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.append_calls: list[tuple[str, pd.DataFrame]] = []

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.created[name] = data.copy()
        return f"id::{name}"

    async def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.created:
            raise KeyError(name)
        return self.created[name]

    async def append_to_dataset(self, name: str, data: pd.DataFrame) -> None:
        if name not in self.created:
            # Mirror PhoenixProvider semantics: treat missing as a hard miss
            # so the manager falls back to create_dataset().
            raise KeyError(name)
        self.append_calls.append((name, data))
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self):
        self.datasets = FakeDatasetStore()
        self.experiments = AsyncMock()


class TestExperimentMetricsDataclass:
    def test_to_row_serialises_extras_to_json_string(self):
        m = ExperimentMetrics(
            tenant_id="t1",
            agent_type="search_agent",
            run_id="r1",
            timestamp="2026-05-08T00:00:00+00:00",
            optimizer="BootstrapFewShot",
            baseline_score=0.5,
            candidate_score=0.8,
            improvement=0.3,
            promoted=True,
            train_examples=64,
            extra_metrics={"latency_p95_ms": 230, "judge_score": 0.91},
        )
        row = m.to_row()
        assert row["tenant_id"] == "t1"
        assert row["promoted"] is True
        # extras flattened to JSON string
        assert isinstance(row["extra_metrics"], str)
        parsed = json.loads(row["extra_metrics"])
        assert parsed == {"latency_p95_ms": 230, "judge_score": 0.91}

    def test_from_row_round_trip(self):
        original = ExperimentMetrics(
            tenant_id="t1",
            agent_type="search_agent",
            run_id="r1",
            timestamp="2026-05-08T00:00:00+00:00",
            optimizer="MIPROv2",
            baseline_score=0.5,
            candidate_score=0.4,
            improvement=-0.1,
            promoted=False,
            train_examples=32,
            extra_metrics={"k": "v"},
        )
        rebuilt = ExperimentMetrics.from_row(original.to_row())
        assert rebuilt == original

    def test_from_row_tolerates_missing_optionals(self):
        rebuilt = ExperimentMetrics.from_row(
            {
                "tenant_id": "t1",
                "agent_type": "x",
                "run_id": "r",
                "timestamp": "now",
                "optimizer": "X",
                # baseline/candidate/improvement/train_examples missing
                "promoted": False,
                "extra_metrics": "{}",
            }
        )
        assert rebuilt.baseline_score is None
        assert rebuilt.candidate_score is None
        assert rebuilt.improvement is None
        assert rebuilt.train_examples is None
        assert rebuilt.extra_metrics == {}


class TestArtifactManagerExperiments:
    @pytest.fixture
    def manager_and_provider(self):
        provider = FakeProvider()
        return ArtifactManager(provider, tenant_id="acme"), provider

    @pytest.mark.asyncio
    async def test_save_first_creates_then_appends(self, manager_and_provider):
        mgr, provider = manager_and_provider
        first = ExperimentMetrics(
            tenant_id="acme",
            agent_type="search_agent",
            run_id="r1",
            timestamp="2026-05-08T01:00:00+00:00",
            optimizer="BootstrapFewShot",
            promoted=True,
        )
        second = ExperimentMetrics(
            tenant_id="acme",
            agent_type="search_agent",
            run_id="r2",
            timestamp="2026-05-08T02:00:00+00:00",
            optimizer="BootstrapFewShot",
            promoted=False,
        )

        await mgr.save_experiment(first)
        await mgr.save_experiment(second)

        # Single dataset created the first time, append on the second.
        ds_name = "dspy-experiments-acme-search_agent"
        assert ds_name in provider.datasets.created
        assert len(provider.datasets.append_calls) == 1
        assert provider.datasets.append_calls[0][0] == ds_name

        # Reading back returns both rows in chronological order.
        history = await mgr.load_experiments("search_agent")
        assert [m.run_id for m in history] == ["r1", "r2"]
        assert history[0].promoted is True
        assert history[1].promoted is False

    @pytest.mark.asyncio
    async def test_tenant_id_mismatch_rejected(self, manager_and_provider):
        mgr, _ = manager_and_provider
        wrong = ExperimentMetrics(
            tenant_id="other_tenant",
            agent_type="x",
            run_id="r",
            timestamp="t",
            optimizer="o",
        )
        with pytest.raises(ValueError, match="tenant_id"):
            await mgr.save_experiment(wrong)

    @pytest.mark.asyncio
    async def test_load_latest_experiment_returns_none_when_empty(
        self, manager_and_provider
    ):
        mgr, _ = manager_and_provider
        latest = await mgr.load_latest_experiment("nope")
        assert latest is None

    @pytest.mark.asyncio
    async def test_log_optimization_run_back_compat_shim(self, manager_and_provider):
        """Old free-form API still works and routes through the typed ledger."""
        mgr, _ = manager_and_provider
        await mgr.log_optimization_run(
            "search_agent",
            {
                "optimizer": "MIPROv2",
                "baseline_score": 0.7,
                "candidate_score": 0.85,
                "improvement": 0.15,
                "promoted": True,
                "train_examples": 128,
                "judge_score": 0.93,  # extras
            },
        )

        latest = await mgr.load_latest_experiment("search_agent")
        assert latest is not None
        assert latest.optimizer == "MIPROv2"
        assert latest.candidate_score == 0.85
        assert latest.promoted is True
        assert latest.train_examples == 128
        assert latest.extra_metrics == {"judge_score": 0.93}

    @pytest.mark.asyncio
    async def test_load_optimization_run_back_compat_shape(self, manager_and_provider):
        """Deprecated load_optimization_run returns the legacy dict shape."""
        mgr, _ = manager_and_provider
        await mgr.log_optimization_run(
            "search_agent",
            {
                "optimizer": "BootstrapFewShot",
                "candidate_score": 0.6,
                "promoted": False,
            },
        )

        legacy = await mgr.load_optimization_run("search_agent")
        assert legacy is not None
        assert legacy["tenant_id"] == "acme"
        assert legacy["agent_type"] == "search_agent"
        assert legacy["metrics"]["optimizer"] == "BootstrapFewShot"
        assert legacy["metrics"]["candidate_score"] == 0.6
        assert legacy["metrics"]["promoted"] is False
