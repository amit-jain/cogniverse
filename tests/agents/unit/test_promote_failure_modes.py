"""failure-mode coverage for ``promote_if_better`` and friends.

Real optimization runs hit transient failures: LM 5xx, Phoenix outage,
process kill mid-write. These tests pin the contract around partial state.

Contract:
  * If ``save_prompts`` raises (the FIRST artefact write), nothing else is
    attempted: no demos, no experiment row. Active state is unchanged.
  * If ``save_demonstrations`` raises after ``save_prompts`` succeeded, the
    prompts ARE persisted (partial promotion). The experiment row is NOT
    written (we cannot truthfully claim promoted=true for a half-written
    artefact set). The original exception propagates.
  * If ``save_experiment`` raises AFTER artefact saves succeeded, both
    artefact halves are persisted. The exception propagates so the
    operator sees the failure; this is preferable to silent loss.
  * On the rejection path, any save_experiment failure also propagates
    (rejection runs are valuable audit data; we do not swallow that loss).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


class _Mode:
    NORMAL = "normal"
    FAIL_PROMPTS = "fail_prompts"
    FAIL_DEMOS = "fail_demos"
    FAIL_EXPERIMENT = "fail_experiment"


class FaultInjectingStore:
    """DatasetStore stub with per-dataset failure injection."""

    def __init__(self, mode: str = _Mode.NORMAL):
        self.mode = mode
        self.created: dict[str, pd.DataFrame] = {}
        self.append_calls: list[tuple[str, pd.DataFrame]] = []

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.mode == _Mode.FAIL_PROMPTS and "prompts" in name:
            raise ConnectionError("simulated Phoenix 503 on prompts save")
        if self.mode == _Mode.FAIL_DEMOS and "demos" in name:
            raise ConnectionError("simulated Phoenix 503 on demos save")
        if self.mode == _Mode.FAIL_EXPERIMENT and "experiments" in name:
            raise ConnectionError("simulated Phoenix 503 on experiment save")
        self.created[name] = data.copy()
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
        if self.mode == _Mode.FAIL_EXPERIMENT and "experiments" in name:
            raise ConnectionError("simulated Phoenix 503 on experiment append")
        if name not in self.created:
            raise KeyError(name)
        self.append_calls.append((name, data))
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self, store: FaultInjectingStore):
        self.datasets = store


def _ds_prompts(agent: str) -> str:
    return f"dspy-prompts-acme:acme-{agent}"


def _ds_demos(agent: str) -> str:
    return f"dspy-demos-acme:acme-{agent}"


def _ds_experiments(agent: str) -> str:
    return f"dspy-experiments-acme:acme-{agent}"


CANDIDATE_PROMPTS = {"system": "v1"}
CANDIDATE_DEMOS = [{"input": "q", "output": "a"}]


@pytest.mark.asyncio
class TestPromptSaveFailure:
    async def test_no_demos_no_experiment_on_prompts_failure(self):
        store = FaultInjectingStore(mode=_Mode.FAIL_PROMPTS)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        with pytest.raises(ConnectionError):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts=CANDIDATE_PROMPTS,
                candidate_demos=CANDIDATE_DEMOS,
                baseline_score=0.5,
                candidate_score=0.7,
            )

        # Nothing else was attempted.
        assert _ds_prompts("x") not in store.created
        assert _ds_demos("x") not in store.created
        assert _ds_experiments("x") not in store.created


@pytest.mark.asyncio
class TestDemoSaveFailure:
    async def test_prompts_persisted_demos_partial_no_experiment(self):
        store = FaultInjectingStore(mode=_Mode.FAIL_DEMOS)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        with pytest.raises(ConnectionError):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts=CANDIDATE_PROMPTS,
                candidate_demos=CANDIDATE_DEMOS,
                baseline_score=0.5,
                candidate_score=0.7,
            )

        # Prompts already landed before demos failed (partial promotion).
        assert _ds_prompts("x") in store.created
        # Demos write failed; not present.
        assert _ds_demos("x") not in store.created
        # Experiment row NOT written — promote_if_better cannot truthfully
        # claim promoted=true when the artefact set is half-written.
        assert _ds_experiments("x") not in store.created


@pytest.mark.asyncio
class TestExperimentSaveFailure:
    async def test_artefacts_persisted_experiment_failure_propagates(self):
        store = FaultInjectingStore(mode=_Mode.FAIL_EXPERIMENT)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        with pytest.raises(ConnectionError):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts=CANDIDATE_PROMPTS,
                candidate_demos=CANDIDATE_DEMOS,
                baseline_score=0.5,
                candidate_score=0.7,
            )

        # Both artefact datasets exist (promotion succeeded at the artefact
        # layer); only the experiment row failed, raising for visibility.
        assert _ds_prompts("x") in store.created
        assert _ds_demos("x") in store.created
        assert _ds_experiments("x") not in store.created

    async def test_rejection_path_experiment_failure_propagates(self):
        """Even rejected runs lose audit data on a save failure — must raise."""
        store = FaultInjectingStore(mode=_Mode.FAIL_EXPERIMENT)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        with pytest.raises(ConnectionError):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts=CANDIDATE_PROMPTS,
                candidate_demos=CANDIDATE_DEMOS,
                baseline_score=0.7,
                candidate_score=0.4,
            )

        # On the rejection path, prompts/demos were never written.
        assert _ds_prompts("x") not in store.created
        assert _ds_demos("x") not in store.created
        # Experiment row also not written — the failure surfaces clearly.
        assert _ds_experiments("x") not in store.created


@pytest.mark.asyncio
class TestEmptyArtefactGuards:
    async def test_empty_prompts_still_logged_as_promoted_if_winner(self):
        """Edge case: a winner with empty prompts dict — record still logged.

        ``save_prompts`` accepts an empty dict (creates an empty dataset);
        ``promote_if_better`` does not double-validate the candidate shape.
        Operators using this gate must validate their candidate before calling.
        """
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        record = await mgr.promote_if_better(
            agent_type="x",
            candidate_prompts={},
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.6,
        )
        assert record.promoted is True
        # An empty-prompt dataset is still a real artefact write.
        assert _ds_prompts("x") in store.created


@pytest.mark.asyncio
class TestRetryCleanlinessOnSecondRun:
    async def test_second_run_after_first_succeeds_uses_append_path(self):
        """Two successful promotions land as two rows in the same dataset."""
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.6,
            run_id="r1",
        )
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v2"},
            candidate_demos=None,
            baseline_score=0.6,
            candidate_score=0.7,
            run_id="r2",
        )

        # Append called exactly once (second run); first run created the dataset.
        ds = _ds_experiments("agent_a")
        assert any(name == ds for name, _ in store.append_calls)

        history = await mgr.load_experiments("agent_a")
        assert [r.run_id for r in history] == ["r1", "r2"]


@pytest.mark.asyncio
class TestTornCanaryPromotion:
    """``promote_canary_to_active`` overwrites the un-versioned active
    artefacts BEFORE saving the state blob. If the state save fails, the
    previously active content must be restored so the init-time read seam
    (un-versioned datasets) and the request-time read seam (state blob)
    stay consistent — otherwise a later ``retire_canary`` reverts traffic
    to prompts that no longer match the active dataset."""

    async def test_state_save_failure_restores_previous_active_prompts(self):
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        await mgr.save_prompts_versioned("search_agent", {"system": "V1"})
        await mgr.promote_to_canary("search_agent", 1, traffic_pct=100)
        await mgr.promote_canary_to_active("search_agent")
        assert await mgr.load_prompts("search_agent") == {"system": "V1"}

        await mgr.save_prompts_versioned("search_agent", {"system": "V2"})
        await mgr.promote_to_canary("search_agent", 2, traffic_pct=100)

        real_save = mgr._save_artefact_state

        async def failing_save(agent_type, state):
            raise ConnectionError("simulated Phoenix 503 on state save")

        mgr._save_artefact_state = failing_save
        try:
            with pytest.raises(ConnectionError, match="state save"):
                await mgr.promote_canary_to_active("search_agent")
        finally:
            mgr._save_artefact_state = real_save

        # Active content restored; state blob still shows v1 active / v2 canary.
        assert await mgr.load_prompts("search_agent") == {"system": "V1"}
        state = await mgr.get_artefact_state("search_agent")
        assert state["active"]["version"] == 1
        assert state["canary"]["version"] == 2
