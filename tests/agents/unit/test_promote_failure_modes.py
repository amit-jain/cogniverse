"""failure-mode coverage for ``promote_if_better`` and friends.

Real optimization runs hit transient failures: LM 5xx, Phoenix outage,
process kill mid-write. These tests pin the contract around partial state.

Contract:
  * If ``save_prompts`` raises (the FIRST artefact write), nothing else is
    attempted: no demos, no experiment row. Active state is unchanged.
  * If ``save_demonstrations`` raises after ``save_prompts`` succeeded, the
    active slot is reverted so the unvetted candidate never serves: with a
    prior active the previous pair is restored, with no prior active (a fresh
    tenant) the candidate prompts are cleared. The experiment row is NOT
    written (we cannot truthfully claim promoted=true for a half-written
    artefact set). The original exception propagates.
  * If ``save_experiment`` raises AFTER artefact saves succeeded, both
    artefact halves are persisted. The exception propagates so the
    operator sees the failure; this is preferable to silent loss.
  * On the rejection path, any save_experiment failure also propagates
    (rejection runs are valuable audit data; we do not swallow that loss).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import DatasetStore


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

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

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


def _io(demos):
    """Normalize loaded demonstrations to their input/output pairs."""
    return [{"input": d["input"], "output": d["output"]} for d in demos or []]


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
    async def test_fresh_tenant_demos_failure_clears_leaked_prompts(self):
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

        # Fresh tenant: the prompts landed before demos failed, but with no
        # prior active to restore the compensation CLEARS the slot so the
        # unvetted candidate never serves via the default path.
        assert _ds_prompts("x") not in store.created
        # Demos write failed; not present.
        assert _ds_demos("x") not in store.created
        # Experiment row NOT written — promote_if_better cannot truthfully
        # claim promoted=true when the artefact set is half-written.
        assert _ds_experiments("x") not in store.created
        # The candidate does not serve to any traffic.
        served = await mgr.load_for_request("x", request_seed="anyseed")
        assert served["served_from"] == "default"
        assert served["prompts"] is None


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

    async def test_restore_demos_failure_restores_previous_active(self):
        """A mid-restore demos-save failure (prompts already advanced) must be
        compensated too — the restore is torn just like a state-save failure, so
        the previous active prompts AND demos are put back, not left with prompts
        at the canary version and demos/state stale."""
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        # Establish active v1 with both prompts and demos.
        await mgr.save_prompts_versioned("search_agent", {"system": "V1"})
        await mgr.save_demonstrations_versioned(
            "search_agent", [{"input": "q1", "output": "a1"}]
        )
        await mgr.promote_to_canary("search_agent", 1, traffic_pct=100)
        await mgr.promote_canary_to_active("search_agent")

        def _io(demos):
            return [{"input": d["input"], "output": d["output"]} for d in demos or []]

        assert await mgr.load_prompts("search_agent") == {"system": "V1"}
        assert _io(await mgr.load_demonstrations("search_agent")) == [
            {"input": "q1", "output": "a1"}
        ]

        # Stage v2 as canary.
        await mgr.save_prompts_versioned("search_agent", {"system": "V2"})
        await mgr.save_demonstrations_versioned(
            "search_agent", [{"input": "q2", "output": "a2"}]
        )
        await mgr.promote_to_canary("search_agent", 2, traffic_pct=100)

        # Fail only the FIRST demos save (the one inside the restore, after the
        # v2 prompts already landed); the compensation's demos restore succeeds.
        real_save_demos = mgr.save_demonstrations
        calls = {"n": 0}

        async def failing_then_ok(agent_type, demos):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated Phoenix 503 on demos save")
            return await real_save_demos(agent_type, demos)

        mgr.save_demonstrations = failing_then_ok
        try:
            with pytest.raises(ConnectionError):
                await mgr.promote_canary_to_active("search_agent")
        finally:
            mgr.save_demonstrations = real_save_demos

        # Torn restore healed: previous active prompts AND demos restored, and
        # the state blob still shows v1 active / v2 canary (never advanced).
        assert await mgr.load_prompts("search_agent") == {"system": "V1"}
        assert _io(await mgr.load_demonstrations("search_agent")) == [
            {"input": "q1", "output": "a1"}
        ]
        state = await mgr.get_artefact_state("search_agent")
        assert state["active"]["version"] == 1
        assert state["canary"]["version"] == 2


@pytest.mark.asyncio
class TestNonVersionedTornPromotion:
    """The non-versioned promote path writes prompts then demos into the active
    slot. A demos failure after prompts landed would leave new prompts + OLD
    demos — a torn active. With a prior active, the pair must be restored."""

    async def test_demos_failure_restores_previous_active_pair(self):
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        # Prior active: P0 + D0 at the un-versioned slot.
        await mgr.save_prompts("x", {"system": "P0"})
        await mgr.save_demonstrations("x", [{"input": "q0", "output": "a0"}])

        # Fail only the FORWARD candidate demos save (first call); the
        # compensation's demos restore (second call) succeeds.
        real_save_demos = mgr.save_demonstrations
        calls = {"n": 0}

        async def failing_then_ok(agent_type, demos):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated Phoenix 503 on demos save")
            return await real_save_demos(agent_type, demos)

        mgr.save_demonstrations = failing_then_ok
        try:
            with pytest.raises(ConnectionError):
                await mgr.promote_if_better(
                    agent_type="x",
                    candidate_prompts={"system": "P_NEW"},
                    candidate_demos=[{"input": "q1", "output": "a1"}],
                    baseline_score=0.5,
                    candidate_score=0.7,
                    snapshot_before_promote=False,
                )
        finally:
            mgr.save_demonstrations = real_save_demos

        # Active restored to the previous pair — NOT P_NEW + D0.
        assert await mgr.load_prompts("x") == {"system": "P0"}
        assert _io(await mgr.load_demonstrations("x")) == [
            {"input": "q0", "output": "a0"}
        ]
        # No experiment row for a rolled-back promotion.
        assert _ds_experiments("x") not in store.created


@pytest.mark.asyncio
class TestFailedActivePromotionRetiresCanary:
    """serve_versioned promotes a candidate to canary@100% then to active. If
    the active promotion fails, the canary@100% must be retired — otherwise the
    unpromoted candidate keeps serving ALL traffic via the canary overlay."""

    async def test_failed_promote_canary_to_active_retires_canary(self):
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        # Prior active v1.
        await mgr.save_prompts_versioned("x", {"system": "V1"})
        await mgr.promote_to_canary("x", 1, traffic_pct=100)
        await mgr.promote_canary_to_active("x")

        async def failing_promote(agent_type):
            raise ConnectionError("simulated Phoenix 503 on active promotion")

        mgr.promote_canary_to_active = failing_promote

        with pytest.raises(ConnectionError):
            await mgr.promote_if_better(
                agent_type="x",
                candidate_prompts={"system": "V2"},
                candidate_demos=None,
                baseline_score=0.5,
                candidate_score=0.7,
                serve_versioned=True,
                snapshot_before_promote=False,
            )

        # The canary@100% was retired — no unpromoted candidate serves all
        # traffic via the overlay; active still points at the prior version.
        state = await mgr.get_artefact_state("x")
        assert state.get("canary") is None
        assert state["active"]["version"] == 1


@pytest.mark.asyncio
class TestRollbackCompensation:
    """rollback_to_version restores the active pair from a versioned snapshot;
    a missing target version or a mid-rollback failure must never leave the
    active slot torn (new prompts + old demos)."""

    async def test_missing_target_version_leaves_active_untouched(self):
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        await mgr.save_prompts("x", {"system": "P0"})
        await mgr.save_demonstrations("x", [{"input": "q0", "output": "a0"}])
        await mgr.save_prompts_versioned("x", {"system": "P1"})  # -> v1, no demos v99

        with pytest.raises(ValueError, match="demos version 99 not found"):
            await mgr.rollback_to_version("x", prompts_version=1, demos_version=99)

        # Both target datasets are resolved before ANY write, so the missing
        # demos aborted before the prompts advanced.
        assert await mgr.load_prompts("x") == {"system": "P0"}
        assert _io(await mgr.load_demonstrations("x")) == [
            {"input": "q0", "output": "a0"}
        ]

    async def test_demos_save_failure_restores_previous_active(self):
        store = FaultInjectingStore(mode=_Mode.NORMAL)
        mgr = ArtifactManager(FakeProvider(store), tenant_id="acme")

        await mgr.save_prompts("x", {"system": "P0"})
        await mgr.save_demonstrations("x", [{"input": "q0", "output": "a0"}])
        await mgr.save_prompts_versioned("x", {"system": "P1"})  # -> v1
        await mgr.save_demonstrations_versioned(
            "x", [{"input": "q1", "output": "a1"}]
        )  # -> v1

        real_save_demos = mgr.save_demonstrations
        calls = {"n": 0}

        async def failing_then_ok(agent_type, demos):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated 503 on demos save")
            return await real_save_demos(agent_type, demos)

        mgr.save_demonstrations = failing_then_ok
        try:
            with pytest.raises(ConnectionError):
                await mgr.rollback_to_version("x", prompts_version=1, demos_version=1)
        finally:
            mgr.save_demonstrations = real_save_demos

        # Restored to the previous active pair, not P1 + D0.
        assert await mgr.load_prompts("x") == {"system": "P0"}
        assert _io(await mgr.load_demonstrations("x")) == [
            {"input": "q0", "output": "a0"}
        ]


class FirstPromoteStore(DatasetStore):
    """create overwrites (last-write-wins). Fails the create for the
    un-versioned demos dataset exactly once — the mid-copy demos write that
    lands after the un-versioned prompts write has already succeeded."""

    def __init__(self):
        self.data: dict[str, pd.DataFrame] = {}
        self.armed = True

    @staticmethod
    def _is_unversioned_demos(name: str) -> bool:
        return name.startswith("dspy-demos-") and not re.search(r"-v\d+$", name)

    async def create_dataset(self, name, data, metadata=None):
        if self.armed and self._is_unversioned_demos(name):
            self.armed = False
            raise ConnectionError("demos create failed mid-copy")
        self.data[name] = data.copy()
        return name

    async def get_dataset(self, name):
        if name not in self.data:
            raise KeyError(name)
        return self.data[name]

    async def append_to_dataset(self, name, data, metadata=None):
        raise KeyError("no dataset")

    async def delete_dataset(self, name):
        return self.data.pop(name, None) is not None


@pytest.mark.asyncio
class TestFirstPromotionClearsSlot:
    """A FRESH tenant's first promotion has no prior un-versioned active. When
    a mid-copy demos save fails AFTER the un-versioned prompts save landed, the
    un-versioned prompts slot holds the unvetted candidate. With no prior to
    restore, the compensation must CLEAR the slot (delete the un-versioned
    dataset) so the failed candidate never serves to any traffic via the
    default path."""

    async def test_serve_versioned_clears_leaked_candidate_prompts(self):
        store = FirstPromoteStore()
        mgr = ArtifactManager(FakeProvider(store), tenant_id="tornco")
        agent = "router"

        with pytest.raises(ConnectionError, match="demos create failed mid-copy"):
            await mgr.promote_if_better(
                agent_type=agent,
                candidate_prompts={"system": "CANDIDATE_PROMPT"},
                candidate_demos=[{"input": "i", "output": "o", "metadata": "{}"}],
                baseline_score=0.5,
                candidate_score=0.9,
                serve_versioned=True,
                snapshot_before_promote=False,
            )

        served = await mgr.load_for_request(agent, request_seed="anyseed")
        assert served["served_from"] == "default"
        assert served["prompts"] is None

        assert await mgr.load_prompts(agent) is None

        state = await mgr.get_artefact_state(agent)
        assert state.get("active") is None
        assert state.get("canary") is None

    async def test_non_serve_versioned_clears_leaked_candidate_prompts(self):
        store = FirstPromoteStore()
        mgr = ArtifactManager(FakeProvider(store), tenant_id="tornco")
        agent = "router"

        with pytest.raises(ConnectionError, match="demos create failed mid-copy"):
            await mgr.promote_if_better(
                agent_type=agent,
                candidate_prompts={"system": "CANDIDATE_PROMPT"},
                candidate_demos=[{"input": "i", "output": "o", "metadata": "{}"}],
                baseline_score=0.5,
                candidate_score=0.9,
                serve_versioned=False,
                snapshot_before_promote=False,
            )

        served = await mgr.load_for_request(agent, request_seed="anyseed")
        assert served["served_from"] == "default"
        assert served["prompts"] is None

        assert await mgr.load_prompts(agent) is None

        state = await mgr.get_artefact_state(agent)
        assert state.get("active") is None
        assert state.get("canary") is None
