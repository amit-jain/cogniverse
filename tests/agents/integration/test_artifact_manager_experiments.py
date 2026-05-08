"""C.1 integration — ArtifactManager.save_experiment against real Phoenix.

Skip-guarded behind Phoenix availability. When Phoenix is up, this verifies:
  * a typed ExperimentMetrics record persists as a row in the dedicated
    experiments dataset (no save_blob, no overwrite-per-run);
  * subsequent saves append rather than overwrite, so the full history
    is queryable via ``load_experiments``;
  * cross-tenant safety: the manager refuses an ExperimentMetrics with a
    mismatched tenant_id before any HTTP I/O happens.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    ArtifactManager,
    ExperimentMetrics,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

PHOENIX_HTTP = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")


def _phoenix_available() -> bool:
    try:
        return httpx.get(f"{PHOENIX_HTTP}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _phoenix_available(),
        reason=f"Phoenix not running at {PHOENIX_HTTP}",
    ),
]


@pytest.fixture
def artifact_manager() -> ArtifactManager:
    """Real PhoenixProvider against the locally-running Phoenix instance."""
    tenant_id = f"c1_int_{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider(
        http_endpoint=PHOENIX_HTTP,
        tenant_id=tenant_id,
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_save_then_load_experiments_round_trip_real_phoenix(
    artifact_manager,
):
    mgr = artifact_manager

    # Two consecutive runs with different scores — must both persist with
    # the second appearing as the most recent.
    first_run_id = uuid.uuid4().hex
    second_run_id = uuid.uuid4().hex
    first = ExperimentMetrics(
        tenant_id=mgr._tenant_id,
        agent_type="c1_int_agent",
        run_id=first_run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        optimizer="BootstrapFewShot",
        baseline_score=0.5,
        candidate_score=0.55,
        improvement=0.05,
        promoted=True,
        train_examples=16,
        extra_metrics={"judge": 0.7},
    )
    second = ExperimentMetrics(
        tenant_id=mgr._tenant_id,
        agent_type="c1_int_agent",
        run_id=second_run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        optimizer="BootstrapFewShot",
        baseline_score=0.55,
        candidate_score=0.5,
        improvement=-0.05,
        promoted=False,
        train_examples=16,
        extra_metrics={"judge": 0.66, "ablation": "noise_only"},
    )

    await mgr.save_experiment(first)
    await mgr.save_experiment(second)

    history = await mgr.load_experiments("c1_int_agent")
    run_ids = [m.run_id for m in history]
    assert first_run_id in run_ids
    assert second_run_id in run_ids
    assert run_ids.index(first_run_id) < run_ids.index(second_run_id), (
        "history must preserve chronological order"
    )

    latest = await mgr.load_latest_experiment("c1_int_agent")
    assert latest is not None
    assert latest.run_id == second_run_id
    assert latest.candidate_score == 0.5
    assert latest.extra_metrics["ablation"] == "noise_only"


@pytest.mark.asyncio
async def test_tenant_mismatch_rejected_before_phoenix_io(artifact_manager):
    mgr = artifact_manager
    wrong = ExperimentMetrics(
        tenant_id="totally-different-tenant",
        agent_type="c1_int_agent",
        run_id=uuid.uuid4().hex,
        timestamp=datetime.now(timezone.utc).isoformat(),
        optimizer="BootstrapFewShot",
    )
    with pytest.raises(ValueError):
        await mgr.save_experiment(wrong)


@pytest.mark.asyncio
async def test_back_compat_shim_persists_to_typed_ledger(artifact_manager):
    """log_optimization_run must land in the experiments dataset, not save_blob."""
    mgr = artifact_manager
    await mgr.log_optimization_run(
        "c1_int_legacy_agent",
        {
            "optimizer": "MIPROv2",
            "candidate_score": 0.42,
            "promoted": False,
            "extra_signal": "from_back_compat",
        },
    )

    latest = await mgr.load_latest_experiment("c1_int_legacy_agent")
    assert latest is not None
    assert latest.optimizer == "MIPROv2"
    assert latest.candidate_score == 0.42
    assert latest.extra_metrics.get("extra_signal") == "from_back_compat"
