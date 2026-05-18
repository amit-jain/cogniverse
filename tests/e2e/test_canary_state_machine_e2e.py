"""Phase 6b — ArtifactManager canary state machine end-to-end.

Pins the FSM behaviour against the deployed cogniverse Phoenix:

  * promote_to_canary stamps a canary entry with version/promoted_at/traffic_pct;
    promote_canary_to_active flips active and retires the prior active with
    reason "superseded_by_canary_promotion";
  * retire_canary moves the canary to retired with the supplied reason and
    leaves active untouched;
  * load_for_request routing is a stable hash of request_seed: per-seed
    determinism + the empirical canary-traffic share matches the configured
    traffic_pct ± a noise band on N=100 distinct seeds;
  * promote_to_canary rejects traffic_pct outside [1, 100] with the exact
    ValueError message;
  * promote_canary_to_active without a canary raises with the canonical
    "no canary set" substring.
"""

from __future__ import annotations

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.e2e.conftest import run_async, skip_if_no_runtime, unique_id

PHOENIX_HTTP = "http://localhost:26006"
PHOENIX_GRPC = "localhost:4317"


def _make_artifact_manager(tenant_id: str) -> ArtifactManager:
    """ArtifactManager bound to the deployed cluster's Phoenix via host NodePort."""
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": PHOENIX_HTTP,
            "grpc_endpoint": PHOENIX_GRPC,
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


def _run(coro):
    """Run a coroutine in a fresh OS thread.

    Direct ``asyncio.new_event_loop().run_until_complete(coro)`` raises
    ``RuntimeError: This event loop is already running`` under
    pytest.ini's ``asyncio_mode = auto``. See conftest.run_async.
    """
    return run_async(coro)


async def _seed_three_versions(am: ArtifactManager, agent_type: str) -> None:
    """Save three versioned prompts so canary/promote tests have versions to flip."""
    await am.save_prompts_versioned(agent_type, {"system": "v1-text"})
    await am.save_prompts_versioned(agent_type, {"system": "v2-text"})
    await am.save_prompts_versioned(agent_type, {"system": "v3-text"})


# ---------------------------------------------------------------------------
# 1. happy path: promote canary → promote-to-active → previous active retired
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestCanaryLifecycleHappyPath:
    """promote_to_canary then promote_canary_to_active flips state correctly."""

    def test_promote_then_active_with_retired_marker(self) -> None:
        tenant_id = unique_id("opt_lc") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _scenario() -> None:
            await _seed_three_versions(am, agent_type)

            # Set v1 active first so the lifecycle has something to retire
            # when v3 gets promoted from canary to active.
            await am.promote_to_canary(agent_type, version=1, traffic_pct=100)
            await am.promote_canary_to_active(agent_type)
            state = await am.get_artefact_state(agent_type)
            assert state["active"]["version"] == 1
            assert state["canary"] is None

            # Promote v3 to canary at 10%.
            state2 = await am.promote_to_canary(agent_type, version=3, traffic_pct=10)
            assert state2["canary"]["version"] == 3
            assert state2["canary"]["traffic_pct"] == 10
            assert isinstance(state2["canary"]["promoted_at"], str)
            assert state2["active"]["version"] == 1, state2

            # Promote canary → active. Prior active (v1) lands in retired.
            state3 = await am.promote_canary_to_active(agent_type)
            assert state3["canary"] is None
            assert state3["active"]["version"] == 3
            retired_versions = [r["version"] for r in state3["retired"]]
            assert 1 in retired_versions, state3["retired"]
            v1_retire = next(r for r in state3["retired"] if r["version"] == 1)
            assert v1_retire["reason"] == "superseded_by_canary_promotion"

        _run(_scenario())


# ---------------------------------------------------------------------------
# 2. retire_canary moves to retired, active untouched
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPromoteWorseCanaryThenRetire:
    """retire_canary uses the supplied reason and leaves active in place."""

    def test_retire_with_reason(self) -> None:
        tenant_id = unique_id("opt_ret") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _scenario() -> None:
            await _seed_three_versions(am, agent_type)
            await am.promote_to_canary(agent_type, version=3, traffic_pct=100)
            await am.promote_canary_to_active(agent_type)
            # Now promote v1 (a regression candidate) to canary.
            await am.promote_to_canary(agent_type, version=1, traffic_pct=10)
            state = await am.retire_canary(agent_type, reason="metric_regression")
            assert state["canary"] is None
            # Active still v3.
            assert state["active"]["version"] == 3
            last_retired = state["retired"][-1]
            assert last_retired["version"] == 1
            assert last_retired["reason"] == "metric_regression"
            assert isinstance(last_retired["retired_at"], str)

        _run(_scenario())


# ---------------------------------------------------------------------------
# 3. load_for_request routing: stable per seed; empirical traffic ~ traffic_pct
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestStableRoutingByRequestSeed:
    """Per-seed determinism + 10% canary band lands in [5, 20] over N=100."""

    def test_routing_is_stable_and_band_correct(self) -> None:
        tenant_id = unique_id("opt_route") + ":t1"
        agent_type = "search_agent"
        am = _make_artifact_manager(tenant_id)

        async def _scenario() -> None:
            await _seed_three_versions(am, agent_type)
            # Active v2, canary v3 at 10%.
            await am.promote_to_canary(agent_type, version=2, traffic_pct=100)
            await am.promote_canary_to_active(agent_type)
            await am.promote_to_canary(agent_type, version=3, traffic_pct=10)

            seeds = [f"user_{i:03d}" for i in range(100)]
            served_first_pass = []
            for seed in seeds:
                view = await am.load_for_request(agent_type, request_seed=seed)
                served_first_pass.append(view["served_from"])

            canary_count = sum(1 for s in served_first_pass if s == "canary")
            # 10% ± noise: hash distribution over 100 seeds gives a tight
            # band. Values outside this would mean either the routing
            # bucket math regressed or the configured traffic_pct didn't
            # reach the load path.
            assert 5 <= canary_count <= 20, (
                f"canary share drifted: {canary_count}/100 (expected ~10)"
            )

            # Per-seed determinism: re-running the same seed must return
            # the exact same served_from / version.
            for seed, expected in list(zip(seeds, served_first_pass))[:8]:
                view = await am.load_for_request(agent_type, request_seed=seed)
                assert view["served_from"] == expected, (
                    f"routing not stable for seed={seed}: "
                    f"first={expected}, second={view['served_from']}"
                )

        _run(_scenario())


# ---------------------------------------------------------------------------
# 4. promote_to_canary input validation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPromoteToCanaryRejectsInvalidTrafficPct:
    """traffic_pct outside [1, 100] raises ValueError with the canonical text."""

    @pytest.mark.parametrize("bad_pct", [0, 101, -5])
    def test_invalid_pct_rejected(self, bad_pct: int) -> None:
        tenant_id = unique_id("opt_pct") + ":t1"
        am = _make_artifact_manager(tenant_id)
        with pytest.raises(ValueError) as exc:
            _run(am.promote_to_canary("search_agent", version=1, traffic_pct=bad_pct))
        assert "traffic_pct must be in [1, 100]" in str(exc.value), exc.value


# ---------------------------------------------------------------------------
# 5. promote_canary_to_active without a canary raises
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPromoteCanaryToActiveWithoutCanaryFails:
    """A fresh agent with no canary set raises ValueError 'no canary set'."""

    def test_no_canary_raises(self) -> None:
        tenant_id = unique_id("opt_nc") + ":t1"
        am = _make_artifact_manager(tenant_id)
        with pytest.raises(ValueError) as exc:
            _run(am.promote_canary_to_active("fresh_agent_with_no_state"))
        assert "no canary set" in str(exc.value), exc.value
