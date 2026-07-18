"""Unit tests for the per-tenant canary promotion state machine."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


class FakeStore:
    """Captures dataset writes + supports get/append for canary tests."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.create_calls: list[str] = []

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
        # Blob saves delete-then-create for last-write-wins (the real
        # provider contract) — absent names are a no-op False.
        return self.created.pop(name, None) is not None

    async def append_to_dataset(
        self, name: str, data: pd.DataFrame, metadata: dict | None = None
    ) -> None:
        if name not in self.created:
            raise KeyError(name)
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self):
        self.datasets = FakeStore()


@pytest.fixture
def manager_provider():
    provider = FakeProvider()
    mgr = ArtifactManager(provider, tenant_id="acme")
    return mgr, provider


def _ds_prompts_v(agent: str, v: int) -> str:
    return f"dspy-prompts-acme-{agent}-v{v}"


def _ds_demos_v(agent: str, v: int) -> str:
    return f"dspy-demos-acme-{agent}-v{v}"


@pytest.mark.asyncio
class TestStateBlobLifecycle:
    async def test_initial_state_is_empty(self, manager_provider):
        mgr, _ = manager_provider
        state = await mgr.get_artefact_state("agent_a")
        assert state == {"active": None, "canary": None, "retired": []}

    async def test_promote_to_canary_writes_state(self, manager_provider):
        mgr, _ = manager_provider
        state = await mgr.promote_to_canary("agent_a", version=3, traffic_pct=15)
        assert state["canary"] == {
            "version": 3,
            "traffic_pct": 15,
            "promoted_at": state["canary"]["promoted_at"],  # iso string
        }

    async def test_invalid_traffic_pct_rejected(self, manager_provider):
        mgr, _ = manager_provider
        with pytest.raises(ValueError):
            await mgr.promote_to_canary("agent_a", version=1, traffic_pct=0)
        with pytest.raises(ValueError):
            await mgr.promote_to_canary("agent_a", version=1, traffic_pct=101)

    async def test_replacing_canary_retires_previous(self, manager_provider):
        mgr, _ = manager_provider
        await mgr.promote_to_canary("agent_a", version=1)
        state = await mgr.promote_to_canary("agent_a", version=2)
        assert state["canary"]["version"] == 2
        retired_versions = [r["version"] for r in state["retired"]]
        assert 1 in retired_versions

    async def test_retire_canary_clears_slot(self, manager_provider):
        mgr, _ = manager_provider
        await mgr.promote_to_canary("agent_a", version=4, traffic_pct=20)
        state = await mgr.retire_canary("agent_a", reason="manual_test")
        assert state["canary"] is None
        assert any(
            r["version"] == 4 and r["reason"] == "manual_test" for r in state["retired"]
        )


@pytest.mark.asyncio
class TestCanaryToActivePromotion:
    async def test_promote_canary_to_active_requires_canary(self, manager_provider):
        mgr, _ = manager_provider
        with pytest.raises(ValueError, match="no canary"):
            await mgr.promote_canary_to_active("agent_a")

    async def test_promote_canary_to_active_retires_old_active(self, manager_provider):
        mgr, _provider = manager_provider
        # Build up to v3: each save_prompts_versioned auto-increments.
        await mgr.save_prompts_versioned("agent_a", {"system": "v1-prompts"})
        await mgr.save_prompts_versioned("agent_a", {"system": "v2-prompts"})
        await mgr.save_prompts_versioned("agent_a", {"system": "v3-prompts"})
        await mgr._save_artefact_state(
            "agent_a",
            {
                "active": {"version": 1, "promoted_at": "old"},
                "canary": None,
                "retired": [],
            },
        )
        await mgr.promote_to_canary("agent_a", version=3, traffic_pct=10)

        state = await mgr.promote_canary_to_active("agent_a")

        # Active is now v3; canary cleared.
        assert state["active"]["version"] == 3
        assert state["canary"] is None
        # Old v1 active retired.
        assert any(r["version"] == 1 for r in state["retired"])
        # Active prompts dataset reflects v3 content (restored from v3 snapshot).
        active = await mgr.load_prompts("agent_a")
        assert active == {"system": "v3-prompts"}


class TestRouteToCanary:
    def test_routing_in_band(self):
        # 100% traffic → every seed routes to canary.
        for seed in ["a", "b", "c", "d"]:
            assert ArtifactManager._route_to_canary(seed, 100) is True

    def test_routing_out_of_band(self):
        # 0% traffic → nothing routes.
        for seed in ["a", "b", "c", "d"]:
            assert ArtifactManager._route_to_canary(seed, 0) is False

    def test_routing_is_stable(self):
        # Same seed always returns the same decision.
        for seed in ["abc", "xyz", "12345"]:
            d1 = ArtifactManager._route_to_canary(seed, 50)
            d2 = ArtifactManager._route_to_canary(seed, 50)
            assert d1 == d2

    def test_routing_distribution_roughly_matches_pct(self):
        seeds = [f"req_{i}" for i in range(10000)]
        canary = sum(1 for s in seeds if ArtifactManager._route_to_canary(s, 10))
        # 10% target with 10k seeds → expect 800–1200 (~3σ window).
        assert 800 <= canary <= 1200, (
            f"distribution outside 8–12% band: got {canary} of 10000"
        )


@pytest.mark.asyncio
class TestLoadForRequest:
    async def test_canary_served_for_in_band_seed(self, manager_provider):
        mgr, _ = manager_provider
        # Seed v2 versioned prompts as the canary content.
        await mgr.save_prompts_versioned("agent_a", {"system": "canary-v2"})
        await mgr.promote_to_canary("agent_a", version=1, traffic_pct=100)

        out = await mgr.load_for_request("agent_a", request_seed="anything")
        assert out["served_from"] == "canary"
        assert out["version"] == 1
        assert out["prompts"] == {"system": "canary-v2"}

    async def test_active_served_when_canary_misses(self, manager_provider):
        mgr, _ = manager_provider
        # Seed v1 prompts as the active dataset.
        await mgr.save_prompts_versioned("agent_a", {"system": "active-v1"})
        # Use traffic_pct=1 (smallest legal value) and a seed that hashes
        # outside the 1% band, so the canary path is *never* taken.
        await mgr.promote_to_canary("agent_a", version=2, traffic_pct=1)
        # Then overwrite state so active=v1, canary=v2 with 1% traffic.
        await mgr._save_artefact_state(
            "agent_a",
            {
                "active": {"version": 1, "promoted_at": "now"},
                "canary": {
                    "version": 2,
                    "promoted_at": "now",
                    "traffic_pct": 1,
                },
                "retired": [],
            },
        )

        # Find a seed that the routing function considers OUT of the 1%
        # canary band; the test then asserts the active arm wins.
        seed = next(
            s
            for s in (f"req_{i}" for i in range(1000))
            if not ArtifactManager._route_to_canary(s, 1)
        )
        out = await mgr.load_for_request("agent_a", request_seed=seed)
        assert out["served_from"] == "active"
        assert out["version"] == 1
        assert out["prompts"] == {"system": "active-v1"}

    async def test_default_path_when_no_state(self, manager_provider):
        mgr, _ = manager_provider
        # No state, no versioned datasets — falls back to load_prompts (None).
        out = await mgr.load_for_request("agent_a", request_seed="seed")
        assert out["served_from"] == "default"
        assert out["version"] is None

    async def test_canary_dataset_missing_falls_back_to_active(self, manager_provider):
        mgr, _ = manager_provider
        # Promote canary v9 but never save the v9 dataset.
        await mgr.save_prompts_versioned("agent_a", {"system": "active-v1"})
        await mgr.promote_to_canary("agent_a", version=9, traffic_pct=100)
        await mgr._save_artefact_state(
            "agent_a",
            {
                "active": {"version": 1, "promoted_at": "now"},
                "canary": {
                    "version": 9,
                    "promoted_at": "now",
                    "traffic_pct": 100,
                },
                "retired": [],
            },
        )

        out = await mgr.load_for_request("agent_a", request_seed="seed")
        # Canary dataset missing — fall back to active.
        assert out["served_from"] == "active"
        assert out["version"] == 1
