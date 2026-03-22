"""
Unit tests for ArtifactManager versioned save/load.

Tests:
1. save_prompts_versioned auto-increments version
2. save_demonstrations_versioned auto-increments version
3. list_versions returns sorted version list
4. get_version_lineage includes row counts
5. Multiple saves produce distinct versions
6. Version numbering starts at 1
"""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


def _make_provider_with_datasets(existing_names: list[str]):
    """Create a mock provider where get_dataset succeeds for named datasets."""
    provider = MagicMock()
    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(return_value="ds-new")

    async def mock_get(name):
        if name in existing_names:
            return pd.DataFrame([{"name": "k", "value": "v"}])
        raise KeyError(f"Dataset not found: {name}")

    provider.datasets.get_dataset = AsyncMock(side_effect=mock_get)
    return provider


@pytest.fixture
def empty_provider():
    return _make_provider_with_datasets([])


@pytest.fixture
def artifact_manager(empty_provider):
    return ArtifactManager(telemetry_provider=empty_provider, tenant_id="test-tenant")


class TestVersionedPromptSave:
    @pytest.mark.asyncio
    async def test_first_version_is_1(self, artifact_manager):
        ds_id, version = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "You are a router"}
        )
        assert version == 1
        assert ds_id == "ds-new"

    @pytest.mark.asyncio
    async def test_auto_increments_past_existing(self):
        provider = _make_provider_with_datasets([
            "dspy-prompts-test-tenant-routing-v1",
            "dspy-prompts-test-tenant-routing-v2",
        ])
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        _, version = await am.save_prompts_versioned("routing", {"system": "v3"})
        assert version == 3

    @pytest.mark.asyncio
    async def test_ignores_other_agents(self):
        provider = _make_provider_with_datasets([
            "dspy-prompts-test-tenant-other_agent-v5",
        ])
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        _, version = await am.save_prompts_versioned("routing", {"system": "prompt"})
        assert version == 1


class TestVersionedDemoSave:
    @pytest.mark.asyncio
    async def test_first_version_is_1(self, artifact_manager):
        ds_id, version = await artifact_manager.save_demonstrations_versioned(
            "routing", [{"input": "q1", "output": "a1"}]
        )
        assert version == 1

    @pytest.mark.asyncio
    async def test_auto_increments(self):
        provider = _make_provider_with_datasets([
            "dspy-demos-test-tenant-routing-v1",
            "dspy-demos-test-tenant-routing-v2",
        ])
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        _, version = await am.save_demonstrations_versioned(
            "routing", [{"input": "q", "output": "a"}]
        )
        assert version == 3


class TestListVersions:
    @pytest.mark.asyncio
    async def test_returns_sorted_versions(self):
        provider = _make_provider_with_datasets([
            "dspy-demos-test-tenant-routing-v1",
            "dspy-demos-test-tenant-routing-v2",
            "dspy-demos-test-tenant-routing-v3",
        ])
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        versions = await am.list_versions("demos", "routing")
        assert len(versions) == 3
        assert [v["version"] for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_when_no_versions(self, artifact_manager):
        versions = await artifact_manager.list_versions("prompts", "routing")
        assert versions == []

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self):
        provider = _make_provider_with_datasets([
            "dspy-demos-test-tenant-routing-v1",
        ])
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        versions = await am.list_versions("demos", "routing")
        assert len(versions) == 1
        assert versions[0]["version"] == 1

        versions_other = await am.list_versions("demos", "other")
        assert versions_other == []


class TestVersionLineage:
    @pytest.mark.asyncio
    async def test_lineage_includes_row_counts(self):
        rows = pd.DataFrame([{"input": "q", "output": "a"}] * 5)

        provider = MagicMock()
        provider.datasets = MagicMock()
        provider.datasets.create_dataset = AsyncMock(return_value="ds-new")

        call_count = 0

        async def mock_get(name):
            nonlocal call_count
            # list_versions probes + get_version_lineage reads
            if "v1" in name or "v2" in name:
                return rows
            raise KeyError("not found")

        provider.datasets.get_dataset = AsyncMock(side_effect=mock_get)
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        lineage = await am.get_version_lineage("demos", "routing")
        assert len(lineage) == 2
        assert lineage[0]["version"] == 1
        assert lineage[0]["row_count"] == 5

    @pytest.mark.asyncio
    async def test_lineage_handles_missing_dataset(self):
        call_count = 0

        async def mock_get(name):
            nonlocal call_count
            call_count += 1
            if "v1" in name and call_count <= 1:
                return pd.DataFrame([{"x": 1}])
            raise KeyError("gone")

        provider = MagicMock()
        provider.datasets = MagicMock()
        provider.datasets.get_dataset = AsyncMock(side_effect=mock_get)
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        lineage = await am.get_version_lineage("demos", "routing")
        assert len(lineage) == 1
        assert lineage[0]["row_count"] == 0


class TestVersionedSaveRoundTrip:
    @pytest.mark.asyncio
    async def test_save_v1_then_v2_then_list(self):
        """Full round-trip: save v1 → save v2 → list → verify both."""
        stored: dict[str, pd.DataFrame] = {}

        async def mock_create(name, data, metadata):
            stored[name] = data
            return f"ds-{len(stored)}"

        async def mock_get(name):
            if name in stored:
                return stored[name]
            raise KeyError(f"Not found: {name}")

        provider = MagicMock()
        provider.datasets = MagicMock()
        provider.datasets.create_dataset = AsyncMock(side_effect=mock_create)
        provider.datasets.get_dataset = AsyncMock(side_effect=mock_get)
        am = ArtifactManager(telemetry_provider=provider, tenant_id="test-tenant")

        _, v1 = await am.save_prompts_versioned("routing", {"system": "v1 prompt"})
        assert v1 == 1

        _, v2 = await am.save_prompts_versioned("routing", {"system": "v2 prompt"})
        assert v2 == 2

        versions = await am.list_versions("prompts", "routing")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2
