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


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(return_value="ds-123")
    provider.datasets.list_datasets = AsyncMock(return_value=[])
    provider.datasets.get_dataset = AsyncMock(return_value=pd.DataFrame())
    provider.experiments = MagicMock()
    provider.experiments.create_experiment = AsyncMock(return_value="exp-1")
    provider.experiments.log_run = AsyncMock(return_value="run-1")
    return provider


@pytest.fixture
def artifact_manager(mock_provider):
    return ArtifactManager(telemetry_provider=mock_provider, tenant_id="test-tenant")


class TestVersionedPromptSave:
    @pytest.mark.asyncio
    async def test_first_version_is_1(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(return_value=[])

        ds_id, version = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "You are a router"}
        )
        assert version == 1
        assert ds_id == "ds-123"

        call_kwargs = mock_provider.datasets.create_dataset.call_args.kwargs
        assert "v1" in call_kwargs["name"]
        assert call_kwargs["metadata"]["version"] == 1

    @pytest.mark.asyncio
    async def test_auto_increments(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-prompts-test-tenant-routing-v1", "id": "ds-1"},
                {"name": "dspy-prompts-test-tenant-routing-v2", "id": "ds-2"},
            ]
        )

        ds_id, version = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "Updated prompt"}
        )
        assert version == 3

    @pytest.mark.asyncio
    async def test_ignores_other_agents(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-prompts-test-tenant-other_agent-v5", "id": "ds-5"},
            ]
        )

        _, version = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "prompt"}
        )
        assert version == 1


class TestVersionedDemoSave:
    @pytest.mark.asyncio
    async def test_first_version_is_1(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(return_value=[])

        ds_id, version = await artifact_manager.save_demonstrations_versioned(
            "routing", [{"input": "q1", "output": "a1"}]
        )
        assert version == 1

    @pytest.mark.asyncio
    async def test_auto_increments(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-demos-test-tenant-routing-v1", "id": "ds-1"},
                {"name": "dspy-demos-test-tenant-routing-v3", "id": "ds-3"},
            ]
        )

        _, version = await artifact_manager.save_demonstrations_versioned(
            "routing", [{"input": "q", "output": "a"}]
        )
        assert version == 4


class TestListVersions:
    @pytest.mark.asyncio
    async def test_returns_sorted_versions(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-demos-test-tenant-routing-v3", "id": "ds-3"},
                {"name": "dspy-demos-test-tenant-routing-v1", "id": "ds-1"},
                {"name": "dspy-demos-test-tenant-routing-v2", "id": "ds-2"},
            ]
        )

        versions = await artifact_manager.list_versions("demos", "routing")
        assert len(versions) == 3
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2
        assert versions[2]["version"] == 3

    @pytest.mark.asyncio
    async def test_empty_when_no_versions(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(return_value=[])
        versions = await artifact_manager.list_versions("prompts", "routing")
        assert versions == []

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-demos-test-tenant-routing-v1", "id": "ds-1"},
                {"name": "dspy-demos-test-tenant-other-v1", "id": "ds-other"},
            ]
        )

        versions = await artifact_manager.list_versions("demos", "routing")
        assert len(versions) == 1
        assert versions[0]["dataset_id"] == "ds-1"


class TestVersionLineage:
    @pytest.mark.asyncio
    async def test_lineage_includes_row_counts(self, artifact_manager, mock_provider):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-demos-test-tenant-routing-v1", "id": "ds-1"},
                {"name": "dspy-demos-test-tenant-routing-v2", "id": "ds-2"},
            ]
        )
        mock_provider.datasets.get_dataset = AsyncMock(
            return_value=pd.DataFrame([{"input": "q", "output": "a"}] * 5)
        )

        lineage = await artifact_manager.get_version_lineage("demos", "routing")
        assert len(lineage) == 2
        assert lineage[0]["version"] == 1
        assert lineage[0]["row_count"] == 5
        assert lineage[1]["version"] == 2
        assert lineage[1]["row_count"] == 5

    @pytest.mark.asyncio
    async def test_lineage_handles_missing_dataset(
        self, artifact_manager, mock_provider
    ):
        mock_provider.datasets.list_datasets = AsyncMock(
            return_value=[
                {"name": "dspy-demos-test-tenant-routing-v1", "id": "ds-1"},
            ]
        )
        mock_provider.datasets.get_dataset = AsyncMock(side_effect=KeyError("gone"))

        lineage = await artifact_manager.get_version_lineage("demos", "routing")
        assert len(lineage) == 1
        assert lineage[0]["row_count"] == 0


class TestVersionedSaveRoundTrip:
    @pytest.mark.asyncio
    async def test_save_v1_then_v2_then_list(self, artifact_manager, mock_provider):
        """Full round-trip: save v1 → save v2 → list → verify both exist."""
        created = []

        async def mock_create(name, data, metadata):
            created.append({"name": name, "metadata": metadata})
            return f"ds-{len(created)}"

        async def mock_list():
            return [{"name": c["name"], "id": f"ds-{i+1}"} for i, c in enumerate(created)]

        mock_provider.datasets.create_dataset = mock_create
        mock_provider.datasets.list_datasets = mock_list

        _, v1 = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "v1 prompt"}
        )
        assert v1 == 1

        _, v2 = await artifact_manager.save_prompts_versioned(
            "routing", {"system": "v2 prompt"}
        )
        assert v2 == 2

        versions = await artifact_manager.list_versions("prompts", "routing")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2
