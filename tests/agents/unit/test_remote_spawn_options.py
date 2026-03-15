"""Unit tests for RemoteSpawnOptions query-level configuration.

Tests RemoteSpawnOptions model, ModalJobConfig integration, and agent input wiring.
"""

import pytest

from cogniverse_core.agents.remote_spawn_options import (
    GPUType,
    RemoteSpawnOptions,
    SpawnPriority,
    SpawnRegion,
)


class TestRemoteSpawnOptions:
    """Test RemoteSpawnOptions configuration."""

    def test_disabled_by_default(self):
        """Remote spawn should be disabled by default."""
        opts = RemoteSpawnOptions()
        assert opts.enabled is False
        assert opts.should_spawn_remote() is False

    def test_explicit_enable(self):
        """Explicit enable should route to remote execution."""
        opts = RemoteSpawnOptions(enabled=True)
        assert opts.should_spawn_remote() is True

    def test_default_values(self):
        """Verify default configuration values."""
        opts = RemoteSpawnOptions()
        assert opts.gpu == GPUType.A10G
        assert opts.gpu_count == 1
        assert opts.cpu == 4
        assert opts.memory == 16384
        assert opts.timeout == 3600
        assert opts.keep_warm == 0
        assert opts.max_concurrency == 1
        assert opts.container_idle_timeout == 300
        assert opts.region == SpawnRegion.AUTO
        assert opts.priority == SpawnPriority.NORMAL
        assert opts.cloud_provider == "modal"
        assert opts.environment == {}
        assert opts.volume_mounts == {}

    def test_custom_configuration(self):
        """Custom configuration should be preserved."""
        opts = RemoteSpawnOptions(
            enabled=True,
            gpu=GPUType.H100,
            gpu_count=2,
            cpu=16,
            memory=65536,
            timeout=7200,
            keep_warm=2,
            max_concurrency=10,
            region=SpawnRegion.US_EAST,
            priority=SpawnPriority.HIGH,
            cloud_provider="kubernetes",
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
            volume_mounts={"model-cache": "/cache"},
        )
        assert opts.enabled is True
        assert opts.gpu == GPUType.H100
        assert opts.gpu_count == 2
        assert opts.cpu == 16
        assert opts.memory == 65536
        assert opts.timeout == 7200
        assert opts.keep_warm == 2
        assert opts.max_concurrency == 10
        assert opts.region == SpawnRegion.US_EAST
        assert opts.priority == SpawnPriority.HIGH
        assert opts.cloud_provider == "kubernetes"
        assert opts.environment == {"CUDA_VISIBLE_DEVICES": "0,1"}
        assert opts.volume_mounts == {"model-cache": "/cache"}

    def test_gpu_count_bounds(self):
        """gpu_count should be bounded between 1 and 8."""
        opts = RemoteSpawnOptions(gpu_count=1)
        assert opts.gpu_count == 1

        opts = RemoteSpawnOptions(gpu_count=8)
        assert opts.gpu_count == 8

        with pytest.raises(ValueError):
            RemoteSpawnOptions(gpu_count=0)

        with pytest.raises(ValueError):
            RemoteSpawnOptions(gpu_count=9)

    def test_timeout_bounds(self):
        """timeout should be bounded between 30 and 7200."""
        opts = RemoteSpawnOptions(timeout=30)
        assert opts.timeout == 30

        opts = RemoteSpawnOptions(timeout=7200)
        assert opts.timeout == 7200

        with pytest.raises(ValueError):
            RemoteSpawnOptions(timeout=29)

        with pytest.raises(ValueError):
            RemoteSpawnOptions(timeout=7201)

    def test_memory_bounds(self):
        """memory should be bounded between 1024 and 262144."""
        opts = RemoteSpawnOptions(memory=1024)
        assert opts.memory == 1024

        opts = RemoteSpawnOptions(memory=262144)
        assert opts.memory == 262144

        with pytest.raises(ValueError):
            RemoteSpawnOptions(memory=512)

        with pytest.raises(ValueError):
            RemoteSpawnOptions(memory=262145)

    def test_cpu_bounds(self):
        """cpu should be bounded between 1 and 32."""
        with pytest.raises(ValueError):
            RemoteSpawnOptions(cpu=0)

        with pytest.raises(ValueError):
            RemoteSpawnOptions(cpu=33)


class TestRemoteSpawnOptionsToModalKwargs:
    """Test to_modal_kwargs conversion."""

    def test_default_modal_kwargs(self):
        """Default options should produce standard Modal kwargs."""
        opts = RemoteSpawnOptions()
        kwargs = opts.to_modal_kwargs()
        assert kwargs == {
            "gpu": "A10G",
            "cpu": 4,
            "memory": 16384,
            "timeout": 3600,
        }

    def test_multi_gpu_modal_kwargs(self):
        """Multiple GPUs should use colon syntax."""
        opts = RemoteSpawnOptions(gpu=GPUType.A100_80GB, gpu_count=4)
        kwargs = opts.to_modal_kwargs()
        assert kwargs["gpu"] == "A100-80GB:4"

    def test_keep_warm_modal_kwargs(self):
        """keep_warm > 0 should appear in kwargs."""
        opts = RemoteSpawnOptions(keep_warm=3)
        kwargs = opts.to_modal_kwargs()
        assert kwargs["keep_warm"] == 3

    def test_concurrency_modal_kwargs(self):
        """max_concurrency > 1 should appear as allow_concurrent_inputs."""
        opts = RemoteSpawnOptions(max_concurrency=10)
        kwargs = opts.to_modal_kwargs()
        assert kwargs["allow_concurrent_inputs"] == 10

    def test_custom_idle_timeout_modal_kwargs(self):
        """Non-default idle timeout should appear in kwargs."""
        opts = RemoteSpawnOptions(container_idle_timeout=600)
        kwargs = opts.to_modal_kwargs()
        assert kwargs["container_idle_timeout"] == 600

    def test_default_idle_timeout_excluded(self):
        """Default idle timeout (300) should not appear in kwargs."""
        opts = RemoteSpawnOptions()
        kwargs = opts.to_modal_kwargs()
        assert "container_idle_timeout" not in kwargs


class TestModalJobConfigFromSpawnOptions:
    """Test ModalJobConfig.from_remote_spawn_options round-trip."""

    def test_default_round_trip(self):
        """Default RemoteSpawnOptions should produce equivalent ModalJobConfig."""
        from cogniverse_finetuning.training.modal_runner import ModalJobConfig

        opts = RemoteSpawnOptions()
        config = ModalJobConfig.from_remote_spawn_options(opts)

        assert config.gpu == "A10G"
        assert config.gpu_count == 1
        assert config.cpu == 4
        assert config.memory == 16384
        assert config.timeout == 3600

    def test_custom_round_trip(self):
        """Custom RemoteSpawnOptions should map correctly to ModalJobConfig."""
        from cogniverse_finetuning.training.modal_runner import ModalJobConfig

        opts = RemoteSpawnOptions(
            gpu=GPUType.H100,
            gpu_count=2,
            cpu=16,
            memory=65536,
            timeout=7200,
        )
        config = ModalJobConfig.from_remote_spawn_options(opts)

        assert config.gpu == "H100"
        assert config.gpu_count == 2
        assert config.cpu == 16
        assert config.memory == 65536
        assert config.timeout == 7200


class TestSearchInputWithRemoteSpawn:
    """Test SearchInput integration with RemoteSpawnOptions."""

    def test_search_input_remote_spawn_none_by_default(self):
        """SearchInput should have remote_spawn=None by default."""
        from cogniverse_agents.search_agent import SearchInput

        input_data = SearchInput(query="test query", tenant_id="test_tenant")
        assert input_data.remote_spawn is None

    def test_search_input_with_remote_spawn_options(self):
        """SearchInput should accept RemoteSpawnOptions."""
        from cogniverse_agents.search_agent import SearchInput

        spawn_opts = RemoteSpawnOptions(enabled=True, gpu=GPUType.H100)
        input_data = SearchInput(
            query="test query", tenant_id="test_tenant", remote_spawn=spawn_opts
        )

        assert input_data.remote_spawn is not None
        assert input_data.remote_spawn.enabled is True
        assert input_data.remote_spawn.gpu == GPUType.H100

    def test_search_input_serialization(self):
        """SearchInput with remote_spawn should serialize correctly."""
        from cogniverse_agents.search_agent import SearchInput

        spawn_opts = RemoteSpawnOptions(enabled=True, gpu=GPUType.A100_80GB)
        input_data = SearchInput(
            query="test", tenant_id="test_tenant", remote_spawn=spawn_opts
        )

        data_dict = input_data.model_dump()
        assert data_dict["remote_spawn"]["enabled"] is True
        assert data_dict["remote_spawn"]["gpu"] == "A100-80GB"

        reconstructed = SearchInput.model_validate(data_dict)
        assert reconstructed.remote_spawn.enabled is True
        assert reconstructed.remote_spawn.gpu == GPUType.A100_80GB


class TestOrchestratorInputWithRemoteSpawn:
    """Test OrchestratorInput integration with RemoteSpawnOptions."""

    def test_orchestrator_input_remote_spawn_none_by_default(self):
        """OrchestratorInput should have remote_spawn=None by default."""
        from cogniverse_agents.orchestrator_agent import OrchestratorInput

        input_data = OrchestratorInput(query="test query")
        assert input_data.remote_spawn is None

    def test_orchestrator_input_with_remote_spawn(self):
        """OrchestratorInput should accept RemoteSpawnOptions."""
        from cogniverse_agents.orchestrator_agent import OrchestratorInput

        spawn_opts = RemoteSpawnOptions(enabled=True, gpu=GPUType.A10G, timeout=120)
        input_data = OrchestratorInput(query="test query", remote_spawn=spawn_opts)

        assert input_data.remote_spawn is not None
        assert input_data.remote_spawn.enabled is True
        assert input_data.remote_spawn.timeout == 120

    def test_orchestrator_input_serialization(self):
        """OrchestratorInput with remote_spawn should serialize correctly."""
        from cogniverse_agents.orchestrator_agent import OrchestratorInput

        spawn_opts = RemoteSpawnOptions(
            enabled=True, gpu=GPUType.T4, priority=SpawnPriority.HIGH
        )
        input_data = OrchestratorInput(query="test", remote_spawn=spawn_opts)

        data_dict = input_data.model_dump()
        assert data_dict["remote_spawn"]["enabled"] is True
        assert data_dict["remote_spawn"]["priority"] == "high"

        reconstructed = OrchestratorInput.model_validate(data_dict)
        assert reconstructed.remote_spawn.priority == SpawnPriority.HIGH


class TestGPUTypeEnum:
    """Test GPUType enum values."""

    def test_all_gpu_types(self):
        """All expected GPU types should be available."""
        assert GPUType.T4.value == "T4"
        assert GPUType.L4.value == "L4"
        assert GPUType.A10G.value == "A10G"
        assert GPUType.A100_40GB.value == "A100-40GB"
        assert GPUType.A100_80GB.value == "A100-80GB"
        assert GPUType.H100.value == "H100"

    def test_gpu_type_from_string(self):
        """GPUType should be constructable from string values."""
        assert GPUType("A10G") == GPUType.A10G
        assert GPUType("H100") == GPUType.H100
        assert GPUType("A100-80GB") == GPUType.A100_80GB
