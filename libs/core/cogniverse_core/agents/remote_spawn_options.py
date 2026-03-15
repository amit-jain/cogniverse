"""Remote spawn options for controlling cloud GPU execution at query level.

Provides RemoteSpawnOptions for per-request control over how agents and
functions are spawned on remote infrastructure (Modal, Kubernetes, etc.).

This enables A/B testing between different GPU configurations, dynamic
resource allocation based on query complexity, and cost-aware routing.

References:
    - Modal: https://modal.com/docs (serverless GPU compute)
    - ModalJobConfig: cogniverse_finetuning.training.modal_runner
"""

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class GPUType(str, Enum):
    """Available GPU types for remote spawning."""

    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    H100 = "H100"


class SpawnRegion(str, Enum):
    """Preferred deployment regions."""

    AUTO = "auto"
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"


class SpawnPriority(str, Enum):
    """Execution priority for the spawned job."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class RemoteSpawnOptions(BaseModel):
    """
    Per-request configuration for remote GPU execution.

    Controls how agents and functions are spawned on cloud infrastructure
    (Modal, Kubernetes). Attach to any AgentInput to override default
    resource allocation for that specific request.

    Attributes:
        enabled: Route this request to remote execution instead of local
        gpu: GPU type for the remote container
        gpu_count: Number of GPUs to allocate (1-8)
        cpu: Number of CPU cores (1-32)
        memory: Memory in MB (1024-262144)
        timeout: Max execution time in seconds (30-7200)
        keep_warm: Number of warm containers to maintain (0=cold start)
        max_concurrency: Max concurrent inputs per container (1-100)
        container_idle_timeout: Seconds before idle container shuts down
        region: Preferred deployment region
        priority: Execution priority
        cloud_provider: Target cloud provider (modal, kubernetes)
        environment: Extra environment variables for the container
        volume_mounts: Named volumes to mount (name -> mount_path)

    Usage:
        # Force remote execution with specific GPU
        spawn = RemoteSpawnOptions(enabled=True, gpu=GPUType.A100_80GB)

        # Cost-optimized spawn for simple queries
        spawn = RemoteSpawnOptions(enabled=True, gpu=GPUType.T4, timeout=60)

        # Auto-detect: let the system decide (default)
        input = SearchInput(query="...", remote_spawn=None)

        # Explicit remote with H100 for heavy workloads
        input = SearchInput(
            query="...",
            remote_spawn=RemoteSpawnOptions(enabled=True, gpu=GPUType.H100),
        )

    Example A/B Testing:
        # Group A: Local execution (no remote_spawn)
        input_a = SearchInput(query="...", remote_spawn=None)

        # Group B: Remote GPU execution
        input_b = SearchInput(
            query="...",
            remote_spawn=RemoteSpawnOptions(enabled=True, gpu=GPUType.A10G),
        )

        # Compare in telemetry/Phoenix dashboard:
        # - remote_spawn_enabled=true vs false
        # - Latency distribution by GPU type
        # - Cost per query
        # - Response quality
    """

    enabled: bool = Field(
        default=False,
        description="Route this request to remote GPU execution",
    )
    gpu: GPUType = Field(
        default=GPUType.A10G,
        description="GPU type for the remote container",
    )
    gpu_count: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPUs to allocate (1-8)",
    )
    cpu: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of CPU cores (1-32)",
    )
    memory: int = Field(
        default=16384,
        ge=1024,
        le=262144,
        description="Memory in MB (1024-262144)",
    )
    timeout: int = Field(
        default=3600,
        ge=30,
        le=7200,
        description="Max execution time in seconds (30-7200)",
    )
    keep_warm: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of warm containers to maintain (0=cold start)",
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Max concurrent inputs per container (1-100)",
    )
    container_idle_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Seconds before idle container shuts down (30-3600)",
    )
    region: SpawnRegion = Field(
        default=SpawnRegion.AUTO,
        description="Preferred deployment region",
    )
    priority: SpawnPriority = Field(
        default=SpawnPriority.NORMAL,
        description="Execution priority for the spawned job",
    )
    cloud_provider: str = Field(
        default="modal",
        description="Target cloud provider (modal, kubernetes)",
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for the container",
    )
    volume_mounts: Dict[str, str] = Field(
        default_factory=dict,
        description="Named volumes to mount (name -> mount_path)",
    )

    def to_modal_kwargs(self) -> Dict[str, Any]:
        """
        Convert to Modal @app.function keyword arguments.

        Returns:
            Dict suitable for passing to modal.Function or ModalJobConfig.

        Example:
            opts = RemoteSpawnOptions(gpu=GPUType.H100, memory=65536)
            kwargs = opts.to_modal_kwargs()
            # {"gpu": "H100", "cpu": 4, "memory": 65536, "timeout": 3600}
        """
        kwargs: Dict[str, Any] = {
            "gpu": self.gpu.value,
            "cpu": self.cpu,
            "memory": self.memory,
            "timeout": self.timeout,
        }
        if self.gpu_count > 1:
            kwargs["gpu"] = f"{self.gpu.value}:{self.gpu_count}"
        if self.keep_warm > 0:
            kwargs["keep_warm"] = self.keep_warm
        if self.max_concurrency > 1:
            kwargs["allow_concurrent_inputs"] = self.max_concurrency
        if self.container_idle_timeout != 300:
            kwargs["container_idle_timeout"] = self.container_idle_timeout
        return kwargs

    def should_spawn_remote(self, estimated_complexity: int = 0) -> bool:
        """
        Determine if this request should use remote execution.

        Args:
            estimated_complexity: Estimated complexity score (0-100).
                Higher values indicate heavier workloads.

        Returns:
            True if remote execution should be used for this request.
        """
        return self.enabled
