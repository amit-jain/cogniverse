"""Durable execution for long-running workflows.

Stage-pipeline checkpoint + resume for long-running optimization / eval jobs
that lose all progress when their Argo pod is killed. Consumed today by
``optimization_cli``'s triggered mode; persists to the telemetry-span
substrate.
"""

from cogniverse_core.durable.pipeline_checkpoint import (
    PipelineCheckpoint,
    PipelineCheckpointConfig,
    PipelineCheckpointStatus,
)
from cogniverse_core.durable.pipeline_checkpoint_storage import (
    PipelineCheckpointStorage,
)

__all__ = [
    "PipelineCheckpoint",
    "PipelineCheckpointConfig",
    "PipelineCheckpointStatus",
    "PipelineCheckpointStorage",
]
