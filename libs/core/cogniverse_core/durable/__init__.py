"""Durable execution for long-running workflows.

Stage-pipeline checkpoint + resume for the long-running optimization / eval
jobs (`optimization_cli` modes, `job_executor`) that today lose all progress
when their Argo pod is killed. Persists to the telemetry-span substrate.
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
