"""Stage-pipeline durable-execution checkpoint records.

A long-running optimization / auto-eval workflow is a sequence of
heterogeneous stages (span-scan -> build-trainset -> compile -> save ->
distill -> eval), not an agent-task DAG. This record captures enough to
resume after a crash: the ordered stages, the index of the next stage to
run, and the per-unit results already completed (e.g. each agent's compiled
artifact) so a resumed run can skip them. ``cursor`` carries a dataset
offset for eval loops that stream over golden queries / spans.

Unlike the orchestrator's ``WorkflowCheckpoint`` (an agent-task DAG keyed by
task_id and indexed into ``execution_order``), this is shaped for a stage
pipeline plus an optional dataset cursor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PipelineCheckpointStatus(Enum):
    """Lifecycle status of a pipeline workflow run."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineCheckpointConfig:
    """Configuration for pipeline checkpointing."""

    project_name: str = "pipeline_checkpoints"
    retain_completed_hours: int = 24 * 7
    retain_failed_hours: int = 24 * 30


@dataclass
class PipelineCheckpoint:
    """Resumable state of a long-running stage-pipeline workflow."""

    checkpoint_id: str
    workflow_id: str
    tenant_id: str
    status: str  # PipelineCheckpointStatus value
    phases: List[str]
    phase_index: int
    completed_units: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    cursor: Optional[Dict[str, Any]] = None
    resume_count: int = 0

    def pending_phases(self) -> List[str]:
        """Stages still to run — a resumed run starts here."""
        return self.phases[self.phase_index :]

    def completed_unit_keys(self) -> Set[str]:
        """Unit keys already completed and therefore skippable on resume.

        A unit recorded with a non-``completed`` status (e.g. ``failed``) is
        NOT skippable — it must re-run.
        """
        return {
            key
            for key, unit in self.completed_units.items()
            if isinstance(unit, dict) and unit.get("status") == "completed"
        }

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to span-attribute-safe scalars (complex fields as JSON)."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "phases": json.dumps(self.phases),
            "phase_index": self.phase_index,
            "completed_units": json.dumps(self.completed_units),
            "cursor": json.dumps(self.cursor) if self.cursor is not None else "",
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "resume_count": self.resume_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        phases = data.get("phases", "[]")
        if isinstance(phases, str):
            phases = json.loads(phases) if phases else []
        # A persisted null/none-list shape must not crash resume later
        # (pending_phases subscripts; completed_unit_keys calls .items()).
        if not isinstance(phases, list):
            phases = []

        completed_units = data.get("completed_units", "{}")
        if isinstance(completed_units, str):
            completed_units = json.loads(completed_units) if completed_units else {}
        if not isinstance(completed_units, dict):
            completed_units = {}

        cursor = data.get("cursor")
        if isinstance(cursor, str):
            cursor = json.loads(cursor) if cursor else None

        metadata = data.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            tenant_id=data["tenant_id"],
            status=data.get("status", "active"),
            phases=phases,
            phase_index=int(data.get("phase_index", 0)),
            completed_units=completed_units,
            metadata=metadata,
            created_at=created_at,
            cursor=cursor,
            resume_count=int(data.get("resume_count", 0)),
        )
