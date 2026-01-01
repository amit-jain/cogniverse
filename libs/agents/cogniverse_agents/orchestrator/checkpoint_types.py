"""
Checkpoint Types for Durable Workflow Execution

Data structures for workflow checkpointing, enabling:
- Resume from failure
- Replay completed tasks
- Time-travel debugging (fork from historical checkpoint)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CheckpointLevel(Enum):
    """Granularity of checkpointing"""

    PHASE = "phase"  # Checkpoint after each phase completes
    TASK = "task"  # Checkpoint after each task completes
    PHASE_AND_TASK = "both"  # Checkpoint at both phase and task boundaries


class CheckpointStatus(Enum):
    """Checkpoint lifecycle status"""

    ACTIVE = "active"  # Current checkpoint for workflow
    SUPERSEDED = "superseded"  # Replaced by newer checkpoint
    FAILED = "failed"  # Workflow failed at this checkpoint
    COMPLETED = "completed"  # Workflow completed successfully


@dataclass
class TaskCheckpoint:
    """Serializable state of a single workflow task"""

    task_id: str
    agent_name: str
    query: str
    dependencies: List[str]  # List for JSON serialization (converted from Set)
    status: str  # TaskStatus.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for span storage"""
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": json.dumps(self.result) if self.result else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskCheckpoint":
        """Deserialize from dictionary"""
        result = data.get("result")
        if isinstance(result, str):
            result = json.loads(result)

        start_time = data.get("start_time")
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        end_time = data.get("end_time")
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        return cls(
            task_id=data["task_id"],
            agent_name=data["agent_name"],
            query=data["query"],
            dependencies=data.get("dependencies", []),
            status=data["status"],
            result=result,
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            start_time=start_time,
            end_time=end_time,
        )


@dataclass
class WorkflowCheckpoint:
    """Complete workflow state at a checkpoint"""

    checkpoint_id: str
    workflow_id: str
    tenant_id: str
    workflow_status: str  # WorkflowStatus.value
    current_phase: int  # Index into execution_order
    original_query: str
    execution_order: List[List[str]]  # Phases of task IDs
    metadata: Dict[str, Any]
    task_states: Dict[str, TaskCheckpoint]  # task_id -> TaskCheckpoint
    checkpoint_time: datetime
    checkpoint_status: CheckpointStatus
    parent_checkpoint_id: Optional[str] = None  # For forking/time-travel
    resume_count: int = 0  # Number of times resumed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for span storage"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
            "workflow_status": self.workflow_status,
            "current_phase": self.current_phase,
            "original_query": self.original_query,
            "execution_order": json.dumps(self.execution_order),
            "metadata": json.dumps(self.metadata),
            "task_states": json.dumps(
                {task_id: task.to_dict() for task_id, task in self.task_states.items()}
            ),
            "checkpoint_time": self.checkpoint_time.isoformat(),
            "checkpoint_status": self.checkpoint_status.value,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "resume_count": self.resume_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Deserialize from dictionary"""
        execution_order = data.get("execution_order", "[]")
        if isinstance(execution_order, str):
            execution_order = json.loads(execution_order)

        metadata = data.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        task_states_raw = data.get("task_states", "{}")
        if isinstance(task_states_raw, str):
            task_states_raw = json.loads(task_states_raw)

        task_states = {
            task_id: TaskCheckpoint.from_dict(task_data)
            for task_id, task_data in task_states_raw.items()
        }

        checkpoint_time = data.get("checkpoint_time")
        if isinstance(checkpoint_time, str):
            checkpoint_time = datetime.fromisoformat(checkpoint_time)
        elif checkpoint_time is None:
            checkpoint_time = datetime.now()

        checkpoint_status = data.get("checkpoint_status", "active")
        if isinstance(checkpoint_status, str):
            checkpoint_status = CheckpointStatus(checkpoint_status)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            tenant_id=data["tenant_id"],
            workflow_status=data["workflow_status"],
            current_phase=data["current_phase"],
            original_query=data["original_query"],
            execution_order=execution_order,
            metadata=metadata,
            task_states=task_states,
            checkpoint_time=checkpoint_time,
            checkpoint_status=checkpoint_status,
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            resume_count=data.get("resume_count", 0),
        )

    def get_completed_task_ids(self) -> Set[str]:
        """Get IDs of completed tasks"""
        return {
            task_id
            for task_id, task in self.task_states.items()
            if task.status == "completed"
        }

    def get_failed_task_ids(self) -> Set[str]:
        """Get IDs of failed tasks"""
        return {
            task_id
            for task_id, task in self.task_states.items()
            if task.status == "failed"
        }

    def get_pending_phases(self) -> List[List[str]]:
        """Get phases that still need execution"""
        return self.execution_order[self.current_phase :]


@dataclass
class CheckpointConfig:
    """Configuration for workflow checkpointing"""

    enabled: bool = True
    level: CheckpointLevel = CheckpointLevel.PHASE
    project_name: str = "workflow_checkpoints"
    # Retention settings
    retain_completed_hours: int = 24 * 7  # 7 days
    retain_failed_hours: int = 24 * 30  # 30 days

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "enabled": self.enabled,
            "level": self.level.value,
            "project_name": self.project_name,
            "retain_completed_hours": self.retain_completed_hours,
            "retain_failed_hours": self.retain_failed_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointConfig":
        """Deserialize from dictionary"""
        level = data.get("level", "phase")
        if isinstance(level, str):
            level = CheckpointLevel(level)

        return cls(
            enabled=data.get("enabled", True),
            level=level,
            project_name=data.get("project_name", "workflow_checkpoints"),
            retain_completed_hours=data.get("retain_completed_hours", 24 * 7),
            retain_failed_hours=data.get("retain_failed_hours", 24 * 30),
        )
