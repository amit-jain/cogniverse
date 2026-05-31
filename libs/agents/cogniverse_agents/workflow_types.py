"""
Shared workflow data types for multi-agent orchestration system

This module contains common data classes and enums used by both the
multi-agent orchestrator and workflow intelligence components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TaskStatus(Enum):
    """Individual task status"""

    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class WorkflowTask:
    """Individual task within a workflow"""

    task_id: str
    agent_name: str
    query: str
    dependencies: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timeout_seconds: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class WorkflowPlan:
    """Complete workflow execution plan"""

    workflow_id: str
    original_query: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    execution_order: List[List[str]] = field(
        default_factory=list
    )  # Parallel execution groups
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    final_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to execute (dependencies met)"""
        completed_task_ids = {
            task.task_id for task in self.tasks if task.status == TaskStatus.COMPLETED
        }

        ready_tasks = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and all(
                dep_id in completed_task_ids for dep_id in task.dependencies
            ):
                ready_tasks.append(task)

        return ready_tasks

    def get_task_by_id(self, task_id: str) -> Optional[WorkflowTask]:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
