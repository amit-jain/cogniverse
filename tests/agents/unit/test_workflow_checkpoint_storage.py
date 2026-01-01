"""
Unit Tests for Workflow Checkpoint Storage

Tests the checkpoint types and storage functionality for durable execution.
"""

import json
from datetime import datetime

from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig,
    CheckpointLevel,
    CheckpointStatus,
    TaskCheckpoint,
    WorkflowCheckpoint,
)


class TestTaskCheckpoint:
    """Tests for TaskCheckpoint data class"""

    def test_create_task_checkpoint(self):
        """Test creating a TaskCheckpoint"""
        task = TaskCheckpoint(
            task_id="task_1",
            agent_name="search_agent",
            query="Find videos about cats",
            dependencies=["task_0"],
            status="completed",
            result={"items": ["result1", "result2"]},
            error=None,
            retry_count=0,
            start_time=datetime(2025, 1, 1, 10, 0, 0),
            end_time=datetime(2025, 1, 1, 10, 1, 0),
        )

        assert task.task_id == "task_1"
        assert task.agent_name == "search_agent"
        assert task.status == "completed"
        assert len(task.dependencies) == 1
        assert task.result == {"items": ["result1", "result2"]}

    def test_task_checkpoint_to_dict(self):
        """Test serializing TaskCheckpoint to dictionary"""
        task = TaskCheckpoint(
            task_id="task_1",
            agent_name="search_agent",
            query="Find videos",
            dependencies=["dep1"],
            status="running",
            result=None,
            error=None,
            retry_count=1,
            start_time=datetime(2025, 1, 1, 10, 0, 0),
            end_time=None,
        )

        data = task.to_dict()

        assert data["task_id"] == "task_1"
        assert data["agent_name"] == "search_agent"
        assert data["status"] == "running"
        assert data["retry_count"] == 1
        assert data["start_time"] == "2025-01-01T10:00:00"
        assert data["end_time"] is None
        assert data["result"] is None

    def test_task_checkpoint_from_dict(self):
        """Test deserializing TaskCheckpoint from dictionary"""
        data = {
            "task_id": "task_2",
            "agent_name": "summarizer_agent",
            "query": "Summarize results",
            "dependencies": ["task_1"],
            "status": "completed",
            "result": '{"summary": "Test summary"}',
            "error": None,
            "retry_count": 0,
            "start_time": "2025-01-01T11:00:00",
            "end_time": "2025-01-01T11:05:00",
        }

        task = TaskCheckpoint.from_dict(data)

        assert task.task_id == "task_2"
        assert task.agent_name == "summarizer_agent"
        assert task.result == {"summary": "Test summary"}
        assert task.start_time == datetime(2025, 1, 1, 11, 0, 0)
        assert task.end_time == datetime(2025, 1, 1, 11, 5, 0)

    def test_task_checkpoint_roundtrip(self):
        """Test serialization roundtrip preserves data"""
        original = TaskCheckpoint(
            task_id="task_3",
            agent_name="report_agent",
            query="Generate report",
            dependencies=["task_1", "task_2"],
            status="failed",
            result=None,
            error="Connection timeout",
            retry_count=2,
            start_time=datetime(2025, 1, 1, 12, 0, 0),
            end_time=datetime(2025, 1, 1, 12, 0, 30),
        )

        data = original.to_dict()
        restored = TaskCheckpoint.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.agent_name == original.agent_name
        assert restored.status == original.status
        assert restored.error == original.error
        assert restored.retry_count == original.retry_count


class TestWorkflowCheckpoint:
    """Tests for WorkflowCheckpoint data class"""

    def test_create_workflow_checkpoint(self):
        """Test creating a WorkflowCheckpoint"""
        task1 = TaskCheckpoint(
            task_id="task_1",
            agent_name="search_agent",
            query="Search",
            dependencies=[],
            status="completed",
        )
        task2 = TaskCheckpoint(
            task_id="task_2",
            agent_name="summarizer_agent",
            query="Summarize",
            dependencies=["task_1"],
            status="waiting",
        )

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_abc123",
            workflow_id="workflow_xyz",
            tenant_id="test_tenant",
            workflow_status="running",
            current_phase=1,
            original_query="Complex query",
            execution_order=[["task_1"], ["task_2"]],
            metadata={"context": "test"},
            task_states={"task_1": task1, "task_2": task2},
            checkpoint_time=datetime(2025, 1, 1, 10, 0, 0),
            checkpoint_status=CheckpointStatus.ACTIVE,
            parent_checkpoint_id=None,
            resume_count=0,
        )

        assert checkpoint.checkpoint_id == "ckpt_abc123"
        assert checkpoint.workflow_id == "workflow_xyz"
        assert checkpoint.current_phase == 1
        assert len(checkpoint.task_states) == 2
        assert checkpoint.checkpoint_status == CheckpointStatus.ACTIVE

    def test_workflow_checkpoint_to_dict(self):
        """Test serializing WorkflowCheckpoint to dictionary"""
        task = TaskCheckpoint(
            task_id="task_1",
            agent_name="search_agent",
            query="Search",
            dependencies=[],
            status="completed",
        )

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_123",
            workflow_id="wf_456",
            tenant_id="tenant_a",
            workflow_status="running",
            current_phase=0,
            original_query="Test query",
            execution_order=[["task_1"]],
            metadata={"key": "value"},
            task_states={"task_1": task},
            checkpoint_time=datetime(2025, 1, 1, 10, 0, 0),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        data = checkpoint.to_dict()

        assert data["checkpoint_id"] == "ckpt_123"
        assert data["workflow_id"] == "wf_456"
        assert data["tenant_id"] == "tenant_a"
        assert json.loads(data["execution_order"]) == [["task_1"]]
        assert json.loads(data["metadata"]) == {"key": "value"}
        assert "task_1" in json.loads(data["task_states"])

    def test_workflow_checkpoint_from_dict(self):
        """Test deserializing WorkflowCheckpoint from dictionary"""
        task_data = {
            "task_id": "task_1",
            "agent_name": "search_agent",
            "query": "Search",
            "dependencies": [],
            "status": "completed",
        }

        data = {
            "checkpoint_id": "ckpt_789",
            "workflow_id": "wf_abc",
            "tenant_id": "tenant_b",
            "workflow_status": "completed",
            "current_phase": 2,
            "original_query": "Another query",
            "execution_order": json.dumps([["task_1"]]),
            "metadata": json.dumps({"info": "test"}),
            "task_states": json.dumps({"task_1": task_data}),
            "checkpoint_time": "2025-01-01T15:00:00",
            "checkpoint_status": "completed",
            "parent_checkpoint_id": None,
            "resume_count": 1,
        }

        checkpoint = WorkflowCheckpoint.from_dict(data)

        assert checkpoint.checkpoint_id == "ckpt_789"
        assert checkpoint.workflow_status == "completed"
        assert checkpoint.current_phase == 2
        assert checkpoint.checkpoint_status == CheckpointStatus.COMPLETED
        assert checkpoint.resume_count == 1
        assert "task_1" in checkpoint.task_states

    def test_get_completed_task_ids(self):
        """Test getting completed task IDs from checkpoint"""
        tasks = {
            "task_1": TaskCheckpoint(
                task_id="task_1",
                agent_name="agent1",
                query="q1",
                dependencies=[],
                status="completed",
            ),
            "task_2": TaskCheckpoint(
                task_id="task_2",
                agent_name="agent2",
                query="q2",
                dependencies=[],
                status="running",
            ),
            "task_3": TaskCheckpoint(
                task_id="task_3",
                agent_name="agent3",
                query="q3",
                dependencies=[],
                status="completed",
            ),
        }

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_1",
            workflow_id="wf_1",
            tenant_id="t",
            workflow_status="running",
            current_phase=1,
            original_query="q",
            execution_order=[],
            metadata={},
            task_states=tasks,
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        completed = checkpoint.get_completed_task_ids()

        assert completed == {"task_1", "task_3"}

    def test_get_failed_task_ids(self):
        """Test getting failed task IDs from checkpoint"""
        tasks = {
            "task_1": TaskCheckpoint(
                task_id="task_1",
                agent_name="agent1",
                query="q1",
                dependencies=[],
                status="completed",
            ),
            "task_2": TaskCheckpoint(
                task_id="task_2",
                agent_name="agent2",
                query="q2",
                dependencies=[],
                status="failed",
                error="Connection error",
            ),
        }

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_1",
            workflow_id="wf_1",
            tenant_id="t",
            workflow_status="running",
            current_phase=1,
            original_query="q",
            execution_order=[],
            metadata={},
            task_states=tasks,
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        failed = checkpoint.get_failed_task_ids()

        assert failed == {"task_2"}

    def test_get_pending_phases(self):
        """Test getting pending phases from checkpoint"""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_1",
            workflow_id="wf_1",
            tenant_id="t",
            workflow_status="running",
            current_phase=1,
            original_query="q",
            execution_order=[["task_1"], ["task_2"], ["task_3"]],
            metadata={},
            task_states={},
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        pending = checkpoint.get_pending_phases()

        assert pending == [["task_2"], ["task_3"]]


class TestCheckpointConfig:
    """Tests for CheckpointConfig data class"""

    def test_default_config(self):
        """Test default CheckpointConfig values"""
        config = CheckpointConfig()

        assert config.enabled is True
        assert config.level == CheckpointLevel.PHASE
        assert config.project_name == "workflow_checkpoints"
        assert config.retain_completed_hours == 24 * 7
        assert config.retain_failed_hours == 24 * 30

    def test_custom_config(self):
        """Test custom CheckpointConfig values"""
        config = CheckpointConfig(
            enabled=False,
            level=CheckpointLevel.TASK,
            project_name="custom_checkpoints",
            retain_completed_hours=48,
            retain_failed_hours=72,
        )

        assert config.enabled is False
        assert config.level == CheckpointLevel.TASK
        assert config.project_name == "custom_checkpoints"

    def test_config_to_dict(self):
        """Test serializing CheckpointConfig to dictionary"""
        config = CheckpointConfig(level=CheckpointLevel.PHASE_AND_TASK)

        data = config.to_dict()

        assert data["enabled"] is True
        assert data["level"] == "both"
        assert data["project_name"] == "workflow_checkpoints"

    def test_config_from_dict(self):
        """Test deserializing CheckpointConfig from dictionary"""
        data = {
            "enabled": True,
            "level": "task",
            "project_name": "test_checkpoints",
            "retain_completed_hours": 100,
            "retain_failed_hours": 200,
        }

        config = CheckpointConfig.from_dict(data)

        assert config.enabled is True
        assert config.level == CheckpointLevel.TASK
        assert config.project_name == "test_checkpoints"
        assert config.retain_completed_hours == 100


class TestCheckpointEnums:
    """Tests for checkpoint enums"""

    def test_checkpoint_level_values(self):
        """Test CheckpointLevel enum values"""
        assert CheckpointLevel.PHASE.value == "phase"
        assert CheckpointLevel.TASK.value == "task"
        assert CheckpointLevel.PHASE_AND_TASK.value == "both"

    def test_checkpoint_status_values(self):
        """Test CheckpointStatus enum values"""
        assert CheckpointStatus.ACTIVE.value == "active"
        assert CheckpointStatus.SUPERSEDED.value == "superseded"
        assert CheckpointStatus.FAILED.value == "failed"
        assert CheckpointStatus.COMPLETED.value == "completed"
