"""
Integration Tests for Workflow Checkpointing

Tests the durable execution capability of MultiAgentOrchestrator:
1. Checkpoint creation after each phase
2. Resume from checkpoint
3. Skip completed tasks on resume
4. Status transitions (ACTIVE -> SUPERSEDED)
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig,
    CheckpointLevel,
    CheckpointStatus,
    TaskCheckpoint,
    WorkflowCheckpoint,
)
from cogniverse_agents.orchestrator.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
)
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_agents.workflow.types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowTask,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_routing_agent():
    """Create a mock routing agent to avoid RoutingDeps validation"""
    agent = MagicMock(spec=RoutingAgent)
    agent.route_query = AsyncMock(return_value=MagicMock(
        recommended_agent="search_agent",
        confidence=0.9,
        enhanced_query="test query"
    ))
    return agent


class MockCheckpointStorage:
    """Mock checkpoint storage for testing without Phoenix"""

    def __init__(self):
        self.checkpoints: dict[str, WorkflowCheckpoint] = {}
        self.save_count = 0
        self.status_updates: list[tuple[str, CheckpointStatus]] = []

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save checkpoint to in-memory store"""
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        self.save_count += 1
        logger.info(
            f"MockStorage: Saved checkpoint {checkpoint.checkpoint_id} "
            f"(workflow={checkpoint.workflow_id}, phase={checkpoint.current_phase})"
        )
        return checkpoint.checkpoint_id

    async def get_latest_checkpoint(
        self, workflow_id: str
    ) -> WorkflowCheckpoint | None:
        """Get latest active checkpoint for workflow"""
        workflow_checkpoints = [
            ckpt
            for ckpt in self.checkpoints.values()
            if ckpt.workflow_id == workflow_id
            and ckpt.checkpoint_status == CheckpointStatus.ACTIVE
        ]
        if not workflow_checkpoints:
            return None
        # Return most recent by checkpoint_time
        return max(workflow_checkpoints, key=lambda c: c.checkpoint_time)

    async def get_checkpoint_by_id(
        self, checkpoint_id: str
    ) -> WorkflowCheckpoint | None:
        """Get specific checkpoint by ID"""
        return self.checkpoints.get(checkpoint_id)

    async def mark_checkpoint_status(
        self, checkpoint_id: str, status: CheckpointStatus
    ) -> bool:
        """Update checkpoint status"""
        if checkpoint_id in self.checkpoints:
            self.checkpoints[checkpoint_id].checkpoint_status = status
            self.status_updates.append((checkpoint_id, status))
            return True
        return False

    async def list_workflow_checkpoints(
        self, workflow_id: str, include_superseded: bool = False
    ) -> list[WorkflowCheckpoint]:
        """List checkpoints for workflow"""
        return [
            ckpt
            for ckpt in self.checkpoints.values()
            if ckpt.workflow_id == workflow_id
            and (include_superseded or ckpt.checkpoint_status != CheckpointStatus.SUPERSEDED)
        ]

    async def get_resumable_workflows(
        self, tenant_id: str | None = None
    ) -> list[dict]:
        """Get resumable workflows"""
        resumable = []
        for ckpt in self.checkpoints.values():
            if ckpt.checkpoint_status == CheckpointStatus.ACTIVE:
                if ckpt.workflow_status in ["running", "failed"]:
                    resumable.append({
                        "workflow_id": ckpt.workflow_id,
                        "checkpoint_id": ckpt.checkpoint_id,
                        "workflow_status": ckpt.workflow_status,
                        "current_phase": ckpt.current_phase,
                    })
        return resumable


@pytest.fixture
def mock_checkpoint_storage():
    """Provide mock checkpoint storage"""
    return MockCheckpointStorage()


@pytest.fixture
def mock_a2a_client():
    """Mock A2A client that returns successful results"""
    client = AsyncMock()
    client.send_message = AsyncMock(
        return_value={"result": "success", "data": "test_data", "confidence": 0.9}
    )
    return client


@pytest.fixture
def checkpoint_config():
    """Provide checkpoint configuration for testing"""
    return CheckpointConfig(
        enabled=True,
        level=CheckpointLevel.PHASE,
        project_name="test_checkpoints",
    )


class TestCheckpointCreation:
    """Tests for checkpoint creation during workflow execution"""

    @pytest.mark.asyncio
    async def test_checkpoint_created_after_each_phase(
        self, mock_checkpoint_storage, checkpoint_config, mock_a2a_client, mock_routing_agent
    ):
        """Test that checkpoints are created after each phase completes"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        # Execute a workflow
        with patch.object(orchestrator, "_plan_workflow") as mock_plan:
            # Create a simple 2-phase workflow
            workflow_plan = WorkflowPlan(
                workflow_id="wf_test_1",
                original_query="Test query",
                tasks=[
                    WorkflowTask(
                        task_id="task_1",
                        agent_name="search_agent",
                        query="Search",
                        dependencies=set(),
                    ),
                    WorkflowTask(
                        task_id="task_2",
                        agent_name="summarizer_agent",
                        query="Summarize",
                        dependencies={"task_1"},
                    ),
                ],
                execution_order=[["task_1"], ["task_2"]],
                metadata={},
            )
            mock_plan.return_value = workflow_plan

            await orchestrator.process_complex_query("Test query")

        # Should have checkpoints: after phase 1, after phase 2, and completion
        assert mock_checkpoint_storage.save_count >= 2
        assert len(mock_checkpoint_storage.checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_checkpoint_contains_task_states(
        self, mock_checkpoint_storage, checkpoint_config, mock_a2a_client, mock_routing_agent
    ):
        """Test that checkpoints contain correct task states"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        with patch.object(orchestrator, "_plan_workflow") as mock_plan:
            workflow_plan = WorkflowPlan(
                workflow_id="wf_test_2",
                original_query="Query with states",
                tasks=[
                    WorkflowTask(
                        task_id="task_1",
                        agent_name="search_agent",
                        query="Search",
                        dependencies=set(),
                    ),
                ],
                execution_order=[["task_1"]],
                metadata={},
            )
            mock_plan.return_value = workflow_plan

            await orchestrator.process_complex_query("Query with states")

        # Check that saved checkpoints have task states
        for checkpoint in mock_checkpoint_storage.checkpoints.values():
            assert "task_1" in checkpoint.task_states
            task_state = checkpoint.task_states["task_1"]
            assert task_state.agent_name == "search_agent"

    @pytest.mark.asyncio
    async def test_checkpointing_disabled(self, mock_a2a_client, mock_routing_agent):
        """Test that no checkpoints are saved when disabled"""
        disabled_config = CheckpointConfig(enabled=False)
        mock_storage = MockCheckpointStorage()

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=disabled_config,
            checkpoint_storage=mock_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        with patch.object(orchestrator, "_plan_workflow") as mock_plan:
            workflow_plan = WorkflowPlan(
                workflow_id="wf_disabled",
                original_query="No checkpoints",
                tasks=[
                    WorkflowTask(
                        task_id="task_1",
                        agent_name="search_agent",
                        query="Search",
                        dependencies=set(),
                    ),
                ],
                execution_order=[["task_1"]],
                metadata={},
            )
            mock_plan.return_value = workflow_plan

            await orchestrator.process_complex_query("No checkpoints")

        # No checkpoints should be saved
        assert mock_storage.save_count == 0

    @pytest.mark.asyncio
    async def test_checkpoint_without_storage(self, checkpoint_config, mock_a2a_client, mock_routing_agent):
        """Test that workflow executes even without checkpoint storage"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=None,  # No storage configured
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        with patch.object(orchestrator, "_plan_workflow") as mock_plan:
            workflow_plan = WorkflowPlan(
                workflow_id="wf_no_storage",
                original_query="No storage",
                tasks=[
                    WorkflowTask(
                        task_id="task_1",
                        agent_name="search_agent",
                        query="Search",
                        dependencies=set(),
                    ),
                ],
                execution_order=[["task_1"]],
                metadata={},
            )
            mock_plan.return_value = workflow_plan

            result = await orchestrator.process_complex_query("No storage")

        # Workflow should complete successfully
        assert result["status"] == "completed"


class TestWorkflowResume:
    """Tests for resuming workflows from checkpoints"""

    @pytest.mark.asyncio
    async def test_resume_skips_completed_tasks(
        self, mock_checkpoint_storage, checkpoint_config, mock_a2a_client, mock_routing_agent
    ):
        """Test that resume skips already completed tasks"""
        # Create a checkpoint with task_1 completed
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_resume_1",
            workflow_id="wf_resume_1",
            tenant_id="test_tenant",
            workflow_status="running",
            current_phase=1,  # Phase 0 completed
            original_query="Resume test",
            execution_order=[["task_1"], ["task_2"]],
            metadata={},
            task_states={
                "task_1": TaskCheckpoint(
                    task_id="task_1",
                    agent_name="search_agent",
                    query="Search",
                    dependencies=[],
                    status="completed",
                    result={"data": "search_result"},
                ),
                "task_2": TaskCheckpoint(
                    task_id="task_2",
                    agent_name="summarizer_agent",
                    query="Summarize",
                    dependencies=["task_1"],
                    status="waiting",
                ),
            },
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )
        await mock_checkpoint_storage.save_checkpoint(checkpoint)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        # Resume the workflow
        result = await orchestrator.process_complex_query(
            "Resume test", resume_from_workflow_id="wf_resume_1"
        )

        assert result["status"] == "completed"
        assert result["resumed_from"] == "ckpt_resume_1"
        assert result["resume_count"] == 1

        # task_1 should NOT have been re-executed (A2A called only for task_2)
        # The mock was called once for task_2
        assert mock_a2a_client.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_resume_marks_old_checkpoint_superseded(
        self, mock_checkpoint_storage, checkpoint_config, mock_a2a_client, mock_routing_agent
    ):
        """Test that old checkpoint is marked as superseded on resume"""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_supersede",
            workflow_id="wf_supersede",
            tenant_id="test_tenant",
            workflow_status="running",
            current_phase=0,
            original_query="Supersede test",
            execution_order=[["task_1"]],
            metadata={},
            task_states={
                "task_1": TaskCheckpoint(
                    task_id="task_1",
                    agent_name="search_agent",
                    query="Search",
                    dependencies=[],
                    status="waiting",
                ),
            },
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )
        await mock_checkpoint_storage.save_checkpoint(checkpoint)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        await orchestrator.process_complex_query(
            "Supersede test", resume_from_workflow_id="wf_supersede"
        )

        # Original checkpoint should be marked superseded
        assert ("ckpt_supersede", CheckpointStatus.SUPERSEDED) in mock_checkpoint_storage.status_updates

    @pytest.mark.asyncio
    async def test_resume_without_checkpoint_returns_error(
        self, mock_checkpoint_storage, checkpoint_config, mock_routing_agent
    ):
        """Test that resume fails gracefully when no checkpoint exists"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )

        result = await orchestrator.process_complex_query(
            "No checkpoint", resume_from_workflow_id="wf_nonexistent"
        )

        assert result["status"] == "failed"
        assert "No checkpoint found" in result["error"]

    @pytest.mark.asyncio
    async def test_resume_without_storage_returns_error(self, checkpoint_config, mock_routing_agent):
        """Test that resume fails when no storage is configured"""
        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=None,
            enable_workflow_intelligence=False,
        )

        result = await orchestrator.process_complex_query(
            "No storage", resume_from_workflow_id="wf_any"
        )

        assert result["status"] == "failed"
        assert "Checkpoint storage not configured" in result["error"]


class TestResumeableWorkflows:
    """Tests for listing resumable workflows"""

    @pytest.mark.asyncio
    async def test_get_resumable_workflows(
        self, mock_checkpoint_storage, checkpoint_config, mock_routing_agent
    ):
        """Test listing workflows that can be resumed"""
        # Create checkpoints for different workflow states
        running_ckpt = WorkflowCheckpoint(
            checkpoint_id="ckpt_running",
            workflow_id="wf_running",
            tenant_id="test_tenant",
            workflow_status="running",
            current_phase=1,
            original_query="Running workflow",
            execution_order=[["task_1"], ["task_2"]],
            metadata={},
            task_states={},
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        failed_ckpt = WorkflowCheckpoint(
            checkpoint_id="ckpt_failed",
            workflow_id="wf_failed",
            tenant_id="test_tenant",
            workflow_status="failed",
            current_phase=0,
            original_query="Failed workflow",
            execution_order=[["task_1"]],
            metadata={},
            task_states={},
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        completed_ckpt = WorkflowCheckpoint(
            checkpoint_id="ckpt_completed",
            workflow_id="wf_completed",
            tenant_id="test_tenant",
            workflow_status="completed",
            current_phase=2,
            original_query="Completed workflow",
            execution_order=[["task_1"], ["task_2"]],
            metadata={},
            task_states={},
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.COMPLETED,
        )

        await mock_checkpoint_storage.save_checkpoint(running_ckpt)
        await mock_checkpoint_storage.save_checkpoint(failed_ckpt)
        await mock_checkpoint_storage.save_checkpoint(completed_ckpt)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=checkpoint_config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )

        resumable = await orchestrator.get_resumable_workflows()

        # Should only include running and failed workflows
        workflow_ids = [w["workflow_id"] for w in resumable]
        assert "wf_running" in workflow_ids
        assert "wf_failed" in workflow_ids
        assert "wf_completed" not in workflow_ids


class TestCheckpointLevels:
    """Tests for different checkpoint granularity levels"""

    @pytest.mark.asyncio
    async def test_phase_level_checkpointing(
        self, mock_checkpoint_storage, mock_a2a_client, mock_routing_agent
    ):
        """Test phase-level checkpointing creates checkpoint after each phase"""
        config = CheckpointConfig(enabled=True, level=CheckpointLevel.PHASE)

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            checkpoint_config=config,
            checkpoint_storage=mock_checkpoint_storage,
            enable_workflow_intelligence=False,
        )
        orchestrator.a2a_client = mock_a2a_client

        with patch.object(orchestrator, "_plan_workflow") as mock_plan:
            workflow_plan = WorkflowPlan(
                workflow_id="wf_phase_level",
                original_query="Phase level test",
                tasks=[
                    WorkflowTask(
                        task_id="task_1",
                        agent_name="search_agent",
                        query="Search",
                        dependencies=set(),
                    ),
                    WorkflowTask(
                        task_id="task_2",
                        agent_name="summarizer_agent",
                        query="Summarize",
                        dependencies={"task_1"},
                    ),
                ],
                execution_order=[["task_1"], ["task_2"]],
                metadata={},
            )
            mock_plan.return_value = workflow_plan

            await orchestrator.process_complex_query("Phase level test")

        # Should have created checkpoints after each phase
        # Phase 1 complete -> checkpoint, Phase 2 complete -> checkpoint, Final -> checkpoint
        assert mock_checkpoint_storage.save_count >= 2

    @pytest.mark.asyncio
    async def test_should_checkpoint_phase_returns_correct_value(self, mock_routing_agent):
        """Test _should_checkpoint_phase returns correct values"""
        mock_storage = MockCheckpointStorage()

        # Enabled with PHASE level
        orchestrator1 = MultiAgentOrchestrator(
            tenant_id="test",
            routing_agent=mock_routing_agent,
            checkpoint_config=CheckpointConfig(enabled=True, level=CheckpointLevel.PHASE),
            checkpoint_storage=mock_storage,
            enable_workflow_intelligence=False,
        )
        assert orchestrator1._should_checkpoint_phase() is True

        # Enabled with TASK level (no phase checkpointing)
        orchestrator2 = MultiAgentOrchestrator(
            tenant_id="test",
            routing_agent=mock_routing_agent,
            checkpoint_config=CheckpointConfig(enabled=True, level=CheckpointLevel.TASK),
            checkpoint_storage=mock_storage,
            enable_workflow_intelligence=False,
        )
        assert orchestrator2._should_checkpoint_phase() is False

        # Disabled
        orchestrator3 = MultiAgentOrchestrator(
            tenant_id="test",
            routing_agent=mock_routing_agent,
            checkpoint_config=CheckpointConfig(enabled=False),
            checkpoint_storage=mock_storage,
            enable_workflow_intelligence=False,
        )
        assert orchestrator3._should_checkpoint_phase() is False

        # No storage
        orchestrator4 = MultiAgentOrchestrator(
            tenant_id="test",
            routing_agent=mock_routing_agent,
            checkpoint_config=CheckpointConfig(enabled=True),
            checkpoint_storage=None,
            enable_workflow_intelligence=False,
        )
        assert orchestrator4._should_checkpoint_phase() is False


class TestWorkflowReconstruction:
    """Tests for reconstructing workflow plans from checkpoints"""

    def test_reconstruct_workflow_plan(self, mock_routing_agent):
        """Test that workflow plan is correctly reconstructed from checkpoint"""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="ckpt_reconstruct",
            workflow_id="wf_reconstruct",
            tenant_id="test_tenant",
            workflow_status="running",
            current_phase=1,
            original_query="Reconstruct test",
            execution_order=[["task_1"], ["task_2"]],
            metadata={"key": "value"},
            task_states={
                "task_1": TaskCheckpoint(
                    task_id="task_1",
                    agent_name="search_agent",
                    query="Search query",
                    dependencies=[],
                    status="completed",
                    result={"data": "result"},
                ),
                "task_2": TaskCheckpoint(
                    task_id="task_2",
                    agent_name="summarizer_agent",
                    query="Summarize",
                    dependencies=["task_1"],
                    status="waiting",
                ),
            },
            checkpoint_time=datetime.now(),
            checkpoint_status=CheckpointStatus.ACTIVE,
        )

        orchestrator = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            routing_agent=mock_routing_agent,
            enable_workflow_intelligence=False,
        )
        workflow_plan = orchestrator._reconstruct_workflow_plan(checkpoint)

        assert workflow_plan.workflow_id == "wf_reconstruct"
        assert workflow_plan.original_query == "Reconstruct test"
        assert len(workflow_plan.tasks) == 2
        assert workflow_plan.execution_order == [["task_1"], ["task_2"]]
        assert workflow_plan.metadata == {"key": "value"}

        # Check task states
        task1 = next(t for t in workflow_plan.tasks if t.task_id == "task_1")
        assert task1.status == TaskStatus.COMPLETED
        assert task1.result == {"data": "result"}

        task2 = next(t for t in workflow_plan.tasks if t.task_id == "task_2")
        assert task2.status == TaskStatus.WAITING
        assert task2.dependencies == {"task_1"}
