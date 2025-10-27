"""
Decision Orchestrator

Orchestrates workflows with approval checkpoints using state machine.
Integrates HumanApprovalAgent with workflow execution.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.interfaces import (
    ApprovalBatch,
    ReviewDecision,
)
from cogniverse_agents.workflow.state_machine import (
    WorkflowState,
    WorkflowStateMachine,
)

logger = logging.getLogger(__name__)


class DecisionOrchestrator:
    """
    Orchestrate workflows with approval checkpoints

    Combines:
    - WorkflowStateMachine: State management
    - HumanApprovalAgent: Approval logic
    - Custom workflows: Domain-specific execution

    Example usage:
        orchestrator = DecisionOrchestrator(
            approval_agent=approval_agent,
            workflow_id="synthetic_generation_001"
        )

        # Register workflow steps
        orchestrator.register_step(
            name="generate",
            executor=lambda ctx: generate_synthetic_data(ctx),
            requires_approval=True
        )

        orchestrator.register_step(
            name="optimize",
            executor=lambda ctx: run_optimization(ctx),
            requires_approval=False
        )

        # Execute workflow
        result = await orchestrator.execute(context={"tenant_id": "acme"})
    """

    def __init__(
        self,
        approval_agent: HumanApprovalAgent,
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize orchestrator

        Args:
            approval_agent: HumanApprovalAgent for approval logic
            workflow_id: Unique workflow identifier
            initial_context: Initial workflow context
        """
        self.approval_agent = approval_agent
        self.workflow_id = workflow_id
        self.state_machine = WorkflowStateMachine(
            initial_state=WorkflowState.INITIALIZING,
            context=initial_context or {},
        )

        self.steps: List[Dict[str, Any]] = []
        self.current_step_index = 0

        # Setup standard state transitions
        self._setup_transitions()

        logger.info(f"Initialized DecisionOrchestrator (workflow: {workflow_id})")

    def _setup_transitions(self) -> None:
        """Setup standard workflow state transitions"""

        # Initialize -> Running
        self.state_machine.register_transition(
            from_state=WorkflowState.INITIALIZING,
            to_state=WorkflowState.RUNNING,
            condition=lambda ctx: ctx.get("steps_registered", False),
            description="All steps registered, ready to execute",
        )

        # Running -> Awaiting Approval
        self.state_machine.register_transition(
            from_state=WorkflowState.RUNNING,
            to_state=WorkflowState.AWAITING_APPROVAL,
            condition=lambda ctx: (
                ctx.get("current_step_requires_approval", False)
                and ctx.get("pending_review_count", 0) > 0
            ),
            description="Step output requires human review",
        )

        # Awaiting Approval -> Approved
        self.state_machine.register_transition(
            from_state=WorkflowState.AWAITING_APPROVAL,
            to_state=WorkflowState.APPROVED,
            condition=lambda ctx: ctx.get("pending_review_count", 0) == 0,
            description="All items approved",
        )

        # Approved -> Running (continue to next step)
        self.state_machine.register_transition(
            from_state=WorkflowState.APPROVED,
            to_state=WorkflowState.RUNNING,
            condition=lambda ctx: not ctx.get("is_last_step", False),
            description="Continue to next step",
        )

        # Approved -> Completed (workflow done)
        self.state_machine.register_transition(
            from_state=WorkflowState.APPROVED,
            to_state=WorkflowState.COMPLETED,
            condition=lambda ctx: ctx.get("is_last_step", False),
            description="All steps completed",
        )

        # Running -> Completed (no approval needed and last step)
        self.state_machine.register_transition(
            from_state=WorkflowState.RUNNING,
            to_state=WorkflowState.COMPLETED,
            condition=lambda ctx: (
                not ctx.get("current_step_requires_approval", False)
                and ctx.get("is_last_step", False)
            ),
            description="Workflow completed without approval",
        )

        # Awaiting Approval -> Rejected (if any rejections)
        self.state_machine.register_transition(
            from_state=WorkflowState.AWAITING_APPROVAL,
            to_state=WorkflowState.REJECTED,
            condition=lambda ctx: ctx.get("rejection_count", 0) > 0,
            description="Items rejected by human",
        )

        # Rejected -> Regenerating
        self.state_machine.register_transition(
            from_state=WorkflowState.REJECTED,
            to_state=WorkflowState.REGENERATING,
            condition=lambda ctx: ctx.get("regenerate_enabled", True),
            description="Regenerating rejected items",
        )

        # Regenerating -> Running (retry with regenerated items)
        self.state_machine.register_transition(
            from_state=WorkflowState.REGENERATING,
            to_state=WorkflowState.RUNNING,
            condition=lambda ctx: ctx.get("regeneration_complete", False),
            description="Regeneration complete, continuing workflow",
        )

    def register_step(
        self,
        name: str,
        executor: Callable[[Dict[str, Any]], Any],
        requires_approval: bool = False,
    ) -> None:
        """
        Register a workflow step

        Args:
            name: Step name
            executor: Function to execute the step (takes context, returns result)
            requires_approval: Whether step output requires human approval
        """
        self.steps.append(
            {
                "name": name,
                "executor": executor,
                "requires_approval": requires_approval,
            }
        )

        logger.info(f"Registered step '{name}' (approval: {requires_approval})")

        # Update context
        self.state_machine.context["steps_registered"] = True
        self.state_machine.context["total_steps"] = len(self.steps)

    async def execute(
        self, context_updates: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow with approval checkpoints

        Args:
            context_updates: Updates to workflow context

        Returns:
            Final workflow context with results
        """
        if context_updates:
            self.state_machine.context.update(context_updates)

        # Transition to running
        self.state_machine.transition()

        while not self.state_machine.is_terminal():
            current_state = self.state_machine.current_state

            if current_state == WorkflowState.RUNNING:
                await self._execute_current_step()

            elif current_state == WorkflowState.AWAITING_APPROVAL:
                logger.info(
                    f"Workflow {self.workflow_id} paused for approval "
                    f"({self.state_machine.context.get('pending_review_count', 0)} items pending)"
                )
                # Workflow pauses here - resume with apply_approvals()
                break

            elif current_state == WorkflowState.REGENERATING:
                await self._handle_regeneration()

            # Try to transition
            self.state_machine.transition()

        return self.state_machine.context

    async def _execute_current_step(self) -> None:
        """Execute the current workflow step"""
        if self.current_step_index >= len(self.steps):
            logger.error("No more steps to execute")
            self.state_machine.force_transition(WorkflowState.FAILED, "No more steps")
            return

        step = self.steps[self.current_step_index]
        logger.info(
            f"Executing step {self.current_step_index + 1}/{len(self.steps)}: "
            f"{step['name']}"
        )

        try:
            # Execute step
            result = step["executor"](self.state_machine.context)

            # Update context
            self.state_machine.context[f"step_{step['name']}_result"] = result
            self.state_machine.context["current_step_name"] = step["name"]
            self.state_machine.context["current_step_requires_approval"] = step[
                "requires_approval"
            ]
            self.state_machine.context["is_last_step"] = (
                self.current_step_index == len(self.steps) - 1
            )

            # If approval required, process with approval agent
            if step["requires_approval"] and isinstance(result, list):
                batch = await self.approval_agent.process_batch(
                    items=result,
                    batch_id=f"{self.workflow_id}_step_{self.current_step_index}",
                    context={
                        "workflow_id": self.workflow_id,
                        "step_name": step["name"],
                        "step_index": self.current_step_index,
                    },
                )

                self.state_machine.context["current_batch"] = batch
                self.state_machine.context["pending_review_count"] = len(
                    batch.pending_review
                )
                logger.info(
                    f"Step '{step['name']}' generated {len(batch.items)} items: "
                    f"{len(batch.auto_approved)} auto-approved, "
                    f"{len(batch.pending_review)} pending review"
                )
            else:
                self.state_machine.context["pending_review_count"] = 0

            # Move to next step if no approval needed
            if not step["requires_approval"]:
                self.current_step_index += 1

        except Exception as e:
            logger.error(f"Step '{step['name']}' failed: {e}")
            self.state_machine.force_transition(WorkflowState.FAILED, str(e))

    async def apply_approvals(self, decisions: List[ReviewDecision]) -> None:
        """
        Apply approval decisions and resume workflow

        Args:
            decisions: List of human decisions
        """
        batch: ApprovalBatch = self.state_machine.context.get("current_batch")
        if not batch:
            raise ValueError("No batch awaiting approval")

        logger.info(
            f"Applying {len(decisions)} approval decisions to workflow {self.workflow_id}"
        )

        # Apply decisions
        for decision in decisions:
            await self.approval_agent.apply_decision(batch.batch_id, decision)

        # Update batch
        updated_batch = await self.approval_agent.storage.get_batch(batch.batch_id)
        if updated_batch:
            batch = updated_batch

        # Update context
        pending_count = len(batch.pending_review)
        rejection_count = len(batch.rejected)

        self.state_machine.context.update(
            {
                "current_batch": batch,
                "pending_review_count": pending_count,
                "rejection_count": rejection_count,
            }
        )

        logger.info(
            f"Approval decisions applied: {pending_count} still pending, "
            f"{rejection_count} rejected"
        )

        # Transition state machine
        if pending_count == 0 and rejection_count == 0:
            # All approved, move to next step
            self.current_step_index += 1
            self.state_machine.transition({"pending_review_count": 0})
        elif rejection_count > 0:
            # Handle rejections
            self.state_machine.transition()

    async def _handle_regeneration(self) -> None:
        """Handle regeneration of rejected items"""
        batch: ApprovalBatch = self.state_machine.context.get("current_batch")
        if not batch:
            logger.error("No batch for regeneration")
            return

        logger.info(f"Regenerating {len(batch.rejected)} rejected items")

        # Mark regeneration complete
        self.state_machine.context["regeneration_complete"] = True

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "workflow_id": self.workflow_id,
            "state": self.state_machine.current_state.value,
            "current_step": self.current_step_index + 1,
            "total_steps": len(self.steps),
            "state_duration": self.state_machine.get_state_duration(),
            "context": self.state_machine.context,
            "state_machine": self.state_machine.to_dict(),
        }
