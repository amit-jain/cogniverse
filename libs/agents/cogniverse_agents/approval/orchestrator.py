"""
Decision Orchestrator

Orchestrates workflows with approval checkpoints using state machine.
Integrates HumanApprovalAgent with workflow execution.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.workflow.state_machine import (
    WorkflowState,
    WorkflowStateMachine,
)
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
)
from cogniverse_core.common.tenant_utils import require_tenant_id, validate_tenant_id

logger = logging.getLogger(__name__)


def _validated_tenant_id(value: Any, *, source: str) -> str:
    tenant_id = require_tenant_id(value, source=source)
    validate_tenant_id(tenant_id)
    return tenant_id


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
            workflow_id="synthetic_generation_001",
            initial_context={
                "agent_type": "routing",
                "tenant_id": "acme",
            },
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
        result = await orchestrator.execute()
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
            initial_context: Initial workflow context. Must include a non-empty
                ``agent_type`` identifying the training-data consumer and an
                explicit ``tenant_id``.
        """
        self.approval_agent = approval_agent
        self.workflow_id = workflow_id
        context = dict(initial_context or {})
        self.tenant_id = _validated_tenant_id(
            context.get("tenant_id"),
            source="DecisionOrchestrator initial_context",
        )
        context["tenant_id"] = self.tenant_id

        storage = approval_agent.storage
        storage_tenant_raw = getattr(storage, "tenant_id", None)
        if storage_tenant_raw is not None:
            storage_tenant = _validated_tenant_id(
                storage_tenant_raw,
                source="DecisionOrchestrator approval storage",
            )
            if storage_tenant != self.tenant_id:
                raise ValueError(
                    "DecisionOrchestrator tenant_id does not match approval "
                    f"storage: context={self.tenant_id} storage={storage_tenant}"
                )

        self.state_machine = WorkflowStateMachine(
            initial_state=WorkflowState.INITIALIZING,
            context=context,
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
            condition=lambda ctx: (
                ctx.get("pending_review_count", 0) == 0
                and ctx.get("rejection_count", 0) == 0
            ),
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

        # Running -> Approved (approval step with nothing left to review:
        # every item auto-approved, an empty result, or a non-list result).
        # Without this the step never leaves RUNNING and re-executes forever,
        # because the human gate (AWAITING_APPROVAL) needs pending>0 and
        # apply_approvals() is never reached. APPROVED then routes to the next
        # step or to COMPLETED via is_last_step, exactly like the human path.
        self.state_machine.register_transition(
            from_state=WorkflowState.RUNNING,
            to_state=WorkflowState.APPROVED,
            condition=lambda ctx: (
                ctx.get("current_step_requires_approval", False)
                and ctx.get("pending_review_count", 0) == 0
                and ctx.get("rejection_count", 0) == 0
            ),
            description="Approval step auto-approved with no pending items",
        )

        # Awaiting Approval -> Rejected (if any rejections)
        self.state_machine.register_transition(
            from_state=WorkflowState.AWAITING_APPROVAL,
            to_state=WorkflowState.REJECTED,
            condition=lambda ctx: (
                ctx.get("pending_review_count", 0) == 0
                and ctx.get("rejection_count", 0) > 0
            ),
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

        Raises:
            ValueError: If ``context.agent_type`` is missing or empty, or a
                context update attempts to change the workflow tenant.
        """
        if context_updates:
            updates = dict(context_updates)
            if "tenant_id" in updates:
                updated_tenant = _validated_tenant_id(
                    updates["tenant_id"],
                    source="DecisionOrchestrator context update",
                )
                if updated_tenant != self.tenant_id:
                    raise ValueError(
                        "DecisionOrchestrator tenant_id cannot change: "
                        f"initial={self.tenant_id} update={updated_tenant}"
                    )
                updates["tenant_id"] = updated_tenant
            self.state_machine.context.update(updates)

        context_tenant = _validated_tenant_id(
            self.state_machine.context.get("tenant_id"),
            source="DecisionOrchestrator workflow context",
        )
        if context_tenant != self.tenant_id:
            raise ValueError(
                "DecisionOrchestrator workflow context tenant_id does not match "
                f"its initial tenant: initial={self.tenant_id} "
                f"context={context_tenant}"
            )
        self.state_machine.context["tenant_id"] = context_tenant

        agent_type = self.state_machine.context.get("agent_type")
        if not isinstance(agent_type, str) or not agent_type.strip():
            raise ValueError("context.agent_type must be a non-empty string")

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
            if inspect.isawaitable(result):
                result = await result

            context_tenant = _validated_tenant_id(
                self.state_machine.context.get("tenant_id"),
                source=f"DecisionOrchestrator step {step['name']} context",
            )
            if context_tenant != self.tenant_id:
                raise ValueError(
                    "DecisionOrchestrator step changed tenant_id: "
                    f"initial={self.tenant_id} step={context_tenant}"
                )
            self.state_machine.context["tenant_id"] = context_tenant

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
                        "agent_type": self.state_machine.context.get("agent_type"),
                        "tenant_id": self.tenant_id,
                        "workflow_id": self.workflow_id,
                        "step_name": step["name"],
                        "step_index": self.current_step_index,
                    },
                )

                self.state_machine.context["current_batch"] = batch
                self.state_machine.context["pending_review_count"] = len(
                    batch.pending_review
                )
                self.state_machine.context["rejection_count"] = len(batch.rejected)
                logger.info(
                    f"Step '{step['name']}' generated {len(batch.items)} items: "
                    f"{len(batch.auto_approved)} auto-approved, "
                    f"{len(batch.pending_review)} pending review"
                )
            else:
                self.state_machine.context["pending_review_count"] = 0
                self.state_machine.context["rejection_count"] = 0

            # Advance when no human gate remains: a non-approval step, or an
            # approval step whose items were all auto-approved (no pending).
            # Approval steps with pending items wait for apply_approvals() to
            # advance the index after the human decides.
            if (
                not step["requires_approval"]
                or self.state_machine.context.get("pending_review_count", 0) == 0
            ):
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

        applied_items = []
        for decision in decisions:
            applied_item = await self.approval_agent.apply_decision(
                batch.batch_id, decision
            )
            if applied_item is None:
                raise RuntimeError(
                    "Approval decision produced no persisted item: "
                    f"batch={batch.batch_id} item={decision.item_id}"
                )
            applied_items.append(applied_item)

        # Phoenix span queries are eventually consistent even after a checked
        # export. Reload the latest visible batch, then overlay the exact items
        # returned by the persistence boundary so this workflow never reverts a
        # just-applied decision to an older pending snapshot.
        updated_batch = await self.approval_agent.storage.get_batch(batch.batch_id)
        if updated_batch:
            batch = updated_batch
        batch_tenant = _validated_tenant_id(
            batch.context.get("tenant_id"),
            source=f"DecisionOrchestrator approval batch {batch.batch_id}",
        )
        if batch_tenant != self.tenant_id:
            raise ValueError(
                "DecisionOrchestrator approval batch tenant_id does not match "
                f"its initial tenant: initial={self.tenant_id} batch={batch_tenant}"
            )
        batch.context["tenant_id"] = batch_tenant
        for applied_item in applied_items:
            original_item_id = applied_item.metadata.get("original_item_id")
            if original_item_id is not None:
                if (
                    not isinstance(original_item_id, str)
                    or not original_item_id
                    or original_item_id == applied_item.item_id
                ):
                    raise RuntimeError(
                        "Invalid approval item lineage: "
                        f"item={applied_item.item_id} "
                        f"original_item_id={original_item_id!r} "
                        f"status={applied_item.status.value}"
                    )

                original = next(
                    (
                        candidate
                        for candidate in batch.items
                        if candidate.item_id == original_item_id
                    ),
                    None,
                )
                if original is None:
                    raise RuntimeError(
                        "Approval item lineage references an unknown original: "
                        f"batch={batch.batch_id} item={applied_item.item_id} "
                        f"original_item_id={original_item_id}"
                    )
                original.status = ApprovalStatus.REJECTED
            elif applied_item.status is ApprovalStatus.REGENERATED:
                raise RuntimeError(
                    "Regenerated approval item is missing its original item: "
                    f"batch={batch.batch_id} item={applied_item.item_id}"
                )

            for index, candidate in enumerate(batch.items):
                if candidate.item_id == applied_item.item_id:
                    batch.items[index] = applied_item
                    break
            else:
                batch.items.append(applied_item)

        # Update context
        pending_count = len(batch.pending_review)
        superseded_item_ids = {
            original_item_id
            for item in batch.items
            if isinstance(
                original_item_id := item.metadata.get("original_item_id"), str
            )
            and original_item_id
        }
        rejection_count = sum(
            item.status is ApprovalStatus.REJECTED
            and item.item_id not in superseded_item_ids
            for item in batch.items
        )

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
        elif pending_count == 0 and rejection_count > 0:
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
