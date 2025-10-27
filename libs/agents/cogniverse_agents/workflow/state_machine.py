"""
Workflow State Machine

Generic state machine for workflow orchestration with approval checkpoints.
Supports custom states, transitions, and callbacks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Standard workflow states with approval support"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGENERATING = "regenerating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StateTransition:
    """
    Define a state transition with condition and callback

    Example:
        StateTransition(
            from_state=WorkflowState.RUNNING,
            to_state=WorkflowState.AWAITING_APPROVAL,
            condition=lambda ctx: ctx["pending_items"] > 0,
            on_transition=lambda ctx: logger.info("Awaiting approval")
        )
    """

    from_state: WorkflowState
    to_state: WorkflowState
    condition: Callable[[Dict[str, Any]], bool]
    on_transition: Optional[Callable[[Dict[str, Any]], None]] = None
    description: str = ""


@dataclass
class StateHistory:
    """Record of a state change"""

    from_state: Optional[WorkflowState]
    to_state: WorkflowState
    timestamp: datetime
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    transition_reason: str = ""


class WorkflowStateMachine:
    """
    Generic state machine for workflow orchestration

    Features:
    - Custom states and transitions
    - Conditional transitions with callbacks
    - State history tracking
    - Context management
    - Approval checkpoint support

    Example usage:
        sm = WorkflowStateMachine(WorkflowState.INITIALIZING)

        # Register transitions
        sm.register_transition(
            from_state=WorkflowState.RUNNING,
            to_state=WorkflowState.AWAITING_APPROVAL,
            condition=lambda ctx: ctx["pending_count"] > 0,
            description="Items need human review"
        )

        # Execute state machine
        context = {"pending_count": 10}
        sm.transition(context)  # -> AWAITING_APPROVAL

        # Later, after approval
        context["pending_count"] = 0
        sm.transition(context)  # -> Next state
    """

    def __init__(
        self,
        initial_state: WorkflowState,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize state machine

        Args:
            initial_state: Starting state
            context: Initial context dictionary
        """
        self.current_state = initial_state
        self.context = context or {}
        self.transitions: List[StateTransition] = []
        self.history: List[StateHistory] = []

        # Record initial state
        self.history.append(
            StateHistory(
                from_state=None,
                to_state=initial_state,
                timestamp=datetime.utcnow(),
                context_snapshot=self.context.copy(),
                transition_reason="initialization",
            )
        )

        logger.info(f"Initialized WorkflowStateMachine (state: {initial_state.value})")

    def register_transition(
        self,
        from_state: WorkflowState,
        to_state: WorkflowState,
        condition: Callable[[Dict[str, Any]], bool],
        on_transition: Optional[Callable[[Dict[str, Any]], None]] = None,
        description: str = "",
    ) -> None:
        """
        Register a state transition

        Args:
            from_state: Source state
            to_state: Target state
            condition: Function that returns True if transition should occur
            on_transition: Optional callback executed during transition
            description: Human-readable description of transition
        """
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            on_transition=on_transition,
            description=description,
        )
        self.transitions.append(transition)
        logger.debug(
            f"Registered transition: {from_state.value} -> {to_state.value} "
            f"({description})"
        )

    def transition(self, context_updates: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt state transition based on registered rules

        Updates context and checks all registered transitions for current state.
        Executes first matching transition.

        Args:
            context_updates: Updates to merge into context

        Returns:
            True if transition occurred, False otherwise
        """
        # Update context
        if context_updates:
            self.context.update(context_updates)

        # Find matching transition
        for trans in self.transitions:
            if trans.from_state != self.current_state:
                continue

            # Check condition
            try:
                if trans.condition(self.context):
                    self._execute_transition(trans)
                    return True
            except Exception as e:
                logger.error(
                    f"Transition condition check failed: {e} "
                    f"({trans.from_state.value} -> {trans.to_state.value})"
                )

        logger.debug(f"No matching transition from {self.current_state.value}")
        return False

    def force_transition(self, to_state: WorkflowState, reason: str = "") -> None:
        """
        Force transition to a specific state (bypassing conditions)

        Use sparingly - typically for error handling or manual intervention.

        Args:
            to_state: Target state
            reason: Reason for forced transition
        """
        old_state = self.current_state
        self.current_state = to_state

        # Record in history
        self.history.append(
            StateHistory(
                from_state=old_state,
                to_state=to_state,
                timestamp=datetime.utcnow(),
                context_snapshot=self.context.copy(),
                transition_reason=f"forced: {reason}",
            )
        )

        logger.warning(
            f"Forced transition: {old_state.value} -> {to_state.value} "
            f"(reason: {reason})"
        )

    def _execute_transition(self, transition: StateTransition) -> None:
        """Execute a state transition"""
        old_state = self.current_state
        new_state = transition.to_state

        logger.info(
            f"Transitioning: {old_state.value} -> {new_state.value} "
            f"({transition.description})"
        )

        # Execute callback
        if transition.on_transition:
            try:
                transition.on_transition(self.context)
            except Exception as e:
                logger.error(f"Transition callback failed: {e}")

        # Update state
        self.current_state = new_state

        # Record in history
        self.history.append(
            StateHistory(
                from_state=old_state,
                to_state=new_state,
                timestamp=datetime.utcnow(),
                context_snapshot=self.context.copy(),
                transition_reason=transition.description,
            )
        )

    def get_history(self, limit: Optional[int] = None) -> List[StateHistory]:
        """
        Get state transition history

        Args:
            limit: Optional limit on number of entries (most recent)

        Returns:
            List of StateHistory entries
        """
        if limit:
            return self.history[-limit:]
        return self.history.copy()

    def get_state_duration(self) -> float:
        """
        Get duration in current state (seconds)

        Returns:
            Seconds since last transition
        """
        if not self.history:
            return 0.0

        last_transition = self.history[-1]
        duration = (datetime.utcnow() - last_transition.timestamp).total_seconds()
        return duration

    def is_terminal(self) -> bool:
        """Check if current state is terminal (completed or failed)"""
        return self.current_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]

    def can_transition_to(self, target_state: WorkflowState) -> bool:
        """
        Check if transition to target state is possible

        Args:
            target_state: State to check

        Returns:
            True if transition is possible with current context
        """
        for trans in self.transitions:
            if trans.from_state == self.current_state and trans.to_state == target_state:
                try:
                    return trans.condition(self.context)
                except Exception:
                    return False

        return False

    def get_available_transitions(self) -> List[Tuple[WorkflowState, str]]:
        """
        Get all possible transitions from current state

        Returns:
            List of (target_state, description) tuples
        """
        available = []
        for trans in self.transitions:
            if trans.from_state == self.current_state:
                try:
                    if trans.condition(self.context):
                        available.append((trans.to_state, trans.description))
                except Exception:
                    pass

        return available

    def to_dict(self) -> Dict[str, Any]:
        """Export state machine state as dictionary"""
        return {
            "current_state": self.current_state.value,
            "context": self.context,
            "state_duration": self.get_state_duration(),
            "is_terminal": self.is_terminal(),
            "history_count": len(self.history),
            "available_transitions": [
                {"state": state.value, "description": desc}
                for state, desc in self.get_available_transitions()
            ],
        }
