"""A2A Protocol executor — bridges a2a-sdk to in-process agent dispatch.

Implements the AgentExecutor interface from the official a2a-sdk so that
external A2A clients can call cogniverse agents via JSON-RPC 2.0.

Multi-turn support: Extracts conversation history from Task.history
(populated by InMemoryTaskStore when contextId is reused) and threads
it through the dispatcher so agents can reason over prior turns.
"""

import json
import logging
from typing import Any, Dict, List

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.utils import get_message_text, new_agent_text_message

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

logger = logging.getLogger(__name__)


class CogniverseAgentExecutor(AgentExecutor):
    """Bridges the a2a-sdk AgentExecutor to the internal AgentDispatcher.

    Extracts agent_name + query from the incoming A2A message, dispatches
    to the correct in-process agent, and enqueues the result as an A2A
    text message.

    Multi-turn: When the same contextId is reused across calls,
    Task.history accumulates previous messages. This executor extracts
    them into a flat list of ConversationTurn dicts and passes them
    through the context dict.
    """

    def __init__(self, dispatcher: AgentDispatcher) -> None:
        self._dispatcher = dispatcher

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Handle an incoming A2A message/send request."""
        user_text = context.get_user_input()
        metadata = context.metadata

        agent_name = metadata.get("agent_name", "")
        query = metadata.get("query", user_text)
        tenant_id = metadata.get("tenant_id", "default")
        top_k = int(metadata.get("top_k", 10))

        if not agent_name:
            agent_name = self._infer_agent_from_text(user_text)

        task_context: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "context_id": context.context_id,
            "conversation_history": self._extract_conversation_history(context),
        }

        try:
            result = await self._dispatcher.dispatch(
                agent_name=agent_name,
                query=query,
                context=task_context,
                top_k=top_k,
            )
            result_text = json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"A2A dispatch failed for agent '{agent_name}': {e}")
            result_text = json.dumps(
                {"status": "error", "error": str(e), "agent": agent_name}
            )

        # Emit a TaskStatusUpdateEvent so the task is persisted to the store
        # with the agent's response as status.message. This enables multi-turn
        # conversations: on subsequent turns with the same taskId, the handler
        # retrieves the task and builds history from prior status messages.
        # Emit TaskStatusUpdateEvent with input-required state so the task
        # stays alive for multi-turn conversations. The handler persists it
        # to InMemoryTaskStore, enabling history accumulation on subsequent
        # turns with the same taskId.
        response_message = new_agent_text_message(result_text)
        event = TaskStatusUpdateEvent(
            task_id=context.task_id or "",
            context_id=context.context_id or "",
            final=False,
            status=TaskStatus(
                state=TaskState.input_required,
                message=response_message,
            ),
        )
        await event_queue.enqueue_event(event)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise NotImplementedError("Task cancellation not supported")

    def _infer_agent_from_text(self, text: str) -> str:
        """Fall back to routing_agent if no agent_name is provided."""
        return "routing_agent"

    def _extract_conversation_history(
        self, context: RequestContext
    ) -> List[Dict[str, str]]:
        """Extract previous conversation turns from A2A Task.history.

        Returns a list of dicts with 'role' and 'content' keys,
        excluding the current message (which is the latest in history).
        """
        task = context.current_task
        if not task or not task.history:
            return []

        # Task.history includes all messages up to and including the current one.
        # Exclude the last message (the current user input being processed).
        prior_messages: List[Message] = list(task.history[:-1])

        turns: List[Dict[str, str]] = []
        for msg in prior_messages:
            text = get_message_text(msg)
            if text:
                turns.append({"role": msg.role, "content": text})

        return turns
