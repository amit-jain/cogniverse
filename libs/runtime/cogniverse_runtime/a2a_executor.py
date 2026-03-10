"""A2A Protocol executor — bridges a2a-sdk to in-process agent dispatch.

Implements the AgentExecutor interface from the official a2a-sdk so that
external A2A clients can call cogniverse agents via JSON-RPC 2.0.
"""

import json
import logging
from typing import Any, Dict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

logger = logging.getLogger(__name__)


class CogniverseAgentExecutor(AgentExecutor):
    """Bridges the a2a-sdk AgentExecutor to the internal AgentDispatcher.

    Extracts agent_name + query from the incoming A2A message, dispatches
    to the correct in-process agent, and enqueues the result as an A2A
    text message.
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

        task_context: Dict[str, Any] = {"tenant_id": tenant_id}

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

        await event_queue.enqueue_event(new_agent_text_message(result_text))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise NotImplementedError("Task cancellation not supported")

    def _infer_agent_from_text(self, text: str) -> str:
        """Fall back to routing_agent if no agent_name is provided."""
        return "routing_agent"
