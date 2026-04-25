"""A2A Protocol executor — bridges a2a-sdk to in-process agent dispatch.

Implements the AgentExecutor interface from the official a2a-sdk so that
external A2A clients can call cogniverse agents via JSON-RPC 2.0.

Streaming: Agents that support streaming (e.g., SummarizerAgent) emit
intermediate progress events to the A2A EventQueue during execution.
Clients using /tasks/sendSubscribe receive these as SSE events.

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

from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

logger = logging.getLogger(__name__)

# All agents support streaming via emit_progress() and call_dspy()
_STREAMING_CAPABILITIES = frozenset(
    {
        "summarization",
        "text_generation",
        "routing",
        "search",
        "video_search",
        "retrieval",
        "detailed_report",
        "orchestration",
        "planning",
        "query_enhancement",
        "entity_extraction",
        "profile_selection",
        "image_search",
        "visual_analysis",
        "audio_analysis",
        "transcription",
        "document_analysis",
        "pdf_processing",
        "coding",
        # GatewayAgent's capabilities are ``gateway`` + ``classification``.
        # Without these in the streaming set the executor falls through to
        # the single-event path; clients subscribed to /a2a/ message/stream
        # then never see emit_progress() events from the GLiNER classifier
        # or downstream dispatch.
        "gateway",
        "classification",
    }
)


class CogniverseAgentExecutor(AgentExecutor):
    """Bridges the a2a-sdk AgentExecutor to the internal AgentDispatcher.

    Extracts agent_name + query from the incoming A2A message, dispatches
    to the correct in-process agent, and enqueues the result as an A2A
    text message.

    Streaming: For agents with streaming capabilities, emits intermediate
    TaskStatusUpdateEvents (state=working, final=False) followed by a
    final event (state=input_required, final=False for multi-turn).
    Clients using /tasks/sendSubscribe receive these as SSE.

    Multi-turn: When the same contextId is reused across calls,
    Task.history accumulates previous messages. This executor extracts
    them into a flat list of ConversationTurn dicts and passes them
    through the context dict.
    """

    def __init__(self, dispatcher: AgentDispatcher) -> None:
        self._dispatcher = dispatcher

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle an incoming A2A message/send request.

        For streaming-capable agents (summarization, text_generation),
        emits intermediate progress events to the queue during execution.
        Non-streaming agents emit a single final event.
        """
        user_text = context.get_user_input()
        metadata = context.metadata

        agent_name = metadata.get("agent_name", "")
        query = metadata.get("query", user_text)
        tenant_id = require_tenant_id(
            metadata.get("tenant_id"), source="A2A request metadata"
        )
        top_k = int(metadata.get("top_k", 10))
        stream = metadata.get("stream", False)

        if not agent_name:
            agent_name = self._infer_agent_from_text(user_text)

        task_context: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "context_id": context.context_id,
            "conversation_history": self._extract_conversation_history(context),
        }

        task_id = context.task_id or ""
        context_id = context.context_id or ""

        # Check if agent supports streaming and client requested it
        agent_entry = self._dispatcher._registry.get_agent(agent_name)
        capabilities = set(agent_entry.capabilities) if agent_entry else set()
        use_streaming = stream and bool(capabilities & _STREAMING_CAPABILITIES)

        if use_streaming:
            await self._execute_streaming(
                agent_name, query, tenant_id, task_id, context_id, event_queue
            )
        else:
            await self._execute_non_streaming(
                agent_name, query, task_context, top_k, task_id, context_id, event_queue
            )

    async def _execute_non_streaming(
        self,
        agent_name: str,
        query: str,
        task_context: Dict[str, Any],
        top_k: int,
        task_id: str,
        context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """Dispatch non-streaming and emit a single result event."""
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

        response_message = new_agent_text_message(result_text)
        event = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            final=False,
            status=TaskStatus(
                state=TaskState.input_required,
                message=response_message,
            ),
        )
        await event_queue.enqueue_event(event)

    async def _execute_streaming(
        self,
        agent_name: str,
        query: str,
        tenant_id: str,
        task_id: str,
        context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """Stream agent results as intermediate A2A events.

        Creates the agent, calls process(stream=True), and maps each
        yielded event dict to a TaskStatusUpdateEvent on the A2A queue.
        """
        try:
            import asyncio
            import contextlib

            agent, typed_input = await asyncio.to_thread(
                self._dispatcher.create_streaming_agent,
                agent_name,
                query,
                tenant_id,
            )

            agent_lm = getattr(agent, "_dspy_lm", None)
            if agent_lm is not None:
                import dspy

                lm_ctx = dspy.context(lm=agent_lm)
            else:
                lm_ctx = contextlib.nullcontext()

            async def _iterate_with_ctx():
                with lm_ctx:
                    async for ev in await agent.process(typed_input, stream=True):
                        yield ev

            async for event in _iterate_with_ctx():
                event_type = event.get("type", "")
                event_text = json.dumps(event, default=str)

                is_final = event_type == "final"
                state = TaskState.input_required if is_final else TaskState.working

                a2a_event = TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    final=False,
                    status=TaskStatus(
                        state=state,
                        message=new_agent_text_message(event_text),
                    ),
                )
                await event_queue.enqueue_event(a2a_event)

        except Exception as e:
            logger.error(f"A2A streaming failed for agent '{agent_name}': {e}")
            error_text = json.dumps(
                {"type": "error", "message": str(e), "agent": agent_name}
            )
            error_event = TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                final=False,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=new_agent_text_message(error_text),
                ),
            )
            await event_queue.enqueue_event(error_event)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

        cancel_event = TaskStatusUpdateEvent(
            status=TaskStatus(
                state=TaskState.canceled,
                message=new_agent_text_message(
                    "Task cancellation acknowledged. Cogniverse does not support "
                    "mid-execution cancellation — the task may have already completed."
                ),
            ),
        )
        await event_queue.enqueue_event(cancel_event)

    def _infer_agent_from_text(self, text: str) -> str:
        """Fall back to orchestrator_agent when no agent_name is provided.

        The orchestrator plans and executes against the agent registry for
        both simple and complex queries.
        """
        return "orchestrator_agent"

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
