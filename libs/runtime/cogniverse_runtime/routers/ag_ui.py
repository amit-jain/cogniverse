"""AG-UI surface for web clients.

``POST /ag-ui/{agent_name}`` runs one turn of a registered agent for an AG-UI
client (CopilotKit's ``HttpAgent`` in particular) and streams the run as AG-UI
events over SSE. The bearer key, key map, dispatcher and continuation store
are the ones the ``/v1`` surface is wired with, so a key serves both surfaces
for the same tenant.

The client holds the conversation and sends the whole of it on every run, so
a run is self-contained exactly as a ``/v1`` request is. A run streams:

- ``RUN_STARTED`` with the client's thread and run ids;
- ``STEP_STARTED`` / ``STEP_FINISHED`` around each agent phase, plus a
  ``CUSTOM`` ``cogniverse.status`` event carrying the phase's message;
- the reply as one ``TEXT_MESSAGE_START`` / ``_CONTENT`` / ``_END`` sequence;
- ``STATE_SNAPSHOT`` with the agent's final payload under ``result``, which
  a client renders as results cards;
- ``RUN_FINISHED``, or ``RUN_ERROR`` when the turn failed.

A search payload carries the ``span_id`` of the search; ``POST
/ag-ui/results/relevance`` stores a reviewer's relevance label for one of its
results as that span's ``result_relevance`` annotation, in the key's tenant.

Frontend tools: the run's ``tools`` reach the agent as its external tools. An
agent that suspends on them streams one ``TOOL_CALL_START`` / ``_ARGS`` /
``_END`` sequence per call and finishes with ``outcome.pendingToolCallIds``;
the client runs the tools and starts a new run with the tool messages
appended, which resumes the turn from the shared continuation store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    CustomEvent,
    DeveloperMessage,
    ImagePart,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    SystemMessage,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TextPart,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from cogniverse_core.registries.agent_registry import AgentRegistryUnavailableError
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.span_contract import (
    RELEVANCE_SCORES,
    SpanNotInProjectError,
    persist_result_relevance,
)
from cogniverse_runtime.routers.openai_compat import (
    UNAUTHORIZED,
    DispatcherNotReady,
    RequestShapeError,
    ToolsForbiddenError,
    answer_token_events,
    build_dispatch_args,
    current_dispatcher,
    dependency_unavailable,
    error_response,
    failure_body,
    in_flight_turn,
    resolve_tenant_off_loop,
    run_turn,
    split_answer_chunks,
    use_token_stream,
)
from cogniverse_runtime.session_state import SessionStateUnavailable
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

STATUS_EVENT = "cogniverse.status"


def _openai_content(content: Any, where: str) -> Any:
    """AG-UI message content in the OpenAI shape ``build_dispatch_args`` reads.

    Raises:
        RequestShapeError: a part other than text or an image.
    """
    if isinstance(content, str) or content is None:
        return content
    parts: List[Dict[str, Any]] = []
    for index, part in enumerate(content):
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            source = part.source
            if source.type == "url":
                url = source.value
            else:
                url = f"data:{source.mime_type};base64,{source.value}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise RequestShapeError(
                f"{where} part {index} has unsupported type {part.type!r}; only "
                "text and image parts are accepted"
            )
    return parts


def to_openai_messages(run_input: RunAgentInput) -> List[Dict[str, Any]]:
    """The run's messages as an OpenAI ``messages`` array.

    Developer messages become system messages. Activity and reasoning
    messages are the client's own rendering of earlier runs and carry nothing
    the agent reads, so they are left out.

    Raises:
        RequestShapeError: a content part this surface does not accept.
    """
    messages: List[Dict[str, Any]] = []
    for index, message in enumerate(run_input.messages):
        where = f"messages[{index}]"
        if isinstance(message, UserMessage):
            messages.append(
                {"role": "user", "content": _openai_content(message.content, where)}
            )
        elif isinstance(message, AssistantMessage):
            entry: Dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(entry)
        elif isinstance(message, ToolMessage):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": _openai_content(message.content, where),
                }
            )
        elif isinstance(message, (SystemMessage, DeveloperMessage)):
            messages.append({"role": "system", "content": message.content})
    return messages


def to_external_tools(run_input: RunAgentInput) -> Optional[List[Dict[str, Any]]]:
    """The run's frontend tools as OpenAI function definitions."""
    if not run_input.tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
                if tool.parameters is not None
                else {"type": "object", "properties": {}},
            },
        }
        for tool in run_input.tools
    ]


class _RunWriter:
    """Encodes one run's events and keeps the open step and message paired."""

    def __init__(self, run_input: RunAgentInput, agent_name: str) -> None:
        self._encoder = EventEncoder()
        self._thread_id = run_input.thread_id
        self._run_id = run_input.run_id
        self._agent_name = agent_name
        self._message_id = f"msg-{uuid.uuid4().hex}"
        self._message_open = False
        self._message_sent = False
        self._step: Optional[str] = None

    @property
    def message_id(self) -> str:
        return self._message_id

    def _encode(self, event: BaseEvent) -> str:
        return self._encoder.encode(event)

    def started(self) -> str:
        return self._encode(
            RunStartedEvent(thread_id=self._thread_id, run_id=self._run_id)
        )

    def status(self, phase: str, message: str) -> List[str]:
        frames: List[str] = []
        if phase and phase != self._step:
            frames.extend(self._close_step())
            self._step = phase
            frames.append(self._encode(StepStartedEvent(step_name=phase)))
        frames.append(
            self._encode(
                CustomEvent(
                    name=STATUS_EVENT, value={"phase": phase, "message": message}
                )
            )
        )
        return frames

    def _close_step(self) -> List[str]:
        if self._step is None:
            return []
        step, self._step = self._step, None
        return [self._encode(StepFinishedEvent(step_name=step))]

    def text(self, delta: str) -> List[str]:
        if not delta:
            return []
        frames: List[str] = []
        if not self._message_open:
            self._message_open = True
            self._message_sent = True
            frames.append(
                self._encode(
                    TextMessageStartEvent(message_id=self._message_id, role="assistant")
                )
            )
        frames.append(
            self._encode(
                TextMessageContentEvent(message_id=self._message_id, delta=delta)
            )
        )
        return frames

    def _close_message(self) -> List[str]:
        if not self._message_open:
            return []
        self._message_open = False
        return [self._encode(TextMessageEndEvent(message_id=self._message_id))]

    def tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        frames = self._close_message()
        for call in tool_calls:
            frames.append(
                self._encode(
                    ToolCallStartEvent(
                        tool_call_id=call["id"],
                        tool_call_name=call["function"]["name"],
                        parent_message_id=self._message_id,
                    )
                )
            )
            frames.append(
                self._encode(
                    ToolCallArgsEvent(
                        tool_call_id=call["id"], delta=call["function"]["arguments"]
                    )
                )
            )
            frames.append(self._encode(ToolCallEndEvent(tool_call_id=call["id"])))
        return frames

    def finished(
        self,
        payload: Optional[Dict[str, Any]] = None,
        pending_tool_call_ids: Optional[List[str]] = None,
    ) -> List[str]:
        frames = self._close_message()
        if payload is not None:
            frames.append(
                self._encode(
                    StateSnapshotEvent(
                        snapshot={
                            "agent": self._agent_name,
                            "result": jsonable_encoder(payload),
                        }
                    )
                )
            )
        frames.extend(self._close_step())
        frames.append(
            self._encode(
                RunFinishedEvent(
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                    outcome=RunFinishedSuccessOutcome(
                        pending_tool_call_ids=pending_tool_call_ids
                    ),
                )
            )
        )
        return frames

    def failed(self, message: str, code: str) -> List[str]:
        frames = self._close_message() + self._close_step()
        frames.append(self._encode(RunErrorEvent(message=message, code=code)))
        return frames


def _failure_message(exc: BaseException, agent_name: str) -> tuple[str, str]:
    """The message and code a run error carries; exception text stays in the log."""
    if isinstance(exc, SessionStateUnavailable):
        return failure_body(exc, agent_name)["message"], "service_unavailable"
    if isinstance(exc, ToolsForbiddenError):
        return str(exc), "tool_choice_violation"
    return failure_body(exc, agent_name)["message"], "internal_error"


async def _stream_run(
    dispatcher: Any,
    agent_name: str,
    run_input: RunAgentInput,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]],
) -> AsyncIterator[str]:
    """One run as encoded AG-UI events."""
    writer = _RunWriter(run_input, agent_name)
    with in_flight_turn():
        async with aclosing(
            _run_frames(
                writer, dispatcher, agent_name, dispatch_args, tenant_id, external_tools
            )
        ) as frames:
            async for frame in frames:
                yield frame


async def _run_frames(
    writer: "_RunWriter",
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]],
) -> AsyncIterator[str]:
    """The run's frames; a failure ends the run on ``RUN_ERROR``."""
    try:
        yield writer.started()
        if use_token_stream(
            dispatcher, agent_name, bool(dispatch_args["tool_results"])
        ):
            async with aclosing(
                answer_token_events(
                    dispatcher, agent_name, dispatch_args, tenant_id, external_tools
                )
            ) as events:
                async for event in events:
                    kind = event["kind"]
                    if kind == "status":
                        frames = writer.status(event["phase"], event["message"])
                    elif kind == "text":
                        frames = writer.text(event["delta"])
                    elif kind == "error":
                        frames = writer.failed(event["message"], "internal_error")
                    elif kind == "tool_calls":
                        calls = event["tool_calls"]
                        frames = writer.tool_calls(calls) + writer.finished(
                            pending_tool_call_ids=[call["id"] for call in calls]
                        )
                    else:
                        frames = writer.finished(payload=event["payload"])
                    for frame in frames:
                        yield frame
            return

        outcome = await run_turn(
            dispatcher, agent_name, dispatch_args, tenant_id, external_tools
        )
        if outcome["kind"] == "tool_calls":
            calls = outcome["tool_calls"]
            frames = writer.tool_calls(calls) + writer.finished(
                pending_tool_call_ids=[call["id"] for call in calls]
            )
        else:
            frames = []
            for part in split_answer_chunks(outcome["answer"]):
                frames.extend(writer.text(part))
            frames.extend(writer.finished(payload=outcome["payload"]))
        for frame in frames:
            yield frame
    except asyncio.CancelledError:
        logger.info("ag-ui run for %s cancelled", agent_name)
        for frame in writer.failed(
            "Run cancelled before the turn completed.", "run_cancelled"
        ):
            yield frame
        raise
    except Exception as exc:
        logger.exception("ag-ui run for %s failed", agent_name)
        message, code = _failure_message(exc, agent_name)
        for frame in writer.failed(message, code):
            yield frame


class RelevanceRequest(BaseModel):
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    result_id: str = Field(min_length=1)
    relevance: Literal[tuple(RELEVANCE_SCORES)]  # type: ignore[valid-type]


def _invalid_body(exc: ValidationError, what: str) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return error_response(400, f"Invalid {what}: {problems}", "invalid_request")


@router.post("/results/relevance")
async def rate_result(
    raw_request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Store a reviewer's relevance label for one result of a search span of
    the key's tenant; answers the stored label and score."""
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)
    try:
        request = RelevanceRequest.model_validate(await raw_request.json())
    except json.JSONDecodeError as exc:
        return error_response(
            400,
            f"Invalid relevance rating: body is not JSON ({exc})",
            "invalid_request",
        )
    except ValidationError as exc:
        return _invalid_body(exc, "relevance rating")

    try:
        manager = get_telemetry_manager()
        project = manager.config.get_project_name(tenant_id)
        score = await persist_result_relevance(
            manager.get_provider(tenant_id=tenant_id, project_name=project),
            project,
            request.span_id,
            request.result_id,
            request.relevance,
        )
    except SpanNotInProjectError:
        return error_response(
            404,
            f"Search span {request.span_id} is not a span of this tenant.",
            "span_not_found",
        )
    except Exception as exc:
        logger.exception(
            "relevance of result %s on span %s for tenant %s was not stored",
            request.result_id,
            request.span_id,
            tenant_id,
        )
        return error_response(
            502,
            f"The relevance of result {request.result_id} was not stored "
            f"({type(exc).__name__}). See server logs for detail.",
            "annotation_not_stored",
            err_type="server_error",
            error_type=type(exc).__name__,
        )
    return {
        "span_id": request.span_id,
        "result_id": request.result_id,
        "relevance": request.relevance,
        "score": score,
    }


@router.post("/{agent_name}")
async def run_agent(
    agent_name: str,
    raw_request: Request,
    authorization: Optional[str] = Header(default=None),
):
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)

    try:
        run_input = RunAgentInput.model_validate(await raw_request.json())
    except json.JSONDecodeError as exc:
        return error_response(
            400, f"Invalid AG-UI run input: body is not JSON ({exc})", "invalid_request"
        )
    except ValidationError as exc:
        return _invalid_body(exc, "AG-UI run input")
    try:
        dispatch_args = build_dispatch_args(to_openai_messages(run_input))
    except RequestShapeError as exc:
        return error_response(400, str(exc), "invalid_request")

    try:
        dispatcher = current_dispatcher()
    except DispatcherNotReady as exc:
        return error_response(
            503, str(exc), "service_unavailable", err_type="server_error"
        )
    except Exception as exc:
        logger.exception("dispatcher provider failed")
        return dependency_unavailable(exc, "dispatcher")
    try:
        await dispatcher.refresh_agent_registry()
    except AgentRegistryUnavailableError as exc:
        logger.warning("agent registry unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "agent registry")
    if not dispatcher.is_registered(agent_name):
        return error_response(
            404, f"Agent '{agent_name}' is not registered.", "agent_not_found"
        )

    return StreamingResponse(
        _stream_run(
            dispatcher,
            agent_name,
            run_input,
            dispatch_args,
            tenant_id,
            to_external_tools(run_input),
        ),
        media_type="text/event-stream",
    )
