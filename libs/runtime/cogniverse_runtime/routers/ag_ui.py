"""AG-UI surface for web clients.

``POST /ag-ui/{agent_name}`` runs one turn of a registered agent for an AG-UI
client (CopilotKit's ``HttpAgent`` in particular) and streams the run as AG-UI
events over SSE. The bearer key, key map, dispatcher and continuation store
are the ones the ``/v1`` surface is wired with, so a key serves both surfaces
for the same tenant.

The client holds the conversation and sends the whole of it on every run, so
a run is self-contained exactly as a ``/v1`` request is. Each finished run
also saves its turn (the user message and the reply, the user message alone
when the run failed, or the user message and a ``run_cancelled`` marker when
the client hung up or cancelled it first) to the tenant's conversation store
under the run's thread, and ``GET /ag-ui/threads/{thread_id}`` reads a thread's saved turns
back, so a client restores a conversation from the runtime. A run streams:

- ``RUN_STARTED`` with the client's thread and run ids;
- ``STEP_STARTED`` / ``STEP_FINISHED`` around each phase, plus a ``CUSTOM``
  ``cogniverse.status`` event carrying the phase's message (and the
  ``themes`` / ``summary`` a partial result reports): first the
  ``starting`` step, sent before the agent runs, then every phase the agent
  reports as it reaches it, whichever path serves the turn;
- the reply as one ``TEXT_MESSAGE_START`` / ``_CONTENT`` / ``_END`` sequence,
  streamed token by token for an agent that declares
  ``streams_answer_tokens`` and sent when the turn completes otherwise;
- ``STATE_SNAPSHOT`` with the agent's final payload under ``result`` and the
  tenant it ran for under ``tenant_id``, which a client renders as results
  cards for that tenant only;
- ``RUN_FINISHED``, or ``RUN_ERROR`` when the turn failed or its reply could
  not be saved to the thread.

Per-run parameters travel in ``forwardedProps.cogniverse``: ``top_k`` (how
many hits a searching agent returns, 1-100) and ``search_results`` (hits the
client already shows, which an answer agent such as the summarizer is grounded
in instead of searching again). Any other key there is refused with 400.

A search payload carries the ``span_id`` of the search; ``POST
/ag-ui/results/relevance`` stores a reviewer's relevance label for one of its
results as that span's ``result_relevance`` annotation, in the key's tenant.
``POST /ag-ui/threads/{thread_id}/evaluation`` stores a reviewer's verdict on
a whole conversation as a ``session_evaluation`` annotation on each search
span of it the client names.

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
import re
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
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from cogniverse_core.agents.base import collect_progress
from cogniverse_core.registries.agent_registry import AgentRegistryUnavailableError
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.span_contract import (
    RELEVANCE_SCORES,
    SESSION_OUTCOMES,
    SpanNotInProjectError,
    persist_result_relevance,
    persist_session_evaluation,
    span_readable_within_s,
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
    progress_details,
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
# The step every run opens with, before the agent reports a phase.
START_PHASE = "starting"


# The forwardedProps key a client's per-run parameters travel under; the
# rest of forwardedProps belongs to the client's framework.
RUN_PARAMETERS_KEY = "cogniverse"
MAX_TOP_K = 100
MAX_THREADED_RESULTS = 50


def thread_context_id(thread_id: str) -> str:
    """The conversation context an AG-UI thread's turns are saved under."""
    return f"ag-ui:{thread_id}"


class RunParameters(BaseModel):
    """A run's parameters from ``forwardedProps.cogniverse``."""

    model_config = ConfigDict(extra="forbid")

    top_k: Optional[StrictInt] = Field(default=None, ge=1, le=MAX_TOP_K)
    search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None, min_length=1, max_length=MAX_THREADED_RESULTS
    )

    def context(self) -> Dict[str, Any]:
        """The parameters placed on the dispatch context, where an agent's
        input of the same field name reads them."""
        return self.model_dump(exclude_none=True)


def run_parameters(run_input: RunAgentInput) -> RunParameters:
    """The run's parameters; none when the client sent none.

    Raises:
        RequestShapeError: ``forwardedProps.cogniverse`` is not an object of
            known, valid parameters.
    """
    props = run_input.forwarded_props
    if props is None:
        return RunParameters()
    if not isinstance(props, dict):
        raise RequestShapeError("forwardedProps must be an object")
    raw = props.get(RUN_PARAMETERS_KEY)
    if raw is None:
        return RunParameters()
    try:
        return RunParameters.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
            for error in exc.errors()
        )
        raise RequestShapeError(
            f"forwardedProps.{RUN_PARAMETERS_KEY} is invalid: {problems}"
        ) from exc


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

    def __init__(
        self, run_input: RunAgentInput, agent_name: str, tenant_id: str
    ) -> None:
        self._encoder = EventEncoder()
        self._thread_id = run_input.thread_id
        self._run_id = run_input.run_id
        self._agent_name = agent_name
        self._tenant_id = tenant_id
        self._message_id = f"msg-{uuid.uuid4().hex}"
        self._message_open = False
        self._message_sent = False
        self._step: Optional[str] = None

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def _encode(self, event: BaseEvent) -> str:
        return self._encoder.encode(event)

    def started(self) -> str:
        return self._encode(
            RunStartedEvent(thread_id=self._thread_id, run_id=self._run_id)
        )

    def status(
        self, phase: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        frames: List[str] = []
        if phase and phase != self._step:
            frames.extend(self._close_step())
            self._step = phase
            frames.append(self._encode(StepStartedEvent(step_name=phase)))
        frames.append(
            self._encode(
                CustomEvent(
                    name=STATUS_EVENT,
                    value={"phase": phase, "message": message, **(details or {})},
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
                            "tenant_id": self._tenant_id,
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


def _progress_status(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An agent progress event as a status event; None for a token event,
    which on the dispatch path is not the reply."""
    if event.get("type") not in ("status", "partial") or event.get("phase") == "token":
        return None
    return {
        "kind": "status",
        "phase": str(event.get("phase", "")),
        "message": str(event.get("message", "")),
        **progress_details(event),
    }


def _shown_details(event: Dict[str, Any]) -> Dict[str, Any]:
    """The themes and summary a status event carries beside its message."""
    return {key: event[key] for key in ("themes", "summary") if key in event}


async def _dispatch_events(
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]],
    parameters: RunParameters,
) -> AsyncIterator[Dict[str, Any]]:
    """The dispatch path of one turn as events.

    Yields ``{"kind": "status", "phase", "message"}`` for each phase the agent
    reports while the turn runs, then ``{"kind": "outcome", "outcome"}`` with
    what ``run_turn`` returned; a failed turn raises its exception after the
    phases reported before it. Closing the iterator cancels the turn.
    """
    with collect_progress() as progress:
        turn = asyncio.create_task(
            run_turn(
                dispatcher,
                agent_name,
                dispatch_args,
                tenant_id,
                external_tools,
                sampling=parameters.context(),
                top_k=parameters.top_k,
            )
        )
    next_event: Optional[asyncio.Future] = None
    try:
        while True:
            next_event = asyncio.ensure_future(progress.get())
            await asyncio.wait({next_event, turn}, return_when=asyncio.FIRST_COMPLETED)
            if not next_event.done():
                break
            status = _progress_status(next_event.result())
            if status is not None:
                yield status
        while not progress.empty():
            status = _progress_status(progress.get_nowait())
            if status is not None:
                yield status
        yield {"kind": "outcome", "outcome": turn.result()}
    finally:
        if next_event is not None and not next_event.done():
            next_event.cancel()
        if not turn.done():
            turn.cancel()
            try:
                await turn
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("ag-ui turn for %s failed while cancelled", agent_name)


async def _stream_run(
    dispatcher: Any,
    agent_name: str,
    run_input: RunAgentInput,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]],
    parameters: RunParameters,
) -> AsyncIterator[str]:
    """One run as encoded AG-UI events."""
    writer = _RunWriter(run_input, agent_name, tenant_id)
    with in_flight_turn():
        async with aclosing(
            _run_frames(
                writer,
                dispatcher,
                agent_name,
                dispatch_args,
                tenant_id,
                external_tools,
                parameters,
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
    parameters: RunParameters,
) -> AsyncIterator[str]:
    """The run's frames; a failure ends the run on ``RUN_ERROR``.

    A run that ends before its turn was saved or suspended — the client hung
    up, or the run was cancelled — saves its user message with a cancelled
    marker on the way out.
    """
    query = dispatch_args["query"]
    settled = False
    try:
        yield writer.started()
        for frame in writer.status(START_PHASE, f"Running {agent_name}"):
            yield frame
        if use_token_stream(
            dispatcher, agent_name, bool(dispatch_args["tool_results"])
        ):
            async with aclosing(
                answer_token_events(
                    dispatcher,
                    agent_name,
                    dispatch_args,
                    tenant_id,
                    external_tools,
                    sampling=parameters.context(),
                )
            ) as events:
                async for event in events:
                    kind = event["kind"]
                    if kind == "status":
                        frames = writer.status(
                            event["phase"], event["message"], _shown_details(event)
                        )
                    elif kind == "text":
                        frames = writer.text(event["delta"])
                    elif kind == "error":
                        await _save_unanswered(
                            dispatcher, writer, tenant_id, query, agent_name
                        )
                        settled = True
                        frames = writer.failed(event["message"], "internal_error")
                    elif kind == "tool_calls":
                        settled = True
                        calls = event["tool_calls"]
                        frames = writer.tool_calls(calls) + writer.finished(
                            pending_tool_call_ids=[call["id"] for call in calls]
                        )
                    else:
                        frames = await _answered(
                            dispatcher,
                            writer,
                            tenant_id,
                            query,
                            event["text"],
                            event["payload"],
                        )
                        settled = True
                    for frame in frames:
                        yield frame
            return

        outcome: Dict[str, Any] = {}
        async with aclosing(
            _dispatch_events(
                dispatcher,
                agent_name,
                dispatch_args,
                tenant_id,
                external_tools,
                parameters,
            )
        ) as events:
            async for event in events:
                if event["kind"] == "status":
                    for frame in writer.status(
                        event["phase"], event["message"], _shown_details(event)
                    ):
                        yield frame
                else:
                    outcome = event["outcome"]
        if outcome["kind"] == "tool_calls":
            settled = True
            calls = outcome["tool_calls"]
            frames = writer.tool_calls(calls) + writer.finished(
                pending_tool_call_ids=[call["id"] for call in calls]
            )
        else:
            frames = []
            for part in split_answer_chunks(outcome["answer"]):
                frames.extend(writer.text(part))
            frames.extend(
                await _answered(
                    dispatcher,
                    writer,
                    tenant_id,
                    query,
                    outcome["answer"],
                    outcome["payload"],
                )
            )
            settled = True
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
        settled = True
        await _save_unanswered(dispatcher, writer, tenant_id, query, agent_name)
        for frame in writer.failed(message, code):
            yield frame
    finally:
        if not settled:
            await _save_cancelled(dispatcher, writer, tenant_id, query, agent_name)


# Saves of cancelled runs still landing; held so none is collected mid-save.
_cancelled_saves: "set[asyncio.Task[None]]" = set()


async def _save_cancelled(
    dispatcher: Any,
    writer: "_RunWriter",
    tenant_id: str,
    query: str,
    agent_name: str,
) -> None:
    """Save a cancelled run's user message and its cancelled marker.

    The run's own task is being cancelled, so the save runs in a task of its
    own that the cancellation cannot interrupt; a save that fails is logged.
    """

    async def save() -> None:
        try:
            await dispatcher.record_cancelled_turn(
                tenant_id, thread_context_id(writer.thread_id), query
            )
        except Exception:
            logger.exception(
                "ag-ui thread %s: the cancelled %s run's message was not saved",
                writer.thread_id,
                agent_name,
            )

    task = asyncio.ensure_future(save())
    _cancelled_saves.add(task)
    task.add_done_callback(_cancelled_saves.discard)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # The run is still being cancelled; its save carries on regardless.
        pass


async def _answered(
    dispatcher: Any,
    writer: "_RunWriter",
    tenant_id: str,
    query: str,
    answer: str,
    payload: Optional[Dict[str, Any]],
) -> List[str]:
    """The answered run's closing frames, once its turn (the reply as
    delivered) has its place in the thread; a turn the ledger cannot place
    ends the run on ``RUN_ERROR``."""
    try:
        await dispatcher.record_conversation_turn(
            tenant_id, thread_context_id(writer.thread_id), query, {"answer": answer}
        )
    except Exception as exc:
        logger.exception("ag-ui thread %s: the turn was not saved", writer.thread_id)
        return writer.failed(
            "The reply was not saved to this conversation "
            f"({type(exc).__name__}). See server logs for detail.",
            "conversation_not_saved",
        )
    return writer.finished(payload=payload)


async def _save_unanswered(
    dispatcher: Any,
    writer: "_RunWriter",
    tenant_id: str,
    query: str,
    agent_name: str,
) -> None:
    """Save a failed run's user message to its thread; the run reports its
    own failure, so a save that fails too is logged."""
    try:
        await dispatcher.record_conversation_turn(
            tenant_id, thread_context_id(writer.thread_id), query, {}
        )
    except Exception:
        logger.exception(
            "ag-ui thread %s: the failed %s run's message was not saved",
            writer.thread_id,
            agent_name,
        )


_SPAN_ID = re.compile(r"[0-9a-f]{16}")


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
            readable_within_s=span_readable_within_s(manager.config.batch_config),
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


class SessionEvaluationRequest(BaseModel):
    outcome: Literal[SESSION_OUTCOMES]  # type: ignore[valid-type]
    score: float = Field(ge=0.0, le=1.0)
    span_ids: List[str] = Field(min_length=1, max_length=200)


@router.post("/threads/{thread_id}/evaluation")
async def evaluate_thread(
    thread_id: str,
    raw_request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Store a reviewer's verdict on one of the key's tenant's conversations:
    its outcome (success, partial or failure) and a 0-1 quality score, as a
    ``session_evaluation`` annotation on each search span of the conversation
    the request names. Answers the verdict and the spans it was written on.
    """
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)
    try:
        request = SessionEvaluationRequest.model_validate(await raw_request.json())
    except json.JSONDecodeError as exc:
        return error_response(
            400,
            f"Invalid session evaluation: body is not JSON ({exc})",
            "invalid_request",
        )
    except ValidationError as exc:
        return _invalid_body(exc, "session evaluation")
    malformed = [
        span_id for span_id in request.span_ids if not _SPAN_ID.fullmatch(span_id)
    ]
    if malformed:
        return error_response(
            400,
            f"Invalid session evaluation: span_ids {malformed} are not span ids",
            "invalid_request",
        )

    try:
        manager = get_telemetry_manager()
        project = manager.config.get_project_name(tenant_id)
        written = await persist_session_evaluation(
            manager.get_provider(tenant_id=tenant_id, project_name=project),
            project,
            thread_id,
            request.span_ids,
            request.outcome,
            request.score,
            readable_within_s=span_readable_within_s(manager.config.batch_config),
        )
    except SpanNotInProjectError as exc:
        return error_response(
            404,
            f"Conversation {thread_id} names search spans this tenant does not "
            f"have: {exc}.",
            "span_not_found",
        )
    except Exception as exc:
        logger.exception(
            "evaluation of conversation %s for tenant %s was not stored",
            thread_id,
            tenant_id,
        )
        return error_response(
            502,
            f"The evaluation of conversation {thread_id} was not stored "
            f"({type(exc).__name__}). See server logs for detail.",
            "annotation_not_stored",
            err_type="server_error",
            error_type=type(exc).__name__,
        )
    return {
        "thread_id": thread_id,
        "outcome": request.outcome,
        "score": request.score,
        "span_ids": written,
    }


@router.get("/threads/{thread_id}")
async def read_thread(
    thread_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """The saved turns of one of the key's tenant's threads, oldest first.

    ``state`` is ``incomplete`` (with its ``reason``) when a turn is still
    being saved or was lost; a thread with no saved turns answers none.
    """
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)
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
        history = await dispatcher.read_conversation(
            tenant_id, thread_context_id(thread_id)
        )
    except SessionStateUnavailable as exc:
        logger.warning("conversation ledger unavailable on /ag-ui: %s", exc)
        return dependency_unavailable(exc, "conversation ledger")
    except Exception as exc:
        logger.exception("ag-ui thread %s could not be read", thread_id)
        return dependency_unavailable(exc, "conversation store")
    return {
        "thread_id": thread_id,
        "state": history.state,
        "reason": history.reason,
        "turns": history.turns,
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
        parameters = run_parameters(run_input)
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
            parameters,
        ),
        media_type="text/event-stream",
    )
