"""OpenAI-compatible chat-completions surface for harness clients.

``POST /v1/chat/completions`` lets any client speaking the OpenAI dialect —
Pi's ``openai-completions`` provider in particular — drive cogniverse agents
as its "model". Bearer keys map to canonical tenants, model names map to
agent names (``cogniverse`` routes through the gateway agent,
``cogniverse/<profile>`` pins one agent).

Module-level DI mirrors routers/agents.py: main.py's lifespan injects the
static api-key map (env-resolved at the entrypoint), the model map, the
dynamic key resolver and a dispatcher provider; tests wire their own.

Clients replay the full transcript on every call, so each request is a
self-contained turn and no server-side conversation state is required. The
per-conversation seed comes from the transcript's first user message, so
canary/variant bucketing stays sticky across the turns of one conversation.

Dual loop: a request may carry OpenAI ``tools`` definitions for the client's
local tools. An agent returning ``pending_tool_calls`` suspends the turn —
the response carries ``tool_calls`` and ``finish_reason: "tool_calls"``; the
client executes locally and replays the transcript with the tool results
appended, which resumes the turn on whichever process receives it. Suspended
``continuation_state`` is kept in the shared Redis ``ContinuationStore``,
keyed by tenant, agent, seed and call ids, for ``CONTINUATION_TTL_SECONDS``; a
miss (expired, already resumed, another tenant) degrades to stateless
re-derivation from the replayed transcript, never another tenant's state. A
store that does not answer fails the turn with 503.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import aclosing, contextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from cogniverse_core.agents.base import leaf_exceptions
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.agent_registry import AgentRegistryUnavailableError
from cogniverse_runtime.harness_keys import HarnessKeyNotFoundError
from cogniverse_runtime.harness_turn import (
    NoAnswerError,
    derive_request_seed,
    extract_answer_text,
    is_answer_field,
    to_openai_tool_calls,
)
from cogniverse_runtime.llm_dependency import llm_dependency_failure
from cogniverse_runtime.session_state import ContinuationStore, SessionStateUnavailable
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

__all__ = ["derive_request_seed", "extract_answer_text", "to_openai_tool_calls"]

logger = logging.getLogger(__name__)

router = APIRouter()

_dispatcher_provider: Optional[Callable[[], Any]] = None
_api_keys: Dict[str, str] = {}
_model_map: Dict[str, str] = {}
_key_resolver: Optional[Callable[[str], Optional[str]]] = None
_continuation_store: Optional[ContinuationStore] = None
_in_flight: set = set()

_CONTENT_CHUNK_CHARS = 256

# How often the non-streamed handler asks the transport whether the client is
# still there; the turn is cancelled within one interval of the hang-up.
_DISCONNECT_POLL_SECONDS = 0.05

_MAX_TEMPERATURE = 2.0


class RequestShapeError(ValueError):
    """The request body violates the OpenAI chat-completions shape."""


class ToolsForbiddenError(RuntimeError):
    """The agent asked for a tool call on a turn that forbade tool calls."""


def set_dispatcher_provider(provider: Optional[Callable[[], Any]]) -> None:
    global _dispatcher_provider
    _dispatcher_provider = provider


def set_api_keys(mapping: Optional[Dict[str, str]]) -> None:
    global _api_keys
    _api_keys = dict(mapping or {})


def set_model_map(mapping: Optional[Dict[str, str]]) -> None:
    global _model_map
    _model_map = dict(mapping or {})


def set_key_resolver(resolver: Optional[Callable[[str], Optional[str]]]) -> None:
    """Dynamic key->tenant lookup (``HarnessKeyStore.resolve``), consulted
    after the static env map. None disables the dynamic path."""
    global _key_resolver
    _key_resolver = resolver


def set_continuation_store(store: Optional[ContinuationStore]) -> None:
    """The shared store suspended turns wait in for their tool results."""
    global _continuation_store
    _continuation_store = store


def require_continuation_store() -> ContinuationStore:
    if _continuation_store is None:
        raise SessionStateUnavailable(
            "a turn with tool calls needs the shared continuation store, and "
            "none is configured"
        )
    return _continuation_store


def in_flight_count() -> int:
    """Number of turns currently executing (streamed and non-streamed)."""
    return len(_in_flight)


@contextmanager
def in_flight_turn() -> Iterator[None]:
    """Count the enclosed turn in ``in_flight_count`` while it runs."""
    handle = object()
    _in_flight.add(handle)
    try:
        yield
    finally:
        _in_flight.discard(handle)


class DispatcherNotReady(RuntimeError):
    """The runtime has not wired or built its dispatcher yet."""


def current_dispatcher() -> Any:
    """The dispatcher turns run on.

    Raises:
        DispatcherNotReady: no provider is wired, or it has not built one.
        Exception: whatever the provider raised building it.
    """
    if _dispatcher_provider is None:
        raise DispatcherNotReady("Runtime initialising; dispatcher not wired.")
    dispatcher = _dispatcher_provider()
    if dispatcher is None:
        raise DispatcherNotReady("Runtime initialising; dispatcher not built yet.")
    return dispatcher


def resolve_tenant(api_key: str) -> Optional[str]:
    """Canonical tenant id for a bearer key; None when the key is unknown.

    Both key sources are canonicalized here and nowhere else, so a static map
    entry written as ``default`` and a stored key written as ``org:tenant``
    reach dispatch in the one storage form the rest of the stack uses.

    Raises:
        ConfigStoreUnavailableError: the key store could not be read. An
            outage is a 503, never a 401 — a client's correct reaction to a
            401 is to stop and ask its user for a new key.
    """
    static = _api_keys.get(api_key)
    if static is not None:
        return canonical_tenant_id(static)
    if _key_resolver is None:
        return None
    try:
        resolved = _key_resolver(api_key)
    except HarnessKeyNotFoundError:
        return None
    if not resolved:
        return None
    return canonical_tenant_id(resolved)


def resolve_agent(model: str) -> Optional[str]:
    return _model_map.get(model)


_IMAGE_PART_TYPES = ("image_url", "input_image")


def _image_part_url(part: Dict[str, Any], where: str) -> str:
    """URL of an OpenAI image part.

    Raises:
        RequestShapeError: the part carries no usable url.
    """
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        raise RequestShapeError(
            f"{where} is an image part with no url string; expected "
            "image_url as a string or as an object with a 'url' string"
        )
    return url


def _content_text(content: Any, *, where: str) -> str:
    """Normalize OpenAI message content to plain text.

    The dialect allows a string or an array of content parts. ``text`` parts
    join into the text; image parts are attachments handled separately (see
    ``extract_attachments``) and contribute no text; anything else is
    rejected naming its position.

    Raises:
        RequestShapeError: a part is not an object, a text part's ``text`` is
            absent or not a string, an image part has no url, or the content
            is neither a string nor a list.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestShapeError(
            f"{where} content must be a string or an array of parts, got "
            f"{type(content).__name__}"
        )
    texts: List[str] = []
    for index, part in enumerate(content):
        position = f"{where} part {index}"
        if not isinstance(part, dict):
            raise RequestShapeError(
                f"{position} must be an object, got {type(part).__name__}"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise RequestShapeError(
                    f"{position} has text of type {type(text).__name__}; "
                    "text parts require a string"
                )
            texts.append(text)
        elif part_type in _IMAGE_PART_TYPES:
            _image_part_url(part, position)
        else:
            raise RequestShapeError(
                f"{position} has unsupported type {part_type!r}; only text "
                "and image parts are accepted"
            )
    return "".join(texts)


def extract_attachments(messages: List[Dict[str, Any]]) -> List[str]:
    """Image URIs attached to the current turn.

    Only the last user message's image parts are returned — that is the turn's
    fresh attachments; earlier turns' images are not replayed.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        content = messages[index].get("content")
        if not isinstance(content, list):
            return []
        where = f"messages[{index}]"
        return [
            _image_part_url(part, f"{where} part {position}")
            for position, part in enumerate(content)
            if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES
        ]
    return []


def _validate_tool_calls(entry: Dict[str, Any], where: str) -> List[Dict[str, Any]]:
    """The assistant message's ``tool_calls``, shape-checked.

    Raises:
        RequestShapeError: ``tool_calls`` is not a non-empty array of objects
            each carrying a string ``id`` and a ``function.name`` string, or
            an ``arguments`` value that is not a JSON-encoded string.
    """
    tool_calls = entry.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise RequestShapeError(
            f"{where} must carry tool_calls as a non-empty array, got "
            f"{type(tool_calls).__name__}"
        )
    for index, call in enumerate(tool_calls):
        position = f"{where} tool_calls[{index}]"
        if not isinstance(call, dict):
            raise RequestShapeError(
                f"{position} must be an object, got {type(call).__name__}"
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise RequestShapeError(f"{position} requires a non-empty string id")
        function = call.get("function")
        if not isinstance(function, dict):
            raise RequestShapeError(
                f"{position} requires a function object, got {type(function).__name__}"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RequestShapeError(
                f"{position} requires a non-empty string function.name"
            )
        if "arguments" in function:
            arguments = function["arguments"]
            if not isinstance(arguments, str):
                raise RequestShapeError(
                    f"{position} function.arguments must be a JSON string, got "
                    f"{type(arguments).__name__}"
                )
            try:
                json.loads(arguments)
            except ValueError as exc:
                raise RequestShapeError(
                    f"{position} function.arguments is not valid JSON: {exc}"
                ) from exc
    return list(tool_calls)


def _match_results_to_calls(
    round_index: int, tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> None:
    """Every replayed tool result answers exactly one call of its round.

    Raises:
        RequestShapeError: a duplicated call id, a result whose
            ``tool_call_id`` matches no call in the round, a duplicated
            result, or a call left unanswered — each naming the ids.
    """
    where = f"tool round {round_index}"
    call_ids: List[str] = [call["id"] for call in tool_calls]
    duplicates = sorted({cid for cid in call_ids if call_ids.count(cid) > 1})
    if duplicates:
        raise RequestShapeError(f"{where} repeats tool_call id(s) {duplicates}")

    seen: List[str] = []
    for result in results:
        result_id = result.get("tool_call_id")
        if not isinstance(result_id, str) or not result_id:
            raise RequestShapeError(
                f"{where} has a tool result with no tool_call_id string"
            )
        if result_id not in call_ids:
            raise RequestShapeError(
                f"{where} has a tool result for unmatched id {result_id!r}; "
                f"the round called {sorted(call_ids)}"
            )
        if result_id in seen:
            raise RequestShapeError(
                f"{where} repeats a tool result for id {result_id!r}"
            )
        seen.append(result_id)

    unanswered = sorted(set(call_ids) - set(seen))
    if unanswered:
        raise RequestShapeError(f"{where} has no tool result for id(s) {unanswered}")


def build_dispatch_args(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map an OpenAI ``messages`` array onto dispatcher inputs.

    The last user message becomes the query; everything before it is the
    conversation history the dispatcher already understands. Messages after
    the last user message are the in-flight turn's tool exchange: an assistant
    message carrying ``tool_calls`` followed by the ``role: "tool"`` results
    the client produced for them.

    A user message whose only content is an image is a valid turn: the query
    is empty and the image travels as an attachment.

    Raises:
        RequestShapeError: no user message, a last user message carrying
            neither text nor an image, or a post-user tail that is not a
            well-formed assistant-tool_calls + tool-results exchange.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestShapeError("messages must be a non-empty array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestShapeError(
                f"messages[{index}] must be an object, got {type(message).__name__}"
            )

    last_user = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user = index
            break
    if last_user is None:
        raise RequestShapeError("messages must contain at least one user message")

    content = _content_text(
        messages[last_user].get("content"), where=f"messages[{last_user}]"
    ).strip()
    attachments = extract_attachments(messages)
    if not content and not attachments:
        raise RequestShapeError(
            f"messages[{last_user}] is the turn's user message and carries "
            "neither text nor an image part"
        )

    history = [
        {
            "role": str(message.get("role")),
            "content": _content_text(
                message.get("content"), where=f"messages[{index}]"
            ),
        }
        for index, message in enumerate(messages[:last_user])
    ]

    tool_exchange: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for offset, entry in enumerate(messages[last_user + 1 :]):
        index = last_user + 1 + offset
        where = f"messages[{index}]"
        role = entry.get("role")
        if role == "assistant":
            if current is not None:
                _match_results_to_calls(
                    len(tool_exchange) - 1, current["tool_calls"], current["results"]
                )
            current = {"tool_calls": _validate_tool_calls(entry, where), "results": []}
            tool_exchange.append(current)
        elif role == "tool":
            if current is None:
                raise RequestShapeError(
                    f"{where} is a tool result but no assistant tool_calls "
                    "message precedes it in this turn"
                )
            current["results"].append(
                {
                    "tool_call_id": entry.get("tool_call_id"),
                    "content": _content_text(entry.get("content"), where=where),
                }
            )
        else:
            raise RequestShapeError(
                f"{where} has role {role!r}; only assistant tool_calls and "
                "tool results may follow the last user message"
            )
    if current is not None:
        _match_results_to_calls(
            len(tool_exchange) - 1, current["tool_calls"], current["results"]
        )

    all_ids = [call["id"] for round_ in tool_exchange for call in round_["tool_calls"]]
    reused = sorted({cid for cid in all_ids if all_ids.count(cid) > 1})
    if reused:
        raise RequestShapeError(f"tool_call id(s) {reused} reused across rounds")

    last_round = tool_exchange[-1] if tool_exchange else None
    return {
        "query": content,
        "attachments": attachments,
        "conversation_history": history,
        "assistant_tool_calls": list(last_round["tool_calls"]) if last_round else [],
        "tool_results": list(last_round["results"]) if last_round else [],
        "tool_exchange": tool_exchange,
    }


def split_answer_chunks(
    text: str, chunk_chars: int = _CONTENT_CHUNK_CHARS
) -> List[str]:
    """Split answer text into stream deltas; concatenation is the identity."""
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _usage_from_tracker(tracker: Any) -> Tuple[int, int]:
    """(prompt_tokens, completion_tokens) summed across tracked LM calls.

    DSPy's ``UsageTracker.usage_data`` is ``{lm_name: [{prompt_tokens,
    completion_tokens, ...}, ...]}``. Cache hits never reach the tracker, so
    zeros are possible and the caller fills gaps from the estimate.
    """
    if tracker is None or not hasattr(tracker, "usage_data"):
        return 0, 0
    prompt = completion = 0
    for entries in tracker.usage_data.values():
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            prompt_tokens = entry.get("prompt_tokens")
            completion_tokens = entry.get("completion_tokens")
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                prompt += prompt_tokens
            if isinstance(completion_tokens, int) and completion_tokens > 0:
                completion += completion_tokens
    return prompt, completion


def _estimate_usage(
    query: str, history: List[Dict[str, Any]], answer: str
) -> Tuple[int, int]:
    """4-chars-per-token estimate for turns resolved from the DSPy cache.

    Mirrors the never-zero convention of RLMInference token reporting so
    clients that require usage (Pi does) always get plausible integers.
    """
    prompt_chars = len(query) + sum(len(str(m.get("content", ""))) for m in history)
    return max(1, prompt_chars // 4), max(1, len(answer) // 4)


def _finalize_usage(
    tracker: Any,
    query: str,
    history: List[Dict[str, Any]],
    completion_text: str,
) -> Dict[str, int]:
    prompt_tokens, completion_tokens = _usage_from_tracker(tracker)
    estimated_prompt, estimated_completion = _estimate_usage(
        query, history, completion_text
    )
    if prompt_tokens <= 0:
        prompt_tokens = estimated_prompt
    if completion_tokens <= 0:
        completion_tokens = estimated_completion
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def error_response(
    status_code: int,
    message: str,
    code: str,
    err_type: str = "invalid_request_error",
    headers: Optional[Dict[str, str]] = None,
    agent: Optional[str] = None,
    error_type: Optional[str] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {"message": message, "type": err_type, "code": code}
    if agent is not None:
        body["agent"] = agent
    if error_type is not None:
        body["error_type"] = error_type
    return JSONResponse(
        status_code=status_code,
        content={"error": body},
        headers=headers,
    )


class StreamedAnswerDiverged(RuntimeError):
    """The final answer does not continue the text already streamed.

    The client holds a prefix the completed answer contradicts, so the turn
    cannot be finished as a reply.
    """


def failure_body(exc: BaseException, agent: str) -> Dict[str, str]:
    """What a ``/v1`` client is told about a failure it did not cause.

    The exception text stays server-side: a backend error carries the URL it
    could not reach, and that URL carries credentials. The client gets the
    agent and the leaf exception type, which is what the A2A surface and the
    agent layer already emit, and enough to route the report.
    """
    leaves = leaf_exceptions(exc)
    leaf_names: List[str] = []
    for leaf in leaves:
        if type(leaf).__name__ not in leaf_names:
            leaf_names.append(type(leaf).__name__)
    return {
        "message": (
            f"{agent} failed with {'; '.join(leaf_names)}. See server logs for detail."
        ),
        "agent": agent,
        "error_type": type(leaves[0]).__name__,
    }


UNAUTHORIZED = dict(
    status_code=401,
    message="Invalid or missing API key.",
    code="invalid_api_key",
    headers={"WWW-Authenticate": "Bearer"},
)


def _unavailable(message: str) -> JSONResponse:
    return error_response(503, message, "service_unavailable", err_type="server_error")


def dependency_unavailable(exc: BaseException, dependency: str) -> JSONResponse:
    """503 for a dependency this replica could not reach.

    The client is told which dependency and which exception type; the text,
    which carries the backend URL and its credentials, stays in the log.
    """
    error_type = type(leaf_exceptions(exc)[0]).__name__
    return error_response(
        503,
        f"The {dependency} is unavailable ({error_type}). See server logs for detail.",
        "service_unavailable",
        err_type="server_error",
        error_type=error_type,
    )


class ChatCompletionRequest(BaseModel):
    """The subset of the OpenAI request this surface acts on.

    ``extra="allow"`` keeps an unknown field from 422-ing a client; every
    field that changes what the agent does is declared here and forwarded.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[Dict[str, Any]] = None


def usage_requested(request: ChatCompletionRequest) -> bool:
    """Whether a streamed turn ends with a usage chunk.

    The dialect sends usage on a stream only when the client asks with
    ``stream_options: {"include_usage": true}``, as a last chunk whose
    ``choices`` is empty.

    Raises:
        RequestShapeError: ``include_usage`` is set to something other than a
            boolean.
    """
    include = (request.stream_options or {}).get("include_usage")
    if include is None:
        return False
    if not isinstance(include, bool):
        raise RequestShapeError(
            f"stream_options.include_usage must be a boolean, got {include!r}"
        )
    return include


def sampling_context(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Sampling parameters to place on the dispatch context.

    ``max_completion_tokens`` is the current spelling of ``max_tokens``; a
    request carrying both must agree.

    Raises:
        RequestShapeError: a parameter is outside the range the dialect
            defines, or the two token ceilings disagree.
    """
    forwarded: Dict[str, Any] = {}
    temperature = request.temperature
    if temperature is not None:
        if not 0.0 <= temperature <= _MAX_TEMPERATURE:
            raise RequestShapeError(
                f"temperature must be between 0 and {_MAX_TEMPERATURE}, got "
                f"{temperature}"
            )
        forwarded["temperature"] = temperature

    ceilings = {
        name: value
        for name, value in (
            ("max_tokens", request.max_tokens),
            ("max_completion_tokens", request.max_completion_tokens),
        )
        if value is not None
    }
    for name, value in ceilings.items():
        if value < 1:
            raise RequestShapeError(f"{name} must be at least 1, got {value}")
    if len(set(ceilings.values())) > 1:
        raise RequestShapeError(
            f"max_tokens and max_completion_tokens disagree: {ceilings}"
        )
    if ceilings:
        forwarded["max_tokens"] = next(iter(ceilings.values()))
    return forwarded


def resolve_tool_policy(
    request: ChatCompletionRequest,
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """The tools to forward, and whether the client forbade tool calls.

    ``"none"`` withholds the definitions AND makes a tool request from the
    agent an error, so a client that said "not this turn" can never be
    answered with ``finish_reason: "tool_calls"``. Forcing a call
    (``"required"``, a named function) is rejected: nothing on this surface
    can compel an agent to call one, and accepting the field would report a
    guarantee that does not exist.

    Raises:
        RequestShapeError: a tool_choice this surface cannot honour.
    """
    choice = request.tool_choice
    if choice is None or choice == "auto":
        return request.tools, False
    if choice == "none":
        return None, True
    raise RequestShapeError(
        f"tool_choice {choice!r} is not supported; this surface accepts "
        '"auto" and "none"'
    )


def build_dispatch_context(
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    seed: str,
    external_tools: Optional[List[Dict[str, Any]]],
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    """The context one turn hands the dispatcher."""
    context: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "conversation_history": dispatch_args["conversation_history"],
        "request_id": seed,
        **sampling,
    }
    if dispatch_args.get("attachments"):
        context["attachments"] = dispatch_args["attachments"]
    if external_tools:
        context["external_tools"] = external_tools
    if dispatch_args.get("tool_results"):
        context["tool_results"] = dispatch_args["tool_results"]
        context["assistant_tool_calls"] = dispatch_args["assistant_tool_calls"]
        context["tool_exchange"] = dispatch_args["tool_exchange"]
    return context


async def run_turn(
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    tools_forbidden: bool = False,
) -> Dict[str, Any]:
    """Execute one dispatch turn.

    Returns ``{"kind": "answer", "answer": str, "usage": {...}}`` for a
    completed turn, or ``{"kind": "tool_calls", "tool_calls": [...],
    "usage": {...}}`` when the agent suspends on external tool calls (OpenAI
    message.tool_calls shape, arguments JSON-encoded).
    """
    from dspy.utils.usage_tracker import track_usage

    query = dispatch_args["query"]
    history = dispatch_args["conversation_history"]
    seed = derive_request_seed(query, history)
    context = build_dispatch_context(
        dispatch_args, tenant_id, seed, external_tools, sampling or {}
    )

    if context.get("tool_results"):
        state = await require_continuation_store().pop(
            tenant_id,
            agent_name,
            seed,
            [call.get("id") for call in context["assistant_tool_calls"]],
        )
        if state is not None:
            context["continuation_state"] = state

    with track_usage() as tracker:
        result = await dispatcher.dispatch(
            agent_name=agent_name, query=query, context=context
        )

    pending = result.get("pending_tool_calls") if isinstance(result, dict) else None
    if pending:
        if tools_forbidden:
            raise ToolsForbiddenError(
                f"Agent '{agent_name}' requested {len(pending)} tool call(s) on a "
                'turn sent with tool_choice "none"'
            )
        tool_calls = to_openai_tool_calls(pending)
        state = result.get("continuation_state")
        if isinstance(state, dict) and state:
            await require_continuation_store().put(
                tenant_id, agent_name, seed, [c["id"] for c in tool_calls], state
            )
        usage = _finalize_usage(tracker, query, history, json.dumps(tool_calls))
        return {"kind": "tool_calls", "tool_calls": tool_calls, "usage": usage}

    answer = extract_answer_text(result)
    usage = _finalize_usage(tracker, query, history, answer)
    return {"kind": "answer", "answer": answer, "usage": usage, "payload": result}


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    choices: List[Dict[str, Any]],
    usage: Optional[Dict[str, int]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return _sse(payload)


def _error_frame(exc: BaseException, agent_name: str) -> str:
    unavailable = isinstance(exc, SessionStateUnavailable)
    return _sse(
        {
            "error": {
                **failure_body(exc, agent_name),
                "type": "server_error",
                "code": "service_unavailable" if unavailable else "internal_error",
            }
        }
    )


_CANCELLED_FRAME = _sse(
    {
        "error": {
            "message": "Stream cancelled before the turn completed.",
            "type": "server_error",
            "code": "stream_cancelled",
        }
    }
)


def _terminal_frames_on_cancel(agent_name: str) -> List[str]:
    """The frames a cancelled stream owes its client.

    A rolling restart cancels the response task at the generator's yield
    point; the writes issued from the except block still reach the socket, so
    the client ends on an error frame and ``[DONE]`` instead of a body that
    simply stops. A client that already left never reads them.
    """
    logger.info("chat.completions stream for %s cancelled", agent_name)
    return [_CANCELLED_FRAME, "data: [DONE]\n\n"]


async def _stream_turn(
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    completion_id: str,
    created: int,
    model: str,
    external_tools: Optional[List[Dict[str, Any]]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    tools_forbidden: bool = False,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Stream one turn as SSE by chunking the finished answer."""
    task = asyncio.create_task(
        run_turn(
            dispatcher,
            agent_name,
            dispatch_args,
            tenant_id,
            external_tools,
            sampling,
            tools_forbidden,
        )
    )
    _in_flight.add(task)
    try:
        yield _chunk(
            completion_id,
            created,
            model,
            [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        )
        outcome = await task
        if outcome["kind"] == "tool_calls":
            yield _chunk(
                completion_id,
                created,
                model,
                [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": position, **call}
                                for position, call in enumerate(outcome["tool_calls"])
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            )
            yield _chunk(
                completion_id,
                created,
                model,
                [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            )
            if include_usage:
                yield _chunk(completion_id, created, model, [], usage=outcome["usage"])
            yield "data: [DONE]\n\n"
            return
        for part in split_answer_chunks(outcome["answer"]):
            yield _chunk(
                completion_id,
                created,
                model,
                [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
            )
        yield _chunk(
            completion_id,
            created,
            model,
            [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        if include_usage:
            yield _chunk(completion_id, created, model, [], usage=outcome["usage"])
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        for frame in _terminal_frames_on_cancel(agent_name):
            yield frame
        raise
    except Exception as exc:
        logger.exception("chat.completions turn failed mid-stream")
        yield _error_frame(exc, agent_name)
        yield "data: [DONE]\n\n"
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass
        _in_flight.discard(task)


def use_token_stream(dispatcher: Any, agent_name: str, has_tool_results: bool) -> bool:
    """Whether a streamed request takes the live-token path.

    Only for an agent whose endpoint declares ``streams_answer_tokens`` and
    only on a turn that is not resuming a tool exchange: a resume carries the
    replayed results and the continuation state, which the dispatch path owns.
    A first turn that merely *offers* tools still streams tokens — the agent
    may answer without calling one, and if it does call one the final event
    carries the pending calls.
    """
    if has_tool_results:
        return False
    return dispatcher.supports_token_stream(agent_name)


async def answer_token_events(
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    external_tools: Optional[List[Dict[str, Any]]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    tools_forbidden: bool = False,
) -> AsyncIterator[Dict[str, Any]]:
    """The token path of one turn as surface-neutral events.

    Yields, in order of arrival:

    - ``{"kind": "status", "phase", "message"}`` for each progress event that
      is not an answer token;
    - ``{"kind": "text", "delta"}`` for each piece of reply text;
    - exactly one terminal event: ``{"kind": "answer", "text", "payload"}``,
      ``{"kind": "tool_calls", "tool_calls", "payload"}`` (OpenAI
      ``message.tool_calls`` shape, its continuation already stored), or
      ``{"kind": "error", "message", "agent", "error_type"}``.

    An agent emits a token event per streamed field, so only the field that
    carries the answer (``is_answer_field``) becomes text, and the first such
    field seen is the one followed: a research agent's decomposition and gap
    list travel on the same channel as its summary and are not the reply.

    Whatever was streamed is reconciled against the answer of the final
    payload, so the deltas of a streamed turn always concatenate to the answer
    of the same turn served non-streamed.

    Raises:
        ToolsForbiddenError: the agent asked for tool calls on a turn that
            forbade them.
        StreamedAnswerDiverged: the final answer does not continue the text
            already streamed.
        SessionStateUnavailable: a suspended turn could not store its
            continuation.
    """
    query = dispatch_args["query"]
    history = dispatch_args["conversation_history"]
    seed = derive_request_seed(query, history)
    context = build_dispatch_context(
        dispatch_args, tenant_id, seed, external_tools, sampling or {}
    )
    streamed = ""
    answer_field: Optional[str] = None
    async with aclosing(
        dispatcher.dispatch_stream(agent_name, query, context)
    ) as events:
        async for event in events:
            if event.get("type") == "error":
                # The agent layer already withheld the exception text; carry
                # the identity it named through unchanged.
                yield {
                    "kind": "error",
                    "message": str(event.get("message", "")),
                    "agent": str(event.get("agent", agent_name)),
                    "error_type": str(event.get("error_type", "")),
                }
                return
            if event.get("phase") == "token":
                field = (event.get("data") or {}).get("output_field")
                if not isinstance(field, str) or not is_answer_field(field):
                    continue
                if answer_field is None:
                    answer_field = field
                elif field != answer_field:
                    continue
                delta = event.get("message") or ""
                if delta:
                    streamed += delta
                    yield {"kind": "text", "delta": delta}
                continue
            if event.get("type") in ("status", "partial"):
                yield {
                    "kind": "status",
                    "phase": str(event.get("phase", "")),
                    "message": str(event.get("message", "")),
                }
                continue
            if event.get("type") != "final":
                continue

            payload = event.get("data") or {}
            pending = payload.get("pending_tool_calls")
            if pending:
                if tools_forbidden:
                    raise ToolsForbiddenError(
                        f"Agent '{agent_name}' requested {len(pending)} tool "
                        'call(s) on a turn sent with tool_choice "none"'
                    )
                tool_calls = to_openai_tool_calls(pending)
                state = payload.get("continuation_state")
                if isinstance(state, dict) and state:
                    await require_continuation_store().put(
                        tenant_id,
                        agent_name,
                        seed,
                        [call["id"] for call in tool_calls],
                        state,
                    )
                yield {
                    "kind": "tool_calls",
                    "tool_calls": tool_calls,
                    "payload": payload,
                }
                return

            answer = extract_answer_text(payload)
            if not streamed:
                if answer:
                    streamed = answer
                    yield {"kind": "text", "delta": answer}
            elif answer != streamed:
                if not answer.startswith(streamed):
                    raise StreamedAnswerDiverged(
                        f"Agent '{agent_name}' streamed {len(streamed)} "
                        f"characters of {answer_field!r} that its final "
                        "answer does not begin with; the streamed reply "
                        "cannot be completed"
                    )
                remainder = answer[len(streamed) :]
                streamed = answer
                yield {"kind": "text", "delta": remainder}
            yield {"kind": "answer", "text": streamed, "payload": payload}
            return
    yield {"kind": "answer", "text": streamed, "payload": None}


async def _stream_tokens(
    dispatcher: Any,
    agent_name: str,
    dispatch_args: Dict[str, Any],
    tenant_id: str,
    completion_id: str,
    created: int,
    model: str,
    external_tools: Optional[List[Dict[str, Any]]] = None,
    sampling: Optional[Dict[str, Any]] = None,
    tools_forbidden: bool = False,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Stream the agent's answer tokens as they are produced."""
    query = dispatch_args["query"]
    history = dispatch_args["conversation_history"]
    handle = object()
    _in_flight.add(handle)

    def content(text: str) -> str:
        return _chunk(
            completion_id,
            created,
            model,
            [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        )

    try:
        yield _chunk(
            completion_id,
            created,
            model,
            [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        )
        streamed = ""
        async with aclosing(
            answer_token_events(
                dispatcher,
                agent_name,
                dispatch_args,
                tenant_id,
                external_tools,
                sampling,
                tools_forbidden,
            )
        ) as events:
            async for event in events:
                kind = event["kind"]
                if kind == "text":
                    yield content(event["delta"])
                elif kind == "error":
                    yield _sse(
                        {
                            "error": {
                                "message": event["message"],
                                "type": "server_error",
                                "code": "internal_error",
                                "agent": event["agent"],
                                "error_type": event["error_type"],
                            }
                        }
                    )
                    yield "data: [DONE]\n\n"
                    return
                elif kind == "tool_calls":
                    tool_calls = event["tool_calls"]
                    yield _chunk(
                        completion_id,
                        created,
                        model,
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": position, **call}
                                        for position, call in enumerate(tool_calls)
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    )
                    yield _chunk(
                        completion_id,
                        created,
                        model,
                        [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    )
                    if include_usage:
                        yield _chunk(
                            completion_id,
                            created,
                            model,
                            [],
                            usage=_finalize_usage(
                                None, query, history, json.dumps(tool_calls)
                            ),
                        )
                    yield "data: [DONE]\n\n"
                    return
                elif kind == "answer":
                    streamed = event["text"]
        yield _chunk(
            completion_id,
            created,
            model,
            [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        if include_usage:
            yield _chunk(
                completion_id,
                created,
                model,
                [],
                usage=_finalize_usage(None, query, history, streamed),
            )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        for frame in _terminal_frames_on_cancel(agent_name):
            yield frame
        raise
    except Exception as exc:
        logger.exception("chat.completions token stream failed mid-stream")
        yield _error_frame(exc, agent_name)
        yield "data: [DONE]\n\n"
    finally:
        _in_flight.discard(handle)


async def _watch_disconnect(request: Request, stop: asyncio.Event) -> None:
    """Return once the client has hung up, or as soon as ``stop`` is set.

    The watcher is retired through ``stop`` rather than ``Task.cancel``:
    ``Request.is_disconnected`` polls the receive channel inside a cancel
    scope that absorbs whatever cancellation arrives while it is open, so a
    cancel aimed at this task can be swallowed and the task would outlive the
    request.
    """
    while not stop.is_set():
        if await request.is_disconnected():
            return
        try:
            await asyncio.wait_for(stop.wait(), _DISCONNECT_POLL_SECONDS)
        except TimeoutError:
            continue


async def run_turn_until_disconnect(
    request: Request, turn: "asyncio.Task[Dict[str, Any]]"
) -> Optional[Dict[str, Any]]:
    """Await ``turn``, cancelling it if the client hangs up first.

    Returns the turn's outcome, or None when the client left — the LM calls,
    the sandbox execution and any mutating tool sequence behind the turn stop
    with it rather than running on for a response nobody will read.
    """
    stop = asyncio.Event()
    watcher = asyncio.create_task(_watch_disconnect(request, stop))
    try:
        done, _ = await asyncio.wait(
            {turn, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if turn in done:
            return turn.result()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        return None
    finally:
        stop.set()
        await asyncio.gather(watcher, return_exceptions=True)


async def resolve_tenant_off_loop(authorization: Optional[str]) -> Optional[str]:
    """Resolve the bearer key to a canonical tenant; None means reject.

    The dynamic resolver does a blocking backend read, so it runs off the
    event loop and cannot stall other in-flight turns.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    key = authorization[len("Bearer ") :].strip()
    if key in _api_keys:
        return resolve_tenant(key)
    return await asyncio.to_thread(resolve_tenant, key)


@router.get("/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    """The configured model map in OpenAI list form, in configuration order."""
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /v1/models: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)
    created = int(time.time())
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "cogniverse",
                }
                for model_id in _model_map
            ],
        }
    )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    authorization: Optional[str] = Header(default=None),
):
    try:
        tenant_id = await resolve_tenant_off_loop(authorization)
    except ConfigStoreUnavailableError as exc:
        logger.warning("harness key store unavailable on /v1/chat: %s", exc)
        return dependency_unavailable(exc, "harness key store")
    if tenant_id is None:
        return error_response(**UNAUTHORIZED)

    agent_name = resolve_agent(request.model)
    if agent_name is None:
        return error_response(
            404,
            f"Model '{request.model}' does not exist or you do not have access to it.",
            "model_not_found",
        )

    try:
        dispatch_args = build_dispatch_args(request.messages)
        sampling = sampling_context(request)
        external_tools, tools_forbidden = resolve_tool_policy(request)
        include_usage = usage_requested(request)
    except RequestShapeError as exc:
        return error_response(400, str(exc), "invalid_request")

    try:
        dispatcher = current_dispatcher()
    except DispatcherNotReady as exc:
        return _unavailable(str(exc))
    except Exception as exc:
        logger.exception("dispatcher provider failed")
        return dependency_unavailable(exc, "dispatcher")
    try:
        await dispatcher.refresh_agent_registry()
    except AgentRegistryUnavailableError as exc:
        logger.warning("agent registry unavailable on /v1/chat: %s", exc)
        return dependency_unavailable(exc, "agent registry")

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        stream = (
            _stream_tokens
            if use_token_stream(
                dispatcher, agent_name, bool(dispatch_args["tool_results"])
            )
            else _stream_turn
        )
        return StreamingResponse(
            stream(
                dispatcher,
                agent_name,
                dispatch_args,
                tenant_id,
                completion_id,
                created,
                request.model,
                external_tools,
                sampling,
                tools_forbidden,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    task = asyncio.create_task(
        run_turn(
            dispatcher,
            agent_name,
            dispatch_args,
            tenant_id,
            external_tools,
            sampling,
            tools_forbidden,
        )
    )
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)
    try:
        outcome = await run_turn_until_disconnect(raw_request, task)
    except ToolsForbiddenError as exc:
        logger.warning("chat.completions turn ignored tool_choice none: %s", exc)
        return error_response(
            502, str(exc), "tool_choice_violation", err_type="server_error"
        )
    except NoAnswerError as exc:
        # The agent's reported detail can carry a backend's error text; it
        # stays in the log.
        logger.warning("chat.completions turn produced no answer: %s", exc)
        return error_response(
            502,
            f"Agent '{agent_name}' finished without an answer to return "
            f"(status={exc.status}).",
            "upstream_no_answer",
            err_type="server_error",
            agent=agent_name,
            error_type=type(exc).__name__,
        )
    except SessionStateUnavailable as exc:
        logger.warning("chat.completions turn lost its session state: %s", exc)
        return dependency_unavailable(exc, "session state store")
    except Exception as exc:
        llm_failure = llm_dependency_failure(exc)
        if llm_failure is not None:
            logger.warning("chat.completions turn failed on the chat LLM: %s", exc)
            return error_response(
                llm_failure.http_status,
                llm_failure.message(agent_name),
                llm_failure.error,
                err_type="server_error",
                headers=llm_failure.headers(),
                agent=agent_name,
                error_type=llm_failure.failure,
            )
        # Message validation already returned 400 above; anything raised inside
        # the dispatch turn (including a ValueError from agent-input
        # validation) is a server-side failure, not a bad model name.
        logger.exception("chat.completions turn failed")
        return error_response(
            500,
            code="internal_error",
            err_type="server_error",
            **failure_body(exc, agent_name),
        )

    if outcome is None:
        logger.info(
            "chat.completions client for %s disconnected; turn cancelled", agent_name
        )
        return error_response(
            499,
            "Client closed the request before the turn completed.",
            "client_disconnected",
        )

    if outcome["kind"] == "tool_calls":
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": outcome["tool_calls"],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": outcome["answer"]}
        finish_reason = "stop"

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": outcome["usage"],
        }
    )
