"""
Generic Type-Safe Agent Base

Provides the foundation for all agents with compile-time type safety
and runtime Pydantic validation. Type safety is inherent - not an option.

Streaming support: All agents support streaming via `stream=True` parameter.
Call `self.emit_progress()` from within `_process_impl()` to emit progress
events during processing. When streaming, these are yielded to the consumer.
When not streaming, they are silently discarded — zero overhead, zero changes
to agent logic.

Usage:
    class SearchAgent(AgentBase[SearchInput, SearchOutput, SearchDeps]):
        async def _process_impl(self, input: SearchInput) -> SearchOutput:
            self.emit_progress("encoding", "Encoding query...")
            encoded = await self.encode(input.query)

            self.emit_progress("retrieval", "Searching...")
            results = await self.deps.vespa_client.search(encoded)

            return SearchOutput(results=results)

    # Non-streaming — emit_progress calls are no-ops
    result = await agent.process(input)

    # Streaming — emit_progress calls yield SSE events
    async for event in await agent.process(input, stream=True):
        print(event)
        # {"type": "status", "phase": "encoding", "message": "Encoding query..."}
        # {"type": "status", "phase": "retrieval", "message": "Searching..."}
        # {"type": "final", "data": {...}}
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    overload,
)

if TYPE_CHECKING:
    from cogniverse_core.agents.rails import RailChain

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation.

    Extracted from A2A Task.history messages for threading through
    the agent pipeline.
    """

    role: str = Field(..., description="Message role: 'user' or 'agent'")
    content: str = Field(..., description="Text content of the message")
    agent_name: Optional[str] = Field(
        default=None, description="Agent that produced this turn (for agent roles)"
    )


class AgentInput(BaseModel):
    """
    Base class for all agent inputs.

    All agent input types must inherit from this.
    Pydantic validation is automatic.
    """

    # Use "ignore" to allow orchestrator to pass additional context fields
    # from previous agent results, while still validating known fields
    model_config = ConfigDict(extra="ignore")


class AgentOutput(BaseModel):
    """
    Base class for all agent outputs.

    All agent output types must inherit from this.
    Pydantic validation is automatic.
    """

    model_config = ConfigDict(extra="forbid")


class AgentDeps(BaseModel):
    """
    Base class for agent dependencies.

    Contains configuration, clients, and services the agent needs.
    Agents are tenant-agnostic at startup — tenant_id arrives per-request.
    """

    model_config = ConfigDict(extra="allow")  # Dependencies can have additional fields


# Type variables for generics
InputT = TypeVar("InputT", bound=AgentInput)
OutputT = TypeVar("OutputT", bound=AgentOutput)
DepsT = TypeVar("DepsT", bound=AgentDeps)


class AgentValidationError(Exception):
    """Raised when agent input or output validation fails."""

    def __init__(
        self, message: str, validation_error: Optional[ValidationError] = None
    ):
        super().__init__(message)
        self.validation_error = validation_error


class AgentBase(ABC, Generic[InputT, OutputT, DepsT]):
    """
    Generic type-safe agent base class with streaming support.

    Type safety is inherent - every agent has typed input, output, and dependencies.
    Pydantic validates at runtime; type checkers validate at write-time.

    Streaming: Use `process(input, stream=True)` for SSE-compatible streaming.
    Call `self.emit_progress()` from `_process_impl()` to emit progress events.

    Type Parameters:
        InputT: Agent input type (must extend AgentInput)
        OutputT: Agent output type (must extend AgentOutput)
        DepsT: Agent dependencies type (must extend AgentDeps)

    Example:
        class MyAgent(AgentBase[MyInput, MyOutput, MyDeps]):
            async def _process_impl(self, input: MyInput) -> MyOutput:
                # Full IDE autocomplete, type checking enforced
                return MyOutput(result=input.query.upper())

        # Non-streaming
        result = await agent.process(input)

        # Streaming
        async for event in agent.process(input, stream=True):
            print(event)  # {"type": "final", "data": {...}}
    """

    # Class-level type references extracted from generics
    _input_type: ClassVar[Type[AgentInput]]
    _output_type: ClassVar[Type[AgentOutput]]
    _deps_type: ClassVar[Type[AgentDeps]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Extract generic type parameters at class definition time."""
        super().__init_subclass__(**kwargs)

        # Walk through the MRO to find our generic parameters
        for base in getattr(cls, "__orig_bases__", []):
            origin = getattr(base, "__origin__", None)
            if origin is AgentBase or (
                origin is not None and issubclass(origin, AgentBase)
            ):
                args = get_args(base)
                if len(args) >= 3:
                    cls._input_type = args[0]
                    cls._output_type = args[1]
                    cls._deps_type = args[2]
                    return

        # If we're a concrete class (not abstract), we need types
        if not getattr(cls, "__abstractmethods__", set()):
            # Check if types were inherited from parent
            if not hasattr(cls, "_input_type"):
                raise TypeError(
                    f"{cls.__name__} must specify generic type parameters: "
                    f"class {cls.__name__}(AgentBase[InputT, OutputT, DepsT])"
                )

    def __init__(self, deps: DepsT) -> None:
        """
        Initialize agent with typed dependencies.

        Args:
            deps: Agent dependencies (validated by Pydantic)

        Raises:
            TypeError: If deps is not the correct type
            ValidationError: If deps fails Pydantic validation
        """
        if not isinstance(deps, self._deps_type):
            raise TypeError(
                f"deps must be {self._deps_type.__name__}, got {type(deps).__name__}"
            )

        self.deps = deps
        self._process_count = 0
        self._error_count = 0
        self._progress_queue: Optional[asyncio.Queue] = None
        self._input_rails: Optional["RailChain"] = None
        self._output_rails: Optional["RailChain"] = None

        logger.debug(f"Initialized {self.__class__.__name__}")

    def set_rails(
        self,
        input_rails: Optional["RailChain"] = None,
        output_rails: Optional["RailChain"] = None,
    ) -> None:
        """Attach input and/or output rail chains to this agent."""
        self._input_rails = input_rails
        self._output_rails = output_rails

    def emit_progress(
        self,
        phase: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a progress event during processing.

        When streaming (process(stream=True)), events are yielded to the consumer.
        When not streaming, this is a no-op — zero overhead.

        Args:
            phase: Processing phase name (e.g., "thinking", "retrieval", "encoding")
            message: Human-readable status message
            data: Optional partial results to include in the event
        """
        if self._progress_queue is None:
            return
        event: Dict[str, Any] = {
            "type": "status" if data is None else "partial",
            "phase": phase,
            "message": message,
        }
        if data is not None:
            event["data"] = data
        self._progress_queue.put_nowait(event)

    async def call_dspy(
        self,
        module,
        output_field: str = "summary",
        **kwargs,
    ):
        """Call a DSPy module, streaming tokens via emit_progress when active.

        When not streaming, calls module.forward(**kwargs) directly.
        When streaming, wraps with dspy.streamify() to emit token-by-token
        progress events for the specified output_field.

        Args:
            module: DSPy module (must have forward() method)
            output_field: Name of the output field to stream tokens for
            **kwargs: Arguments to pass to module.forward()

        Returns:
            DSPy Prediction from the module
        """
        if self._progress_queue is not None:
            import dspy

            streaming_fn = dspy.streamify(
                module,
                stream_listeners=[
                    dspy.streaming.StreamListener(output_field),
                ],
                include_final_prediction_in_output_stream=True,
            )
            accumulated = ""
            prediction = None
            async for chunk in streaming_fn(**kwargs):
                if isinstance(chunk, dspy.Prediction):
                    prediction = chunk
                else:
                    accumulated += str(chunk)
                    self.emit_progress(
                        "token", str(chunk), data={"accumulated": accumulated}
                    )
            return prediction or module.forward(**kwargs)
        else:
            return module.forward(**kwargs)

    def validate_input(self, raw_input: Dict[str, Any]) -> InputT:
        """
        Validate raw input dict and convert to typed InputT.

        Args:
            raw_input: Raw dictionary input

        Returns:
            Validated InputT instance

        Raises:
            AgentValidationError: If validation fails
        """
        try:
            return self._input_type.model_validate(raw_input)
        except ValidationError as e:
            raise AgentValidationError(
                f"Input validation failed for {self._input_type.__name__}: {e}",
                validation_error=e,
            )

    def validate_output(self, raw_output: Dict[str, Any]) -> OutputT:
        """
        Validate raw output dict and convert to typed OutputT.

        Args:
            raw_output: Raw dictionary output

        Returns:
            Validated OutputT instance

        Raises:
            AgentValidationError: If validation fails
        """
        try:
            return self._output_type.model_validate(raw_output)
        except ValidationError as e:
            raise AgentValidationError(
                f"Output validation failed for {self._output_type.__name__}: {e}",
                validation_error=e,
            )

    @overload
    async def process(
        self, input: InputT, stream: Literal[False] = False
    ) -> OutputT: ...

    @overload
    async def process(
        self, input: InputT, stream: Literal[True]
    ) -> AsyncGenerator[Dict[str, Any], None]: ...

    async def process(
        self, input: Union[InputT, Dict[str, Any]], stream: bool = False
    ) -> Union[OutputT, AsyncGenerator[Dict[str, Any], None]]:
        """
        Process typed input. Returns result or async generator based on stream param.

        Runs input rails before _process_impl() and output rails after.

        Args:
            input: Input of type InputT or dict (auto-validated to InputT)
            stream: If True, returns async generator yielding events

        Returns:
            If stream=False: Output of type OutputT
            If stream=True: AsyncGenerator yielding event dicts

        Raises:
            RailBlockedError: If input or output violates a rail
        """
        if isinstance(input, dict):
            typed_input = self.validate_input(input)
        else:
            typed_input = input

        if self._input_rails:
            self._input_rails.check(typed_input.model_dump())

        if stream:
            return self._stream_with_progress(typed_input)

        result = await self._process_impl(typed_input)

        if self._output_rails:
            self._output_rails.check(result.model_dump())

        return result

    @abstractmethod
    async def _process_impl(self, input: InputT) -> OutputT:
        """Core processing logic. Subclasses must implement this.

        Call self.emit_progress() during processing to emit streaming events.
        When not streaming, emit_progress is a no-op.

        Args:
            input: Validated input of type InputT

        Returns:
            Output of type OutputT
        """
        pass

    async def _stream_with_progress(
        self, input: InputT
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run _process_impl() while yielding progress events from emit_progress().

        Creates a progress queue, runs _process_impl in a concurrent task,
        and yields events as they arrive. After _process_impl completes,
        yields the final result. On error, yields an error event.

        Agents do NOT override this. They call self.emit_progress() from
        within _process_impl() instead.
        """
        self._progress_queue = asyncio.Queue()
        _SENTINEL = object()

        result_holder: list = []
        error_holder: list = []

        async def _run_impl():
            try:
                result = await self._process_impl(input)
                result_holder.append(result)
            except Exception as e:
                error_holder.append(e)
            finally:
                self._progress_queue.put_nowait(_SENTINEL)

        task = asyncio.create_task(_run_impl())

        try:
            while True:
                event = await self._progress_queue.get()
                if event is _SENTINEL:
                    break
                yield event

            if error_holder:
                yield {
                    "type": "error",
                    "message": str(error_holder[0]),
                }
            elif result_holder:
                yield {
                    "type": "final",
                    "data": result_holder[0].model_dump(),
                }
        finally:
            self._progress_queue = None
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def run(
        self, raw_input: Dict[str, Any], stream: bool = False
    ) -> Union[OutputT, AsyncGenerator[Dict[str, Any], None]]:
        """
        Run agent with raw input dict.

        Validates input, processes, and validates output.

        Args:
            raw_input: Raw dictionary input
            stream: If True, returns async generator yielding events

        Returns:
            If stream=False: Validated output of type OutputT
            If stream=True: AsyncGenerator yielding event dicts

        Raises:
            AgentValidationError: If input or output validation fails
        """
        self._process_count += 1

        try:
            validated_input = self.validate_input(raw_input)

            if stream:
                return self._stream_with_progress(validated_input)
            else:
                output = await self._process_impl(validated_input)

                if not isinstance(output, self._output_type):
                    raise TypeError(
                        f"_process_impl() must return {self._output_type.__name__}, "
                        f"got {type(output).__name__}"
                    )

                return output

        except Exception:
            self._error_count += 1
            raise

    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for input type."""
        return cls._input_type.model_json_schema()

    @classmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for output type."""
        return cls._output_type.model_json_schema()

    @classmethod
    def get_deps_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for dependencies type."""
        return cls._deps_type.model_json_schema()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent processing statistics."""
        return {
            "agent": self.__class__.__name__,
            "process_count": self._process_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._process_count - self._error_count) / self._process_count
                if self._process_count > 0
                else 1.0
            ),
        }
