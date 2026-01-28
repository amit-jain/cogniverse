"""
Generic Type-Safe Agent Base

Provides the foundation for all agents with compile-time type safety
and runtime Pydantic validation. Type safety is inherent - not an option.

Streaming support: All agents support streaming via `stream=True` parameter.
Override `_process_stream_impl()` for custom streaming behavior.

Usage:
    class SearchInput(AgentInput):
        query: str
        top_k: int = 10

    class SearchOutput(AgentOutput):
        results: List[SearchResult]

    class SearchDeps(AgentDeps):
        vespa_client: VespaClient

    class SearchAgent(AgentBase[SearchInput, SearchOutput, SearchDeps]):
        async def _process_impl(self, input: SearchInput) -> SearchOutput:
            # IDE autocomplete works, types are enforced
            results = await self.deps.vespa_client.search(input.query)
            return SearchOutput(results=results)

    # Usage:
    result = await agent.process(input, stream=False)  # Returns SearchOutput
    async for event in agent.process(input, stream=True):  # Streams events
        print(event)
"""

import logging
from abc import ABC, abstractmethod
from typing import (
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

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


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
    tenant_id is required for multi-tenancy isolation.
    """

    tenant_id: str

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
    Override `_process_stream_impl()` for custom streaming behavior.

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
        # Validate deps type at runtime
        if not isinstance(deps, self._deps_type):
            raise TypeError(
                f"deps must be {self._deps_type.__name__}, "
                f"got {type(deps).__name__}"
            )

        self.deps = deps
        self._process_count = 0
        self._error_count = 0

        logger.debug(
            f"Initialized {self.__class__.__name__} for tenant {deps.tenant_id}"
        )

    @property
    def tenant_id(self) -> str:
        """Get tenant ID from dependencies."""
        return self.deps.tenant_id

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

        Args:
            input: Input of type InputT or dict (auto-validated to InputT)
            stream: If True, returns async generator yielding events (OpenAI style)

        Returns:
            If stream=False: Output of type OutputT
            If stream=True: AsyncGenerator yielding event dicts

        Example:
            # Non-streaming
            result = await agent.process(input)

            # Streaming
            async for event in agent.process(input, stream=True):
                print(event)
        """
        # Auto-validate dict inputs to typed InputT
        if isinstance(input, dict):
            typed_input = self.validate_input(input)
        else:
            typed_input = input

        if stream:
            return self._process_stream_impl(typed_input)
        else:
            return await self._process_impl(typed_input)

    @abstractmethod
    async def _process_impl(self, input: InputT) -> OutputT:
        """
        Core processing logic. Subclasses must implement this.

        This replaces the old `process()` method. Rename your existing
        `process()` implementations to `_process_impl()`.

        Args:
            input: Validated input of type InputT

        Returns:
            Output of type OutputT
        """
        pass

    async def _process_stream_impl(
        self, input: InputT
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming implementation. Override for custom streaming behavior.

        Default implementation wraps _process_impl() result in a final event.
        Override this to yield intermediate progress events.

        Args:
            input: Validated input of type InputT

        Yields:
            Event dicts with "type" key: "status", "partial", "final", "error"

        Example override:
            async def _process_stream_impl(self, input: MyInput):
                yield {"type": "status", "message": "Processing..."}
                result = await self._process_impl(input)
                yield {"type": "final", "data": result.model_dump()}
        """
        result = await self._process_impl(input)
        yield {"type": "final", "data": result.model_dump()}

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
            # Validate input
            validated_input = self.validate_input(raw_input)

            if stream:
                # Return streaming generator
                return self._process_stream_impl(validated_input)
            else:
                # Process non-streaming
                output = await self._process_impl(validated_input)

                # Verify output type
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
            "tenant_id": self.tenant_id,
            "process_count": self._process_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._process_count - self._error_count) / self._process_count
                if self._process_count > 0
                else 1.0
            ),
        }
